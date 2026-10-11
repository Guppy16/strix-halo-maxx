"""Re-implemented single-tile inference path for moondream2 (rev 2025-06-21).

Differences from the stock HF code path (all numerically equivalent for a 378x378 input):
  * the global crop and the single local crop are bit-identical for a 378x378 image, so the
    vision encoder runs on a batch of 1 and the features are reused for both halves of the
    projection input (stock code encodes both crops: 2x vision FLOPs);
  * the 730-token image prefill attends over exactly 730 keys with no mask (stock: full 2048-wide
    KV cache + bool mask, which forces the memory-efficient attention kernel);
  * no KV-cache cloning / reloading between encode and query;
  * the question is prefilled once (fixed-length padded prompt) and the answer is read off the
    yes/no logits of the last real position -- no sampling loop, no decode step;
  * static shapes everywhere so the whole thing can be captured in CUDA/HIP graphs.
"""
import math, numpy as np, torch, torch.nn.functional as F
try:
    from fused import rope_cache
except Exception:
    rope_cache = None
try:
    from hattn import Attn as HAttn
except Exception:
    HAttn = None
from PIL import Image

IMG = 378
PATCH = 14
GRID = IMG // PATCH          # 27
NTOK = GRID * GRID           # 729
PREFIX = NTOK + 1            # 730 = bos + image tokens
ROT = 32


class Lin:
    """Weight holder; subclasses can implement other GEMM paths."""
    def __init__(self, weight, bias, dtype):
        self.w = weight.detach().to(dtype).contiguous()
        self.b = None if bias is None else bias.detach().to(dtype).contiguous()

    def __call__(self, x):
        return F.linear(x, self.w, self.b)


def _ln(x, wb):
    return F.layer_norm(x, (x.shape[-1],), wb[0], wb[1])


def gelu(x):
    return F.gelu(x, approximate="tanh")


USE_FUSED_RES = False
SHARE_QUANT = True  # quantise the LN output once for qkv and fc1 (bitwise identical, saves 24 kernels)


def mlp(fc1, fc2, h, res=None, pre=None):
    kw = {"pre": pre} if pre is not None else {}
    a = fc1(h, gelu=True, **kw) if getattr(fc1, "fused_gelu", False) else gelu(fc1(h, **kw))
    if res is not None and USE_FUSED_RES and getattr(fc2, "fused_res", False):
        return fc2(a, res=res)
    out = fc2(a)
    return out if res is None else out + res


def lin_res(l, x, res):
    if USE_FUSED_RES and getattr(l, "fused_res", False):
        return l(x, res=res)
    return l(x) + res


class FastMD2:
    def __init__(self, model, dtype=torch.bfloat16, prompt_len=32, lin_cls=Lin, device="cuda", pool=0, tlin_cls=None, fuse=True, n_q=1, fold_proj=False, fused_res=False, hattn=False):
        mm = model.model
        self.mm = mm
        self.cfg = mm.config
        self.tok = mm.tokenizer
        self.dtype = dtype
        self.dev = torch.device(device)
        self.L = prompt_len
        self.fuse = fuse and rope_cache is not None
        self.NQ = n_q  # questions answered per encoded frame (batched prompt prefill)
        assert self.NQ == 1 or self.fuse, 'batched questions need the fused rope/cache kernel'
        self.pool = pool  # 0 = stock 27x27 image tokens; k = avg-pool projected tokens to k x k
        self.n_img = pool * pool if pool else NTOK
        self.prefix = 1 + self.n_img
        V, T = mm.vision, mm.text
        c = lambda t: t.detach().to(dtype).contiguous()
        lin = lambda m: lin_cls(m.weight, m.bias, dtype)
        tlin = lambda m: (tlin_cls or lin_cls)(m.weight, m.bias, dtype)
        # ---- vision weights
        self.patch_emb = lin(V.patch_emb)
        self.pos_emb = c(V.pos_emb)  # (1,729,1152)
        self.vblocks = [dict(ln1=(c(b.ln1.weight), c(b.ln1.bias)), qkv=lin(b.attn.qkv), proj=lin(b.attn.proj),
                             ln2=(c(b.ln2.weight), c(b.ln2.bias)), fc1=lin(b.mlp.fc1), fc2=lin(b.mlp.fc2)) for b in V.blocks]
        self.post_ln = (c(V.post_ln.weight), c(V.post_ln.bias))
        # projection input is cat([x, x]) for a single tile -> optionally fold the two K-halves of fc1 together
        # (exact in real arithmetic, but the folded bf16 weight rounds differently from the stock two-half GEMM)
        self.fold_proj = fold_proj
        if fold_proj:
            w1 = V.proj_mlp.fc1.weight.detach().float(); w1 = (w1[:, :self.cfg.vision.enc_dim] + w1[:, self.cfg.vision.enc_dim:]).to(V.proj_mlp.fc1.weight.dtype)
            self.proj_fc1 = lin_cls(w1, V.proj_mlp.fc1.bias, dtype)
        else:
            self.proj_fc1 = lin(V.proj_mlp.fc1)
        self.proj_fc2 = lin(V.proj_mlp.fc2)
        global USE_FUSED_RES
        USE_FUSED_RES = fused_res
        self.vh = self.cfg.vision.enc_n_heads
        self.hattn = hattn and HAttn is not None
        if self.hattn:
            self.attn_v = HAttn(self.vh, NTOK, self.cfg.vision.enc_dim // self.vh)
        # ---- text weights
        self.wte = c(T.wte)
        self.tblocks = [dict(ln=(c(b.ln.weight), c(b.ln.bias)), qkv=tlin(b.attn.qkv), proj=tlin(b.attn.proj),
                             fc1=tlin(b.mlp.fc1), fc2=tlin(b.mlp.fc2)) for b in T.blocks]
        self.tpost_ln = (c(T.post_ln.weight), c(T.post_ln.bias))
        self.lm_head = lin(T.lm_head)
        self.full_logits = False  # True: compute all 51200 logits (diagnostics); False: only the yes/no rows
        self.th = self.cfg.text.n_heads; self.hd = self.cfg.text.dim // self.th
        if self.hattn:
            self.attn_t = HAttn(self.th, self.prefix, self.hd)
        fc = T.freqs_cis  # (2048, 16, 2) fp32
        P = self.prefix
        self.cos_pre = fc[:P, :, 0].contiguous(); self.sin_pre = fc[:P, :, 1].contiguous()
        self.cos_q = fc[P:P + self.L, :, 0].contiguous(); self.sin_q = fc[P:P + self.L, :, 1].contiguous()
        # ---- static buffers
        CTX = P + self.L
        self.kc = [torch.zeros(self.NQ, self.th, CTX, self.hd, device=self.dev, dtype=dtype) for _ in T.blocks]
        self.vc = [torch.zeros(self.NQ, self.th, CTX, self.hd, device=self.dev, dtype=dtype) for _ in T.blocks]
        self.img_buf = torch.zeros(IMG, IMG, 3, device=self.dev, dtype=torch.uint8)
        self.tok_buf = torch.zeros(self.NQ, self.L, device=self.dev, dtype=torch.long)
        self.last_idx = torch.zeros(self.NQ, device=self.dev, dtype=torch.long)  # flat row index into (NQ*L) of each question's last token
        self.bos = torch.tensor([[self.cfg.tokenizer.bos_id]], device=self.dev)
        # query mask: prompt position i sees all 730 prefix tokens and prompt tokens <= i
        m = torch.zeros(self.L, CTX, dtype=torch.bool)
        m[:, :P] = True
        m[:, P:] = torch.tril(torch.ones(self.L, self.L, dtype=torch.bool))
        self.qmask = m[None, None].to(self.dev)
        # yes / no token ids
        ids = lambda s: self.tok.encode(s).ids
        self.yes_ids = sorted({ids(s)[0] for s in ("Yes", "yes", " Yes", " yes") if len(ids(s)) == 1})
        self.no_ids = sorted({ids(s)[0] for s in ("No", "no", " No", " no") if len(ids(s)) == 1})
        self.yes_t = torch.tensor(self.yes_ids, device=self.dev); self.no_t = torch.tensor(self.no_ids, device=self.dev)
        self.yn_ids = self.yes_ids + self.no_ids
        self.lm_small = Lin(T.lm_head.weight[self.yn_ids], T.lm_head.bias[self.yn_ids], dtype)
        self.graphs = {}

    # ------------------------------------------------------------------ vision
    def _vision(self, img_u8):
        x = img_u8.permute(2, 0, 1)[None].to(self.dtype).div_(255.0).sub_(0.5).div_(0.5)  # (1,3,378,378)
        x = x.reshape(1, 3, GRID, PATCH, GRID, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(1, NTOK, 3 * PATCH * PATCH)
        x = self.patch_emb(x) + self.pos_emb
        H = self.vh
        for b in self.vblocks:
            h = _ln(x, b["ln1"])
            q, k, v = [t.view(1, NTOK, H, -1).transpose(1, 2) for t in b["qkv"](h).chunk(3, dim=-1)]
            a = self.attn_v(q, k, v) if self.hattn else F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, NTOK, -1)
            x = lin_res(b["proj"], a, x)
            x = mlp(b["fc1"], b["fc2"], _ln(x, b["ln2"]), res=x)
        x = _ln(x, self.post_ln)[0]  # (729,1152)
        feats = x if self.fold_proj else torch.cat([x, x], dim=-1)
        return mlp(self.proj_fc1, self.proj_fc2, feats)  # (729, 2048)

    def _rope(self, x, cos, sin):
        xr, xp = x[..., :ROT], x[..., ROT:]
        a, b = xr[..., :ROT // 2], xr[..., ROT // 2:]
        cos = cos[None, None]; sin = sin[None, None]
        o = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
        return torch.cat([o.to(x.dtype), xp], dim=-1)

    def _text_layers(self, x, pos0, n, cos, sin, mask):
        H, hd = self.th, self.hd
        B = x.shape[0]
        for i, b in enumerate(self.tblocks):
            h = _ln(x, b["ln"])
            pre = b["qkv"].quant(h) if SHARE_QUANT and hasattr(b["qkv"], "quant") and hasattr(b["fc1"], "quant") else None  # shared int8 activation
            qkv = b["qkv"](h, pre=pre) if pre is not None else b["qkv"](h)
            if self.fuse:
                q = torch.empty(B, H, n, hd, device=x.device, dtype=x.dtype)
                rope_cache(qkv.view(B * n, -1), cos, sin, q, self.kc[i], self.vc[i], pos0, bcast=(pos0 == 0))
                if mask is None and self.hattn:
                    a = self.attn_t(q, self.kc[i][:1, :, :n], self.vc[i][:1, :, :n])   # (1, n, H*hd) directly
                    x = mlp(b["fc1"], b["fc2"], h, res=lin_res(b["proj"], a, x), pre=pre); continue
                if mask is None:
                    a = F.scaled_dot_product_attention(q, self.kc[i][:1, :, :n], self.vc[i][:1, :, :n])
                else:
                    a = F.scaled_dot_product_attention(q, self.kc[i][:, :, :pos0 + n], self.vc[i][:, :, :pos0 + n], attn_mask=mask)
            else:
                q, k, v = qkv.split([H * hd, H * hd, H * hd], dim=-1)
                q = self._rope(q.view(1, n, H, hd).transpose(1, 2), cos, sin)
                k = self._rope(k.view(1, n, H, hd).transpose(1, 2), cos, sin)
                v = v.view(1, n, H, hd).transpose(1, 2)
                self.kc[i][:, :, pos0:pos0 + n] = k
                self.vc[i][:, :, pos0:pos0 + n] = v
                if mask is None:
                    a = F.scaled_dot_product_attention(q, k, v)
                else:
                    a = F.scaled_dot_product_attention(q, self.kc[i][:, :, :pos0 + n], self.vc[i][:, :, :pos0 + n], attn_mask=mask)
            a = a.transpose(1, 2).reshape(B, n, H * hd)
            x = mlp(b["fc1"], b["fc2"], h, res=lin_res(b["proj"], a, x), pre=pre)
        return x

    def _encode(self):
        emb = self._vision(self.img_buf)
        if self.pool:
            emb = F.adaptive_avg_pool2d(emb.view(GRID, GRID, -1).permute(2, 0, 1), (self.pool, self.pool)).permute(1, 2, 0).reshape(self.n_img, -1)
        x = torch.cat([F.embedding(self.bos, self.wte), emb[None]], dim=1)  # (1,prefix,2048)
        self._text_layers(x, 0, self.prefix, self.cos_pre, self.sin_pre, None)

    def _query(self):
        x = F.embedding(self.tok_buf, self.wte)  # (NQ, L, 2048)
        x = self._text_layers(x, self.prefix, self.L, self.cos_q, self.sin_q, self.qmask)
        h = _ln(x.reshape(-1, x.shape[-1]).index_select(0, self.last_idx), self.tpost_ln)  # (NQ, 2048)
        return self.lm_head(h) if self.full_logits else self.lm_small(h)  # (NQ, V) or (NQ, n_yes+n_no)

    # ------------------------------------------------------------------ public
    def prompt_ids(self, question):
        t = self.cfg.tokenizer.templates["query"]
        ids = t["prefix"] + self.tok.encode(question).ids + t["suffix"] + t["suffix"]
        assert len(ids) <= self.L, (len(ids), self.L)
        return ids

    def set_image(self, img):
        if isinstance(img, Image.Image):
            if img.size != (IMG, IMG):
                img = img.resize((IMG, IMG), Image.Resampling.LANCZOS)
            img = torch.from_numpy(np.array(img.convert("RGB")))
        self.img_buf.copy_(img, non_blocking=True)

    def set_question(self, question):
        self.set_questions([question] * self.NQ if isinstance(question, str) else question)

    def set_questions(self, questions):
        assert len(questions) == self.NQ, (len(questions), self.NQ)
        buf = torch.zeros(self.NQ, self.L, dtype=torch.long); last = torch.zeros(self.NQ, dtype=torch.long)
        for b, q in enumerate(questions):
            ids = self.prompt_ids(q); buf[b, :len(ids)] = torch.tensor(ids); last[b] = b * self.L + len(ids) - 1
        self.tok_buf.copy_(buf, non_blocking=True); self.last_idx.copy_(last, non_blocking=True)

    def capture(self, which=("encode", "query")):
        s = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(s):
            for _ in range(3):
                self._encode(); self._query()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.inference_mode():
            if "encode" in which:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._encode()
                self.graphs["encode"] = g
            if "query" in which:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.logits_out = self._query()
                self.graphs["query"] = g
        torch.cuda.synchronize()

    def encode(self):
        if "encode" in self.graphs:
            self.graphs["encode"].replay()
        else:
            with torch.inference_mode():
                self._encode()

    def query(self):
        if "query" in self.graphs:
            self.graphs["query"].replay(); return self.logits_out
        with torch.inference_mode():
            return self._query()

    def decide_all(self, logits):
        """batched version: list of (pred_yes, argmax_or_-1, margin) per question"""
        return [self.decide(logits[b:b + 1]) for b in range(logits.shape[0])]

    def decide(self, logits):
        """returns (pred_yes, argmax_token_id or -1, margin) -- margin = logsumexp(yes) - logsumexp(no)"""
        lg = logits[0].float()
        if lg.numel() == len(self.yn_ids):
            ny = len(self.yes_ids)
            m = (torch.logsumexp(lg[:ny], 0) - torch.logsumexp(lg[ny:], 0)).item()
            return m > 0, -1, m
        m = (torch.logsumexp(lg[self.yes_t], 0) - torch.logsumexp(lg[self.no_t], 0)).item()
        return m > 0, int(lg.argmax().item()), m

    def answer(self, img, question):
        self.set_image(img); self.set_question(question)
        self.encode(); return self.decide(self.query())
