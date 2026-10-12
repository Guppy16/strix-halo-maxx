"""Crop study: can TAPNext++ track a compact group of points from a small, moving crop of the frame?

CPU only (bf16 reference, depth 12). Runs in npu-iron-torch with
    -v $SHM_SCRATCH:/work -v <repo>:<repo>   (see run.sh)
Model / data loading follow tapnext/npu (step8.load_model, davis_cpu.preprocess / load_dv); they are restated here
because those modules import pyxrt (step8) or exec the model from a /work path that differs in this mount (davis_cpu).

    python3 crop_study.py stats   [--videos 10]                  groups, spreads, run-time inputs
    python3 crop_study.py run     --variants base,c256,a128s,... [--videos 10]   appends out/runs.jsonl, out/pred/*.npz
    python3 crop_study.py speed                                  CPU ms/frame per window size, same session
    python3 crop_study.py report                                 tables -> results.json, results_*.csv

Variants: base = model.step on the full frame (+ attention mass outside each W window);
c256 = the crop path with W=256 (must equal base); a<W>s / a<W>n = moving crop of W px with / without state shift.
"""

import argparse
import json
import os
import pathlib
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

TAPNET = "/work/tapnet"
TAPNEXT_SRC = os.environ.get("TAPNEXT_SRC", "/work/tapnext_model.py")  # the TAPNext torch model file
WEIGHTS = os.environ.get("TAPNEXT_WEIGHTS", f"{TAPNET}/tapnextpp_256.safetensors")
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
PATCH, GRID, IMAGE = 8, 32, 256
CELL = 64  # grouping rule: queries whose query-frame (y, x) fall in the same 64x64 cell form one group
MARGIN = 16  # px kept between the group's bbox and the crop edge
SIZES = (128, 160, 192, 256)
ATTN_W = (128, 160, 192)

tn = types.ModuleType("tn")  # the vendored model targets Python 3.14 (lazy annotations); the container is 3.13
exec(open(TAPNEXT_SRC).read().replace("-> TAPNext:", '-> "TAPNext":'), tn.__dict__)


def load_model(depth=12):
    model = tn.TAPNext()
    tn.load_state(model, WEIGHTS)
    model.to_half(torch.bfloat16).eval()
    model.blocks = model.blocks[:depth]
    return model


def preprocess(frames_u8, res=256):
    t = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
    return (t / 127.5 - 1.0).permute(0, 2, 3, 1).contiguous()


def load_dv():
    src = "\n".join(l for l in open(f"{TAPNET}/davis.py").read().split("\n") if "import tnpp" not in l)
    dv = types.ModuleType("dv")
    dv.__dict__["__file__"] = f"{TAPNET}/davis.py"
    exec(src.split("def track_video")[0], dv.__dict__)
    dv.ROOT = pathlib.Path(TAPNET)
    return dv


def load_videos(n):
    dv = load_dv()
    data = dv.load_davis(dv.ROOT / "data" / "tapvid_davis.zip")
    names = sorted(data.keys())[:n]
    vids = {}
    for name in names:
        d = data[name]
        pts = d["points"] * np.array([256.0, 256.0])
        qp, gt_tracks, gt_occ = dv.sample_queries_first(d["occluded"], pts)
        vids[name] = dict(frames=d["video"], qp=qp, gt_tracks=gt_tracks, gt_occ=gt_occ)
    return dv, vids


def group_queries(qp):
    """[1, Q, 3] (t, y, x) -> list of index arrays, one per occupied 64x64 cell of the query-frame position."""
    cells = {}
    for i, (_, y, x) in enumerate(qp[0]):
        key = (min(int(y // CELL), IMAGE // CELL - 1), min(int(x // CELL), IMAGE // CELL - 1))
        cells.setdefault(key, []).append(i)
    return [np.array(cells[k]) for k in sorted(cells)]


def choose_window(anchors, w0, fit=True):
    """anchors [k, 2] (y, x) in 256 px -> (r0, c0, n) patch origin and side. The window is the smallest size >= w0
    whose bbox + 2*MARGIN fits (fallback: enlarge, up to the full frame), centred on the bbox, on the 8 px grid."""
    a = np.clip(anchors, 0, IMAGE - 1e-3)
    lo, hi = a.min(0), a.max(0)
    w = w0
    if fit:
        for w in (s for s in SIZES if s >= w0):
            if (hi - lo).max() + 2 * MARGIN <= w:
                break
    n = w // PATCH
    origin = np.round(((lo + hi) / 2 - w / 2) / PATCH).astype(int)
    origin = np.clip(origin, 0, GRID - n)
    return int(origin[0]), int(origin[1]), n


def anchors_at(t, qp_g, prev_tracks):
    """Points the crop must cover at frame t: active points at their t-1 prediction, points queried at t at their
    query position; before any query, the next queries' positions."""
    qt = qp_g[:, 0]
    if (qt <= t).any():
        a = np.where((qt < t)[:, None], prev_tracks, qp_g[:, 1:])
        return a[qt <= t]
    return qp_g[qt == qt.min(), 1:]


def run_vit(vit, x, rec):
    """VitBlock.forward, plus (when rec is not None) the softmax attention probabilities [H, N, N] in fp32."""
    h = vit.ln_1(x)
    if rec is not None:
        sa = vit.self_attention
        n, c = h.shape
        qkv = F.linear(h, sa.in_proj_weight, sa.in_proj_bias)
        q, k, _ = qkv.view(n, 3, sa.num_heads, c // sa.num_heads).permute(1, 2, 0, 3).float()
        rec.append(torch.softmax(q @ k.transpose(1, 2) / (c // sa.num_heads) ** 0.5, -1))
    x = x + vit.self_attention(h)
    return x + vit.mlp(vit.ln_2(x))


def attention_outside(probs, active, windows):
    """Per W: (query->image mass outside the window, in-window image->outside image mass, share of the query rows'
    outside mass held by their 8 largest outside keys), averaged over heads and rows.
    probs [H, N, N] with the 1024 image tokens first; active: bool [Q] of query rows already queried."""
    out = {}
    img = probs[:, :, :1024]
    qrows = img[:, 1024:][:, torch.from_numpy(active)]
    for w, (r0, c0, n) in windows.items():
        m = torch.zeros(GRID, GRID, dtype=torch.bool)
        m[r0 : r0 + n, c0 : c0 + n] = True
        m = m.flatten()
        qo = qrows[:, :, ~m]
        q_mass = qo.sum(-1)
        top8 = (qo.topk(8, -1).values.sum(-1) / q_mass.clamp_min(1e-12)).mean().item()
        i_out = img[:, :1024][:, m][:, :, ~m].sum(-1).mean().item()
        out[w] = (q_mass.mean().item(), i_out, top8)
    return out


class CropTracker:
    """The TAPNext step over a W x W crop of patch tokens (absolute positions: the pos-emb slice of those patches).
    Per-token recurrent state lives in a 32x32 buffer per block: with shift=True it is indexed by the absolute patch,
    so state moves with the image content and tokens leaving the crop drop theirs (entering ones start at zero);
    with shift=False it is indexed by the crop-local slot, i.e. the state stays put while the crop moves."""

    def __init__(self, model, shift=True):
        self.m, self.shift = model, shift
        L = len(model.blocks)
        self.lru = torch.zeros(L, GRID * GRID, tn.WIDTH)
        self.conv = torch.zeros(L, GRID * GRID, tn.CONV_WIDTH - 1, tn.WIDTH, dtype=torch.bfloat16)
        self.q = None  # per block BlockCache of the query tokens

    def step(self, frame, rel, r0, c0, n):
        m = self.m
        ar = torch.arange(n)
        idx = ((r0 + ar)[:, None] * GRID + (c0 + ar)[None]).flatten()
        sidx = idx if self.shift else (ar[:, None] * GRID + ar[None]).flatten()
        pix = frame[r0 * PATCH : (r0 + n) * PATCH, c0 * PATCH : (c0 + n) * PATCH]
        x = pix.permute(2, 0, 1)[None].to(m.lin_proj.weight.dtype)
        video = m.lin_proj(x)[0].flatten(1).transpose(0, 1) + m.image_pos_emb[0, idx]
        tokens = torch.cat([video, m.embed_queries(rel).to(video.dtype)])
        hw = video.shape[0]
        first = self.q is None
        newq = []
        for i, block in enumerate(m.blocks):
            cache = None
            if not first:
                cache = tn.BlockCache(
                    lru=torch.cat([self.lru[i, sidx], self.q[i].lru]), conv=torch.cat([self.conv[i, sidx], self.q[i].conv])
                )
            tokens, c = block(tokens, cache)
            if self.shift:  # tokens outside the crop lose their state
                self.lru[i].zero_()
                self.conv[i].zero_()
            self.lru[i, sidx] = c.lru[:hw]
            self.conv[i, sidx] = c.conv[:hw]
            newq.append(tn.BlockCache(lru=c.lru[hw:], conv=c.conv[hw:]))
        self.q = newq
        tokens = m.encoder_norm(tokens)
        return m.heads(tokens[hw:])


def base_step(model, frame, rel, caches, rec):
    """model.step, block by block so the attention can be recorded (identical arithmetic)."""
    x = frame.permute(2, 0, 1)[None].to(model.lin_proj.weight.dtype)
    video = model.lin_proj(x)[0].flatten(1).transpose(0, 1) + model.image_pos_emb[0]
    hw = video.shape[0]
    tokens = torch.cat([video, model.embed_queries(rel).to(video.dtype)])
    new = []
    for i, block in enumerate(model.blocks):
        tokens, c = block.ssm_block(tokens, None if caches is None else caches[i])
        tokens = run_vit(block.vit_block, tokens, rec)
        new.append(c)
    tokens = model.encoder_norm(tokens)
    tracks, logits, vis = model.heads(tokens[hw:])
    return tracks, logits, vis, new


def zoom_crop(frames_u8, y0, x0, side):
    """The square [y0, y0 + side) x [x0, x0 + side) of 256 model space, sampled from the full-resolution frames
    straight to 256 x 256 (bilinear, same sampling convention as preprocess)."""
    i = (torch.arange(IMAGE) + 0.5) * side / IMAGE
    gy, gx = torch.meshgrid((y0 + i) / IMAGE * 2 - 1, (x0 + i) / IMAGE * 2 - 1, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None].expand(frames_u8.shape[0], -1, -1, -1)
    t = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float()
    t = F.grid_sample(t, grid, mode="bilinear", align_corners=False)
    return (t / 127.5 - 1.0).permute(0, 2, 3, 1).contiguous()


def zoom_window(qp_g, s0):
    """B's fixed crop: the smallest side >= s0 covering the query-frame positions + 2*MARGIN, centred, in the frame."""
    a = qp_g[:, 1:]
    lo, hi = a.min(0), a.max(0)
    side = next((s for s in SIZES if s >= s0 and (hi - lo).max() + 2 * MARGIN <= s), IMAGE)
    y0, x0 = np.clip((lo + hi) / 2 - side / 2, 0, IMAGE - side)
    return float(y0), float(x0), side


def track_group(model, video, qp_g, variant, frames_u8=None):
    """One group through one variant. Returns tracks [T, Q, 2] (y, x), vis [T, Q], window sizes [T], extra."""
    T = video.shape[0]
    if variant[0] == "b" and variant[1].isdigit():  # B: fixed zoomed crop from the full-resolution frames, full pos-emb, coords mapped back
        y0, x0, side = zoom_window(qp_g, int(variant[1:4]))
        video = zoom_crop(frames_u8, y0, x0, side)
        qz = qp_g.copy()
        qz[:, 1:] = (qp_g[:, 1:] - [y0, x0]) * IMAGE / side
        tracks, vis, _, ms, _ = track_group(model, video, qz, "base_noattn")
        tracks = tracks * side / IMAGE + np.array([y0, x0], np.float32)
        return tracks, vis, np.full(T, side), ms, dict(window=np.array([y0, x0, side]))
    q = torch.from_numpy(qp_g[None].astype(np.float32))
    prev = qp_g[:, 1:].copy()
    tr, vl, sizes, attn = [], [], [], []
    caches = None
    if not variant.startswith("base"):
        w0 = int(variant[1:4])
        crop = CropTracker(model, shift=variant == "c256" or variant.endswith("s"))
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(T):
            rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
            anc = anchors_at(t, qp_g, prev)
            if variant.startswith("base"):
                rec = [] if variant == "base" and (qp_g[:, 0] <= t).any() else None
                tracks, _, vis, caches = base_step(model, video[t], rel, caches, rec)
                sizes.append(IMAGE)
                if rec is not None:
                    wins = {w: choose_window(anc, w, fit=False) for w in ATTN_W}
                    active = qp_g[:, 0] <= t
                    per_block = [attention_outside(p, active, wins) for p in rec]
                    attn.append([[v for w in ATTN_W for v in ab[w]] for ab in per_block])
            else:
                r0, c0, n = choose_window(anc, w0)
                tracks, _, vis = crop.step(video[t], rel, r0, c0, n)
                sizes.append(n * PATCH)
            prev = tracks.numpy().astype(np.float64)
            tr.append(tracks.clone())
            vl.append(vis.clone())
    ms = (time.perf_counter() - t0) * 1e3 / T
    extra = {}
    if attn:
        a = np.array(attn)  # [frames, blocks, 3 * len(ATTN_W)]
        extra["attn"] = a
    return torch.stack(tr).numpy(), torch.stack(vl).numpy(), np.array(sizes), ms, extra


def metrics(dv, qp, gt_occ, gt_tracks, tracks, vis):
    pred_tracks = tracks[..., ::-1].transpose(1, 0, 2)[None]
    pred_occ = ~(vis > 0).T[None]
    m = dv.compute_tapvid_metrics(qp, gt_occ, gt_tracks, pred_occ, pred_tracks)
    return {k: float(np.mean(v)) for k, v in m.items()}


def spread(qp_g, gt_tracks_g, gt_occ_g):
    """Query-frame bbox max side, and per-frame bbox max side of the visible GT points (median / max over time)."""
    qs = qp_g[:, 1:]
    q_side = float((qs.max(0) - qs.min(0)).max())
    sides = []
    for t in range(gt_tracks_g.shape[1]):
        v = ~gt_occ_g[:, t]
        if v.sum() >= 2:
            p = gt_tracks_g[v, t]
            sides.append(float((p.max(0) - p.min(0)).max()))
    return q_side, float(np.median(sides)) if sides else 0.0, float(max(sides)) if sides else 0.0


def cmd_stats(o):
    _, vids = load_videos(o.videos)
    tot_frames = tot_groups = 0
    for name, v in vids.items():
        gs = group_queries(v["qp"])
        T = v["frames"].shape[0]
        tot_frames += T * len(gs)
        tot_groups += len(gs)
        print(f"{name:20s} T={T:3d} Q={v['qp'].shape[1]:3d} frame={v['frames'].shape[1:]} groups={len(gs)} "
              f"sizes={[len(g) for g in gs]}")
    print(f"groups={tot_groups} group-frames={tot_frames}")


def done_set():
    p = OUT / "runs.jsonl"
    if not p.exists():
        return set()
    return {(r["video"], r["group"], r["variant"]) for r in map(json.loads, p.open())}


def cmd_run(o):
    torch.set_num_threads(o.threads)
    OUT.mkdir(exist_ok=True)
    (OUT / "pred").mkdir(exist_ok=True)
    dv, vids = load_videos(o.videos)
    if o.only:
        vids = {k: v for k, v in vids.items() if k in o.only.split(",")}
    model = load_model(12)
    done = done_set()
    for variant in o.variants.split(","):
        for name, v in vids.items():
            video = preprocess(v["frames"])
            for gi, g in enumerate(group_queries(v["qp"])):
                if (name, gi, variant) in done:
                    continue
                qp_g = v["qp"][0, g]
                tracks, vis, sizes, ms, extra = track_group(model, video, qp_g, variant, v["frames"])
                gm = metrics(dv, v["qp"][:, g], v["gt_occ"][:, g], v["gt_tracks"][:, g], tracks, vis)
                np.savez(OUT / "pred" / f"{name}_{gi}_{variant}.npz", tracks=tracks, vis=vis, sizes=sizes, **extra)
                w0 = IMAGE if variant == "base" else int(variant[1:4])
                row = dict(video=name, group=gi, variant=variant, q=len(g), T=len(sizes), ms_per_frame=round(ms, 1),
                           frac_enlarged=float(np.mean(sizes > w0)), frac_full=float(np.mean(sizes == IMAGE)),
                           **{k: round(gm[k] * 100, 3) for k in ("average_jaccard", "occlusion_accuracy",
                                                                 "average_pts_within_thresh")})
                with (OUT / "runs.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    print("RUN_DONE")


def flops_per_frame(n_img, n_q, depth=12):
    """MACs x 2 for one step: patch embed, per-block SSM + ViT linears, attention (QK^T and PV), heads."""
    w = tn.WIDTH
    n = n_img + n_q
    lin = w * 2 * w + 2 * w * (w // tn.HEADS) + w * w + w * 8 * w + 4 * w * w  # ssm: yx, rg-lru gates, out, mlp
    lin += 3 * w * w + w * w + 8 * w * w  # vit: qkv, out, mlp
    blk = n * lin + 2 * n * n * w
    heads = n_q * 2 * (w * 256 + 256 * 256) + n_q * (256 * 512 + 256)
    return 2 * (n_img * 3 * PATCH * PATCH * w + depth * blk + heads)


def cmd_speed(o):
    torch.set_num_threads(o.threads)
    dv, vids = load_videos(1)
    v = next(iter(vids.values()))
    video = preprocess(v["frames"][: o.warm + o.iters])
    model = load_model(12)
    qp = v["qp"][0, : o.q]
    q = torch.from_numpy(np.concatenate([np.zeros((o.q, 1)), qp[:, 1:]], 1)[None].astype(np.float32))
    res = {}
    for rep in range(o.reps):  # interleave sizes so load drift on the shared box hits them alike
        for w in (256, 192, 160, 128):
            crop = CropTracker(model, shift=True)
            n = w // PATCH
            r0 = (GRID - n) // 2
            ts = []
            with torch.no_grad():
                for t in range(o.warm + o.iters):
                    rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
                    t0 = time.perf_counter()
                    crop.step(video[t], rel, r0, r0, n)
                    if t >= o.warm:
                        ts.append((time.perf_counter() - t0) * 1e3)
            res.setdefault(w, []).extend(ts)
    rows = []
    for w, ts in res.items():
        n_img = (w // PATCH) ** 2
        rows.append(dict(W=w, img_tokens=n_img, q=o.q, ms_median=round(float(np.median(ts)), 1),
                         ms_min=round(float(np.min(ts)), 1), gflop=round(flops_per_frame(n_img, o.q) / 1e9, 1),
                         threads=o.threads))
        print(json.dumps(rows[-1]), flush=True)
    (HERE / "speed.json").write_text(json.dumps(rows, indent=1) + "\n")


def cmd_report(o):
    dv, vids = load_videos(o.videos)
    runs = [json.loads(l) for l in (OUT / "runs.jsonl").open()]
    by = {}
    for r in runs:
        by.setdefault(r["variant"], {})[(r["video"], r["group"])] = r
    variants = [v for v in ("base", "c256", "a192s", "a192n", "a160s", "a160n", "a128s", "a128n", "b192", "b160", "b128") if v in by]
    keys = ("average_jaccard", "occlusion_accuracy", "average_pts_within_thresh")
    table, pervideo = [], {}
    for var in variants:
        per_video = []
        for name, v in vids.items():
            gs = group_queries(v["qp"])
            if not all((name, gi) in by[var] for gi in range(len(gs))):
                continue
            Q, T = v["qp"].shape[1], v["frames"].shape[0]
            tracks, vis = np.zeros((T, Q, 2)), np.zeros((T, Q))
            for gi, g in enumerate(gs):
                p = np.load(OUT / "pred" / f"{name}_{gi}_{var}.npz")
                tracks[:, g], vis[:, g] = p["tracks"], p["vis"]
            m = metrics(dv, v["qp"], v["gt_occ"], v["gt_tracks"], tracks, vis)
            per_video.append(m)
            pervideo.setdefault(name, {})[var] = round(m["average_jaccard"] * 100, 2)
        rs = list(by[var].values())
        gf = sum(r["T"] for r in rs)
        table.append(dict(variant=var, videos=len(per_video), **{k: round(float(np.mean([m[k] for m in per_video])) * 100, 2)
                                                                for k in keys},
                          frac_enlarged=round(sum(r["frac_enlarged"] * r["T"] for r in rs) / gf, 4),
                          frac_full=round(sum(r["frac_full"] * r["T"] for r in rs) / gf, 4),
                          ms_per_frame_median=float(np.median([r["ms_per_frame"] for r in rs]))))
        print(json.dumps(table[-1]))
    # B on the frames where the point's GT is inside its group's fixed crop: zoom gain separated from points leaving
    inside = {}
    for var in (v for v in variants if v[0] == "b" and v[1].isdigit()):
        aj = {"base": [], var: []}
        for name, v in vids.items():
            gs = group_queries(v["qp"])
            Q, T = v["qp"].shape[1], v["frames"].shape[0]
            out_mask = np.zeros((Q, T), bool)
            preds = {k: (np.zeros((T, Q, 2)), np.zeros((T, Q))) for k in aj}
            for gi, g in enumerate(gs):
                y0, x0, side = np.load(OUT / "pred" / f"{name}_{gi}_{var}.npz")["window"]
                xy = v["gt_tracks"][0, g]  # [q, T, 2] (x, y)
                out_mask[g] = ~((xy[..., 1] >= y0) & (xy[..., 1] < y0 + side) & (xy[..., 0] >= x0) & (xy[..., 0] < x0 + side))
                for k in aj:
                    p = np.load(OUT / "pred" / f"{name}_{gi}_{k}.npz")
                    preds[k][0][:, g], preds[k][1][:, g] = p["tracks"], p["vis"]
            gt_occ = v["gt_occ"] | out_mask[None]
            for k, (tr, vi) in preds.items():
                vi = np.where(out_mask.T, -1.0, vi)  # outside frames: predicted occluded, so they score neither way
                aj[k].append(metrics(dv, v["qp"], gt_occ, v["gt_tracks"], tr, vi))
        inside[var] = {k: {m: round(float(np.mean([x[m] for x in ms])) * 100, 2) for m in ("average_jaccard",
                       "average_pts_within_thresh")} for k, ms in aj.items()}
    print(json.dumps(inside))
    # per-group: spread, attention mass outside each W window (baseline), AJ drop per variant
    groups = []
    for name, v in vids.items():
        for gi, g in enumerate(group_queries(v["qp"])):
            if (name, gi) not in by.get("base", {}):
                continue
            qs, med, mx = spread(v["qp"][0, g], v["gt_tracks"][0, g], v["gt_occ"][0, g])
            row = dict(video=name, group=gi, q=len(g), q_side=round(qs, 1), gt_side_median=round(med, 1),
                       gt_side_max=round(mx, 1), aj_base=by["base"][(name, gi)]["average_jaccard"])
            p = np.load(OUT / "pred" / f"{name}_{gi}_base.npz")
            if "attn" in p:
                a = p["attn"].mean(0)  # [blocks, 3 * len(ATTN_W)]
                for j, w in enumerate(ATTN_W):
                    row[f"q_out_{w}"] = round(float(a[:, 3 * j].mean()), 5)
                    row[f"i_out_{w}"] = round(float(a[:, 3 * j + 1].mean()), 5)
                    row[f"q_top8_{w}"] = round(float(a[:, 3 * j + 2].mean()), 5)
                    row[f"q_out_{w}_blocks"] = [round(float(x), 4) for x in a[:, 3 * j]]
                    row[f"i_out_{w}_blocks"] = [round(float(x), 4) for x in a[:, 3 * j + 1]]
            for var in variants:
                if var != "base" and (name, gi) in by[var]:
                    if var[0] == "b" and var[1].isdigit():
                        row[f"side_{var}"] = int(np.load(OUT / "pred" / f"{name}_{gi}_{var}.npz")["sizes"][0])
                    row[f"drop_{var}"] = round(row["aj_base"] - by[var][(name, gi)]["average_jaccard"], 3)
            groups.append(row)
    corr = {}
    for w in ATTN_W:
        for s in "sn":
            var = f"a{w}{s}"
            xs = [(g[f"q_out_{w}"], g[f"drop_{var}"]) for g in groups if f"drop_{var}" in g and f"q_out_{w}" in g]
            if len(xs) > 3:
                x, y = np.array(xs).T
                rank = lambda z: np.argsort(np.argsort(z))
                corr[var] = dict(n_groups=len(xs), pearson=round(float(np.corrcoef(x, y)[0, 1]), 3),
                                 spearman=round(float(np.corrcoef(rank(x), rank(y))[0, 1]), 3))
    corr_v = {}  # per video: mean query->outside mass over its groups vs the video-level AJ drop
    for w in ATTN_W:
        for s in "sn":
            var = f"a{w}{s}"
            xs = [(np.mean([g[f"q_out_{w}"] for g in groups if g["video"] == name]), aj["base"] - aj[var])
                  for name, aj in pervideo.items() if var in aj and "base" in aj]
            if len(xs) > 3:
                x, y = np.array(xs).T
                rank = lambda z: np.argsort(np.argsort(z))
                corr_v[var] = dict(n_videos=len(xs), pearson=round(float(np.corrcoef(x, y)[0, 1]), 3),
                                   spearman=round(float(np.corrcoef(rank(x), rank(y))[0, 1]), 3))
    res = dict(table=table, b_inside_crop=inside, per_video_aj=pervideo, corr_group=corr, corr_video=corr_v, groups=groups)
    (HERE / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(dict(group=corr, video=corr_v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "run", "speed", "report"])
    ap.add_argument("--videos", type=int, default=10)
    ap.add_argument("--only", default="")
    ap.add_argument("--variants", default="base,c256")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    o = ap.parse_args()
    dict(stats=cmd_stats, run=cmd_run, speed=cmd_speed, report=cmd_report)[o.cmd](o)


if __name__ == "__main__":
    sys.exit(main())
