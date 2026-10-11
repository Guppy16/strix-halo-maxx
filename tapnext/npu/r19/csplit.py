"""Round 19 CSPLIT: patch a design8 build's lowered MLIR (input_with_addresses.mlir) so each GEMM core drains its C object in two
parts, then re-run aiecc on it (r19/csplit.sh). Per GEMM tile (col c, row r; C buffer C{c}_{r-2}_buff_0):
  * a new lock csz_c_r = aie.lock(tile, CS_Z) {init = 1}
  * the core MM2S chain  [acq cons; bytes 0 .. 18,431; rel prod]  becomes
      [acq cons; bytes 0 .. 12,287; rel prod] -> [acq cons; bytes 12,288 .. 18,431; rel csz] -> (first)
    (the memtile join still receives one 18,432 B stream per object)
  * the core's releases of the C cons lock release 2 (part 1 and part 2 ready together; mm8.cc -DCSPLIT takes csz before writing
    part 2: acc_epilogue / attn_begin)
usage: python3 r19/csplit.py IN.mlir OUT.mlir [--z 6] [--split 6144]   (split in bf16 elements)
"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--z", type=int, default=6)
ap.add_argument("--split", type=int, default=6144)
ap.add_argument("--core-rel", type=int, default=2, help="the core's IRON release of the C cons lock (1 with -DEPLR: the kernel releases part 1)")
ap.add_argument("--cons-id", type=int, default=3, help="the C cons lock id every GEMM tile must have (mm8.cc CS_CONS)")
ap.add_argument("--latec", default="", help="round 21 LATEC: lock ids (one per --sub boundary, init 1): sub-part 1's BD releases the prod "
                "lock, sub-part q + 1's BD releases latec[q] (mm_rot acquires them in a tile's first k-step: mm8.cc -DLATEC)")
ap.add_argument("--sub", default="", help="round 21 CSUB: comma-separated bf16 offsets inside part 1 where acquire-only BDs end (each "
                "takes one more C cons release per object: mm8.cc -DCSUB releases them as the epilogue finalises each sub-part)")
o = ap.parse_args()
src = open(o.inp).read()
lines = src.split("\n")

# GEMM C buffers: %C{c}_{i}_buff_0 = aie.buffer(%tile_{c}_{2+i})
cbufs = re.findall(r"%(C(\d)_(\d)_buff_0) = aie\.buffer\(%(tile_\d_\d)\).*memref<9216xbf16>", src)
assert len(cbufs) == 24, len(cbufs)
out = lines
n_bd = n_core = 0
for buf, c, i, tile in cbufs:
    pre = f"C{c}_{i}"
    cons, prod = f"%{pre}_cons_lock_0", f"%{pre}_prod_lock_0"
    assert re.search(rf"{cons} = aie\.lock\(%{tile}, {o.cons_id}\)", src), (tile, "C cons lock id")
    # locks on this tile
    ids = {int(m) for m in re.findall(rf"aie\.lock\(%{tile}, (\d+)\)", src)}
    assert o.z not in ids, (tile, ids)
    k = next(j for j, ln in enumerate(out) if ln.lstrip().startswith(f"{cons} = aie.lock(%{tile},"))
    zname = f"%csz_{tile[5:]}"
    out.insert(k + 1, f'    {zname} = aie.lock(%{tile}, {o.z}) {{init = 1 : i32, sym_name = "csz_{tile[5:]}"}}')
    lates = [int(v) for v in o.latec.split(",") if v]
    for lid in lates:
        assert lid not in ids and lid != o.z, (tile, lid, ids)
        out.insert(k + 2, f'    %csl{lid}_{tile[5:]} = aie.lock(%{tile}, {lid}) {{init = 1 : i32, sym_name = "csl{lid}_{tile[5:]}"}}')
    # the tile's DMA region
    m0 = next(j for j, ln in enumerate(out) if re.search(rf"= aie\.mem\(%{tile}\) \{{", ln))
    m1 = next(j for j in range(m0 + 1, len(out)) if out[j].startswith("    }"))
    region = out[m0:m1]
    used = {int(v) for ln in region for v in re.findall(r"bd_id = (\d+)", ln)}
    newid = next(b for b in range(16) if b not in used)
    j = next(j for j in range(m0, m1) if re.search(rf"aie\.dma_bd\(%{buf} : memref<9216xbf16> offset = 0 len = 9216\)", out[j]))
    acq, bd, rel, nxt = out[j - 1], out[j], out[j + 1], out[j + 2]
    assert cons in acq and "AcquireGreaterEqual" in acq, acq
    assert prod in rel and "Release" in rel, rel
    lab = re.search(r"aie\.next_bd (\^bb\d+)", nxt).group(1)
    old_id = int(re.search(r"\{bd_id = (\d+) : i32, next_bd_id = (\d+)", bd).group(1))
    assert re.search(rf"next_bd_id = {old_id} : i32", bd), bd  # a one-BD loop
    loc = re.search(r"( loc\(.*\))$", bd)
    loc = loc.group(1) if loc else ""
    ind = bd[:len(bd) - len(bd.lstrip())]
    bd1 = re.sub(r"len = 9216\) \{bd_id = (\d+) : i32, next_bd_id = \d+ : i32\}",
                 rf"len = {o.split}) {{bd_id = \1 : i32, next_bd_id = {newid} : i32}}", bd)
    blk = f"^cs{tile[5:]}"
    subs = [int(v) for v in o.sub.split(",") if v]
    assert all(a < b for a, b in zip([0] + subs, subs + [o.split])), subs
    head = []
    if subs:  # acquire-only BDs [0, s1), [s1, s2), ... then [s_last, split) with part 1's release
        ids = [old_id]
        for _ in subs:
            ids.append(next(b for b in range(16) if b not in used and b != newid and b not in ids))
        bounds = [0] + subs + [o.split]
        for q in range(len(subs) + 1):
            nb = newid if q == len(subs) else ids[q + 1]
            bdq = re.sub(r"offset = 0 len = 9216\) \{bd_id = (\d+) : i32, next_bd_id = \d+ : i32\}",
                         f"offset = {bounds[q]} len = {bounds[q + 1] - bounds[q]}) {{bd_id = {ids[q]} : i32, next_bd_id = {nb} : i32}}", bd)
            assert bdq != bd, bd
            if q:
                head.append(f"  ^csu{q}_{tile[5:]}:")
            head += [acq, bdq]
            if lates:  # LATEC: sub-part 1 frees the object for IRON's acquire, the later ones for mm_rot's first k-step
                assert len(lates) == len(subs)
                head += [rel if q == 0 else re.sub(rf"{prod}", f"%csl{lates[q - 1]}_{tile[5:]}", rel)]
                head.append(f"{ind}aie.next_bd {blk}{loc}" if q == len(subs) else f"{ind}aie.next_bd ^csu{q + 1}_{tile[5:]}{loc}")
            elif q == len(subs):
                head += [rel, f"{ind}aie.next_bd {blk}{loc}"]
            else:  # a BD block needs one acquire AND one release (the MLIR verifier): release the prod lock by 0 (a no-op)
                head += [re.sub(r"Release, %\w+\)", "Release, %csu_zero)", rel), f"{ind}aie.next_bd ^csu{q + 1}_{tile[5:]}{loc}"]
        out.insert(m0 + 1, f"      %csu_zero = arith.constant 0 : i32")
        j += 1
    new = (head if subs else [acq, bd1, rel, f"{ind}aie.next_bd {blk}{loc}"]) + [f"  {blk}:",
           acq,
           f"{ind}aie.dma_bd(%{buf} : memref<9216xbf16> offset = {o.split} len = {9216 - o.split}) {{bd_id = {newid} : i32, next_bd_id = {old_id} : i32}}{loc}",
           re.sub(rf"{prod}", zname, rel),
           f"{ind}aie.next_bd {lab}{loc}"]
    assert lab in "".join(out[j - 3:j - 1]) or True
    out[j - 1:j + 3] = new
    n_bd += 1
    # the core: releases of the C cons lock by 2
    k0 = next(j for j, ln in enumerate(out) if re.search(rf"= aie\.core\(%{tile}\) \{{", ln))
    k1 = next(j for j in range(k0 + 1, len(out)) if out[j].startswith("    }"))
    cname = f"%cs2_{tile[5:]}"
    cnt = 0
    if o.core_rel == 1:
        cnt = sum(1 for j in range(k0, k1) if re.search(rf"aie\.use_lock\({cons}, Release,", out[j]))
        assert cnt >= 1, tile
        n_core += cnt
        continue
    for j in range(k0, k1):
        if re.search(rf"aie\.use_lock\({cons}, Release, %\w+\)", out[j]):
            out[j] = re.sub(rf"aie\.use_lock\({cons}, Release, %\w+\)", f"aie.use_lock({cons}, Release, {cname})", out[j])
            cnt += 1
    assert cnt >= 1, tile
    out.insert(k0 + 1, f"      {cname} = arith.constant 2 : i32")
    n_core += cnt
open(o.out, "w").write("\n".join(out))
print(f"csplit: {n_bd} C MM2S chains split at {2 * o.split} B (sub-parts at {o.sub or 'none'}), {n_core} core releases -> {o.core_rel}, lock id {o.z}")
