"""Rev 16: dump the post-sync CP instruction run of chosen segments (block 1) of a .bin: kind + push tag, compressed.
python3 r21/cpdump.py BIN seg_index [...]   (npu-iron: aie.utils.txn_trace)"""
import sys

sys.path.insert(0, "/work/npu-research/r20")
import cpseq  # noqa: E402

segs = cpseq.segments(sys.argv[1])
for k in (int(x) for x in sys.argv[2:]):
    seg = segs[24 + k]
    print(f"## segment {k} ({cpseq.NAMES[k]}), {len(seg)} instructions")
    out = []
    for i, kind, t in seg:
        s = kind if t is None else f"PUSH[{t[0]} {t[1]}{t[2]} c{t[3]} len{t[4]}]"
        out.append(f"{i}:{s}")
    print(" ".join(out))
