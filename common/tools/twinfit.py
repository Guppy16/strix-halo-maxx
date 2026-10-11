"""Rev 5: is the no-MAC twin (full12p1n) really evidence of a >= 7.6 B/cycle A path?

Refits sim9's per-op law on the twin timeline with t_ks held at fixed values and prints the span rms, so we can see
whether the timeline separates t_ks from t_tile / L_dep at all. No device. Usage: python3 twinfit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim9  # noqa: E402

B = Path(__file__).resolve().parents[2] / "npu-tapnext" / "build"
F = 1.81e9

for build in ("p1n", "r6a"):
    _, ops = sim9.load(B / f"tl_full12{build}_d12.json")
    seg = sim9.segments(ops)
    labs = sorted({s[2] for s in seg})
    print(f"\nfull12{build}: {len(seg)} spans; distinct span kinds: {len(labs)}")
    print("  " + ", ".join(labs))
    p0, rms0, _, _ = sim9.fit(ops)
    print(f"  free fit: t_ks {p0['t_ks']:.3f} us  t_tile {p0['t_tile']:.2f}  L_dep {p0['L_dep']:.1f}  rms {rms0:.2f} us")
    for tks in (0.6, 0.753, 0.9, 1.0, 1.1, 1.2, 1.3, 1.43, 1.6):
        p, rms, _, _ = sim9.fit(ops, fixed={"t_ks": tks})
        print(f"  t_ks {tks:.3f} us ({10368 / (tks * F / 1e6):.1f} B/cyc): rms {rms:6.2f} us  t_tile {p['t_tile']:6.2f}"
              f"  L_dep {p['L_dep']:6.1f}  t_gelu {p['t_gelu']:6.2f}  ssm {p['ssm']:6.0f}  attn {p['attn']:6.0f}")
