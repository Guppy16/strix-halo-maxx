"""Round 12: compact loop bodies whose last bundle is still 16 B aligned (zero-overhead loop rule, SKILL 3).

msched emits every body bundle in the full 16-byte form, which makes the aligned last bundle automatic but costs text
(the EW core had 144 B free). Here: assemble the function once with a compact body (first / last bundles full), read every
body bundle's size from llvm-objdump, then expand the fewest bytes' worth of bundles to the full form so that the last
bundle's offset from the (aligned) start is a multiple of 16 (a subset sum modulo 16), re-emit, re-assemble and verify.
No padding is ever inserted inside the body, so the schedule is unchanged. Needs Peano (npu-iron); elsewhere the caller
keeps the full-form body.

    fit(lp, render, name, work_dir) -> text      (render() returns the function's asm text for the current lp.body_full)
"""

import re
import subprocess
from pathlib import Path

B = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin"


def available():
    return Path(B, "clang").exists()


def body_offsets(text, name, work):
    s, o = Path(work) / f"{name}_fit.s", Path(work) / f"{name}_fit.o"
    s.write_text(text)
    subprocess.run([f"{B}/clang", "--target=aie2p-none-unknown-elf", "-c", str(s), "-o", str(o)], check=True)
    dis = subprocess.run([f"{B}/llvm-objdump", "-d", str(o)], capture_output=True, text=True, check=True).stdout
    syms = subprocess.run([f"{B}/llvm-objdump", "-t", str(o)], capture_output=True, text=True, check=True).stdout
    addr = {m.group(2): int(m.group(1), 16) for m in re.finditer(r"^([0-9a-f]+)\s.*\s(\S+)$", syms, re.M)}
    start, last = addr[f"{name}_start"], addr[f"{name}_last"]
    offs = [int(m.group(1), 16) for m in re.finditer(r"^\s*([0-9a-f]+):", dis, re.M)]
    body = [a for a in offs if start <= a <= last]
    assert body[0] == start and body[-1] == last, (hex(start), hex(last), [hex(a) for a in body[:3]])
    return body


def fit(lp, render, name, work):
    nb = len(lp.program()[1])
    lp.body_full = set()
    text = render()
    offs = body_offsets(text, name, work)
    assert len(offs) == nb, f"{len(offs)} bundles assembled, {nb} scheduled"
    sizes = [b - a for a, b in zip(offs, offs[1:])]  # bundles 0 .. nb - 2 (the last is full-form anyway)
    need = (-(offs[-1] - offs[0])) % 16
    # DP over residues mod 16: the cheapest set of expansions (each adds 16 - size bytes) with the needed residue
    best = {0: (0, frozenset())}
    for c in range(1, nb - 1):  # bundle 0 is already full
        g = 16 - sizes[c]
        if g == 0:
            continue
        nxt = dict(best)
        for r, (cost, sel) in best.items():
            r2, c2 = (r + g) % 16, cost + g
            if r2 not in nxt or c2 < nxt[r2][0]:
                nxt[r2] = (c2, sel | {c})
        best = nxt
    assert need in best, f"no expansion set reaches residue {need}"
    lp.body_full = set(best[need][1])
    text = render()
    offs = body_offsets(text, name, work)
    assert (offs[-1] - offs[0]) % 16 == 0 and offs[0] % 16 == 0, "the last body bundle is not 16 B aligned"
    return text, best[need][0], nb * 16 - (offs[-1] - offs[0] + 16)


def emit(lp, render, name, out, do_fit):
    """round 13: write the generated function to `out`; fit = compact the loop body (asmfit) when Peano is here. Returns the
    bytes saved (0 unfitted)."""
    if do_fit and available():
        text, added, saved = fit(lp, render, name, Path(out).parent)
        print(f"asmfit {name}: body compacted, {saved} B saved ({added} B of expansions for the alignment)")
    else:
        text, saved = render(), 0
    Path(out).write_text(text)
    return saved
