"""Round 25 P0FREE edit of design8.py (exact replacements, each must match once)."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
R = [
    ("""                        pst = {}  # round 24 CPPRE: part g + 1 configured right after part g is pushed (all four at once: BD peak 20 > 16)
                        for g in range(-1, 4):
                            if cppre and os.environ.get("CPPRE_OP") == "1" and g + 1 < 4:  # (off: static BD peak 20 > 16 at out_proj)
                                pst[g + 1] = [prefill(a_inb[c], part_(c, g + 1), tg, False) for c in range(COLS)]
                            want = an + ("#p0" if g < 0 else f"#h{g}")
                            for n_, g_ in list(pending):
                                if n_ == want:
                                    g_.finish()
                                    pending.remove((n_, g_))
""",
     """                        pst = {}  # round 24 CPPRE: part g + 1 configured right after part g is pushed (all four at once: BD peak 20 > 16)
                        for g in range(-1, 4):
                            early = cppre and os.environ.get("CPPRE_OP") == "1" and g + 1 < 4  # (round 24: static BD peak 20 > 16 at out_proj)
                            if early and not (p0free and g < 0):
                                pst[g + 1] = [prefill(a_inb[c], part_(c, g + 1), tg, False) for c in range(COLS)]
                            want = an + ("#p0" if g < 0 else f"#h{g}")
                            for n_, g_ in list(pending):
                                if n_ == want:
                                    g_.finish()
                                    pending.remove((n_, g_))
                            if early and p0free and g < 0:  # round 25 P0FREE: part 0 configured once #p0 freed the pass-0 fills' BDs
                                pst[0] = [prefill(a_inb[c], part_(c, 0), tg, False) for c in range(COLS)]
"""),
    ("""                if aqpre:  # round 20: qkv's Q drains (steps 0-3, its first drain group) were finished inside qkv's sequence when its
                    # third group was issued; the Q fill (into half 1) goes before the wait on the K / V groups
                    assert ops[ops.index(op) - 1].get("rr") and ops[ops.index(op) - 1]["epi"] == "qkv"
                    for c in range(COLS):
                        fill_a(op, c, group=tg, wait=False)""",
     """                # round 25 P0FREE: the Q fill and pass 0's K / V fills ride in pass 0's drain group, so out_proj's wait on "#p0" frees
                # their shim BDs (pass 0's drains complete after pass 0 consumed them); round 24 kept them in tg until the end
                kvg0 = p0g if p0free and op2 else tg
                if aqpre:  # round 20: qkv's Q drains (steps 0-3, its first drain group) were finished inside qkv's sequence when its
                    # third group was issued; the Q fill (into half 1) goes before the wait on the K / V groups
                    assert ops[ops.index(op) - 1].get("rr") and ops[ops.index(op) - 1]["epi"] == "qkv"
                    for c in range(COLS):
                        fill_a(op, c, group=kvg0, wait=False)"""),
    ("""                    for r in range(ROWS_G):
                        fill(B_prods[r], op["b"][r][0], group=tg, wait=False)
                finish_deps(op)""",
     """                    for r in range(ROWS_G):
                        fill(B_prods[r], op["b"][r][0], group=kvg0, wait=False)
                finish_deps(op)"""),
    ("""                            if not (akvpre and pi == 0 and i_ == 0):
                                fill(B_prods[r], t, group=tg, wait=False)""",
     """                            if not (akvpre and pi == 0 and i_ == 0):
                                fill(B_prods[r], t, group=kvg0 if pi == 0 else tg, wait=False)"""),
    ("""        oppre2 = os.environ.get("OPPRE2") == "1"  # round 24: out_proj's A per attention head group (pass 1)
""",
     """        oppre2 = os.environ.get("OPPRE2") == "1"  # round 24: out_proj's A per attention head group (pass 1)
        p0free = os.environ.get("P0FREE") == "1"  # round 25: the attention's Q + pass-0 K / V fills in pass 0's drain group
        assert not p0free or oppre2
"""),
]
for a, b in R:
    assert s.count(a) == 1, a[:120]
    s = s.replace(a, b)
p.write_text(s)
print("P0FREE patched")
