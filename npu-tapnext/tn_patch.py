from pathlib import Path
d = Path("design6.py").read_text()
d = d.replace("RA_K = 1152  # one memtile slot holds half an M-tile of A for K <= 2 * RA_K", "RA_K = int(os.environ.get(\"RA_K\", \"1152\"))  # one memtile slot holds half an M-tile of A for K <= 2 * RA_K (TAPNext: 768)")
assert "RA_K = int(" in d
Path("design6.py").write_text(d)
k = Path("attn_tn.cc").read_text()
k = k.replace("constexpr int QT = 48, HD = 72, KC = 48;", "constexpr int QT = 48, HD = 64, KC = 48;   // TAPNext: 12 heads x 64 dims")
k = k.replace("constexpr int GQ = 704;   // Q / K group: 9 panels (648 B) padded to 704 B (the streams need 64 B-aligned seeks)", "constexpr int GQ = 576;   // Q / K group: 8 panels = 576 B = 9 words (64 B-aligned by itself)")
k = k.replace("constexpr int OBJB = RB * GQ;  // 4224 B per Q / K / V object", "constexpr int OBJB = NB * GV;  // 3584 B per Q / K / V object (V^T: 8 groups of 448 B; Q / K use 3456 of it)")
assert "HD = 64" in k and "GQ = 576" in k and "OBJB = NB * GV" in k
Path("attn_tn.cc").write_text(k)
a = Path("attn_tn.py").read_text()
a = a.replace("QT, HD, KC, NKEYS = 48, 72, 48, 729", "QT, HD, KC, NKEYS = 48, 64, 48, 1072")
a = a.replace("HEADS_PER_COL = 2\nITEMS = HEADS_PER_COL * PASSES  # 8 per core", "HEADS = 12\nQTILES = 23          # ceil(1072 / 48) query tiles per head, padded to 24 = 6 passes of 4 cores\nITEMS = HEADS * 6 // COLS  # 9 (head, pass) items per core")
a = a.replace("OBJW = 6 * 704 // 4  # 1056 words: Q / K / V object = 6 groups of 9 bfp panels (648 B) padded to 704 B (64 B-aligned seeks)", "OBJW = 8 * 448 // 4  # 896 words: Q / K objects = 6 groups of 8 panels (576 B); V^T objects = 8 groups of 6 panels padded to 448 B")
a = a.replace('source_file=str(HERE / "attn13.cc")', 'source_file=str(HERE / "attn_tn.cc")').replace('object_file_name=f"attn13_', 'object_file_name=f"attn_tn_')
a = a.replace('f"-DATTN_SCALE={1.0 / 72 ** 0.5}f"', 'f"-DATTN_SCALE={1.0 / HD ** 0.5}f"')
a = a.replace("dict(QT=QT, HD=HD, KC=KC, NKEYS=NKEYS, ITEMS=ITEMS, reps=o.reps, COLS=COLS)", "dict(QT=QT, HD=HD, KC=KC, NKEYS=NKEYS, ITEMS=ITEMS, reps=o.reps, COLS=COLS, HEADS=HEADS, QTILES=QTILES)")
for s in ("HD, KC, NKEYS = 48, 64, 48, 1072", "ITEMS = HEADS * 6 // COLS", "OBJW = 8 * 448 // 4", "attn_tn.cc", "HD ** 0.5", "HEADS=HEADS"):
    assert s in a, s
Path("attn_tn.py").write_text(a)
r = Path("runattn_tn.py").read_text()
r = r.replace("NCH = (NK + KC - 1) // KC; NQT = 768 // QT; H = 16", "NCH = (NK + KC - 1) // KC; H = cfg.get(\"HEADS\", 16); QTILES = cfg.get(\"QTILES\", 16); NQT = -(-QTILES // 4) * 4  # tiles incl. the padding tiles of the last pass")
r = r.replace("def qk_obj(t):  # 48 x 72 (rows x dims) -> [6 row blocks][9 dim panels], 704 B per group\n    return groups([to_panels(blocked(t[rb * 8:(rb + 1) * 8])) for rb in range(t.shape[0] // 8)], 704, 4224)",
              "GQ, GV, OBJB = (576, 448, 3584) if HD == 64 else (704, 448, 4224)\ndef qk_obj(t):  # 48 x HD (rows x dims) -> [6 row blocks][HD/8 dim panels], GQ B per group\n    return groups([to_panels(blocked(t[rb * 8:(rb + 1) * 8])) for rb in range(t.shape[0] // 8)], GQ, OBJB)")
r = r.replace("                   for j in range(v.shape[1] // 8)], 448, 4224)", "                   for j in range(v.shape[1] // 8)], GV, OBJB)")
r = r.replace("obj = 4224 // 4  # words per bfp object (6 padded groups)", "obj = OBJB // 4  # words per bfp object")
r = r.replace("def item_map(c, i, r):  # column c: heads 2c + i // 4; pass i % 4: core r takes query tile 4 * (i % 4) + r\n    return 2 * c + i // 4, 4 * (i % 4) + r",
              "PASSES = NQT // 4\ndef item_map(c, i, r):  # (head, pass) items dealt round-robin to the columns; core r takes query tile 4 * pass + r (>= QTILES: padding)\n    g = c + COLS * i\n    return g // PASSES, 4 * (g % PASSES) + r")
for s in ("QTILES = cfg.get", "GQ, GV, OBJB =", "PASSES = NQT // 4"):
    assert s in r, s
Path("runattn_tn.py").write_text(r)
print("patched")
