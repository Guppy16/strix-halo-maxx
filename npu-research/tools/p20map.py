"""p20map: the conflict-free accumulator piece map for P20's mm_kstep on the full12r8c placement (npu-research rev 4).

A piece = one group's accumulators = (row block rb, column pair) = 8 rows x 32 columns fp32 = 1 KB (16 quarters of 64 B).
Group order in a k-step: stripe j = 0..8, groups G = (rb 2j, pair 0), (2j, 1), (2j+1, 0), (2j+1, 1). At every boundary the
next piece's 16 loads co-issue with the previous piece's 16 stores, and the B pops (bank 3) with both; the A pops of row
block 1 co-issue with group 1's stores, the next stripe's A pops with group 3's stores (P20 pipelines that boundary).
Rule (MEASURED, tools/bankmodel.py): a bundle costs max over banks of its accesses in that bank.

Placement (full12r8c core 0,2; input_with_addresses.mlir + llvm-nm): C object 0x78900 (12 pieces C0..C11 at +1 KB s,
bank 2), acc_hi 0x71900 (24 pieces H0..H23 at +1 KB h: H0-H8 bank 0, H9 straddles 0/1 (last 256 B in bank 1), H10-H23
bank 1), A ring 0x77B80 / 0x78000 / 0x78480 (banks 1, 2, 2; contiguous, stride 1152), B fifo bank 3.
In-place epilogue rule: C slot s may only hold a piece of row block <= s (output rb is written over C bytes [rb KB, +1 KB)).

usage: python3 p20map.py        # checks the map, prints the table in k-step order and the epilogue's (rb, pair) map
"""

C0, H0 = 0x78900, 0x71900
A_SLOT_BANK = [1, 2, 2]
B_BANK = 3


def bank_of(addr, size=1024):
    return {((addr + o) - 0x70000) // 0x4000 for o in (0, size - 1)}


def piece_addr(p):
    kind, i = p[0], int(p[1:])
    return (C0 if kind == "C" else H0) + 1024 * i


# (stripe j) -> pieces of groups 0..3; stripes 0-5: pair-0 pieces in the C object at slot = row block
X = {0: "H6", 1: "H16", 2: "H17", 3: "H7", 4: "H18", 5: "H19"}  # (2j, pair 1)
Y = {0: "H20", 1: "H21", 2: "H9", 3: "H22", 4: "H23", 5: "H8"}  # (2j+1, pair 1)
MAP = {j: [f"C{2 * j}", X[j], f"C{2 * j + 1}", Y[j]] for j in range(6)}
MAP[6] = ["H10", "H0", "H11", "H1"]
MAP[7] = ["H12", "H2", "H13", "H3"]
MAP[8] = ["H14", "H4", "H15", "H5"]


def main():
    seq = [(j, g, MAP[j][g]) for j in range(9) for g in range(4)]
    assert len({p for _, _, p in seq}) == 36
    stalls = 0
    for (j, g, p), (j2, g2, p2) in zip(seq, seq[1:]):
        bs, bl = bank_of(piece_addr(p)), bank_of(piece_addr(p2))
        common = bs & bl
        if common:
            print(f"  CONFLICT stores {p} (banks {bs}) with loads {p2} (banks {bl})")
            stalls += 16
        assert B_BANK not in bs | bl
        # A pops co-issued with stores: row block 1 of stripe j at its G1 -> G2 boundary, stripe j+1 at G3 -> G0
        if g in (1, 3):
            ja = j if g == 1 else j + 1
            if ja < 9 and A_SLOT_BANK[ja % 3] in bs:
                part = 4 if len(bs) > 1 else 9  # straddling piece: only its last 4 quarters are in the other bank
                print(f"  A-store overlap: stripe {ja} A pops (bank {A_SLOT_BANK[ja % 3]}) with stores of {p} {sorted(bs)}: <= {part} cycles")
                stalls += part
    # in-place rule
    for j in range(9):
        for g, p in enumerate(MAP[j]):
            rb = 2 * j + g // 2
            if p[0] == "C":
                assert int(p[1:]) >= rb and int(p[1:]) == rb, (p, rb)
    print(f"map OK: no Cl/Cs bank pair, nothing in B's bank 3; worst-case residual {stalls} cycles per k-step")
    print("\nk-step order (the table mm_kstep walks; one 32-bit L1 address per group):")
    for j in range(9):
        row = ", ".join(f"0x{piece_addr(p):05x} ({p}, bank {'/'.join(map(str, sorted(bank_of(piece_addr(p)))))})" for p in MAP[j])
        print(f"  stripe {j}: {row}")
    print("\nepilogue map acc_piece(rb, pair) (rb = 0..17):")
    for rb in range(18):
        j, h = divmod(rb, 2)
        print(f"  rb {rb:2d}: pair0 0x{piece_addr(MAP[j][2 * h]):05x}  pair1 0x{piece_addr(MAP[j][2 * h + 1]):05x}")


if __name__ == "__main__":
    main()
