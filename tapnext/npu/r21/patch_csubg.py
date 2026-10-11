# round 21 CSUB-G: the gated tiles (flags 2 / 3) join the CSUB group loop
p = "mm8.cc"
s = open(p).read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)
rep('''#ifdef GUP  // round 12 (npu-research G1a): the up half''', '''#if defined(GUP) && !defined(CSUB)  // (CSUB: the group loop below) round 12 (npu-research G1a): the up half''')
rep('''#ifdef CSUB
  if (flag != 4)
#else''', '''#ifdef CSUB
  if (flag >= 5)
#else''')
rep('''    for (int g = 0; g < 3; g++) {
      bfloat16 *v = g < 2 ? c_out + 3072 * g : (bfloat16 *)acc_hi, *u = (bfloat16 *)(acc_hi + 256 * (6 + 6 * g));
      if (flag == 4) {
        bgelu((const float *)v, v, 8 * 6 / BGELU_U, serf_ab, serf_cd, bias);
        bgelu((const float *)u, u, 8 * 6 / BGELU_U, serf_ab, serf_cd, bias + 256);
      }
      pan(v, u, (bfp16ebs8 *)((uint8_t *)c_out + (flag == 4 ? 3456 : 4608) * g), 6 / PAN_U, jump, 5);
      release(c_out, 48 + CS_CONS, 1);
    }''', '''    // Gated (2 / 3, r12 gup): the gate half's bias + GELU (pair 0) per group, gup's panels (flag 2 into the stash in part 2, no
    // release: the object continues; flag 3 at 576 B rows) and flag 3's stash copy behind them, then the release.
#pragma clang loop unroll(disable)
    for (int g = 0; g < 3; g++) {
      bfloat16 *v = g < 2 ? c_out + 3072 * g : (bfloat16 *)acc_hi, *u = (bfloat16 *)(acc_hi + 256 * (6 + 6 * g));
      uint8_t *const stash = (uint8_t *)c_out + STASH_OFF + 6 * STASH_STRIDE * g;
      if (flag <= 3) {
        bgelu((const float *)v, v, 8 * 6 / BGELU_U, sgelu_ab, sgelu_cd, bias);
        uint8_t *const dst = flag == 2 ? stash : (uint8_t *)c_out + 3456 * g;
        gup(v, (const float *)u, (bfp16ebs8 *)dst, 24 / GUP_U, bias + 256, (flag == 2 ? STASH_STRIDE : 8 * PB) - 4 * PB);
        if (flag == 2) continue;
        cpy32(stash, dst + 4 * PB, 6 * 9 / CPY32_U, STASH_STRIDE - 8 * 32, 8 * PB - 8 * 32, 8);
      } else {
        if (flag == 4) {
          bgelu((const float *)v, v, 8 * 6 / BGELU_U, serf_ab, serf_cd, bias);
          bgelu((const float *)u, u, 8 * 6 / BGELU_U, serf_ab, serf_cd, bias + 256);
        }
        pan(v, u, (bfp16ebs8 *)((uint8_t *)c_out + (flag == 4 ? 3456 : 4608) * g), 6 / PAN_U, jump, 5);
      }
      release(c_out, 48 + CS_CONS, 1);
    }''')
rep('''#ifdef CSUB  // round 21: every other epilogue released its three sub-parts itself
    if (flag == 3) release(c_out, 48 + CS_CONS, 3);
#else''', '''#ifdef CSUB  // round 21: every epilogue released its three sub-parts itself
    (void)flag;
#else''')
open(p, "w").write(s)
print("CSUB-G applied")
