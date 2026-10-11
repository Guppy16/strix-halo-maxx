	.text
	.section	.text.attk,"ax",@progbits
	.globl	attk
	.p2align	4
	.type	attk,@function
attk:
	mova	r3, #780
	mova	r4, #60
	mova	r24, #0
	mova	r25, #0
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex0, [p0, lf0, r24]
	vlda.pop.576	ex1, [p0, lf0, r24]
	vlda.pop.576	ex2, [p0, lf0, r24]
	vlda.pop.576	ex3, [p0, lf0, r24]
	vlda.pop.576	ex4, [p0, lf0, r24]
	vlda.pop.576	ex5, [p0, lf0, r24]
	vlda.pop.576	ex6, [p0, lf0, r24]
	vlda.pop.576	ex7, [p0, lf0, r24]
	movxm	r16, #52428
	movxm	r2, #aexp_ab
	vbcst.32	x0, r2
	mov	r2, p5
	vbcst.32	x1, r2
	movxm	r2, #704
	vbcst.32	x2, r2
	vsel.32	x3, x0, x1, r16
	nop
	nop
	vadd.32	x10, x3, x2
	movxm	r2, #17039
	vbcst.16	x8, r2
	movxm	r2, #-15696
	vbcst.16	x9, r2
	movxm	r2, #16256
	vbcst.16	x11, r2
	mova	r1, #828
	mova	r2, #24
	mova	r3, #5
	mov	s0, #3
	mova	r26, #0
	mov	p5, p2
	movxm	ls, #attk_start
	movxm	le, #attk_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 33, U 1, stages 3, ops 79
	vldb.fill.512 [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	nop
	nop
	vldb.pop.576 ex8, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmul.f dm2, ex0, ex8, r3
	vldb.pop.576 ex8, [p1, lf1, r25];		vmul.f dm0, ex1, ex9, r3
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm2, dm2, ex2, ex8, r3
	vmac.f dm0, dm0, ex3, ex9, r3
	nop
	vmac.f dm2, dm2, ex4, ex8, r3
	vmac.f dm0, dm0, ex5, ex9, r3
	nop
	vmac.f dm2, dm2, ex6, ex8, r3
	vmac.f dm0, dm0, ex7, ex9, r3
	nop
	vlda x3, [p3, #0]
	vadd.f dm1, dm2, dm0, r4
	nop
	nop
	nop
	nop
	nop
	vconv.bf16.fp32 x5, cml1
	nop
	vmul.f dm1, x5, x11, r1
	nop
	nop
	vlda x6, [p3, #64];		vldb.fill.512 [p1, lf1, r25];		vst x5, [p2], #64;		vmsc.f dm2, dm1, x3, x11, r1
	vldb.pop.576 ex8, [p1, lf1, r25];		vconv.bf16.fp32 x5, cmh1
	vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25];		vst x5, [p2], #64;		vmul.f dm2, x5, x11, r1
	vldb.pop.576 ex9, [p1, lf1, r25]
	nop
	vconv.bf16.fp32 x2, cml2
	vldb.pop.576 ex8, [p1, lf1, r25];		vmsc.f dm3, dm2, x6, x11, r1
	vldb.pop.576 ex9, [p1, lf1, r25];		vmin_ge.bf16 x2, r16, x8, x2
	vmul.f dm2, ex0, ex8, r3
	vldb.pop.576 ex8, [p1, lf1, r25];		vmax_lt.bf16 x2, r16, x9, x2;		vmul.f dm0, ex1, ex9, r3
	vldb.pop.576 ex9, [p1, lf1, r25]
	vfloor.s32.bf16 x6, wl2, s0;		vmac.f dm2, dm2, ex2, ex8, r3
	vmac.f dm0, dm0, ex3, ex9, r3
	vfloor.s32.bf16 x6, wh2, s0;		vadd.32 x2, x6, x10
	vconv.bf16.fp32 x7, cml3;		vmac.f dm2, dm2, ex4, ex8, r3
	vldb.4x64.hi wh4, wh2;		vadd.32 x3, x6, x10;		vmac.f dm0, dm0, ex5, ex9, r3
	vldb.4x64.lo wl4, wh2;		vmin_ge.bf16 x6, r16, x8, x7
	vldb.4x64.lo wl5, wl2;		vmac.f dm2, dm2, ex6, ex8, r3
	vldb.4x64.hi wh0, wl3;		vmax_lt.bf16 x6, r16, x9, x6;		vmac.f dm0, dm0, ex7, ex9, r3
	vldb.4x64.hi wh5, wl2
	vlda x3, [p3, #0];		vfloor.s32.bf16 x0, wl6, s0
	vfloor.s32.bf16 x6, wh6, s0;		vadd.f dm1, dm2, dm0, r4
	vldb.4x64.lo wl0, wl3;		vadd.32 x2, x0, x10
	vldb.4x64.lo wl7, wh3;		vadd.32 x1, x6, x10
	vldb.4x64.hi wh7, wh3
	vldb.4x64.lo wl1, wh1
	vldb.4x64.hi wh1, wh1;		vshuffle bmll0, x5, x4, r3
	vldb.4x64.lo wl6, wl1;		vconv.bf16.fp32 x5, cml1;		vshuffle x4, x5, x4, r2
	vldb.4x64.hi wh3, wh2
	vldb.4x64.hi wh6, wl1;		vmul.f dm1, x5, x11, r1
	vldb.4x64.lo wl5, wl2
	vldb.4x64.lo wl3, wh2
	.p2align	4
attk_start:
	vlda x6, [p3, #64];		vldb.fill.512 [p1, lf1, r25];		vst x5, [p2], #64;		nopx;		vshuffle bmlh0, x0, x7, r3;		vmsc.f dm2, dm1, x3, x11, r1
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		vconv.bf16.fp32 x5, cmh1;		nopx;		vshuffle x0, x0, x7, r2;		nopv
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		nopv
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		vst x5, [p2], #64;		nopx;		nopm;		vmul.f dm2, x5, x11, r1
	vlda x4, [p3, #0];		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		vshuffle bmlh1, x6, x1, r3;		nopv
	nopa;		nopb;		nops;		nopx;		vshuffle x1, x6, x1, r2;		nopv
	nopa;		vldb.4x64.hi wh5, wl2;		vconv.bf16.fp32 x2, cml2;		nopx;		nopm;		nopv
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		vmsc.f dm3, dm2, x6, x11, r1
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		vmin_ge.bf16 x2, r16, x8, x2;		nopv
	nopa;		nopb;		nops;		nopx;		vmov wl0, wh4;		vmul.f dm2, ex0, ex8, r3
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		vmax_lt.bf16 x2, r16, x9, x2;		vmul.f dm0, ex1, ex9, r3
	vlda x5, [p3, #64];		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmsc.f dm3, dm0, x0, x4, r1
	nopa;		nopb;		vfloor.s32.bf16 x6, wl2, s0;		nopx;		nopm;		vmac.f dm2, dm2, ex2, ex8, r3
	nopa;		nopb;		nops;		nopx;		vshuffle x4, x5, x3, r2;		vmac.f dm0, dm0, ex3, ex9, r3
	nopa;		vldb x4, [p5], #64;		vfloor.s32.bf16 x6, wh2, s0;		nopx;		vadd.32 x2, x6, x10;		nopv
	nopa;		vldb x2, [p5], #64;		vconv.bf16.fp32 x7, cml3;		nopx;		vshuffle bmll1, x5, x3, r3;		vmac.f dm2, dm2, ex4, ex8, r3
	nopa;		vldb.4x64.hi wh4, wh2;		nops;		nopx;		vadd.32 x3, x6, x10;		vmac.f dm0, dm0, ex5, ex9, r3
	nopa;		vldb.4x64.lo wl4, wh2;		nops;		nopx;		vmin_ge.bf16 x6, r16, x8, x7;		nopv
	nopa;		vldb.4x64.lo wl5, wl2;		nops;		nopx;		vmov wl1, wh4;		vmac.f dm2, dm2, ex6, ex8, r3
	nopa;		vldb.4x64.hi wh0, wl3;		nops;		nopx;		vmax_lt.bf16 x6, r16, x9, x6;		vmac.f dm0, dm0, ex7, ex9, r3
	nopa;		vldb.4x64.hi wh5, wl2;		nops;		nopx;		nopm;		vmsc.f dm1, dm1, x1, x5, r1
	vlda x3, [p3, #0];		nopb;		vfloor.s32.bf16 x0, wl6, s0;		nopx;		nopm;		vmac.f dm2, dm3, x0, x4, r1
	nopa;		nopb;		vfloor.s32.bf16 x6, wh6, s0;		nopx;		nopm;		vadd.f dm1, dm2, dm0, r4
	nopa;		vldb.4x64.lo wl0, wl3;		nops;		nopx;		vadd.32 x2, x0, x10;		vmac.f dm3, dm1, x1, x2, r1
	nopa;		vldb.4x64.lo wl7, wh3;		nops;		nopx;		vadd.32 x1, x6, x10;		nopv
	nopa;		vldb.4x64.hi wh7, wh3;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.lo wl1, wh1;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh1, wh1;		nops;		nopx;		vshuffle bmll0, x5, x4, r3;		nopv
	nopa;		vldb.4x64.lo wl6, wl1;		vconv.bf16.fp32 x5, cml1;		nopx;		vshuffle x4, x5, x4, r2;		nopv
	nopa;		vldb.4x64.hi wh3, wh2;		nops;		nopx;		vmov cmh2, cml3;		nopv
	nopa;		vldb.4x64.hi wh6, wl1;		nops;		nopx;		nopm;		vmul.f dm1, x5, x11, r1
	nopa;		vldb.4x64.lo wl5, wl2;		vst.push.576.conv.bfp16ebs8.fp32 dm2, [p4, sf, r26];		nopx;		nopm;		nopv
attk_last:
	nopa;		vldb.4x64.lo wl3, wh2;		vst.flush.512.conv [p4, sf, r26];		nopx;		nopm;		nopv
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_attk:
	.size	attk, .Lfunc_end_attk-attk
