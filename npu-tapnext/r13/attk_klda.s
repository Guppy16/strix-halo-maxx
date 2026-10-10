	.text
	.section	.text.attk,"ax",@progbits
	.globl	attk
	.p2align	4
	.type	attk,@function
attk:
	mova	r5, #780
	mova	r6, #60
	mova	r24, #0
	mova	r25, #0
	vldb.fill.512	[p0, lf1, r25]
	vldb.pop.576	ex0, [p0, lf1, r25]
	vldb.pop.576	ex1, [p0, lf1, r25]
	vldb.pop.576	ex2, [p0, lf1, r25]
	vldb.pop.576	ex3, [p0, lf1, r25]
	vldb.pop.576	ex4, [p0, lf1, r25]
	vldb.pop.576	ex5, [p0, lf1, r25]
	vldb.pop.576	ex6, [p0, lf1, r25]
	vldb.pop.576	ex7, [p0, lf1, r25]
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
	mov	p5, p4
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
	vlda.fill.512 [p1, lf0, r24]
	vlda.pop.576 ex8, [p1, lf0, r24]
	vlda.pop.576 ex9, [p1, lf0, r24]
	vlda.pop.576 ex8, [p1, lf0, r24]
	vlda.pop.576 ex9, [p1, lf0, r24]
	nop
	nop
	vlda.pop.576 ex8, [p1, lf0, r24]
	vlda.pop.576 ex9, [p1, lf0, r24]
	vmul.f dm1, ex0, ex8, r5
	vlda.pop.576 ex8, [p1, lf0, r24];		vmul.f dm2, ex1, ex9, r5
	vlda.pop.576 ex9, [p1, lf0, r24]
	vmac.f dm1, dm1, ex2, ex8, r5
	vmac.f dm2, dm2, ex3, ex9, r5
	nop
	vmac.f dm1, dm1, ex4, ex8, r5
	vmac.f dm2, dm2, ex5, ex9, r5
	nop
	vmac.f dm1, dm1, ex6, ex8, r5
	vmac.f dm2, dm2, ex7, ex9, r5
	vlda x1, [p3, #64]
	nop
	vadd.f dm0, dm1, dm2, r6
	nop
	nop
	nop
	nop
	nop
	vconv.bf16.fp32 x2, cmh0
	nop
	vlda x6, [p3, #0];		vmul.f dm0, x2, x11, r1
	nop
	nop
	vlda.fill.512 [p1, lf0, r24];		vconv.bf16.fp32 x7, cml0;		vmsc.f dm0, dm0, x1, x11, r1
	vlda.pop.576 ex8, [p1, lf0, r24]
	vlda.pop.576 ex9, [p1, lf0, r24];		vst x7, [p4], #64;		vmul.f dm2, x7, x11, r1
	vlda.pop.576 ex8, [p1, lf0, r24]
	vlda.pop.576 ex9, [p1, lf0, r24];		vst x2, [p4], #64
	vmsc.f dm2, dm2, x6, x11, r1
	vconv.bf16.fp32 x6, cml0
	vlda.pop.576 ex8, [p1, lf0, r24]
	vlda.pop.576 ex9, [p1, lf0, r24];		vmin_ge.bf16 x6, r16, x8, x6
	vmul.f dm1, ex0, ex8, r5
	vlda.pop.576 ex8, [p1, lf0, r24];		vmax_lt.bf16 x5, r16, x9, x6;		vmul.f dm2, ex1, ex9, r5
	vlda.pop.576 ex9, [p1, lf0, r24];		vconv.bf16.fp32 x4, cml2
	vfloor.s32.bf16 x6, wl5, s0;		vmac.f dm1, dm1, ex2, ex8, r5
	vmin_ge.bf16 x4, r16, x8, x4;		vmac.f dm2, dm2, ex3, ex9, r5
	vadd.32 x6, x6, x10
	vfloor.s32.bf16 x5, wh5, s0;		vmax_lt.bf16 x4, r16, x9, x4;		vmac.f dm1, dm1, ex4, ex8, r5
	vldb.4x64.hi wh3, wl6;		vmac.f dm2, dm2, ex5, ex9, r5
	vldb.4x64.lo wl7, wh6
	vldb.4x64.lo wl3, wl6;		vmac.f dm1, dm1, ex6, ex8, r5
	vldb.4x64.hi wh7, wh6;		vadd.32 x0, x5, x10;		vmac.f dm2, dm2, ex7, ex9, r5
	vlda x1, [p3, #64];		vfloor.s32.bf16 x5, wh4, s0
	vldb.4x64.lo wl5, wh0
	vfloor.s32.bf16 x4, wl4, s0;		vadd.f dm0, dm1, dm2, r6
	vldb.4x64.lo wl7, wl0
	vldb.4x64.hi wh7, wl0;		vadd.32 x4, x4, x10
	vldb.4x64.hi wh5, wh0;		vadd.32 x0, x5, x10
	vldb.4x64.lo wl3, wh4
	vldb.4x64.hi wh3, wh4;		vshuffle x6, x3, x7, r2
	vldb.4x64.lo wl5, wl4;		vconv.bf16.fp32 x2, cmh0;		vshuffle bmll1, x3, x7, r3
	vldb.4x64.hi wh1, wl0
	vlda x6, [p3, #0];		vmul.f dm0, x2, x11, r1
	vldb.4x64.hi wh5, wl4
	vlda x2, [p3, #64];		vshuffle bmlh1, x7, x5, r3
	.p2align	4
attk_start:
	vlda.fill.512 [p1, lf0, r24];		nopb;		vconv.bf16.fp32 x7, cml0;		nopx;		vshuffle x4, x7, x5, r2;		vmsc.f dm0, dm0, x1, x11, r1
	vlda.pop.576 ex8, [p1, lf0, r24];		nopb;		nops;		nopx;		vmov wl4, wh6;		nopv
	vlda.pop.576 ex9, [p1, lf0, r24];		nopb;		vst x7, [p4], #64;		nopx;		nopm;		vmul.f dm2, x7, x11, r1
	vlda.pop.576 ex8, [p1, lf0, r24];		vldb x2, [p5], #64;		nops;		nopx;		nopm;		nopv
	vlda.pop.576 ex9, [p1, lf0, r24];		vldb x7, [p5], #64;		vst x2, [p4], #64;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.lo wl3, wh0;		nops;		nopx;		nopm;		vmsc.f dm2, dm2, x6, x11, r1
	nopa;		vldb.4x64.hi wh3, wh0;		vconv.bf16.fp32 x6, cml0;		nopx;		nopm;		nopv
	vlda.pop.576 ex8, [p1, lf0, r24];		nopb;		nops;		nopx;		vshuffle bmll0, x5, x3, r3;		nopv
	vlda.pop.576 ex9, [p1, lf0, r24];		nopb;		nops;		nopx;		vmin_ge.bf16 x6, r16, x8, x6;		vmsc.f dm2, dm1, x4, x2, r1
	nopa;		vldb.4x64.lo wl1, wl0;		nops;		nopx;		vshuffle x0, x5, x3, r2;		vmul.f dm1, ex0, ex8, r5
	vlda.pop.576 ex8, [p1, lf0, r24];		nopb;		nops;		nopx;		vmax_lt.bf16 x5, r16, x9, x6;		vmul.f dm2, ex1, ex9, r5
	vlda.pop.576 ex9, [p1, lf0, r24];		nopb;		vconv.bf16.fp32 x4, cml2;		nopx;		nopm;		vmac.f dm3, dm2, x4, x7, r1
	nopa;		nopb;		vfloor.s32.bf16 x6, wl5, s0;		nopx;		nopm;		vmac.f dm1, dm1, ex2, ex8, r5
	vlda x3, [p3, #0];		nopb;		nops;		nopx;		vmin_ge.bf16 x4, r16, x8, x4;		vmac.f dm2, dm2, ex3, ex9, r5
	nopa;		nopb;		nops;		nopx;		vadd.32 x6, x6, x10;		nopv
	nopa;		nopb;		vfloor.s32.bf16 x5, wh5, s0;		nopx;		vmax_lt.bf16 x4, r16, x9, x4;		vmac.f dm1, dm1, ex4, ex8, r5
	nopa;		vldb.4x64.hi wh3, wl6;		nops;		nopx;		vshuffle bmlh0, x1, x3, r3;		vmac.f dm2, dm2, ex5, ex9, r5
	nopa;		vldb.4x64.lo wl7, wh6;		nops;		nopx;		vshuffle x1, x1, x3, r2;		nopv
	nopa;		vldb.4x64.lo wl3, wl6;		nops;		nopx;		vmov wl1, wh0;		vmac.f dm1, dm1, ex6, ex8, r5
	nopa;		vldb.4x64.hi wh7, wh6;		nops;		nopx;		vadd.32 x0, x5, x10;		vmac.f dm2, dm2, ex7, ex9, r5
	vlda x1, [p3, #64];		nopb;		vfloor.s32.bf16 x5, wh4, s0;		nopx;		nopm;		vmsc.f dm0, dm0, x1, x3, r1
	nopa;		vldb.4x64.lo wl5, wh0;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		vfloor.s32.bf16 x4, wl4, s0;		nopx;		nopm;		vadd.f dm0, dm1, dm2, r6
	nopa;		vldb.4x64.lo wl7, wl0;		nops;		nopx;		nopm;		vmac.f dm2, dm0, x1, x2, r1
	nopa;		vldb.4x64.hi wh7, wl0;		nops;		nopx;		vadd.32 x4, x4, x10;		nopv
	nopa;		vldb.4x64.hi wh5, wh0;		nops;		nopx;		vadd.32 x0, x5, x10;		nopv
	nopa;		vldb.4x64.lo wl3, wh4;		nops;		nopx;		vmov cmh2, cml3;		nopv
	nopa;		vldb.4x64.hi wh3, wh4;		nops;		nopx;		vshuffle x6, x3, x7, r2;		nopv
	nopa;		vldb.4x64.lo wl5, wl4;		vconv.bf16.fp32 x2, cmh0;		nopx;		vshuffle bmll1, x3, x7, r3;		nopv
	nopa;		vldb.4x64.hi wh1, wl0;		vst.push.576.conv.bfp16ebs8.fp32 dm2, [p2, sf, r26];		nopx;		nopm;		nopv
	vlda x6, [p3, #0];		nopb;		vst.flush.512.conv [p2, sf, r26];		nopx;		nopm;		vmul.f dm0, x2, x11, r1
	nopa;		vldb.4x64.hi wh5, wl4;		nops;		nopx;		nopm;		nopv
attk_last:
	vlda x2, [p3, #64];		nopb;		nops;		nopx;		vshuffle bmlh1, x7, x5, r3;		nopv
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
