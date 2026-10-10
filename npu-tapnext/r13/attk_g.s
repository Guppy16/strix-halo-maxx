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
	paddxm	[sp], #64
	st	p0, [sp, #-64]
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
	vadd.32	x11, x3, x2
	movxm	r2, #17039
	vbcst.16	x9, r2
	movxm	r2, #-15696
	vbcst.16	x10, r2
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
	// modulo schedule: II 35, U 1, stages 3, ops 92
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25]
	nop
	nop
	mov r24, #0
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vlda.fill.512 [p0, lf0, r24]
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vldb.pop.576 ex2, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25]
	vlda.pop.576 ex5, [p0, lf0, r24]
	vlda.pop.576 ex2, [p0, lf0, r24]
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex6, [p1, lf1, r25]
	vlda.pop.576 ex7, [p0, lf0, r24]
	vmul.f dm0, ex5, ex6, r5
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		vmul.f dm2, ex0, ex2, r5
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm0, ex5, ex1, r5
	vlda x0, [p3, #0];		vmac.f dm2, dm2, ex2, ex6, r5
	vlda x2, [p3, #64]
	vmac.f dm0, dm0, ex4, ex5, r5
	vmac.f dm2, dm2, ex7, ex6, r5
	nop
	vmac.f dm0, dm0, ex4, ex5, r5
	vmac.f dm2, dm2, ex6, ex7, r5
	vconv.fp32.bf16 cml1, x0
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25]
	vadd.f dm2, dm0, dm2, r6
	nop
	mov r24, #0
	nop
	nop
	nop
	vconv.bf16.fp32 x8, cml2
	vadd.f dm2, dm3, dm1, r6
	vconv.fp32.bf16 cml3, x8
	nop
	vconv.fp32.bf16 cml3, x2
	vconv.bf16.fp32 x4, cmh2
	vlda.fill.512 [p0, lf0, r24];		vst x8, [p4], #64
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vldb.pop.576 ex2, [p1, lf1, r25];		vadd.f dm2, dm2, dm3, r6
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25];		vconv.bf16.fp32 x0, cml2;		vconv.fp32.bf16 cml2, x4
	vlda.pop.576 ex5, [p0, lf0, r24];		vst x4, [p4], #64
	vlda.pop.576 ex2, [p0, lf0, r24]
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex6, [p1, lf1, r25];		vconv.bf16.fp32 x7, cml2
	vlda.pop.576 ex7, [p0, lf0, r24];		vmin_ge.bf16 x3, r16, x9, x0
	vmin_ge.bf16 x6, r16, x9, x7;		vmul.f dm0, ex5, ex6, r5
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		vmax_lt.bf16 x0, r16, x10, x3;		vmul.f dm2, ex0, ex2, r5
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vmax_lt.bf16 x3, r16, x10, x6
	vfloor.s32.bf16 x7, wh0, s0;		vmac.f dm0, dm0, ex5, ex1, r5
	vlda x0, [p3, #0];		vfloor.s32.bf16 x1, wh3, s0;		vmac.f dm2, dm2, ex2, ex6, r5
	vlda x2, [p3, #64];		vadd.32 x2, x7, x11
	vlda x0, [p3, #0];		vfloor.s32.bf16 x0, wl0, s0;		vmac.f dm0, dm0, ex4, ex5, r5
	vldb.4x64.hi wh5, wh2;		vmac.f dm2, dm2, ex7, ex6, r5
	vldb.4x64.lo wl5, wh2
	vldb.4x64.hi wh7, wl2;		vadd.32 x4, x0, x11;		vmac.f dm0, dm0, ex4, ex5, r5
	vldb.4x64.lo wl7, wl2;		vmac.f dm2, dm2, ex6, ex7, r5
	vldb.4x64.hi wh3, wh4;		vconv.fp32.bf16 cml1, x0
	.p2align	4
attk_start:
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25];		vfloor.s32.bf16 x3, wl3, s0;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.lo wl7, wl4;		nops;		nopx;		nopm;		vadd.f dm2, dm0, dm2, r6
	nopa;		vldb.4x64.hi wh7, wl4;		nops;		nopx;		vadd.32 x6, x3, x11;		nopv
	nopa;		vldb.4x64.lo wl3, wh4;		nops;		nopx;		mov r24, #0;		nopv
	nopa;		vldb.4x64.lo wl5, wl6;		nops;		nopx;		vadd.32 x4, x1, x11;		nopv
	nopa;		vldb.4x64.hi wh5, wl6;		nops;		nopx;		vshuffle x1, x7, x5, r2;		nopv
	nopa;		vldb.4x64.hi wh2, wh4;		nops;		nopx;		vshuffle bmlh0, x7, x5, r3;		nopv
	nopa;		vldb.4x64.lo wl6, wh6;		vconv.bf16.fp32 x8, cml2;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh6, wh6;		nops;		nopx;		nopm;		vadd.f dm2, dm3, dm1, r6
	nopa;		vldb.4x64.lo wl2, wh4;		nops;		nopx;		vconv.fp32.bf16 cml3, x8;		nopv
	vlda x3, [p3, #64];		nopb;		nops;		nopx;		vshuffle bmll0, x7, x3, r3;		nopv
	nopa;		vldb.4x64.hi wh7, wl4;		nops;		nopx;		vconv.fp32.bf16 cml3, x2;		nopv
	nopa;		vldb.4x64.lo wl7, wl4;		vconv.bf16.fp32 x4, cmh2;		nopx;		vshuffle x3, x7, x3, r2;		nopv
	vlda.fill.512 [p0, lf0, r24];		vldb x6, [p5], #64;		vst x8, [p4], #64;		nopx;		nopm;		nopv
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		nops;		nopx;		vmov wl1, wh3;		nopv
	nopa;		vldb.pop.576 ex2, [p1, lf1, r25];		nops;		nopx;		vshuffle bmll1, x5, x6, r3;		vadd.f dm2, dm2, dm3, r6
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25];		vconv.bf16.fp32 x0, cml2;		nopx;		vconv.fp32.bf16 cml2, x4;		vmac.f dm0, dm0, x1, x0, r1
	vlda.pop.576 ex5, [p0, lf0, r24];		nopb;		vst x4, [p4], #64;		nopx;		vshuffle x5, x5, x6, r2;		nopv
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb x7, [p5], #64;		nops;		nopx;		nopm;		nopv
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		nops;		nopx;		vshuffle x4, x7, x2, r2;		nopv
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		vmov wl4, wh5;		vmac.f dm3, dm0, x1, x6, r1
	nopa;		vldb.pop.576 ex6, [p1, lf1, r25];		vconv.bf16.fp32 x7, cml2;		nopx;		vshuffle bmlh1, x7, x2, r3;		nopv
	vlda.pop.576 ex7, [p0, lf0, r24];		nopb;		nops;		nopx;		vmin_ge.bf16 x3, r16, x9, x0;		vmac.f dm1, dm1, x4, x3, r1
	nopa;		nopb;		nops;		nopx;		vmin_ge.bf16 x6, r16, x9, x7;		vmul.f dm0, ex5, ex6, r5
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		vmax_lt.bf16 x0, r16, x10, x3;		vmul.f dm2, ex0, ex2, r5
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		vmax_lt.bf16 x3, r16, x10, x6;		vmac.f dm1, dm1, x4, x7, r1
	nopa;		nopb;		vfloor.s32.bf16 x7, wh0, s0;		nopx;		nopm;		vmac.f dm0, dm0, ex5, ex1, r5
	vlda x0, [p3, #0];		nopb;		vfloor.s32.bf16 x1, wh3, s0;		nopx;		nopm;		vmac.f dm2, dm2, ex2, ex6, r5
	vlda x2, [p3, #64];		nopb;		nops;		nopx;		vadd.32 x2, x7, x11;		nopv
	vlda x0, [p3, #0];		nopb;		vfloor.s32.bf16 x0, wl0, s0;		nopx;		nopm;		vmac.f dm0, dm0, ex4, ex5, r5
	nopa;		vldb.4x64.hi wh5, wh2;		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex7, ex6, r5
	nopa;		vldb.4x64.lo wl5, wh2;		nops;		nopx;		vmov cmh3, cml1;		nopv
	nopa;		vldb.4x64.hi wh7, wl2;		nops;		nopx;		vadd.32 x4, x0, x11;		vmac.f dm0, dm0, ex4, ex5, r5
	nopa;		vldb.4x64.lo wl7, wl2;		vst.push.576.conv.bfp16ebs8.fp32 dm3, [p2, sf, r26];		nopx;		nopm;		vmac.f dm2, dm2, ex6, ex7, r5
attk_last:
	nopa;		vldb.4x64.hi wh3, wh4;		vst.flush.512.conv [p2, sf, r26];		nopx;		vconv.fp32.bf16 cml1, x0;		nopv
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
	paddxm	[sp], #-64
	nop
.Lfunc_end_attk:
	.size	attk, .Lfunc_end_attk-attk
