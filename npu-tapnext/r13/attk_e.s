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
	vldb.fill.512 [p1, lf1, r25]
	lda p0, [sp, #-64]
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
	nop
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24]
	nop
	vldb.pop.576 ex1, [p1, lf1, r25]
	nop
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24]
	vldb.pop.576 ex4, [p1, lf1, r25];		vmul.f dm1, ex3, ex5, r5
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vmul.f dm2, ex0, ex1, r5
	vlda.pop.576 ex6, [p0, lf0, r24]
	vmac.f dm1, dm1, ex4, ex5, r5
	vmac.f dm2, dm2, ex6, ex7, r5
	nop
	vmac.f dm1, dm1, ex2, ex4, r5
	vmac.f dm2, dm2, ex6, ex7, r5
	vlda x5, [p3, #64]
	vlda x1, [p3, #0];		vldb.fill.512 [p1, lf1, r25];		vmac.f dm1, dm1, ex0, ex5, r5
	lda p0, [sp, #-64];		vmac.f dm2, dm2, ex6, ex4, r5
	nop
	nop
	mov r24, #0;		vadd.f dm3, dm1, dm2, r6
	nop
	nop
	vconv.fp32.bf16 cml1, x5
	nop
	vconv.fp32.bf16 cml2, x1
	vconv.bf16.fp32 x2, cmh3
	vconv.bf16.fp32 x3, cml3;		vadd.f dm1, dm3, dm1, r6
	vconv.fp32.bf16 cml3, x2
	vst x3, [p4], #64;		vadd.f dm3, dm3, dm2, r6
	vlda.fill.512 [p0, lf0, r24];		vst x2, [p4], #64;		vconv.fp32.bf16 cml3, x3
	nop
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24]
	nop
	vldb.pop.576 ex1, [p1, lf1, r25];		vconv.bf16.fp32 x6, cml3
	nop
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		vmin_ge.bf16 x6, r16, x9, x6
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24];		vmax_lt.bf16 x2, r16, x10, x6
	vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x3, cml1;		vmul.f dm1, ex3, ex5, r5
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vfloor.s32.bf16 x4, wl2, s0
	vldb.pop.576 ex5, [p1, lf1, r25];		vfloor.s32.bf16 x3, wh2, s0;		vmin_ge.bf16 x2, r16, x9, x3
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vadd.32 x0, x4, x11;		vmul.f dm2, ex0, ex1, r5
	vlda.pop.576 ex6, [p0, lf0, r24];		vadd.32 x1, x3, x11
	vldb.4x64.hi wh2, wh0;		vmax_lt.bf16 x3, r16, x10, x2;		vmac.f dm1, dm1, ex4, ex5, r5
	vldb.4x64.lo wl0, wl1;		vmac.f dm2, dm2, ex6, ex7, r5
	vldb.4x64.lo wl3, wl0
	vldb.4x64.hi wh3, wl0;		vmac.f dm1, dm1, ex2, ex4, r5
	vldb.4x64.lo wl2, wh0;		vmac.f dm2, dm2, ex6, ex7, r5
	vlda x5, [p3, #64]
	.p2align	4
attk_start:
	vlda x1, [p3, #0];		vldb.fill.512 [p1, lf1, r25];		vfloor.s32.bf16 x5, wl3, s0;		nopx;		nopm;		vmac.f dm1, dm1, ex0, ex5, r5
	lda p0, [sp, #-64];		nopb;		vfloor.s32.bf16 x4, wh3, s0;		nopx;		nopm;		vmac.f dm2, dm2, ex6, ex4, r5
	nopa;		vldb.4x64.lo wl5, wh1;		nops;		nopx;		vadd.32 x6, x5, x11;		nopv
	nopa;		vldb.4x64.hi wh5, wh1;		nops;		nopx;		vadd.32 x4, x4, x11;		nopv
	nopa;		vldb.4x64.lo wl1, wl6;		nops;		nopx;		mov r24, #0;		vadd.f dm3, dm1, dm2, r6
	nopa;		vldb.4x64.hi wh0, wl1;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh1, wl6;		nops;		nopx;		vshuffle bmll0, x3, x2, r3;		nopv
	nopa;		vldb.4x64.lo wl4, wh6;		nops;		nopx;		vconv.fp32.bf16 cml1, x5;		nopv
	nopa;		vldb.4x64.hi wh4, wh6;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh2, wl4;		nops;		nopx;		vconv.fp32.bf16 cml2, x1;		nopv
	nopa;		vldb.4x64.lo wl3, wh4;		vconv.bf16.fp32 x2, cmh3;		nopx;		vshuffle x6, x3, x2, r2;		nopv
	nopa;		vldb.4x64.hi wh3, wh4;		vconv.bf16.fp32 x3, cml3;		nopx;		nopm;		vadd.f dm1, dm3, dm1, r6
	nopa;		vldb.4x64.lo wl2, wl4;		nops;		nopx;		vconv.fp32.bf16 cml3, x2;		nopv
	vlda x4, [p3, #0];		nopb;		vst x3, [p4], #64;		nopx;		vshuffle bmlh0, x0, x5, r3;		vadd.f dm3, dm3, dm2, r6
	vlda.fill.512 [p0, lf0, r24];		nopb;		vst x2, [p4], #64;		nopx;		vconv.fp32.bf16 cml3, x3;		nopv
	vlda x5, [p3, #64];		vldb x4, [p5], #64;		nops;		nopx;		vshuffle x0, x0, x5, r2;		nopv
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		vmov wl0, wh6;		nopv
	vlda.pop.576 ex0, [p0, lf0, r24];		nopb;		nops;		nopx;		vshuffle bmll2, x1, x4, r3;		nopv
	nopa;		vldb x4, [p5], #64;		nops;		nopx;		vshuffle x5, x1, x4, r2;		nopv
	nopa;		vldb.pop.576 ex1, [p1, lf1, r25];		vconv.bf16.fp32 x6, cml3;		nopx;		vshuffle x1, x2, x3, r2;		nopv
	nopa;		nopb;		nops;		nopx;		vmov wl1, wh5;		vmac.f dm0, dm0, x0, x4, r1
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		vmin_ge.bf16 x6, r16, x9, x6;		nopv
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		vshuffle bmlh2, x2, x3, r3;		vmac.f dm1, dm2, x1, x5, r1
	vlda.pop.576 ex2, [p0, lf0, r24];		nopb;		nops;		nopx;		vmax_lt.bf16 x2, r16, x10, x6;		vmac.f dm0, dm0, x0, x4, r1
	nopa;		vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x3, cml1;		nopx;		nopm;		vmul.f dm1, ex3, ex5, r5
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vfloor.s32.bf16 x4, wl2, s0;		nopx;		nopm;		vmac.f dm2, dm1, x1, x4, r1
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		vfloor.s32.bf16 x3, wh2, s0;		nopx;		vmin_ge.bf16 x2, r16, x9, x3;		nopv
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		nops;		nopx;		vadd.32 x0, x4, x11;		vmul.f dm2, ex0, ex1, r5
	vlda.pop.576 ex6, [p0, lf0, r24];		nopb;		nops;		nopx;		vadd.32 x1, x3, x11;		nopv
	nopa;		vldb.4x64.hi wh2, wh0;		nops;		nopx;		vmax_lt.bf16 x3, r16, x10, x2;		vmac.f dm1, dm1, ex4, ex5, r5
	nopa;		vldb.4x64.lo wl0, wl1;		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex6, ex7, r5
	nopa;		vldb.4x64.lo wl3, wl0;		nops;		nopx;		vmov cmh0, cml2;		nopv
	nopa;		vldb.4x64.hi wh3, wl0;		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex2, ex4, r5
	nopa;		vldb.4x64.lo wl2, wh0;		vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26];		nopx;		nopm;		vmac.f dm2, dm2, ex6, ex7, r5
attk_last:
	vlda x5, [p3, #64];		nopb;		vst.flush.512.conv [p2, sf, r26];		nopx;		nopm;		nopv
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
