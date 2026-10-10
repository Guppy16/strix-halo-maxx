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
	lda p0, [sp, #-64]
	nop
	nop
	nop
	nop
	mov r24, #0
	nop
	vldb.fill.512 [p1, lf1, r25]
	nop
	nop
	vlda.fill.512 [p0, lf0, r24]
	vlda.pop.576 ex2, [p0, lf0, r24]
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24]
	nop
	vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24]
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex4, [p1, lf1, r25];		vmul.f dm2, ex2, ex5, r5
	nop
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vmul.f dm3, ex1, ex5, r5
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24]
	vldb.pop.576 ex2, [p1, lf1, r25];		vmac.f dm2, dm2, ex2, ex7, r5
	vlda x4, [p3, #0]
	vlda x7, [p3, #64];		vmac.f dm3, dm3, ex0, ex7, r5
	vmac.f dm2, dm2, ex2, ex4, r5
	nop
	vmac.f dm3, dm3, ex2, ex4, r5
	vmac.f dm2, dm2, ex7, ex8, r5
	nop
	vconv.fp32.bf16 cml1, x4;		vmac.f dm3, dm3, ex0, ex2, r5
	vconv.fp32.bf16 cml0, x7
	lda p0, [sp, #-64]
	vadd.f dm3, dm2, dm3, r6
	nop
	nop
	nop
	mov r24, #0
	nop
	vldb.fill.512 [p1, lf1, r25];		vconv.bf16.fp32 x3, cml3
	nop
	nop
	vlda.fill.512 [p0, lf0, r24];		vst x3, [p4], #64;		vadd.f dm1, dm3, dm1, r6
	vlda.pop.576 ex2, [p0, lf0, r24];		vconv.bf16.fp32 x3, cmh3;		vconv.fp32.bf16 cml3, x3
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vadd.f dm1, dm3, dm0, r6
	vldb.pop.576 ex5, [p1, lf1, r25];		vst x3, [p4], #64;		vconv.fp32.bf16 cml3, x3
	vlda.pop.576 ex2, [p0, lf0, r24]
	vconv.bf16.fp32 x1, cml1
	vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24];		vmin_ge.bf16 x7, r16, x9, x1
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x7, cml1;		vmax_lt.bf16 x6, r16, x10, x7;		vmul.f dm2, ex2, ex5, r5
	nop
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vfloor.s32.bf16 x5, wh6, s0;		vmin_ge.bf16 x1, r16, x9, x7;		vmul.f dm3, ex1, ex5, r5
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25];		vfloor.s32.bf16 x8, wl6, s0
	vlda.pop.576 ex0, [p0, lf0, r24];		vmax_lt.bf16 x6, r16, x10, x1
	vldb.pop.576 ex2, [p1, lf1, r25];		vadd.32 x1, x8, x11;		vmac.f dm2, dm2, ex2, ex7, r5
	vlda x4, [p3, #0];		vfloor.s32.bf16 x5, wl6, s0;		vadd.32 x3, x5, x11
	vlda x7, [p3, #64];		vfloor.s32.bf16 x0, wh6, s0;		vmac.f dm3, dm3, ex0, ex7, r5
	vldb.4x64.hi wh0, wh1;		vadd.32 x5, x5, x11;		vmac.f dm2, dm2, ex2, ex4, r5
	vldb.4x64.lo wl2, wh3;		vadd.32 x6, x0, x11
	vldb.4x64.lo wl4, wh5;		vmac.f dm3, dm3, ex2, ex4, r5
	vldb.4x64.lo wl6, wl5;		vmac.f dm2, dm2, ex7, ex8, r5
	vldb.4x64.hi wh5, wh6
	vldb.4x64.lo wl5, wh6;		vconv.fp32.bf16 cml1, x4;		vmac.f dm3, dm3, ex0, ex2, r5
	vldb.4x64.hi wh4, wh5;		vconv.fp32.bf16 cml0, x7
	.p2align	4
attk_start:
	lda p0, [sp, #-64];		vldb.4x64.lo wl7, wl6;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh7, wl6;		nops;		nopx;		nopm;		vadd.f dm3, dm2, dm3, r6
	nopa;		vldb.4x64.hi wh6, wl5;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.lo wl0, wh1;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh1, wl3;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.lo wl1, wl3;		nops;		nopx;		mov r24, #0;		nopv
	nopa;		vldb.4x64.hi wh2, wh3;		nops;		nopx;		nopm;		nopv
	vlda x7, [p3, #64];		vldb.fill.512 [p1, lf1, r25];		vconv.bf16.fp32 x3, cml3;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh6, wl1;		nops;		nopx;		vshuffle bmlh2, x7, x5, r3;		nopv
	nopa;		vldb.4x64.lo wl6, wl1;		nops;		nopx;		vshuffle bmll2, x6, x4, r3;		nopv
	vlda.fill.512 [p0, lf0, r24];		vldb x4, [p5], #64;		vst x3, [p4], #64;		nopx;		vshuffle x4, x6, x4, r2;		vadd.f dm1, dm3, dm1, r6
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb x7, [p5], #64;		vconv.bf16.fp32 x3, cmh3;		nopx;		vconv.fp32.bf16 cml3, x3;		nopv
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		vshuffle x5, x7, x5, r2;		nopv
	nopa;		nopb;		nops;		nopx;		vmov wl5, wh4;		vadd.f dm1, dm3, dm0, r6
	vlda x0, [p3, #0];		vldb.pop.576 ex5, [p1, lf1, r25];		vst x3, [p4], #64;		nopx;		vconv.fp32.bf16 cml3, x3;		nopv
	vlda.pop.576 ex2, [p0, lf0, r24];		nopb;		nops;		nopx;		vshuffle x3, x1, x2, r2;		vmac.f dm2, dm2, x5, x7, r1
	nopa;		nopb;		vconv.bf16.fp32 x1, cml1;		nopx;		vshuffle bmlh0, x1, x2, r3;		nopv
	nopa;		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		vshuffle bmll0, x6, x0, r3;		nopv
	vlda.pop.576 ex0, [p0, lf0, r24];		nopb;		nops;		nopx;		vmin_ge.bf16 x7, r16, x9, x1;		vmac.f dm1, dm2, x5, x7, r1
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		vshuffle x8, x6, x0, r2;		nopv
	nopa;		vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x7, cml1;		nopx;		vmax_lt.bf16 x6, r16, x10, x7;		vmul.f dm2, ex2, ex5, r5
	nopa;		nopb;		nops;		nopx;		vmov wl3, wh8;		nopv
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vfloor.s32.bf16 x5, wh6, s0;		nopx;		vmin_ge.bf16 x1, r16, x9, x7;		vmul.f dm3, ex1, ex5, r5
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25];		vfloor.s32.bf16 x8, wl6, s0;		nopx;		nopm;		vmac.f dm1, dm0, x3, x0, r1
	vlda.pop.576 ex0, [p0, lf0, r24];		nopb;		nops;		nopx;		vmax_lt.bf16 x6, r16, x10, x1;		nopv
	nopa;		vldb.pop.576 ex2, [p1, lf1, r25];		nops;		nopx;		vadd.32 x1, x8, x11;		vmac.f dm2, dm2, ex2, ex7, r5
	vlda x4, [p3, #0];		nopb;		vfloor.s32.bf16 x5, wl6, s0;		nopx;		vadd.32 x3, x5, x11;		vmac.f dm0, dm1, x3, x4, r1
	vlda x7, [p3, #64];		nopb;		vfloor.s32.bf16 x0, wh6, s0;		nopx;		vmov cmh0, cml1;		vmac.f dm3, dm3, ex0, ex7, r5
	nopa;		vldb.4x64.hi wh0, wh1;		nops;		nopx;		vadd.32 x5, x5, x11;		vmac.f dm2, dm2, ex2, ex4, r5
	nopa;		vldb.4x64.lo wl2, wh3;		nops;		nopx;		vadd.32 x6, x0, x11;		nopv
	nopa;		vldb.4x64.lo wl4, wh5;		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex2, ex4, r5
	nopa;		vldb.4x64.lo wl6, wl5;		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex7, ex8, r5
	nopa;		vldb.4x64.hi wh5, wh6;		vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26];		nopx;		nopm;		nopv
	nopa;		vldb.4x64.lo wl5, wh6;		vst.flush.512.conv [p2, sf, r26];		nopx;		vconv.fp32.bf16 cml1, x4;		vmac.f dm3, dm3, ex0, ex2, r5
attk_last:
	nopa;		vldb.4x64.hi wh4, wh5;		nops;		nopx;		vconv.fp32.bf16 cml0, x7;		nopv
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
