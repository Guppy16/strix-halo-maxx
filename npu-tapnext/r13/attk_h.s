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
	// modulo schedule: II 33, U 1, stages 3, ops 92
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	mov r24, #0
	nop
	vlda.fill.512 [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vlda.pop.576 ex8, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vlda.pop.576 ex7, [p0, lf0, r24]
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vmul.f dm1, ex8, ex6, r5
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		vmul.f dm2, ex2, ex7, r5
	nop
	vlda x6, [p3, #0];		vmac.f dm1, dm1, ex7, ex4, r5
	vmac.f dm2, dm2, ex3, ex5, r5
	nop
	vmac.f dm1, dm1, ex3, ex4, r5
	vmac.f dm2, dm2, ex5, ex6, r5
	nop
	vmac.f dm1, dm1, ex3, ex4, r5
	vconv.fp32.bf16 cml0, x6;		vmac.f dm2, dm2, ex2, ex5, r5
	nop
	nop
	vadd.f dm2, dm1, dm2, r6
	nop
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25]
	vlda x5, [p3, #64]
	nop
	nop
	vconv.bf16.fp32 x6, cml2
	vadd.f dm2, dm3, dm0, r6
	vconv.fp32.bf16 cml3, x6
	vconv.bf16.fp32 x7, cmh2;		mov r24, #0
	vconv.fp32.bf16 cml0, x5;		vadd.f dm2, dm3, dm0, r6
	vlda.fill.512 [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vst x6, [p4], #64;		vconv.fp32.bf16 cml3, x7
	vlda.pop.576 ex8, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vst x7, [p4], #64
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x7, cml2
	nop
	vlda.pop.576 ex7, [p0, lf0, r24];		vmin_ge.bf16 x7, r16, x9, x7
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		vconv.bf16.fp32 x0, cml2
	nop
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vmax_lt.bf16 x1, r16, x10, x7
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vfloor.s32.bf16 x5, wl1, s0;		vmul.f dm1, ex8, ex6, r5
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		vmul.f dm2, ex2, ex7, r5
	vadd.32 x2, x5, x11
	vlda x6, [p3, #0];		vfloor.s32.bf16 x0, wh1, s0;		vmin_ge.bf16 x1, r16, x9, x0;		vmac.f dm1, dm1, ex7, ex4, r5
	vldb.4x64.hi wh3, wh2;		vmac.f dm2, dm2, ex3, ex5, r5
	vldb.4x64.hi wh4, wl2;		vmax_lt.bf16 x1, r16, x10, x1
	vldb.4x64.lo wl3, wh2;		vadd.32 x0, x0, x11;		vmac.f dm1, dm1, ex3, ex4, r5
	vldb.4x64.lo wl4, wl2;		vmac.f dm2, dm2, ex5, ex6, r5
	vldb.4x64.hi wh5, wl0
	vlda x0, [p3, #0];		vfloor.s32.bf16 x7, wl1, s0;		vmac.f dm1, dm1, ex3, ex4, r5
	vldb.4x64.hi wh7, wh0;		vconv.fp32.bf16 cml0, x6;		vmac.f dm2, dm2, ex2, ex5, r5
	vldb.4x64.lo wl7, wh0;		vadd.32 x6, x7, x11
	vldb.4x64.lo wl5, wl0
	vldb.4x64.lo wl3, wh6;		vadd.f dm2, dm1, dm2, r6
	vldb.4x64.lo wl4, wl6
	.p2align	4
attk_start:
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25];		vfloor.s32.bf16 x1, wh1, s0;		nopx;		nopm;		nopv
	vlda x5, [p3, #64];		nopb;		nops;		nopx;		vshuffle x2, x4, x3, r2;		nopv
	nopa;		vldb.4x64.hi wh4, wl6;		nops;		nopx;		vadd.32 x8, x1, x11;		nopv
	nopa;		vldb.4x64.hi wh3, wh6;		nops;		nopx;		vshuffle bmll1, x4, x3, r3;		nopv
	nopa;		vldb.4x64.lo wl5, wl8;		vconv.bf16.fp32 x6, cml2;		nopx;		vshuffle x1, x5, x7, r2;		nopv
	nopa;		vldb.4x64.lo wl2, wh8;		nops;		nopx;		vshuffle bmlh1, x5, x7, r3;		vadd.f dm2, dm3, dm0, r6
	nopa;		vldb.4x64.hi wh2, wh8;		nops;		nopx;		vconv.fp32.bf16 cml3, x6;		nopv
	nopa;		vldb.4x64.hi wh5, wl8;		vconv.bf16.fp32 x7, cmh2;		nopx;		mov r24, #0;		nopv
	vlda x4, [p3, #64];		vldb x8, [p5], #64;		nops;		nopx;		vconv.fp32.bf16 cml0, x5;		vadd.f dm2, dm3, dm0, r6
	vlda.fill.512 [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vst x6, [p4], #64;		nopx;		vconv.fp32.bf16 cml3, x7;		nopv
	vlda.pop.576 ex8, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vst x7, [p4], #64;		nopx;		vmov wl1, wh2;		nopv
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x7, cml2;		nopx;		vshuffle bmll0, x4, x3, r3;		nopv
	nopa;		nopb;		nops;		nopx;		vshuffle x6, x4, x3, r2;		vmac.f dm1, dm1, x1, x0, r1
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb x6, [p5], #64;		nops;		nopx;		vmin_ge.bf16 x7, r16, x9, x7;		nopv
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		vconv.bf16.fp32 x0, cml2;		nopx;		vshuffle x3, x5, x2, r2;		nopv
	nopa;		nopb;		nops;		nopx;		vmov wl3, wh6;		vmac.f dm3, dm1, x1, x8, r1
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		nops;		nopx;		vmax_lt.bf16 x1, r16, x10, x7;		nopv
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		nops;		nopx;		vshuffle bmlh0, x5, x2, r3;		vmac.f dm0, dm0, x3, x4, r1
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vfloor.s32.bf16 x5, wl1, s0;		nopx;		nopm;		vmul.f dm1, ex8, ex6, r5
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmul.f dm2, ex2, ex7, r5
	nopa;		nopb;		nops;		nopx;		vadd.32 x2, x5, x11;		vmac.f dm0, dm0, x3, x6, r1
	vlda x6, [p3, #0];		nopb;		vfloor.s32.bf16 x0, wh1, s0;		nopx;		vmin_ge.bf16 x1, r16, x9, x0;		vmac.f dm1, dm1, ex7, ex4, r5
	nopa;		vldb.4x64.hi wh3, wh2;		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex3, ex5, r5
	nopa;		vldb.4x64.hi wh4, wl2;		nops;		nopx;		vmax_lt.bf16 x1, r16, x10, x1;		nopv
	nopa;		vldb.4x64.lo wl3, wh2;		nops;		nopx;		vadd.32 x0, x0, x11;		vmac.f dm1, dm1, ex3, ex4, r5
	nopa;		vldb.4x64.lo wl4, wl2;		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex5, ex6, r5
	nopa;		vldb.4x64.hi wh5, wl0;		nops;		nopx;		vmov cmh3, cml0;		nopv
	vlda x0, [p3, #0];		nopb;		vfloor.s32.bf16 x7, wl1, s0;		nopx;		nopm;		vmac.f dm1, dm1, ex3, ex4, r5
	nopa;		vldb.4x64.hi wh7, wh0;		vst.push.576.conv.bfp16ebs8.fp32 dm3, [p2, sf, r26];		nopx;		vconv.fp32.bf16 cml0, x6;		vmac.f dm2, dm2, ex2, ex5, r5
	nopa;		vldb.4x64.lo wl7, wh0;		vst.flush.512.conv [p2, sf, r26];		nopx;		vadd.32 x6, x7, x11;		nopv
	nopa;		vldb.4x64.lo wl5, wl0;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.lo wl3, wh6;		nops;		nopx;		nopm;		vadd.f dm2, dm1, dm2, r6
attk_last:
	nopa;		vldb.4x64.lo wl4, wl6;		nops;		nopx;		nopm;		nopv
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
