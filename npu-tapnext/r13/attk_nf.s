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
	movxm	r2, #17023
	vbcst.16	x9, r2
	movxm	r2, #-15696
	vbcst.16	x10, r2
	mova	r1, #828
	mova	r2, #24
	mova	r3, #5
	mov	s0, #3
	mova	r26, #0
	mova	r17, #0
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
	nop
	nop
	mov r24, #0
	nop
	nop
	nop
	nop
	nop
	nop
	vlda.fill.512 [p0, lf0, r24]
	nop
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25]
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vlda.pop.576 ex1, [p0, lf0, r24]
	nop
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vldb.pop.576 ex3, [p1, lf1, r25]
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vmul.f dm2, ex0, ex1, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vmul.f dm3, ex3, ex4, r5
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	nop
	vlda x3, [p3, #64];		vmac.f dm2, dm2, ex1, ex5, r5
	vlda x6, [p3, #0];		vldb.pop.576 ex4, [p1, lf1, r25]
	vlda.pop.576 ex6, [p0, lf0, r24];		vmac.f dm3, dm3, ex2, ex3, r5
	vmac.f dm2, dm2, ex4, ex6, r5
	nop
	vmac.f dm3, dm3, ex3, ex4, r5
	vmac.f dm2, dm2, ex5, ex6, r5
	nop
	vconv.fp32.bf16 cml1, x6
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25];		vconv.fp32.bf16 cml0, x3
	vmac.f dm3, dm3, ex6, ex4, r5
	nop
	nop
	vadd.f dm3, dm2, dm3, r6
	mov r24, #0
	nop
	nop
	nop
	nop
	nop
	vconv.bf16.fp32 x4, cml3
	vlda.fill.512 [p0, lf0, r24];		vadd.f dm3, dm3, dm1, r6
	vconv.bf16.fp32 x5, cmh3;		vconv.fp32.bf16 cml3, x4
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25];		vst x4, [p4], #64;		vadd.f dm2, dm1, dm0, r6
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vconv.fp32.bf16 cml1, x5
	vlda.pop.576 ex1, [p0, lf0, r24]
	vst x5, [p4], #64
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vconv.bf16.fp32 x6, cml3
	vldb.pop.576 ex3, [p1, lf1, r25]
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vmin_ge.bf16 x5, r16, x9, x6
	vmul.f dm2, ex0, ex1, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x4, cml2;		vmax_lt.bf16 x3, r16, x10, x5;		vmul.f dm3, ex3, ex4, r5
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vfloor.s32.bf16 x4, wh3, s0;		vmin_ge.bf16 x0, r16, x9, x4
	vlda x3, [p3, #64];		vfloor.s32.bf16 x5, wl3, s0;		vmac.f dm2, dm2, ex1, ex5, r5
	vlda x6, [p3, #0];		vldb.pop.576 ex4, [p1, lf1, r25];		vadd.32 x1, x4, x11
	vlda.pop.576 ex6, [p0, lf0, r24];		vadd.32 x2, x5, x11;		vmac.f dm3, dm3, ex2, ex3, r5
	vldb.4x64.hi wh0, wl1;		vmac.f dm2, dm2, ex4, ex6, r5
	vldb.4x64.lo wl2, wl2
	vldb.4x64.lo wl4, wh2;		vmac.f dm3, dm3, ex3, ex4, r5
	vldb.4x64.lo wl0, wl1;		vmac.f dm2, dm2, ex5, ex6, r5
	vldb.4x64.hi wh4, wh2;		vmax_lt.bf16 x5, r16, x10, x0
	vldb.4x64.hi wh2, wl2;		vconv.fp32.bf16 cml1, x6
	.p2align	4
attk_start:
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25];		vfloor.s32.bf16 x3, wl5, s0;		nopx;		vconv.fp32.bf16 cml0, x3;		nopv
	vlda x5, [p3, #0];		nopb;		vfloor.s32.bf16 x5, wh5, s0;		nopx;		nopm;		vmac.f dm3, dm3, ex6, ex4, r5
	nopa;		vldb.4x64.lo wl6, wh1;		nops;		nopx;		vadd.32 x6, x3, x11;		nopv
	nopa;		vldb.4x64.hi wh6, wh1;		nops;		nopx;		vadd.32 x3, x5, x11;		nopv
	nopa;		vldb.4x64.hi wh2, wl6;		nops;		nopx;		nopm;		vadd.f dm3, dm2, dm3, r6
	nopa;		vldb.4x64.lo wl2, wl6;		nops;		nopx;		mov r24, #0;		nopv
	nopa;		vldb.4x64.hi wh3, wh6;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.lo wl3, wh6;		nops;		nopx;		vshuffle bmll2, x2, x4, r3;		nopv
	nopa;		vldb.4x64.hi wh0, wh3;		nops;		nopx;		vshuffle x4, x2, x4, r2;		nopv
	nopa;		vldb.4x64.lo wl4, wl3;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh4, wl3;		nops;		nopx;		vshuffle x1, x0, x6, r2;		nopv
	nopa;		vldb.4x64.lo wl0, wh3;		vconv.bf16.fp32 x4, cml3;		nopx;		vmov wl1, wh4;		nopv
	vlda.fill.512 [p0, lf0, r24];		nopb;		nops;		nopx;		vshuffle bmlh2, x0, x6, r3;		vadd.f dm3, dm3, dm1, r6
	vlda x3, [p3, #64];		vldb x5, [p5], #64;		vconv.bf16.fp32 x5, cmh3;		nopx;		vconv.fp32.bf16 cml3, x4;		vmac.f dm1, dm2, x1, x5, r1
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25];		vst x4, [p4], #64;		nopx;		nopm;		vadd.f dm2, dm1, dm0, r6
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		nops;		nopx;		vconv.fp32.bf16 cml1, x5;		nopv
	vlda.pop.576 ex1, [p0, lf0, r24];		nopb;		nops;		nopx;		vshuffle x6, x2, x3, r2;		nopv
	nopa;		vldb x0, [p5], #64;		vst x5, [p4], #64;		nopx;		vshuffle bmll0, x2, x3, r3;		nopv
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		vshuffle x2, x4, x0, r2;		nopv
	nopa;		nopb;		vconv.bf16.fp32 x6, cml3;		nopx;		vmov wl2, wh6;		nopv
	nopa;		vldb.pop.576 ex3, [p1, lf1, r25];		nops;		nopx;		vshuffle bmlh0, x4, x0, r3;		vmac.f dm0, dm1, x1, x5, r1
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		nops;		nopx;		vmin_ge.bf16 x5, r16, x9, x6;		vmac.f dm1, dm0, x2, x3, r1
	nopa;		nopb;		nops;		nopx;		nopm;		vmul.f dm2, ex0, ex1, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x4, cml2;		nopx;		vmax_lt.bf16 x3, r16, x10, x5;		vmul.f dm3, ex3, ex4, r5
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, x2, x0, r1
	nopa;		nopb;		vfloor.s32.bf16 x4, wh3, s0;		nopx;		vmin_ge.bf16 x0, r16, x9, x4;		nopv
	vlda x3, [p3, #64];		nopb;		vfloor.s32.bf16 x5, wl3, s0;		nopx;		nopm;		vmac.f dm2, dm2, ex1, ex5, r5
	vlda x6, [p3, #0];		vldb.pop.576 ex4, [p1, lf1, r25];		nops;		nopx;		vadd.32 x1, x4, x11;		nopv
	vlda.pop.576 ex6, [p0, lf0, r24];		nopb;		nops;		nopx;		vadd.32 x2, x5, x11;		vmac.f dm3, dm3, ex2, ex3, r5
	nopa;		vldb.4x64.hi wh0, wl1;		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex4, ex6, r5
	nopa;		vldb.4x64.lo wl2, wl2;		nops;		nopx;		vmov cmh0, cml1;		nopv
	nopa;		vldb.4x64.lo wl4, wh2;		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex3, ex4, r5
	nopa;		vldb.4x64.lo wl0, wl1;		vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26];		nopx;		nopm;		vmac.f dm2, dm2, ex5, ex6, r5
	nopa;		vldb.4x64.hi wh4, wh2;		vst.flush.512.conv [p2, sf, r26];		nopx;		vmax_lt.bf16 x5, r16, x10, x0;		nopv
attk_last:
	nopa;		vldb.4x64.hi wh2, wl2;		nops;		nopx;		vconv.fp32.bf16 cml1, x6;		nopv
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r0, r17
	ret	lr
	nop
	nop
	nop
	paddxm	[sp], #-64
	nop
.Lfunc_end_attk:
	.size	attk, .Lfunc_end_attk-attk
