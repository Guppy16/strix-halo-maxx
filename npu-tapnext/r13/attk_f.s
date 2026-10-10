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
	mov r24, #0
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vlda.fill.512 [p0, lf0, r24]
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25]
	nop
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25]
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	nop
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex2, [p1, lf1, r25]
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vlda.pop.576 ex1, [p0, lf0, r24];		vmul.f dm2, ex5, ex6, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vmul.f dm1, ex0, ex1, r5
	vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm2, dm2, ex0, ex1, r5
	vmac.f dm1, dm1, ex5, ex6, r5
	nop
	vmac.f dm2, dm2, ex1, ex2, r5
	vmac.f dm1, dm1, ex5, ex6, r5
	vlda x0, [p3, #0]
	vlda x1, [p3, #64];		vmac.f dm2, dm2, ex1, ex4, r5
	vmac.f dm1, dm1, ex3, ex5, r5
	nop
	nop
	vadd.f dm1, dm2, dm1, r6
	nop
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25];		vconv.fp32.bf16 cml0, x0
	mov r24, #0
	nop
	vconv.fp32.bf16 cml2, x1
	vconv.bf16.fp32 x4, cml1
	vadd.f dm0, dm1, dm0, r6
	vconv.bf16.fp32 x7, cmh1;		vconv.fp32.bf16 cml1, x4
	nop
	nop
	vlda.fill.512 [p0, lf0, r24];		vst x4, [p4], #64;		vadd.f dm1, dm1, dm2, r6
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vst x7, [p4], #64;		vconv.fp32.bf16 cml1, x7
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25];		vconv.bf16.fp32 x5, cml0
	nop
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25];		vmin_ge.bf16 x5, r16, x9, x5
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vconv.bf16.fp32 x4, cml1
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex2, [p1, lf1, r25];		vmax_lt.bf16 x2, r16, x10, x5
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vlda.pop.576 ex1, [p0, lf0, r24];		vfloor.s32.bf16 x3, wl2, s0;		vmul.f dm2, ex5, ex6, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vmul.f dm1, ex0, ex1, r5
	vldb.pop.576 ex5, [p1, lf1, r25];		vadd.32 x3, x3, x11
	vlda x8, [p3, #0];		vfloor.s32.bf16 x0, wh2, s0;		vmac.f dm2, dm2, ex0, ex1, r5
	vldb.4x64.hi wh6, wl3;		vmin_ge.bf16 x4, r16, x9, x4;		vmac.f dm1, dm1, ex5, ex6, r5
	vldb.4x64.lo wl6, wl3
	vldb.4x64.hi wh5, wh3;		vmax_lt.bf16 x2, r16, x10, x4;		vmac.f dm2, dm2, ex1, ex2, r5
	vldb.4x64.lo wl5, wh3;		vadd.32 x0, x0, x11;		vmac.f dm1, dm1, ex5, ex6, r5
	vlda x0, [p3, #0]
	vlda x1, [p3, #64];		vfloor.s32.bf16 x1, wl2, s0;		vmac.f dm2, dm2, ex1, ex4, r5
	vldb.4x64.hi wh0, wl0;		vmac.f dm1, dm1, ex3, ex5, r5
	vldb.4x64.lo wl3, wh0
	vldb.4x64.lo wl0, wl0;		vadd.32 x4, x1, x11
	vldb.4x64.hi wh3, wh0;		vadd.f dm1, dm2, dm1, r6
	vldb.4x64.lo wl1, wh4
	.p2align	4
attk_start:
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25];		vfloor.s32.bf16 x2, wh2, s0;		nopx;		vconv.fp32.bf16 cml0, x0;		nopv
	nopa;		vldb.4x64.lo wl6, wl4;		nops;		nopx;		mov r24, #0;		nopv
	nopa;		vldb.4x64.hi wh6, wl4;		nops;		nopx;		vadd.32 x2, x2, x11;		nopv
	nopa;		vldb.4x64.hi wh1, wh4;		nops;		nopx;		vconv.fp32.bf16 cml2, x1;		nopv
	nopa;		vldb.4x64.hi wh0, wl2;		vconv.bf16.fp32 x4, cml1;		nopx;		vshuffle bmll3, x6, x5, r3;		nopv
	nopa;		vldb.4x64.lo wl3, wh2;		nops;		nopx;		vshuffle x5, x6, x5, r2;		vadd.f dm0, dm1, dm0, r6
	nopa;		vldb.4x64.lo wl0, wl2;		vconv.bf16.fp32 x7, cmh1;		nopx;		vconv.fp32.bf16 cml1, x4;		nopv
	nopa;		vldb.4x64.hi wh3, wh2;		nops;		nopx;		vshuffle x2, x0, x3, r2;		nopv
	vlda x1, [p3, #64];		vldb x6, [p5], #64;		nops;		nopx;		vmov wl2, wh5;		nopv
	vlda.fill.512 [p0, lf0, r24];		vldb x8, [p5], #64;		vst x4, [p4], #64;		nopx;		vshuffle bmlh3, x0, x3, r3;		vadd.f dm1, dm1, dm2, r6
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vst x7, [p4], #64;		nopx;		vconv.fp32.bf16 cml1, x7;		nopv
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25];		vconv.bf16.fp32 x5, cml0;		nopx;		vshuffle x4, x6, x1, r2;		nopv
	nopa;		nopb;		nops;		nopx;		vshuffle bmll0, x6, x1, r3;		nopv
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex1, [p1, lf1, r25];		nops;		nopx;		vmin_ge.bf16 x5, r16, x9, x5;		vmac.f dm1, dm3, x2, x8, r1
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		nops;		nopx;		vshuffle x7, x0, x3, r2;		nopv
	nopa;		nopb;		vconv.bf16.fp32 x4, cml1;		nopx;		vmov wl7, wh4;		nopv
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex2, [p1, lf1, r25];		nops;		nopx;		vmax_lt.bf16 x2, r16, x10, x5;		vmac.f dm0, dm1, x2, x6, r1
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		nops;		nopx;		vshuffle bmlh0, x0, x3, r3;		vmac.f dm1, dm0, x7, x1, r1
	vlda.pop.576 ex1, [p0, lf0, r24];		nopb;		vfloor.s32.bf16 x3, wl2, s0;		nopx;		nopm;		vmul.f dm2, ex5, ex6, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		nops;		nopx;		nopm;		vmul.f dm1, ex0, ex1, r5
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		vadd.32 x3, x3, x11;		vmac.f dm3, dm1, x7, x8, r1
	vlda x8, [p3, #0];		nopb;		vfloor.s32.bf16 x0, wh2, s0;		nopx;		nopm;		vmac.f dm2, dm2, ex0, ex1, r5
	nopa;		vldb.4x64.hi wh6, wl3;		nops;		nopx;		vmin_ge.bf16 x4, r16, x9, x4;		vmac.f dm1, dm1, ex5, ex6, r5
	nopa;		vldb.4x64.lo wl6, wl3;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh5, wh3;		nops;		nopx;		vmax_lt.bf16 x2, r16, x10, x4;		vmac.f dm2, dm2, ex1, ex2, r5
	nopa;		vldb.4x64.lo wl5, wh3;		nops;		nopx;		vadd.32 x0, x0, x11;		vmac.f dm1, dm1, ex5, ex6, r5
	vlda x0, [p3, #0];		nopb;		nops;		nopx;		vmov cmh0, cml3;		nopv
	vlda x1, [p3, #64];		nopb;		vfloor.s32.bf16 x1, wl2, s0;		nopx;		nopm;		vmac.f dm2, dm2, ex1, ex4, r5
	nopa;		vldb.4x64.hi wh0, wl0;		vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26];		nopx;		nopm;		vmac.f dm1, dm1, ex3, ex5, r5
	nopa;		vldb.4x64.lo wl3, wh0;		vst.flush.512.conv [p2, sf, r26];		nopx;		nopm;		nopv
	nopa;		vldb.4x64.lo wl0, wl0;		nops;		nopx;		vadd.32 x4, x1, x11;		nopv
	nopa;		vldb.4x64.hi wh3, wh0;		nops;		nopx;		nopm;		vadd.f dm1, dm2, dm1, r6
attk_last:
	nopa;		vldb.4x64.lo wl1, wh4;		nops;		nopx;		nopm;		nopv
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
