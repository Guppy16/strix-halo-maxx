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
	mova	r22, #0
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
	// modulo schedule: II 33, U 1, stages 3, ops 96
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
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	nop
	vlda.pop.576 ex5, [p0, lf0, r24]
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex1, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24];		vmul.f dm1, ex2, ex4, r5
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vmul.f dm2, ex6, ex7, r5
	vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm1, dm1, ex2, ex5, r5
	vlda x6, [p3, #64];		vmac.f dm2, dm2, ex6, ex7, r5
	nop
	vmac.f dm1, dm1, ex5, ex7, r5
	vmac.f dm2, dm2, ex6, ex1, r5
	nop
	vmac.f dm1, dm1, ex0, ex6, r5
	vmac.f dm2, dm2, ex5, ex3, r5
	nop
	vlda x8, [p3, #0]
	vadd.f dm2, dm1, dm2, r6
	nop
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25];		vconv.fp32.bf16 cml0, x6
	mov r24, #0
	nop
	nop
	vconv.bf16.fp32 x7, cml2;		vconv.fp32.bf16 cml3, x8
	vadd.f dm2, dm2, dm3, r6
	vconv.bf16.fp32 x8, cmh2;		vconv.fp32.bf16 cml2, x7
	nop
	nop
	vlda.fill.512 [p0, lf0, r24];		vst x7, [p4], #64;		vconv.fp32.bf16 cml3, x8;		vadd.f dm0, dm3, dm0, r6
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vst x8, [p4], #64
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vconv.bf16.fp32 x7, cml2
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vlt.bf16 r20, x9, x7
	vmin_ge.bf16 x5, r16, x9, x7
	vlda.pop.576 ex5, [p0, lf0, r24];		vconv.bf16.fp32 x3, cml0
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex1, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24];		vmax_lt.bf16 x4, r16, x10, x5;		vmul.f dm1, ex2, ex4, r5
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vmul.f dm2, ex6, ex7, r5
	vldb.pop.576 ex3, [p1, lf1, r25];		vfloor.s32.bf16 x8, wh4, s0
	vlda x1, [p3, #0];		vmac.f dm1, dm1, ex2, ex5, r5
	vlda x6, [p3, #64];		vadd.32 x2, x8, x11;		vmac.f dm2, dm2, ex6, ex7, r5
	vlda x0, [p3, #64];		vfloor.s32.bf16 x4, wl4, s0
	vldb.4x64.hi wh2, wl2;		vlt.bf16 r21, x9, x3;		vmac.f dm1, dm1, ex5, ex7, r5
	vldb.4x64.lo wl2, wl2;		vadd.32 x4, x4, x11;		vmac.f dm2, dm2, ex6, ex1, r5
	vldb.4x64.lo wl5, wh2;		vmin_ge.bf16 x7, r16, x9, x3
	vldb.4x64.lo wl3, wh4;		vmac.f dm1, dm1, ex0, ex6, r5
	vldb.4x64.lo wl4, wl4;		vmax_lt.bf16 x8, r16, x10, x7;		vmac.f dm2, dm2, ex5, ex3, r5
	vldb.4x64.hi wh5, wh2
	vlda x8, [p3, #0];		vfloor.s32.bf16 x3, wh8, s0
	vldb.4x64.hi wh4, wl4;		vadd.f dm2, dm1, dm2, r6
	vldb.4x64.hi wh3, wh4;		vadd.32 x7, x3, x11
	.p2align	4
attk_start:
	lda p0, [sp, #-64];		vldb.fill.512 [p1, lf1, r25];		vfloor.s32.bf16 x6, wl8, s0;		nopx;		vconv.fp32.bf16 cml0, x6;		nopv
	nopa;		vldb.4x64.lo wl5, wl7;		nops;		nopx;		mov r24, #0;		nopv
	nopa;		vldb.4x64.hi wh5, wl7;		nops;		nopx;		vadd.32 x6, x6, x11;		nopv
	nopa;		vldb.4x64.hi wh3, wh7;		nops;		nopx;		vshuffle bmlh1, x2, x5, r3;		nopv
	nopa;		vldb.4x64.lo wl3, wh7;		vconv.bf16.fp32 x7, cml2;		nopx;		vconv.fp32.bf16 cml3, x8;		nopv
	nopa;		vldb.4x64.lo wl4, wl6;		nops;		nopx;		vshuffle x2, x2, x5, r2;		vadd.f dm2, dm2, dm3, r6
	nopa;		vldb.4x64.hi wh6, wh6;		vconv.bf16.fp32 x8, cmh2;		nopx;		vconv.fp32.bf16 cml2, x7;		nopv
	nopa;		vldb.4x64.hi wh4, wl6;		nops;		nopx;		vshuffle bmll1, x4, x3, r3;		nopv
	nopa;		vldb.4x64.lo wl6, wh6;		nops;		nopx;		vshuffle x4, x4, x3, r2;		nopv
	vlda.fill.512 [p0, lf0, r24];		vldb x7, [p5], #64;		vst x7, [p4], #64;		nopx;		vconv.fp32.bf16 cml3, x8;		vadd.f dm0, dm3, dm0, r6
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vst x8, [p4], #64;		or r22, r22, r20;		vmov wl2, wh4;		nopv
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vconv.bf16.fp32 x7, cml2;		nopx;		vshuffle bmlh2, x5, x3, r3;		nopv
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		vshuffle x1, x5, x3, r2;		vmac.f dm0, dm1, x2, x1, r1
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		vlt.bf16 r20, x9, x7;		nopv
	nopa;		nopb;		nops;		or r22, r22, r21;		vmin_ge.bf16 x5, r16, x9, x7;		nopv
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb x0, [p5], #64;		vconv.bf16.fp32 x3, cml0;		nopx;		vshuffle x8, x4, x6, r2;		nopv
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		vshuffle bmll2, x4, x6, r3;		vmac.f dm0, dm0, x2, x7, r1
	nopa;		vldb.pop.576 ex1, [p1, lf1, r25];		nops;		nopx;		vmov wl1, wh8;		nopv
	vlda.pop.576 ex0, [p0, lf0, r24];		nopb;		nops;		nopx;		vmax_lt.bf16 x4, r16, x10, x5;		vmul.f dm1, ex2, ex4, r5
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		nops;		nopx;		nopm;		vmul.f dm2, ex6, ex7, r5
	nopa;		vldb.pop.576 ex3, [p1, lf1, r25];		vfloor.s32.bf16 x8, wh4, s0;		nopx;		nopm;		vmac.f dm3, dm2, x1, x0, r1
	vlda x1, [p3, #0];		nopb;		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex2, ex5, r5
	vlda x6, [p3, #64];		nopb;		nops;		nopx;		vadd.32 x2, x8, x11;		vmac.f dm2, dm2, ex6, ex7, r5
	vlda x0, [p3, #64];		nopb;		vfloor.s32.bf16 x4, wl4, s0;		nopx;		nopm;		vmac.f dm3, dm3, x1, x0, r1
	nopa;		vldb.4x64.hi wh2, wl2;		nops;		nopx;		vlt.bf16 r21, x9, x3;		vmac.f dm1, dm1, ex5, ex7, r5
	nopa;		vldb.4x64.lo wl2, wl2;		nops;		nopx;		vadd.32 x4, x4, x11;		vmac.f dm2, dm2, ex6, ex1, r5
	nopa;		vldb.4x64.lo wl5, wh2;		nops;		nopx;		vmin_ge.bf16 x7, r16, x9, x3;		nopv
	nopa;		vldb.4x64.lo wl3, wh4;		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex0, ex6, r5
	nopa;		vldb.4x64.lo wl4, wl4;		nops;		nopx;		vmax_lt.bf16 x8, r16, x10, x7;		vmac.f dm2, dm2, ex5, ex3, r5
	nopa;		vldb.4x64.hi wh5, wh2;		nops;		nopx;		vmov cmh0, cml3;		nopv
	vlda x8, [p3, #0];		nopb;		vfloor.s32.bf16 x3, wh8, s0;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh4, wl4;		vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26];		nopx;		nopm;		vadd.f dm2, dm1, dm2, r6
attk_last:
	nopa;		vldb.4x64.hi wh3, wh4;		vst.flush.512.conv [p2, sf, r26];		nopx;		vadd.32 x7, x3, x11;		nopv
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r0, r22
	ret	lr
	nop
	nop
	nop
	paddxm	[sp], #-64
	nop
.Lfunc_end_attk:
	.size	attk, .Lfunc_end_attk-attk
