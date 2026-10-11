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
	// modulo schedule: II 11, U 2, stages 3, ops 22
	vldb.fill.512 [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25]
	nop
	vldb.pop.576 ex9, [p1, lf1, r25]
	nop
	vldb.pop.576 ex8, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25];		vmul.f dm1, ex0, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmul.f dm2, ex1, ex9, r5
	vldb.fill.512 [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25];		vmac.f dm1, dm1, ex2, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm2, dm2, ex3, ex9, r5
	vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm1, dm1, ex4, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm2, dm2, ex5, ex9, r5
	nop
	vldb.pop.576 ex8, [p1, lf1, r25];		vmac.f dm1, dm1, ex6, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm2, dm2, ex7, ex9, r5
	vldb.pop.576 ex8, [p1, lf1, r25];		vmul.f dm0, ex0, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmul.f dm2, ex1, ex9, r5
	.p2align	4
attk_start:
	nopa;		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vadd.f dm1, dm1, dm2, r6
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex2, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex3, ex9, r5
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex4, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex5, ex9, r5
	nopa;		nopb;		vconv.bf16.fp32 x0, cml1;		nopx;		nopm;		nopv
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		vconv.bf16.fp32 x1, cmh1;		nopx;		nopm;		vmac.f dm0, dm0, ex6, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		vst x0, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex7, ex9, r5
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		vst x1, [p4], #64;		nopx;		nopm;		vmul.f dm1, ex0, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmul.f dm2, ex1, ex9, r5
	nopa;		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vadd.f dm0, dm0, dm2, r6
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex2, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex3, ex9, r5
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex4, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex5, ex9, r5
	nopa;		nopb;		vconv.bf16.fp32 x0, cml0;		nopx;		nopm;		nopv
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		vconv.bf16.fp32 x1, cmh0;		nopx;		nopm;		vmac.f dm1, dm1, ex6, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		vst x0, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex7, ex9, r5
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		vst x1, [p4], #64;		nopx;		nopm;		vmul.f dm0, ex0, ex8, r5
attk_last:
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmul.f dm2, ex1, ex9, r5
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
