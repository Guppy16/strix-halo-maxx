	.text
	.section	.text.attqk,"ax",@progbits
	.globl	attqk
	.p2align	4
	.type	attqk,@function
attqk:
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
	movxm	ls, #attqk_start
	movxm	le, #attqk_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 11, U 2, stages 3, ops 20
	vldb.fill.512 [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	nop
	vldb.pop.576 ex8, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	nop
	vldb.pop.576 ex8, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25];		vmul.f dm1, ex0, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmul.f dm2, ex1, ex9, r5
	vldb.fill.512 [p1, lf1, r25]
	vldb.pop.576 ex8, [p1, lf1, r25];		vmac.f dm1, dm1, ex2, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm2, dm2, ex3, ex9, r5
	nop
	vldb.pop.576 ex8, [p1, lf1, r25];		vmac.f dm1, dm1, ex4, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm2, dm2, ex5, ex9, r5
	nop
	vldb.pop.576 ex8, [p1, lf1, r25];		vmac.f dm1, dm1, ex6, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm2, dm2, ex7, ex9, r5
	vldb.pop.576 ex8, [p1, lf1, r25];		vmul.f dm0, ex0, ex8, r5
	vldb.pop.576 ex9, [p1, lf1, r25];		vmul.f dm2, ex1, ex9, r5
	.p2align	4
attqk_start:
	nopa;		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vadd.f dm1, dm1, dm2, r6
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex2, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex3, ex9, r5
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex4, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex5, ex9, r5
	nopa;		nopb;		vst.conv.bf16.fp32 cml1, [p2], #64;		nopx;		nopm;		nopv
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		vst.conv.bf16.fp32 cmh1, [p2], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex6, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex7, ex9, r5
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		vmul.f dm1, ex0, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmul.f dm2, ex1, ex9, r5
	nopa;		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vadd.f dm0, dm0, dm2, r6
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex2, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex3, ex9, r5
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex4, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex5, ex9, r5
	nopa;		nopb;		vst.conv.bf16.fp32 cml0, [p2], #64;		nopx;		nopm;		nopv
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		vst.conv.bf16.fp32 cmh0, [p2], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex6, ex8, r5
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex7, ex9, r5
	nopa;		vldb.pop.576 ex8, [p1, lf1, r25];		nops;		nopx;		nopm;		vmul.f dm0, ex0, ex8, r5
attqk_last:
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
.Lfunc_end_attqk:
	.size	attqk, .Lfunc_end_attqk-attqk
