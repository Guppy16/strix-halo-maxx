	.text
	.section	.text.ggk,"ax",@progbits
	.globl	ggk
	.p2align	4
	.type	ggk,@function
ggk:
	paddxm	[sp], #64
	st	p6, [sp, #-64]
	mov	p6, p0
	mov	r0, p4
	mov	r2, p2
	sub	r0, r0, r2
	movxm	r3, #-960
	add	r0, r0, r3
	mov	m2, r0
	mova	dn2, #15
	mova	dj2, #64
	mova	dc2, #0
	movxm	r0, #16256
	vbcst.16	x11, r0
	mova	r1, #828
	mova	r5, #780
	mova	r25, #0
	movxm	ls, #ggk_start
	movxm	le, #ggk_last
	mova	r0, #8
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 24, U 1, stages 3, ops 61
	mov p0, p6
	nop
	mov r24, #0
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vldb.pop.576 ex2, [p1, lf1, r25]
	vldb.pop.576 ex4, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24]
	vlda.pop.576 ex3, [p0, lf0, r24]
	vldb.pop.576 ex1, [p1, lf1, r25]
	vldb.pop.576 ex6, [p1, lf1, r25]
	vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex6, [p1, lf1, r25];		vmul.f dm0, ex1, ex4, r5
	vlda.pop.576 ex4, [p0, lf0, r24]
	vldb.pop.576 ex5, [p1, lf1, r25];		vmul.f dm1, ex1, ex2, r5
	vldb.fill.512 [p1, lf1, r25];		vmac.f dm0, dm0, ex0, ex4, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vldb.pop.576 ex4, [p1, lf1, r25];		vmac.f dm1, dm1, ex0, ex1, r5
	vldb.pop.576 ex5, [p1, lf1, r25];		vmac.f dm0, dm0, ex3, ex6, r5
	vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm1, ex3, ex5, r5
	vldb.pop.576 ex4, [p1, lf1, r25];		vmac.f dm0, dm0, ex4, ex6, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vmac.f dm1, dm1, ex4, ex5, r5
	vldb.pop.576 ex5, [p1, lf1, r25];		mov p0, p6;		vmac.f dm0, dm0, ex3, ex6, r5
	nop
	vldb.pop.576 ex3, [p1, lf1, r25];		mov r24, #0;		vmac.f dm1, dm1, ex3, ex4, r5
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25];		vmac.f dm0, dm0, ex2, ex5, r5
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vldb.pop.576 ex2, [p1, lf1, r25];		vmac.f dm1, dm1, ex2, ex4, r5
	vldb.pop.576 ex4, [p1, lf1, r25];		vmac.f dm0, dm0, ex3, ex5, r5
	vlda.pop.576 ex0, [p0, lf0, r24]
	vlda.pop.576 ex3, [p0, lf0, r24];		vmac.f dm1, dm1, ex3, ex4, r5
	vldb.pop.576 ex1, [p1, lf1, r25];		vmac.f dm0, dm0, ex2, ex5, r5
	vldb.pop.576 ex6, [p1, lf1, r25]
	vldb.pop.576 ex5, [p1, lf1, r25];		vmac.f dm1, dm1, ex2, ex3, r5
	vldb.pop.576 ex6, [p1, lf1, r25];		vmul.f dm0, ex1, ex4, r5
	vlda.pop.576 ex4, [p0, lf0, r24]
	vldb.pop.576 ex5, [p1, lf1, r25];		vmul.f dm1, ex1, ex2, r5
	vlda.2d.conv.fp32.bf16 cml2, [p2], d2;		vldb.fill.512 [p1, lf1, r25];		vconv.bf16.fp32 x4, cml0;		vmac.f dm0, dm0, ex0, ex4, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vconv.bf16.fp32 x2, cmh0
	vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x1, cml1;		vmac.f dm1, dm1, ex0, ex1, r5
	vldb.pop.576 ex5, [p1, lf1, r25];		vconv.bf16.fp32 x0, cmh1;		vmac.f dm0, dm0, ex3, ex6, r5
	vlda.pop.576 ex2, [p0, lf0, r24];		vmac.f dm2, dm2, x4, x11, r1
	vlda.2d.conv.fp32.bf16 cml2, [p2], d2;		vmac.f dm1, dm1, ex3, ex5, r5
	vldb.pop.576 ex4, [p1, lf1, r25];		vmac.f dm0, dm0, ex4, ex6, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		vmac.f dm1, dm1, ex4, ex5, r5
	.p2align	4
ggk_start:
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		mov p0, p6;		vmac.f dm0, dm0, ex3, ex6, r5
	nopa;		nopb;		vst.conv.bf16.fp32 cml2, [p3], #64;		nopx;		nopm;		vmac.f dm2, dm2, x2, x11, r1
	vlda.2d.conv.fp32.bf16 cml2, [p2], d2;		vldb.pop.576 ex3, [p1, lf1, r25];		nops;		nopx;		mov r24, #0;		vmac.f dm1, dm1, ex3, ex4, r5
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex2, ex5, r5
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vldb.pop.576 ex2, [p1, lf1, r25];		vmac.f dm1, dm1, ex2, ex4, r5
	vldb.pop.576 ex4, [p1, lf1, r25];		vmac.f dm0, dm0, ex3, ex5, r5
	vlda.pop.576 ex0, [p0, lf0, r24];		vst.conv.bf16.fp32 cml2, [p3], #64;		vmac.f dm2, dm2, x1, x11, r1
	vlda.pop.576 ex3, [p0, lf0, r24];		vmac.f dm1, dm1, ex3, ex4, r5
	vlda.2d.conv.fp32.bf16 cml2, [p2], d2;		vldb.pop.576 ex1, [p1, lf1, r25];		vmac.f dm0, dm0, ex2, ex5, r5
	vldb.pop.576 ex6, [p1, lf1, r25]
	vldb.pop.576 ex5, [p1, lf1, r25];		vmac.f dm1, dm1, ex2, ex3, r5
	vldb.pop.576 ex6, [p1, lf1, r25];		vmul.f dm0, ex1, ex4, r5
	vlda.pop.576 ex4, [p0, lf0, r24];		vst.conv.bf16.fp32 cml2, [p3], #64;		vmac.f dm2, dm2, x0, x11, r1
	vldb.pop.576 ex5, [p1, lf1, r25];		vmul.f dm1, ex1, ex2, r5
	vlda.2d.conv.fp32.bf16 cml2, [p2], d2;		vldb.fill.512 [p1, lf1, r25];		vconv.bf16.fp32 x4, cml0;		vmac.f dm0, dm0, ex0, ex4, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vconv.bf16.fp32 x2, cmh0
	vldb.pop.576 ex4, [p1, lf1, r25];		vconv.bf16.fp32 x1, cml1;		vmac.f dm1, dm1, ex0, ex1, r5
	vldb.pop.576 ex5, [p1, lf1, r25];		vconv.bf16.fp32 x0, cmh1;		vmac.f dm0, dm0, ex3, ex6, r5
	vlda.pop.576 ex2, [p0, lf0, r24];		vst.conv.bf16.fp32 cml2, [p3], #64;		vmac.f dm2, dm2, x4, x11, r1
	vlda.2d.conv.fp32.bf16 cml2, [p2], d2;		vmac.f dm1, dm1, ex3, ex5, r5
	vldb.pop.576 ex4, [p1, lf1, r25];		vmac.f dm0, dm0, ex4, ex6, r5
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
ggk_last:
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex4, ex5, r5
	lda	p6, [sp, #-64]
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
	paddxm	[sp], #-64
.Lfunc_end_ggk:
	.size	ggk, .Lfunc_end_ggk-ggk
