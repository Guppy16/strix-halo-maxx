	.text
	.section	.text.attkb,"ax",@progbits
	.globl	attkb
	.p2align	4
	.type	attkb,@function
attkb:
	mov	r7, r0
	paddxm	[sp], #64
	st	p6, [sp, #-64]
	st	p7, [sp, #-60]
	mov	p6, p5
	mov	p7, p3
	mov	p5, p4
	mova	r5, #780
	mova	r6, #60
	mova	r24, #0
	mova	r25, #0
	mov	r2, p4
	movxm	r3, #896
	add	r2, r2, r3
	movxm	r16, #52428
	vbcst.32	x1, r2
	movxm	r2, #aexp_ab
	vbcst.32	x0, r2
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
	mova	dn0, #5
	movxm	dj0, #-576
	movxm	m0, #0
	mova	dc0, #0
	mova	dn1, #5
	movxm	dj1, #0
	movxm	m1, #-3456
	mova	dc1, #0
	mova	dn2, #5
	movxm	dj2, #-64
	movxm	m2, #64
	mova	dc2, #0
	mova	dn3, #5
	movxm	dj3, #-64
	movxm	m3, #64
	mova	dc3, #0
	mova	dn4, #6
	movxm	dj4, #64
	movxm	m4, #-832
	mova	dc4, #0
	mova	dn5, #6
	movxm	dj5, #64
	movxm	m5, #-832
	mova	dc5, #0
	mova	dn6, #5
	movxm	dj6, #0
	movxm	m6, #16
	mova	dc6, #0
	movxm	ls, #attkb_start
	movxm	le, #attkb_last
	add.nc	lc, r7, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 34, U 1, stages 3, ops 95
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25]
	nop
	nop
	nop
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25]
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24]
	vldb.pop.576 ex5, [p1, lf1, r25];		vmul.f dm1, ex3, ex7, r5
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25];		vmul.f dm2, ex6, ex8, r5
	vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25]
	vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm1, dm1, ex6, ex8, r5
	vlda.conv.fp32.bf16 cml0, [p3], #64;		vmac.f dm2, dm2, ex5, ex7, r5
	padds.2d [p1], d1
	vmac.f dm1, dm1, ex0, ex5, r5
	vmac.f dm2, dm2, ex7, ex8, r5
	padda.2d [p0], d0
	vmac.f dm1, dm1, ex7, ex5, r5
	vmac.f dm2, dm2, ex1, ex6, r5
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	nop
	vadd.f dm2, dm1, dm2, r6
	nop
	nop
	nop
	nop
	nop
	vconv.bf16.fp32 x7, cml2
	vconv.bf16.fp32 x6, cmh2;		vadd.f dm3, dm3, dm0, r6
	vlda.2d.conv.fp32.bf16 cml3, [p3], d2;		vconv.fp32.bf16 cml3, x7
	nop
	vst x7, [p4], #64
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25];		vst.2d x6, [p4], d4;		vconv.fp32.bf16 cml0, x6;		vadd.f dm0, dm0, dm3, r6
	vconv.bf16.fp32 x4, cml3
	nop
	nop
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25];		vmin_ge.bf16 x6, r16, x9, x4
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24];		vmax_lt.bf16 x1, r16, x10, x6
	vldb.pop.576 ex5, [p1, lf1, r25];		vconv.bf16.fp32 x3, cml0;		vmul.f dm1, ex3, ex7, r5
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25];		vfloor.s32.bf16 x4, wh1, s0;		vlt.bf16 r18, x9, x4;		vmul.f dm2, ex6, ex8, r5
	vldb.pop.576 ex5, [p1, lf1, r25];		vfloor.s32.bf16 x1, wl1, s0;		vmin_ge.bf16 x7, r16, x9, x3
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vadd.32 x4, x4, x11
	vlda.pop.576 ex1, [p0, lf0, r24];		vmax_lt.bf16 x2, r16, x10, x7
	vldb.4x64.hi wh0, wh4;		vmac.f dm1, dm1, ex6, ex8, r5
	vlda.conv.fp32.bf16 cml0, [p3], #64;		vfloor.s32.bf16 x6, wl2, s0;		vmac.f dm2, dm2, ex5, ex7, r5
	vldb.4x64.hi wh1, wl4;		padds.2d [p1], d1;		vlt.bf16 r19, x9, x3
	vldb.4x64.lo wl1, wl4;		vadd.32 x3, x6, x11;		vmac.f dm1, dm1, ex0, ex5, r5
	vldb.4x64.lo wl0, wh4;		vmac.f dm2, dm2, ex7, ex8, r5
	padda.2d [p0], d0;		vldb.4x64.lo wl2, wl3;		vadd.32 x4, x1, x11
	vldb.4x64.hi wh3, wh3;		vmac.f dm1, dm1, ex7, ex5, r5
	vldb.4x64.hi wh2, wl3;		vmac.f dm2, dm2, ex1, ex6, r5
	.p2align	4
attkb_start:
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25];		vfloor.s32.bf16 x5, wh2, s0;		nopx;		nopm;		nopv
	vldb.4x64.lo wl4, wh4;		or r4, r18, r19
	vldb.4x64.lo wl3, wh3;		or r17, r17, r4;		vadd.f dm2, dm1, dm2, r6
	nopa;		vldb.4x64.hi wh4, wh4;		nops;		nopx;		nopm;		nopv
	vldb.4x64.lo wl8, wl4;		vadd.32 x5, x5, x11
	vldb.4x64.hi wh8, wl4
	vldb.4x64.lo wl1, wl5
	vldb.4x64.lo wl3, wh5;		vshuffle bmlh1, x1, x0, r3
	vldb.4x64.hi wh1, wl5;		vconv.bf16.fp32 x7, cml2;		vshuffle x0, x1, x0, r2
	vldb.4x64.hi wh3, wh5;		vconv.bf16.fp32 x6, cmh2;		vshuffle bmll2, x2, x3, r3;		vadd.f dm3, dm3, dm0, r6
	vlda.2d.conv.fp32.bf16 cml3, [p3], d2;		vconv.fp32.bf16 cml3, x7
	vlda x6, [p7], #64;		vshuffle x5, x2, x3, r2
	vlda.2d x5, [p7], d3;		vst x7, [p4], #64;		vshuffle x2, x8, x4, r2
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vshuffle bmll1, x8, x4, r3
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25];		vst.2d x6, [p4], d4;		vconv.fp32.bf16 cml0, x6;		vadd.f dm0, dm0, dm3, r6
	vconv.bf16.fp32 x4, cml3;		vmov wl0, wh2
	vldb x5, [p5], #64;		vshuffle x2, x1, x3, r2
	vldb.2d x6, [p5], d5;		vmov wl2, wh5
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25];		vmin_ge.bf16 x6, r16, x9, x4;		vmac.f dm0, dm1, x0, x6, r1
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vshuffle bmlh2, x1, x3, r3
	vlda.pop.576 ex0, [p0, lf0, r24];		vmax_lt.bf16 x1, r16, x10, x6;		vmac.f dm3, dm2, x2, x5, r1
	vldb.pop.576 ex5, [p1, lf1, r25];		vconv.bf16.fp32 x3, cml0;		vmul.f dm1, ex3, ex7, r5
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb.pop.576 ex8, [p1, lf1, r25];		vfloor.s32.bf16 x4, wh1, s0;		vlt.bf16 r18, x9, x4;		vmul.f dm2, ex6, ex8, r5
	vldb.pop.576 ex5, [p1, lf1, r25];		vfloor.s32.bf16 x1, wl1, s0;		vmin_ge.bf16 x7, r16, x9, x3
	vlda.pop.576 ex7, [p0, lf0, r24];		vldb.pop.576 ex6, [p1, lf1, r25];		vadd.32 x4, x4, x11;		vmac.f dm3, dm3, x2, x6, r1
	vlda.pop.576 ex1, [p0, lf0, r24];		st r4, [p6], #4;		vmax_lt.bf16 x2, r16, x10, x7;		vmac.f dm0, dm0, x0, x5, r1
	vldb.4x64.hi wh0, wh4;		vmac.f dm1, dm1, ex6, ex8, r5
	vlda.conv.fp32.bf16 cml0, [p3], #64;		vfloor.s32.bf16 x6, wl2, s0;		vmac.f dm2, dm2, ex5, ex7, r5
	vldb.4x64.hi wh1, wl4;		padds.2d [p1], d1;		vlt.bf16 r19, x9, x3
	vldb.4x64.lo wl1, wl4;		vadd.32 x3, x6, x11;		vmac.f dm1, dm1, ex0, ex5, r5
	vldb.4x64.lo wl0, wh4;		vmov cmh0, cml3;		vmac.f dm2, dm2, ex7, ex8, r5
	padda.2d [p0], d0;		vldb.4x64.lo wl2, wl3;		vadd.32 x4, x1, x11
	vldb.4x64.hi wh3, wh3;		vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26];		vmac.f dm1, dm1, ex7, ex5, r5
attkb_last:
	nopa;		vldb.4x64.hi wh2, wl3;		vst.flush.512.conv.2d [p2, sf, r26, d6];		nopx;		nopm;		vmac.f dm2, dm2, ex1, ex6, r5
	st	r17, [p6, #0]
	lda	p6, [sp, #-64]
	lda	p7, [sp, #-60]
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
.Lfunc_end_attkb:
	.size	attkb, .Lfunc_end_attkb-attkb
