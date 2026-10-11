	.text
	.section	.text.gup,"ax",@progbits
	.globl	gup
	.p2align	4
	.type	gup,@function
gup:
	mov	m4, r1
	mova	dn4, #3
	mova	dj4, #0
	mova	dc4, #0
	mova	dn0, #7
	mova	dj0, #64
	movxm	m0, #576
	mova	dc0, #0
	mova	dn1, #3
	mova	dj1, #64
	mova	m1, #64
	mova	dc1, #0
	mova	dn3, #7
	mova	dj3, #64
	movxm	m3, #-448
	mova	dc3, #0
	movxm	r3, #16256
	vbcst.16	x11, r3
	mova	r2, #828
	mova	r26, #0
	movxm	ls, #gup_start
	movxm	le, #gup_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 6, U 1, stages 6, ops 20
	vldb.2d x3, [p3], d3
	nop
	vlda.2d bmll2, [p1], d1
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3
	vlda.2d bmll1, [p1], d1
	nop
	vldb.2d x3, [p3], d3
	vlda.2d bmlh1, [p1], d1;		vmac.f dm0, dm2, x3, x11, r2
	vlda.2d bmll2, [p1], d1
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3
	vlda.2d bmll1, [p1], d1;		vldb.2d x2, [p0], d0
	vmac.f dm4, dm1, x1, x11, r2
	vldb.2d x3, [p3], d3
	vlda.2d bmlh1, [p1], d1;		vldb.2d x2, [p0], d0;		vmac.f dm0, dm2, x3, x11, r2
	vlda.2d bmll2, [p1], d1
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3
	vlda.2d bmll1, [p1], d1;		vldb.2d x2, [p0], d0;		vconv.bf16.fp32 x0, cml0
	vconv.bf16.fp32 x1, cml4;		vmac.f dm4, dm1, x1, x11, r2
	vldb.2d x3, [p3], d3;		vmul.f dm3, x2, x0, r2
	vlda.2d bmlh1, [p1], d1;		vldb.2d x2, [p0], d0;		vmac.f dm0, dm2, x3, x11, r2
	vlda.2d bmll2, [p1], d1;		vmul.f dm3, x2, x1, r2
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3
	vlda.2d bmll1, [p1], d1;		vldb.2d x2, [p0], d0;		vconv.bf16.fp32 x0, cml0
	vconv.bf16.fp32 x1, cml4;		vmac.f dm4, dm1, x1, x11, r2
	vldb.2d x3, [p3], d3;		vconv.bf16.fp32 x0, cml3;		vmul.f dm3, x2, x0, r2
	vlda.2d bmlh1, [p1], d1;		vldb.2d x2, [p0], d0;		vmac.f dm0, dm2, x3, x11, r2
	vlda.2d bmll2, [p1], d1;		vmul.f dm3, x2, x1, r2
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3;		vconv.bf16.fp32 x3, cml3
	vlda.2d bmll1, [p1], d1;		vldb.2d x2, [p0], d0;		vconv.bf16.fp32 x0, cml0;		vconv.fp32.bf16 cml2, x0
	vconv.bf16.fp32 x1, cml4;		vconv.fp32.bf16 cmh2, x3;		vmac.f dm4, dm1, x1, x11, r2
	.p2align	4
gup_start:
	nopa;		vldb.2d x3, [p3], d3;		vconv.bf16.fp32 x0, cml3;		nopx;		nopm;		vmul.f dm3, x2, x0, r2
	vlda.2d bmlh1, [p1], d1;		vldb.2d x2, [p0], d0;		vst.push.576.conv.bfp16ebs8.fp32 dm2, [p2, sf, r26];		nopx;		nopm;		vmac.f dm0, dm2, x3, x11, r2
	vlda.2d bmll2, [p1], d1;		nopb;		vst.flush.512.conv.2d [p2, sf, r26, d4];		nopx;		nopm;		vmul.f dm3, x2, x1, r2
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3;		vconv.bf16.fp32 x3, cml3;		nopx;		nopm;		nopv
	vlda.2d bmll1, [p1], d1;		vldb.2d x2, [p0], d0;		vconv.bf16.fp32 x0, cml0;		nopx;		vconv.fp32.bf16 cml2, x0;		nopv
gup_last:
	nopa;		nopb;		vconv.bf16.fp32 x1, cml4;		nopx;		vconv.fp32.bf16 cmh2, x3;		vmac.f dm4, dm1, x1, x11, r2
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
.Lfunc_end_gup:
	.size	gup, .Lfunc_end_gup-gup
