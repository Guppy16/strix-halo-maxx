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
	// modulo schedule: II 7, U 2, stages 5, ops 20
	vlda.2d bmll2, [p1], d1
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3
	vlda.2d bmll3, [p1], d1
	vlda.2d bmlh3, [p1], d1;		vldb.2d x4, [p3], d3
	nop
	nop
	nop
	vlda.2d bmll2, [p1], d1
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3;		vmac.f dm0, dm2, x1, x11, r2
	vlda.2d bmll3, [p1], d1;		vldb.2d x2, [p0], d0
	vlda.2d bmlh3, [p1], d1;		vldb.2d x4, [p3], d3;		vmac.f dm0, dm3, x4, x11, r2
	vldb.2d x2, [p0], d0
	nop
	nop
	vlda.2d bmll2, [p1], d1;		vconv.bf16.fp32 x3, cml0
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3;		vmac.f dm1, dm2, x1, x11, r2
	vlda.2d bmll3, [p1], d1;		vldb.2d x2, [p0], d0;		vconv.bf16.fp32 x3, cml0;		vmul.f dm4, x2, x3, r2
	vlda.2d bmlh3, [p1], d1;		vldb.2d x4, [p3], d3;		vmac.f dm1, dm3, x4, x11, r2
	vldb.2d x2, [p0], d0;		vmul.f dm4, x2, x3, r2
	nop
	nop
	vlda.2d bmll2, [p1], d1;		vconv.bf16.fp32 x3, cml1
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3;		vconv.bf16.fp32 x1, cml4;		vmac.f dm0, dm2, x1, x11, r2
	vlda.2d bmll3, [p1], d1;		vldb.2d x2, [p0], d0;		vconv.bf16.fp32 x3, cml1;		vmul.f dm4, x2, x3, r2
	vlda.2d bmlh3, [p1], d1;		vldb.2d x4, [p3], d3;		vconv.bf16.fp32 x0, cml4;		vconv.fp32.bf16 cml1, x1;		vmac.f dm0, dm3, x4, x11, r2
	vldb.2d x2, [p0], d0;		vmul.f dm4, x2, x3, r2
	nop
	nop
	.p2align	4
gup_start:
	vlda.2d bmll2, [p1], d1;		nopb;		vconv.bf16.fp32 x3, cml0;		nopx;		nopm;		nopv
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3;		vconv.bf16.fp32 x1, cml4;		nopx;		nopm;		vmac.f dm1, dm2, x1, x11, r2
	vlda.2d bmll3, [p1], d1;		vldb.2d x2, [p0], d0;		vconv.bf16.fp32 x3, cml0;		nopx;		vconv.fp32.bf16 cmh1, x0;		vmul.f dm4, x2, x3, r2
	vlda.2d bmlh3, [p1], d1;		vldb.2d x4, [p3], d3;		vconv.bf16.fp32 x0, cml4;		nopx;		vconv.fp32.bf16 cml0, x1;		vmac.f dm1, dm3, x4, x11, r2
	nopa;		vldb.2d x2, [p0], d0;		vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26];		nopx;		nopm;		vmul.f dm4, x2, x3, r2
	nopa;		nopb;		vst.flush.512.conv.2d [p2, sf, r26, d4];		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	vlda.2d bmll2, [p1], d1;		nopb;		vconv.bf16.fp32 x3, cml1;		nopx;		nopm;		nopv
	vlda.2d bmlh2, [p1], d1;		vldb.2d x1, [p3], d3;		vconv.bf16.fp32 x1, cml4;		nopx;		nopm;		vmac.f dm0, dm2, x1, x11, r2
	vlda.2d bmll3, [p1], d1;		vldb.2d x2, [p0], d0;		vconv.bf16.fp32 x3, cml1;		nopx;		vconv.fp32.bf16 cmh0, x0;		vmul.f dm4, x2, x3, r2
	vlda.2d bmlh3, [p1], d1;		vldb.2d x4, [p3], d3;		vconv.bf16.fp32 x0, cml4;		nopx;		vconv.fp32.bf16 cml1, x1;		vmac.f dm0, dm3, x4, x11, r2
	nopa;		vldb.2d x2, [p0], d0;		vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26];		nopx;		nopm;		vmul.f dm4, x2, x3, r2
	nopa;		nopb;		vst.flush.512.conv.2d [p2, sf, r26, d4];		nopx;		nopm;		nopv
gup_last:
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
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
