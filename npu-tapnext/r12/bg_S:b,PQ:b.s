	.text
	.section	.text.bgelu,"ax",@progbits
	.globl	bgelu
	.p2align	4
	.type	bgelu,@function
bgelu:
	movxm	r16, #52428
	mov	r1, p2
	vbcst.32	x0, r1
	mov	r1, p3
	vbcst.32	x1, r1
	mova	r1, #576
	vbcst.32	x2, r1
	vsel.32	x3, x0, x1, r16
	nop
	nop
	vadd.32	x10, x3, x2
	movxm	r1, #16515
	vbcst.16	x8, r1
	movxm	r1, #-16252
	vbcst.16	x9, r1
	movxm	r1, #-16512
	vbcst.16	x11, r1
	movxm	r1, #16256
	vbcst.16	x7, r1
	mova	r1, #828
	mova	r2, #24
	mova	r3, #5
	mov	s0, #7
	mova	dn0, #15
	mova	dj0, #64
	mova	m0, #64
	mova	dc0, #0
	mova	dn5, #7
	mova	dj5, #64
	movxm	m5, #-448
	mova	dc5, #0
	mova	dn1, #7
	mova	dj1, #64
	movxm	m1, #576
	mova	dc1, #0
	mov	p4, p1
	mova	dn2, #7
	mova	dj2, #64
	movxm	m2, #576
	mova	dc2, #0
	mov	p2, p1
	mova	dn3, #7
	mova	dj3, #64
	movxm	m3, #576
	mova	dc3, #0
	mov	p3, p1
	mova	dn4, #7
	mova	dj4, #64
	movxm	m4, #576
	mova	dc4, #0
	movxm	ls, #bgelu_start
	movxm	le, #bgelu_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 13, U 1, stages 6, ops 34
	vlda.2d bmll0, [p0], d0
	vlda.2d bmlh0, [p0], d0;		vldb.2d x2, [p5], d5
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm1, dm0, x2, x7, r1
	nop
	nop
	nop
	nop
	vlda.2d bmll0, [p0], d0
	vlda.2d bmlh0, [p0], d0;		vldb.2d x2, [p5], d5;		vst.2d.conv.bf16.fp32 cml1, [p1], d1
	vconv.bf16.fp32 x4, cml1
	nop
	nop
	nop
	vmin_ge.bf16 x4, r16, x8, x4
	nop
	vmax_lt.bf16 x0, r16, x9, x4;		vmac.f dm1, dm0, x2, x7, r1
	nop
	vfloor.s32.bf16 x1, wh0, s0
	vfloor.s32.bf16 x0, wl0, s0
	vadd.32 x1, x1, x10
	vlda.2d bmll0, [p0], d0;		vadd.32 x0, x0, x10
	vlda.2d bmlh0, [p0], d0;		vldb.2d x2, [p5], d5;		vst.2d.conv.bf16.fp32 cml1, [p1], d1
	vldb.4x64.lo wl5, wl0;		vconv.bf16.fp32 x4, cml1
	vldb.4x64.hi wh4, wh0
	vldb.4x64.lo wl6, wh1
	vldb.4x64.hi wh6, wh1
	vldb.4x64.hi wh5, wl0;		vmin_ge.bf16 x4, r16, x8, x4
	vldb.4x64.lo wl4, wh0
	vldb.4x64.hi wh3, wl1;		vmax_lt.bf16 x0, r16, x9, x4;		vmac.f dm1, dm0, x2, x7, r1
	vldb.4x64.lo wl3, wl1
	vfloor.s32.bf16 x1, wh0, s0
	vfloor.s32.bf16 x0, wl0, s0
	vadd.32 x1, x1, x10
	vlda.2d bmll0, [p0], d0;		vldb.2d x5, [p4], d2;		vadd.32 x0, x0, x10
	vlda.2d bmlh0, [p0], d0;		vldb.2d x2, [p5], d5;		vst.2d.conv.bf16.fp32 cml1, [p1], d1;		vshuffle bmll2, x5, x4, r3
	vldb.4x64.lo wl5, wl0;		vconv.bf16.fp32 x4, cml1;		vshuffle x5, x5, x4, r2
	vldb.4x64.hi wh4, wh0;		vshuffle bmlh2, x3, x6, r3
	vldb.4x64.lo wl6, wh1;		vshuffle x6, x3, x6, r2
	vldb.4x64.hi wh6, wh1;		vmov wl6, wh5
	vldb.4x64.hi wh5, wl0;		vmin_ge.bf16 x4, r16, x8, x4
	vldb.4x64.lo wl4, wh0;		vmac.f dm0, dm2, x6, x5, r1
	vldb.4x64.hi wh3, wl1;		vmax_lt.bf16 x0, r16, x9, x4;		vmac.f dm1, dm0, x2, x7, r1
	vldb.4x64.lo wl3, wl1
	vfloor.s32.bf16 x1, wh0, s0
	vfloor.s32.bf16 x0, wl0, s0
	vldb.2d x3, [p2], d3;		vadd.32 x1, x1, x10
	vlda.2d bmll0, [p0], d0;		vldb.2d x5, [p4], d2;		vconv.bf16.fp32 x2, cml0;		vadd.32 x0, x0, x10
	vlda.2d bmlh0, [p0], d0;		vldb.2d x2, [p5], d5;		vst.2d.conv.bf16.fp32 cml1, [p1], d1;		vshuffle bmll2, x5, x4, r3
	vldb.4x64.lo wl5, wl0;		vconv.bf16.fp32 x4, cml1;		vshuffle x5, x5, x4, r2;		vmac.f dm1, dm0, x2, x11, r1
	vldb.4x64.hi wh4, wh0;		vshuffle bmlh2, x3, x6, r3
	vldb.4x64.lo wl6, wh1;		vshuffle x6, x3, x6, r2
	vldb.4x64.hi wh6, wh1;		vmov wl6, wh5
	vldb.4x64.hi wh5, wl0;		vmin_ge.bf16 x4, r16, x8, x4;		vmul.f dm2, x3, x2, r1
	vldb.4x64.lo wl4, wh0;		vmac.f dm0, dm2, x6, x5, r1
	vldb.4x64.hi wh3, wl1;		vconv.bf16.fp32 x2, cml1;		vmax_lt.bf16 x0, r16, x9, x4;		vmac.f dm1, dm0, x2, x7, r1
	vldb.4x64.lo wl3, wl1
	vfloor.s32.bf16 x1, wh0, s0
	vfloor.s32.bf16 x0, wl0, s0;		vmac.f dm1, dm2, x3, x2, r1
	vldb.2d x3, [p2], d3;		vadd.32 x1, x1, x10
	.p2align	4
bgelu_start:
	vlda.2d bmll0, [p0], d0;		vldb.2d x5, [p4], d2;		vconv.bf16.fp32 x2, cml0;		nopx;		vadd.32 x0, x0, x10;		nopv
	vlda.2d bmlh0, [p0], d0;		vldb.2d x2, [p5], d5;		vst.2d.conv.bf16.fp32 cml1, [p1], d1;		nopx;		vshuffle bmll2, x5, x4, r3;		nopv
	nopa;		vldb.4x64.lo wl5, wl0;		vconv.bf16.fp32 x4, cml1;		nopx;		vshuffle x5, x5, x4, r2;		vmac.f dm1, dm0, x2, x11, r1
	nopa;		vldb.4x64.hi wh4, wh0;		nops;		nopx;		vshuffle bmlh2, x3, x6, r3;		nopv
	nopa;		vldb.4x64.lo wl6, wh1;		vst.2d.conv.bf16.fp32 cml1, [p3], d4;		nopx;		vshuffle x6, x3, x6, r2;		nopv
	nopa;		vldb.4x64.hi wh6, wh1;		nops;		nopx;		vmov wl6, wh5;		nopv
	nopa;		vldb.4x64.hi wh5, wl0;		nops;		nopx;		vmin_ge.bf16 x4, r16, x8, x4;		vmul.f dm2, x3, x2, r1
	nopa;		vldb.4x64.lo wl4, wh0;		nops;		nopx;		nopm;		vmac.f dm0, dm2, x6, x5, r1
	nopa;		vldb.4x64.hi wh3, wl1;		vconv.bf16.fp32 x2, cml1;		nopx;		vmax_lt.bf16 x0, r16, x9, x4;		vmac.f dm1, dm0, x2, x7, r1
	nopa;		vldb.4x64.lo wl3, wl1;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		vfloor.s32.bf16 x1, wh0, s0;		nopx;		nopm;		nopv
	nopa;		nopb;		vfloor.s32.bf16 x0, wl0, s0;		nopx;		nopm;		vmac.f dm1, dm2, x3, x2, r1
bgelu_last:
	nopa;		vldb.2d x3, [p2], d3;		nops;		nopx;		vadd.32 x1, x1, x10;		nopv
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
.Lfunc_end_bgelu:
	.size	bgelu, .Lfunc_end_bgelu-bgelu
