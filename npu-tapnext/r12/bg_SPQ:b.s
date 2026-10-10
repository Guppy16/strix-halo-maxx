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
	movxm	ls, #bgelu_start
	movxm	le, #bgelu_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 13, U 2, stages 6, ops 33
	vlda.2d bmll3, [p0], d0;		vldb.2d x5, [p5], d5
	nop
	nop
	nop
	nop
	vlda.2d bmlh3, [p0], d0
	nop
	nop
	nop
	vmac.f dm3, dm3, x5, x7, r1
	nop
	nop
	nop
	vlda.2d bmll3, [p0], d0;		vldb.2d x5, [p5], d5
	nop
	vst.2d.conv.bf16.fp32 cml3, [p1], d1
	vconv.bf16.fp32 x6, cml3
	nop
	vlda.2d bmlh3, [p0], d0
	vmin_ge.bf16 x6, r16, x8, x6
	nop
	vmax_lt.bf16 x6, r16, x9, x6
	vmac.f dm3, dm3, x5, x7, r1
	vfloor.s32.bf16 x5, wl6, s0
	vfloor.s32.bf16 x2, wh6, s0
	vadd.32 x5, x5, x10
	vlda.2d bmll3, [p0], d0;		vldb.2d x5, [p5], d5
	vldb.4x64.lo wl3, wl5;		vadd.32 x2, x2, x10
	vldb.4x64.lo wl4, wh5;		vst.2d.conv.bf16.fp32 cml3, [p1], d1
	vldb.4x64.hi wh4, wh5;		vconv.bf16.fp32 x6, cml3
	vldb.4x64.hi wh3, wl5
	vlda.2d bmlh3, [p0], d0
	vldb.4x64.hi wh1, wl2;		vmin_ge.bf16 x6, r16, x8, x6
	vldb.4x64.lo wl1, wl2
	vldb.4x64.hi wh4, wh2;		vmax_lt.bf16 x6, r16, x9, x6
	vldb.4x64.lo wl4, wh2;		vmac.f dm3, dm3, x5, x7, r1
	vfloor.s32.bf16 x5, wl6, s0
	vfloor.s32.bf16 x2, wh6, s0;		vshuffle bmll0, x3, x4, r3
	vldb.2d x1, [p4], d2;		vadd.32 x5, x5, x10
	vlda.2d bmll3, [p0], d0;		vldb.2d x5, [p5], d5;		vshuffle x3, x3, x4, r2
	vldb.4x64.lo wl3, wl5;		vadd.32 x2, x2, x10
	vldb.4x64.lo wl4, wh5;		vst.2d.conv.bf16.fp32 cml3, [p1], d1
	vldb.4x64.hi wh4, wh5;		vconv.bf16.fp32 x6, cml3;		vshuffle bmlh0, x1, x4, r3
	vldb.4x64.hi wh3, wl5;		vshuffle x4, x1, x4, r2
	vlda.2d bmlh3, [p0], d0;		vmov wl4, wh3
	vldb.4x64.hi wh0, wl2;		vmin_ge.bf16 x6, r16, x8, x6
	vldb.4x64.lo wl0, wl2;		vmac.f dm1, dm0, x4, x1, r1
	vldb.4x64.hi wh4, wh2;		vmax_lt.bf16 x6, r16, x9, x6
	vldb.4x64.lo wl4, wh2;		vmac.f dm3, dm3, x5, x7, r1
	vfloor.s32.bf16 x5, wl6, s0
	vfloor.s32.bf16 x2, wh6, s0;		vshuffle bmll0, x3, x4, r3
	vldb.2d x0, [p4], d2;		vadd.32 x5, x5, x10
	vlda.2d bmll3, [p0], d0;		vldb.2d x5, [p5], d5;		vconv.bf16.fp32 x6, cml1;		vshuffle x3, x3, x4, r2
	vldb.4x64.lo wl3, wl5;		vadd.32 x2, x2, x10
	vldb.4x64.lo wl4, wh5;		vst.2d.conv.bf16.fp32 cml3, [p1], d1;		vmul.f dm1, x1, x6, r1
	vldb.4x64.hi wh4, wh5;		vconv.bf16.fp32 x6, cml3;		vshuffle bmlh0, x0, x4, r3;		vmac.f dm2, dm1, x6, x11, r1
	vldb.4x64.hi wh3, wl5;		vshuffle x4, x0, x4, r2
	vlda.2d bmlh3, [p0], d0;		vmov wl4, wh3
	vldb.4x64.hi wh1, wl2;		vmin_ge.bf16 x6, r16, x8, x6
	vldb.4x64.lo wl1, wl2;		vmac.f dm2, dm0, x4, x0, r1
	vldb.4x64.hi wh4, wh2;		vmax_lt.bf16 x6, r16, x9, x6
	vldb.4x64.lo wl4, wh2;		vconv.bf16.fp32 x2, cml2;		vmac.f dm3, dm3, x5, x7, r1
	vfloor.s32.bf16 x5, wl6, s0
	vfloor.s32.bf16 x2, wh6, s0;		vshuffle bmll0, x3, x4, r3;		vmac.f dm1, dm1, x1, x2, r1
	vldb.2d x1, [p4], d2;		vadd.32 x5, x5, x10
	.p2align	4
bgelu_start:
	vlda.2d bmll3, [p0], d0;		vldb.2d x5, [p5], d5;		vconv.bf16.fp32 x6, cml2;		nopx;		vshuffle x3, x3, x4, r2;		nopv
	nopa;		vldb.4x64.lo wl3, wl5;		nops;		nopx;		vadd.32 x2, x2, x10;		nopv
	nopa;		vldb.4x64.lo wl4, wh5;		vst.2d.conv.bf16.fp32 cml3, [p1], d1;		nopx;		nopm;		vmul.f dm2, x0, x6, r1
	nopa;		vldb.4x64.hi wh4, wh5;		vconv.bf16.fp32 x6, cml3;		nopx;		vshuffle bmlh0, x1, x4, r3;		vmac.f dm1, dm2, x6, x11, r1
	nopa;		vldb.4x64.hi wh3, wl5;		vst.2d.conv.bf16.fp32 cml1, [p2], d3;		nopx;		vshuffle x4, x1, x4, r2;		nopv
	vlda.2d bmlh3, [p0], d0;		nopb;		nops;		nopx;		vmov wl4, wh3;		nopv
	nopa;		vldb.4x64.hi wh0, wl2;		nops;		nopx;		vmin_ge.bf16 x6, r16, x8, x6;		nopv
	nopa;		vldb.4x64.lo wl0, wl2;		nops;		nopx;		nopm;		vmac.f dm1, dm0, x4, x1, r1
	nopa;		vldb.4x64.hi wh4, wh2;		nops;		nopx;		vmax_lt.bf16 x6, r16, x9, x6;		nopv
	nopa;		vldb.4x64.lo wl4, wh2;		vconv.bf16.fp32 x2, cml1;		nopx;		nopm;		vmac.f dm3, dm3, x5, x7, r1
	nopa;		nopb;		vfloor.s32.bf16 x5, wl6, s0;		nopx;		nopm;		nopv
	nopa;		nopb;		vfloor.s32.bf16 x2, wh6, s0;		nopx;		vshuffle bmll0, x3, x4, r3;		vmac.f dm2, dm2, x0, x2, r1
	nopa;		vldb.2d x0, [p4], d2;		nops;		nopx;		vadd.32 x5, x5, x10;		nopv
	vlda.2d bmll3, [p0], d0;		vldb.2d x5, [p5], d5;		vconv.bf16.fp32 x6, cml1;		nopx;		vshuffle x3, x3, x4, r2;		nopv
	nopa;		vldb.4x64.lo wl3, wl5;		nops;		nopx;		vadd.32 x2, x2, x10;		nopv
	nopa;		vldb.4x64.lo wl4, wh5;		vst.2d.conv.bf16.fp32 cml3, [p1], d1;		nopx;		nopm;		vmul.f dm1, x1, x6, r1
	nopa;		vldb.4x64.hi wh4, wh5;		vconv.bf16.fp32 x6, cml3;		nopx;		vshuffle bmlh0, x0, x4, r3;		vmac.f dm2, dm1, x6, x11, r1
	nopa;		vldb.4x64.hi wh3, wl5;		vst.2d.conv.bf16.fp32 cml2, [p2], d3;		nopx;		vshuffle x4, x0, x4, r2;		nopv
	vlda.2d bmlh3, [p0], d0;		nopb;		nops;		nopx;		vmov wl4, wh3;		nopv
	nopa;		vldb.4x64.hi wh1, wl2;		nops;		nopx;		vmin_ge.bf16 x6, r16, x8, x6;		nopv
	nopa;		vldb.4x64.lo wl1, wl2;		nops;		nopx;		nopm;		vmac.f dm2, dm0, x4, x0, r1
	nopa;		vldb.4x64.hi wh4, wh2;		nops;		nopx;		vmax_lt.bf16 x6, r16, x9, x6;		nopv
	nopa;		vldb.4x64.lo wl4, wh2;		vconv.bf16.fp32 x2, cml2;		nopx;		nopm;		vmac.f dm3, dm3, x5, x7, r1
	nopa;		nopb;		vfloor.s32.bf16 x5, wl6, s0;		nopx;		nopm;		nopv
	nopa;		nopb;		vfloor.s32.bf16 x2, wh6, s0;		nopx;		vshuffle bmll0, x3, x4, r3;		vmac.f dm1, dm1, x1, x2, r1
bgelu_last:
	nopa;		vldb.2d x1, [p4], d2;		nops;		nopx;		vadd.32 x5, x5, x10;		nopv
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
