	.text
	.section	.text.bgelu,"ax",@progbits
	.globl	bgelu
	.p2align	4
	.type	bgelu,@function
bgelu:
	mov	p5, p4
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
	movxm	r1, #16256
	vbcst.16	x11, r1
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
	mov	p3, p1
	mova	dn4, #7
	mova	dj4, #64
	movxm	m4, #576
	mova	dc4, #0
	paddxm	[sp], #512
	mov	p1, sp
	padda	[p1], #-512
	mova	dn1, #7
	mova	dj1, #64
	movxm	m1, #-448
	mova	dc1, #0
	mov	p4, p1
	mova	dn2, #7
	mova	dj2, #64
	movxm	m2, #-448
	mova	dc2, #0
	mov	p2, p1
	mova	dn3, #7
	mova	dj3, #64
	movxm	m3, #-448
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
	// modulo schedule: II 12, U 1, stages 7, ops 34
	vlda.2d bmll3, [p0], d0
	vldb.2d x3, [p5], d5
	nop
	vlda.2d bmlh3, [p0], d0
	nop
	nop
	nop
	nop
	vmac.f dm2, dm3, x3, x11, r1
	nop
	nop
	nop
	vlda.2d bmll3, [p0], d0
	vldb.2d x3, [p5], d5
	nop
	vlda.2d bmlh3, [p0], d0;		vst.2d.conv.bf16.fp32 cml2, [p1], d1
	nop
	nop
	vconv.bf16.fp32 x7, cml2
	nop
	vmin_ge.bf16 x7, r16, x8, x7;		vmac.f dm2, dm3, x3, x11, r1
	nop
	vmax_lt.bf16 x6, r16, x9, x7
	nop
	vlda.2d bmll3, [p0], d0;		vfloor.s32.bf16 x5, wh6, s0
	vldb.2d x3, [p5], d5
	vfloor.s32.bf16 x5, wl6, s0;		vadd.32 x4, x5, x10
	vlda.2d bmlh3, [p0], d0;		vst.2d.conv.bf16.fp32 cml2, [p1], d1
	vldb.4x64.lo wl3, wl4
	vldb.4x64.hi wh3, wl4;		vadd.32 x5, x5, x10
	vldb.4x64.hi wh2, wh4;		vconv.bf16.fp32 x7, cml2
	vldb.4x64.hi wh1, wl5
	vldb.4x64.lo wl2, wh4;		vmin_ge.bf16 x7, r16, x8, x7;		vmac.f dm2, dm3, x3, x11, r1
	vldb.4x64.lo wl1, wl5
	vldb.4x64.hi wh6, wh5;		vmax_lt.bf16 x6, r16, x9, x7
	vldb.4x64.lo wl6, wh5
	vlda.2d bmll3, [p0], d0;		vfloor.s32.bf16 x5, wh6, s0
	vldb.2d x3, [p5], d5
	vfloor.s32.bf16 x5, wl6, s0;		vadd.32 x4, x5, x10
	vlda.2d bmlh3, [p0], d0;		vldb.2d x1, [p4], d2;		vst.2d.conv.bf16.fp32 cml2, [p1], d1;		vshuffle bmlh0, x3, x2, r3
	vldb.4x64.lo wl3, wl4;		vshuffle x2, x3, x2, r2
	vldb.4x64.hi wh3, wl4;		vadd.32 x5, x5, x10
	vldb.4x64.hi wh2, wh4;		vconv.bf16.fp32 x7, cml2;		vshuffle bmll0, x1, x6, r3
	vldb.4x64.hi wh1, wl5;		vshuffle x6, x1, x6, r2
	vldb.4x64.lo wl2, wh4;		vmin_ge.bf16 x7, r16, x8, x7;		vmac.f dm2, dm3, x3, x11, r1
	vldb.4x64.lo wl1, wl5;		vmov wl2, wh6
	vldb.4x64.hi wh6, wh5;		vmax_lt.bf16 x6, r16, x9, x7
	vldb.4x64.lo wl6, wh5;		vmac.f dm1, dm0, x2, x1, r1
	vlda.2d bmll3, [p0], d0;		vfloor.s32.bf16 x5, wh6, s0
	vldb.2d x3, [p5], d5
	vlda.2d x0, [p2], d3;		vfloor.s32.bf16 x5, wl6, s0;		vadd.32 x4, x5, x10
	vlda.2d bmlh3, [p0], d0;		vldb.2d x1, [p4], d2;		vst.2d.conv.bf16.fp32 cml2, [p1], d1;		vshuffle bmlh0, x3, x2, r3
	vldb.4x64.lo wl3, wl4;		vshuffle x2, x3, x2, r2
	vldb.4x64.hi wh3, wl4;		vadd.32 x5, x5, x10
	vldb.4x64.hi wh2, wh4;		vconv.bf16.fp32 x7, cml2;		vshuffle bmll0, x1, x6, r3
	vldb.4x64.hi wh1, wl5;		vshuffle x6, x1, x6, r2
	vldb.4x64.lo wl2, wh4;		vconv.bf16.fp32 x4, cml1;		vmin_ge.bf16 x7, r16, x8, x7;		vmac.f dm2, dm3, x3, x11, r1
	vldb.4x64.lo wl1, wl5;		vmov wl2, wh6
	vldb.4x64.hi wh6, wh5;		vmax_lt.bf16 x6, r16, x9, x7;		vmsc.f dm3, dm1, x4, x11, r1
	vldb.4x64.lo wl6, wh5;		vmac.f dm1, dm0, x2, x1, r1
	vlda.2d bmll3, [p0], d0;		vfloor.s32.bf16 x5, wh6, s0
	vldb.2d x3, [p5], d5
	vlda.2d x0, [p2], d3;		vfloor.s32.bf16 x5, wl6, s0;		vadd.32 x4, x5, x10;		vmul.f dm2, x0, x4, r1
	vlda.2d bmlh3, [p0], d0;		vldb.2d x1, [p4], d2;		vst.2d.conv.bf16.fp32 cml2, [p1], d1;		vshuffle bmlh0, x3, x2, r3
	vldb.4x64.lo wl3, wl4;		vconv.bf16.fp32 x3, cml3;		vshuffle x2, x3, x2, r2
	vldb.4x64.hi wh3, wl4;		vadd.32 x5, x5, x10
	vldb.4x64.hi wh2, wh4;		vconv.bf16.fp32 x7, cml2;		vshuffle bmll0, x1, x6, r3;		vmac.f dm4, dm2, x0, x3, r1
	vldb.4x64.hi wh1, wl5;		vshuffle x6, x1, x6, r2
	vldb.4x64.lo wl2, wh4;		vconv.bf16.fp32 x4, cml1;		vmin_ge.bf16 x7, r16, x8, x7;		vmac.f dm2, dm3, x3, x11, r1
	vldb.4x64.lo wl1, wl5;		vmov wl2, wh6
	vldb.4x64.hi wh6, wh5;		vmax_lt.bf16 x6, r16, x9, x7;		vmsc.f dm3, dm1, x4, x11, r1
	vldb.4x64.lo wl6, wh5;		vmac.f dm1, dm0, x2, x1, r1
	.p2align	4
bgelu_start:
	vlda.2d bmll3, [p0], d0;		nopb;		vfloor.s32.bf16 x5, wh6, s0;		nopx;		nopm;		nopv
	nopa;		vldb.2d x3, [p5], d5;		vst.2d.conv.bf16.fp32 cml4, [p3], d4;		nopx;		nopm;		nopv
	vlda.2d x0, [p2], d3;		nopb;		vfloor.s32.bf16 x5, wl6, s0;		nopx;		vadd.32 x4, x5, x10;		vmul.f dm2, x0, x4, r1
	vlda.2d bmlh3, [p0], d0;		vldb.2d x1, [p4], d2;		vst.2d.conv.bf16.fp32 cml2, [p1], d1;		nopx;		vshuffle bmlh0, x3, x2, r3;		nopv
	nopa;		vldb.4x64.lo wl3, wl4;		vconv.bf16.fp32 x3, cml3;		nopx;		vshuffle x2, x3, x2, r2;		nopv
	nopa;		vldb.4x64.hi wh3, wl4;		nops;		nopx;		vadd.32 x5, x5, x10;		nopv
	nopa;		vldb.4x64.hi wh2, wh4;		vconv.bf16.fp32 x7, cml2;		nopx;		vshuffle bmll0, x1, x6, r3;		vmac.f dm4, dm2, x0, x3, r1
	nopa;		vldb.4x64.hi wh1, wl5;		nops;		nopx;		vshuffle x6, x1, x6, r2;		nopv
	nopa;		vldb.4x64.lo wl2, wh4;		vconv.bf16.fp32 x4, cml1;		nopx;		vmin_ge.bf16 x7, r16, x8, x7;		vmac.f dm2, dm3, x3, x11, r1
	nopa;		vldb.4x64.lo wl1, wl5;		nops;		nopx;		vmov wl2, wh6;		nopv
	nopa;		vldb.4x64.hi wh6, wh5;		nops;		nopx;		vmax_lt.bf16 x6, r16, x9, x7;		vmsc.f dm3, dm1, x4, x11, r1
bgelu_last:
	nopa;		vldb.4x64.lo wl6, wh5;		nops;		nopx;		nopm;		vmac.f dm1, dm0, x2, x1, r1
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
	paddxm	[sp], #-512
	nop
.Lfunc_end_bgelu:
	.size	bgelu, .Lfunc_end_bgelu-bgelu
