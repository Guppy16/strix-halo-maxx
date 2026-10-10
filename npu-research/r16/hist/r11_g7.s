	.text
	.section	.text.probe7,"ax",@progbits
	.globl	probe7
	.p2align	4
	.type	probe7,@function
probe7:
	movxm	r16, #52428
	mov	m0, r1
	mov	m1, r1
	mov	m2, r1
	mov	r1, p2
	vbcst.32	x0, r1
	mov	r1, p3
	vbcst.32	x1, r1
	mova	r1, #512
	vbcst.32	x2, r1
	vsel.32	x3, x0, x1, r16
	nop
	nop
	vadd.32	x10, x3, x2
	movxm	r1, #16511
	vbcst.16	x8, r1
	movxm	r1, #-16256
	vbcst.16	x9, r1
	movxm	r1, #-16512
	vbcst.16	x11, r1
	mova	r1, #828
	mova	r2, #24
	mova	r3, #5
	mov	s0, #7
	mov	p4, p0
	paddxm	[sp], #256
	mov	p2, sp
	padda	[p2], #-256
	mov	p3, p2
	mova	dn3, #3
	mova	dj3, #64
	mova	m3, #-192
	mova	dc3, #0
	mova	dn4, #3
	mova	dj4, #64
	mova	m4, #-192
	mova	dc4, #0
	mova	dn0, #7
	mova	dj0, #64
	mova	dc0, #0
	mova	dn1, #7
	mova	dj1, #64
	mova	dc1, #0
	mova	dn2, #7
	mova	dj2, #64
	mova	dc2, #0
	movxm	ls, #probe7_start
	movxm	le, #probe7_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 10, U 1, stages 4, ops 27
	vlda.2d x2, [p0], d0
	vlda.2d x6, [p4], d2
	nop
	nop
	nop
	nop
	vmin_ge.bf16 x5, r16, x8, x2
	vmax_lt.bf16 x5, r16, x9, x5
	vst.2d x5, [p2], d3
	vfloor.s32.bf16 x0, wl5, s0
	vlda.2d x2, [p0], d0;		vfloor.s32.bf16 x0, wh5, s0;		vadd.32 x5, x0, x10
	vlda.2d x6, [p4], d2;		vldb.4x64.lo wl2, wl5;		vadd.32 x1, x0, x10
	vldb.4x64.hi wh2, wl5
	vldb.4x64.lo wl3, wh5
	vldb.4x64.hi wh3, wh5
	vldb.4x64.lo wl4, wl1
	vlda.2d x0, [p3], d4;		vldb.4x64.hi wh4, wl1;		vmin_ge.bf16 x5, r16, x8, x2
	vldb.4x64.lo wl6, wh1;		vmax_lt.bf16 x5, r16, x9, x5
	vldb.4x64.hi wh6, wh1;		vst.2d x5, [p2], d3
	vfloor.s32.bf16 x0, wl5, s0
	vlda.2d x2, [p0], d0;		vfloor.s32.bf16 x0, wh5, s0;		vadd.32 x5, x0, x10
	vlda.2d x6, [p4], d2;		vldb.4x64.lo wl2, wl5;		vadd.32 x1, x0, x10
	vldb.4x64.hi wh2, wl5;		vst.2d x0, [p1], d1;		vshuffle bmll0, x2, x3, r3
	vldb.4x64.lo wl3, wh5;		vshuffle x3, x2, x3, r2
	vldb.4x64.hi wh3, wh5;		vshuffle bmlh0, x4, x6, r3
	vldb.4x64.lo wl4, wl1;		vshuffle x4, x4, x6, r2
	vlda.2d x0, [p3], d4;		vldb.4x64.hi wh4, wl1;		vmin_ge.bf16 x5, r16, x8, x2
	vldb.4x64.lo wl6, wh1;		vmax_lt.bf16 x5, r16, x9, x5
	vldb.4x64.hi wh6, wh1;		vst.2d x5, [p2], d3;		vmov wl4, wh3
	vfloor.s32.bf16 x0, wl5, s0;		vmac.f dm1, dm0, x4, x0, r1
	.p2align	4
probe7_start:
	vlda.2d x2, [p0], d0;		nopb;		vfloor.s32.bf16 x0, wh5, s0;		nopx;		vadd.32 x5, x0, x10;		nopv
	vlda.2d x6, [p4], d2;		vldb.4x64.lo wl2, wl5;		vadd.32 x1, x0, x10
	vldb.4x64.hi wh2, wl5;		vst.2d x0, [p1], d1;		vshuffle bmll0, x2, x3, r3
	vldb.4x64.lo wl3, wh5;		vshuffle x3, x2, x3, r2
	vldb.4x64.hi wh3, wh5;		vconv.bf16.fp32 x2, cml1;		vshuffle bmlh0, x4, x6, r3
	vldb.4x64.lo wl4, wl1;		vshuffle x4, x4, x6, r2;		vmac.f dm1, dm1, x2, x11, r1
	vlda.2d x0, [p3], d4;		vldb.4x64.hi wh4, wl1;		vmin_ge.bf16 x5, r16, x8, x2
	vldb.4x64.lo wl6, wh1;		vmax_lt.bf16 x5, r16, x9, x5
	vldb.4x64.hi wh6, wh1;		vst.2d x5, [p2], d3;		vmov wl4, wh3
probe7_last:
	nopa;		nopb;		vfloor.s32.bf16 x0, wl5, s0;		nopx;		nopm;		vmac.f dm1, dm0, x4, x0, r1
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
	paddxm	[sp], #-256
	nop
.Lfunc_end_probe7:
	.size	probe7, .Lfunc_end_probe7-probe7
