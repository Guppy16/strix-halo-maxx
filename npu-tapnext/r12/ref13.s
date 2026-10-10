	.text
	.section	.text.gelu_ms,"ax",@progbits
	.globl	gelu_ms
	.p2align	4
	.type	gelu_ms,@function
gelu_ms:
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
	movxm	ls, #gelu_ms_start
	movxm	le, #gelu_ms_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 13, U 1, stages 5, ops 30
	vlda.2d x2, [p0], d0
	nop
	nop
	nop
	nop
	nop
	nop
	vmin_ge.bf16 x7, r16, x8, x2
	nop
	vmax_lt.bf16 x7, r16, x9, x7
	nop
	vst.2d x7, [p2], d3
	vfloor.s32.bf16 x5, wl7, s0
	vlda.2d x2, [p0], d0;		vfloor.s32.bf16 x4, wh7, s0
	vadd.32 x7, x5, x10
	vadd.32 x0, x4, x10
	vldb.4x64.lo wl3, wl7
	vldb.4x64.hi wh3, wl7
	vldb.4x64.lo wl6, wh7
	nop
	vldb.4x64.hi wh6, wh7;		vmin_ge.bf16 x7, r16, x8, x2
	vldb.4x64.lo wl1, wl0
	vldb.4x64.hi wh1, wl0;		vmax_lt.bf16 x7, r16, x9, x7
	vldb.4x64.lo wl4, wh0
	vst.2d x7, [p2], d3
	vfloor.s32.bf16 x5, wl7, s0
	vlda.2d x2, [p0], d0;		vfloor.s32.bf16 x4, wh7, s0
	vldb.4x64.hi wh4, wh0;		vadd.32 x7, x5, x10
	vadd.32 x0, x4, x10
	vldb.4x64.lo wl3, wl7;		vshuffle bmll0, x3, x6, r3
	vldb.4x64.hi wh3, wl7;		vshuffle x5, x3, x6, r2
	vldb.4x64.lo wl6, wh7
	vlda.2d x4, [p3], d4
	vldb.4x64.hi wh6, wh7;		vmin_ge.bf16 x7, r16, x8, x2
	vldb.4x64.lo wl1, wl0;		vshuffle bmlh0, x1, x4, r3
	vldb.4x64.hi wh1, wl0;		vmax_lt.bf16 x7, r16, x9, x7
	vldb.4x64.lo wl4, wh0;		vshuffle x1, x1, x4, r2
	vst.2d x7, [p2], d3;		vmov wl1, wh5
	vfloor.s32.bf16 x5, wl7, s0
	vlda.2d x2, [p0], d0;		vfloor.s32.bf16 x4, wh7, s0;		vmac.f dm1, dm0, x1, x4, r1
	vldb.4x64.hi wh4, wh0;		vadd.32 x7, x5, x10
	vlda.2d x2, [p4], d2;		vadd.32 x0, x4, x10
	vldb.4x64.lo wl3, wl7;		vshuffle bmll0, x3, x6, r3
	vldb.4x64.hi wh3, wl7;		vshuffle x5, x3, x6, r2
	vldb.4x64.lo wl6, wh7
	vlda.2d x4, [p3], d4;		vconv.bf16.fp32 x6, cml1
	vldb.4x64.hi wh6, wh7;		vmin_ge.bf16 x7, r16, x8, x2
	vldb.4x64.lo wl1, wl0;		vshuffle bmlh0, x1, x4, r3;		vmac.f dm1, dm1, x6, x11, r1
	vldb.4x64.hi wh1, wl0;		vmax_lt.bf16 x7, r16, x9, x7;		vmul.f dm2, x2, x6, r1
	vldb.4x64.lo wl4, wh0;		vshuffle x1, x1, x4, r2
	vst.2d x7, [p2], d3;		vmov wl1, wh5
	vfloor.s32.bf16 x5, wl7, s0
	.p2align	4
gelu_ms_start:
	vlda.2d x2, [p0], d0;		nopb;		vfloor.s32.bf16 x4, wh7, s0;		nopx;		nopm;		vmac.f dm1, dm0, x1, x4, r1
	nopa;		vldb.4x64.hi wh4, wh0;		vconv.bf16.fp32 x5, cml1;		nopx;		vadd.32 x7, x5, x10;		nopv
	vlda.2d x2, [p4], d2;		nopb;		nops;		nopx;		vadd.32 x0, x4, x10;		nopv
	nopa;		vldb.4x64.lo wl3, wl7;		nops;		nopx;		vshuffle bmll0, x3, x6, r3;		vmac.f dm2, dm2, x2, x5, r1
	nopa;		vldb.4x64.hi wh3, wl7;		nops;		nopx;		vshuffle x5, x3, x6, r2;		nopv
	nopa;		vldb.4x64.lo wl6, wh7;		nops;		nopx;		nopm;		nopv
	vlda.2d x4, [p3], d4;		nopb;		vconv.bf16.fp32 x6, cml1;		nopx;		nopm;		nopv
	nopa;		vldb.4x64.hi wh6, wh7;		nops;		nopx;		vmin_ge.bf16 x7, r16, x8, x2;		nopv
	nopa;		vldb.4x64.lo wl1, wl0;		nops;		nopx;		vshuffle bmlh0, x1, x4, r3;		vmac.f dm1, dm1, x6, x11, r1
	nopa;		vldb.4x64.hi wh1, wl0;		vst.2d.conv.bf16.fp32 cml2, [p1], d1;		nopx;		vmax_lt.bf16 x7, r16, x9, x7;		vmul.f dm2, x2, x6, r1
	nopa;		vldb.4x64.lo wl4, wh0;		nops;		nopx;		vshuffle x1, x1, x4, r2;		nopv
	nopa;		nopb;		vst.2d x7, [p2], d3;		nopx;		vmov wl1, wh5;		nopv
gelu_ms_last:
	nopa;		nopb;		vfloor.s32.bf16 x5, wl7, s0;		nopx;		nopm;		nopv
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
.Lfunc_end_gelu_ms:
	.size	gelu_ms, .Lfunc_end_gelu_ms-gelu_ms
