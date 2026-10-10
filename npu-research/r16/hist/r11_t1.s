	.text
	.section	.text.probe1,"ax",@progbits
	.globl	probe1
	.p2align	4
	.type	probe1,@function
probe1:
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
	movxm	ls, #probe1_start
	movxm	le, #probe1_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 9, U 2, stages 5, ops 27
	vlda.2d x1, [p0], d0
	vlda.2d x3, [p4], d2
	nop
	nop
	nop
	nop
	vmin_ge.bf16 x1, r16, x8, x1
	vmax_lt.bf16 x6, r16, x9, x1
	vst.2d x6, [p2], d3
	vlda.2d x0, [p0], d0;		vfloor.s32.bf16 x4, wl6, s0
	vlda.2d x3, [p4], d2;		vfloor.s32.bf16 x2, wh6, s0;		vadd.32 x7, x4, x10
	vldb.4x64.lo wl0, wl7;		vadd.32 x2, x2, x10
	vldb.4x64.hi wh0, wl7;		vst.2d x6, [p1], d1
	vldb.4x64.lo wl6, wh7
	vldb.4x64.hi wh6, wh7
	vmin_ge.bf16 x0, r16, x8, x0
	vldb.4x64.hi wh4, wl2;		vmax_lt.bf16 x5, r16, x9, x0
	vldb.4x64.lo wl5, wh2;		vst.2d x5, [p2], d3
	vlda.2d x1, [p0], d0;		vldb.4x64.hi wh5, wh2;		vfloor.s32.bf16 x4, wl5, s0
	vlda.2d x3, [p4], d2;		vldb.4x64.lo wl4, wl2;		vfloor.s32.bf16 x2, wh5, s0;		vadd.32 x7, x4, x10
	vldb.4x64.lo wl1, wl7;		vadd.32 x2, x2, x10
	vldb.4x64.hi wh1, wl7;		vst.2d x5, [p1], d1;		vshuffle bmll0, x0, x6, r3
	vldb.4x64.lo wl5, wh7;		vshuffle x0, x0, x6, r2
	vldb.4x64.hi wh5, wh7
	vmin_ge.bf16 x1, r16, x8, x1
	vldb.4x64.hi wh4, wl2;		vmax_lt.bf16 x6, r16, x9, x1
	vlda.2d x5, [p3], d4;		vldb.4x64.lo wl6, wh2;		vst.2d x6, [p2], d3;		vshuffle bmlh0, x4, x5, r3
	vlda.2d x0, [p0], d0;		vldb.4x64.hi wh6, wh2;		vfloor.s32.bf16 x4, wl6, s0;		vshuffle x3, x4, x5, r2
	vlda.2d x3, [p4], d2;		vldb.4x64.lo wl4, wl2;		vfloor.s32.bf16 x2, wh6, s0;		vadd.32 x7, x4, x10
	vldb.4x64.lo wl0, wl7;		vadd.32 x2, x2, x10
	vldb.4x64.hi wh0, wl7;		vst.2d x6, [p1], d1;		vshuffle bmll1, x1, x5, r3
	vldb.4x64.lo wl6, wh7;		vshuffle x1, x1, x5, r2
	vldb.4x64.hi wh6, wh7;		vmov wl3, wh0
	vmin_ge.bf16 x0, r16, x8, x0;		vmac.f dm2, dm0, x3, x5, r1
	vldb.4x64.hi wh4, wl2;		vmax_lt.bf16 x5, r16, x9, x0
	vlda.2d x6, [p3], d4;		vldb.4x64.lo wl5, wh2;		vst.2d x5, [p2], d3;		vshuffle bmlh1, x4, x6, r3
	.p2align	4
probe1_start:
	vlda.2d x1, [p0], d0;		vldb.4x64.hi wh5, wh2;		vfloor.s32.bf16 x4, wl5, s0;		nopx;		vshuffle x3, x4, x6, r2;		nopv
	vlda.2d x3, [p4], d2;		vldb.4x64.lo wl4, wl2;		vfloor.s32.bf16 x2, wh5, s0;		nopx;		vadd.32 x7, x4, x10;		nopv
	nopa;		vldb.4x64.lo wl1, wl7;		vconv.bf16.fp32 x4, cml2;		nopx;		vadd.32 x2, x2, x10;		nopv
	nopa;		vldb.4x64.hi wh1, wl7;		vst.2d x5, [p1], d1;		nopx;		vshuffle bmll0, x0, x6, r3;		vmac.f dm2, dm2, x4, x11, r1
	nopa;		vldb.4x64.lo wl5, wh7;		nops;		nopx;		vshuffle x0, x0, x6, r2;		nopv
	nopa;		vldb.4x64.hi wh5, wh7;		nops;		nopx;		vmov wl3, wh1;		nopv
	nopa;		nopb;		nops;		nopx;		vmin_ge.bf16 x1, r16, x8, x1;		vmac.f dm2, dm1, x3, x6, r1
	nopa;		vldb.4x64.hi wh4, wl2;		nops;		nopx;		vmax_lt.bf16 x6, r16, x9, x1;		nopv
	vlda.2d x5, [p3], d4;		vldb.4x64.lo wl6, wh2;		vst.2d x6, [p2], d3;		nopx;		vshuffle bmlh0, x4, x5, r3;		nopv
	vlda.2d x0, [p0], d0;		vldb.4x64.hi wh6, wh2;		vfloor.s32.bf16 x4, wl6, s0;		nopx;		vshuffle x3, x4, x5, r2;		nopv
	vlda.2d x3, [p4], d2;		vldb.4x64.lo wl4, wl2;		vfloor.s32.bf16 x2, wh6, s0;		nopx;		vadd.32 x7, x4, x10;		nopv
	nopa;		vldb.4x64.lo wl0, wl7;		vconv.bf16.fp32 x4, cml2;		nopx;		vadd.32 x2, x2, x10;		nopv
	nopa;		vldb.4x64.hi wh0, wl7;		vst.2d x6, [p1], d1;		nopx;		vshuffle bmll1, x1, x5, r3;		vmac.f dm2, dm2, x4, x11, r1
	nopa;		vldb.4x64.lo wl6, wh7;		nops;		nopx;		vshuffle x1, x1, x5, r2;		nopv
	nopa;		vldb.4x64.hi wh6, wh7;		nops;		nopx;		vmov wl3, wh0;		nopv
	nopa;		nopb;		nops;		nopx;		vmin_ge.bf16 x0, r16, x8, x0;		vmac.f dm2, dm0, x3, x5, r1
	nopa;		vldb.4x64.hi wh4, wl2;		nops;		nopx;		vmax_lt.bf16 x5, r16, x9, x0;		nopv
probe1_last:
	vlda.2d x6, [p3], d4;		vldb.4x64.lo wl5, wh2;		vst.2d x5, [p2], d3;		nopx;		vshuffle bmlh1, x4, x6, r3;		nopv
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
.Lfunc_end_probe1:
	.size	probe1, .Lfunc_end_probe1-probe1
