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
	// modulo schedule: II 11, U 2, stages 7, ops 34
	vlda.2d bmll2, [p0], d0;		vldb.2d x5, [p5], d5
	nop
	nop
	vlda.2d bmlh2, [p0], d0
	nop
	nop
	nop
	vmac.f dm3, dm2, x5, x11, r1
	nop
	nop
	nop
	vlda.2d bmll0, [p0], d0;		vldb.2d x6, [p5], d5
	nop
	vconv.bf16.fp32 x5, cml3
	vlda.2d bmlh0, [p0], d0
	nop
	vmin_ge.bf16 x5, r16, x8, x5
	vst.2d.conv.bf16.fp32 cml3, [p1], d1
	vmac.f dm1, dm0, x6, x11, r1
	vmax_lt.bf16 x6, r16, x9, x5
	nop
	nop
	vlda.2d bmll2, [p0], d0;		vldb.2d x5, [p5], d5;		vfloor.s32.bf16 x4, wl6, s0
	nop
	vconv.bf16.fp32 x5, cml1;		vadd.32 x4, x4, x10
	vlda.2d bmlh2, [p0], d0;		vfloor.s32.bf16 x7, wh6, s0
	vldb.4x64.lo wl0, wh4
	vldb.4x64.hi wh0, wh4;		vmin_ge.bf16 x3, r16, x8, x5
	vldb.4x64.hi wh6, wl4;		vst.2d.conv.bf16.fp32 cml1, [p1], d1;		vadd.32 x1, x7, x10
	vldb.4x64.lo wl6, wl4;		vmac.f dm3, dm2, x5, x11, r1
	vldb.4x64.hi wh1, wh1;		vmax_lt.bf16 x3, r16, x9, x3
	vldb.4x64.lo wl1, wh1
	nop
	vlda.2d bmll0, [p0], d0;		vldb.2d x6, [p5], d5;		vfloor.s32.bf16 x7, wl3, s0
	vldb.4x64.lo wl2, wl1
	vldb.4x64.hi wh2, wl1;		vconv.bf16.fp32 x5, cml3;		vadd.32 x4, x7, x10
	vlda.2d bmlh0, [p0], d0;		vldb.2d x5, [p4], d2;		vfloor.s32.bf16 x7, wh3, s0;		vshuffle x3, x6, x0, r2
	vldb.4x64.lo wl0, wh4;		vshuffle bmll2, x6, x0, r3
	vldb.4x64.hi wh0, wh4;		vmin_ge.bf16 x5, r16, x8, x5
	vldb.4x64.hi wh3, wl4;		vst.2d.conv.bf16.fp32 cml3, [p1], d1;		vadd.32 x7, x7, x10
	vldb.4x64.lo wl3, wl4;		vmac.f dm1, dm0, x6, x11, r1
	vldb.4x64.hi wh2, wh7;		vmax_lt.bf16 x6, r16, x9, x5
	vldb.4x64.lo wl2, wh7;		vshuffle bmlh2, x2, x1, r3
	vshuffle x2, x2, x1, r2
	vlda.2d bmll2, [p0], d0;		vldb.2d x5, [p5], d5;		vfloor.s32.bf16 x4, wl6, s0;		vmov wl2, wh3
	vldb.4x64.lo wl7, wl7
	vldb.4x64.hi wh7, wl7;		vconv.bf16.fp32 x5, cml1;		vadd.32 x4, x4, x10;		vmac.f dm1, dm2, x2, x5, r1
	vlda.2d bmlh2, [p0], d0;		vldb.2d x5, [p4], d2;		vfloor.s32.bf16 x7, wh6, s0;		vshuffle x6, x3, x0, r2
	vldb.4x64.lo wl0, wh4;		vshuffle bmll0, x3, x0, r3
	vldb.4x64.hi wh0, wh4;		vmin_ge.bf16 x3, r16, x8, x5
	vldb.4x64.hi wh6, wl4;		vst.2d.conv.bf16.fp32 cml1, [p1], d1;		vadd.32 x1, x7, x10
	vldb.4x64.lo wl6, wl4;		vmac.f dm3, dm2, x5, x11, r1
	vldb.4x64.hi wh1, wh1;		vmax_lt.bf16 x3, r16, x9, x3
	vldb.4x64.lo wl1, wh1;		vshuffle bmlh0, x7, x2, r3
	vlda.2d x0, [p2], d3;		vconv.bf16.fp32 x2, cml1;		vshuffle x4, x7, x2, r2
	vlda.2d bmll0, [p0], d0;		vldb.2d x6, [p5], d5;		vfloor.s32.bf16 x7, wl3, s0;		vmov wl4, wh6
	vldb.4x64.lo wl2, wl1;		vmsc.f dm1, dm1, x2, x11, r1
	vldb.4x64.hi wh2, wl1;		vconv.bf16.fp32 x5, cml3;		vadd.32 x4, x7, x10;		vmac.f dm3, dm0, x4, x5, r1
	vlda.2d bmlh0, [p0], d0;		vldb.2d x5, [p4], d2;		vfloor.s32.bf16 x7, wh3, s0;		vshuffle x3, x6, x0, r2
	vldb.4x64.lo wl0, wh4;		vshuffle bmll2, x6, x0, r3
	vldb.4x64.hi wh0, wh4;		vmin_ge.bf16 x5, r16, x8, x5
	vldb.4x64.hi wh3, wl4;		vst.2d.conv.bf16.fp32 cml3, [p1], d1;		vadd.32 x7, x7, x10;		vmul.f dm0, x0, x2, r1
	vldb.4x64.lo wl3, wl4;		vconv.bf16.fp32 x4, cml1;		vmac.f dm1, dm0, x6, x11, r1
	vldb.4x64.hi wh2, wh7;		vmax_lt.bf16 x6, r16, x9, x5
	vldb.4x64.lo wl2, wh7;		vshuffle bmlh2, x2, x1, r3;		vmac.f dm0, dm0, x0, x4, r1
	vlda.2d x0, [p2], d3;		vconv.bf16.fp32 x1, cml3;		vshuffle x2, x2, x1, r2
	.p2align	4
bgelu_start:
	vlda.2d bmll2, [p0], d0;		vldb.2d x5, [p5], d5;		vfloor.s32.bf16 x4, wl6, s0;		nopx;		vmov wl2, wh3;		nopv
	nopa;		vldb.4x64.lo wl7, wl7;		nops;		nopx;		nopm;		vmsc.f dm3, dm3, x1, x11, r1
	nopa;		vldb.4x64.hi wh7, wl7;		vconv.bf16.fp32 x5, cml1;		nopx;		vadd.32 x4, x4, x10;		vmac.f dm1, dm2, x2, x5, r1
	vlda.2d bmlh2, [p0], d0;		vldb.2d x5, [p4], d2;		vfloor.s32.bf16 x7, wh6, s0;		nopx;		vshuffle x6, x3, x0, r2;		nopv
	nopa;		vldb.4x64.lo wl0, wh4;		vst.2d.conv.bf16.fp32 cml0, [p3], d4;		nopx;		vshuffle bmll0, x3, x0, r3;		nopv
	nopa;		vldb.4x64.hi wh0, wh4;		nops;		nopx;		vmin_ge.bf16 x3, r16, x8, x5;		nopv
	nopa;		vldb.4x64.hi wh6, wl4;		vst.2d.conv.bf16.fp32 cml1, [p1], d1;		nopx;		vadd.32 x1, x7, x10;		vmul.f dm2, x0, x1, r1
	nopa;		vldb.4x64.lo wl6, wl4;		vconv.bf16.fp32 x4, cml3;		nopx;		nopm;		vmac.f dm3, dm2, x5, x11, r1
	nopa;		vldb.4x64.hi wh1, wh1;		nops;		nopx;		vmax_lt.bf16 x3, r16, x9, x3;		nopv
	nopa;		vldb.4x64.lo wl1, wh1;		nops;		nopx;		vshuffle bmlh0, x7, x2, r3;		vmac.f dm2, dm2, x0, x4, r1
	vlda.2d x0, [p2], d3;		nopb;		vconv.bf16.fp32 x2, cml1;		nopx;		vshuffle x4, x7, x2, r2;		nopv
	vlda.2d bmll0, [p0], d0;		vldb.2d x6, [p5], d5;		vfloor.s32.bf16 x7, wl3, s0;		nopx;		vmov wl4, wh6;		nopv
	nopa;		vldb.4x64.lo wl2, wl1;		nops;		nopx;		nopm;		vmsc.f dm1, dm1, x2, x11, r1
	nopa;		vldb.4x64.hi wh2, wl1;		vconv.bf16.fp32 x5, cml3;		nopx;		vadd.32 x4, x7, x10;		vmac.f dm3, dm0, x4, x5, r1
	vlda.2d bmlh0, [p0], d0;		vldb.2d x5, [p4], d2;		vfloor.s32.bf16 x7, wh3, s0;		nopx;		vshuffle x3, x6, x0, r2;		nopv
	nopa;		vldb.4x64.lo wl0, wh4;		vst.2d.conv.bf16.fp32 cml2, [p3], d4;		nopx;		vshuffle bmll2, x6, x0, r3;		nopv
	nopa;		vldb.4x64.hi wh0, wh4;		nops;		nopx;		vmin_ge.bf16 x5, r16, x8, x5;		nopv
	nopa;		vldb.4x64.hi wh3, wl4;		vst.2d.conv.bf16.fp32 cml3, [p1], d1;		nopx;		vadd.32 x7, x7, x10;		vmul.f dm0, x0, x2, r1
	nopa;		vldb.4x64.lo wl3, wl4;		vconv.bf16.fp32 x4, cml1;		nopx;		nopm;		vmac.f dm1, dm0, x6, x11, r1
	nopa;		vldb.4x64.hi wh2, wh7;		nops;		nopx;		vmax_lt.bf16 x6, r16, x9, x5;		nopv
	nopa;		vldb.4x64.lo wl2, wh7;		nops;		nopx;		vshuffle bmlh2, x2, x1, r3;		vmac.f dm0, dm0, x0, x4, r1
bgelu_last:
	vlda.2d x0, [p2], d3;		nopb;		vconv.bf16.fp32 x1, cml3;		nopx;		vshuffle x2, x2, x1, r2;		nopv
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
