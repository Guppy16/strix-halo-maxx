	.text
	.section	.text.sigk,"ax",@progbits
	.globl	sigk
	.p2align	4
	.type	sigk,@function
sigk:
	movxm	r16, #52428
	mov	r1, p1
	vbcst.32	x0, r1
	mov	r1, p2
	vbcst.32	x1, r1
	movxm	r1, #1536
	vbcst.32	x2, r1
	vsel.32	x3, x0, x1, r16
	nop
	nop
	vadd.32	x10, x3, x2
	movxm	r1, #16638
	vbcst.16	x8, r1
	movxm	r1, #-16064
	vbcst.16	x9, r1
	mova	r1, #828
	mova	r2, #24
	mova	r3, #5
	mov	s0, #7
	mov	p3, p0
	padda	[p3], #256
	padda	[p0], #64
	nop
	vmov	x4, x10
	vmov	x7, x10
	movxm	ls, #sigk_start
	movxm	le, #sigk_last
	add.nc	lc, r0, #2
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 12, U 2, stages 4, ops 22, kernel-only (+2 trips)
	.p2align	4
sigk_start:
	vlda x5, [p0], #-64;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		vfloor.s32.bf16 x4, wh0, s0;		nopx;		vadd.32 x7, x4, x10;		nopv
	nopa;		nopb;		vst.conv.bf16.fp32 cml0, [p3], #-64;		nopx;		vshuffle x11, x2, x5, r2;		nopv
	nopa;		vldb.4x64.lo wl3, wl7;		nops;		nopx;		vadd.32 x4, x4, x10;		nopv
	nopa;		vldb.4x64.lo wl5, wh7;		nops;		nopx;		vshuffle bmll0, x2, x5, r3;		nopv
	nopa;		vldb.4x64.hi wh3, wl7;		nops;		nopx;		vshuffle x2, x6, x3, r2;		nopv
	nopa;		vldb.4x64.hi wh5, wh7;		nops;		nopx;		vmov wl2, wh11;		nopv
	nopa;		vldb.4x64.hi wh2, wh4;		nops;		nopx;		vmin_ge.bf16 x5, r16, x8, x5;		nopv
	nopa;		vldb.4x64.lo wl2, wh4;		nops;		nopx;		vshuffle bmlh0, x6, x3, r3;		vmac.f dm0, dm0, x2, x1, r1
	nopa;		vldb.4x64.lo wl6, wl4;		nops;		nopx;		vmax_lt.bf16 x1, r16, x9, x5;		nopv
	nopa;		vldb.4x64.hi wh6, wl4;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		vfloor.s32.bf16 x4, wl1, s0;		nopx;		nopm;		nopv
	vlda x5, [p0], #-64;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		vfloor.s32.bf16 x4, wh1, s0;		nopx;		vadd.32 x7, x4, x10;		nopv
	nopa;		nopb;		vst.conv.bf16.fp32 cml0, [p3], #-64;		nopx;		vshuffle x11, x3, x5, r2;		nopv
	nopa;		vldb.4x64.lo wl2, wl7;		nops;		nopx;		vadd.32 x4, x4, x10;		nopv
	nopa;		vldb.4x64.lo wl5, wh7;		nops;		nopx;		vshuffle bmll0, x3, x5, r3;		nopv
	nopa;		vldb.4x64.hi wh2, wl7;		nops;		nopx;		vshuffle x3, x6, x2, r2;		nopv
	nopa;		vldb.4x64.hi wh5, wh7;		nops;		nopx;		vmov wl3, wh11;		nopv
	nopa;		vldb.4x64.hi wh3, wh4;		nops;		nopx;		vmin_ge.bf16 x5, r16, x8, x5;		nopv
	nopa;		vldb.4x64.lo wl3, wh4;		nops;		nopx;		vshuffle bmlh0, x6, x2, r3;		vmac.f dm0, dm0, x3, x0, r1
	nopa;		vldb.4x64.lo wl6, wl4;		nops;		nopx;		vmax_lt.bf16 x0, r16, x9, x5;		nopv
	nopa;		vldb.4x64.hi wh6, wl4;		nops;		nopx;		nopm;		nopv
sigk_last:
	nopa;		nopb;		vfloor.s32.bf16 x4, wl0, s0;		nopx;		nopm;		nopv
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_sigk:
	.size	sigk, .Lfunc_end_sigk-sigk
