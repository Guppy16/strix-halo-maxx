	.text
	.section	.text.probe0,"ax",@progbits
	.globl	probe0                       // -- Begin function probe0
	.p2align	4
	.type	probe0,@function
probe0:                              // @probe0
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	r16, #52428;		nopv	
	nops	;		mov	r1, p2
	vbcst.32	 x0, r1
	mov	r1, p3
	vbcst.32	 x2, r1
	mova	r1, #512;		vsel.32	 x0, x0, x2, r16
	vbcst.32	 x2, r1
	movxm	r1, #16511
	vadd.32	 x0, x0, x2
	vldb	 x8, [p0], #64;		vbcst.16	 x4, r1
	movxm	r1, #-16256
	vmov	bmll4, x4
	vbcst.16	 x4, r1
	vmov	bmhl1, x0
	vmov	bmlh4, x4
	vmov	x0, bmll4
	vmin_ge.bf16	 x0, r16, x8, x0
	vmov	x2, bmlh4
	vmax_lt.bf16	 x0, r16, x0, x2
	mov	s0, #7
	vfloor.s32.bf16	x10, wh0, s0
	vmov	x6, bmhl1
	vadd.32	 x1, x10, x6
	vldb	 x10, [p0], #64;		vfloor.s32.bf16	x2, wl0, s0
	vldb.4x64.lo	 wl3, wl1
	vldb.4x64.hi	 wh3, wl1;		add.nc	lc, r0, #-3
	vldb.4x64.lo	 wl11, wh1;		vadd.32	 x2, x2, x6
	vldb.4x64.hi	 wh11, wh1;		movxm	ls, #.LBB14_1_probe0
	vldb.4x64.lo	 wl1, wl2;		movxm	le, #.L_LEnd17_probe0
	vldb.4x64.hi	 wh1, wl2;		movxm	r1, #-16512
	vldb.4x64.lo	 wl7, wh2;		vmov	x6, bmlh4
	vldb.4x64.hi	 wh7, wh2;		vmov	x2, bmll4
	vmin_ge.bf16	 x2, r16, x10, x2
	vmax_lt.bf16	 x5, r16, x2, x6
	mova	r2, #24;		vmov	x6, bmhl1
	mova	r3, #5;		vfloor.s32.bf16	x2, wl5, s0;		vshuffle	x9, x3, x11, r2
	vshuffle	bmlh1, x3, x11, r3
	vfloor.s32.bf16	x3, wh5, s0;		vadd.32	 x2, x2, x6
	vldb	 x11, [p0], #64;		vshuffle	x11, x1, x7, r2
	vldb.4x64.lo	 wl0, wl2;		vadd.32	 x3, x3, x6
	vldb.4x64.hi	 wh0, wl2;		vmov	wl9, wh11
	mova	r1, #828;		vldb.4x64.lo	 wl7, wl3;		vbcst.16	 x4, r1
	vldb.4x64.hi	 wh7, wl3;		vmov	bmll2, x4;		vmac.f	dm0, dm1, x9, x0, r1
	vldb.4x64.lo	 wl9, wh3;		vshuffle	bmll1, x1, x7, r3
	vldb.4x64.hi	 wh9, wh3;		vmov	x6, bmlh4
	vldb.4x64.lo	 wl1, wh2;		vmov	x2, bmll4
	vldb.4x64.hi	 wh1, wh2;		vmov	bmlh1, bmll2
	vmin_ge.bf16	 x2, r16, x11, x2
	vconv.bf16.fp32	 x4, cml0;		vmax_lt.bf16	 x3, r16, x2, x6
	vmov	x2, bmll2
	vmov	x6, x4
	vshuffle	x2, x7, x9, r2
	vfloor.s32.bf16	x4, wl3, s0;		vshuffle	bmhh1, x7, x9, r3;		vmac.f	dm0, dm0, x4, x2, r1
	vfloor.s32.bf16	x9, wh3, s0;		vmov	x6, bmhl1
	vadd.32	 x7, x4, x6;		vmul.f	dm2, x8, x6, r1
	vadd.32	 x9, x9, x6
	vmov	x6, bmlh1
.LBB14_1_probe0:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopx	;		vshuffle	x4, x0, x1, r2
	vldb	 x11, [p0], #64;		vmov	bmlh1, x11
	vldb.4x64.hi	 wh2, wh9;		vshuffle	bmll3, x0, x1, r3
	vldb.4x64.lo	 wl2, wh9;		vmov	wl2, wh4
	vldb.4x64.hi	 wh9, wl9;		vmov	bmlh3, bmhh1
	vconv.bf16.fp32	 x5, cml0;		vldb.4x64.lo	 wl9, wl9;		vmov	x4, bmll4;		vmac.f	dm3, dm3, x2, x5, r1
	vmov	x8, x10
	vldb.4x64.hi	 wh1, wh7;		vmov	x5, x3;		vmac.f	dm2, dm2, x8, x5, r1
	vldb.4x64.lo	 wl1, wh7;		vmov	x3, bmlh4
	vldb.4x64.hi	 wh0, wl7;		vmin_ge.bf16	 x10, r16, x11, x4
	vldb.4x64.lo	 wl0, wl7;		vmax_lt.bf16	 x3, r16, x10, x3
	vconv.bf16.fp32	 x7, cml3;		vmov	x10, bmhl1
	vfloor.s32.bf16	x4, wh3, s0;		vshuffle	bmhh1, x9, x2, r3
	vfloor.s32.bf16	x7, wl3, s0;		vshuffle	x2, x9, x2, r2;		vmac.f	dm0, dm3, x7, x6, r1
	vadd.32	 x9, x4, x10;		vmul.f	dm2, x8, x7, r1
	vadd.32	 x7, x7, x10
.L_LEnd17_probe0:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml2, [p1], #64;		nopx	;		vmov	x10, bmlh1;		nopv	
// %bb.2:
	nopa	;		nopxm	
	nop	
	vconv.bf16.fp32	 x4, cml0
	nop	
	vmac.f	dm0, dm2, x8, x4, r1
	nop	
	vshuffle	x4, x0, x1, r2
	vmov	wl2, wh4
	nop	
	vshuffle	bmll0, x0, x1, r3;		vmac.f	dm0, dm0, x2, x5, r1
	vst.conv.bf16.fp32	 cml0, [p1], #64;		vmov	bmlh0, bmhh1
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm0, x0, x6, r1
	vldb.4x64.lo	 wl4, wh9
	vldb.4x64.hi	 wh4, wh9
	vldb.4x64.lo	 wl2, wl9
	vldb.4x64.hi	 wh2, wl9
	vldb.4x64.lo	 wl6, wh7;		vmul.f	dm1, x10, x0, r1
	vldb.4x64.hi	 wh6, wh7;		vconv.bf16.fp32	 x0, cml0
	vldb.4x64.lo	 wl0, wl7
	vldb.4x64.hi	 wh0, wl7;		vmac.f	dm0, dm1, x10, x0, r1
	nop	
	nop	
	vmov	x5, x6
	nop	
	vshuffle	bmlh0, x2, x4, r3
	vst.conv.bf16.fp32	 cml0, [p1], #64;		vshuffle	x2, x2, x4, r2
	vshuffle	x8, x0, x6, r2
	vmov	wl2, wh8
	nop	
	vmac.f	dm0, dm0, x2, x3, r1
	vshuffle	bmll0, x0, x6, r3
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm0, x0, x5, r1
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm1, x11, x0, r1
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm1, x11, x0, r1
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml0, [p1], #64 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end14_probe0:
	.size	probe0, .Lfunc_end14_probe0-probe0
	.section	.text.probe1,"ax",@progbits
	.globl	probe1                         // -- Begin function probe1
	.p2align	4
	.type	probe1,@function
probe1:                                // @probe1
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	r16, #52428
	mov	r1, p2
	vbcst.32	 x0, r1
	mov	r1, p3
	vbcst.32	 x2, r1
	mova	r1, #512;		vsel.32	 x6, x0, x2, r16
	vldb	 x8, [p0], #64;		vbcst.32	 x8, r1
	movxm	r1, #16511
	vadd.32	 x6, x6, x8
	vbcst.16	 x0, r1
	movxm	r1, #-16256
	vbcst.16	 x4, r1
	vmov	bmll3, x0
	vmin_ge.bf16	 x10, r16, x8, x0
	vmax_lt.bf16	 x0, r16, x10, x4
	mov	s0, #7
	vfloor.s32.bf16	x10, wh0, s0
	nop	
	vadd.32	 x10, x10, x6
	nop	
	vldb.4x64.lo	 wl10, wh10
	vldb.4x64.hi	 wh10, wh10
	nop	
	nop	
	nop	
	vldb.4x64.lo	 wl7, wl10
	vldb.4x64.hi	 wh7, wl10
	vfloor.s32.bf16	x10, wl0, s0
	vmov	bmll2, x10
	vadd.32	 x3, x10, x6
	vldb	 x10, [p0], #64
	vldb.4x64.lo	 wl11, wl3
	vldb.4x64.hi	 wh11, wl3
	vldb.4x64.lo	 wl9, wh3
	vldb.4x64.hi	 wh9, wh3
	nop	
	nop	
	nop	
	vmov	bmll1, x0
	vmov	x0, bmll3
	mova	r2, #24;		vmin_ge.bf16	 x1, r16, x10, x0
	vshuffle	x0, x11, x9, r2
	paddxm	 [sp], #64
	vst	 x0, [sp, #-64]                 // 64-byte Folded Spill
	vlda	 x0, [sp, #-64]                 // 64-byte Folded Reload
	movxm	r1, #-16512
	vbcst.16	 x2, r1
	vmax_lt.bf16	 x1, r16, x1, x4
	mova	r3, #5;		vmov	x0, bmll2
	vshuffle	bmlh0, x7, x0, r3
	vshuffle	x7, x7, x0, r2
	vfloor.s32.bf16	x5, wh1, s0;		vmov	wl7, wh0
	vmov	x0, bmll1
	mova	r1, #828;		vadd.32	 x5, x5, x6
	vmac.f	dm4, dm0, x7, x0, r1
	vldb.4x64.lo	 wl3, wl5;		vshuffle	bmll0, x11, x9, r3
	vldb.4x64.hi	 wh3, wl5;		add.nc	lc, r0, #-2
	vfloor.s32.bf16	x7, wl1, s0;		movxm	ls, #.LBB11_1_probe1
	vldb.4x64.lo	 wl5, wh5;		movxm	le, #.L_LEnd14_probe1
	vldb.4x64.hi	 wh5, wh5;		vadd.32	 x7, x7, x6
.LBB11_1_probe1:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		vconv.bf16.fp32	 x9, cml4;		nopx	
	nop	
	vmac.f	dm2, dm4, x9, x2, r1
	nop	
	vldb	 x10, [p0], #64
	nop	
	nop	
	vmul.f	dm0, x8, x9, r1
	vconv.bf16.fp32	 x9, cml2;		vmov	x0, bmll3
	vmov	x8, x10
	vldb.4x64.hi	 wh11, wh7;		vmov	x9, x1;		vmac.f	dm0, dm0, x8, x9, r1
	vldb.4x64.lo	 wl11, wh7;		vmin_ge.bf16	 x0, r16, x10, x0
	vldb.4x64.hi	 wh7, wl7;		vmax_lt.bf16	 x1, r16, x0, x4
	vldb.4x64.lo	 wl7, wl7
	vfloor.s32.bf16	x0, wh1, s0
	nop	
	vadd.32	 x0, x0, x6
	nop	
	vldb.4x64.hi	 wh5, wh0;		vmov	bmll2, x2
	vldb.4x64.lo	 wl5, wh0;		vshuffle	x2, x3, x5, r2
	vldb.4x64.hi	 wh3, wl0;		vshuffle	x0, x7, x11, r2
	vldb.4x64.lo	 wl3, wl0;		vmov	wl2, wh0
	nop	
	vshuffle	bmlh1, x3, x5, r3;		vmac.f	dm4, dm1, x2, x9, r1
	vshuffle	bmll1, x7, x11, r3
	vfloor.s32.bf16	x0, wl1, s0
	nop	
	vadd.32	 x7, x0, x6
.L_LEnd14_probe1:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml0, [p1], #64;		nopx	;		vmov	x2, bmll2;		nopv	
// %bb.2:
	vconv.bf16.fp32	 x0, cml4;		nopx	
	nop	
	vmac.f	dm0, dm4, x0, x2, r1
	nop	
	vldb.4x64.lo	 wl0, wl7
	vldb.4x64.hi	 wh0, wl7
	vldb.4x64.lo	 wl2, wh7
	vldb.4x64.hi	 wh2, wh7;		vmul.f	dm1, x8, x0, r1
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm1, x8, x0, r1
	nop	
	vmov	x8, x2
	vshuffle	x6, x3, x5, r2
	vshuffle	x4, x0, x2, r2
	vmov	wl6, wh4
	nop	
	vshuffle	bmlh0, x3, x5, r3;		vmac.f	dm0, dm0, x6, x1, r1
	vst.conv.bf16.fp32	 cml0, [p1], #64;		vshuffle	bmll0, x0, x2, r3
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm0, x0, x8, r1
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm1, x10, x0, r1
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm1, x10, x0, r1
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml0, [p1], #64;		paddxm	 [sp], #-64 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end11_probe1:
	.size	probe1, .Lfunc_end11_probe1-probe1
	.section	.text.probe2,"ax",@progbits
	.globl	probe2                         // -- Begin function probe2
	.p2align	4
	.type	probe2,@function
probe2:                                // @probe2
// %bb.0:                               // %entry
	movxm	r16, #52428
	movxm	r2, #-16512
	movxm	ls, #.LBB12_1_probe2
	mova	r1, #2;		movxm	le, #.L_LEnd15_probe2
	ltu	 r27, r1, r0;		mov	s0, #7
	sel.nez	 r0, r0, r1, r27;		mov	r1, p2
	vbcst.32	 x0, r1
	mov	r1, p3
	vbcst.32	 x2, r1
	mova	r1, #512;		vsel.32	 x6, x0, x2, r16
	vbcst.32	 x8, r1
	movxm	r1, #16511
	vbcst.16	 x0, r1
	movxm	r1, #-16256
	vmov	bmll2, x0
	vbcst.16	 x0, r1
	vmov	bmll3, x0
	mova	r1, #-1;		add	r0, r0, #-1;		vbcst.16	 x0, r2
	lshl	 r1, r0, r1;		vmov	bmll4, x0
	add	r3, r1, #1;		vadd.32	 x0, x6, x8
	mova	r0, #24;		add.nc	lc, r3, #0
	mova	r1, #5;		movx	r2, #828;		vmov	bmlh2, x0
.LBB12_1_probe2:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	vlda	 x8, [p0], #64;		nopxm	
	nop	
	nop	
	nop	
	nop	
	vmov	x0, bmll2
	vmov	x2, bmll3
	vmin_ge.bf16	 x10, r16, x8, x0
	vmax_lt.bf16	 x10, r16, x10, x2
	nop	
	vlda	 x1, [p0], #64;		vfloor.s32.bf16	x7, wh10, s0
	vfloor.s32.bf16	x3, wl10, s0;		vmov	x6, bmlh2
	vadd.32	 x11, x7, x6
	vadd.32	 x3, x3, x6
	vldb.4x64.lo	 wl9, wl11
	vldb.4x64.lo	 wl5, wl3
	vldb.4x64.hi	 wh5, wl3
	vldb.4x64.lo	 wl7, wh3;		vmin_ge.bf16	 x3, r16, x1, x0
	vldb.4x64.hi	 wh7, wh3;		vmax_lt.bf16	 x3, r16, x3, x2
	vldb.4x64.hi	 wh9, wl11
	vfloor.s32.bf16	x4, wl3, s0
	vldb.4x64.lo	 wl2, wh11
	vfloor.s32.bf16	x0, wh3, s0;		vadd.32	 x4, x4, x6
	vldb.4x64.hi	 wh2, wh11
	vldb.4x64.lo	 wl6, wl4;		vadd.32	 x0, x0, x6
	vldb.4x64.hi	 wh6, wl4
	vldb.4x64.lo	 wl11, wl0
	vldb.4x64.hi	 wh11, wl0
	vldb.4x64.lo	 wl0, wh0
	vldb.4x64.hi	 wh0, wh0
	vldb.4x64.lo	 wl4, wh4
	vldb.4x64.hi	 wh4, wh4;		vshuffle	bmll0, x5, x7, r1
	vshuffle	x5, x5, x7, r0
	vshuffle	bmlh0, x9, x2, r1
	vshuffle	x2, x9, x2, r0
	vmov	wl2, wh5
	vshuffle	x2, x11, x0, r0
	vshuffle	bmlh1, x11, x0, r1;		vmac.f	dm0, dm0, x2, x10, r2
	vshuffle	x0, x6, x4, r0
	vmov	wl2, wh0
	nop	
	vmac.f	dm1, dm1, x2, x3, r2
	vshuffle	bmll1, x6, x4, r1
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0;		vmov	x4, bmll4
	vconv.bf16.fp32	 x2, cml1
	vmac.f	dm0, dm0, x0, x4, r2
	vmac.f	dm1, dm1, x2, x4, r2
	nop	
	nop	
	nop	
	vmul.f	dm0, x8, x0, r2
	vconv.bf16.fp32	 x4, cml0;		vmul.f	dm1, x1, x2, r2
	vconv.bf16.fp32	 x0, cml1
	vmac.f	dm0, dm0, x8, x4, r2
	vmac.f	dm1, dm1, x1, x0, r2
	nop	
	nop	
	nop	
	nop	
	vst.conv.bf16.fp32	 cml0, [p1], #64
.L_LEnd15_probe2:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml1, [p1], #64;		nopxm	;		nopv	
// %bb.2:                               // %for.cond.cleanup
	nopa	;		ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end12_probe2:
	.size	probe2, .Lfunc_end12_probe2-probe2
	.section	.text.probe3,"ax",@progbits
	.globl	probe3                       // -- Begin function probe3
	.p2align	4
	.type	probe3,@function
probe3:                              // @probe3
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	r16, #52428;		nopv	
	nops	;		mov	r1, p2
	vbcst.32	 x0, r1
	mov	r1, p3
	vbcst.32	 x2, r1
	mova	r1, #512;		vsel.32	 x0, x0, x2, r16
	vbcst.32	 x2, r1
	movxm	r1, #16511
	vadd.32	 x0, x0, x2
	vldb	 x8, [p0], #64;		vbcst.16	 x4, r1
	movxm	r1, #-16256
	vmov	bmll4, x4
	vbcst.16	 x4, r1
	vmov	bmhl1, x0
	vmov	bmlh4, x4
	vmov	x0, bmll4
	vmin_ge.bf16	 x0, r16, x8, x0
	vmov	x2, bmlh4
	vmax_lt.bf16	 x0, r16, x0, x2
	mov	s0, #7
	vfloor.s32.bf16	x10, wh0, s0
	vmov	x6, bmhl1
	vadd.32	 x1, x10, x6
	vldb	 x10, [p0], #64;		vfloor.s32.bf16	x2, wl0, s0
	vldb.4x64.lo	 wl3, wl1
	vldb.4x64.hi	 wh3, wl1;		add.nc	lc, r0, #-3
	vldb.4x64.lo	 wl11, wh1;		vadd.32	 x2, x2, x6
	vldb.4x64.hi	 wh11, wh1;		movxm	ls, #.LBB15_1_probe3
	vldb.4x64.lo	 wl1, wl2;		movxm	le, #.L_LEnd18_probe3
	vldb.4x64.hi	 wh1, wl2;		movxm	r1, #-16512
	vldb.4x64.lo	 wl7, wh2;		vmov	x6, bmlh4
	vldb.4x64.hi	 wh7, wh2;		vmov	x2, bmll4
	vmin_ge.bf16	 x2, r16, x10, x2
	vmax_lt.bf16	 x5, r16, x2, x6
	mova	r2, #24;		vmov	x6, bmhl1
	mova	r3, #5;		vfloor.s32.bf16	x2, wl5, s0;		vshuffle	x9, x3, x11, r2
	vshuffle	bmlh1, x3, x11, r3
	vfloor.s32.bf16	x3, wh5, s0;		vadd.32	 x2, x2, x6
	vldb	 x11, [p0], #64;		vshuffle	x11, x1, x7, r2
	vldb.4x64.lo	 wl0, wl2;		vadd.32	 x3, x3, x6
	vldb.4x64.hi	 wh0, wl2;		vmov	wl9, wh11
	mova	r1, #828;		vldb.4x64.lo	 wl7, wl3;		vbcst.16	 x4, r1
	vldb.4x64.hi	 wh7, wl3;		vmov	bmll2, x4;		vmac.f	dm0, dm1, x9, x0, r1
	vldb.4x64.lo	 wl9, wh3;		vshuffle	bmll1, x1, x7, r3
	vldb.4x64.hi	 wh9, wh3;		vmov	x6, bmlh4
	vldb.4x64.lo	 wl1, wh2;		vmov	x2, bmll4
	vldb.4x64.hi	 wh1, wh2;		vmov	bmlh1, bmll2
	vmin_ge.bf16	 x2, r16, x11, x2
	vconv.bf16.fp32	 x4, cml0;		vmax_lt.bf16	 x3, r16, x2, x6
	vmov	x2, bmll2
	vmov	x6, x4
	vshuffle	x2, x7, x9, r2
	vfloor.s32.bf16	x4, wl3, s0;		vshuffle	bmhh1, x7, x9, r3;		vmac.f	dm0, dm0, x4, x2, r1
	vfloor.s32.bf16	x9, wh3, s0;		vmov	x6, bmhl1
	vadd.32	 x7, x4, x6;		vmul.f	dm2, x8, x6, r1
	vadd.32	 x9, x9, x6
	vmov	x6, bmlh1
.LBB15_1_probe3:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopx	;		vshuffle	x4, x0, x1, r2
	vldb	 x11, [p0], #64;		vmov	bmlh1, x11
	vldb.4x64.hi	 wh2, wh9;		vshuffle	bmll3, x0, x1, r3
	vldb.4x64.lo	 wl2, wh9;		vmov	wl2, wh4
	vldb.4x64.hi	 wh9, wl9;		vmov	bmlh3, bmhh1
	vconv.bf16.fp32	 x5, cml0;		vldb.4x64.lo	 wl9, wl9;		vmov	x4, bmll4;		vmac.f	dm3, dm3, x2, x5, r1
	vmov	x8, x10
	vldb.4x64.hi	 wh1, wh7;		vmov	x5, x3;		vmac.f	dm2, dm2, x8, x5, r1
	vldb.4x64.lo	 wl1, wh7;		vmov	x3, bmlh4
	vldb.4x64.hi	 wh0, wl7;		vmin_ge.bf16	 x10, r16, x11, x4
	vldb.4x64.lo	 wl0, wl7;		vmax_lt.bf16	 x3, r16, x10, x3
	vconv.bf16.fp32	 x7, cml3;		vmov	x10, bmhl1
	vfloor.s32.bf16	x4, wh3, s0;		vshuffle	bmhh1, x9, x2, r3
	vfloor.s32.bf16	x7, wl3, s0;		vshuffle	x2, x9, x2, r2;		vmac.f	dm0, dm3, x7, x6, r1
	vadd.32	 x9, x4, x10;		vmul.f	dm2, x8, x7, r1
	vadd.32	 x7, x7, x10
.L_LEnd18_probe3:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml2, [p1], #64;		nopx	;		vmov	x10, bmlh1;		nopv	
// %bb.2:
	nopa	;		nopxm	
	nop	
	vconv.bf16.fp32	 x4, cml0
	nop	
	vmac.f	dm0, dm2, x8, x4, r1
	nop	
	vshuffle	x4, x0, x1, r2
	vmov	wl2, wh4
	nop	
	vshuffle	bmll0, x0, x1, r3;		vmac.f	dm0, dm0, x2, x5, r1
	vst.conv.bf16.fp32	 cml0, [p1], #64;		vmov	bmlh0, bmhh1
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm0, x0, x6, r1
	vldb.4x64.lo	 wl4, wh9
	vldb.4x64.hi	 wh4, wh9
	vldb.4x64.lo	 wl2, wl9
	vldb.4x64.hi	 wh2, wl9
	vldb.4x64.lo	 wl6, wh7;		vmul.f	dm1, x10, x0, r1
	vldb.4x64.hi	 wh6, wh7;		vconv.bf16.fp32	 x0, cml0
	vldb.4x64.lo	 wl0, wl7
	vldb.4x64.hi	 wh0, wl7;		vmac.f	dm0, dm1, x10, x0, r1
	nop	
	nop	
	vmov	x5, x6
	nop	
	vshuffle	bmlh0, x2, x4, r3
	vst.conv.bf16.fp32	 cml0, [p1], #64;		vshuffle	x2, x2, x4, r2
	vshuffle	x8, x0, x6, r2
	vmov	wl2, wh8
	nop	
	vmac.f	dm0, dm0, x2, x3, r1
	vshuffle	bmll0, x0, x6, r3
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm0, x0, x5, r1
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm1, x11, x0, r1
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm1, x11, x0, r1
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml0, [p1], #64 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end15_probe3:
	.size	probe3, .Lfunc_end15_probe3-probe3
	.section	.text.probe4,"ax",@progbits
	.globl	probe4                       // -- Begin function probe4
	.p2align	4
	.type	probe4,@function
probe4:                              // @probe4
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	r16, #52428;		nopv	
	nops	;		mov	r1, p2
	vbcst.32	 x0, r1
	mov	r1, p3
	vbcst.32	 x2, r1
	mova	r1, #512;		vsel.32	 x0, x0, x2, r16
	vbcst.32	 x2, r1
	movxm	r1, #16511
	vadd.32	 x0, x0, x2
	vldb	 x8, [p0], #64;		vbcst.16	 x4, r1
	movxm	r1, #-16256
	vmov	bmll4, x4
	vbcst.16	 x4, r1
	vmov	bmhl1, x0
	vmov	bmlh4, x4
	vmov	x0, bmll4
	vmin_ge.bf16	 x0, r16, x8, x0
	vmov	x2, bmlh4
	vmax_lt.bf16	 x0, r16, x0, x2
	mov	s0, #7
	vfloor.s32.bf16	x10, wh0, s0
	vmov	x6, bmhl1
	vadd.32	 x1, x10, x6
	vldb	 x10, [p0], #64;		vfloor.s32.bf16	x2, wl0, s0
	vldb.4x64.lo	 wl3, wl1
	vldb.4x64.hi	 wh3, wl1;		add.nc	lc, r0, #-3
	vldb.4x64.lo	 wl11, wh1;		vadd.32	 x2, x2, x6
	vldb.4x64.hi	 wh11, wh1;		movxm	ls, #.LBB16_1_probe4
	vldb.4x64.lo	 wl1, wl2;		movxm	le, #.L_LEnd19_probe4
	vldb.4x64.hi	 wh1, wl2;		movxm	r1, #-16512
	vldb.4x64.lo	 wl7, wh2;		vmov	x6, bmlh4
	vldb.4x64.hi	 wh7, wh2;		vmov	x2, bmll4
	vmin_ge.bf16	 x2, r16, x10, x2
	vmax_lt.bf16	 x5, r16, x2, x6
	mova	r2, #24;		vmov	x6, bmhl1
	mova	r3, #5;		vfloor.s32.bf16	x2, wl5, s0;		vshuffle	x9, x3, x11, r2
	vshuffle	bmlh1, x3, x11, r3
	vfloor.s32.bf16	x3, wh5, s0;		vadd.32	 x2, x2, x6
	vldb	 x11, [p0], #64;		vshuffle	x11, x1, x7, r2
	vldb.4x64.lo	 wl0, wl2;		vadd.32	 x3, x3, x6
	vldb.4x64.hi	 wh0, wl2;		vmov	wl9, wh11
	mova	r1, #828;		vldb.4x64.lo	 wl7, wl3;		vbcst.16	 x4, r1
	vldb.4x64.hi	 wh7, wl3;		vmov	bmll2, x4;		vmac.f	dm0, dm1, x9, x0, r1
	vldb.4x64.lo	 wl9, wh3;		vshuffle	bmll1, x1, x7, r3
	vldb.4x64.hi	 wh9, wh3;		vmov	x6, bmlh4
	vldb.4x64.lo	 wl1, wh2;		vmov	x2, bmll4
	vldb.4x64.hi	 wh1, wh2;		vmov	bmlh1, bmll2
	vmin_ge.bf16	 x2, r16, x11, x2
	vconv.bf16.fp32	 x4, cml0;		vmax_lt.bf16	 x3, r16, x2, x6
	vmov	x2, bmll2
	vmov	x6, x4
	vshuffle	x2, x7, x9, r2
	vfloor.s32.bf16	x4, wl3, s0;		vshuffle	bmhh1, x7, x9, r3;		vmac.f	dm0, dm0, x4, x2, r1
	vfloor.s32.bf16	x9, wh3, s0;		vmov	x6, bmhl1
	vadd.32	 x7, x4, x6;		vmul.f	dm2, x8, x6, r1
	vadd.32	 x9, x9, x6
	vmov	x6, bmlh1
.LBB16_1_probe4:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopx	;		vshuffle	x4, x0, x1, r2
	vldb	 x11, [p0], #64;		vmov	bmlh1, x11
	vldb.4x64.hi	 wh2, wh9;		vshuffle	bmll3, x0, x1, r3
	vldb.4x64.lo	 wl2, wh9;		vmov	wl2, wh4
	vldb.4x64.hi	 wh9, wl9;		vmov	bmlh3, bmhh1
	vconv.bf16.fp32	 x5, cml0;		vldb.4x64.lo	 wl9, wl9;		vmov	x4, bmll4;		vmac.f	dm3, dm3, x2, x5, r1
	vmov	x8, x10
	vldb.4x64.hi	 wh1, wh7;		vmov	x5, x3;		vmac.f	dm2, dm2, x8, x5, r1
	vldb.4x64.lo	 wl1, wh7;		vmov	x3, bmlh4
	vldb.4x64.hi	 wh0, wl7;		vmin_ge.bf16	 x10, r16, x11, x4
	vldb.4x64.lo	 wl0, wl7;		vmax_lt.bf16	 x3, r16, x10, x3
	vconv.bf16.fp32	 x7, cml3;		vmov	x10, bmhl1
	vfloor.s32.bf16	x4, wh3, s0;		vshuffle	bmhh1, x9, x2, r3
	vfloor.s32.bf16	x7, wl3, s0;		vshuffle	x2, x9, x2, r2;		vmac.f	dm0, dm3, x7, x6, r1
	vadd.32	 x9, x4, x10;		vmul.f	dm2, x8, x7, r1
	vadd.32	 x7, x7, x10
.L_LEnd19_probe4:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml2, [p1], #64;		nopx	;		vmov	x10, bmlh1;		nopv	
// %bb.2:
	nopa	;		nopxm	
	nop	
	vconv.bf16.fp32	 x4, cml0
	nop	
	vmac.f	dm0, dm2, x8, x4, r1
	nop	
	vshuffle	x4, x0, x1, r2
	vmov	wl2, wh4
	nop	
	vshuffle	bmll0, x0, x1, r3;		vmac.f	dm0, dm0, x2, x5, r1
	vst.conv.bf16.fp32	 cml0, [p1], #64;		vmov	bmlh0, bmhh1
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm0, x0, x6, r1
	vldb.4x64.lo	 wl4, wh9
	vldb.4x64.hi	 wh4, wh9
	vldb.4x64.lo	 wl2, wl9
	vldb.4x64.hi	 wh2, wl9
	vldb.4x64.lo	 wl6, wh7;		vmul.f	dm1, x10, x0, r1
	vldb.4x64.hi	 wh6, wh7;		vconv.bf16.fp32	 x0, cml0
	vldb.4x64.lo	 wl0, wl7
	vldb.4x64.hi	 wh0, wl7;		vmac.f	dm0, dm1, x10, x0, r1
	nop	
	nop	
	vmov	x5, x6
	nop	
	vshuffle	bmlh0, x2, x4, r3
	vst.conv.bf16.fp32	 cml0, [p1], #64;		vshuffle	x2, x2, x4, r2
	vshuffle	x8, x0, x6, r2
	vmov	wl2, wh8
	nop	
	vmac.f	dm0, dm0, x2, x3, r1
	vshuffle	bmll0, x0, x6, r3
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm0, x0, x5, r1
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm1, x11, x0, r1
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm1, x11, x0, r1
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml0, [p1], #64 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end16_probe4:
	.size	probe4, .Lfunc_end16_probe4-probe4
	.section	.text.probe5,"ax",@progbits
	.globl	probe5
	.p2align	4
	.type	probe5,@function
probe5:
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
	movxm	ls, #probe5_start
	movxm	le, #probe5_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 10, U 1, stages 5, ops 30
	vlda.2d x5, [p0], d0
	nop
	nop
	nop
	nop
	nop
	vmin_ge.bf16 x5, r16, x8, x5
	vmax_lt.bf16 x5, r16, x9, x5
	vst.2d x5, [p2], d3
	vfloor.s32.bf16 x1, wl5, s0
	vlda.2d x5, [p0], d0;		vfloor.s32.bf16 x1, wh5, s0;		vadd.32 x5, x1, x10
	vldb.4x64.lo wl2, wl5;		vadd.32 x1, x1, x10
	vldb.4x64.hi wh2, wl5
	vldb.4x64.lo wl3, wh5
	vldb.4x64.hi wh3, wh5
	vldb.4x64.lo wl4, wl1
	vldb.4x64.hi wh4, wl1;		vmin_ge.bf16 x5, r16, x8, x5
	vldb.4x64.lo wl6, wh1;		vmax_lt.bf16 x5, r16, x9, x5
	vldb.4x64.hi wh6, wh1;		vst.2d x5, [p2], d3
	vfloor.s32.bf16 x1, wl5, s0
	vlda.2d x5, [p0], d0;		vfloor.s32.bf16 x1, wh5, s0;		vadd.32 x5, x1, x10
	vldb.4x64.lo wl2, wl5;		vadd.32 x1, x1, x10
	vldb.4x64.hi wh2, wl5;		vshuffle bmll0, x2, x3, r3
	vlda.2d x1, [p3], d4;		vldb.4x64.lo wl3, wh5;		vshuffle x3, x2, x3, r2
	vldb.4x64.hi wh3, wh5;		vshuffle bmlh0, x4, x6, r3
	vldb.4x64.lo wl4, wl1;		vshuffle x4, x4, x6, r2
	vldb.4x64.hi wh4, wl1;		vmin_ge.bf16 x5, r16, x8, x5
	vldb.4x64.lo wl6, wh1;		vmax_lt.bf16 x5, r16, x9, x5
	vldb.4x64.hi wh6, wh1;		vst.2d x5, [p2], d3;		vmov wl4, wh3
	vlda.2d x0, [p4], d2;		vfloor.s32.bf16 x1, wl5, s0;		vmac.f dm2, dm0, x4, x1, r1
	vlda.2d x5, [p0], d0;		vfloor.s32.bf16 x1, wh5, s0;		vadd.32 x5, x1, x10
	vldb.4x64.lo wl2, wl5;		vadd.32 x1, x1, x10
	vldb.4x64.hi wh2, wl5;		vshuffle bmll0, x2, x3, r3
	vlda.2d x1, [p3], d4;		vldb.4x64.lo wl3, wh5;		vshuffle x3, x2, x3, r2
	vldb.4x64.hi wh3, wh5;		vconv.bf16.fp32 x2, cml2;		vshuffle bmlh0, x4, x6, r3
	vldb.4x64.lo wl4, wl1;		vshuffle x4, x4, x6, r2;		vmac.f dm2, dm2, x2, x11, r1
	vldb.4x64.hi wh4, wl1;		vmin_ge.bf16 x5, r16, x8, x5;		vmul.f dm1, x0, x2, r1
	vldb.4x64.lo wl6, wh1;		vmax_lt.bf16 x5, r16, x9, x5
	vldb.4x64.hi wh6, wh1;		vst.2d x5, [p2], d3;		vmov wl4, wh3
	vlda.2d x0, [p4], d2;		vfloor.s32.bf16 x1, wl5, s0;		vmac.f dm2, dm0, x4, x1, r1
	.p2align	4
probe5_start:
	vlda.2d x5, [p0], d0;		nopb;		vfloor.s32.bf16 x1, wh5, s0;		nopx;		vadd.32 x5, x1, x10;		nopv
	vldb.4x64.lo wl2, wl5;		vconv.bf16.fp32 x6, cml2;		vadd.32 x1, x1, x10
	vldb.4x64.hi wh2, wl5;		vshuffle bmll0, x2, x3, r3;		vmac.f dm1, dm1, x0, x6, r1
	vlda.2d x1, [p3], d4;		vldb.4x64.lo wl3, wh5;		vshuffle x3, x2, x3, r2
	vldb.4x64.hi wh3, wh5;		vconv.bf16.fp32 x2, cml2;		vshuffle bmlh0, x4, x6, r3
	vldb.4x64.lo wl4, wl1;		vshuffle x4, x4, x6, r2;		vmac.f dm2, dm2, x2, x11, r1
	vldb.4x64.hi wh4, wl1;		vmin_ge.bf16 x5, r16, x8, x5;		vmul.f dm1, x0, x2, r1
	vldb.4x64.lo wl6, wh1;		vst.2d.conv.bf16.fp32 cml1, [p1], d1;		vmax_lt.bf16 x5, r16, x9, x5
	vldb.4x64.hi wh6, wh1;		vst.2d x5, [p2], d3;		vmov wl4, wh3
probe5_last:
	vlda.2d x0, [p4], d2;		nopb;		vfloor.s32.bf16 x1, wl5, s0;		nopx;		nopm;		vmac.f dm2, dm0, x4, x1, r1
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
.Lfunc_end_probe5:
	.size	probe5, .Lfunc_end_probe5-probe5
	.section	.text.probe6,"ax",@progbits
	.globl	probe6
	.p2align	4
	.type	probe6,@function
probe6:
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
	movxm	ls, #probe6_start
	movxm	le, #probe6_last
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
probe6_start:
	vlda.2d x1, [p0], d0;		vldb.4x64.hi wh5, wh2;		vfloor.s32.bf16 x4, wl5, s0;		nopx;		vshuffle x3, x4, x6, r2;		nopv
	vlda.2d x3, [p4], d2;		vldb.4x64.lo wl4, wl2;		vfloor.s32.bf16 x2, wh5, s0;		vadd.32 x7, x4, x10
	vldb.4x64.lo wl1, wl7;		vconv.bf16.fp32 x4, cml2;		vadd.32 x2, x2, x10
	vldb.4x64.hi wh1, wl7;		vst.2d x5, [p1], d1;		vshuffle bmll0, x0, x6, r3;		vmac.f dm2, dm2, x4, x11, r1
	vldb.4x64.lo wl5, wh7;		vshuffle x0, x0, x6, r2
	vldb.4x64.hi wh5, wh7;		vmov wl3, wh1
	vmin_ge.bf16 x1, r16, x8, x1;		vmac.f dm2, dm1, x3, x6, r1
	vldb.4x64.hi wh4, wl2;		vmax_lt.bf16 x6, r16, x9, x1
	vlda.2d x5, [p3], d4;		vldb.4x64.lo wl6, wh2;		vst.2d x6, [p2], d3;		vshuffle bmlh0, x4, x5, r3
	vlda.2d x0, [p0], d0;		vldb.4x64.hi wh6, wh2;		vfloor.s32.bf16 x4, wl6, s0;		vshuffle x3, x4, x5, r2
	vlda.2d x3, [p4], d2;		vldb.4x64.lo wl4, wl2;		vfloor.s32.bf16 x2, wh6, s0;		vadd.32 x7, x4, x10
	vldb.4x64.lo wl0, wl7;		vconv.bf16.fp32 x4, cml2;		vadd.32 x2, x2, x10
	vldb.4x64.hi wh0, wl7;		vst.2d x6, [p1], d1;		vshuffle bmll1, x1, x5, r3;		vmac.f dm2, dm2, x4, x11, r1
	vldb.4x64.lo wl6, wh7;		vshuffle x1, x1, x5, r2
	vldb.4x64.hi wh6, wh7;		vmov wl3, wh0
	vmin_ge.bf16 x0, r16, x8, x0;		vmac.f dm2, dm0, x3, x5, r1
	vldb.4x64.hi wh4, wl2;		vmax_lt.bf16 x5, r16, x9, x0
probe6_last:
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
.Lfunc_end_probe6:
	.size	probe6, .Lfunc_end_probe6-probe6
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
