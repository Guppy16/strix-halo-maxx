_ZN4ssm9L6loga_pEP8bfloat16PKN3aie6vectorIS0_Lj32EEE: 
	nopa	;		nopb	;		nops	;		movxm	m0, #20032;		nopv	
	movxm	p1, ##_ZL7g_arena
	movxm	m1, #13376
	padda	 [p1], m1;		movxm	p0, ##_ZL7g_arena
	padda	 [p0], m0;		mov	p2, p1
	vlda	 x5, [p0], #64;		vldb	 x7, [p2], #128
	vldb	 x3, [p1, #64]
	nop	
	nop	
	movxm	r0, #-16128
	vbcst.16	 x0, r0
	mova	r0, #828
	vmul.f	dm2, x7, x0, r0
	vmul.f	dm0, x3, x0, r0
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cml2
	vconv.bf16.fp32	 x1, cml0
	vmul.f	dm1, x9, x5, r0
	vmul.f	dm4, x1, x5, r0
	nop	
	nop	
	nop	
	movxm	ls, #.LBB18_1
	mova	r1, #8;		vst.conv.bf16.fp32	 cml1, [p1, #0];		movxm	le, #.L_LEnd14
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml4, [p1, #64];		nopx	;		add.nc	lc, r1, #-1;		nopv	
.LBB18_1:                               
	vlda	 x5, [p0], #64;		vldb	 x7, [p2], #128;		movs	p1, p2;		nopxm	;		nopv	
	nopa	;		vldb	 x3, [p1, #64];		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nopxm	
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm2, x7, x0, r0
	vmul.f	dm0, x3, x0, r0
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cml2
	vconv.bf16.fp32	 x1, cml0
	vmul.f	dm1, x9, x5, r0
	vmul.f	dm4, x1, x5, r0
	nop	
	nop	
	nop	
	nop	
	vst.conv.bf16.fp32	 cml1, [p1, #0]
.L_LEnd14:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml4, [p1, #64];		nopxm	;		nopv	
	movs	p1, p2;		ret	lr
	nop	                                
	nop	                                
	nop	                                
	nop	                                
	nop	                                
.Lfunc_end18:
_ZN4ssm9L6exp_pAEPK8bfloat16PS0_Ps:     
	movxm	r0, #-15760
	movxm	p0, ##_ZL7g_arena
	movxm	m2, #13376
	nopx	;		vbcst.16	 x0, r0
	movxm	r0, #17008
	vbcst.16	 x2, r0
	movxm	r0, #16313
	padda	 [p0], m2;		vbcst.16	 x4, r0
	vldb	 x11, [p0], #64;		movxm	r0, #-17620
	vbcst.16	 x6, r0
	movxm	r0, #16256
	vbcst.16	 x8, r0
	movxm	r0, #19264
	vbcst.16	 x10, r0
	movxm	r0, #-13504
	vmax_lt.bf16	 x7, r16, x11, x0
	vmin_ge.bf16	 x7, r16, x7, x2
	mova	r0, #828;		vbcst.16	 x1, r0
	vmul.f	dm0, x7, x4, r0
	movxm	m0, #10240
	movxm	p1, ##_ZL7g_arena
	movxm	m1, #9216;		vmac.f	dm3, dm0, x7, x6, r0
	movxm	p2, ##_ZL7g_arena
	movxm	r1, #-16512
	movxm	ls, #.LBB19_1;		vmac.f	dm1, dm3, x8, x10, r0
	movxm	le, #.L_LEnd15
	mova	r2, #16;		vbcst.16	 x3, r1
	padda	 [p2], m1;		paddb	 [p1], m0;		nops	;		movx	r1, #2;		add.nc	lc, r2, #-1;		vmac.f	dm4, dm1, x8, x1, r0
.LBB19_1:                               
	nopa	;		vldb	 x11, [p0], #64;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopx	;		vmov	x9, bmlh1
	vmov	x11, bmll1
	vshuffle	x11, x11, x9, r1
	vconv.bf16.fp32	 x5, cml4
	vst	 x11, [p1], #64
	vmax_lt.bf16	 x7, r16, x11, x0;		vmac.f	dm2, dm3, x5, x3, r0
	vmin_ge.bf16	 x7, r16, x7, x2
	nop	
	vmul.f	dm0, x7, x4, r0
	nop	
	nop	
	vst.conv.bf16.fp32	 cml2, [p2], #64;		vmac.f	dm3, dm0, x7, x6, r0
	nop	
	nop	
	vmac.f	dm1, dm3, x8, x10, r0
	nop	
	nop	
.L_LEnd15:
	nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm4, dm1, x8, x1, r0
	nopx	
	nop	
	vmov	x9, bmlh1
	vmov	x11, bmll1
	vshuffle	x11, x11, x9, r1
	vconv.bf16.fp32	 x5, cml4
	vst	 x11, [p1], #64
	vmac.f	dm2, dm3, x5, x3, r0
	nop	
	ret	lr
	nop	                                
	nop	                                
	nop	                                
	vst.conv.bf16.fp32	 cml2, [p2], #64 
	nop	                                
.Lfunc_end19:
_ZN4ssm9L6exp_pBEPK8bfloat16PS0_S3_:    
	nopa	;		nopb	;		nops	;		movxm	r0, #16177;		nopv	
	movxm	p0, ##_ZL7g_arena
	movxm	m2, #9216
	vbcst.16	 x0, r0
	movxm	r0, #1065353216
	padda	 [p0], m2;		vbcst.32	 x6, r0
	vldb	 x5, [p0], #64;		movxm	r0, #15990
	vbcst.16	 x2, r0
	movxm	r0, #15715
	vmov	bmll0, x6
	vbcst.16	 x4, r0
	movxm	r0, #-16512
	mova	r0, #828;		vbcst.16	 x6, r0
	vmul.f	dm3, x5, x5, r0
	nop	
	nop	
	nop	
	nop	
	vldb	 x5, [p0], #64
	vconv.bf16.fp32	 x7, cml3
	nop	
	vmul.f	dm2, x7, x5, r0
	nop	
	nop	
	vmac.f	dm4, dm0, x5, x0, r0
	vmov	bmlh0, bmll0;		vmul.f	dm3, x5, x5, r0
	nop	
	vconv.bf16.fp32	 x9, cml2;		vmac.f	dm4, dm4, x7, x2, r0
	nop	
	nop	
	vldb	 x5, [p0], #64;		vmac.f	dm4, dm4, x9, x4, r0
	vconv.bf16.fp32	 x7, cml3
	nop	
	vmul.f	dm2, x7, x5, r0
	nop	
	movxm	m1, #14400
	vconv.bf16.fp32	 x11, cml4;		movxm	p2, ##_ZL7g_arena;		vmac.f	dm4, dm0, x5, x0, r0
	padda	 [p2], m1;		movxm	m0, #15424;		vmul.f	dm3, x5, x5, r0
	vst	 x11, [p2], #64;		movxm	p1, ##_ZL7g_arena;		vmac.f	dm1, dm4, x11, x6, r0
	vconv.bf16.fp32	 x9, cml2;		movxm	ls, #.LBB20_1;		vmac.f	dm4, dm4, x7, x2, r0
	mova	r1, #16;		movxm	le, #.L_LEnd16
	padda	 [p1], m0;		nopb	;		nops	;		nopx	;		add.nc	lc, r1, #-3;		nopv	
.LBB20_1:                               
	nopa	;		vldb	 x5, [p0], #64;		nops	;		nopxm	;		vmac.f	dm4, dm4, x9, x4, r0
	nopa	;		nopb	;		vconv.bf16.fp32	 x7, cml3;		nopxm	;		nopv	
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml1, [p1], #64;		nopxm	;		nopv	
	nopa	;		nopb	;		nopxm	;		vmul.f	dm2, x7, x5, r0
	nop	
	nop	
	vconv.bf16.fp32	 x11, cml4;		vmac.f	dm4, dm0, x5, x0, r0
	vmul.f	dm3, x5, x5, r0
	vst	 x11, [p2], #64;		vmac.f	dm1, dm4, x11, x6, r0
	vconv.bf16.fp32	 x9, cml2;		vmac.f	dm4, dm4, x7, x2, r0
	nop	
.L_LEnd16:
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm4, dm4, x9, x4, r0
	vconv.bf16.fp32	 x7, cml3;		nopx	
	vst.conv.bf16.fp32	 cml1, [p1], #64
	vmul.f	dm2, x7, x5, r0
	nop	
	nop	
	vconv.bf16.fp32	 x11, cml4;		vmac.f	dm4, dm0, x5, x0, r0
	nop	
	vst	 x11, [p2], #64;		vmac.f	dm1, dm4, x11, x6, r0
	vconv.bf16.fp32	 x9, cml2;		vmac.f	dm4, dm4, x7, x2, r0
	nop	
	nop	
	vmac.f	dm4, dm4, x9, x4, r0
	nop	
	vst.conv.bf16.fp32	 cml1, [p1], #64
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x11, cml4
	nop	
	vst	 x11, [p2], #64;		vmac.f	dm1, dm4, x11, x6, r0
	nop	
	ret	lr
	nop	                                
	nop	                                
	nop	                                
	vst.conv.bf16.fp32	 cml1, [p1], #64 
	nop	                                
.Lfunc_end20:
_ZN4ssm9L6exp_pCEPK8bfloat16S2_PKsPS0_S5_: 
	nopa	;		movxm	m2, #10240
	movxm	p3, ##_ZL7g_arena
	padda	 [p3], m2
	vlda.ups.2x	 cml4, s0, upssign1, [p3], #64;		movxm	p0, ##_ZL7g_arena
	movxm	m4, #14400
	padda	 [p0], m4
	vldb	 x3, [p0], #64;		movxm	m3, #15424
	movxm	p4, ##_ZL7g_arena
	padda	 [p4], m3;		mov	s0, #7
	vlda.ups.2x	 cml4, s0, upssign1, [p3], #64;		vldb	 x7, [p4], #64;		mov	crupsmode, #0
	mov	s1, #0
	mov	crsrsmode, #0
	vsrs.2x	x5, cml4, s1, srssign1;		vldb	 x3, [p0], #64;		movxm	r0, #16256
	vbcst.16	 x0, r0
	movxm	r0, #-16512
	vlda.ups.2x	 cml4, s0, upssign1, [p3], #64;		vldb	 x7, [p4], #64;		vbcst.16	 x2, r0
	mova	r0, #16;		vadd.16	 x9, x5, x0
	mova	r0, #828;		add.nc	lc, r0, #-5
	vsrs.2x	x5, cml4, s1, srssign1;		vldb	 x3, [p0], #64;		vmul.f	dm0, x3, x9, r0
	nop	
	nop	
	vlda.ups.2x	 cml4, s0, upssign1, [p3], #64;		vldb	 x7, [p4], #64;		vmac.f	dm1, dm0, x7, x9, r0
	vadd.16	 x9, x5, x0
	nop	
	vsrs.2x	x5, cml4, s1, srssign1;		vldb	 x3, [p0], #64;		vmul.f	dm0, x3, x9, r0
	movxm	m0, #9216
	movxm	m1, #13376
	vlda.ups.2x	 cml4, s0, upssign1, [p3], #64;		vldb	 x7, [p4], #64;		vconv.bf16.fp32	 x11, cml1;		movxm	p2, ##_ZL7g_arena;		vmac.f	dm1, dm0, x7, x9, r0
	padda	 [p2], m1;		vadd.16	 x9, x5, x0
	vst	 x11, [p2], #64;		movxm	p1, ##_ZL7g_arena;		vmac.f	dm2, dm1, x11, x2, r0
	vsrs.2x	x5, cml4, s1, srssign1;		vldb	 x3, [p0], #64;		movxm	ls, #.LBB21_1;		vmul.f	dm0, x3, x9, r0
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd17;		nopv	
	padda	 [p1], m0;		nopb	;		nops	;		nopxm	;		nopv	
.LBB21_1:                               
	vlda.ups.2x	 cml4, s0, upssign1, [p3], #64;		vldb	 x7, [p4], #64;		vconv.bf16.fp32	 x11, cml1;		nopxm	;		vmac.f	dm1, dm0, x7, x9, r0
	nopa	;		nopb	;		nops	;		nopx	;		vadd.16	 x9, x5, x0;		nopv	
	nopa	;		nopb	;		vst	 x11, [p2], #64;		nopxm	;		vmac.f	dm2, dm1, x11, x2, r0
	nopa	;		vldb	 x3, [p0], #64;		vsrs.2x	x5, cml4, s1, srssign1;		nopxm	;		vmul.f	dm0, x3, x9, r0
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml2, [p1], #64;		nopxm	;		nopv	
.L_LEnd17:
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		vldb	 x7, [p4], #64;		vconv.bf16.fp32	 x11, cml1;		nopxm	;		vmac.f	dm1, dm0, x7, x9, r0
	nopa	;		nopb	;		nopx	;		vadd.16	 x9, x5, x0
	vst	 x11, [p2], #64;		vmac.f	dm2, dm1, x11, x2, r0
	vsrs.2x	x5, cml4, s1, srssign1;		vmul.f	dm0, x3, x9, r0
	vst.conv.bf16.fp32	 cml2, [p1], #64
	nop	
	vconv.bf16.fp32	 x11, cml1;		vmac.f	dm1, dm0, x7, x9, r0
	vadd.16	 x9, x5, x0
	vst	 x11, [p2], #64;		vmac.f	dm2, dm1, x11, x2, r0
	vmul.f	dm0, x3, x9, r0
	vst.conv.bf16.fp32	 cml2, [p1], #64
	nop	
	vconv.bf16.fp32	 x11, cml1;		vmac.f	dm1, dm0, x7, x9, r0
	nop	
	vst	 x11, [p2], #64;		vmac.f	dm2, dm1, x11, x2, r0
	nop	
	vst.conv.bf16.fp32	 cml2, [p1], #64
	nop	
	vconv.bf16.fp32	 x11, cml1
	nop	
	vst	 x11, [p2], #64;		vmac.f	dm2, dm1, x11, x2, r0
	nop	
	vst.conv.bf16.fp32	 cml2, [p1], #64
	nop	
	ret	lr
	nop	                                
	nop	                                
	nop	                                
	vst.conv.bf16.fp32	 cml2, [p1], #64 
	nop	                                
.Lfunc_end21:
_ZN4ssm9L6exp_pDEPK8bfloat16S2_PS0_:    
	nopa	;		nopb	;		nops	;		movxm	m1, #9216;		nopv	
	nopa	;		movxm	p2, ##_ZL7g_arena
	padda	 [p2], m1;		movxm	r0, #16384
	vldb	 x1, [p2], #64;		movxm	p0, ##_ZL7g_arena
	movxm	m2, #13376
	vbcst.16	 x0, r0
	movxm	r0, #1065353216
	vldb	 x1, [p2], #64;		vbcst.32	 x4, r0
	padda	 [p0], m2;		movxm	r0, #16256
	vlda	 x3, [p0], #64;		movx	r0, #828;		vbcst.16	 x2, r0
	vmul.f	dm1, x1, x0, r0
	vldb	 x1, [p2], #64
	nop	
	vlda	 x3, [p0], #64
	vmul.f	dm1, x1, x0, r0
	vldb	 x1, [p2], #64
	vconv.bf16.fp32	 x5, cml1;		vmul.f	dm2, x3, x3, r0
	vlda	 x3, [p0], #64
	vmul.f	dm1, x1, x0, r0
	vldb	 x1, [p2], #64;		vmac.f	dm3, dm2, x3, x5, r0
	vconv.bf16.fp32	 x5, cml1;		vmul.f	dm2, x3, x3, r0
	vlda	 x3, [p0], #64
	vmul.f	dm1, x1, x0, r0
	vldb	 x1, [p2], #64;		vmac.f	dm3, dm2, x3, x5, r0
	vconv.bf16.fp32	 x5, cml1;		vmul.f	dm2, x3, x3, r0
	vlda	 x3, [p0], #64;		vconv.bf16.fp32	 x7, cml3
	vmul.f	dm1, x1, x0, r0
	vldb	 x1, [p2], #64;		movxm	ls, #.LBB22_1;		vmac.f	dm3, dm2, x3, x5, r0
	vconv.bf16.fp32	 x5, cml1;		movxm	le, #.L_LEnd18;		vmul.f	dm2, x3, x3, r0
	vlda	 x3, [p0], #64;		vconv.bf16.fp32	 x7, cml3;		vmov	bmll0, x4;		vmsc.f	dm4, dm0, x7, x2, r0
	mova	r1, #16;		vmov	bmlh0, x4;		vmul.f	dm1, x1, x0, r0
	nopa	;		vldb	 x1, [p2], #64;		nops	;		nopx	;		add.nc	lc, r1, #-8;		vmac.f	dm3, dm2, x3, x5, r0
	nopa	;		nopb	;		vconv.bf16.fp32	 x5, cml1;		movxm	m0, #10240;		vmul.f	dm2, x3, x3, r0
	vlda	 x3, [p0], #64;		nopb	;		vconv.bf16.fp32	 x7, cml3;		movxm	p1, ##_ZL7g_arena;		vmsc.f	dm4, dm0, x7, x2, r0
	padda	 [p1], m0;		nopb	;		nops	;		nopxm	;		vmul.f	dm1, x1, x0, r0
.LBB22_1:                               
	nopa	;		vldb	 x1, [p2], #64;		vst.conv.bf16.fp32	 cml4, [p1], #64;		nopxm	;		vmac.f	dm3, dm2, x3, x5, r0
	nopa	;		nopb	;		vconv.bf16.fp32	 x5, cml1;		nopxm	;		vmul.f	dm2, x3, x3, r0
	vlda	 x3, [p0], #64;		nopb	;		vconv.bf16.fp32	 x7, cml3;		nopxm	;		vmsc.f	dm4, dm0, x7, x2, r0
.L_LEnd18:
	nopa	;		nopb	;		nops	;		nopxm	;		vmul.f	dm1, x1, x0, r0
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml4, [p1], #64;		nopxm	;		vmac.f	dm3, dm2, x3, x5, r0
	nopa	;		vconv.bf16.fp32	 x5, cml1;		nopx	;		vmul.f	dm2, x3, x3, r0
	vlda	 x3, [p0], #64;		vconv.bf16.fp32	 x7, cml3;		vmsc.f	dm4, dm0, x7, x2, r0
	vmul.f	dm1, x1, x0, r0
	vst.conv.bf16.fp32	 cml4, [p1], #64;		vmac.f	dm3, dm2, x3, x5, r0
	vconv.bf16.fp32	 x5, cml1;		vmul.f	dm2, x3, x3, r0
	vconv.bf16.fp32	 x7, cml3;		vmsc.f	dm4, dm0, x7, x2, r0
	nop	
	vst.conv.bf16.fp32	 cml4, [p1], #64;		vmac.f	dm3, dm2, x3, x5, r0
	vconv.bf16.fp32	 x5, cml1;		vmul.f	dm2, x3, x3, r0
	vconv.bf16.fp32	 x7, cml3;		vmsc.f	dm4, dm0, x7, x2, r0
	nop	
	vst.conv.bf16.fp32	 cml4, [p1], #64;		vmac.f	dm3, dm2, x3, x5, r0
	nop	
	vconv.bf16.fp32	 x7, cml3;		vmsc.f	dm4, dm0, x7, x2, r0
	nop	
	vst.conv.bf16.fp32	 cml4, [p1], #64
	nop	
	vconv.bf16.fp32	 x7, cml3;		vmsc.f	dm4, dm0, x7, x2, r0
	nop	
	vst.conv.bf16.fp32	 cml4, [p1], #64
	nop	
	vmsc.f	dm4, dm0, x7, x2, r0
	nop	
	vst.conv.bf16.fp32	 cml4, [p1], #64;		ret	lr
	nop	                                
	nop	                                
