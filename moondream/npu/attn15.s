	.file	"attn15.cc"
	.section	.text.attn_begin,"ax",@progbits
	.globl	attn_begin                      // -- Begin function attn_begin
	.p2align	4
	.type	attn_begin,@function
attn_begin:                             // @attn_begin
// %bb.0:                               // %entry
	movxm	p2, ##_ZL6g_init
	lda.s8	 r2, [p2, #0];		nopx	
	nop	
	nop	
	nop	
	nop	
	nop	
	movx	r1, #1
	and	 r4, r2, r1
	jnz	 r4, #.LBB0_12
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mova	r0, #0;		movxm	p0, ##_ZL6g_o_lo //  Delay Slot 2
	mova	r2, #3;		movxm	r3, #-15744     //  Delay Slot 1
// %bb.1:                               // %if.then
	movxm	p5, ##_ZL8ones_grp
	movxm	r18, #1077952639
	movxm	r19, #1077952576
	st.s8	 r1, [p2, #0];		movxm	p2, ##_ZL9mask_full
	movxm	r4, #16256
	movxm	ls, #.LBB0_2
	mova	r20, #108;		movxm	le, #.L_LEnd3
	nopa	;		nopb	;		nops	;		nopx	;		add.nc	lc, r20, #-1;		nopv	
	nopa	;		nopb	;		nops	;		movxm	p3, ##_ZL9mask_last;		nopv	
	mova	r6, #8;		nopb	;		nops	;		movxm	r5, #15857;		nopv	
	mova	r7, #64;		nopb	;		nops	;		movxm	p4, ##_ZL7g_scale;		nopv	
	mova	r16, #9;		nopb	;		nops	;		add	r20, r2, #-3;		mov	crrnd, #12;		nopv	
	mova	r17, #6;		nopb	;		nops	;		add	r20, r2, #4;		mov	dj0, r20;		nopv	
.LBB0_2:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		add	r21, r20, #-3;		nopm	;		nopv	
.L_LEnd3:
	nopa	;		nopb	;		st	 r0, [p5, dj0];		add	r20, r20, #4;		mov	dj0, r21;		nopv	
// %bb.3:
	st	 r0, [p5, dj0]
	st	 r18, [p5], #4
	mova	m0, #64;		st	 r19, [p5], #4
	st.s8	 r7, [p5], m0
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	st	 r18, [p5], #4
	st	 r19, [p5], #4
	st.s8	 r7, [p5], m0
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	st	 r18, [p5], #4
	st	 r19, [p5], #4
	st.s8	 r7, [p5], m0
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	st	 r18, [p5], #4
	st	 r19, [p5], #4
	st.s8	 r7, [p5], m0
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	st	 r18, [p5], #4
	st	 r19, [p5], #4
	st.s8	 r7, [p5], m0
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	st	 r18, [p5], #4
	st	 r19, [p5, #0]
	st.s8	 r7, [p5, #4]
	nop	
	nop	
	nop	
	nop	
	mova	r20, #0
	mova	r19, #0;		movx	r18, #1;		mov	r21, #0
.LBB0_4:                                // %for.cond24.preheader
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_5 Depth 2
                                        //       Child Loop BB0_6 Depth 3
	mova	r22, #0;		nopb	;		nops	;		nopxm	;		nopv	
.LBB0_5:                                // %for.cond29.preheader
                                        //   Parent Loop BB0_4 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_6 Depth 3
	mova	r24, #0;		nopb	;		nops	;		add	 r23, r20, r22;		nopm	;		nopv	
.LBB0_6:                                // %for.body32
                                        //   Parent Loop BB0_4 Depth=1
                                        //     Parent Loop BB0_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		nops	;		jnz	 r21, #.LBB0_8;		nopv	
	nopa	;		nopx	                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	add	 r25, r23, r24                  //  Delay Slot 2
	lshl	 r25, r25, r18                  //  Delay Slot 1
// %bb.7:                               // %if.then38
                                        //   in Loop: Header=BB0_6 Depth=3
	mov	dj0, r25
	st.s16	 r4, [p2, dj0]
	nop	
	nop	
	nop	
.LBB0_8:                                // %if.end
                                        //   in Loop: Header=BB0_6 Depth=3
	nopa	;		add	 r26, r19, r24
	add	r24, r24, #1
	eq	 r25, r24, r6;		mov	dj0, r25
	st.s16	 r26, [p3, dj0];		jz	 r25, #.LBB0_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	ltu	 r27, r26, r16                  //  Delay Slot 2
	sel.nez	 r26, r4, r0, r27               //  Delay Slot 1
// %bb.9:                               // %for.cond.cleanup31
                                        //   in Loop: Header=BB0_5 Depth=2
	add	r22, r22, #8
	eq	 r23, r22, r7
	jz	 r23, #.LBB0_5
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.10:                              // %for.cond.cleanup26
                                        //   in Loop: Header=BB0_4 Depth=1
	add	r21, r21, #1
	eq	 r22, r21, r17
	jz	 r22, #.LBB0_4
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	add	r19, r19, #8                    //  Delay Slot 2
	add	 r20, r20, r7                   //  Delay Slot 1
// %bb.11:                              // %for.cond.cleanup21
	st.s16	 r5, [p4, #0]
.LBB0_12:                               // %if.end53
	nopa	;		nopb	;		nops	;		nopx	;		vbcst.16	 x2, r3;		nopv	
	add.nc	lc, r2, #-1
	movxm	p5, ##_ZL6g_maxb
	movxm	ls, #.LBB0_13
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd2;		nopv	
	nopa	;		nopb	;		nops	;		lshl	 r2, r0, r1;		vbcst.32	 x0, r0;		nopv	
	nopa	;		nopb	;		nops	;		movxm	p4, ##_ZL4o_hi;		nopv	
	nopa	;		nopb	;		nops	;		movxm	p3, ##_ZL5o_sum;		nopv	
	mova	r3, #12;		nopb	;		nops	;		movxm	p2, ##_ZL7g_chunk;		nopv	
	mova	r4, #54;		nopb	;		st	 p1, [p0, #0];		add	r2, r0, #32;		mov	dj0, r2;		nopv	
.LBB0_13:                               // %for.body58
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		lshl	 r5, r2, r1;		nopm	;		nopv	
.L_LEnd2:
	nopa	;		nopb	;		vst	 x2, [p5, dj0];		add	r2, r2, #32;		mov	dj0, r5;		nopv	
// %bb.14:
	movxm	ls, #.LBB0_15
	mova	r1, #2;		movxm	le, #.L_LEnd1
	nopa	;		nopb	;		nops	;		lshl	 r2, r0, r1;		add.nc	lc, r4, #-1;		nopv	
	nopa	;		nopb	;		movs	p0, p1;		nopx	;		mov	m0, r2;		nopv	
	padda	 [p0], m0;		nopb	;		vst	 x2, [p5, dj0];		nopxm	;		nopv	
	padda	 [p4], m0;		nopb	;		vst	 x0, [p0, #64];		add	r2, r0, #32;		nopm	;		nopv	
.LBB0_15:                               // %for.body67
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		vst	 x0, [p0, #0];		lshl	 r4, r2, r1;		mov	p0, p1;		nopv	
	nopa	;		nopb	;		vst	 x0, [p4, #64];		nopx	;		mov	m0, r4;		nopv	
	padda	 [p0], m0;		nopb	;		vst	 x0, [p4, #0];		movxm	p4, ##_ZL4o_hi;		nopv	
.L_LEnd1:
	padda	 [p4], m0;		nopb	;		vst	 x0, [p0, #64];		add	r2, r2, #32;		nopm	;		nopv	
// %bb.16:
	nopa	;		nopx	;		add.nc	lc, r3, #-2
	movxm	ls, #.LBB0_17
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd0;		nopv	
	mova	r1, #2;		nopb	;		vst	 x0, [p0, #0];		movxm	p0, ##_ZL5o_sum;		nopv	
	nopa	;		nopb	;		nops	;		lshl	 r2, r0, r1;		nopm	;		nopv	
	nopa	;		nopb	;		vst	 x0, [p4, #0];		add	r2, r0, #32;		mov	m0, r2;		nopv	
	padda	 [p0], m0;		nopb	;		vst	 x0, [p4, #64];		lshl	 r3, r2, r1;		nopm	;		nopv	
	nopa	;		nopb	;		vst	 x0, [p0, #64];		add	r2, r2, #32;		mov	m0, r3;		nopv	
.LBB0_17:                               // %for.body85
                                        // =>This Inner Loop Header: Depth=1
	vst	 x0, [p0, #0];		movxm	p0, ##_ZL5o_sum
	padda	 [p0], m0;		lshl	 r3, r2, r1
.L_LEnd0:
	nopa	;		nopb	;		vst	 x0, [p0, #64];		add	r2, r2, #32;		mov	m0, r3;		nopv	
// %bb.18:
	nopa	;		nopb	;		ret	lr;		nopm	
	vst	 x0, [p0, #0]                   //  Delay Slot 5
	padda	 [p3], m0;		st	 r0, [p2, #0] //  Delay Slot 4
	vst	 x0, [p3, #64]                  //  Delay Slot 3
	vst	 x0, [p3, #0]                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end0:
	.size	attn_begin, .Lfunc_end0-attn_begin
                                        // -- End function
	.section	.text.attn_chunk_k,"ax",@progbits
	.globl	attn_chunk_k                    // -- Begin function attn_chunk_k
	.p2align	4
	.type	attn_chunk_k,@function
attn_chunk_k:                           // @attn_chunk_k
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		paddxm	 [sp], #4608;		nopv	
	st	 r12, [sp, #-40];		nopb	;		nopx	 // 4-byte Folded Spill
	st	 r13, [sp, #-36]                // 4-byte Folded Spill
	st	 r14, [sp, #-32];		movxm	r0, #-16512 // 4-byte Folded Spill
	st	 r15, [sp, #-28];		movxm	p2, ##_ZL7g_chunk // 4-byte Folded Spill
	st	 r8, [sp, #-56];		movxm	r2, #16313 // 4-byte Folded Spill
	st	 p7, [sp, #-20];		movxm	r3, #-17620 // 4-byte Folded Spill
	st	 r9, [sp, #-52];		movxm	r5, #16256 // 4-byte Folded Spill
	st	 r10, [sp, #-48];		movxm	r7, #15715 // 4-byte Folded Spill
	st	 lr, [sp, #-60];		movxm	r16, ##_ZL9mask_last // 4-byte Folded Spill
	st	 r11, [sp, #-44];		movxm	r17, ##_ZL9mask_full;		vclr	dm0 // 4-byte Folded Spill
	lda	 r1, [p2, #0];		st	 p6, [sp, #-24];		event	#0;		vbcst.16	 x0, r0 // 4-byte Folded Spill
	movxm	r0, #-15840
	st	 p0, [sp, #-112];		vbcst.16	 x0, r0 // 4-byte Folded Spill
	vst	 x0, [sp, #-2880];		movxm	p2, ##_ZL7g_scale // 64-byte Folded Spill
	vst	 x0, [sp, #-3264];		vbcst.16	 x0, r2 // 64-byte Folded Spill
	mova	r4, #15;		vst	 bmll0, [sp, #-2688];		vbcst.16	 x0, r3 // 64-byte Folded Spill
	vst	 x0, [sp, #-3200];		movxm	r3, #19264 // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-2624];		eq	 r27, r1, r4 // 64-byte Folded Spill
	vst	 bmhl0, [sp, #-2560];		movxm	r4, #-13504 // 64-byte Folded Spill
	mova	r6, #6;		vst	 bmhh0, [sp, #-2496];		sel.nez	 r16, r16, r17, r27 // 64-byte Folded Spill
	vst	 x0, [sp, #-3136];		lshl	 r2, r27, r6;		vbcst.16	 x0, r5 // 64-byte Folded Spill
	mova	r1, #3;		st	 r27, [sp, #-116];		movxm	r5, #1065353216 // 4-byte Folded Spill
	mova	r10, #1;		st	 r16, [sp, #-108];		mul	r1, r2, r1;		vbcst.16	 x0, r3 // 4-byte Folded Spill
	vst	 x0, [sp, #-2816];		movxm	r6, #15990 // 64-byte Folded Spill
	vst	 x0, [sp, #-3072];		lshl	 r1, r1, r10;		vbcst.16	 x0, r4 // 64-byte Folded Spill
	mova	r17, #8;		st	 r1, [sp, #-84];		movxm	r4, #16177 // 4-byte Folded Spill
	vst	 x0, [sp, #-2752];		lshl	 r4, r27, r17;		vbcst.16	 x2, r4 // 64-byte Folded Spill
	mova	r20, #7;		st	 r4, [sp, #-92];		mov	p0, r16 // 4-byte Folded Spill
	vst	 x2, [sp, #-3008];		lshl	 r5, r27, r20;		vbcst.32	 x0, r5 // 64-byte Folded Spill
	lda.s16	 r0, [p2, #0];		movs	m0, r5;		vmov	bmll0, x0
	padda	 [p0], m0;		st	 r5, [sp, #-100];		movx	r3, #5;		vbcst.16	 x0, r6 // 4-byte Folded Spill
	mova	r5, #9;		st	 p0, [sp, #-104];		mul	r2, r2, r3 // 4-byte Folded Spill
	vst	 x0, [sp, #-2944];		lshl	 r3, r27, r5;		vbcst.16	 x2, r7 // 64-byte Folded Spill
	st	 r3, [sp, #-80]                 // 4-byte Folded Spill
	vst	 x2, [sp, #-3328]               // 64-byte Folded Spill
	vst	 bmll0, [sp, #-2432];		mov	m0, r4 // 64-byte Folded Spill
	mova	r0, #0;		vst	 bmll0, [sp, #-2368];		vbcst.16	 x0, r0 // 64-byte Folded Spill
	st	 r0, [sp, #-64];		mov	p0, r16 // 4-byte Folded Spill
	padda	 [p0], m0;		st	 r0, [sp, #-128];		lshl	 r1, r2, r10;		mov	m0, r1 // 4-byte Folded Spill
	mova	r19, #-2;		st	 r1, [sp, #-72];		movxm	r18, #1408 // 4-byte Folded Spill
	mova	r21, #0;		st	 p0, [sp, #-96];		movx	r22, #128;		mov	p0, r16 // 4-byte Folded Spill
	padda	 [p0], m0;		movs	m0, r3;		j	#.LBB1_3
	mova	r13, #780;		st	 p0, [sp, #-88];		movx	r12, #704;		mov	p0, r16 // 4-byte Folded Spill Delay Slot 5
	padda	 [p0], m0;		vst	 x0, [sp, #-2304];		movx	r14, #60;		mov	m0, r1 // 64-byte Folded Spill Delay Slot 4
	mova	r8, #4;		st	 p0, [sp, #-76];		movx	r15, #828;		mov	p0, r16 // 4-byte Folded Spill Delay Slot 3
	padda	 [p0], m0;		vst	 x0, [sp, #-2240];		movx	r9, #-128;		mov	p7, p1 // 64-byte Folded Spill Delay Slot 2
	mova	r2, #0;		st	 p0, [sp, #-68];		movx	r20, #16;		mov	r17, #0 // 4-byte Folded Spill Delay Slot 1
.LBB1_1:                                // %if.then85
                                        //   in Loop: Header=BB1_3 Depth=1
	vlda	 x0, [sp, #-2880];		nopb	;		nops	;		nopxm	;		nopv	 // 64-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.fp32.bf16	bmll0, wh0
	vmac.f	dm0, dm0, x2, x0, r15
	nop	
	jl	#_ZL10rescale_rbiN3aie6vectorI8bfloat16Lj16EEE
	nop	                                //  Delay Slot 5
	movxm	p0, ##_ZL6g_maxb                //  Delay Slot 4
	vst	 wl2, [p0, dj0]                 //  Delay Slot 3
	vconv.bf16.fp32	 wl0, bmll0             //  Delay Slot 2
	or	 r11, r21, r21;		mov	r0, r21 //  Delay Slot 1
	vlda	 x4, [sp, #-2880];		nopxm	 // 64-byte Folded Reload
	lda.s16	 r7, [p6], #2
	lda	 r17, [sp, #-124]               // 4-byte Folded Reload
	lda.s16	 r5, [p6], #2
	lda.s16	 r6, [p6], #2
	lda.s16	 r4, [p6], #2
	lda.s16	 r3, [p6], #2
	lda.s16	 r2, [p6], #2;		movx	r22, #128;		mov	r21, r11
	lda.s16	 r1, [p6, #0];		movx	r19, #-2;		mov	r20, #16
	lda.s16	 r0, [p6, #2];		movxm	r18, #1408
.LBB1_2:                                // %if.end93
                                        //   in Loop: Header=BB1_3 Depth=1
	nopx	;		vbcst.16	 x0, r7
	vbcst.16	 x2, r5
	movxm	p0, ##_ZL4s_rb
	vshift	x0, x0, x0, r20
	vshift	x0, x0, x2, r20
	vbcst.16	 x2, r6
	vshift	x0, x0, x2, r20
	vlda.conv.fp32.bf16	 cml0, [p0], #64;		vbcst.16	 x2, r4
	vshift	x0, x0, x2, r20
	vbcst.16	 x2, r3
	vshift	x2, x0, x2, r20
	vlda	 x11, [sp, #-3264];		vconv.fp32.bf16	cml1, x0;		vsub.f	dm0, dm0, dm1, r14 // 64-byte Folded Reload
	vbcst.16	 x0, r2
	vshift	x0, x2, x0, r20
	vlda	 x1, [sp, #-3200];		vbcst.16	 x2, r1 // 64-byte Folded Reload
	vshift	x0, x0, x2, r20
	vbcst.16	 x2, r0
	vlda	 x10, [sp, #-3136];		vconv.bf16.fp32	 x2, cml0 // 64-byte Folded Reload
	vlda	 x7, [sp, #-2816];		vshift	x0, x0, x2, r20 // 64-byte Folded Reload
	vlda.conv.fp32.bf16	 cml0, [p0], #64;		vmax_lt.bf16	 x2, r16, x2, x11
	vlda	 x9, [sp, #-3072]               // 64-byte Folded Reload
	vmul.f	dm1, x2, x1, r15
	nop	
	vlda	 x6, [sp, #-2752];		vst	 bmll1, [sp, #-4096];		vconv.fp32.bf16	cml1, x0;		vsub.f	dm0, dm0, dm1, r14 // 64-byte Folded Reload64-byte Folded Spill
	vst	 bmlh1, [sp, #-4032];		vmac.f	dm2, dm1, x2, x10, r15 // 64-byte Folded Spill
	vst	 bmll1, [sp, #-4608]            // 64-byte Folded Spill
	vst	 bmlh1, [sp, #-4544]            // 64-byte Folded Spill
	vmac.f	dm1, dm2, x7, x9, r15
	nop	
	nop	
	vmac.f	dm0, dm1, x7, x6, r15
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmax_lt.bf16	 x0, r16, x0, x11
	nop	
	vconv.bf16.fp32	 x2, cml0;		vmul.f	dm2, x0, x1, r15
	nop	
	vmac.f	dm0, dm2, x2, x4, r15
	vmac.f	dm2, dm2, x0, x10, r15
	nop	
	nop	
	vmac.f	dm0, dm2, x7, x9, r15
	nop	
	nop	
	vmac.f	dm3, dm0, x7, x6, r15
	nop	
	vconv.bf16.fp32	 x2, cml0
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml3
	vmul.f	dm3, x2, x2, r15
	vmac.f	dm4, dm2, x0, x4, r15
	nop	
	vlda	 x5, [sp, #-3008];		vmov	lfh0, x4 // 64-byte Folded Reload
	vlda	 x1, [sp, #-2944];		vmov	sfl, x10 // 64-byte Folded Reload
	vlda	 bmll2, [sp, #-2432];		vmov	lfe, x6 // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-2368];		vconv.bf16.fp32	 x4, cml3;		vmov	x6, bmll1 // 64-byte Folded Reload
	mova	r1, #2;		vconv.bf16.fp32	 x6, cml4;		vmov	x0, bmlh1
	vlda	 x4, [sp, #-3328];		vshuffle	x10, x6, x0, r1;		vmul.f	dm1, x4, x2, r15 // 64-byte Folded Reload
	vmul.f	dm3, x6, x6, r15
	vst	 bmhl2, [sp, #-4224];		vmov	sfh, x1;		vmac.f	dm2, dm2, x2, x5, r15 // 64-byte Folded Spill
	vst	 bmhh2, [sp, #-4160]            // 64-byte Folded Spill
	vst	 bmll2, [sp, #-4352]            // 64-byte Folded Spill
	vst	 bmlh2, [sp, #-4288];		vmac.f	dm2, dm2, x4, x1, r15 // 64-byte Folded Spill
	vconv.bf16.fp32	 x2, cml1
	vconv.bf16.fp32	 x2, cml3
	vmac.f	dm1, dm2, x2, x4, r15
	vlda	 bmhl0, [sp, #-4224];		vmul.f	dm0, x2, x6, r15 // 64-byte Folded Reload
	vlda	 bmhh0, [sp, #-4160]            // 64-byte Folded Reload
	vlda	 bmll0, [sp, #-4352]            // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-4288];		mov	crupsmode, #0 // 64-byte Folded Reload
	vmov	x8, bmll0
	vmov	x0, bmlh0
	mova	r0, #7;		mov	crsrsmode, #0
	mov	s0, r0;		vmac.f	dm0, dm0, x6, x5, r15
	vconv.bf16.fp32	 x8, cml0;		vshuffle	x3, x8, x0, r1
	mova	r0, #0;		vups.2x	cml2, x10, s0, upssign1
	mov	s1, r0;		vmac.f	dm0, dm0, x2, x1, r15
	vups.2x	cml1, x3, s0, upssign1
	vsrs.2x	x0, cml2, s1, srssign1
	vlda	 bmhl2, [sp, #-3968];		vconv.bf16.fp32	 x6, cml1;		vmac.f	dm0, dm0, x8, x4, r15 // 64-byte Folded Reload
	vlda	 bmhh2, [sp, #-3904];		vsrs.2x	x4, cml1, s1, srssign1 // 64-byte Folded Reload
	lda	 p1, [sp, #-108]                // 4-byte Folded Reload
	vlda	 bmll2, [sp, #-4096];		vadd.16	 x0, x0, x7 // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-4032];		movxm	p1, ##_ZL9mask_last // 64-byte Folded Reload
	vlda.conv.fp32.bf16	 cml0, [p0], #64;		vadd.16	 x4, x4, x7;		vmul.f	dm0, x6, x0, r15
	lda	 r0, [sp, #-116];		vldb	 x6, [p1, #64];		movxm	p1, ##_ZL9mask_full;		vconv.bf16.fp32	 x2, cml0 // 4-byte Folded Reload
	vldb	 x8, [p1, #64]
	vmul.f	dm1, x2, x4, r15
	vldb	 x4, [p1, #0];		vsub.f	dm0, dm0, dm2, r14
	vlda.conv.fp32.bf16	 cml1, [p0], #64
	vlda	 bmll2, [sp, #-4608];		vconv.bf16.fp32	 x0, cml0 // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-4544]            // 64-byte Folded Reload
	vlda	 bmhl2, [sp, #-4480];		add	r16, r0, #-1;		vmov	lfh1, x5 // 64-byte Folded Reload
	vlda	 bmhh2, [sp, #-4416];		vconv.bf16.fp32	 x1, cml1;		vsel.32	 x5, x6, x8, r16 // 64-byte Folded Reload
	vconv.bf16.fp32	 x2, cml0;		vmov	lfl1, x1
	vmov	x6, sfh;		vmul.f	dm0, y0, y2, r15
	vmax_lt.bf16	 x2, r16, x2, x11
	vsub.f	dm2, dm1, dm2, r14
	vmul.f	dm1, x2, x6, r15
	vmov	x3, sfl
	nop	
	vconv.bf16.fp32	 x0, cml0;		vmac.f	dm4, dm1, x2, x3, r15
	vconv.bf16.fp32	 x4, cmh0;		movxm	p2, ##_ZL4e_rb
	movs	m0, r17;		vconv.fp32.bf16	cml0, x0
	vconv.bf16.fp32	 x0, cml2;		vconv.fp32.bf16	cmh0, x4;		vmac.f	dm0, dm4, x7, x9, r15
	padda	 [p2], m0;		movx	r26, #0
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26];		vmax_lt.bf16	 x0, r16, x0, x11
	nop	
	vlda	 bmhl2, [sp, #-3968];		vmov	x1, lfe;		vmul.f	dm0, x0, x6, r15 // 64-byte Folded Reload
	vlda	 bmhh2, [sp, #-3904]            // 64-byte Folded Reload
	vlda	 bmll2, [sp, #-4096];		vst	 bmll0, [sp, #-1920];		vmac.f	dm3, dm0, x7, x1, r15 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh2, [sp, #-4032];		vst	 bmlh0, [sp, #-1856];		vmac.f	dm1, dm0, x0, x3, r15 // 64-byte Folded Reload64-byte Folded Spill
	vlda.conv.fp32.bf16	 cml3, [p0], #64;		vst	 bmhl0, [sp, #-1792];		vmov	x10, x9 // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-1728]            // 64-byte Folded Spill
	vmac.f	dm0, dm1, x7, x10, r15
	nop	
	vsub.f	dm3, dm3, dm2, r14
	nop	
	vconv.bf16.fp32	 x0, cml3
	vmac.f	dm0, dm0, x7, x1, r15
	vlda	 bmll2, [sp, #-4608];		vst	 bmll0, [sp, #-2176] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh2, [sp, #-4544];		vst	 bmlh0, [sp, #-2112];		vmov	x5, x11 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmhl2, [sp, #-4480];		vconv.bf16.fp32	 x2, cml3;		vmov	x9, x6 // 64-byte Folded Reload
	vlda	 bmhh2, [sp, #-4416];		vst	 bmhl0, [sp, #-2048];		vmov	x6, lfh0 // 64-byte Folded Reload64-byte Folded Spill
	vlda.conv.fp32.bf16	 cml1, [p0], #64;		vst	 bmhh0, [sp, #-1984];		vmax_lt.bf16	 x2, r16, x2, x5 // 64-byte Folded Spill
	vconv.bf16.fp32	 x4, cml0;		vmac.f	dm4, dm4, x0, x6, r15
	vmul.f	dm1, x2, x9, r15
	vmac.f	dm0, dm1, x4, x6, r15
	vsub.f	dm4, dm1, dm2, r14
	vmac.f	dm1, dm1, x2, x3, r15
	nop	
	nop	
	vmac.f	dm0, dm1, x7, x10, r15
	vconv.bf16.fp32	 x11, cml4
	nop	
	nop	
	vlda	 bmhl0, [sp, #-4224];		vmac.f	dm0, dm0, x7, x1, r15 // 64-byte Folded Reload
	vlda	 bmhh0, [sp, #-4160];		vconv.bf16.fp32	 x4, cml0;		vmul.f	dm3, x11, x11, r15 // 64-byte Folded Reload
	vlda	 bmll0, [sp, #-4352];		vst	 bmll0, [sp, #-3840] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh0, [sp, #-4288];		vst	 bmlh0, [sp, #-3776] // 64-byte Folded Reload64-byte Folded Spill
	vst	 bmhl0, [sp, #-3712];		vmov	x2, x10 // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-3648];		vmov	x10, lfh1 // 64-byte Folded Spill
	nop	
	vconv.bf16.fp32	 x8, cml3;		vmov	x0, x3;		vmac.f	dm2, dm0, x11, x10, r15
	vconv.bf16.fp32	 x7, cml0;		vmov	x3, lfl1
	vmov	lfh0, x7
	vmov	lfl0, lfh0;		vmac.f	dm2, dm2, x8, x3, r15
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x6, cml4;		vmul.f	dm3, x4, x4, r15
	nop	
	vlda	 x5, [sp, #-3328];		vst	 bmll2, [sp, #-3584];		vmax_lt.bf16	 x6, r16, x6, x5;		vmul.f	dm3, x8, x11, r15 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll0, [sp, #-3584];		vst	 bmlh2, [sp, #-3520] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh0, [sp, #-3520];		vst	 bmhl2, [sp, #-3456];		vmul.f	dm4, x6, x9, r15 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmhl0, [sp, #-3456];		vst	 bmhh2, [sp, #-3392];		vmov	lfe, x5 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmhh0, [sp, #-3392];		vconv.bf16.fp32	 x6, cml3;		vmov	x5, lfl0;		vmac.f	dm1, dm0, x4, x10, r15 // 64-byte Folded Reload
	vmac.f	dm4, dm4, x6, x0, r15
	vconv.bf16.fp32	 x0, cml3;		vmac.f	dm2, dm1, x7, x5, r15
	vmov	lfh1, x0;		vmul.f	dm3, x6, x4, r15
	vmac.f	dm0, dm0, x0, x5, r15
	nop	
	nop	
	vmov	x11, lfh0
	nop	
	vlda	 bmhl3, [sp, #-4480];		vconv.bf16.fp32	 x4, cml3;		vmac.f	dm1, dm4, x11, x2, r15 // 64-byte Folded Reload
	vlda	 bmhh3, [sp, #-4416];		vst	 bmll0, [sp, #-1664];		vmac.f	dm0, dm1, x6, x3, r15 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmhl2, [sp, #-3968];		vst	 bmlh0, [sp, #-1600] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmhh2, [sp, #-3904];		vst	 bmhl0, [sp, #-1536] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll2, [sp, #-4096];		vst	 bmhh0, [sp, #-1472];		vmac.f	dm0, dm0, x4, x5, r15 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh2, [sp, #-4032];		vconv.bf16.fp32	 x0, cml2 // 64-byte Folded Reload
	vlda.conv.fp32.bf16	 cml1, [p0], #64;		vst	 bmll1, [sp, #-3584];		vmac.f	dm1, dm1, x11, x1, r15 // 64-byte Folded Spill
	vlda	 bmll3, [sp, #-4608];		vst	 bmlh1, [sp, #-3520] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh3, [sp, #-4544];		vst	 bmhl1, [sp, #-3456] // 64-byte Folded Reload64-byte Folded Spill
	vlda.conv.fp32.bf16	 cml1, [p0], #64;		vst	 bmhh1, [sp, #-3392];		vmul.f	dm0, x0, x0, r15 // 64-byte Folded Spill
	vlda	 bmhl2, [sp, #-4224];		vst	 bmll0, [sp, #-1408];		vsub.f	dm2, dm1, dm2, r14 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmhh2, [sp, #-4160];		vst	 bmlh0, [sp, #-1344];		vmov	lfl1, x1 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll2, [sp, #-4352];		vconv.bf16.fp32	 x2, cml1;		vmov	lfh0, x2 // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-4288];		vst	 bmhl0, [sp, #-1280];		vmov	x1, lfl0 // 64-byte Folded Reload64-byte Folded Spill
	vst	 bmhh0, [sp, #-1216]            // 64-byte Folded Spill
	vmac.f	dm4, dm4, x2, x1, r15
	vconv.bf16.fp32	 x2, cml0;		vsub.f	dm3, dm1, dm3, r14
	vmac.f	dm0, dm2, x0, x10, r15
	vconv.bf16.fp32	 x4, cml2;		vmul.f	dm0, x2, x0, r15
	nop	
	vmov	x7, x11;		vmac.f	dm1, dm0, x2, x3, r15
	vmov	x11, lfe
	vconv.bf16.fp32	 x6, cml3;		vmax_lt.bf16	 x4, r16, x4, x11
	nop	
	vconv.bf16.fp32	 x2, cml0;		vmax_lt.bf16	 x0, r16, x6, x11;		vmul.f	dm3, x4, x9, r15
	vmov	x6, lfh1
	vmac.f	dm1, dm1, x2, x5, r15
	vmac.f	dm3, dm3, x4, x6, r15
	vmov	x4, lfh0
	vmul.f	dm0, x0, x9, r15
	vmac.f	dm1, dm3, x7, x4, r15
	vconv.bf16.fp32	 x0, cml4
	vst	 bmll1, [sp, #-896];		vmov	x9, lfl1;		vmac.f	dm0, dm0, x0, x6, r15 // 64-byte Folded Spill
	vst	 bmlh1, [sp, #-832]             // 64-byte Folded Spill
	vst	 bmhl1, [sp, #-768];		vmac.f	dm1, dm1, x7, x9, r15 // 64-byte Folded Spill
	vst	 bmhh1, [sp, #-704];		vmul.f	dm1, x0, x0, r15 // 64-byte Folded Spill
	vst	 bmll1, [sp, #-384];		vmac.f	dm1, dm0, x7, x4, r15 // 64-byte Folded Spill
	vst	 bmlh1, [sp, #-320]             // 64-byte Folded Spill
	vst	 bmhl1, [sp, #-256]             // 64-byte Folded Spill
	vst	 bmhh1, [sp, #-192]             // 64-byte Folded Spill
	vconv.bf16.fp32	 x2, cml1;		vmov	x6, lfl0;		vmac.f	dm1, dm1, x7, x9, r15
	vconv.bf16.fp32	 x2, cml1
	vst	 bmll1, [sp, #-640];		vmac.f	dm4, dm3, x2, x6, r15 // 64-byte Folded Spill
	vst	 bmlh1, [sp, #-576]             // 64-byte Folded Spill
	vst	 bmhl1, [sp, #-512];		vmac.f	dm1, dm2, x0, x10, r15 // 64-byte Folded Spill
	vst	 bmhh1, [sp, #-448]             // 64-byte Folded Spill
	vconv.bf16.fp32	 x4, cml1
	vmac.f	dm3, dm1, x2, x3, r15
	vconv.bf16.fp32	 x4, cml4;		vmac.f	dm1, dm0, x4, x6, r15
	vmul.f	dm0, x2, x0, r15
	vmul.f	dm0, x4, x4, r15
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	vconv.bf16.fp32	 x0, cml0
	vmac.f	dm3, dm3, x0, x5, r15
	vmul.f	dm1, x0, x4, r15
	nop	
	vmac.f	dm0, dm2, x4, x10, r15
	nop	
	nop	
	vconv.bf16.fp32	 x10, cml1;		vmac.f	dm0, dm0, x0, x3, r15
	vlda	 bmll3, [sp, #-1920];		vconv.bf16.fp32	 x0, cml1;		vmov	x9, x10 // 64-byte Folded Reload
	vlda	 bmlh3, [sp, #-1856];		vmul.f	dm1, x10, x10, r15 // 64-byte Folded Reload
	vlda	 bmll0, [sp, #-1664];		vmac.f	dm0, dm0, x0, x5, r15 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-1600]            // 64-byte Folded Reload
	vlda	 bmll0, [sp, #-2176]            // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-2112];		vst	 bmll3, [sp, #-1152] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll0, [sp, #-1408];		vst	 bmlh3, [sp, #-1088] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh0, [sp, #-1344];		vconv.bf16.fp32	 x1, cml1 // 64-byte Folded Reload
	vlda	 bmll4, [sp, #-4608];		vst	 bmll0, [sp, #-1920] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh4, [sp, #-4544];		vst	 bmlh0, [sp, #-1856];		vmul.f	dm0, x1, x10, r15 // 64-byte Folded Reload64-byte Folded Spill
	vlda.conv.fp32.bf16	 cml0, [p0], #64;		vconv.bf16.fp32	 x11, cml0;		vmov	x8, x3;		vmac.f	dm0, dm2, x10, x9, r15
	vmov	x2, bmlh3
	vst	 bmhl0, [sp, #-1792];		vmov	x3, bmll0 // 64-byte Folded Spill
	vlda.conv.fp32.bf16	 cml0, [p0], #64;		vst	 bmhh0, [sp, #-1728];		vmov	x4, bmlh0;		vmac.f	dm2, dm0, x1, x8, r15 // 64-byte Folded Spill
	vconv.bf16.fp32	 x2, cml0;		vmov	x6, bmll3
	vlda	 bmhl3, [sp, #-1536];		vconv.bf16.fp32	 x0, cml0;		vshuffle	x6, x6, x2, r1 // 64-byte Folded Reload
	vlda	 bmhh3, [sp, #-1472]            // 64-byte Folded Reload
	lda	 p3, [sp, #-104];		vst	 bmll0, [sp, #-1664];		vmov	cml1, cml4;		vsub.f	dm1, dm0, dm1, r14 // 4-byte Folded Reload64-byte Folded Spill
	vlda	 bmll3, [sp, #-1664];		vst	 bmlh0, [sp, #-1600];		vups.2x	cml3, x6, s0, upssign1 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh3, [sp, #-1600];		vshuffle	x4, x3, x4, r1 // 64-byte Folded Reload
	vlda	 bmll2, [sp, #-4096];		vst	 bmhl3, [sp, #-1024] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh2, [sp, #-4032];		vst	 bmhh3, [sp, #-960];		vups.2x	cml1, x4, s0, upssign1 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmhl2, [sp, #-3968];		vsrs.2x	x6, cml3, s1, srssign1 // 64-byte Folded Reload
	vlda	 bmhh2, [sp, #-3904];		vconv.bf16.fp32	 x8, cml1;		vmac.f	dm0, dm2, x0, x5, r15 // 64-byte Folded Reload
	lda	 dj0, [sp, #-100];		vsrs.2x	x6, cml1, s1, srssign1 // 4-byte Folded Reload
	vlda.conv.fp32.bf16	 cml0, [p0, #0]
	vlda	 x5, [sp, #-3264];		vadd.16	 x4, x6, x7 // 64-byte Folded Reload
	vlda	 bmll2, [sp, #-4608];		vsub.f	dm4, dm3, dm2, r14 // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-4544];		vadd.16	 x4, x6, x7;		vmul.f	dm1, x11, x4, r15 // 64-byte Folded Reload
	vlda	 bmhl2, [sp, #-4480];		vsub.f	dm0, dm0, dm2, r14 // 64-byte Folded Reload
	vlda	 bmhh2, [sp, #-4416];		vldb	 x3, [p3, #64];		vst	 bmll0, [sp, #-2176];		vmul.f	dm1, x2, x4, r15 // 64-byte Folded Reload64-byte Folded Spill
	vlda.conv.fp32.bf16	 cml1, [p0, #64];		vldb	 x2, [p1, dj0];		vst	 bmlh0, [sp, #-2112] // 64-byte Folded Spill
	vlda	 x11, [sp, #-3200];		vst	 bmhl0, [sp, #-2048] // 64-byte Folded Reload64-byte Folded Spill
	vst	 bmhh0, [sp, #-1984]            // 64-byte Folded Spill
	vlda	 x10, [sp, #-3136];		vconv.bf16.fp32	 x0, cml1 // 64-byte Folded Reload
	vconv.bf16.fp32	 x4, cml0;		vsub.f	dm1, dm1, dm2, r14
	vconv.bf16.fp32	 x1, cml1
	vmax_lt.bf16	 x2, r16, x4, x5
	vlda	 x9, [sp, #-3072];		vmul.f	dm0, y0, y1, r15 // 64-byte Folded Reload
	vmul.f	dm1, x2, x11, r15
	vconv.bf16.fp32	 x0, cml4
	vst	 bmll1, [sp, #-4608];		vmax_lt.bf16	 x4, r16, x8, x5 // 64-byte Folded Spill
	vst	 bmlh1, [sp, #-4544];		vmax_lt.bf16	 x0, r16, x0, x5;		vmac.f	dm1, dm1, x2, x10, r15 // 64-byte Folded Spill
	vst	 bmhl1, [sp, #-4480];		vmul.f	dm3, x4, x11, r15 // 64-byte Folded Spill
	vst	 bmhh1, [sp, #-4416];		vmul.f	dm2, x0, x11, r15 // 64-byte Folded Spill
	vmac.f	dm0, dm1, x7, x9, r15
	vmov	lfh0, x10
	vmov	x10, lfl1;		vmac.f	dm4, dm3, x4, x10, r15
	vmac.f	dm2, dm2, x0, x10, r15
	vconv.bf16.fp32	 x0, cml0;		vmac.f	dm0, dm0, x7, x10, r15
	vconv.bf16.fp32	 x2, cmh0
	vst	 bmll0, [sp, #-1664]            // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-1600]            // 64-byte Folded Spill
	vst	 bmhl0, [sp, #-1536]            // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-1472];		vmov	lfh1, x11 // 64-byte Folded Spill
	vconv.bf16.fp32	 x4, cml0;		vmov	x11, lfl0
	nop	
	vlda	 bmlh0, [sp, #-3776];		vmac.f	dm0, dm1, x4, x11, r15 // 64-byte Folded Reload
	vlda	 bmll0, [sp, #-3840]            // 64-byte Folded Reload
	vst.flush.512.conv	 [p2, sf, r26];		vconv.fp32.bf16	cmh0, x2
	vlda	 bmlh0, [sp, #-3520];		vconv.fp32.bf16	cml0, x0;		vmac.f	dm1, dm2, x7, x9, r15 // 64-byte Folded Reload
	vlda	 bmll0, [sp, #-3584]            // 64-byte Folded Reload
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26]
	vst	 bmlh0, [sp, #-4032];		vmac.f	dm0, dm1, x7, x10, r15 // 64-byte Folded Spill
	vst	 bmll0, [sp, #-4096];		vmov	x0, bmlh0 // 64-byte Folded Spill
	vmov	x2, bmll0;		vmac.f	dm0, dm4, x7, x9, r15
	vshuffle	x0, x2, x0, r1
	vst	 bmhl0, [sp, #-3968];		vmov	x4, bmlh0 // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-3904];		vmov	x6, bmll0;		vmac.f	dm3, dm0, x7, x10, r15 // 64-byte Folded Spill
	vconv.bf16.fp32	 x0, cml0
	vmov	lfl0, x0
	vmac.f	dm2, dm2, x0, x11, r15
	nop	
	vlda	 bmll2, [sp, #-4096];		vshuffle	x2, x6, x4, r1 // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-4032];		vconv.bf16.fp32	 x2, cml3 // 64-byte Folded Reload
	vmov	lfl1, x2
	vlda	 bmll2, [sp, #-384];		vmac.f	dm2, dm4, x2, x11, r15 // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-320];		vst	 bmlh2, [sp, #-1344] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll2, [sp, #-640];		vst	 bmhl2, [sp, #-1280] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh2, [sp, #-576];		vst	 bmll2, [sp, #-1408] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll2, [sp, #-4608];		vst	 bmhh2, [sp, #-1216] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh2, [sp, #-4544];		vconv.bf16.fp32	 x1, cml2 // 64-byte Folded Reload
	vlda	 x6, [sp, #-2944];		vst	 bmll2, [sp, #-4096] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll4, [sp, #-4352];		vst	 bmlh2, [sp, #-4032] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 x8, [sp, #-3008];		vmov	x2, bmll2 // 64-byte Folded Reload
	vlda	 bmlh4, [sp, #-4288];		vmov	x0, bmlh2 // 64-byte Folded Reload
	vlda	 bmhl4, [sp, #-4224];		vmov	x3, bmll2;		vmul.f	dm2, x1, x1, r15 // 64-byte Folded Reload
	vlda	 bmhh4, [sp, #-4160];		vmov	x4, bmlh2 // 64-byte Folded Reload
	vconv.bf16.fp32	 x8, cml2
	vmov	bmhl1, x7
	vst	 bmhl2, [sp, #-3968];		vmax_lt.bf16	 x5, r16, x8, x5 // 64-byte Folded Spill
	vst	 bmhh2, [sp, #-3904];		vshuffle	x0, x2, x0, r1;		vmac.f	dm2, dm4, x1, x8, r15 // 64-byte Folded Spill
	vconv.bf16.fp32	 x7, cml2;		vmov	x2, bmll1
	vmov	lfe, x0
	vmov	x0, lfh1;		vmac.f	dm2, dm2, x7, x6, r15
	vshuffle	x0, x3, x4, r1
	vmov	x3, bmlh0;		vmul.f	dm3, x5, x0, r15
	vmov	lfh1, x0
	vmov	x0, lfh0
	vmov	x5, bmll0
	vst	 bmll2, [sp, #-4608];		vmov	x0, bmlh1;		vmac.f	dm2, dm3, x5, x0, r15 // 64-byte Folded Spill
	vst	 bmlh2, [sp, #-4544];		vshuffle	x0, x2, x0, r1 // 64-byte Folded Spill
	vst	 bmhl2, [sp, #-4480];		vshuffle	x0, x5, x3, r1 // 64-byte Folded Spill
	vst	 x0, [sp, #-3584];		vmov	x0, bmhl1 // 64-byte Folded Spill
	vst	 x0, [sp, #-3840]               // 64-byte Folded Spill
	vst	 bmhh2, [sp, #-4416];		vmac.f	dm1, dm2, x0, x9, r15 // 64-byte Folded Spill
	nop	
	vlda	 x2, [sp, #-3328];		vmul.f	dm0, x7, x1, r15 // 64-byte Folded Reload
	vlda	 bmhl0, [sp, #-4480];		vmac.f	dm0, dm1, x0, x10, r15 // 64-byte Folded Reload
	vlda	 bmhh0, [sp, #-4416]            // 64-byte Folded Reload
	vlda	 bmll0, [sp, #-4608]            // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-4544]            // 64-byte Folded Reload
	nop	
	vconv.bf16.fp32	 x1, cml0
	nop	
	vmac.f	dm0, dm0, x1, x2, r15
	vlda	 bmll0, [sp, #-896];		vconv.bf16.fp32	 x3, cml0 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-832]             // 64-byte Folded Reload
	vlda	 bmll2, [sp, #-1152]            // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-1088]            // 64-byte Folded Reload
	vlda	 bmll2, [sp, #-1408]            // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-1344];		vst	 bmhl0, [sp, #-4480];		vmac.f	dm0, dm2, x3, x11, r15 // 64-byte Folded Reload64-byte Folded Spill
	vst	 bmll0, [sp, #-4608]            // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-4544]            // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-4416]            // 64-byte Folded Spill
	vconv.bf16.fp32	 x1, cml0
	vconv.bf16.fp32	 x0, cml2
	vmov	x9, x0
	vconv.bf16.fp32	 x5, cml2
	vmov	bmll3, x0
	vlda	 bmlh2, [sp, #-1600];		vmov	x0, lfl0;		vmul.f	dm2, x5, x5, r15 // 64-byte Folded Reload
	vlda	 bmll2, [sp, #-1664]            // 64-byte Folded Reload
	vups.2x	cml2, x0, s0, upssign1
	nop	
	vmov	x0, lfl1
	vsrs.2x	x10, cml2, s1, srssign1
	vconv.bf16.fp32	 x7, cml2;		vups.2x	cml2, x0, s0, upssign1
	vmov	x3, bmlh2
	vmov	x11, bmll2
	lda	 p0, [sp, #-96];		vsrs.2x	x10, cml2, s1, srssign1 // 4-byte Folded Reload
	lda	 dj0, [sp, #-92]                // 4-byte Folded Reload
	nop	
	vadd.16	 x4, x10, x9
	vadd.16	 x4, x10, x9
	vmov	x10, bmll3;		vmul.f	dm2, x1, x4, r15
	nop	
	vldb	 x11, [p0, #64];		vmul.f	dm2, x10, x4, r15
	vldb	 x10, [p1, dj0]
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml2;		vmul.f	dm2, x7, x5, r15
	vconv.bf16.fp32	 x1, cml2;		vshuffle	x4, x11, x3, r1
	vmac.f	dm3, dm4, x5, x8, r15
	vmul.f	dm4, y0, y5, r15
	nop	
	vmac.f	dm3, dm3, x7, x6, r15
	vconv.bf16.fp32	 x3, cml2;		vmov	x11, x2
	vmov	x2, lfe
	vlda	 bmll3, [sp, #-1920];		vmov	bmhl1, x8;		vmac.f	dm2, dm3, x3, x2, r15 // 64-byte Folded Reload
	vlda	 bmlh3, [sp, #-1856];		vconv.bf16.fp32	 x10, cmh4;		vups.2x	cml3, x2, s0, upssign1 // 64-byte Folded Reload
	vlda	 bmll3, [sp, #-2176];		vconv.bf16.fp32	 x0, cml4 // 64-byte Folded Reload
	vlda	 bmlh3, [sp, #-2112];		vconv.fp32.bf16	cmh3, x10 // 64-byte Folded Reload
	vst.flush.512.conv	 [p2, sf, r26];		vconv.fp32.bf16	cml3, x0
	vsrs.2x	x8, cml3, s1, srssign1
	vst.push.576.conv.bfp16ebs8.fp32	 dm3, [p2, sf, r26];		vmov	x2, lfh1
	nop	
	vconv.bf16.fp32	 x0, cml3;		vups.2x	cml3, x2, s0, upssign1
	nop	
	lda	 dj0, [sp, #-84];		vconv.bf16.fp32	 x10, cml3 // 4-byte Folded Reload
	lda	 p0, [sp, #-88];		vsrs.2x	x8, cml3, s1, srssign1 // 4-byte Folded Reload
	nop	
	vmov	lfh0, x6
	vadd.16	 x6, x8, x9
	vadd.16	 x0, x8, x9
	vmul.f	dm3, x0, x6, r15
	vldb	 x8, [p1, dj0];		vmul.f	dm3, x10, x0, r15
	vldb	 x9, [p0, #64]
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x6, cml3
	vconv.bf16.fp32	 x7, cml3
	vmov	bmhl2, x9
	vlda	 bmll0, [sp, #-4096];		vmul.f	dm0, y3, y4, r15 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-4032]            // 64-byte Folded Reload
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmul.f	dm0, x0, x0, r15
	vconv.bf16.fp32	 x6, cml0
	vlda	 bmhl3, [sp, #-4224]            // 64-byte Folded Reload
	vlda	 bmhh3, [sp, #-4160];		vconv.bf16.fp32	 x1, cml0 // 64-byte Folded Reload
	vlda	 bmll3, [sp, #-4352]            // 64-byte Folded Reload
	vlda	 bmlh3, [sp, #-4288];		vconv.bf16.fp32	 x8, cmh0;		vmul.f	dm0, x1, x1, r15 // 64-byte Folded Reload
	vconv.bf16.fp32	 x10, cml0
	vmov	x9, bmhl1
	vmul.f	dm0, x10, x0, r15
	vmac.f	dm1, dm3, x0, x9, r15
	vlda	 x2, [sp, #-3584];		vmov	x0, lfh0 // 64-byte Folded Reload
	vconv.bf16.fp32	 x3, cml0;		vmac.f	dm3, dm3, x1, x9, r15
	vmac.f	dm1, dm1, x10, x0, r15
	vlda	 x2, [sp, #-3840];		vmov	x5, bmlh1;		vmul.f	dm4, x3, x1, r15 // 64-byte Folded Reload
	vconv.bf16.fp32	 x0, cml0;		vmov	x7, bmll1
	vmac.f	dm3, dm3, x3, x0, r15
	vmac.f	dm0, dm1, x0, x11, r15
	vups.2x	cml1, x2, s0, upssign1
	nop	
	vconv.bf16.fp32	 x0, cml4
	vsrs.2x	x0, cml1, s1, srssign1;		vups.2x	cml1, x2, s0, upssign1
	lda	 p0, [sp, #-76];		vmac.f	dm3, dm3, x0, x11, r15 // 4-byte Folded Reload
	vlda	 bmll1, [sp, #-4608];		vshuffle	x10, x7, x5, r1 // 64-byte Folded Reload
	vlda	 bmlh1, [sp, #-4544];		vsrs.2x	x2, cml1, s1, srssign1;		vmov	x5, bmhl2 // 64-byte Folded Reload
	lda	 dj0, [sp, #-80];		vconv.bf16.fp32	 x6, cml2;		vadd.16	 x0, x0, x5 // 4-byte Folded Reload
	vst.flush.512.conv	 [p2, sf, r26];		vconv.fp32.bf16	cml1, x6
	vconv.bf16.fp32	 x3, cml0;		vconv.fp32.bf16	cmh1, x8;		vmul.f	dm0, x6, x0, r15
	vconv.bf16.fp32	 x8, cml3;		vadd.16	 x0, x2, x5
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		vups.2x	cml1, x4, s0, upssign1
	vmul.f	dm0, x8, x0, r15
	vldb	 x7, [p0, #64];		vconv.bf16.fp32	 x1, cml1
	vsrs.2x	x2, cml1, s1, srssign1;		vldb	 x6, [p1, dj0];		vups.2x	cml1, x10, s0, upssign1
	vconv.bf16.fp32	 x4, cml0
	lda	 p0, [sp, #-68]                 // 4-byte Folded Reload
	lda	 dj0, [sp, #-72];		vsrs.2x	x0, cml1, s1, srssign1 // 4-byte Folded Reload
	vconv.bf16.fp32	 x5, cml0;		vadd.16	 x0, x2, x5
	nop	
	vmov	x2, bmhl2;		vmul.f	dm0, x1, x0, r15
	vadd.16	 x2, x0, x2;		vmul.f	dm0, y2, y3, r15
	nop	
	vldb	 x3, [p0, #64];		vmul.f	dm1, x3, x2, r15
	vldb	 x2, [p1, dj0]
	nop	
	vconv.bf16.fp32	 x0, cml0
	vconv.bf16.fp32	 x2, cml0
	lda	 r0, [sp, #-128];		vconv.bf16.fp32	 x2, cmh0 // 4-byte Folded Reload
	vconv.bf16.fp32	 x1, cml1;		vconv.fp32.bf16	cml0, x2
	vconv.fp32.bf16	cmh0, x2
	vmul.f	dm1, y0, y1, r15
	nop	
	nop	
	nop	
	add	r0, r0, #16
	st	 r0, [sp, #-128]                // 4-byte Folded Spill
	vconv.bf16.fp32	 x0, cml1
	lda	 r2, [sp, #-120];		vconv.bf16.fp32	 x0, cmh1 // 4-byte Folded Reload
	mova	r0, #6;		vst.flush.512.conv	 [p2, sf, r26];		add	r21, r21, #1;		vconv.fp32.bf16	cml0, x0
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26];		eq	 r0, r21, r0;		vconv.fp32.bf16	cmh0, x0
	vst.flush.512.conv	 [p2, sf, r26];		jnz	 r0, #.LBB1_9
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26] //  Delay Slot 5
	vst.flush.512.conv	 [p2, sf, r26]  //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mova	r1, #448;		add	 r2, r2, r12 //  Delay Slot 2
	add	 r17, r17, r1                   //  Delay Slot 1
.LBB1_3:                                // %_ZN3aie41block_vector_restrict_input_buffer_streamI9bfp16ebs8Lj64EL15aie_dm_resource5EE3popEv.exit.8
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB1_4 Depth 2
	lda	 p0, [sp, #-112];		nopb	;		nopxm	 // 4-byte Folded Reload
	lda	 r24, [sp, #-64]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	mov	m0, r2
	padda	 [p0], m0
	vldb.fill.512	 [p0, lf0, r24]
	vldb.pop.576	 ex0, [p0, lf0, r24]
	nop	
	nop	
	vldb.pop.576	 ex0, [p0, lf0, r24]
	nop	
	nop	
	st	 r17, [sp, #-124];		vldb.pop.576	 ex0, [p0, lf0, r24];		mov	p1, sp // 4-byte Folded Spill
	st	 r2, [sp, #-120];		movxm	dj0, #-4608 // 4-byte Folded Spill
	vst	 x0, [p1, dj0];		movxm	dj0, #-4544 // 64-byte Folded Spill
	st	 el0, [p1, dj0];		vldb.pop.576	 ex0, [p0, lf0, r24];		movxm	dj0, #-4540 // 4-byte Folded Spill
	st	 eh0, [p1, dj0];		movxm	dj0, #-4352 // 4-byte Folded Spill
	vst	 x0, [p1, dj0];		movxm	dj0, #-4288 // 64-byte Folded Spill
	st	 el0, [p1, dj0];		movxm	dj0, #-4284 // 4-byte Folded Spill
	st	 eh0, [p1, dj0];		movxm	dj0, #-4096 // 4-byte Folded Spill
	vst	 x0, [p1, dj0];		vldb.pop.576	 ex8, [p0, lf0, r24];		movxm	dj0, #-4032 // 64-byte Folded Spill
	st	 el0, [p1, dj0];		vldb.pop.576	 ex2, [p0, lf0, r24];		movxm	dj0, #-4028 // 4-byte Folded Spill
	st	 eh0, [p1, dj0];		vldb.pop.576	 ex1, [p0, lf0, r24];		movxm	dj0, #-3840 // 4-byte Folded Spill
	vst	 x0, [p1, dj0];		vldb.pop.576	 ex3, [p0, lf0, r24];		movxm	dj0, #-3776 // 64-byte Folded Spill
	st	 el0, [p1, dj0];		vldb.fill.512	 [p0, lf0, r24];		movxm	dj0, #-3772 // 4-byte Folded Spill
	mova	r26, #0;		vldb.pop.576	 ex9, [p0, lf0, r24];		movx	r1, #-2;		mov	r0, #64;		st	 eh0, [p1, dj0] // 4-byte Folded Spill
.LBB1_4:                                // %if.end54.4
                                        //   Parent Loop BB1_3 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	 r2, r26, r12;		mov	p2, sp
	movxm	dj0, #-4608
	movs	p0, p7;		mov	m0, r2
	padda	 [p0], m0;		movs	p1, p7;		movx	r24, #0;		mov	m0, r26
	padda	 [p1], m0;		vldb.fill.512	 [p0, lf0, r24];		mov	r25, r24
	vlda	 x6, [p2, dj0];		vldb.fill.512	 [p1, lf1, r25];		movxm	dj0, #-4544 // 64-byte Folded Reload
	lda	 el6, [p2, dj0];		vldb.pop.576	 ex4, [p1, lf1, r25];		movxm	dj0, #-4540 // 4-byte Folded Reload
	lda	 eh6, [p2, dj0];		vldb.pop.576	 ex4, [p0, lf0, r24];		vmov	sfh, x11 // 4-byte Folded Reload
	vldb.pop.576	 ex5, [p1, lf1, r25];		movxm	dj0, #-4096
	vlda	 x4, [p2, dj0];		vldb.pop.576	 ex0, [p1, lf1, r25];		movxm	dj0, #-4032 // 64-byte Folded Reload
	lda	 el4, [p2, dj0];		vldb.pop.576	 ex11, [p0, lf0, r24];		movxm	dj0, #-4028 // 4-byte Folded Reload
	lda	 eh4, [p2, dj0];		vldb.pop.576	 ex0, [p0, lf0, r24];		vmov	sfl, x4 // 4-byte Folded Reload
	movxm	dj0, #-4352
	vlda	 x4, [p2, dj0];		movxm	dj0, #-4288 // 64-byte Folded Reload
	lda	 el4, [p2, dj0];		movxm	dj0, #-4284;		vmul.f	dm0, ex6, ex4, r13 // 4-byte Folded Reload
	lda	 eh4, [p2, dj0];		movxm	dj0, #-3840;		vmul.f	dm1, ex6, ex4, r13 // 4-byte Folded Reload
	vlda	 x6, [p2, dj0];		movxm	dj0, #-3776 // 64-byte Folded Reload
	lda	 el6, [p2, dj0];		vldb.pop.576	 ex0, [p1, lf1, r25];		movxm	dj0, #-3772 // 4-byte Folded Reload
	lda	 eh6, [p2, dj0];		vldb.pop.576	 ex0, [p0, lf0, r24];		vmac.f	dm0, dm0, ex4, ex0, r13 // 4-byte Folded Reload
	vldb.pop.576	 ex0, [p1, lf1, r25];		vmac.f	dm1, dm1, ex4, ex0, r13
	vldb.pop.576	 ex0, [p0, lf0, r24]
	vldb.pop.576	 ex0, [p1, lf1, r25]
	vldb.pop.576	 ex0, [p0, lf0, r24];		vmul.f	dm2, ex4, ex5, r13
	vldb.pop.576	 ex0, [p1, lf1, r25];		vmul.f	dm3, ex4, ex11, r13
	vldb.pop.576	 ex0, [p0, lf0, r24]
	vldb.pop.576	 ex0, [p1, lf1, r25];		vmac.f	dm2, dm2, ex6, ex0, r13
	vldb.fill.512	 [p1, lf1, r25];		vmac.f	dm3, dm3, ex6, ex0, r13
	vldb.pop.576	 ex0, [p0, lf0, r24];		vmac.f	dm0, dm0, ex8, ex0, r13
	vldb.pop.576	 ex0, [p1, lf1, r25];		vmac.f	dm1, dm1, ex8, ex0, r13
	vmac.f	dm4, dm2, ex2, ex0, r13
	vmac.f	dm2, dm3, ex2, ex0, r13
	vldb.fill.512	 [p0, lf0, r24];		vmac.f	dm0, dm0, ex1, ex0, r13
	vldb.pop.576	 ex0, [p0, lf0, r24];		vmac.f	dm1, dm1, ex1, ex0, r13
	nop	
	vmac.f	dm3, dm4, ex3, ex0, r13
	vmac.f	dm2, dm2, ex3, ex0, r13
	vmac.f	dm0, dm0, ex9, ex0, r13
	nop	
	vlda	 x10, [sp, #-2304]              // 64-byte Folded Reload
	vlda	 x11, [sp, #-2240];		vadd.f	dm3, dm0, dm3, r14 // 64-byte Folded Reload
	vmac.f	dm1, dm1, ex9, ex0, r13
	nop	
	nop	
	vadd.f	dm0, dm1, dm2, r14
	nop	
	vconv.bf16.fp32	 x4, cml3
	vconv.bf16.fp32	 x5, cmh3
	nop	
	vmul.f	dm1, y2, y5, r15
	vconv.bf16.fp32	 x6, cml0
	vconv.bf16.fp32	 x7, cmh0
	lshl	 r2, r0, r10
	add	 r3, r2, r9;		vmul.f	dm0, y3, y5, r15
	mov	m0, r3
	vconv.bf16.fp32	 x0, cml1;		movxm	p0, ##_ZL4s_rb
	padda	 [p0], m0;		vconv.bf16.fp32	 x6, cmh1;		eq	 r2, r1, r19;		mov	m0, r2
	vst	 x0, [p0, #0];		jnz	 r2, #.LBB1_6
	vst	 x6, [p0, #64]                  //  Delay Slot 5
	vconv.bf16.fp32	 x4, cml0;		movxm	p1, ##_ZL4s_rb //  Delay Slot 4
	padda	 [p1], m0;		vconv.bf16.fp32	 x0, cmh0 //  Delay Slot 3
	vst	 x4, [p1, #0];		vmax_lt.bf16	 x11, r16, x0, x4 //  Delay Slot 2
	vst	 x0, [p1, #64];		vmax_lt.bf16	 x4, r16, x6, x0 //  Delay Slot 1
// %bb.5:                               // %cond.false
                                        //   in Loop: Header=BB1_4 Depth=2
	vmov	x0, sfh
	vmax_lt.bf16	 x11, r16, x0, x11
	vmov	x0, sfl
	vmax_lt.bf16	 x4, r16, x0, x4
.LBB1_6:                                // %cond.end
                                        //   in Loop: Header=BB1_4 Depth=2
	nopa	;		nopb	;		add	r1, r1, #2;		nopm	;		nops	
	ltu	 r2, r1, r8
	jnz	 r2, #.LBB1_4
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	add	 r26, r26, r18                  //  Delay Slot 2
	add	 r0, r0, r22                    //  Delay Slot 1
// %bb.7:                               // %for.cond.cleanup10
                                        //   in Loop: Header=BB1_3 Depth=1
	lda	 r0, [sp, #-128]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	vlda	 bmll1, [sp, #-2688];		movxm	p6, ##_ZL6g_maxb // 64-byte Folded Reload
	vlda	 bmlh1, [sp, #-2624];		lshl	 r0, r0, r10 // 64-byte Folded Reload
	vlda	 bmhl1, [sp, #-2560];		mov	m0, r0 // 64-byte Folded Reload
	padda	 [p6], m0
	mova	r0, #52;		vldb	 wh0, [p6, #0];		mov	dj0, r0
	vlda	 bmhh1, [sp, #-2496];		vshuffle	x2, x11, x4, r0 // 64-byte Folded Reload
	mova	r0, #53;		vmov	wl6, wh2
	vshuffle	x4, x11, x4, r0
	vmax_lt.bf16	 x2, r16, x2, x6
	vmov	wl6, wh4
	vmax_lt.bf16	 x4, r16, x4, x6
	vmov	wl0, wh0;		vadd.f	dm0, dm0, dm1, r14
	vconv.fp32.bf16	cml0, x0
	vmax_lt.bf16	 x2, r16, x2, x4
	vmov	wh4, wl2
	vmov	wl4, wl2
	vshift	x4, x4, x0, r20
	vconv.bf16.fp32	 x4, cml0;		vmax_lt.bf16	 x2, r16, x2, x4
	vmax_lt.bf16	 x2, r16, x2, x0
	vlt.bf16	 r16, x4, x2
	nop	
	extend.u16	 r0, r16
	jnz	 r0, #.LBB1_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.8:                               //   in Loop: Header=BB1_3 Depth=1
	vextract.16	 r5, x0, #1, vaddsign1
	vextract.16	 r6, x0, #2, vaddsign1
	vextract.16	 r4, x0, #3, vaddsign1
	j	#.LBB1_2
	vextract.16	 r3, x0, #4, vaddsign1  //  Delay Slot 5
	vextract.16	 r2, x0, #5, vaddsign1  //  Delay Slot 4
	vextract.16	 r1, x0, #6, vaddsign1  //  Delay Slot 3
	vlda	 x4, [sp, #-2880];		vextract.16	 r7, x0, #0, vaddsign1 // 64-byte Folded Reload Delay Slot 2
	lda	 r17, [sp, #-124];		vextract.16	 r0, x0, #7, vaddsign1 // 4-byte Folded Reload Delay Slot 1
.LBB1_9:                                // %for.cond.cleanup
	lda	 p7, [sp, #-20];		event	#1      // 4-byte Folded Reload
	lda	 p6, [sp, #-24]                 // 4-byte Folded Reload
	lda	 r15, [sp, #-28]                // 4-byte Folded Reload
	lda	 r14, [sp, #-32]                // 4-byte Folded Reload
	lda	 lr, [sp, #-60]                 // 4-byte Folded Reload
	lda	 r13, [sp, #-36]                // 4-byte Folded Reload
	lda	 r12, [sp, #-40]                // 4-byte Folded Reload
	lda	 r11, [sp, #-44]                // 4-byte Folded Reload
	lda	 r10, [sp, #-48]                // 4-byte Folded Reload
	lda	 r9, [sp, #-52]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-56]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-4608                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end1:
	.size	attn_chunk_k, .Lfunc_end1-attn_chunk_k
                                        // -- End function
	.section	.text._ZL10rescale_rbiN3aie6vectorI8bfloat16Lj16EEE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL10rescale_rbiN3aie6vectorI8bfloat16Lj16EEE
	.type	_ZL10rescale_rbiN3aie6vectorI8bfloat16Lj16EEE,@function
_ZL10rescale_rbiN3aie6vectorI8bfloat16Lj16EEE: // @_ZL10rescale_rbiN3aie6vectorI8bfloat16Lj16EEE
// %bb.0:                               // %entry
	mova	r2, #0;		movxm	r1, #-15840
	vbcst.32	 x4, r2
	vbcst.16	 x2, r1
	movxm	r1, #16313
	vmov	wh0, wl4
	vmax_lt.bf16	 x0, r16, x0, x2
	vbcst.16	 x2, r1
	mova	r1, #828;		movxm	r3, #-17620
	vbcst.16	 x4, r3;		vmul.f	dm0, x0, x2, r1
	movxm	r3, #16256
	vbcst.16	 x2, r3
	movxm	r3, #19264;		vmac.f	dm1, dm0, x0, x4, r1
	vbcst.16	 x0, r3
	nop	
	movxm	r3, #-13504;		vmac.f	dm0, dm1, x2, x0, r1
	vbcst.16	 x0, r3
	nop	
	vmac.f	dm2, dm0, x2, x0, r1
	nop	
	nop	
	nop	
	nop	
	movxm	r3, #-16512
	vconv.bf16.fp32	 x4, cml2;		vbcst.16	 x0, r3
	nop	
	vmac.f	dm1, dm1, x4, x0, r1
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x4, cml1
	nop	
	movxm	r4, #15715;		vmul.f	dm1, x4, x4, r1
	mov	s0, #7
	movxm	r3, #16177
	vbcst.16	 x6, r3
	movxm	r3, #1065353216
	vbcst.32	 x10, r3
	vconv.bf16.fp32	 x8, cml1;		movxm	r3, #15990
	vmov	x3, bmlh0
	vmov	x5, bmll0;		vmul.f	dm1, x8, x4, r1
	vmov	bmll0, x10
	vmov	bmlh0, x10;		vmac.f	dm0, dm0, x4, x6, r1
	vbcst.16	 x4, r3
	mova	r3, #2;		mov	crupsmode, #0
	vshuffle	x10, x5, x3, r3;		vmac.f	dm0, dm0, x8, x4, r1
	vconv.bf16.fp32	 x4, cml1;		vbcst.16	 x6, r4
	vups.2x	cml1, x10, s0, upssign1
	mov	crsrsmode, #0;		vmac.f	dm0, dm0, x4, x6, r1
	mov	s0, r2
	vsrs.2x	x4, cml1, s0, srssign1
	nop	
	nop	
	movxm	p2, ##_ZL6g_o_lo
	lda	 p2, [p2, #0];		vconv.bf16.fp32	 x4, cml0;		vadd.16	 x2, x4, x2
	movxm	p0, ##_ZL4o_hi
	mova	r4, #576;		movxm	p1, ##_ZL5o_sum;		vmul.f	dm0, x4, x2, r1
	mul	r6, r0, r4
	movxm	r7, #4187392
	lshl	 r6, r6, r3
	add	 r7, r6, r7;		mov	m0, r6
	padda	 [p2], m0;		mov	m0, r7
	padda	 [p0], m0;		vconv.bf16.fp32	 x4, cml0;		mov	r6, p2
	mov	r7, p0
	vextbcst.16	 x2, x4, #0
	vextbcst.16	 x6, x4, #1
	vextbcst.16	 x8, x4, #2
	mova	r5, #16;		vextbcst.16	 x10, x4, #3
	vshift	x2, x0, x2, r5
	vshift	x2, x2, x6, r5
	vextbcst.16	 x6, x4, #4
	vshift	x2, x2, x8, r5
	vextbcst.16	 x8, x4, #5
	vshift	x6, x0, x6, r5
	mova	r16, #3;		vshift	x2, x2, x10, r5
	lt	 r27, r0, r16;		vextbcst.16	 x10, x4, #6
	mova	r7, #8;		sel.nez	 r6, r6, r7, r27;		vextbcst.16	 x4, x4, #7
	lshl	 r0, r0, r7;		vshift	x6, x6, x8, r5
	movs	m0, r0;		vshift	x6, x6, x10, r5
	padda	 [p1], m0;		vshift	x3, x6, x4, r5
	movs	p0, r6;		lshl	 r6, r2, r3;		mov	r5, p1
	movs	m0, r6;		mov	p1, p0
	padda	 [p1], m0
	eq	 r27, r2, r4;		mov	r6, p1
	sel.nez	 r6, r5, r6, r27
	mov	p1, r6
	vlda	 bmhl0, [p1, #128]
	vlda	 bmll0, [p1, #0]
	vlda	 bmlh0, [p1, #64]
	vlda	 bmhh0, [p1, #192]
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x4, cml0
	vconv.bf16.fp32	 x5, cmh0;		vmov	x1, x0
	nop	
	vmac.f	dm0, dm0, y2, y0, r1
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x4, cml0;		vmul.f	dm1, y2, y1, r1
	vconv.bf16.fp32	 x5, cmh0
	nop	
	vmac.f	dm2, dm1, y2, y1, r1
	nop	
	nop	
	movxm	ls, #.LBB2_1
	movxm	le, #.L_LEnd4
	mova	r0, #64;		movx	r7, #10
	vst	 bmll2, [p1, #0];		add	 r2, r2, r0;		add.nc	lc, r7, #-1
.LBB2_1:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		lshl	 r6, r2, r3;		nopm	;		nops	
	movs	p2, p0;		mov	m0, r6
	padda	 [p2], m0
	vst	 bmlh2, [p1, #64];		eq	 r27, r2, r4;		mov	r6, p2
	vst	 bmhl2, [p1, #128];		sel.nez	 r6, r5, r6, r27
	vst	 bmhh2, [p1, #192];		mov	p1, r6
	vlda	 bmlh4, [p1, #64]
	vlda	 bmhh4, [p1, #192]
	vlda	 bmhl4, [p1, #128]
	vlda	 bmll4, [p1, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cmh4
	vconv.bf16.fp32	 x8, cml4
	nop	
	vmac.f	dm0, dm4, y4, y0, r1
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x11, cmh0;		vmul.f	dm1, y4, y1, r1
	vconv.bf16.fp32	 x10, cml0
	nop	
	vmac.f	dm2, dm1, y5, y1, r1
	nop	
	nop	
	nop	
	nop	
	nop	
.L_LEnd4:
	nopa	;		nopb	;		vst	 bmll2, [p1, #0];		add	 r2, r2, r0;		nopm	;		nopv	
// %bb.2:
	nopa	;		nopb	;		nops	;		ret	lr;		nopm	;		nopv	
	nop	                                //  Delay Slot 5
	vst	 bmlh2, [p1, #64]               //  Delay Slot 4
	vst	 bmhl2, [p1, #128]              //  Delay Slot 3
	vst	 bmhh2, [p1, #192]              //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end2:
	.size	_ZL10rescale_rbiN3aie6vectorI8bfloat16Lj16EEE, .Lfunc_end2-_ZL10rescale_rbiN3aie6vectorI8bfloat16Lj16EEE
                                        // -- End function
	.section	.text.attn_chunk_v,"ax",@progbits
	.globl	attn_chunk_v                    // -- Begin function attn_chunk_v
	.p2align	4
	.type	attn_chunk_v,@function
attn_chunk_v:                           // @attn_chunk_v
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		paddxm	 [sp], #64;		nopv	
	mova	r1, #5;		nopb	;		movx	r0, #3;		st	 r8, [sp, #-64] // 4-byte Folded Spill
	mova	r4, #448;		st	 r9, [sp, #-60];		movx	r3, #64;		mov	r2, #9 // 4-byte Folded Spill
	mova	r7, #896;		st	 r10, [sp, #-56];		movx	r6, #60;		mov	r5, #780 // 4-byte Folded Spill
	mova	r19, #6;		st	 r11, [sp, #-52];		movx	r18, #576;		mov	r16, #128 // 4-byte Folded Spill
	mova	r17, #16;		st	 r12, [sp, #-48];		movxm	p2, ##_ZL6g_o_lo // 4-byte Folded Spill
	mova	r20, #2;		st	 r13, [sp, #-44];		movxm	r26, ##_ZL8ones_grp // 4-byte Folded Spill
	mova	r22, #256;		st	 r14, [sp, #-40];		movxm	r21, #4187392 // 4-byte Folded Spill
	mova	r27, #0;		st	 r15, [sp, #-36];		movx	r25, #0;		mov	r23, p0 // 4-byte Folded Spill
	mova	r29, #0;		st	 p6, [sp, #-32];		event	#0;		mov	r28, #0 // 4-byte Folded Spill
	lda	 p5, [p2, #0];		movxm	p2, ##_ZL7g_chunk
	mova	r9, #0;		movx	r30, #0;		mov	p6, p0
.LBB3_1:                                // %_ZN3aie41block_vector_restrict_input_buffer_streamI9bfp16ebs8Lj64EL15aie_dm_resource5EE3popEv.exit.5
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB3_2 Depth 2
	nopa	;		nopb	;		movxm	p0, ##_ZL4e_rb;		movs	m0, r29
	padda	 [p0], m0;		mov	r24, r25
	vldb.fill.512	 [p0, lf0, r24]
	vldb.pop.576	 ex0, [p0, lf0, r24]
	vldb.pop.576	 ex2, [p0, lf0, r24]
	vldb.pop.576	 ex4, [p0, lf0, r24];		mov	r12, r29
	vldb.pop.576	 ex6, [p0, lf0, r24];		movxm	p1, ##_ZL5o_sum
	vldb.pop.576	 ex8, [p0, lf0, r24];		movxm	ls, #.LBB3_2
	vldb.pop.576	 ex10, [p0, lf0, r24];		movxm	le, #.L_LEnd5
	lshl	 r24, r27, r20;		mov	r10, r27
	movxm	p0, ##_ZL4o_hi
	mova	r29, #-2;		movs	p3, p5;		or	 r11, r28, r28;		mov	m0, r24
	padda	 [p3], m0;		add	 r25, r24, r21;		add.nc	lc, r1, #0
	mova	r14, #0;		movs	m0, r25;		lshl	 r28, r28, r20;		mov	r24, p3
	padda	 [p0], m0;		or	 r13, r30, r30;		mov	m0, r28
	padda	 [p1], m0;		ltu	 r27, r30, r0;		mov	r25, p0
	mova	r15, #0;		sel.nez	 r30, r24, r25, r27;		mov	r28, p1
.LBB3_2:                                // %for.inc73.2
                                        //   Parent Loop BB3_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		nops	;		add	r25, r29, #3;		nopm	;		nopv	
	lshl	 r24, r14, r20;		mov	p4, r30
	add	 r8, r15, r4;		mov	p3, r30
	nopa	;		movs	m0, r24;		ltu	 r27, r25, r2;		mov	r25, r9
	padda	 [p3], m0;		add	 r31, r24, r22;		mov	m0, r15
	sel.nez	 r24, r23, r26, r27
	sel.nez	 r8, r8, r9, r27;		mov	p1, p6
	mov	p0, r24
	padda	 [p1], m0;		mov	m0, r8
	padda	 [p0], m0;		mov	m0, r31
	vlda	 bmll1, [p3, #0];		vldb.fill.512	 [p1, lf1, r25];		padds	 [p4], m0;		mov	r24, r9
	vlda	 bmlh1, [p3, #64];		vldb.fill.512	 [p0, lf0, r24];		mov	r31, p4
	vlda	 bmhl1, [p3, #128];		vldb.pop.576	 ex1, [p1, lf1, r25];		sel.nez	 r27, r31, r28, r27
	vlda	 bmhh1, [p3, #192];		vldb.pop.576	 ex3, [p0, lf0, r24];		movs	p4, r27
	vlda	 bmll2, [p4, #0];		vldb.pop.576	 ex5, [p1, lf1, r25]
	vlda	 bmlh2, [p4, #64];		vldb.pop.576	 ex7, [p0, lf0, r24]
	vlda	 bmhl2, [p4, #128];		vldb.pop.576	 ex9, [p1, lf1, r25]
	vlda	 bmhh2, [p4, #192];		vldb.pop.576	 ex1, [p1, lf1, r25]
	vldb.pop.576	 ex11, [p0, lf0, r24]
	vldb.pop.576	 ex5, [p1, lf1, r25]
	vldb.pop.576	 ex11, [p1, lf1, r25];		vmac.f	dm3, dm1, ex0, ex1, r5
	vldb.pop.576	 ex3, [p0, lf0, r24];		vmac.f	dm4, dm2, ex0, ex3, r5
	vldb.pop.576	 ex9, [p0, lf0, r24];		vmul.f	dm1, ex2, ex5, r5
	vmul.f	dm2, ex2, ex7, r5
	vldb.pop.576	 ex1, [p0, lf0, r24];		vmac.f	dm0, dm3, ex4, ex9, r5
	vmac.f	dm1, dm1, ex6, ex1, r5
	vmac.f	dm4, dm4, ex4, ex11, r5
	vmac.f	dm0, dm0, ex8, ex5, r5
	vmac.f	dm1, dm1, ex10, ex11, r5
	vmac.f	dm2, dm2, ex6, ex3, r5
	vmac.f	dm4, dm4, ex8, ex9, r5
	vadd.f	dm0, dm0, dm1, r6
	vmac.f	dm2, dm2, ex10, ex1, r5
	nop	
	nop	
	vadd.f	dm4, dm4, dm2, r6
	nop	
	vst	 bmll0, [p3, #0]
	vst	 bmlh0, [p3, #64]
	vst	 bmhl0, [p3, #128]
	vst	 bmhh0, [p3, #192]
	vst	 bmll4, [p4, #0]
	vst	 bmlh4, [p4, #64];		add	r29, r29, #2
	vst	 bmhl4, [p4, #128];		add	 r14, r14, r16
.L_LEnd5:
	nopa	;		nopb	;		vst	 bmhh4, [p4, #192];		add	 r15, r15, r7;		nopm	;		nopv	
// %bb.3:                               // %for.cond.cleanup7
                                        //   in Loop: Header=BB3_1 Depth=1
	nopa	;		nopb	;		add	r30, r13, #1;		nopm	
	eq	 r24, r30, r19
	jz	 r24, #.LBB3_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	add	 r29, r12, r4                   //  Delay Slot 3
	add	 r28, r11, r3                   //  Delay Slot 2
	mova	r25, #0;		add	 r27, r10, r18  //  Delay Slot 1
// %bb.4:                               // %for.cond.cleanup
	lda	 r0, [p2, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	add	r0, r0, #1
	eq	 r27, r0, r17
	sel.nez	 r0, r25, r0, r27
	lda	 p6, [sp, #-32];		st	 r0, [p2, #0];		event	#1 // 4-byte Folded Reload
	lda	 r15, [sp, #-36]                // 4-byte Folded Reload
	lda	 r14, [sp, #-40]                // 4-byte Folded Reload
	lda	 r13, [sp, #-44]                // 4-byte Folded Reload
	lda	 r12, [sp, #-48]                // 4-byte Folded Reload
	lda	 r11, [sp, #-52]                // 4-byte Folded Reload
	lda	 r10, [sp, #-56]                // 4-byte Folded Reload
	lda	 r9, [sp, #-60]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-64]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end3:
	.size	attn_chunk_v, .Lfunc_end3-attn_chunk_v
                                        // -- End function
	.section	.text.attn_copyq,"ax",@progbits
	.globl	attn_copyq                      // -- Begin function attn_copyq
	.p2align	4
	.type	attn_copyq,@function
attn_copyq:                             // @attn_copyq
// %bb.0:                               // %entry
	mova	r0, #66;		nopb	;		nops	;		nopxm	;		nopv	
	mova	r0, #2;		movx	r1, #0;		add.nc	lc, r0, #-1
	lshl	 r2, r1, r0
	mov	dj0, r2
	vldb	 x2, [p1, dj0]
	nop	
	nop	
	movxm	ls, #.LBB4_1
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd6;		nopv	
	nopa	;		nopb	;		nops	;		add	r1, r1, #16;		nopm	;		nopv	
.LBB4_1:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		lshl	 r2, r1, r0;		nopm	;		nopv	
	nopa	;		nopb	;		vst	 x2, [p0, dj0];		nopx	;		mov	dj0, r2;		nopv	
	nopa	;		vldb	 x2, [p1, dj0];		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nopxm	
	nop	
	nop	
.L_LEnd6:
	nopa	;		nopb	;		nops	;		add	r1, r1, #16;		nopm	;		nopv	
// %bb.2:
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst	 x2, [p0, dj0]                  //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end4:
	.size	attn_copyq, .Lfunc_end4-attn_copyq
                                        // -- End function
	.section	.text.attn_end,"ax",@progbits
	.globl	attn_end                        // -- Begin function attn_end
	.p2align	4
	.type	attn_end,@function
attn_end:                               // @attn_end
// %bb.0:                               // %entry
	nopa	;		movxm	p2, ##_ZL6g_o_lo
	mova	r0, #0;		movxm	r5, #-16512
	mova	r1, #3;		movxm	r7, #3456
	mova	r16, #7;		paddxm	 [sp], #128
	mova	r2, #9;		st	 r8, [sp, #-32];		movxm	ls, #.LBB5_1 // 4-byte Folded Spill
	mova	r3, #16;		st	 r9, [sp, #-28];		movxm	le, #.L_LEnd8 // 4-byte Folded Spill
	mova	r17, #6;		st	 r10, [sp, #-24];		movx	r4, #64;		vbcst.16	 x0, r5 // 4-byte Folded Spill
	mova	r18, #1;		st	 r11, [sp, #-20];		movx	r6, #576;		add.nc	lc, r17, #0 // 4-byte Folded Spill
	mova	r5, #828;		st	 r12, [sp, #-16];		movx	r19, #0;		mov	p1, sp // 4-byte Folded Spill
	padda	 [p1], #-128;		st	 r13, [sp, #-12];		movx	r17, #2;		vmov	x1, x0 // 4-byte Folded Spill
.LBB5_1:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		lshl	 r20, r19, r17;		nopm	;		nopv	
	movs	m0, r20;		movxm	p3, ##_ZL5o_sum
	paddb	 [p3], m0
	vlda	 bmhl2, [p3], #64
	vlda	 bmhh2, [p3], #64
	vlda	 bmhl4, [p3, #0]
	vlda	 bmhh4, [p3, #64]
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cmh2
	vconv.bf16.fp32	 x8, cmh4
	vconv.fp32.bf16	cmh3, x2
	vconv.fp32.bf16	cml0, x8
	vmov	x4, bmhh3
	vmov	x6, bmhl3
	vextract.64	 r29:r28, x6, #4, vaddsign1
	vextract.64	 r21:r20, x6, #0, vaddsign1
	inv	 r28, r28;		vextract.64	 r27:r26, x6, #3, vaddsign1
	inv	 r20, r20;		vextract.64	 r23:r22, x6, #1, vaddsign1
	inv	 r21, r21;		vextract.64	 r31:r30, x6, #5, vaddsign1
	inv	 r22, r22;		vextract.64	 r9:r8, x4, #3, vaddsign1
	inv	 r23, r23;		vextract.64	 r25:r24, x6, #2, vaddsign1
	inv	 r29, r29;		vextract.64	 r11:r10, x4, #6, vaddsign1
	inv	 r24, r24;		vpush.hi.64	 x10, x0, r21:r20
	inv	 r25, r25;		vextract.64	 r21:r20, x6, #6, vaddsign1
	inv	 r30, r30;		vpush.hi.64	 x3, x10, r23:r22
	inv	 r26, r26;		vextract.64	 r23:r22, x6, #7, vaddsign1
	inv	 r27, r27;		vmov	x6, bmll0
	inv	 r31, r31;		vpush.hi.64	 x5, x3, r25:r24
	inv	 r20, r20;		vextract.64	 r25:r24, x4, #0, vaddsign1
	inv	 r21, r21;		vextract.64	 r13:r12, x6, #7, vaddsign1
	inv	 r22, r22;		vpush.hi.64	 x7, x5, r27:r26
	inv	 r23, r23;		vextract.64	 r27:r26, x4, #1, vaddsign1
	inv	 r24, r24;		vpush.hi.64	 x9, x7, r29:r28
	inv	 r25, r25;		vextract.64	 r29:r28, x4, #2, vaddsign1
	inv	 r26, r26;		vpush.hi.64	 x11, x9, r31:r30
	inv	 r20, r8;		vpush.hi.64	 x11, x11, r21:r20
	inv	 r30, r28;		vextract.64	 r29:r28, x4, #4, vaddsign1
	inv	 r31, r29;		vextract.64	 r9:r8, x4, #5, vaddsign1
	inv	 r21, r9;		vpush.hi.64	 x11, x11, r23:r22
	inv	 r27, r27;		vpush.hi.64	 x2, x0, r25:r24
	inv	 r22, r28;		vextract.64	 r29:r28, x4, #7, vaddsign1
	inv	 r23, r29;		vmov	x4, bmlh0
	inv	 r24, r8;		vextract.64	 r9:r8, x6, #1, vaddsign1
	inv	 r25, r9;		vmov	bmhh0, x11
	inv	 r26, r10;		vpush.hi.64	 x2, x2, r27:r26
	inv	 r27, r11;		vpush.hi.64	 x2, x2, r31:r30
	inv	 r28, r28;		vextract.64	 r31:r30, x6, #0, vaddsign1
	inv	 r29, r29;		vpush.hi.64	 x2, x2, r21:r20
	inv	 r22, r8;		vpush.hi.64	 x2, x2, r23:r22
	inv	 r20, r30;		vextract.64	 r31:r30, x6, #2, vaddsign1
	inv	 r21, r31;		vextract.64	 r9:r8, x6, #3, vaddsign1
	inv	 r23, r9;		vpush.hi.64	 x2, x2, r25:r24
	vpush.hi.64	 x2, x2, r27:r26
	inv	 r24, r30;		vextract.64	 r27:r26, x6, #4, vaddsign1
	inv	 r25, r31;		vpush.hi.64	 x2, x2, r29:r28
	inv	 r30, r8;		vextract.64	 r29:r28, x6, #5, vaddsign1
	inv	 r31, r9;		vpush.hi.64	 x8, x0, r21:r20
	inv	 r8, r26;		vextract.64	 r27:r26, x6, #6, vaddsign1
	inv	 r9, r27;		vextract.64	 r21:r20, x4, #0, vaddsign1
	inv	 r10, r28;		vmov	bmhh1, x2
	inv	 r11, r29;		vpush.hi.64	 x8, x8, r23:r22
	inv	 r28, r26;		vextract.64	 r23:r22, x4, #1, vaddsign1
	inv	 r29, r27;		vpush.hi.64	 x8, x8, r25:r24
	inv	 r26, r12;		vextract.64	 r25:r24, x4, #2, vaddsign1
	inv	 r27, r13;		vpush.hi.64	 x8, x8, r31:r30
	inv	 r20, r20;		vextract.64	 r31:r30, x4, #3, vaddsign1
	inv	 r21, r21;		vpush.hi.64	 x8, x8, r9:r8
	inv	 r22, r22;		vextract.64	 r9:r8, x4, #4, vaddsign1
	inv	 r23, r23;		vpush.hi.64	 x8, x8, r11:r10
	inv	 r24, r24;		vextract.64	 r11:r10, x4, #7, vaddsign1
	inv	 r25, r25;		vpush.hi.64	 x8, x8, r29:r28
	inv	 r28, r30;		vpush.hi.64	 x10, x0, r21:r20
	inv	 r29, r31;		vextract.64	 r31:r30, x4, #5, vaddsign1
	inv	 r26, r8;		vpush.hi.64	 x10, x10, r23:r22
	vconv.bf16.fp32	 wl3, bmhh0;		inv	 r27, r9;		vpush.hi.64	 x8, x8, r27:r26
	vconv.bf16.fp32	 wh3, bmhh1;		inv	 r30, r30;		vextract.64	 r9:r8, x4, #6, vaddsign1
	inv	 r31, r31;		vpush.hi.64	 x10, x10, r25:r24
	inv	 r20, r8;		vextract.16	 r24, x3, #8, vaddsign1
	inv	 r21, r9;		vmov	bmlh1, x8
	inv	 r22, r10;		vpush.hi.64	 x10, x10, r29:r28
	inv	 r23, r11;		vpush.hi.64	 x10, x10, r27:r26
	vpush.hi.64	 x10, x10, r31:r30
	vpush.hi.64	 x10, x10, r21:r20
	lshl	 r21, r16, r18;		vextract.16	 r20, x3, #0, vaddsign1
	lshl	 r22, r24, r3;		vpush.hi.64	 x10, x10, r23:r22
	add	r26, r21, #-14;		vmov	bmll1, x10
	vconv.bf16.fp32	 wl5, bmlh1;		extend.u16	 r20, r20;		vextract.16	 r23, x3, #16, vaddsign1
	vconv.bf16.fp32	 wh5, bmll1;		or	 r20, r20, r22;		vextract.16	 r22, x3, #24, vaddsign1
	extend.u16	 r23, r23
	lshl	 r22, r22, r3;		vextract.16	 r24, x5, #0, vaddsign1
	or	 r22, r23, r22;		vextract.16	 r23, x5, #8, vaddsign1
	extend.u16	 r24, r24;		vextract.16	 r25, x5, #16, vaddsign1
	lshl	 r23, r23, r3
	extend.u16	 r25, r25;		vextract.16	 r24, x5, #24, vaddsign1
	or	 r23, r24, r23
	lshl	 r24, r24, r3
	movs	dj0, r26;		or	 r24, r25, r24
	st	 r20, [p1, dj0];		add	r25, r21, #-10
	movs	dj0, r25
	st	 r22, [p1, dj0];		add	r20, r21, #-6
	movs	dj0, r20
	st	 r23, [p1, dj0];		add	r21, r21, #-2
	movs	dj0, r21;		add	 r19, r19, r4
.L_LEnd8:
	nopa	;		nopb	;		st	 r24, [p1, dj0];		add	r16, r16, #8;		nopm	;		nopv	
// %bb.2:                               // %for.cond15.preheader
	mova	r17, #2;		nopb	;		movx	r16, #1;		nopm	
	mova	r19, #0;		movxm	r18, #4187392
	lda	 p2, [p2, #0];		movx	r21, #0;		mov	r20, #0
.LBB5_3:                                // %for.body18
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_4 Depth 2
	nopa	;		nopb	;		lshl	 r22, r19, r16
	movs	p3, p1;		mov	m0, r22
	padda	 [p3], m0
	lda.s16	 r22, [p3], #2
	lda.s16	 r23, [p3], #2
	lda.s16	 r24, [p3], #2;		lshl	 r25, r20, r17
	lda.s16	 r26, [p3], #2;		movxm	p4, ##_ZL4o_hi
	lda.s16	 r28, [p3], #2;		movs	p5, p2;		add	 r27, r25, r18;		mov	m0, r25
	lda.s16	 r25, [p3], #2;		paddb	 [p5], m0;		mov	m0, r27
	lda.s16	 r22, [p3, #0];		paddb	 [p4], m0;		mov	r29, p5
	ltu	 r27, r21, r1;		mov	r30, p4
	sel.nez	 r27, r29, r30, r27
	vbcst.16	 x2, r22
	vbcst.16	 x4, r23
	vshift	x2, x0, x2, r3
	vshift	x2, x2, x4, r3
	lshl	 r22, r0, r17;		vbcst.16	 x4, r22
	lda.s16	 r23, [p3, #2];		movs	p4, r27;		mov	m0, r22
	padda	 [p4], m0
	vlda	 bmll0, [p4], #64;		vbcst.16	 x6, r24
	vlda	 bmlh0, [p4], #64;		vbcst.16	 x8, r26
	vlda	 bmhl0, [p4, #0];		vshift	x2, x2, x6, r3
	vlda	 bmhh0, [p4, #64];		vbcst.16	 x10, r28
	vshift	x2, x2, x8, r3
	vbcst.16	 x3, r25
	vshift	x8, x0, x10, r3
	vshift	x8, x8, x3, r3
	vbcst.16	 x6, r23
	vconv.bf16.fp32	 x6, cml0;		vshift	x4, x8, x4, r3
	vconv.bf16.fp32	 x7, cmh0;		vshift	x3, x4, x6, r3
	nop	
	vmac.f	dm0, dm0, y3, y0, r5
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x4, cml0;		vmul.f	dm1, y3, y1, r5
	vconv.bf16.fp32	 x5, cmh0
	nop	
	movxm	ls, #.LBB5_4;		vmac.f	dm2, dm1, y2, y1, r5
	movxm	le, #.L_LEnd7
	add	 r22, r20, r0
	lshl	 r22, r22, r16;		add.nc	lc, r2, #-1
	movs	m0, r22;		mov	p4, p0
	padda	 [p4], m0;		add	 r22, r0, r4;		mov	p3, r27
.LBB5_4:                                // %for.body27
                                        //   Parent Loop BB5_3 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	vst.conv.bf16.fp32	 cml2, [p4, #0];		nopb	;		nopx	
	vst.conv.bf16.fp32	 cmh2, [p4, #64];		lshl	 r23, r22, r17
	movs	p4, p3;		mov	m0, r23
	padda	 [p4], m0
	vlda	 bmll4, [p4], #64
	vlda	 bmlh4, [p4], #64
	vlda	 bmhh4, [p4, #64]
	vlda	 bmhl4, [p4, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x8, cml4
	vconv.bf16.fp32	 x9, cmh4
	nop	
	vmac.f	dm0, dm4, y4, y0, r5
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x11, cmh0;		vmul.f	dm1, y4, y1, r5
	vconv.bf16.fp32	 x10, cml0
	nop	
	vmac.f	dm2, dm1, y5, y1, r5
	nop	
	add	 r23, r20, r22
	lshl	 r23, r23, r16
	movs	m0, r23;		mov	p4, p0
.L_LEnd7:
	padda	 [p4], m0;		nopb	;		nops	;		add	 r22, r22, r4;		nopm	;		nopv	
// %bb.5:                               //   in Loop: Header=BB5_3 Depth=1
	nopa	;		add	 r20, r20, r6;		nopm	
	eq	 r22, r20, r7
	jz	 r22, #.LBB5_3
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml2, [p4, #0];		add	r21, r21, #1 //  Delay Slot 2
	vst.conv.bf16.fp32	 cmh2, [p4, #64];		add	r19, r19, #8 //  Delay Slot 1
// %bb.6:                               // %for.cond.cleanup17
	lda	 r13, [sp, #-12]                // 4-byte Folded Reload
	lda	 r12, [sp, #-16]                // 4-byte Folded Reload
	lda	 r11, [sp, #-20]                // 4-byte Folded Reload
	lda	 r10, [sp, #-24]                // 4-byte Folded Reload
	lda	 r9, [sp, #-28]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-32]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-128                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end5:
	.size	attn_end, .Lfunc_end5-attn_end
                                        // -- End function
	.type	_ZL6g_init,@object              // @_ZL6g_init
	.section	.bss._ZL6g_init,"aw",@nobits
	.p2align	2, 0x0
_ZL6g_init:
	.byte	0                               // 0x0
	.size	_ZL6g_init, 1

	.type	_ZL8ones_grp,@object            // @_ZL8ones_grp
	.section	.bss._ZL8ones_grp,"aw",@nobits
	.p2align	6, 0x0
_ZL8ones_grp:
	.space	432
	.size	_ZL8ones_grp, 432

	.type	_ZL9mask_full,@object           // @_ZL9mask_full
	.section	.bss._ZL9mask_full,"aw",@nobits
	.p2align	6, 0x0
_ZL9mask_full:
	.space	128
	.size	_ZL9mask_full, 128

	.type	_ZL9mask_last,@object           // @_ZL9mask_last
	.section	.bss._ZL9mask_last,"aw",@nobits
	.p2align	6, 0x0
_ZL9mask_last:
	.space	768
	.size	_ZL9mask_last, 768

	.type	_ZL7g_scale,@object             // @_ZL7g_scale
	.section	.bss._ZL7g_scale,"aw",@nobits
	.p2align	1, 0x0
_ZL7g_scale:
	.short	0x0000                          // bfloat 0
	.size	_ZL7g_scale, 2

	.type	_ZL6g_o_lo,@object              // @_ZL6g_o_lo
	.section	.bss._ZL6g_o_lo,"aw",@nobits
	.p2align	2, 0x0
_ZL6g_o_lo:
	.word	0
	.size	_ZL6g_o_lo, 4

	.type	_ZL6g_maxb,@object              // @_ZL6g_maxb
	.section	.bss._ZL6g_maxb,"aw",@nobits
	.p2align	6, 0x0
_ZL6g_maxb:
	.space	192
	.size	_ZL6g_maxb, 192

	.type	_ZL4o_hi,@object                // @_ZL4o_hi
	.section	.bss._ZL4o_hi,"aw",@nobits
	.p2align	6, 0x0
_ZL4o_hi:
	.space	6912
	.size	_ZL4o_hi, 6912

	.type	_ZL5o_sum,@object               // @_ZL5o_sum
	.section	.bss._ZL5o_sum,"aw",@nobits
	.p2align	6, 0x0
_ZL5o_sum:
	.space	1536
	.size	_ZL5o_sum, 1536

	.type	_ZL7g_chunk,@object             // @_ZL7g_chunk
	.section	.bss._ZL7g_chunk,"aw",@nobits
	.p2align	2, 0x0
_ZL7g_chunk:
	.word	0                               // 0x0
	.size	_ZL7g_chunk, 4

	.type	_ZL4s_rb,@object                // @_ZL4s_rb
	.section	.bss._ZL4s_rb,"aw",@nobits
	.p2align	6, 0x0
_ZL4s_rb:
	.space	768
	.size	_ZL4s_rb, 768

	.type	_ZL4e_rb,@object                // @_ZL4e_rb
	.section	.bss._ZL4e_rb,"aw",@nobits
	.p2align	6, 0x0
_ZL4e_rb:
	.space	2688
	.size	_ZL4e_rb, 2688

	.ident	"clang version 22.0.0 (https://github.com/Xilinx/llvm-aie 0006955eb0bd0a9033185a5be72b621dce63ce21)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZL8ones_grp
	.addrsig_sym _ZL4e_rb
