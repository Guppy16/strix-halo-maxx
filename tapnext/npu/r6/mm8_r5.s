	.file	"mm8.cc"
	.section	.text.acc_zero,"ax",@progbits
	.globl	acc_zero                        // -- Begin function acc_zero
	.p2align	4
	.type	acc_zero,@function
acc_zero:                               // @acc_zero
// %bb.0:                               // %entry
	nopa	;		movxm	p2, ##_ZL6acc_hi
	movxm	p1, ##_ZL8g_stripe
	movxm	ls, #.LBB0_1
	mova	r2, #96;		movxm	le, #.L_LEnd1
	mova	r2, #2;		nopb	;		nops	;		movx	r0, #0;		add.nc	lc, r2, #-2;		nopv	
	nopa	;		nopb	;		nops	;		lshl	 r3, r0, r2;		vbcst.32	 x0, r0;		nopv	
	nopa	;		nopb	;		movs	p3, p0;		add	r3, r0, #32;		mov	m0, r3;		nopv	
	padda	 [p3], m0;		nopb	;		nops	;		lshl	 r4, r3, r2;		nopm	;		nopv	
	mova	r1, #192;		nopb	;		vst	 x0, [p3, #64];		add	r3, r3, #32;		mov	m0, r4;		nopv	
.LBB0_1:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		vst	 x0, [p3, #0];		nopx	;		mov	p3, p0;		nopv	
	padda	 [p3], m0;		nopb	;		nops	;		lshl	 r4, r3, r2;		nopm	;		nopv	
.L_LEnd1:
	nopa	;		nopb	;		vst	 x0, [p3, #64];		add	r3, r3, #32;		mov	m0, r4;		nopv	
// %bb.2:
	nopa	;		nopx	;		add.nc	lc, r1, #-2
	movxm	ls, #.LBB0_3
	mova	r1, #2;		nopb	;		nops	;		movxm	le, #.L_LEnd0;		nopv	
	padda	 [p0], m0;		nopb	;		vst	 x0, [p3, #0];		lshl	 r2, r0, r1;		nopm	;		nopv	
	nopa	;		nopb	;		vst	 x0, [p0, #64];		add	r2, r0, #32;		mov	m0, r2;		nopv	
	nopa	;		nopb	;		vst	 x0, [p0, #0];		movxm	p0, ##_ZL6acc_hi;		nopv	
	padda	 [p0], m0;		nopb	;		nops	;		lshl	 r3, r2, r1;		nopm	;		nopv	
	nopa	;		nopb	;		vst	 x0, [p0, #64];		add	r2, r2, #32;		mov	m0, r3;		nopv	
.LBB0_3:                                // %for.body5
                                        // =>This Inner Loop Header: Depth=1
	vst	 x0, [p0, #0];		movxm	p0, ##_ZL6acc_hi
	padda	 [p0], m0;		lshl	 r3, r2, r1
.L_LEnd0:
	nopa	;		nopb	;		vst	 x0, [p0, #64];		add	r2, r2, #32;		mov	m0, r3;		nopv	
// %bb.4:
	nopa	;		nopb	;		ret	lr;		nopm	
	vst	 x0, [p0, #0]                   //  Delay Slot 5
	padda	 [p2], m0;		st	 r0, [p1, #0] //  Delay Slot 4
	vst	 x0, [p2, #64]                  //  Delay Slot 3
	vst	 x0, [p2, #0]                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end0:
	.size	acc_zero, .Lfunc_end0-acc_zero
                                        // -- End function
	.section	.text.mm_acc,"ax",@progbits
	.globl	mm_acc                          // -- Begin function mm_acc
	.p2align	4
	.type	mm_acc,@function
mm_acc:                                 // @mm_acc
// %bb.0:                               // %entry
	movxm	p3, ##_ZL8g_stripe
	paddxm	 [sp], #64
	lda	 r16, [p3, #0];		nopb	;		event	#0;		st	 p6, [sp, #-64] // 4-byte Folded Spill
	nop	
	nop	
	nop	
	mova	r4, #576;		movx	r3, #48
	mova	r7, #128;		movx	r6, #780;		mov	r5, #384
	movxm	r17, #4182016
	mova	r0, #8;		add	r1, r16, #1;		mov	r21, #1
	mova	r2, #0;		eq	 r27, r16, r0;		mov	r19, #9
	mova	r1, #4;		movs	p4, p0;		sel.nez	 r18, r2, r1, r27;		mov	r24, #0
	mova	r18, #2;		st	 r18, [p3, #0];		lshl	 r16, r16, r1;		mov	r2, #3
.LBB1_1:                                // %for.body
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB1_2 Depth 2
	nopa	;		nopb	;		lshl	 r20, r24, r2;		nopm	;		nops	
	movxm	p3, ##_ZL6acc_hi
	mul	r22, r24, r4
	or	 r20, r20, r16
	lshl	 r23, r20, r0
	movs	p5, p2;		add	 r24, r23, r17;		mov	m0, r23
	padda	 [p5], m0;		lt	 r27, r20, r3;		mov	m0, r24
	padda	 [p3], m0;		movs	p0, p4;		movx	r23, #0;		mov	r20, p5
	movs	m0, r22;		or	 r24, r23, r23;		mov	r22, p3
	padda	 [p0], m0;		sel.nez	 r20, r20, r22, r27
	vldb.fill.512	 [p0, lf0, r24];		mov	p5, r20
	vlda	 bmhh2, [p5], #64;		vldb.pop.576	 ex0, [p0, lf0, r24]
	vlda	 bmhl2, [p5], #64;		vldb.pop.576	 ex2, [p0, lf0, r24]
	vlda	 bmlh2, [p5], #64;		vldb.pop.576	 ex4, [p0, lf0, r24]
	vlda	 bmll2, [p5], #64;		vldb.pop.576	 ex6, [p0, lf0, r24]
	vldb.pop.576	 ex8, [p0, lf0, r24]
	vldb.pop.576	 ex10, [p0, lf0, r24]
	vldb.pop.576	 ex1, [p0, lf0, r24]
	movs	p3, r20;		vldb.pop.576	 ex3, [p0, lf0, r24];		mov	r20, r21
	mova	r25, #0;		or	 r24, r23, r23;		mov	p0, p1
	vldb.fill.512	 [p0, lf0, r24];		lshl	 r21, r25, r18;		vmov	bmll1, bmhh2
	movs	m0, r21;		vldb.pop.576	 ex5, [p0, lf0, r24];		ne	 r21, r25, r5;		vmov	bmlh1, bmhl2
	vlda	 bmhh3, [p5], #64;		vldb.pop.576	 ex7, [p0, lf0, r24];		lshl	 r21, r21, r19;		vmov	bmhl1, bmlh2
	vlda	 bmhl3, [p5], #64;		vldb.pop.576	 ex9, [p0, lf0, r24];		add	 r25, r25, r7;		vmov	bmhh1, bmll2
	vlda	 bmlh3, [p5, #0];		vldb.pop.576	 ex11, [p0, lf0, r24]
	vlda	 bmll3, [p5, #64];		vldb.pop.576	 ex5, [p0, lf0, r24]
	vldb.pop.576	 ex7, [p0, lf0, r24]
	vldb.pop.576	 ex9, [p0, lf0, r24]
	nop	
	vldb.pop.576	 ex11, [p0, lf0, r24];		vmac.f	dm0, dm1, ex0, ex5, r6
	vldb.fill.512	 [p0, lf0, r24]
	vldb.pop.576	 ex5, [p0, lf0, r24];		vmov	bmll4, bmhh3
	vldb.pop.576	 ex7, [p0, lf0, r24];		vmov	bmlh4, bmhl3;		vmac.f	dm0, dm0, ex2, ex9, r6
	vldb.pop.576	 ex9, [p0, lf0, r24];		vmov	bmhl4, bmlh3;		vmac.f	dm4, dm4, ex0, ex7, r6
	vmov	bmhh4, bmll3
	vldb.pop.576	 ex11, [p0, lf0, r24];		mov	p5, p3;		vmac.f	dm0, dm0, ex4, ex5, r6
	padds	 [p5], m0;		vldb.pop.576	 ex5, [p0, lf0, r24];		mov	m0, r21;		vmac.f	dm4, dm4, ex2, ex11, r6
	movs	p6, p5
	padda	 [p6], m0;		vldb.pop.576	 ex7, [p0, lf0, r24];		vmac.f	dm0, dm0, ex6, ex9, r6
	vlda	 bmhh2, [p6], #64;		vldb.pop.576	 ex9, [p0, lf0, r24];		vmac.f	dm4, dm4, ex4, ex7, r6
	vlda	 bmhl2, [p6], #64
	vlda	 bmlh2, [p6], #64;		vldb.pop.576	 ex11, [p0, lf0, r24];		vmac.f	dm0, dm0, ex8, ex5, r6
	vlda	 bmll2, [p6], #64;		vmac.f	dm4, dm4, ex6, ex11, r6
	nop	
	vmac.f	dm0, dm0, ex10, ex9, r6
	add.nc	lc, r1, #-1;		vmac.f	dm4, dm4, ex8, ex7, r6
	movxm	ls, #.LBB1_2
	vlda	 bmhh3, [p6], #64;		movxm	le, #.L_LEnd2;		vmac.f	dm0, dm0, ex1, ex5, r6
.LBB1_2:                                // %_ZN3aie41block_vector_restrict_input_buffer_streamI9bfp16ebs8Lj64EL15aie_dm_resource6EE3popEv.exit135.7
                                        //   Parent Loop BB1_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	vlda	 bmhl3, [p6], #64;		vldb.fill.512	 [p0, lf0, r24];		nops	;		lshl	 r21, r25, r18;		vmov	bmll1, bmhh2;		vmac.f	dm4, dm4, ex10, ex11, r6
	vlda	 bmlh3, [p6, #0];		vldb.pop.576	 ex5, [p0, lf0, r24];		movs	m0, r21;		ne	 r21, r25, r5;		vmov	bmlh1, bmhl2;		nopv	
	vlda	 bmll3, [p6, #64];		vldb.pop.576	 ex7, [p0, lf0, r24];		nops	;		lshl	 r21, r21, r19;		vmov	bmhl1, bmlh2;		vmac.f	dm0, dm0, ex3, ex9, r6
	vldb.pop.576	 ex9, [p0, lf0, r24];		add	 r25, r25, r7;		vmov	bmhh1, bmll2;		vmac.f	dm4, dm4, ex1, ex7, r6
	vldb.pop.576	 ex11, [p0, lf0, r24]
	vldb.pop.576	 ex5, [p0, lf0, r24]
	vldb.pop.576	 ex7, [p0, lf0, r24];		vmac.f	dm4, dm4, ex3, ex11, r6
	vldb.pop.576	 ex9, [p0, lf0, r24]
	vst	 bmll0, [p5], #64
	vst	 bmlh0, [p5], #64;		vldb.pop.576	 ex11, [p0, lf0, r24];		vmac.f	dm0, dm1, ex0, ex5, r6
	vldb.fill.512	 [p0, lf0, r24];		vst	 bmhl0, [p5], #64
	vst	 bmhh0, [p5], #64;		vldb.pop.576	 ex5, [p0, lf0, r24];		vmov	bmll4, bmhh3
	vst	 bmll4, [p5], #64;		vldb.pop.576	 ex7, [p0, lf0, r24];		vmov	bmlh4, bmhl3;		vmac.f	dm0, dm0, ex2, ex9, r6
	vst	 bmlh4, [p5], #64;		vldb.pop.576	 ex9, [p0, lf0, r24];		vmov	bmhl4, bmlh3;		vmac.f	dm4, dm4, ex0, ex7, r6
	vst	 bmhl4, [p5, #0];		vmov	bmhh4, bmll3
	vst	 bmhh4, [p5, #64];		vldb.pop.576	 ex11, [p0, lf0, r24];		mov	p5, p3;		vmac.f	dm0, dm0, ex4, ex5, r6
	padds	 [p5], m0;		vldb.pop.576	 ex5, [p0, lf0, r24];		mov	m0, r21;		vmac.f	dm4, dm4, ex2, ex11, r6
	movs	p6, p5
	padda	 [p6], m0;		vldb.pop.576	 ex7, [p0, lf0, r24];		vmac.f	dm0, dm0, ex6, ex9, r6
	vlda	 bmhh2, [p6], #64;		vldb.pop.576	 ex9, [p0, lf0, r24];		vmac.f	dm4, dm4, ex4, ex7, r6
	vlda	 bmhl2, [p6], #64
	vlda	 bmlh2, [p6], #64;		vldb.pop.576	 ex11, [p0, lf0, r24];		vmac.f	dm0, dm0, ex8, ex5, r6
	vlda	 bmll2, [p6], #64;		vmac.f	dm4, dm4, ex6, ex11, r6
	nop	
	vmac.f	dm0, dm0, ex10, ex9, r6
	vmac.f	dm4, dm4, ex8, ex7, r6
	nop	
.L_LEnd2:
	vlda	 bmhh3, [p6], #64;		nopb	;		nops	;		nopxm	;		vmac.f	dm0, dm0, ex1, ex5, r6
// %bb.3:                               // %for.cond.cleanup6
                                        //   in Loop: Header=BB1_1 Depth=1
	vlda	 bmhl3, [p6], #64;		nopb	;		nops	;		nopxm	;		vmac.f	dm4, dm4, ex10, ex11, r6
	vlda	 bmlh3, [p6, #0];		movx	r24, #1;		mov	r21, #0
	vlda	 bmll3, [p6, #64];		vmac.f	dm0, dm0, ex3, ex9, r6
	vmac.f	dm4, dm4, ex1, ex7, r6
	nop	
	nop	
	vmac.f	dm4, dm4, ex3, ex11, r6
	nop	
	vst	 bmll0, [p5], #64
	vst	 bmlh0, [p5], #64
	vst	 bmhl0, [p5], #64
	vst	 bmhh0, [p5], #64;		jnz	 r20, #.LBB1_1
	vst	 bmll4, [p5], #64               //  Delay Slot 5
	vst	 bmlh4, [p5], #64               //  Delay Slot 4
	vst	 bmhl4, [p5, #0]                //  Delay Slot 3
	vst	 bmhh4, [p5, #64]               //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.4:                               // %for.cond.cleanup
	lda	 p6, [sp, #-64];		event	#1      // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end1:
	.size	mm_acc, .Lfunc_end1-mm_acc
                                        // -- End function
	.section	.text.acc_epilogue,"ax",@progbits
	.globl	acc_epilogue                    // -- Begin function acc_epilogue
	.p2align	4
	.type	acc_epilogue,@function
acc_epilogue:                           // @acc_epilogue
// %bb.0:                               // %entry
	paddxm	 [sp], #1408
	st	 p6, [sp, #-60];		nopx	        // 4-byte Folded Spill
	st	 p7, [sp, #-56];		movxm	r0, #1065353216 // 4-byte Folded Spill
	st	 lr, [sp, #-96];		movxm	dj0, #1024 // 4-byte Folded Spill
	st	 r8, [sp, #-92];		movxm	m0, #-1408 // 4-byte Folded Spill
	st	 r9, [sp, #-88];		vbcst.32	 x1, r0 // 4-byte Folded Spill
	st	 r10, [sp, #-84];		mov	p2, sp // 4-byte Folded Spill
	st	 p0, [sp, #-128];		movxm	r0, #16256 // 4-byte Folded Spill
	st	 r11, [sp, #-80];		mov	p0, sp // 4-byte Folded Spill
	st	 r12, [sp, #-76];		vbcst.16	 x2, r0 // 4-byte Folded Spill
	lda	 r2, [p1, dj0];		st	 r13, [sp, #-72];		movxm	r0, #16177 // 4-byte Folded Spill
	st	 r14, [sp, #-68];		vbcst.16	 x4, r0 // 4-byte Folded Spill
	st	 r15, [sp, #-64];		movxm	r0, #15990 // 4-byte Folded Spill
	mova	m0, #-1024;		paddb	 [p2], m0;		st	 p1, [sp, #-124];		vbcst.16	 x0, r0 // 4-byte Folded Spill
	padda	 [p0], m0;		st	 p2, [sp, #-512];		movxm	r0, #15715 // 4-byte Folded Spill
	movs	p1, p2;		mov	r14, p0
	vst	 x1, [p2, #64];		movx	r3, #2;		mov	p0, sp
	vst	 x1, [p1], #128;		ge	 r0, r2, r3;		vbcst.16	 x2, r0
	padda	 [p0], #-256;		vst	 x2, [p1], #64;		jnz	 r0, #.LBB2_7
	st	 p0, [sp, #-120]                // 4-byte Folded Spill Delay Slot 5
	vst	 x4, [p1], #64;		mov	p6, sp  //  Delay Slot 4
	vst	 x0, [p1, #0];		mov	p0, sp  //  Delay Slot 3
	padda	 [p0], #-192;		vst	 x2, [p1, #64];		mov	p7, sp //  Delay Slot 2
	padda	 [p7], #-320;		paddb	 [p6], #-384;		movx	r1, #144;		mov	crrnd, #12;		st	 p0, [sp, #-116] // 4-byte Folded Spill Delay Slot 1
// %bb.1:                               // %for.cond.preheader
	j	#.LBB2_5
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	mova	r15, #0                         //  Delay Slot 3
	mova	r9, #1;		movx	r12, #64;		mov	r10, #7 //  Delay Slot 2
	mova	r14, #0;		ne	 r11, r2, r9;		mov	r13, #-3 //  Delay Slot 1
.LBB2_2:                                // %for.body
                                        //   in Loop: Header=BB2_5 Depth=1
	lda	 p1, [sp, #-128];		nopxm	 // 4-byte Folded Reload
	jl	#_ZL7blk_srcP8bfloat16ii
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	and	 r8, r14, r10                   //  Delay Slot 2
	lshl	 r0, r14, r13;		mov	r1, r8  //  Delay Slot 1
	lda	 p1, [sp, #-124];		nopb	;		nopx	 // 4-byte Folded Reload
	jl	#_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	p2, p6                          //  Delay Slot 2
	movs	p3, p7;		mov	r0, r8          //  Delay Slot 1
	nopa	;		nopb	;		jnz	 r11, #.LBB2_4;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %if.then4
                                        //   in Loop: Header=BB2_5 Depth=1
	lda	 p2, [sp, #-512]                // 4-byte Folded Reload
	jl	#_ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	p0, p6                          //  Delay Slot 2
	movs	p1, p7                          //  Delay Slot 1
.LBB2_4:                                // %if.end
                                        //   in Loop: Header=BB2_5 Depth=1
	lda	 p0, [sp, #-128];		nopb	;		nopx	 // 4-byte Folded Reload
	vlda	 x0, [sp, #-384]
	vlda	 x2, [sp, #-320]
	nop	
	nop	
	lshl	 r0, r15, r9
	mov	m0, r0
	padda	 [p0], m0
	vst	 x0, [p0, #0];		add	r14, r14, #1
	mova	r1, #144;		vst	 x2, [p0, #64];		add	 r15, r15, r12
.LBB2_5:                                // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		eq	 r0, r14, r1;		nopm	;		nopv	
	jz	 r0, #.LBB2_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB2_6:                                // %if.end199
	lda	 p7, [sp, #-56];		nopx	        // 4-byte Folded Reload
	lda	 p6, [sp, #-60]                 // 4-byte Folded Reload
	lda	 r15, [sp, #-64]                // 4-byte Folded Reload
	lda	 r14, [sp, #-68]                // 4-byte Folded Reload
	lda	 lr, [sp, #-96]                 // 4-byte Folded Reload
	lda	 r13, [sp, #-72]                // 4-byte Folded Reload
	lda	 r12, [sp, #-76]                // 4-byte Folded Reload
	lda	 r11, [sp, #-80]                // 4-byte Folded Reload
	lda	 r10, [sp, #-84]                // 4-byte Folded Reload
	lda	 r9, [sp, #-88]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-92]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-1408                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB2_7:                                // %if.else
	nopa	;		nopb	;		movxm	r0, #2147483646;		nops	
	and	 r0, r2, r0
	ne	 r1, r0, r3
	jnz	 r1, #.LBB2_21
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r10, #4                         //  Delay Slot 1
// %bb.8:                               // %for.cond14.preheader
	j	#.LBB2_19
	nop	                                //  Delay Slot 5
	mova	r0, #0                          //  Delay Slot 4
	mova	r0, #3;		st	 r0, [sp, #-112] // 4-byte Folded Spill Delay Slot 3
	mova	r12, #828;		st	 r2, [sp, #-104];		eq	 r0, r2, r0 // 4-byte Folded Spill Delay Slot 2
	mova	r8, #0;		st	 r0, [sp, #-100];		movx	r6, #0;		mov	r13, #7 // 4-byte Folded Spill Delay Slot 1
.LBB2_9:                                // %for.body30
                                        //   in Loop: Header=BB2_10 Depth=2
	lda	 r15, [sp, #-128];		nopb	;		nopx	 // 4-byte Folded Reload
	nop	
	jl	#_ZL7blk_srcP8bfloat16ii
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	r0, r8                          //  Delay Slot 2
	or	 r1, r9, r9;		mov	p1, r15 //  Delay Slot 1
	lda	 r11, [sp, #-124]               // 4-byte Folded Reload
	nop	
	jl	#_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	p2, p6                          //  Delay Slot 2
	movs	p3, p7;		or	 r0, r9, r9;		mov	p1, r11 //  Delay Slot 1
	nopa	;		nopb	;		jl	#_ZL7blk_srcP8bfloat16ii
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	add	r10, r9, #4                     //  Delay Slot 2
	movs	p1, r15;		or	 r0, r8, r8;		mov	r1, r10 //  Delay Slot 1
	lda	 p2, [sp, #-120]                // 4-byte Folded Reload
	lda	 p3, [sp, #-116]                // 4-byte Folded Reload
	jl	#_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r10, #4;		movs	p1, r11;		mov	r0, r10 //  Delay Slot 1
	lda	 p2, [sp, #-512];		nopb	;		nopxm	 // 4-byte Folded Reload
	jl	#_ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	p0, p6                          //  Delay Slot 2
	movs	p1, p7                          //  Delay Slot 1
	vlda	 x4, [sp, #-320];		nopx	
	vlda	 x0, [sp, #-384]
	vlda	 x2, [sp, #-256]
	vlda	 x6, [sp, #-192]
	nop	
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm0, x0, x2, r12
	vmul.f	dm1, x4, x6, r12
	nop	
	lshl	 r0, r9, r13
	movs	p0, r14;		mov	m0, r0
	padda	 [p0], m0
	vst.conv.bf16.fp32	 cml0, [p0, #0]
	vst.conv.bf16.fp32	 cml1, [p0, #64];		add	r9, r9, #1
.LBB2_10:                               // %for.cond27
                                        //   Parent Loop BB2_19 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ne	 r0, r9, r10
	jnz	 r0, #.LBB2_9
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.11:                              // %for.cond43.preheader
                                        //   in Loop: Header=BB2_19 Depth=1
	lda	 r3, [sp, #-112]                // 4-byte Folded Reload
	lda	 p1, [sp, #-128]                // 4-byte Folded Reload
	lda	 r6, [sp, #-108]                // 4-byte Folded Reload
	nop	
	lda	 r0, [sp, #-104]                // 4-byte Folded Reload
	nop	
	nop	
	mova	r5, #1;		movxm	m0, #12288
	movs	p0, p1;		j	#.LBB2_13
	padda	 [p0], m0;		lshl	 r0, r3, r5;		mov	m0, r6 //  Delay Slot 5
	padda	 [p0], m0;		movx	r1, #2;		mov	m0, r0 //  Delay Slot 4
	padda	 [p1], m0;		eq	 r27, r0, r1;		mov	r0, p0 //  Delay Slot 3
	mova	r4, #144;		movx	r26, #0;		mov	r1, p1 //  Delay Slot 2
	mova	r0, #0;		sel.nez	 r1, r0, r1, r27;		mov	dj1, #288 //  Delay Slot 1
.LBB2_12:                               // %for.body46
                                        //   in Loop: Header=BB2_13 Depth=2
	nopa	;		nopb	;		lshl	 r2, r0, r13;		nopm	;		nops	
	movs	p2, r14;		mov	m0, r2
	padda	 [p2], m0
	vlda.conv.fp32.bf16	 cml0, [p2, #0]
	vlda.conv.fp32.bf16	 cmh0, [p2, #64]
	nop	
	nop	
	nop	
	nop	
	nop	
	mov	p2, r1
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26]
	vst.flush.512.conv	 [p2, sf, r26]
	nop	
	add	r0, r0, #1;		mov	r1, p2
.LBB2_13:                               // %for.cond43
                                        //   Parent Loop BB2_19 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		ne	 r2, r0, r10;		nopm	;		nops	
	jnz	 r2, #.LBB2_12
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.14:                              // %for.cond.cleanup45
                                        //   in Loop: Header=BB2_19 Depth=1
	lda	 r0, [sp, #-100]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	jz	 r0, #.LBB2_18
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.15:                              // %if.then54
                                        //   in Loop: Header=BB2_19 Depth=1
	j	#.LBB2_17
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r0, #0                          //  Delay Slot 1
.LBB2_16:                               // %for.body63
                                        //   in Loop: Header=BB2_17 Depth=2
	nopa	;		nopb	;		lshl	 r1, r0, r5;		nopm	;		nops	
	mov	dj0, r1
	vldb.128	 wh0, [p0, dj0]
	nop	
	nop	
	nop	
	nop	
	movs	p2, p1;		mov	m0, r1
	padda	 [p2], m0
	vst.128	 wh0, [p2, dj1];		add	r0, r0, #8
.LBB2_17:                               // %for.cond60
                                        //   Parent Loop BB2_19 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		nops	;		eq	 r1, r0, r4;		nopm	;		nopv	
	jz	 r1, #.LBB2_16
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB2_18:                               // %if.end76
                                        //   in Loop: Header=BB2_19 Depth=1
	mova	r0, #320;		nopb	;		nops	;		nopxm	;		nopv	
	mova	r0, #288;		add	 r6, r6, r0
	add	 r3, r3, r0
	st	 r3, [sp, #-112];		add	r8, r8, #1 // 4-byte Folded Spill
.LBB2_19:                               // %for.cond14
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB2_10 Depth 2
                                        //     Child Loop BB2_13 Depth 2
                                        //     Child Loop BB2_17 Depth 2
	mova	r0, #18;		nopb	;		nopx	
	eq	 r0, r8, r0
	jnz	 r0, #.LBB2_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.20:                              // %for.body17
                                        //   in Loop: Header=BB2_19 Depth=1
	j	#.LBB2_10
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r9, #0;		st	 r6, [sp, #-108] // 4-byte Folded Spill Delay Slot 1
.LBB2_21:                               // %if.else80
	mova	r9, #6;		ne	 r0, r0, r10;		nopm	
	ne	 r1, r2, r9
	and	 r0, r1, r0
	jnz	 r0, #.LBB2_35
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.22:                              // %if.then86
	lda	 p0, [sp, #-128]                // 4-byte Folded Reload
	nop	
	nop	
	j	#.LBB2_33
	nop	                                //  Delay Slot 5
	mov	r15, r2                         //  Delay Slot 4
	movxm	m0, #12288                      //  Delay Slot 3
	padda	 [p0], m0;		movx	r11, #7;		mov	r13, #0 //  Delay Slot 2
	mova	r8, #0;		movx	r12, #0;		mov	r14, p0 //  Delay Slot 1
.LBB2_23:                               // %for.body97
                                        //   in Loop: Header=BB2_26 Depth=2
	lda	 p1, [sp, #-128];		nopb	;		nopx	 // 4-byte Folded Reload
	jl	#_ZL7blk_srcP8bfloat16ii
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mova	r0, #4;		mov	r1, r9          //  Delay Slot 2
	ne	 r10, r15, r0;		mov	r0, r8  //  Delay Slot 1
	lda	 p1, [sp, #-124];		nopb	;		nopx	 // 4-byte Folded Reload
	jl	#_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	p2, p6                          //  Delay Slot 2
	movs	p3, p7;		mov	r0, r9          //  Delay Slot 1
	nopa	;		nopb	;		jnz	 r10, #.LBB2_25;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.24:                              // %if.then100
                                        //   in Loop: Header=BB2_26 Depth=2
	lda	 p2, [sp, #-512]                // 4-byte Folded Reload
	jl	#_ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	p0, p6                          //  Delay Slot 2
	movs	p1, p7                          //  Delay Slot 1
.LBB2_25:                               // %if.end101
                                        //   in Loop: Header=BB2_26 Depth=2
	vlda	 x0, [sp, #-384];		nopxm	
	vlda	 x2, [sp, #-320]
	nop	
	nop	
	lshl	 r0, r9, r11
	movs	p0, r14;		mov	m0, r0
	padda	 [p0], m0
	vst	 x0, [p0, #0]
	mova	r10, #4;		vst	 x2, [p0, #64];		add	r9, r9, #1
.LBB2_26:                               // %for.cond94
                                        //   Parent Loop BB2_33 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	mova	r0, #8;		nopb	;		nopxm	;		nops	
	ne	 r0, r9, r0
	jnz	 r0, #.LBB2_23
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.27:                              // %for.cond.cleanup96
                                        //   in Loop: Header=BB2_33 Depth=1
	eq	 r1, r15, r10
	jnz	 r1, #.LBB2_29
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mov	r0, r13                         //  Delay Slot 1
// %bb.28:                              // %cond.false114
                                        //   in Loop: Header=BB2_33 Depth=1
	mova	r0, #-1
	lshl	 r0, r12, r0
.LBB2_29:                               // %cond.end124
                                        //   in Loop: Header=BB2_33 Depth=1
	lda	 p2, [sp, #-128];		nopb	;		nopxm	;		nops	 // 4-byte Folded Reload
	nop	
	j	#.LBB2_31
	nop	                                //  Delay Slot 5
	mova	r1, #1                          //  Delay Slot 4
	lshl	 r0, r0, r1                     //  Delay Slot 3
	lda	 r12, [sp, #-120];		mov	m0, r0 // 4-byte Folded Reload Delay Slot 2
	padda	 [p2], m0;		movx	r0, #0;		mov	r26, #0 //  Delay Slot 1
.LBB2_30:                               // %for.body131
                                        //   in Loop: Header=BB2_31 Depth=2
	lshl	 r1, r0, r11
	movs	p0, r14;		mov	m0, r1
	padda	 [p0], m0
	vlda.conv.fp32.bf16	 cml0, [p0, #0]
	vlda.conv.fp32.bf16	 cmh0, [p0, #64]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26]
	vst.flush.512.conv	 [p2, sf, r26]
	add	r0, r0, #1
.LBB2_31:                               // %for.cond128
                                        //   Parent Loop BB2_33 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	mova	r1, #8;		nopb	;		nopxm	
	ne	 r1, r0, r1
	jnz	 r1, #.LBB2_30
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.32:                              // %for.cond.cleanup130
                                        //   in Loop: Header=BB2_33 Depth=1
	mova	r0, #576;		add	r8, r8, #1
	mova	r0, #288;		add	 r12, r12, r0
	add	 r13, r13, r0
.LBB2_33:                               // %for.cond89
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB2_26 Depth 2
                                        //     Child Loop BB2_31 Depth 2
	nopa	;		nopb	;		nops	;		jl	#__udivsi3;		nopv	
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r2, #6;		mov	r1, r8          //  Delay Slot 1
	mova	r1, #18;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		eq	 r1, r8, r1
	jnz	 r1, #.LBB2_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.34:                              // %for.cond94.preheader
                                        //   in Loop: Header=BB2_33 Depth=1
	j	#.LBB2_26
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	movxm	r1, #1152                       //  Delay Slot 3
	mac	r12, r12, r0, r1                //  Delay Slot 2
	st	 r12, [sp, #-120];		movx	r9, #0 // 4-byte Folded Spill Delay Slot 1
.LBB2_35:                               // %if.else141
	lda	 p0, [sp, #-128];		nopx	 // 4-byte Folded Reload
	nop	
	nop	
	j	#.LBB2_45
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	mova	r0, #0;		movxm	m0, #14848      //  Delay Slot 3
	padda	 [p0], m0;		st	 r0, [sp, #-112] // 4-byte Folded Spill Delay Slot 2
	mova	r12, #0;		st	 p0, [sp, #-120];		movx	r13, #53;		mov	r11, #52 // 4-byte Folded Spill Delay Slot 1
.LBB2_36:                               // %for.body160
                                        //   in Loop: Header=BB2_37 Depth=3
	lda	 p1, [sp, #-128];		nopb	;		nopx	 // 4-byte Folded Reload
	jl	#_ZL7blk_srcP8bfloat16ii
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	add	 r0, r12, r14;		mov	r15, p2 //  Delay Slot 2
	or	 r1, r8, r8;		mov	r10, r26 //  Delay Slot 1
	lda	 p1, [sp, #-124];		nopb	;		nopx	 // 4-byte Folded Reload
	jl	#_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	p2, p6                          //  Delay Slot 2
	movs	p3, p7;		mov	r0, r8          //  Delay Slot 1
	vlda	 x0, [sp, #-384];		nopb	;		nopxm	;		nops	
	vlda	 x2, [sp, #-320]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vlda	 sfl, [sp, #-512];		vshuffle	x4, x0, x2, r11 // 64-byte Folded Reload
	vlda	 sfh, [sp, #-448];		vshuffle	x0, x0, x2, r13 // 64-byte Folded Reload
	vconv.fp32.bf16	cml0, x4
	vconv.fp32.bf16	cmh0, x0
	or	 r26, r10, r10;		mov	p2, r15
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26]
	vst.flush.512.conv	 [p2, sf, r26]
	nop	
	nop	
	nop	
	vst	 sfl, [sp, #-512]               // 64-byte Folded Spill
	vst	 sfh, [sp, #-448];		add	r14, r14, #1 // 64-byte Folded Spill
.LBB2_37:                               // %for.cond157
                                        //   Parent Loop BB2_45 Depth=1
                                        //     Parent Loop BB2_39 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		ne	 r0, r14, r9
	jnz	 r0, #.LBB2_36
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.38:                              // %for.cond.cleanup159
                                        //   in Loop: Header=BB2_39 Depth=2
	lda	 r1, [sp, #-116]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	add	r8, r8, #1
	movx	r0, #448
	add	 r1, r1, r0
.LBB2_39:                               // %for.cond148
                                        //   Parent Loop BB2_45 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_37 Depth 3
	mova	r0, #8;		nopxm	
	ne	 r0, r8, r0
	jz	 r0, #.LBB2_41
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.40:                              // %for.body151
                                        //   in Loop: Header=BB2_39 Depth=2
	lda	 p2, [sp, #-120]                // 4-byte Folded Reload
	nop	
	j	#.LBB2_37
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	m0, r1                          //  Delay Slot 2
	padda	 [p2], m0;		st	 r1, [sp, #-116];		movx	r26, #0;		mov	r14, #0 // 4-byte Folded Spill Delay Slot 1
.LBB2_41:                               // %for.cond.cleanup150
                                        //   in Loop: Header=BB2_45 Depth=1
	lda	 r3, [sp, #-112];		nopxm	 // 4-byte Folded Reload
	nop	
	lda	 p0, [sp, #-128]                // 4-byte Folded Reload
	nop	
	j	#.LBB2_43
	mova	r4, #1                          //  Delay Slot 5
	movxm	r2, #1792                       //  Delay Slot 4
	lshl	 r0, r3, r4                     //  Delay Slot 3
	mov	m0, r0                          //  Delay Slot 2
	padda	 [p0], m0;		movx	r0, #0  //  Delay Slot 1
.LBB2_42:                               // %for.body182
                                        //   in Loop: Header=BB2_43 Depth=2
	lda	 p1, [sp, #-120];		nopxm	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	lshl	 r1, r0, r4
	mov	dj0, r1
	vldb	 x0, [p1, dj0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vst	 x0, [p0, dj0];		add	r0, r0, #32
.LBB2_43:                               // %for.cond179
                                        //   Parent Loop BB2_45 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		nops	;		ne	 r1, r0, r2;		nopm	;		nopv	
	jnz	 r1, #.LBB2_42
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.44:                              // %for.cond.cleanup181
                                        //   in Loop: Header=BB2_45 Depth=1
	movxm	r0, #2304
	add	 r3, r3, r0
	st	 r3, [sp, #-112];		add	r12, r12, #6 // 4-byte Folded Spill
.LBB2_45:                               // %for.cond144
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB2_39 Depth 2
                                        //       Child Loop BB2_37 Depth 3
                                        //     Child Loop BB2_43 Depth 2
	mova	r0, #18;		nopb	;		nopx	
	eq	 r0, r12, r0
	jnz	 r0, #.LBB2_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.46:                              // %for.cond148.preheader
                                        //   in Loop: Header=BB2_45 Depth=1
	j	#.LBB2_39
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r8, #0;		movx	r1, #0          //  Delay Slot 1
.Lfunc_end2:
	.size	acc_epilogue, .Lfunc_end2-acc_epilogue
                                        // -- End function
	.section	.text._ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_
	.type	_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_,@function
_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_: // @_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_
// %bb.0:                               // %entry
	mova	r1, #7
	lshl	 r0, r0, r1
	mov	m0, r0
	padda	 [p1], m0
	vldb	 x0, [p1, #0]
	vlda	 bmll0, [p0], #64;		vldb	 x2, [p1, #64]
	vlda	 bmlh0, [p0], #64
	vlda	 bmll1, [p0, #0]
	vlda	 bmlh1, [p0, #64];		movxm	r0, #16256
	vbcst.16	 x4, r0
	mova	r0, #828
	vmac.f	dm0, dm0, x0, x4, r0
	vmac.f	dm1, dm1, x2, x4, r0
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst.conv.bf16.fp32	 cml0, [p2, #0] //  Delay Slot 3
	vst.conv.bf16.fp32	 cml1, [p3, #0] //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end3:
	.size	_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_, .Lfunc_end3-_ZL8blk_biasPKfPK8bfloat16iRN3aie6vectorIS1_Lj32EEES7_
                                        // -- End function
	.section	.text._ZL7blk_srcP8bfloat16ii,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL7blk_srcP8bfloat16ii
	.type	_ZL7blk_srcP8bfloat16ii,@function
_ZL7blk_srcP8bfloat16ii:                // @_ZL7blk_srcP8bfloat16ii
// %bb.0:                               // %entry
	movxm	p0, ##_ZL6acc_hi
	mova	r2, #11;		nopb	;		movxm	r3, #4182016
	lshl	 r2, r0, r2
	add	 r3, r2, r3;		mov	m0, r2
	padda	 [p1], m0;		movx	r4, #6;		mov	m0, r3
	padda	 [p0], m0;		lt	 r27, r0, r4;		mov	r0, p1
	mov	r2, p0
	ret	lr
	mova	r2, #8;		sel.nez	 r0, r0, r2, r27 //  Delay Slot 5
	lshl	 r1, r1, r2                     //  Delay Slot 4
	movs	p0, r0;		mov	m0, r1          //  Delay Slot 3
	padda	 [p0], m0                       //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end4:
	.size	_ZL7blk_srcP8bfloat16ii, .Lfunc_end4-_ZL7blk_srcP8bfloat16ii
                                        // -- End function
	.section	.text._ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE
	.type	_ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE,@function
_ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE: // @_ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE
// %bb.0:                               // %entry
	vldb	 x0, [p0, #0];		nopxm	
	movxm	r1, #-16128
	movxm	r0, #16640
	movxm	r2, #1061962282
	vldb	 x2, [p1, #0];		vbcst.16	 x4, r1
	movxm	r1, #-16512
	vbcst.16	 x8, r0
	vmax_lt.bf16	 x6, r16, x0, x4
	vmin_ge.bf16	 x6, r16, x6, x8
	mova	r0, #828;		vbcst.32	 x10, r2
	movxm	r2, #15634;		vmul.f	dm1, x6, x6, r0
	vmax_lt.bf16	 x4, r16, x2, x4
	vmin_ge.bf16	 x8, r16, x4, x8
	nop	
	vmov	bmll0, x10;		vmul.f	dm2, x8, x8, r0
	vbcst.16	 x10, r2
	vconv.bf16.fp32	 x1, cml1;		vmov	bmlh0, bmll0
	vbcst.16	 x4, r1
	movxm	r1, #14346;		vmac.f	dm3, dm0, x1, x10, r0
	vbcst.16	 x3, r1
	vconv.bf16.fp32	 x1, cml2;		vmac.f	dm1, dm1, x1, x4, r0
	vmac.f	dm3, dm3, x1, x3, r0
	vmac.f	dm2, dm2, x1, x4, r0
	nop	
	vmac.f	dm0, dm0, x1, x10, r0
	nop	
	vconv.bf16.fp32	 x5, cml1
	vmac.f	dm0, dm0, x1, x3, r0
	vconv.bf16.fp32	 x1, cml2;		vmac.f	dm1, dm3, x5, x10, r0
	nop	
	vmac.f	dm0, dm0, x1, x10, r0
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x10, cml1
	nop	
	vconv.bf16.fp32	 x1, cml0;		vmac.f	dm1, dm1, x10, x4, r0
	nop	
	vmac.f	dm0, dm0, x1, x4, r0
	nop	
	nop	
	vmul.f	dm2, x6, x10, r0
	vconv.bf16.fp32	 x10, cml1
	vmul.f	dm0, x8, x1, r0
	vconv.bf16.fp32	 x6, cml0;		vmac.f	dm1, dm2, x6, x10, r0
	nop	
	vmac.f	dm0, dm0, x8, x6, r0
	nop	
	nop	
	movxm	r1, #-16384
	vconv.bf16.fp32	 x8, cml1;		vbcst.16	 x6, r1
	nop	
	vconv.bf16.fp32	 x8, cml0;		vmul.f	dm2, x8, x6, r0
	vmac.f	dm1, dm1, x8, x4, r0
	movxm	r3, #16313;		vmul.f	dm2, x8, x6, r0
	movxm	r2, #17008
	vbcst.16	 x3, r2
	movxm	r1, #-15760
	vconv.bf16.fp32	 x10, cml2;		vbcst.16	 x1, r1
	movxm	r1, #-17620
	vconv.bf16.fp32	 x5, cml2;		vmax_lt.bf16	 x10, r16, x10, x1
	vmin_ge.bf16	 x10, r16, x10, x3
	vconv.bf16.fp32	 x7, cml1;		vmax_lt.bf16	 x1, r16, x5, x1;		vmac.f	dm0, dm0, x8, x4, r0
	vbcst.16	 x5, r3
	vmin_ge.bf16	 x1, r16, x1, x3;		vmul.f	dm1, x7, x6, r0
	vmul.f	dm2, x10, x5, r0
	vbcst.16	 x8, r1;		vmul.f	dm1, x1, x5, r0
	vlda	 bmlh0, [p2, #64]
	vlda	 bmll0, [p2, #0];		vconv.bf16.fp32	 x8, cml0;		vmac.f	dm2, dm2, x10, x8, r0
	padda	 [p2], #128;		vmac.f	dm1, dm1, x1, x8, r0
	vldb	 x6, [p2], #64;		vmul.f	dm3, x8, x6, r0
	vconv.bf16.fp32	 x10, cml1
	nop	
	nop	
	movxm	r1, #19264;		vmac.f	dm2, dm2, x10, x5, r0
	vbcst.16	 x8, r1
	vconv.bf16.fp32	 x10, cml3;		paddxm	 [sp], #768
	movxm	r1, #-13504;		vmac.f	dm3, dm2, x6, x8, r0
	vst	 bmhl0, [sp, #-384];		vbcst.16	 x10, r1;		vmac.f	dm4, dm1, x10, x5, r0 // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-320]             // 64-byte Folded Spill
	vst	 bmll2, [sp, #-256];		vmac.f	dm1, dm3, x6, x10, r0 // 64-byte Folded Spill
	vlda	 bmll0, [sp, #-256];		vst	 bmlh2, [sp, #-192];		vmac.f	dm0, dm4, x6, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh0, [sp, #-192];		vst	 bmhl2, [sp, #-128] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmhl0, [sp, #-128];		vst	 bmhh2, [sp, #-64] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmhh0, [sp, #-64];		vst	 bmll0, [sp, #-512];		vmac.f	dm2, dm0, x6, x10, r0 // 64-byte Folded Reload64-byte Folded Spill
	vst	 bmlh0, [sp, #-448]             // 64-byte Folded Spill
	vconv.bf16.fp32	 x8, cml1
	vst	 bmll0, [sp, #-768]             // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-704];		vmac.f	dm1, dm0, x8, x4, r0 // 64-byte Folded Spill
	vst	 bmhl0, [sp, #-640]             // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-576]             // 64-byte Folded Spill
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x3, cml1
	nop	
	vconv.bf16.fp32	 x8, cml2;		vmul.f	dm1, x3, x3, r0
	nop	
	vlda	 bmll0, [sp, #-512];		vldb	 x10, [p2], #64;		vmac.f	dm4, dm4, x8, x4, r0 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-448]             // 64-byte Folded Reload
	vlda	 bmhl0, [sp, #-384]             // 64-byte Folded Reload
	vlda	 bmhh0, [sp, #-320]             // 64-byte Folded Reload
	vldb	 x8, [p2, #0];		vconv.bf16.fp32	 x5, cml1
	vmov	x7, bmll3
	mova	r1, #2;		vconv.bf16.fp32	 x7, cml4;		vmov	x1, bmlh3;		vmul.f	dm2, x5, x3, r0
	vldb	 x3, [p2, #64];		vshuffle	x1, x7, x1, r1;		vmac.f	dm1, dm0, x3, x10, r0
	vmul.f	dm2, x7, x7, r0
	nop	
	nop	
	vlda	 bmlh1, [sp, #-704];		vmac.f	dm1, dm1, x5, x8, r0 // 64-byte Folded Reload
	vlda	 bmll1, [sp, #-768];		vconv.bf16.fp32	 x5, cml2 // 64-byte Folded Reload
	nop	
	vconv.bf16.fp32	 x5, cml2;		vmac.f	dm3, dm1, x5, x3, r0
	nop	
	mov	crupsmode, #0;		vmul.f	dm1, x5, x7, r0
	mov	crsrsmode, #0
	vmov	x9, bmlh1;		vmac.f	dm2, dm0, x7, x10, r0
	vmov	x11, bmll1
	mova	r1, #7;		vshuffle	x9, x11, x9, r1
	mov	s0, r1;		vmac.f	dm1, dm2, x5, x8, r0
	vconv.bf16.fp32	 x10, cml1;		vups.2x	cml4, x1, s0, upssign1
	mova	r1, #0;		vups.2x	cml2, x9, s0, upssign1
	mov	s1, r1;		vmac.f	dm1, dm1, x10, x3, r0
	vsrs.2x	x8, cml4, s1, srssign1
	vsrs.2x	x10, cml2, s1, srssign1
	movxm	r1, #16256
	vbcst.16	 x1, r1
	vconv.bf16.fp32	 x1, cml3;		vadd.16	 x8, x8, x1
	vconv.bf16.fp32	 x3, cml1;		vadd.16	 x10, x10, x1
	vmul.f	dm1, x1, x8, r0
	vmul.f	dm1, x3, x10, r0
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x8, cml1
	vconv.bf16.fp32	 x10, cml1
	vmac.f	dm1, dm0, x8, x6, r0
	vmac.f	dm2, dm0, x10, x6, r0
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x10, cml1
	vconv.bf16.fp32	 x8, cml2
	vmac.f	dm1, dm1, x10, x4, r0
	movxm	r1, #32499;		vmac.f	dm2, dm2, x8, x4, r0
	vbcst.16	 x3, r1
	vsub.16	 x5, x3, x10
	vsub.16	 x7, x3, x8
	vmsc.f	dm1, dm0, x10, x5, r0
	vconv.bf16.fp32	 x1, cml1;		vmsc.f	dm2, dm0, x8, x7, r0
	vconv.bf16.fp32	 x3, cml2
	vmsc.f	dm1, dm1, x1, x5, r0
	vmsc.f	dm2, dm2, x3, x7, r0
	nop	
	nop	
	nop	
	vmul.f	dm1, x5, x6, r0
	vconv.bf16.fp32	 x9, cml1;		vmul.f	dm2, x7, x6, r0
	vconv.bf16.fp32	 x5, cml2
	vmac.f	dm1, dm1, x5, x9, r0
	vmac.f	dm2, dm2, x7, x5, r0
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x6, cml1
	vconv.bf16.fp32	 x5, cml2
	vmac.f	dm3, dm1, x6, x4, r0
	vmac.f	dm3, dm2, x5, x4, r0
	nop	
	nop	
	nop	
	vmsc.f	dm4, dm0, x10, x6, r0
	vconv.bf16.fp32	 x7, cml3;		vmsc.f	dm0, dm0, x8, x5, r0
	vconv.bf16.fp32	 x10, cml3
	vmsc.f	dm4, dm4, x10, x7, r0
	vmsc.f	dm0, dm0, x8, x10, r0
	nop	
	vmsc.f	dm4, dm4, x1, x6, r0
	vmsc.f	dm0, dm0, x3, x5, r0
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x8, cml4
	vconv.bf16.fp32	 x6, cml0
	vmac.f	dm1, dm1, x6, x8, r0
	vmac.f	dm0, dm2, x5, x6, r0
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x6, cml1
	vconv.bf16.fp32	 x8, cml0
	vmac.f	dm1, dm1, x6, x4, r0
	vmac.f	dm0, dm0, x8, x4, r0
	nop	
	nop	
	nop	
	vmul.f	dm1, x0, x6, r0
	vconv.bf16.fp32	 x4, cml1;		vmul.f	dm0, x2, x8, r0
	vconv.bf16.fp32	 x0, cml0
	vmac.f	dm1, dm1, x0, x4, r0
	vmac.f	dm0, dm0, x2, x0, r0
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst.conv.bf16.fp32	 cml1, [p0, #0] //  Delay Slot 3
	vst.conv.bf16.fp32	 cml0, [p1, #0];		paddxm	 [sp], #-768 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end5:
	.size	_ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE, .Lfunc_end5-_ZN3nk8L8gelu32x2ERN3aie6vectorI8bfloat16Lj32EEES4_RKNS_4ExpCE
                                        // -- End function
	.section	.text.attn_begin,"ax",@progbits
	.globl	attn_begin                      // -- Begin function attn_begin
	.p2align	4
	.type	attn_begin,@function
attn_begin:                             // @attn_begin
// %bb.0:                               // %entry
	movxm	r6, #1045504
	movxm	p2, ##_ZN2atL3g_oE
	movxm	p1, ##_ZN2atL5g_srbE
	movxm	p5, ##_ZN2atL3g_qE
	movxm	m0, #15360
	movxm	m1, #17088
	movxm	m2, #15744
	movxm	m3, #19200
	movxm	r3, #-15744
	mov	crrnd, #12
	mova	r1, #0;		movs	p3, p0;		paddxm	 [sp], #64
	st	 p7, [sp, #-60];		vbcst.32	 x0, r1 // 4-byte Folded Spill
	st	 p6, [sp, #-64];		movxm	p7, ##_ZN2atL6g_etmpE // 4-byte Folded Spill
	padda	 [p3], m0;		st	 p0, [p2, #0];		movxm	p6, ##_ZL6acc_hi
	mova	r2, #1;		st	 p3, [p1, #0];		movxm	m0, #12288
	mova	r20, #95;		movs	p4, p0;		movx	r17, #192;		mov	p3, p0
	padda	 [p0], m1;		movs	p1, p0;		movxm	m1, #21888
	mova	r21, #96;		st	 p0, [p7, #0];		movxm	p0, ##_ZL6acc_hi
	padda	 [p0], m2;		paddb	 [p4], m0;		movxm	m0, #16128;		padds	 [p6], m0
	mova	r19, #768;		st	 p0, [p5, #4];		movxm	p0, ##_ZL6acc_hi
	padda	 [p0], m3;		st	 p6, [p5, #0];		movxm	p5, ##_ZN2atL5g_erbE
	mova	r18, #432;		st	 p0, [p5, #0];		movxm	p0, ##_ZL6acc_hi
	padda	 [p0], m1;		paddb	 [p3], m0;		movxm	m0, #16640
	mova	r4, #127;		st	 p0, [p5, #4];		movxm	p0, ##_ZN2atL6g_osumE
	mova	r5, #8;		st	 p4, [p0, #0];		movxm	p0, ##_ZN2atL6g_maxbE
	mova	r7, #64;		j	#.LBB6_2
	padda	 [p1], m0;		st	 p3, [p0, #0];		movxm	p0, ##_ZN2atL6g_onesE //  Delay Slot 5
	mova	r16, #72;		st	 p1, [p0, #0];		movxm	p0, ##_ZL6acc_hi //  Delay Slot 4
	mova	r22, #0;		st	 p0, [p2, #4];		movxm	p0, ##_ZN2atL8g_ntilesE //  Delay Slot 3
	mova	r23, #0;		st	 r0, [p0, #0];		movxm	p0, ##_ZN2atL7g_chunkE //  Delay Slot 2
	mova	r0, #2;		st	 r1, [p0, #0];		movx	r3, #433;		vbcst.16	 x2, r3 //  Delay Slot 1
.LBB6_1:                                // %for.body
                                        //   in Loop: Header=BB6_2 Depth=1
	nopa	;		nopb	;		ltu	 r24, r20, r23;		nopm	;		nops	
	lshl	 r24, r24, r0
	mov	dj0, r24
	lda	 p0, [p2, dj0]
	nop	
	ltu	 r27, r23, r21
	add	 r24, r22, r6
	sel.nez	 r24, r22, r24, r27
	lshl	 r24, r24, r0
	mov	m0, r24
	padda	 [p0], m0
	vst	 x0, [p0, #0];		add	r23, r23, #1
	vst	 x0, [p0, #64];		add	r22, r22, #32
.LBB6_2:                                // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		ne	 r24, r23, r17;		nopm	
	jnz	 r24, #.LBB6_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:
	j	#.LBB6_5
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r6, #0;		movx	r0, #2          //  Delay Slot 1
.LBB6_4:                                // %for.body11
                                        //   in Loop: Header=BB6_5 Depth=1
	nopa	;		lshl	 r20, r6, r0;		nopm	
	movs	p0, p4;		mov	m0, r20
	padda	 [p0], m0
	vst	 x0, [p0, #0]
	vst	 x0, [p0, #64];		add	r6, r6, #32
.LBB6_5:                                // %for.cond8
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		ne	 r20, r6, r19;		nopm	
	jnz	 r20, #.LBB6_4
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.6:
	j	#.LBB6_8
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r6, #0;		movx	r0, #1          //  Delay Slot 1
.LBB6_7:                                // %for.body24
                                        //   in Loop: Header=BB6_8 Depth=1
	nopa	;		lshl	 r19, r6, r0
	mov	dj0, r19
	vst	 x2, [p3, dj0];		add	r6, r6, #32
.LBB6_8:                                // %for.cond21
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		ne	 r19, r6, r17;		nopm	;		nops	
	jnz	 r19, #.LBB6_7
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.9:
	j	#.LBB6_11
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r0, #0                          //  Delay Slot 1
.LBB6_10:                               // %for.body37
                                        //   in Loop: Header=BB6_11 Depth=1
	nopa	;		nopb	;		nops	;		nopx	;		mov	dj0, r0;		nopv	
	st.s8	 r1, [p1, dj0]
	nop	
	nop	
	nop	
	nop	
	add	r0, r0, #1
.LBB6_11:                               // %for.cond34
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		ne	 r6, r0, r18;		nopm	;		nopv	
	jz	 r6, #.LBB6_15
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
	j	#.LBB6_10
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB6_12:                               // %for.body51
                                        //   in Loop: Header=BB6_13 Depth=2
	nopa	;		nopb	;		add	 r1, r2, r0;		nopm	
	mov	dj0, r1
	st.s8	 r7, [p1, dj0]
	nop	
	nop	
	nop	
	nop	
	add	r0, r0, #1
.LBB6_13:                               // %for.cond48
                                        //   Parent Loop BB6_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		ne	 r1, r0, r5;		nopm	
	jnz	 r1, #.LBB6_12
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.14:                              // %for.cond.cleanup50
                                        //   in Loop: Header=BB6_15 Depth=1
	add	 r2, r2, r16
.LBB6_15:                               // %for.cond42
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_13 Depth 2
	nopa	;		nopb	;		ne	 r0, r2, r3
	jz	 r0, #.LBB6_17
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.16:                              // %for.body45
                                        //   in Loop: Header=BB6_15 Depth=1
	j	#.LBB6_13
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	add	r0, r2, #-1                     //  Delay Slot 3
	mov	dj0, r0                         //  Delay Slot 2
	st.s8	 r4, [p1, dj0];		movx	r0, #0  //  Delay Slot 1
.LBB6_17:                               // %for.cond.cleanup44
	lda	 p7, [sp, #-60];		nopxm	        // 4-byte Folded Reload
	lda	 p6, [sp, #-64]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end6:
	.size	attn_begin, .Lfunc_end6-attn_begin
                                        // -- End function
	.section	.text.attn_qcopy,"ax",@progbits
	.globl	attn_qcopy                      // -- Begin function attn_qcopy
	.p2align	4
	.type	attn_qcopy,@function
attn_qcopy:                             // @attn_qcopy
// %bb.0:                               // %entry
	nopa	;		ne	 r0, r0, r3;		nopm	
	jnz	 r0, #.LBB7_8
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	paddxm	 [sp], #64                      //  Delay Slot 3
	st	 p6, [sp, #-52]                 // 4-byte Folded Spill Delay Slot 2
	st	 p7, [sp, #-48]                 // 4-byte Folded Spill Delay Slot 1
// %bb.1:                               // %if.end
	movxm	p6, ##_ZL6acc_hi
	movxm	p5, ##_ZN2atL5g_erbE
	mov	r6, p2
	movxm	p2, ##_ZL6acc_hi
	movxm	p7, ##_ZN2atL3g_qE
	movxm	p4, ##_ZL6acc_hi
	mov	r27, r2
	eqz	 r3, r2;		mov	r7, p1
	movxm	p3, ##_ZL6acc_hi
	movxm	m0, #12288
	mova	r5, #1;		movxm	p1, ##_ZN2atL3g_oE
	mova	r2, #2;		xor	 r3, r3, r5;		mov	r0, #3
	sel.eqz	 r4, r2, r0, r27
	lshl	 r3, r3, r5
	ge	 r4, r1, r4
	movxm	r5, ##_ZL6acc_hi
	st	 r5, [p1, #4];		lt	 r5, r1, r3
	or	 r4, r5, r4
	jnz	 r4, #.LBB7_8
	padda	 [p3], m0;		movxm	m1, #15744 //  Delay Slot 5
	padda	 [p4], m1;		st	 p3, [p7, #0];		movxm	m2, #19200 //  Delay Slot 4
	padda	 [p6], m2;		st	 p4, [p7, #4];		movxm	m0, #21888 //  Delay Slot 3
	padda	 [p2], m0;		st	 p6, [p5, #0] //  Delay Slot 2
	st	 p2, [p5, #4]                   //  Delay Slot 1
// %bb.2:                               // %if.end7
	mov	p4, sp
	j	#.LBB7_6
	padda	 [p4], #-64                     //  Delay Slot 5
	movs	p2, p0;		sub	 r1, r1, r3;		mov	p0, p4 //  Delay Slot 4
	st	 p2, [p0], #4;		lshl	 r1, r1, r2 //  Delay Slot 3
	mova	r3, #288;		st	 r7, [p0, #0];		mov	dj0, r1 //  Delay Slot 2
	lda	 p3, [p7, dj0];		st	 r6, [p0, #4];		movx	r1, #0;		mov	r4, #0 //  Delay Slot 1
.LBB7_3:                                // %for.body13
                                        //   in Loop: Header=BB7_4 Depth=2
	nopa	;		lshl	 r6, r5, r2
	mov	dj0, r6
	vldb	 x0, [p0, dj0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vst	 x0, [p1, dj0];		add	r5, r5, #16
.LBB7_4:                                // %for.cond10
                                        //   Parent Loop BB7_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		ne	 r6, r5, r3
	jnz	 r6, #.LBB7_3
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.5:                               // %for.cond.cleanup12
                                        //   in Loop: Header=BB7_6 Depth=1
	add	r4, r4, #1
	add	 r1, r1, r3
.LBB7_6:                                // %for.cond
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_4 Depth 2
	nopa	;		nopb	;		eq	 r5, r4, r0;		nopm	;		nops	
	jnz	 r5, #.LBB7_8
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.7:                               // %for.cond10.preheader
                                        //   in Loop: Header=BB7_6 Depth=1
	j	#.LBB7_4
	nop	                                //  Delay Slot 5
	lshl	 r5, r4, r2                     //  Delay Slot 4
	lshl	 r5, r1, r2;		mov	dj0, r5 //  Delay Slot 3
	lda	 p0, [p4, dj0];		movs	p1, p3;		mov	m0, r5 //  Delay Slot 2
	padda	 [p1], m0;		movx	r5, #0  //  Delay Slot 1
.LBB7_8:                                // %cleanup.cont
	lda	 p7, [sp, #-48];		nopxm	        // 4-byte Folded Reload
	lda	 p6, [sp, #-52]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end7:
	.size	attn_qcopy, .Lfunc_end7-attn_qcopy
                                        // -- End function
	.section	.text.attn_chunk_k,"ax",@progbits
	.globl	attn_chunk_k                    // -- Begin function attn_chunk_k
	.p2align	4
	.type	attn_chunk_k,@function
attn_chunk_k:                           // @attn_chunk_k
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		paddxm	 [sp], #1664;		nopv	
	st	 lr, [sp, #-60]                 // 4-byte Folded Spill
	st	 r8, [sp, #-56]                 // 4-byte Folded Spill
	st	 r9, [sp, #-52]                 // 4-byte Folded Spill
	st	 r10, [sp, #-48]                // 4-byte Folded Spill
	st	 r11, [sp, #-44]                // 4-byte Folded Spill
	st	 r12, [sp, #-40]                // 4-byte Folded Spill
	st	 r13, [sp, #-36]                // 4-byte Folded Spill
	st	 r14, [sp, #-32]                // 4-byte Folded Spill
	st	 r15, [sp, #-28]                // 4-byte Folded Spill
	st	 p6, [sp, #-24];		movxm	p1, ##_ZN2atL7g_chunkE // 4-byte Folded Spill
	lda	 r0, [p1, #0];		st	 p7, [sp, #-20];		event	#0 // 4-byte Folded Spill
	nop	
	nop	
	nop	
	nop	
	mova	r1, #-48
	movxm	r2, #1072
	mac	r2, r2, r0, r1
	movx	r4, #0
	lt	 r27, r4, r2
	mova	r3, #47;		sel.nez	 r0, r2, r4, r27
	mova	r1, #-3;		lt	 r27, r3, r2
	mova	r5, #6;		lshl	 r0, r0, r1;		mov	p2, sp
	sel.nez	 r22, r5, r0, r27;		mov	p3, sp
	jz	 r22, #.LBB8_18
	mova	m0, #-832;		mov	p1, sp  //  Delay Slot 5
	mova	m0, #-768;		paddb	 [p1], m0;		mov	p5, sp //  Delay Slot 4
	padda	 [p2], m0;		movxm	m0, #-1344 //  Delay Slot 3
	padda	 [p3], m0;		movxm	m0, #-1664 //  Delay Slot 2
	padda	 [p5], m0;		st	 p3, [sp, #-100] // 4-byte Folded Spill Delay Slot 1
// %bb.1:                               // %if.then
	movs	p7, p0;		movxm	r0, #1065353216
	st	 p5, [sp, #-84];		movxm	p3, ##_ZN2atL8g_ntilesE;		vclr	dm0 // 4-byte Folded Spill
	st	 p2, [sp, #-88];		movxm	p2, ##_ZN2atL6g_maxbE // 4-byte Folded Spill
	lda	 r6, [p2, #0];		st	 p1, [sp, #-92];		movxm	p1, ##_ZN2atL5g_srbE // 4-byte Folded Spill
	lda	 r0, [p3, #0];		st	 r22, [sp, #-124];		vbcst.32	 x0, r0 // 4-byte Folded Spill
	lda	 p1, [p1, #0];		st	 p7, [sp, #-96] // 4-byte Folded Spill
	lda	 p3, [p1, #0];		vst	 x0, [p5, #64]
	vst	 bmll0, [sp, #-512];		movxm	r0, #1060372480 // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-448];		vbcst.32	 x1, r0 // 64-byte Folded Spill
	vst	 bmhl0, [sp, #-384];		movxm	r0, #16256 // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-320];		vbcst.16	 x2, r0 // 64-byte Folded Spill
	st	 r6, [sp, #-76]                 // 4-byte Folded Spill
	st	 p1, [sp, #-72]                 // 4-byte Folded Spill
	st	 r0, [sp, #-64];		mov	p1, p5  // 4-byte Folded Spill
	st	 p3, [sp, #-80];		movxm	r0, #16640 // 4-byte Folded Spill
	vst	 x0, [p1], #128;		vbcst.16	 x6, r0
	vst	 x1, [p1, #0];		movxm	r0, #-16512
	vst	 x1, [p1, #64];		vbcst.16	 x6, r0
	mova	r7, #60;		vst	 x6, [sp, #-640];		movxm	r0, #15872 // 64-byte Folded Spill
	mova	r17, #828;		vst	 x2, [p1, #128];		j	#.LBB8_16
	mova	r0, #0;		vst	 x6, [sp, #-576];		movx	r10, #780;		vbcst.16	 x8, r0 // 64-byte Folded Spill Delay Slot 5
	mova	r18, #128;		st	 r0, [sp, #-68];		movxm	p4, ##_ZN2atL6g_etmpE // 4-byte Folded Spill Delay Slot 4
	mova	r21, #-128;		vst	 x8, [sp, #-256];		movx	r20, #1;		vbcst.16	 x6, r4 // 64-byte Folded Spill Delay Slot 3
	lda	 p6, [p4, #0];		vst	 x8, [sp, #-192];		movxm	r19, #1152 // 64-byte Folded Spill Delay Slot 2
	mova	r4, #-576;		vst	 x6, [sp, #-704];		movx	r9, #7;		mov	r23, #0 // 64-byte Folded Spill Delay Slot 1
.LBB8_2:                                // %for.cond.cleanup14.i
                                        //   in Loop: Header=BB8_5 Depth=3
	nopa	;		nopb	;		nops	;		add	 r2, r0, r4;		nopm	;		nopv	
	mova	r2, #0;		movs	p0, p7;		mov	m0, r2
	padda	 [p0], m0;		movs	p1, p7;		or	 r24, r2, r2;		mov	m0, r0
	padda	 [p1], m0;		vldb.fill.512	 [p0, lf0, r24];		mov	r25, r2
	vlda	 x0, [sp, #-1216];		vldb.fill.512	 [p1, lf1, r25] // 64-byte Folded Reload
	lda	 el0, [sp, #-1152];		vldb.pop.576	 ex10, [p0, lf0, r24] // 4-byte Folded Reload
	lda	 eh0, [sp, #-1148];		vldb.pop.576	 ex10, [p1, lf1, r25] // 4-byte Folded Reload
	vldb.pop.576	 ex11, [p0, lf0, r24]
	vlda	 x0, [sp, #-960];		vldb.pop.576	 ex6, [p0, lf0, r24] // 64-byte Folded Reload
	lda	 el0, [sp, #-896];		vldb.pop.576	 ex9, [p1, lf1, r25] // 4-byte Folded Reload
	lda	 eh0, [sp, #-892];		vldb.pop.576	 ex6, [p1, lf1, r25] // 4-byte Folded Reload
	nop	
	vlda	 x0, [sp, #-1088]               // 64-byte Folded Reload
	lda	 el0, [sp, #-1024];		vmul.f	dm0, ex0, ex10, r10 // 4-byte Folded Reload
	lda	 eh0, [sp, #-1020];		vmov	sfl, x11;		vmul.f	dm1, ex0, ex10, r10 // 4-byte Folded Reload
	nop	
	vldb.pop.576	 ex6, [p0, lf0, r24];		vmov	sfh, x9
	vldb.pop.576	 ex6, [p1, lf1, r25];		vmac.f	dm0, dm0, ex0, ex6, r10
	vldb.pop.576	 ex6, [p0, lf0, r24];		vmac.f	dm2, dm1, ex0, ex6, r10
	vldb.pop.576	 ex6, [p1, lf1, r25]
	vldb.pop.576	 ex6, [p0, lf0, r24]
	vldb.pop.576	 ex6, [p1, lf1, r25];		vmul.f	dm1, ex0, ex11, r10
	vldb.pop.576	 ex6, [p0, lf0, r24];		vmul.f	dm1, ex0, ex9, r10
	vldb.pop.576	 ex9, [p0, lf0, r24]
	vmac.f	dm3, dm1, ex2, ex6, r10
	vldb.pop.576	 ex10, [p1, lf1, r25];		vmac.f	dm4, dm1, ex2, ex6, r10
	vmac.f	dm1, dm0, ex4, ex6, r10
	vldb.pop.576	 ex6, [p1, lf1, r25];		vmac.f	dm0, dm2, ex4, ex6, r10
	vmac.f	dm3, dm3, ex3, ex6, r10
	vmac.f	dm2, dm4, ex3, ex6, r10
	vmac.f	dm1, dm1, ex5, ex6, r10
	vmac.f	dm3, dm3, ex7, ex9, r10
	nop	
	vlda	 x1, [sp, #-192];		vmac.f	dm0, dm0, ex5, ex10, r10 // 64-byte Folded Reload
	vlda	 x0, [sp, #-256];		vadd.f	dm1, dm1, dm3, r7 // 64-byte Folded Reload
	vmac.f	dm2, dm2, ex7, ex6, r10
	nop	
	nop	
	vadd.f	dm0, dm0, dm2, r7
	nop	
	vconv.bf16.fp32	 x10, cml1
	vconv.bf16.fp32	 x11, cmh1
	nop	
	vmul.f	dm1, y5, y0, r17
	vconv.bf16.fp32	 x8, cml0
	vconv.bf16.fp32	 x9, cmh0
	lshl	 r2, r1, r20
	add	 r3, r2, r21;		vmul.f	dm0, y4, y0, r17
	mov	m0, r3
	vconv.bf16.fp32	 x6, cml1;		mov	p1, p3
	padda	 [p1], m0;		vconv.bf16.fp32	 x10, cmh1
	vst	 x6, [p1, #0];		jz	 r5, #.LBB8_4
	vst	 x10, [p1, #64];		mov	m0, r2  //  Delay Slot 5
	vconv.bf16.fp32	 x8, cml0;		mov	p1, p3 //  Delay Slot 4
	padda	 [p1], m0;		vconv.bf16.fp32	 x6, cmh0 //  Delay Slot 3
	vst	 x8, [p1, #0];		vmax_lt.bf16	 x9, r16, x6, x8 //  Delay Slot 2
	vst	 x6, [p1, #64];		vmax_lt.bf16	 x11, r16, x10, x6 //  Delay Slot 1
// %bb.3:                               // %cond.false.i
                                        //   in Loop: Header=BB8_5 Depth=3
	vmov	x0, sfh
	vmax_lt.bf16	 x9, r16, x0, x9
	vmov	x0, sfl
	vmax_lt.bf16	 x11, r16, x0, x11
.LBB8_4:                                // %cond.end.i
                                        //   in Loop: Header=BB8_5 Depth=3
	nopa	;		nopb	;		add	r5, r5, #2
	add	 r1, r1, r18
	add	 r0, r0, r19
.LBB8_5:                                // %for.cond5.i
                                        //   Parent Loop BB8_16 Depth=1
                                        //     Parent Loop BB8_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		nops	;		ltu	 r2, r5, r22;		nopm	;		nopv	
	jnz	 r2, #.LBB8_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.6:                               // %for.cond.cleanup7.i
                                        //   in Loop: Header=BB8_13 Depth=2
	lshl	 r0, r27, r20
	movs	p1, r6;		mov	m0, r0
	padda	 [p1], m0
	vldb	 wl0, [p1, #0]
	vlda	 x2, [sp, #-640]                // 64-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.fp32.bf16	bmll0, wl0
	vconv.fp32.bf16	bmll1, wl2;		vadd.f	dm1, dm2, dm1, r7
	vmov	bmll2, bmll0
	nop	
	nop	
	vlda	 bmll2, [sp, #-512]             // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-448];		movx	r0, #52;		mov	dj0, r0 // 64-byte Folded Reload
	vlda	 bmhl2, [sp, #-384];		vconv.bf16.fp32	 wl4, bmll1;		movx	r0, #53;		vshuffle	x2, x9, x11, r0 // 64-byte Folded Reload
	vlda	 bmhh2, [sp, #-320];		vshuffle	x6, x9, x11, r0 // 64-byte Folded Reload
	vconv.fp32.bf16	cml1, x4
	vmov	wl4, wh2
	vmax_lt.bf16	 x2, r16, x2, x4
	vmov	wl4, wh6;		vadd.f	dm1, dm1, dm2, r7
	vmax_lt.bf16	 x4, r16, x6, x4
	vmax_lt.bf16	 x2, r16, x2, x4
	vmov	wh4, wl2
	mova	r0, #16;		vmov	wl4, wl2
	vshift	x4, x4, x0, r0
	vconv.bf16.fp32	 x4, cml1;		vmax_lt.bf16	 x2, r16, x2, x4
	vmax_lt.bf16	 x2, r16, x2, x0
	vlt.bf16	 r16, x4, x2
	nop	
	extend.u16	 r0, r16
	jz	 r0, #.LBB8_8
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	st	 r27, [sp, #-960];		mov	p7, p3 // 4-byte Folded Spill Delay Slot 1
// %bb.7:                               // %if.then82.i
                                        //   in Loop: Header=BB8_13 Depth=2
	vlda	 x0, [sp, #-576]                // 64-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	lda	 r0, [sp, #-112];		vmac.f	dm0, dm0, x2, x0, r17 // 4-byte Folded Reload
	lda	 r1, [sp, #-128]                // 4-byte Folded Reload
	jl	#_ZN2atL10rescale_rbEiiN3aie6vectorI8bfloat16Lj16EEE
	nop	                                //  Delay Slot 5
	mov	p0, r6                          //  Delay Slot 4
	vst	 wl2, [p0, dj0]                 //  Delay Slot 3
	vconv.bf16.fp32	 wl0, bmll0             //  Delay Slot 2
	mov	p7, p1                          //  Delay Slot 1
	lda	 p7, [sp, #-72];		nopb	;		nops	;		nopx	;		mov	p1, p7;		nopv	 // 4-byte Folded Reload
.LBB8_8:                                // %if.end87.i
                                        //   in Loop: Header=BB8_13 Depth=2
	lda	 p0, [sp, #-100];		nopb	;		nopxm	 // 4-byte Folded Reload
	nop	
	jl	#_ZN2atL5rows8EPK8bfloat16
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mov	r8, p0                          //  Delay Slot 1
	vlda	 x0, [sp, #-1344];		nopb	;		nopx	;		mov	p0, r8
	vldb	 x0, [p0, #64]
	nop	
	nop	
	nop	
	lda	 r11, [sp, #-92]                // 4-byte Folded Reload
	lda	 r15, [sp, #-88]                // 4-byte Folded Reload
	lda	 r14, [sp, #-84];		vst	 x0, [sp, #-1216] // 4-byte Folded Reload64-byte Folded Spill
	lda	 r12, [sp, #-124];		vst	 x0, [sp, #-1088];		movx	r8, #0;		mov	r13, #0 // 4-byte Folded Reload64-byte Folded Spill
.LBB8_9:                                // %for.body.i.i
                                        //   Parent Loop BB8_16 Depth=1
                                        //     Parent Loop BB8_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	mova	r0, #1;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		lshl	 r0, r13, r0
	movs	p0, p7;		mov	m0, r0
	padda	 [p0], m0
	vldb	 x1, [p0, #0]
	vldb	 x3, [p0, #64]
	vlda	 x0, [sp, #-1216]               // 64-byte Folded Reload
	vlda	 x2, [sp, #-1088]               // 64-byte Folded Reload
	jl	#_ZN3nk8L8expd32x2ERN3aie6vectorI8bfloat16Lj32EEES4_S3_S3_RKNS_4ExpQE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst	 x1, [sp, #-832]                //  Delay Slot 3
	vst	 x3, [sp, #-768];		mov	p1, r15 //  Delay Slot 2
	movs	p2, r14;		mov	p0, r11         //  Delay Slot 1
	vlda	 x5, [sp, #-832];		nopxm	
	vlda	 x4, [sp, #-704];		ltu	 r0, r8, r12 // 64-byte Folded Reload
	vlda	 x7, [sp, #-768];		add	r16, r0, #-1
	lshl	 r0, r8, r9
	mova	r0, #6;		add	r8, r8, #1;		mov	m0, r0
	eq	 r0, r8, r0
	jz	 r0, #.LBB8_9
	nop	                                //  Delay Slot 5
	movs	p0, p6;		vsel.32	 x9, x5, x4, r16 //  Delay Slot 4
	padda	 [p0], m0;		vsel.32	 x11, x7, x4, r16 //  Delay Slot 3
	mova	r3, #64;		vst	 x9, [p0, #0]   //  Delay Slot 2
	vst	 x11, [p0, #64];		add	 r13, r13, r3 //  Delay Slot 1
// %bb.10:                              // %for.cond.cleanup.i.i
                                        //   in Loop: Header=BB8_13 Depth=2
	lda	 dj0, [sp, #-108]               // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	lda	 r3, [sp, #-120];		movxm	p0, ##_ZN2atL5g_erbE // 4-byte Folded Reload
	lda	 p2, [p0, dj0]
	nop	
	nop	
	nop	
	mova	r0, #6;		movx	r2, #7
	lda	 r23, [sp, #-112];		movx	r26, #0;		add.nc	lc, r0, #-1 // 4-byte Folded Reload
	lda	 p7, [sp, #-96];		lshl	 r0, r26, r2;		mov	m0, r3 // 4-byte Folded Reload
	padda	 [p2], m0;		movs	p0, p6;		mov	m0, r0
	padda	 [p0], m0
	vlda.conv.fp32.bf16	 cml1, [p0, #0]
	vlda.conv.fp32.bf16	 cmh1, [p0, #64];		movx	r7, #60;		mov	r5, #6
	lda	 p3, [sp, #-80];		movx	r4, #-576;		mov	r17, #828 // 4-byte Folded Reload
	lda	 r6, [sp, #-76];		movx	r20, #1;		mov	r18, #128 // 4-byte Folded Reload
	lda	 r22, [sp, #-124];		movxm	ls, #.LBB8_11 // 4-byte Folded Reload
	lda	 r27, [sp, #-960];		nopb	;		nops	;		movxm	le, #.L_LEnd3;		nopv	 // 4-byte Folded Reload
	lda	 r16, [sp, #-116];		nopb	;		nops	;		movxm	r19, #1152;		nopv	 // 4-byte Folded Reload
	lda	 r25, [sp, #-128];		nopb	;		nops	;		add	r0, r26, #1;		mov	r21, #-128;		nopv	 // 4-byte Folded Reload
.LBB8_11:                               // %for.body15.i.i
                                        //   Parent Loop BB8_16 Depth=1
                                        //     Parent Loop BB8_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		nopxm	;		nopv	
	nopa	;		nopb	;		nopxm	;		vst.flush.512.conv	 [p2, sf, r26]
	lshl	 r1, r0, r2
	movs	p1, p6;		mov	m0, r1
	padda	 [p1], m0
	vlda.conv.fp32.bf16	 cmh1, [p1, #64]
	vlda.conv.fp32.bf16	 cml1, [p1, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
.L_LEnd3:
	nopa	;		nopb	;		nops	;		add	r0, r0, #1;		nopm	;		nopv	
// %bb.12:                              //   in Loop: Header=BB8_13 Depth=2
	nopa	;		add	r25, r25, #1;		nopm	
	mova	r0, #576;		add	r27, r27, #16
	mova	r0, #448;		vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		add	 r16, r16, r0
	vst.flush.512.conv	 [p2, sf, r26];		add	 r3, r3, r0
.LBB8_13:                               // %for.cond.i
                                        //   Parent Loop BB8_16 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB8_5 Depth 3
                                        //       Child Loop BB8_9 Depth 3
                                        //       Child Loop BB8_11 Depth 3
	nopa	;		eq	 r0, r25, r5;		nopm	
	jnz	 r0, #.LBB8_15
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.14:                              // %for.cond5.preheader.i
                                        //   in Loop: Header=BB8_13 Depth=2
	lda	 p1, [sp, #-104]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	mov	m0, r16
	padda	 [p1], m0;		st	 r25, [sp, #-128];		movx	r25, #0 // 4-byte Folded Spill
	vldb.fill.512	 [p1, lf1, r25]
	vldb.pop.576	 ex0, [p1, lf1, r25]
	nop	
	nop	
	vldb.pop.576	 ex0, [p1, lf1, r25]
	nop	
	nop	
	vldb.pop.576	 ex0, [p1, lf1, r25];		st	 r3, [sp, #-120] // 4-byte Folded Spill
	st	 r16, [sp, #-116]               // 4-byte Folded Spill
	vst	 x0, [sp, #-1216]               // 64-byte Folded Spill
	st	 el0, [sp, #-1152]              // 4-byte Folded Spill
	st	 eh0, [sp, #-1148]              // 4-byte Folded Spill
	vst	 x0, [sp, #-1088];		j	#.LBB8_5 // 64-byte Folded Spill
	vldb.pop.576	 ex2, [p1, lf1, r25];		st	 el0, [sp, #-1024] // 4-byte Folded Spill Delay Slot 5
	vldb.pop.576	 ex4, [p1, lf1, r25];		st	 eh0, [sp, #-1020] // 4-byte Folded Spill Delay Slot 4
	vldb.pop.576	 ex3, [p1, lf1, r25];		vst	 x0, [sp, #-960] // 64-byte Folded Spill Delay Slot 3
	vldb.pop.576	 ex5, [p1, lf1, r25];		st	 el0, [sp, #-896] // 4-byte Folded Spill Delay Slot 2
	mova	r5, #0;		vldb.pop.576	 ex7, [p1, lf1, r25];		movx	r1, #64;		mov	r0, #576;		st	 eh0, [sp, #-892] // 4-byte Folded Spill Delay Slot 1
.LBB8_15:                               // %_ZN2atL7chunk_kEiPK9bfp16ebs8S2_iRKN3nk84ExpQE.exit
                                        //   in Loop: Header=BB8_16 Depth=1
	lda	 r1, [sp, #-68];		nopb	;		nopx	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	movx	r0, #96
	add	 r1, r1, r0
	st	 r1, [sp, #-68];		add	r23, r23, #1 // 4-byte Folded Spill
.LBB8_16:                               // %for.cond
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_13 Depth 2
                                        //       Child Loop BB8_5 Depth 3
                                        //       Child Loop BB8_9 Depth 3
                                        //       Child Loop BB8_11 Depth 3
	lda	 r0, [sp, #-64];		nopb	;		nopx	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	lt	 r0, r23, r0
	jz	 r0, #.LBB8_18
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.17:                              // %for.body
                                        //   in Loop: Header=BB8_16 Depth=1
	mova	r0, #2;		movxm	p1, ##_ZN2atL3g_qE
	lshl	 r0, r23, r0
	mov	dj0, r0
	lda	 p1, [p1, dj0]
	nop	
	j	#.LBB8_13
	lda	 r27, [sp, #-68]                // 4-byte Folded Reload Delay Slot 5
	nop	                                //  Delay Slot 4
	st	 r23, [sp, #-112]               // 4-byte Folded Spill Delay Slot 3
	st	 r0, [sp, #-108]                // 4-byte Folded Spill Delay Slot 2
	mova	r25, #0;		st	 p1, [sp, #-104];		movx	r16, #0;		mov	r3, #0 // 4-byte Folded Spill Delay Slot 1
.LBB8_18:                               // %if.end
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
	paddxm	 [sp], #-1664                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end8:
	.size	attn_chunk_k, .Lfunc_end8-attn_chunk_k
                                        // -- End function
	.section	.text.attn_chunk_v,"ax",@progbits
	.globl	attn_chunk_v                    // -- Begin function attn_chunk_v
	.p2align	4
	.type	attn_chunk_v,@function
attn_chunk_v:                           // @attn_chunk_v
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	p1, ##_ZN2atL7g_chunkE;		nops	
	lda	 r0, [p1, #0]
	nop	
	nop	
	nop	
	nop	
	mova	r1, #-48
	movxm	r2, #1072
	mac	r2, r2, r0, r1
	nop	
	mova	r1, #0;		paddxm	 [sp], #64
	st	 r8, [sp, #-64];		ge	 r2, r1, r2 // 4-byte Folded Spill
	st	 r9, [sp, #-60];		jnz	 r2, #.LBB9_10 // 4-byte Folded Spill
	st	 r10, [sp, #-56]                // 4-byte Folded Spill Delay Slot 5
	st	 r11, [sp, #-52]                // 4-byte Folded Spill Delay Slot 4
	st	 r12, [sp, #-48]                // 4-byte Folded Spill Delay Slot 3
	st	 p6, [sp, #-44]                 // 4-byte Folded Spill Delay Slot 2
	mova	r1, #24;		st	 p7, [sp, #-40] // 4-byte Folded Spill Delay Slot 1
// %bb.1:                               // %for.cond.preheader
	movs	p3, p0;		movxm	p0, ##_ZN2atL8g_ntilesE
	lda	 r2, [p0, #0];		movx	r4, #8
	mova	r7, #448;		movx	r6, #60;		mov	r5, #780
	mova	r18, #-448;		movx	r17, #64;		mov	r16, #512
	mova	r21, #384;		movx	r20, #896;		mov	r19, #128
	mova	r22, #2;		j	#.LBB9_8
	mova	r3, #0;		movxm	p1, ##_ZN2atL6g_onesE //  Delay Slot 5
	movxm	p0, ##_ZN2atL6g_osumE           //  Delay Slot 4
	mova	r31, #0;		lt	 r27, r3, r2;		mov	r23, #256 //  Delay Slot 3
	lda	 r30, [p1, #0];		sel.nez	 r2, r2, r3, r27;		mov	r8, #0 //  Delay Slot 2
	lda	 r29, [p0, #0];		movxm	r3, #3072 //  Delay Slot 1
.LBB9_2:                                // %for.cond.cleanup26.i
                                        //   in Loop: Header=BB9_3 Depth=3
	nopa	;		nopb	;		nops	;		lshl	 r24, r25, r22;		nopm	;		nopv	
	add	 r28, r24, r23;		mov	p1, p3
	movs	p6, p2;		add	 r24, r27, r18;		mov	m0, r24
	mova	r24, #0;		paddb	 [p6], m0;		movs	p0, p3;		mov	m0, r24
	vlda	 bmll0, [p6, #0];		paddb	 [p1], m0;		or	 r11, r25, r25;		mov	r25, r24;		movs	m0, r27
	padda	 [p0], m0;		vldb.fill.512	 [p1, lf1, r25]
	vlda	 bmlh0, [p6, #64];		vldb.fill.512	 [p0, lf0, r24]
	vlda	 bmhl0, [p6, #128];		vldb.pop.576	 ex3, [p1, lf1, r25]
	vlda	 bmhh0, [p6, #192];		vldb.pop.576	 ex5, [p0, lf0, r24];		movs	p5, p2;		mov	m0, r28
	padda	 [p5], m0;		vldb.pop.576	 ex7, [p1, lf1, r25]
	vlda	 bmll1, [p5, #0];		vldb.pop.576	 ex1, [p0, lf0, r24]
	vlda	 bmlh1, [p5, #64];		vldb.pop.576	 ex3, [p1, lf1, r25]
	vlda	 bmhl1, [p5, #128];		vldb.pop.576	 ex5, [p0, lf0, r24]
	vlda	 bmhh1, [p5, #192];		vldb.pop.576	 ex7, [p1, lf1, r25]
	nop	
	vldb.pop.576	 ex5, [p1, lf1, r25]
	vldb.pop.576	 ex9, [p1, lf1, r25];		vmac.f	dm0, dm0, ex10, ex3, r5
	vldb.pop.576	 ex3, [p0, lf0, r24];		vmac.f	dm1, dm1, ex10, ex5, r5
	vldb.pop.576	 ex7, [p0, lf0, r24];		vmul.f	dm2, ex8, ex7, r5
	vmac.f	dm0, dm0, ex2, ex3, r5
	vldb.pop.576	 ex5, [p0, lf0, r24];		vmac.f	dm1, dm1, ex2, ex5, r5
	vmac.f	dm2, dm2, ex0, ex7, r5
	vmul.f	dm0, ex8, ex1, r5
	vmac.f	dm3, dm0, ex4, ex5, r5
	vmac.f	dm2, dm2, ex6, ex9, r5
	vmac.f	dm4, dm0, ex0, ex3, r5
	vmac.f	dm0, dm1, ex4, ex7, r5
	vadd.f	dm1, dm3, dm2, r6
	vmac.f	dm2, dm4, ex6, ex5, r5
	nop	
	nop	
	vadd.f	dm0, dm0, dm2, r6
	nop	
	vst	 bmll1, [p6, #0]
	vst	 bmlh1, [p6, #64]
	vst	 bmhl1, [p6, #128]
	vst	 bmhh1, [p6, #192]
	vst	 bmll0, [p5, #0]
	vst	 bmlh0, [p5, #64];		add	 r27, r27, r20
	vst	 bmhl0, [p5, #128];		add	 r25, r11, r19
	vst	 bmhh0, [p5, #192];		add	r24, r12, #2
.LBB9_3:                                // %for.cond6.i
                                        //   Parent Loop BB9_8 Depth=1
                                        //     Parent Loop BB9_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		nops	;		geu	 r24, r24, r4;		mov	r12, r24;		nopv	
	jz	 r24, #.LBB9_2
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.4:                               // %for.cond.cleanup86.i
                                        //   in Loop: Header=BB9_5 Depth=2
	mova	r25, #0;		lshl	 r24, r26, r22;		mov	p1, r30
	movs	p0, r29;		vldb.fill.512	 [p1, lf1, r25];		mov	m0, r24
	padda	 [p0], m0;		vldb.pop.576	 ex1, [p1, lf1, r25]
	vlda	 bmll0, [p0, #0];		vldb.pop.576	 ex10, [p1, lf1, r25]
	vlda	 bmlh0, [p0, #64]
	vlda	 bmhl0, [p0, #128];		vldb.pop.576	 ex1, [p1, lf1, r25]
	vlda	 bmhh0, [p0, #192];		vldb.pop.576	 ex8, [p1, lf1, r25]
	nop	
	vldb.pop.576	 ex10, [p1, lf1, r25]
	vldb.pop.576	 ex2, [p1, lf1, r25]
	vmac.f	dm0, dm0, ex10, ex1, r5
	vmul.f	dm1, ex8, ex10, r5
	nop	
	vmac.f	dm0, dm0, ex2, ex1, r5
	vmac.f	dm1, dm1, ex0, ex8, r5
	nop	
	vmac.f	dm0, dm0, ex4, ex10, r5
	vmac.f	dm1, dm1, ex6, ex2, r5
	nop	
	nop	
	vadd.f	dm0, dm0, dm1, r6
	nop	
	nop	
	nop	
	nop	
	nop	
	vst	 bmll0, [p0, #0]
	vst	 bmlh0, [p0, #64];		add	 r10, r10, r7
	vst	 bmhl0, [p0, #128];		add	 r26, r26, r17
	vst	 bmhh0, [p0, #192];		add	 r25, r9, r16
.LBB9_5:                                // %for.cond.i
                                        //   Parent Loop BB9_8 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB9_3 Depth 3
	nopa	;		eq	 r24, r25, r3
	jnz	 r24, #.LBB9_7
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.6:                               // %for.cond6.preheader.i
                                        //   in Loop: Header=BB9_5 Depth=2
	movs	p0, p7;		mov	m0, r10
	padda	 [p0], m0;		movx	r24, #0
	vldb.fill.512	 [p0, lf0, r24]
	vldb.pop.576	 ex10, [p0, lf0, r24]
	vldb.pop.576	 ex8, [p0, lf0, r24]
	vldb.pop.576	 ex2, [p0, lf0, r24]
	vldb.pop.576	 ex0, [p0, lf0, r24];		j	#.LBB9_3
	vldb.pop.576	 ex4, [p0, lf0, r24]    //  Delay Slot 5
	vldb.pop.576	 ex6, [p0, lf0, r24]    //  Delay Slot 4
	lshl	 r24, r25, r22                  //  Delay Slot 3
	mova	r27, #448;		movs	p2, p4;		or	 r9, r25, r25;		mov	m0, r24 //  Delay Slot 2
	padda	 [p2], m0;		movx	r24, #0;		mov	r25, #0 //  Delay Slot 1
.LBB9_7:                                // %_ZN2atL7chunk_vEiPK9bfp16ebs8.exit
                                        //   in Loop: Header=BB9_8 Depth=1
	nopa	;		nopb	;		add	r8, r8, #1;		nopm	
	add	 r31, r31, r21
.LBB9_8:                                // %for.cond
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_5 Depth 2
                                        //       Child Loop BB9_3 Depth 3
	nopa	;		eq	 r24, r2, r8
	jnz	 r24, #.LBB9_10
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.9:                               // %for.body
                                        //   in Loop: Header=BB9_8 Depth=1
	j	#.LBB9_5
	lshl	 r24, r8, r22                   //  Delay Slot 5
	movxm	p0, ##_ZN2atL3g_oE              //  Delay Slot 4
	or	 r26, r31, r31;		mov	dj0, r24 //  Delay Slot 3
	lda	 p4, [p0, dj0];		movxm	p0, ##_ZN2atL5g_erbE //  Delay Slot 2
	lda	 p7, [p0, dj0];		movx	r10, #0;		mov	r25, #0 //  Delay Slot 1
.LBB9_10:                               // %if.end
	lda	 p7, [sp, #-40];		nopx	        // 4-byte Folded Reload
	lda	 r12, [sp, #-48]                // 4-byte Folded Reload
	lda	 r11, [sp, #-52]                // 4-byte Folded Reload
	lda	 p6, [sp, #-44]                 // 4-byte Folded Reload
	lda	 r10, [sp, #-56]                // 4-byte Folded Reload
	lda	 r9, [sp, #-60]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-64];		add	r0, r0, #1 // 4-byte Folded Reload
	ret	lr
	movxm	p0, ##_ZN2atL7g_chunkE          //  Delay Slot 5
	mova	r1, #0;		eq	 r27, r0, r1    //  Delay Slot 4
	sel.nez	 r0, r1, r0, r27                //  Delay Slot 3
	st	 r0, [p0, #0];		paddxm	 [sp], #-64 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end9:
	.size	attn_chunk_v, .Lfunc_end9-attn_chunk_v
                                        // -- End function
	.section	.text.attn_end,"ax",@progbits
	.globl	attn_end                        // -- Begin function attn_end
	.p2align	4
	.type	attn_end,@function
attn_end:                               // @attn_end
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		paddxm	 [sp], #512;		nopv	
	st	 r11, [sp, #-56]                // 4-byte Folded Spill
	st	 r12, [sp, #-52];		movxm	p1, ##_ZN2atL8g_ntilesE // 4-byte Folded Spill
	lda	 r0, [p1, #0];		st	 r13, [sp, #-48] // 4-byte Folded Spill
	st	 r14, [sp, #-44]                // 4-byte Folded Spill
	st	 r9, [sp, #-64]                 // 4-byte Folded Spill
	st	 r8, [sp, #-68]                 // 4-byte Folded Spill
	st	 r10, [sp, #-60];		movxm	p1, ##_ZN2atL6g_osumE // 4-byte Folded Spill
	lda	 p1, [p1, #0];		st	 p7, [sp, #-32] // 4-byte Folded Spill
	st	 lr, [sp, #-72];		movx	r1, #0;		mov	p7, sp // 4-byte Folded Spill
	st	 p0, [sp, #-96];		lt	 r27, r1, r0 // 4-byte Folded Spill
	st	 r15, [sp, #-40];		sel.nez	 r0, r0, r1, r27;		mov	p0, sp // 4-byte Folded Spill
	st	 p6, [sp, #-36];		j	#.LBB10_10 // 4-byte Folded Spill
	padda	 [p0], #-384;		st	 r0, [sp, #-76];		movxm	r0, #-16512 // 4-byte Folded Spill Delay Slot 5
	mova	r2, #0;		st	 p0, [sp, #-92];		vbcst.16	 x4, r0 // 4-byte Folded Spill Delay Slot 4
	mova	r13, #828;		st	 p1, [sp, #-100];		movx	r12, #512;		mov	r11, #8 // 4-byte Folded Spill Delay Slot 3
	mova	r9, #2;		vst	 x4, [sp, #-256];		movx	r14, #64;		mov	p1, sp // 64-byte Folded Spill Delay Slot 2
	padda	 [p1], #-512;		paddb	 [p7], #-128;		movx	r10, #0;		mov	r8, #1;		vst	 x4, [sp, #-192] // 64-byte Folded Spill Delay Slot 1
.LBB10_1:                               // %for.body12
                                        //   in Loop: Header=BB10_2 Depth=3
	nopa	;		nopb	;		nops	;		lshl	 r2, r0, r8;		nopm	;		nopv	
	lshl	 r3, r1, r8;		mov	dj0, r2
	lda.s16	 r2, [p1, dj0];		mov	dj0, r3
	st.s16	 r2, [p7, dj0];		nopx	
	nop	
	nop	
	nop	
	add	r1, r1, #1
	add	r0, r0, #8
.LBB10_2:                               // %for.cond9
                                        //   Parent Loop BB10_10 Depth=1
                                        //     Parent Loop BB10_7 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		nops	;		ne	 r2, r1, r11;		nopm	;		nopv	
	jnz	 r2, #.LBB10_1
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %for.cond.cleanup11
                                        //   in Loop: Header=BB10_7 Depth=2
	lda	 p6, [sp, #-92]                 // 4-byte Folded Reload
	nop	
	jl	#_ZN2atL5rows8EPK8bfloat16
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	r10, p1                         //  Delay Slot 2
	movs	p1, p7;		mov	p0, p6          //  Delay Slot 1
	lda	 dj0, [sp, #-104];		nopxm	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	lda	 r2, [sp, #-108];		movxm	p0, ##_ZN2atL3g_oE // 4-byte Folded Reload
	lda	 p0, [p0, dj0]
	lda	 p2, [sp, #-96]                 // 4-byte Folded Reload
	nop	
	j	#.LBB10_5
	vlda	 x0, [sp, #-384]                //  Delay Slot 5
	vlda	 x4, [sp, #-256];		lshl	 r0, r15, r9 // 64-byte Folded Reload Delay Slot 4
	vlda	 x5, [sp, #-192];		mov	m0, r0 // 64-byte Folded Reload Delay Slot 3
	padda	 [p0], m0;		mov	m0, r2  //  Delay Slot 2
	padda	 [p2], m0;		vldb	 x1, [p6, #64];		movx	r0, #0;		mov	r26, #0 //  Delay Slot 1
.LBB10_4:                               // %for.body26
                                        //   in Loop: Header=BB10_5 Depth=3
	nopa	;		nopb	;		lshl	 r1, r0, r9
	movs	p1, p0;		mov	m0, r1
	padda	 [p1], m0
	vlda	 bmll0, [p1], #64
	vlda	 bmlh0, [p1], #64
	vlda	 bmhl0, [p1, #0]
	vlda	 bmhh0, [p1, #64]
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml0
	vconv.bf16.fp32	 x3, cmh0
	nop	
	vmac.f	dm0, dm0, y1, y2, r13
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml0;		vmul.f	dm1, y1, y0, r13
	vconv.bf16.fp32	 x3, cmh0
	nop	
	vmac.f	dm0, dm1, y1, y0, r13
	nop	
	nop	
	nop	
	nop	
	nop	
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26]
	vst.flush.512.conv	 [p2, sf, r26]
	nop	
	add	 r0, r0, r14
.LBB10_5:                               // %for.cond23
                                        //   Parent Loop BB10_10 Depth=1
                                        //     Parent Loop BB10_7 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ne	 r1, r0, r12
	jnz	 r1, #.LBB10_4
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.6:                               // %for.cond.cleanup25
                                        //   in Loop: Header=BB10_7 Depth=2
	lda	 r1, [sp, #-112]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	mova	r0, #576;		add	 r15, r15, r12
	add	 r2, r2, r0
	add	 r1, r1, r14;		mov	p1, r10
.LBB10_7:                               // %for.cond1
                                        //   Parent Loop BB10_10 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_2 Depth 3
                                        //       Child Loop BB10_5 Depth 3
	nopa	;		movxm	r0, #3072
	ne	 r0, r15, r0
	jz	 r0, #.LBB10_9
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.8:                               // %for.body4
                                        //   in Loop: Header=BB10_7 Depth=2
	lda	 p0, [sp, #-100]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	lshl	 r0, r1, r9
	mov	m0, r0
	padda	 [p0], m0
	vlda	 bmll0, [p0], #64
	vlda	 bmlh0, [p0], #64
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vconv.fp32.bf16	cml0, x0
	nop	
	vmov	x0, bmll0
	vextract.64	 r1:r0, x0, #0, vaddsign1
	vlda	 bmll0, [p0, #0];		st	 r1, [sp, #-112] // 4-byte Folded Spill
	vlda	 bmlh0, [p0, #64];		inv	 r4, r0;		vextract.64	 r3:r2, x0, #1, vaddsign1
	st	 r2, [sp, #-108];		inv	 r5, r1;		vmov	x2, bmlh0 // 4-byte Folded Spill
	inv	 r6, r2;		vextract.64	 r3:r2, x0, #3, vaddsign1
	inv	 r7, r3;		vextract.64	 r21:r20, x0, #6, vaddsign1
	inv	 r18, r2;		vextract.64	 r1:r0, x0, #2, vaddsign1
	inv	 r19, r3;		vextract.64	 r23:r22, x0, #7, vaddsign1
	inv	 r16, r0;		vextract.64	 r1:r0, x0, #4, vaddsign1
	vconv.bf16.fp32	 x4, cml0;		inv	 r17, r1;		vextract.64	 r25:r24, x2, #6, vaddsign1
	inv	 r0, r0;		vextract.64	 r3:r2, x0, #5, vaddsign1
	inv	 r1, r1;		vconv.fp32.bf16	cml0, x4
	inv	 r2, r2;		vpush.hi.64	 x4, x0, r5:r4
	inv	 r4, r20;		vextract.64	 r21:r20, x2, #0, vaddsign1
	inv	 r5, r21;		vpush.hi.64	 x0, x4, r7:r6
	inv	 r6, r22;		vextract.64	 r23:r22, x2, #1, vaddsign1
	inv	 r7, r23;		vpush.hi.64	 x0, x0, r17:r16
	inv	 r3, r3;		vmov	x6, bmll0
	inv	 r16, r20;		vpush.hi.64	 x0, x0, r19:r18
	inv	 r17, r21;		vextract.64	 r19:r18, x2, #2, vaddsign1
	inv	 r20, r22;		vextract.64	 r23:r22, x2, #3, vaddsign1
	inv	 r21, r23;		vpush.hi.64	 x0, x0, r1:r0
	inv	 r2, r22;		vpush.hi.64	 x0, x0, r3:r2
	inv	 r0, r18;		vextract.64	 r19:r18, x2, #4, vaddsign1
	inv	 r1, r19;		vextract.64	 r23:r22, x2, #5, vaddsign1
	inv	 r3, r23;		vpush.hi.64	 x0, x0, r5:r4
	vpush.hi.64	 x0, x0, r7:r6
	inv	 r4, r18;		vextract.64	 r19:r18, x2, #7, vaddsign1
	inv	 r5, r19;		vmov	x2, bmlh0
	inv	 r6, r22;		vextract.64	 r23:r22, x6, #1, vaddsign1
	inv	 r7, r23;		vpush.hi.64	 x4, x0, r17:r16
	inv	 r16, r24;		vmov	bmll0, x0
	inv	 r17, r25;		vpush.hi.64	 x4, x4, r21:r20
	inv	 r18, r18;		vextract.64	 r21:r20, x6, #0, vaddsign1
	inv	 r19, r19;		vpush.hi.64	 x4, x4, r1:r0
	inv	 r2, r22;		vpush.hi.64	 x4, x4, r3:r2
	inv	 r0, r20;		vextract.64	 r21:r20, x6, #2, vaddsign1
	inv	 r1, r21;		vpush.hi.64	 x4, x4, r5:r4
	inv	 r3, r23;		vextract.64	 r5:r4, x6, #3, vaddsign1
	inv	 r20, r20;		vpush.hi.64	 x4, x4, r7:r6
	inv	 r21, r21;		vextract.64	 r7:r6, x6, #4, vaddsign1
	inv	 r22, r4;		vpush.hi.64	 x4, x4, r17:r16
	inv	 r23, r5;		vextract.64	 r5:r4, x6, #5, vaddsign1
	inv	 r6, r6;		vextract.64	 r17:r16, x6, #6, vaddsign1
	inv	 r7, r7;		vpush.hi.64	 x4, x4, r19:r18
	inv	 r18, r4;		vextract.64	 r5:r4, x6, #7, vaddsign1
	inv	 r19, r5;		vpush.hi.64	 x6, x0, r1:r0
	inv	 r16, r16;		vextract.64	 r1:r0, x2, #0, vaddsign1
	inv	 r17, r17;		vpush.hi.64	 x6, x6, r3:r2
	inv	 r4, r4;		vextract.64	 r3:r2, x2, #1, vaddsign1
	inv	 r5, r5;		vpush.hi.64	 x6, x6, r21:r20
	inv	 r0, r0;		vextract.64	 r21:r20, x2, #2, vaddsign1
	inv	 r1, r1;		vpush.hi.64	 x6, x6, r23:r22
	inv	 r2, r2;		vextract.64	 r23:r22, x2, #3, vaddsign1
	inv	 r3, r3;		vpush.hi.64	 x6, x6, r7:r6
	inv	 r6, r20;		vextract.64	 r21:r20, x2, #4, vaddsign1
	inv	 r7, r21;		vpush.hi.64	 x6, x6, r19:r18
	inv	 r18, r22;		vextract.64	 r23:r22, x2, #5, vaddsign1
	inv	 r19, r23;		vpush.hi.64	 x6, x6, r17:r16
	inv	 r16, r20;		vextract.64	 r21:r20, x2, #6, vaddsign1
	inv	 r17, r21;		vpush.hi.64	 x6, x6, r5:r4
	inv	 r4, r22;		vextract.64	 r23:r22, x2, #7, vaddsign1
	inv	 r5, r23;		vpush.hi.64	 x2, x0, r1:r0
	vmov	bmll0, x4
	vconv.bf16.fp32	 wl0, bmll0;		inv	 r0, r20;		vpush.hi.64	 x2, x2, r3:r2
	inv	 r1, r21;		vpush.hi.64	 x2, x2, r7:r6
	inv	 r2, r22;		vmov	bmll0, x6
	vconv.bf16.fp32	 wh0, bmll0;		inv	 r3, r23;		vpush.hi.64	 x2, x2, r19:r18
	vpush.hi.64	 x2, x2, r17:r16
	vpush.hi.64	 x2, x2, r5:r4
	vpush.hi.64	 x2, x2, r1:r0
	vpush.hi.64	 x2, x2, r3:r2
	j	#.LBB10_2
	vmov	bmll0, x2                       //  Delay Slot 5
	vconv.bf16.fp32	 wl4, bmll0             //  Delay Slot 4
	vconv.bf16.fp32	 wh4, bmll0             //  Delay Slot 3
	vst	 x0, [sp, #-512]                //  Delay Slot 2
	mova	r1, #0;		vst	 x4, [p1, #64];		movx	r0, #0 //  Delay Slot 1
.LBB10_9:                               // %for.cond.cleanup3
                                        //   in Loop: Header=BB10_10 Depth=1
	lda	 r10, [sp, #-84];		nopb	;		nopxm	 // 4-byte Folded Reload
	lda	 r2, [sp, #-80]                 // 4-byte Folded Reload
	lda	 r1, [sp, #-88]                 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	movxm	r0, #3456
	add	r2, r2, #1
	mova	r0, #384;		add	 r10, r10, r0
	add	 r1, r1, r0
.LBB10_10:                              // %for.cond
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_7 Depth 2
                                        //       Child Loop BB10_2 Depth 3
                                        //       Child Loop BB10_5 Depth 3
	lda	 r0, [sp, #-76];		nopb	;		nops	;		nopxm	;		nopv	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	eq	 r0, r0, r2
	jnz	 r0, #.LBB10_12
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.11:                              // %for.cond1.preheader
                                        //   in Loop: Header=BB10_10 Depth=1
	j	#.LBB10_7
	nop	                                //  Delay Slot 5
	st	 r2, [sp, #-80]                 // 4-byte Folded Spill Delay Slot 4
	st	 r1, [sp, #-88]                 // 4-byte Folded Spill Delay Slot 3
	st	 r10, [sp, #-84];		lshl	 r0, r2, r9 // 4-byte Folded Spill Delay Slot 2
	mova	r15, #0;		st	 r0, [sp, #-104];		mov	r2, r10 // 4-byte Folded Spill Delay Slot 1
.LBB10_12:                              // %for.cond.cleanup
	lda	 p7, [sp, #-32];		nopx	        // 4-byte Folded Reload
	lda	 p6, [sp, #-36]                 // 4-byte Folded Reload
	lda	 r15, [sp, #-40]                // 4-byte Folded Reload
	lda	 r14, [sp, #-44]                // 4-byte Folded Reload
	lda	 lr, [sp, #-72]                 // 4-byte Folded Reload
	lda	 r13, [sp, #-48]                // 4-byte Folded Reload
	lda	 r12, [sp, #-52]                // 4-byte Folded Reload
	lda	 r11, [sp, #-56]                // 4-byte Folded Reload
	lda	 r10, [sp, #-60]                // 4-byte Folded Reload
	lda	 r9, [sp, #-64]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-68]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-512                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end10:
	.size	attn_end, .Lfunc_end10-attn_end
                                        // -- End function
	.section	.text._ZN2atL5rows8EPK8bfloat16,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN2atL5rows8EPK8bfloat16
	.type	_ZN2atL5rows8EPK8bfloat16,@function
_ZN2atL5rows8EPK8bfloat16:              // @_ZN2atL5rows8EPK8bfloat16
// %bb.0:                               // %entry
	lda.s16	 r0, [p1], #2;		nopb	;		nopx	
	lda.s16	 r1, [p1], #2
	lda.s16	 r2, [p1], #2
	lda.s16	 r3, [p1], #2
	lda.s16	 r4, [p1], #2
	nop	
	nop	
	vbcst.16	 x0, r0
	lda.s16	 r0, [p1], #2;		vbcst.16	 x2, r1
	vbcst.16	 x4, r2
	lda.s16	 r1, [p1, #0];		vbcst.16	 x6, r3
	mova	r3, #16;		vbcst.16	 x8, r4
	lda.s16	 r2, [p1, #2];		vshift	x0, x0, x0, r3
	vshift	x0, x0, x2, r3
	vshift	x0, x0, x4, r3
	vbcst.16	 x2, r0
	vshift	x0, x0, x6, r3
	vbcst.16	 x4, r1
	vshift	x8, x0, x8, r3
	vbcst.16	 x6, r2
	ret	lr;		vshift	x2, x8, x2, r3
	vshift	x2, x2, x4, r3                  //  Delay Slot 5
	vshift	x2, x2, x6, r3                  //  Delay Slot 4
	vst	 x0, [p0, #0]                   //  Delay Slot 3
	vst	 x2, [p0, #64]                  //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end11:
	.size	_ZN2atL5rows8EPK8bfloat16, .Lfunc_end11-_ZN2atL5rows8EPK8bfloat16
                                        // -- End function
	.section	.text._ZN2atL10rescale_rbEiiN3aie6vectorI8bfloat16Lj16EEE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN2atL10rescale_rbEiiN3aie6vectorI8bfloat16Lj16EEE
	.type	_ZN2atL10rescale_rbEiiN3aie6vectorI8bfloat16Lj16EEE,@function
_ZN2atL10rescale_rbEiiN3aie6vectorI8bfloat16Lj16EEE: // @_ZN2atL10rescale_rbEiiN3aie6vectorI8bfloat16Lj16EEE
// %bb.0:                               // %entry
	paddxm	 [sp], #768
	st	 r8, [sp, #-60];		nopx	        // 4-byte Folded Spill
	st	 r9, [sp, #-56];		movxm	r2, #1065353216 // 4-byte Folded Spill
	st	 r15, [sp, #-32];		mov	p1, sp // 4-byte Folded Spill
	st	 p7, [sp, #-24];		vbcst.32	 x5, r2 // 4-byte Folded Spill
	st	 r11, [sp, #-48];		mov	p7, sp // 4-byte Folded Spill
	st	 p6, [sp, #-28];		movxm	r2, #16256 // 4-byte Folded Spill
	st	 r12, [sp, #-44];		mov	p6, sp // 4-byte Folded Spill
	st	 r13, [sp, #-40];		vbcst.16	 x10, r2 // 4-byte Folded Spill
	st	 r14, [sp, #-36];		movxm	r2, #16177 // 4-byte Folded Spill
	mova	m0, #-768;		st	 r10, [sp, #-52];		vbcst.16	 x8, r2 // 4-byte Folded Spill
	padda	 [p1], m0;		st	 lr, [sp, #-64];		movxm	r2, #15990 // 4-byte Folded Spill
	movs	p0, p1;		vbcst.16	 x1, r2
	vst	 x5, [p1, #64];		movxm	r2, #15715
	vst	 x5, [p0], #128;		vbcst.16	 x9, r2
	vst	 x10, [p0], #64;		movxm	r2, #-16512
	vst	 x8, [p0], #64;		jl	#_ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_
	mova	r10, #0;		vst	 x1, [p0, #0];		or	 r8, r0, r0;		vbcst.16	 x6, r2 //  Delay Slot 5
	vst	 x9, [p0, #64];		or	 r9, r1, r1;		vbcst.32	 x3, r10 //  Delay Slot 4
	mova	r11, #576;		vst	 x6, [sp, #-192];		movx	r15, #6;		vbcst.16	 x2, r10 // 64-byte Folded Spill Delay Slot 3
	padda	 [p6], #-384;		paddb	 [p7], #-256;		movx	r12, #512;		vmov	wh0, wl3;		vst	 x6, [sp, #-128] // 64-byte Folded Spill Delay Slot 2
	mova	r14, #64;		movx	r13, #828;		mov	p0, p7 //  Delay Slot 1
	nopa	;		nopb	;		nops	;		jl	#_ZN2atL5rows8EPK8bfloat16;		nopv	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	movs	p1, p7;		mov	p0, p6          //  Delay Slot 1
	nopa	;		nopb	;		nops	;		movxm	p0, ##_ZN2atL3g_oE;		nopv	
	mova	r0, #2;		movxm	p1, ##_ZN2atL6g_osumE
	lshl	 r1, r8, r0
	lda	 p1, [p1, #0];		mov	dj0, r1
	lda	 p0, [p0, dj0]
	mova	r1, #11
	lshl	 r1, r9, r1
	mac	r9, r9, r8, r15
	vlda	 x4, [sp, #-192];		j	#.LBB12_2 // 64-byte Folded Reload
	mova	r2, #8                          //  Delay Slot 5
	vlda	 x0, [sp, #-384];		lshl	 r2, r9, r2;		mov	m0, r1 //  Delay Slot 4
	padda	 [p0], m0;		mov	m0, r2  //  Delay Slot 3
	padda	 [p1], m0                       //  Delay Slot 2
	vlda	 x5, [sp, #-128];		vldb	 x1, [p6, #64];		mov	r2, p1 // 64-byte Folded Reload Delay Slot 1
.LBB12_1:                               // %for.body
                                        //   in Loop: Header=BB12_2 Depth=1
	nopa	;		lshl	 r1, r10, r0
	movs	p2, p0;		mov	m0, r1
	padda	 [p2], m0
	eq	 r27, r10, r12;		mov	r1, p2
	sel.nez	 r1, r2, r1, r27
	mov	p2, r1
	vlda	 bmhl0, [p2, #128]
	vlda	 bmll0, [p2, #0]
	vlda	 bmlh0, [p2, #64]
	vlda	 bmhh0, [p2, #192]
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml0
	vconv.bf16.fp32	 x3, cmh0
	nop	
	vmac.f	dm0, dm0, y1, y2, r13
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml0;		vmul.f	dm1, y1, y0, r13
	vconv.bf16.fp32	 x3, cmh0
	nop	
	vmac.f	dm0, dm1, y1, y0, r13
	nop	
	nop	
	nop	
	nop	
	nop	
	vst	 bmll0, [p2, #0]
	vst	 bmlh0, [p2, #64]
	vst	 bmhl0, [p2, #128]
	vst	 bmhh0, [p2, #192];		add	 r10, r10, r14
.LBB12_2:                               // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		ne	 r1, r10, r11;		nopm	;		nopv	
	jnz	 r1, #.LBB12_1
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %for.cond.cleanup
	lda	 p7, [sp, #-24]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-28]                 // 4-byte Folded Reload
	lda	 r15, [sp, #-32]                // 4-byte Folded Reload
	lda	 r14, [sp, #-36]                // 4-byte Folded Reload
	lda	 lr, [sp, #-64]                 // 4-byte Folded Reload
	lda	 r13, [sp, #-40]                // 4-byte Folded Reload
	lda	 r12, [sp, #-44]                // 4-byte Folded Reload
	lda	 r11, [sp, #-48]                // 4-byte Folded Reload
	lda	 r10, [sp, #-52]                // 4-byte Folded Reload
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-768                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end12:
	.size	_ZN2atL10rescale_rbEiiN3aie6vectorI8bfloat16Lj16EEE, .Lfunc_end12-_ZN2atL10rescale_rbEiiN3aie6vectorI8bfloat16Lj16EEE
                                        // -- End function
	.section	.text._ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_
	.type	_ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_,@function
_ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_: // @_ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	r0, #-15760;		nopv	
	nops	;		vbcst.16	 x4, r0
	movxm	r0, #17008
	vmax_lt.bf16	 x0, r16, x0, x4
	vbcst.16	 x6, r0
	movxm	r0, #16313
	vmin_ge.bf16	 x0, r16, x0, x6
	vlda	 bmlh0, [p1, #64];		vbcst.16	 x2, r0
	mova	r0, #828
	vlda	 bmll0, [p1, #0];		movxm	r1, #-17620;		vmul.f	dm1, x0, x2, r0
	padda	 [p1], #128;		vbcst.16	 x4, r1
	vldb	 x6, [p1], #64
	mova	r1, #0;		vmac.f	dm1, dm1, x0, x4, r0
	vbcst.16	 x0, r1
	nop	
	movxm	r1, #19264;		vmac.f	dm2, dm1, x0, x2, r0
	vbcst.16	 x0, r1
	nop	
	movxm	r1, #-13504;		vmac.f	dm1, dm2, x6, x0, r0
	vbcst.16	 x0, r1
	nop	
	vmac.f	dm3, dm1, x6, x0, r0
	nop	
	nop	
	nop	
	nop	
	movxm	r1, #-16512
	vconv.bf16.fp32	 x0, cml3;		vbcst.16	 x2, r1
	nop	
	vmac.f	dm2, dm2, x0, x2, r0
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml2
	nop	
	vmul.f	dm2, x0, x0, r0
	nop	
	nop	
	vldb	 x2, [p1], #64
	nop	
	nop	
	vldb	 x6, [p1, #0];		vconv.bf16.fp32	 x4, cml2
	nop	
	vmul.f	dm1, x4, x0, r0
	vldb	 x1, [p1, #64]
	mov	s0, #7;		vmac.f	dm0, dm0, x0, x2, r0
	vmov	x8, bmlh1
	mova	r1, #2;		vmov	x10, bmll1
	vshuffle	x0, x10, x8, r1;		vmac.f	dm0, dm0, x4, x6, r0
	vconv.bf16.fp32	 x2, cml1;		mov	crupsmode, #0
	vups.2x	cml1, x0, s0, upssign1
	mov	crsrsmode, #0;		vmac.f	dm0, dm0, x2, x1, r0
	mov	s0, #0
	vsrs.2x	x0, cml1, s0, srssign1
	nop	
	movxm	r1, #16256
	vbcst.16	 x2, r1
	vconv.bf16.fp32	 x2, cml0;		vadd.16	 x0, x0, x2
	nop	
	vmul.f	dm0, x2, x0, r0
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml0, [p0, #0] //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end13:
	.size	_ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_, .Lfunc_end13-_ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_
                                        // -- End function
	.section	.text._ZN3nk8L8expd32x2ERN3aie6vectorI8bfloat16Lj32EEES4_S3_S3_RKNS_4ExpQE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN3nk8L8expd32x2ERN3aie6vectorI8bfloat16Lj32EEES4_S3_S3_RKNS_4ExpQE
	.type	_ZN3nk8L8expd32x2ERN3aie6vectorI8bfloat16Lj32EEES4_S3_S3_RKNS_4ExpQE,@function
_ZN3nk8L8expd32x2ERN3aie6vectorI8bfloat16Lj32EEES4_S3_S3_RKNS_4ExpQE: // @_ZN3nk8L8expd32x2ERN3aie6vectorI8bfloat16Lj32EEES4_S3_S3_RKNS_4ExpQE
// %bb.0:                               // %entry
	nopa	;		vldb	 x8, [p0, #0];		nopx	
	nop	
	nop	
	nop	
	vldb	 x10, [p1, #0];		movxm	r0, #16313
	vbcst.16	 x4, r0
	mova	r0, #828
	movxm	r1, #-17620;		vmul.f	dm0, x8, x4, r0
	vbcst.16	 x6, r1
	nop	
	vmac.f	dm0, dm0, x8, x6, r0
	vmul.f	dm1, x10, x4, r0
	nop	
	vmsc.f	dm0, dm0, x0, x4, r0
	vmac.f	dm0, dm1, x10, x6, r0
	vldb	 x8, [p2, #256]
	vmsc.f	dm2, dm0, x0, x6, r0
	vmsc.f	dm0, dm0, x2, x4, r0
	nop	
	movxm	r1, #19264
	vbcst.16	 x0, r1;		vmsc.f	dm4, dm0, x2, x6, r0
	nop	
	movxm	r1, #-13504;		vmac.f	dm1, dm2, x8, x0, r0
	vbcst.16	 x10, r1;		vmac.f	dm0, dm4, x8, x0, r0
	nop	
	vmac.f	dm3, dm1, x8, x10, r0
	vmac.f	dm3, dm0, x8, x10, r0
	nop	
	nop	
	nop	
	movxm	r1, #-16512
	vconv.bf16.fp32	 x2, cml3;		vbcst.16	 x0, r1
	vconv.bf16.fp32	 x4, cml3
	mov	crupsmode, #0;		vmac.f	dm2, dm2, x2, x0, r0
	vmov	x6, bmlh1;		vmac.f	dm3, dm4, x4, x0, r0
	movxm	r1, #15990
	vlda	 bmlh1, [p2, #192];		vbcst.16	 x10, r1
	vlda	 bmll1, [p2, #128];		movxm	r1, #65416
	vmov	x8, bmll1
	mova	r2, #2;		vconv.bf16.fp32	 x0, cml2;		vbcst.16	 x1, r1
	mova	r1, #7;		vconv.bf16.fp32	 x2, cml3;		vshuffle	x6, x8, x6, r2
	mov	s0, r1;		vmac.f	dm2, dm1, x0, x10, r0
	vmax_lt.16	 x6, r16, x6, x1, vaddsign1;		vmac.f	dm2, dm1, x2, x10, r0
	vlda	 bmlh1, [p2, #64];		vmov	x8, bmlh0
	vlda	 bmll1, [p2, #0];		vmov	x10, bmll0
	vups.2x	cml0, x6, s0, upssign1
	vshuffle	x8, x10, x8, r2
	vconv.bf16.fp32	 x4, cml2;		vmax_lt.16	 x8, r16, x8, x1, vaddsign1
	vconv.bf16.fp32	 x6, cml2;		mov	crsrsmode, #0
	mova	r1, #0;		vups.2x	cml2, x8, s0, upssign1;		vmac.f	dm3, dm1, x0, x4, r0
	mov	s0, r1;		vmac.f	dm1, dm1, x2, x6, r0
	vsrs.2x	x0, cml0, s0, srssign1
	vsrs.2x	x2, cml2, s0, srssign1
	nop	
	nop	
	vconv.bf16.fp32	 x4, cml3
	vconv.bf16.fp32	 x6, cml1;		ret	lr
	vadd.16	 x0, x0, x4                     //  Delay Slot 5
	vadd.16	 x2, x2, x6                     //  Delay Slot 4
	vst	 x0, [p0, #0]                   //  Delay Slot 3
	vst	 x2, [p1, #0]                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end14:
	.size	_ZN3nk8L8expd32x2ERN3aie6vectorI8bfloat16Lj32EEES4_S3_S3_RKNS_4ExpQE, .Lfunc_end14-_ZN3nk8L8expd32x2ERN3aie6vectorI8bfloat16Lj32EEES4_S3_S3_RKNS_4ExpQE
                                        // -- End function
	.type	_ZL6acc_hi,@object              // @_ZL6acc_hi
	.section	.bss._ZL6acc_hi,"aw",@nobits
	.p2align	6, 0x0
_ZL6acc_hi:
	.space	24576
	.size	_ZL6acc_hi, 24576

	.type	_ZL8g_stripe,@object            // @_ZL8g_stripe
	.section	.bss._ZL8g_stripe,"aw",@nobits
	.p2align	2, 0x0
_ZL8g_stripe:
	.word	0                               // 0x0
	.size	_ZL8g_stripe, 4

	.type	_ZN2atL3g_oE,@object            // @_ZN2atL3g_oE
	.section	.bss._ZN2atL3g_oE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL3g_oE:
	.space	8
	.size	_ZN2atL3g_oE, 8

	.type	_ZN2atL6g_osumE,@object         // @_ZN2atL6g_osumE
	.section	.bss._ZN2atL6g_osumE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL6g_osumE:
	.word	0
	.size	_ZN2atL6g_osumE, 4

	.type	_ZN2atL5g_srbE,@object          // @_ZN2atL5g_srbE
	.section	.bss._ZN2atL5g_srbE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL5g_srbE:
	.word	0
	.size	_ZN2atL5g_srbE, 4

	.type	_ZN2atL6g_maxbE,@object         // @_ZN2atL6g_maxbE
	.section	.bss._ZN2atL6g_maxbE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL6g_maxbE:
	.word	0
	.size	_ZN2atL6g_maxbE, 4

	.type	_ZN2atL6g_onesE,@object         // @_ZN2atL6g_onesE
	.section	.bss._ZN2atL6g_onesE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL6g_onesE:
	.word	0
	.size	_ZN2atL6g_onesE, 4

	.type	_ZN2atL6g_etmpE,@object         // @_ZN2atL6g_etmpE
	.section	.bss._ZN2atL6g_etmpE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL6g_etmpE:
	.word	0
	.size	_ZN2atL6g_etmpE, 4

	.type	_ZN2atL8g_ntilesE,@object       // @_ZN2atL8g_ntilesE
	.section	.bss._ZN2atL8g_ntilesE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL8g_ntilesE:
	.word	0                               // 0x0
	.size	_ZN2atL8g_ntilesE, 4

	.type	_ZN2atL7g_chunkE,@object        // @_ZN2atL7g_chunkE
	.section	.bss._ZN2atL7g_chunkE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL7g_chunkE:
	.word	0                               // 0x0
	.size	_ZN2atL7g_chunkE, 4

	.type	_ZN2atL3g_qE,@object            // @_ZN2atL3g_qE
	.section	.bss._ZN2atL3g_qE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL3g_qE:
	.space	8
	.size	_ZN2atL3g_qE, 8

	.type	_ZN2atL5g_erbE,@object          // @_ZN2atL5g_erbE
	.section	.bss._ZN2atL5g_erbE,"aw",@nobits
	.p2align	2, 0x0
_ZN2atL5g_erbE:
	.space	8
	.size	_ZN2atL5g_erbE, 8

	.ident	"clang version 22.0.0 (https://github.com/Xilinx/llvm-aie 0006955eb0bd0a9033185a5be72b621dce63ce21)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZL6acc_hi
