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
	nopa	;		nopb	;		nops	;		movxm	dj0, #1024;		nopv	
	lda	 r13, [p1, dj0];		nopb	;		nopx	
	nop	
	paddxm	 [sp], #832
	st	 r8, [sp, #-56]                 // 4-byte Folded Spill
	st	 p6, [sp, #-24]                 // 4-byte Folded Spill
	st	 r9, [sp, #-52]                 // 4-byte Folded Spill
	st	 r13, [sp, #-36];		movx	r0, #2 // 4-byte Folded Spill
	st	 lr, [sp, #-60];		ge	 r0, r13, r0 // 4-byte Folded Spill
	st	 r10, [sp, #-48];		jnz	 r0, #.LBB2_3 // 4-byte Folded Spill
	st	 r11, [sp, #-44]                // 4-byte Folded Spill Delay Slot 5
	st	 r12, [sp, #-40];		mov	p3, p1 // 4-byte Folded Spill Delay Slot 4
	mova	m0, #-832;		st	 p7, [sp, #-20];		mov	crrnd, #12 // 4-byte Folded Spill Delay Slot 3
	mova	r8, #6;		st	 r14, [sp, #-32];		mov	p7, sp // 4-byte Folded Spill Delay Slot 2
	padda	 [p7], m0;		st	 r15, [sp, #-28];		movx	r9, #512;		mov	p6, p0 // 4-byte Folded Spill Delay Slot 1
// %bb.1:                               // %if.then
	jl	#_ZL9bias_passPKfP8bfloat16PKS1_ii
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mova	r0, #6;		movs	p0, p6;		mov	p1, p6 //  Delay Slot 2
	mova	r8, #1;		movs	p2, p3;		movx	r1, #512;		mov	p7, p3 //  Delay Slot 1
	nopa	;		nopb	;		nops	;		jl	#_ZL9bias_passPKfP8bfloat16PKS1_ii;		nopv	
	nop	                                //  Delay Slot 5
	movxm	m0, #6144                       //  Delay Slot 4
	mov	p1, p6                          //  Delay Slot 3
	padda	 [p1], m0;		movxm	p0, ##_ZL6acc_hi //  Delay Slot 2
	mova	r1, #512;		movx	r0, #12;		mov	p2, p7 //  Delay Slot 1
	nopa	;		nopb	;		nops	;		ne	 r0, r13, r8;		nopm	;		nopv	
	jnz	 r0, #.LBB2_34
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.2:                               // %if.then2
	lda	 p7, [sp, #-20]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-24]                 // 4-byte Folded Reload
	lda	 r15, [sp, #-28]                // 4-byte Folded Reload
	lda	 r14, [sp, #-32]                // 4-byte Folded Reload
	lda	 r13, [sp, #-36]                // 4-byte Folded Reload
	lda	 r12, [sp, #-40]                // 4-byte Folded Reload
	lda	 r11, [sp, #-44]                // 4-byte Folded Reload
	lda	 r10, [sp, #-48];		movs	p0, p6 // 4-byte Folded Reload
	lda	 r9, [sp, #-52]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-56]                 // 4-byte Folded Reload
	lda	 lr, [sp, #-60]                 // 4-byte Folded Reload
	j	#_ZL10gelu_lut_nP8bfloat16iPKfS2_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	movxm	p1, ##sgelu_ab                  //  Delay Slot 3
	paddxm	 [sp], #-832                    //  Delay Slot 2
	mova	r0, #288;		movxm	p2, ##sgelu_cd //  Delay Slot 1
.LBB2_3:                                // %if.end3
	mova	r0, #7;		nopx	
	ne	 r0, r13, r0
	jnz	 r0, #.LBB2_14
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mova	r12, #18                        //  Delay Slot 2
	mova	r15, #828;		movx	r28, #0;		mov	r11, #64 //  Delay Slot 1
// %bb.4:                               // %if.then5
	mova	r2, #53
	mova	r4, #448;		movxm	r0, #16256
	mova	r5, #56;		movxm	r1, #3072
	mova	r6, #1;		movxm	r3, #2304
	mova	r7, #2;		movxm	m0, #14848
	mova	r17, #7;		movxm	r16, #4182016
	mova	r20, #0;		movs	p0, p6;		movx	r19, #0;		mov	r18, #0
	padda	 [p0], m0;		movx	r0, #52;		vbcst.16	 x0, r0
.LBB2_5:                                // %for.cond8.preheader
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB2_6 Depth 2
                                        //       Child Loop BB2_7 Depth 3
                                        //       Child Loop BB2_9 Depth 3
                                        //     Child Loop BB2_12 Depth 2
	mova	r22, #0;		nopb	;		nops	;		movx	r21, #0;		nopm	;		nopv	
.LBB2_6:                                // %for.body11
                                        //   Parent Loop BB2_5 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_7 Depth 3
                                        //       Child Loop BB2_9 Depth 3
	nopa	;		lshl	 r23, r22, r6;		nopm	
	mova	r24, #0;		movs	m0, r23;		mov	r23, r19
	movs	p1, p3;		add	 r25, r20, r24;		mov	p2, p6
	padda	 [p1], m0;		lshl	 r26, r23, r7
	movs	m0, r26;		add	 r27, r26, r16
	vlda	 x2, [p1, #0];		paddb	 [p2], m0;		ltu	 r27, r25, r8;		movs	m0, r27
	vldb	 x4, [p1, #64];		add	 r23, r23, r9;		mov	r25, p2
	movxm	p1, ##_ZL6acc_hi
	paddb	 [p1], m0
	mov	r26, p1
	sel.nez	 r25, r25, r26, r27
	movs	p1, r25;		lshl	 r26, r22, r7
	lshl	 r25, r24, r17;		mov	m0, r26
	paddb	 [p1], m0;		add	r24, r24, #1;		mov	m0, r25
	vlda	 bmll4, [p1], #64
	vlda	 bmlh4, [p1], #64
	vlda	 bmll0, [p1, #0]
	vlda	 bmlh0, [p1, #64];		movs	p1, p7
	paddb	 [p1], m0
	vmac.f	dm1, dm4, x2, x0, r15
	nop	
	vmac.f	dm2, dm0, x4, x0, r15
	nop	
	nop	
	add.nc	lc, r8, #-1
	vconv.bf16.fp32	 x3, cml1;		movxm	ls, #.LBB2_7
	movxm	le, #.L_LEnd5
.LBB2_7:                                // %for.body15
                                        //   Parent Loop BB2_5 Depth=1
                                        //     Parent Loop BB2_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		vconv.bf16.fp32	 x5, cml2;		add	 r25, r20, r24;		mov	p2, p6;		nopv	
	nopa	;		lshl	 r26, r23, r7;		nopm	
	movs	m0, r26;		add	 r27, r26, r16;		vshuffle	x7, x3, x5, r0
	movs	m0, r27;		paddb	 [p2], m0;		ltu	 r27, r25, r8;		vshuffle	x9, x3, x5, r2
	vst	 x7, [p1, #0];		add	 r23, r23, r9;		mov	r25, p2
	vst	 x9, [p1, #64];		movxm	p1, ##_ZL6acc_hi
	paddb	 [p1], m0
	mov	r26, p1
	sel.nez	 r25, r25, r26, r27
	movs	p1, r25;		lshl	 r26, r22, r7
	lshl	 r25, r24, r17;		mov	m0, r26
	paddb	 [p1], m0;		add	r24, r24, #1;		mov	m0, r25
	vlda	 bmll4, [p1], #64
	vlda	 bmlh4, [p1], #64
	vlda	 bmll0, [p1, #0]
	vlda	 bmlh0, [p1, #64];		movs	p1, p7
	paddb	 [p1], m0
	vmac.f	dm1, dm4, x2, x0, r15
	nop	
	vmac.f	dm2, dm0, x4, x0, r15
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x3, cml1
.L_LEnd5:
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
// %bb.8:                               // %for.cond.cleanup14
                                        //   in Loop: Header=BB2_6 Depth=2
	mova	r23, #0;		vconv.bf16.fp32	 x5, cml2;		movx	r26, #0;		mov	m0, r21
	mov	p2, p0
	padda	 [p2], m0;		vshuffle	x7, x3, x5, r0
	vshuffle	x9, x3, x5, r2
	vst	 x7, [p1, #0];		add.nc	lc, r8, #-4
	vst	 x9, [p1, #64];		movxm	ls, #.LBB2_9
	lshl	 r24, r23, r17
	movs	m0, r24;		add	r23, r23, #1;		mov	p1, p7
	paddb	 [p1], m0
	vlda.conv.fp32.bf16	 cml1, [p1, #0];		lshl	 r24, r23, r17
	vlda.conv.fp32.bf16	 cmh1, [p1, #64];		movs	m0, r24;		add	r23, r23, #1;		mov	p1, p7
	nopa	;		paddb	 [p1], m0;		nops	;		movxm	le, #.L_LEnd4;		nopv	
	vlda.conv.fp32.bf16	 cml1, [p1, #0];		nopb	;		nops	;		lshl	 r24, r23, r17;		nopm	;		nopv	
	vlda.conv.fp32.bf16	 cmh1, [p1, #64];		nopb	;		movs	m0, r24;		add	r23, r23, #1;		mov	p1, p7;		nopv	
	nopa	;		paddb	 [p1], m0;		nops	;		nopxm	;		nopv	
	vlda.conv.fp32.bf16	 cml1, [p1, #0];		nopb	;		lshl	 r24, r23, r17;		nopm	;		nops	
	vlda.conv.fp32.bf16	 cmh1, [p1, #64];		movs	m0, r24;		add	r23, r23, #1;		mov	p1, p7
	paddb	 [p1], m0;		vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26]
.LBB2_9:                                // %for.body44
                                        //   Parent Loop BB2_5 Depth=1
                                        //     Parent Loop BB2_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	vlda.conv.fp32.bf16	 cml1, [p1, #0];		nopb	;		vst.flush.512.conv	 [p2, sf, r26];		lshl	 r24, r23, r17;		nopm	;		nopv	
	vlda.conv.fp32.bf16	 cmh1, [p1, #64];		nopb	;		movs	m0, r24;		add	r23, r23, #1;		mov	p1, p7;		nopv	
.L_LEnd4:
	nopa	;		paddb	 [p1], m0;		vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		nopxm	;		nopv	
// %bb.10:                              // %for.cond.cleanup43
                                        //   in Loop: Header=BB2_6 Depth=2
	vlda.conv.fp32.bf16	 cml1, [p1, #0];		nopb	;		vst.flush.512.conv	 [p2, sf, r26];		add	 r22, r22, r11;		nopm	;		nopv	
	vlda.conv.fp32.bf16	 cmh1, [p1, #64];		nopb	;		eq	 r23, r22, r9;		nopm	;		nops	
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26]
	vst.flush.512.conv	 [p2, sf, r26];		add	 r21, r21, r4
	nop	
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26]
	vst.flush.512.conv	 [p2, sf, r26]
	nop	
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		jz	 r23, #.LBB2_6
	vst.flush.512.conv	 [p2, sf, r26]  //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.11:                              // %for.cond.cleanup10
                                        //   in Loop: Header=BB2_5 Depth=1
	lshl	 r21, r18, r6
	lshl	 r21, r28, r6;		mov	m0, r21
	mov	dj0, r21
	vldb	 x4, [p0, dj0]
	nop	
	nop	
	movxm	ls, #.LBB2_12
	movs	p1, p6;		movxm	le, #.L_LEnd3
	padda	 [p1], m0;		nopb	;		nops	;		add	r21, r28, #32;		add.nc	lc, r5, #-1;		nopv	
.LBB2_12:                               // %for.body60
                                        //   Parent Loop BB2_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		nops	;		lshl	 r22, r21, r6;		nopm	;		nopv	
	nopa	;		nopb	;		vst	 x4, [p1, dj0];		nopx	;		mov	dj0, r22;		nopv	
	nopa	;		vldb	 x4, [p0, dj0];		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nopxm	;		nops	
	nop	
.L_LEnd3:
	nopa	;		nopb	;		nops	;		add	r21, r21, #32;		nopm	;		nopv	
// %bb.13:                              //   in Loop: Header=BB2_5 Depth=1
	nopa	;		add	r20, r20, #6
	eq	 r21, r20, r12
	jz	 r21, #.LBB2_5
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst	 x4, [p1, dj0];		add	 r19, r19, r1 //  Delay Slot 2
	add	 r18, r18, r3                   //  Delay Slot 1
	j	#.LBB2_34
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB2_14:                               // %if.end75
	nopa	;		nopb	;		jl	#_ZL9bias_passPKfP8bfloat16PKS1_ii;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	movs	p0, p6;		movxm	r10, #2147483646 //  Delay Slot 3
	movs	p1, p6;		movxm	r1, #1024       //  Delay Slot 2
	mova	r0, #6;		movs	p2, p3;		mov	r14, p3 //  Delay Slot 1
	jl	#_ZL9bias_passPKfP8bfloat16PKS1_ii
	nop	                                //  Delay Slot 5
	movxm	p0, ##_ZL6acc_hi                //  Delay Slot 4
	movxm	p1, ##_ZL6acc_hi                //  Delay Slot 3
	movxm	r1, #1024                       //  Delay Slot 2
	mova	r0, #12;		mov	p2, r14         //  Delay Slot 1
	mova	r1, #2;		nopb	;		and	 r0, r13, r10;		nopm	
	ne	 r0, r0, r1
	jnz	 r0, #.LBB2_27
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	st	 r13, [sp, #-64]                // 4-byte Folded Spill Delay Slot 1
// %bb.15:
	mova	r10, #2
	mova	r12, #0;		movx	r14, #18;		mov	r13, #0
.LBB2_16:                               // %for.body82
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		lshl	 r0, r13, r10
	movxm	r1, #4182016
	movxm	p0, ##_ZL6acc_hi
	movxm	p2, ##sgelu_cd
	movs	p1, p6;		add	 r1, r0, r1;		mov	m0, r0
	padda	 [p1], m0;		jl	#_ZL10gelu_lut_nP8bfloat16iPKfS2_
	movs	m0, r1;		mov	r0, p1          //  Delay Slot 5
	padda	 [p0], m0;		movxm	p1, ##sgelu_ab //  Delay Slot 4
	ltu	 r27, r12, r8;		mov	r1, p0  //  Delay Slot 3
	sel.nez	 r1, r0, r1, r27                //  Delay Slot 2
	mova	r0, #8;		mov	p0, r1          //  Delay Slot 1
	nopa	;		nopb	;		nops	;		add	r12, r12, #1;		nopm	;		nopv	
	eq	 r0, r12, r14
	jz	 r0, #.LBB2_16
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	add	 r13, r13, r9                   //  Delay Slot 1
// %bb.17:
	lda	 r22, [sp, #-64]                // 4-byte Folded Reload
	mova	r2, #1;		movx	r1, #2;		mov	m0, #512
	mova	r6, #0;		movx	r5, #0;		mov	r3, #7
	mova	r20, #18;		movx	r16, #0;		mov	r7, #0
	mova	r21, #0;		movxm	r0, #4182016
	mova	r23, #288;		movxm	m1, #12288
	mova	dj0, #288;		movx	r4, #3;		mov	r24, #4
	mova	r28, #320;		eq	 r4, r22, r4;		mov	r25, #2
.LBB2_18:                               // %for.body93
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB2_19 Depth 2
                                        //     Child Loop BB2_21 Depth 2
                                        //     Child Loop BB2_24 Depth 2
	nopa	;		nopb	;		lshl	 r17, r5, r1
	movxm	p0, ##_ZL6acc_hi
	movs	p1, p6;		add	 r18, r17, r0;		mov	m2, r17
	padda	 [p1], m2;		mov	m2, r18
	padda	 [p0], m2;		mov	r17, p1
	ltu	 r27, r16, r8;		mov	r18, p0
	sel.nez	 r17, r17, r18, r27
	mova	r17, #0;		mov	p0, r17
	mova	r18, #0;		lshl	 r19, r17, r2
	movs	m2, r19;		lshl	 r19, r18, r3
	add	r18, r18, #1;		mov	p1, p0
	padda	 [p1], m2;		add	 r17, r17, r11;		mov	m2, r19
	vldb	 x3, [p1], m0
	vlda	 x1, [p1], #-448
	vldb	 x10, [p1], m0
	vlda	 x8, [p1, #0];		mov	p1, p7
	paddb	 [p1], m2
	nop	
	nop	
	nop	
	vmul.f	dm3, x3, x1, r15
	nop	
	vmul.f	dm2, x10, x8, r15
	add.nc	lc, r24, #-1
	movxm	ls, #.LBB2_19
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd8;		nopv	
.LBB2_19:                               // %for.body115
                                        //   Parent Loop BB2_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml3, [p1, #0];		lshl	 r19, r17, r2;		nopm	;		nopv	
	nopa	;		nopb	;		movs	m2, r19;		lshl	 r19, r18, r3;		nopm	;		nopv	
	nopa	;		vst.conv.bf16.fp32	 cml2, [p1, #64];		add	r18, r18, #1;		mov	p1, p0
	padda	 [p1], m2;		add	 r17, r17, r11;		mov	m2, r19
	vldb	 x3, [p1], m0
	vlda	 x1, [p1], #-448
	vldb	 x10, [p1], m0
	vlda	 x8, [p1, #0];		mov	p1, p7
	paddb	 [p1], m2
	nop	
	nop	
	nop	
	vmul.f	dm3, x3, x1, r15
	nop	
	vmul.f	dm2, x10, x8, r15
	nop	
	nop	
.L_LEnd8:
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
// %bb.20:                              // %for.body149.preheader
                                        //   in Loop: Header=BB2_18 Depth=1
	mova	r26, #0;		nopb	;		vst.conv.bf16.fp32	 cml3, [p1, #0];		lshl	 r17, r6, r2;		mov	p0, p6;		nopv	
	padda	 [p0], m1;		nopb	;		eq	 r27, r22, r25;		mov	m2, r7;		nops	
	padda	 [p0], m2;		vst.conv.bf16.fp32	 cml2, [p1, #64];		mov	m2, r17
	mov	p1, p6
	padda	 [p1], m2;		mov	r17, p0
	mov	r18, p1
	mova	r17, #0;		sel.nez	 r18, r17, r18, r27
	lshl	 r19, r17, r3
	movs	m2, r19;		add	r17, r17, #1;		mov	p2, p7
	paddb	 [p2], m2
	vlda.conv.fp32.bf16	 cml1, [p2, #0]
	vlda.conv.fp32.bf16	 cmh1, [p2, #64];		mov	p2, r18
	nop	
	nop	
	nop	
	nop	
	nop	
	add.nc	lc, r24, #-1
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		movxm	ls, #.LBB2_21
	nopa	;		nopb	;		vst.flush.512.conv	 [p2, sf, r26];		movxm	le, #.L_LEnd7;		nopv	
.LBB2_21:                               // %for.body149
                                        //   Parent Loop BB2_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		nops	;		lshl	 r19, r17, r3;		mov	r18, p2;		nopv	
	nopa	;		nopb	;		movs	m2, r19;		add	r17, r17, #1;		mov	p2, p7;		nopv	
	nopa	;		paddb	 [p2], m2;		nops	;		nopxm	;		nopv	
	vlda.conv.fp32.bf16	 cml1, [p2, #0];		nopb	;		nops	;		nopxm	;		nopv	
	vlda.conv.fp32.bf16	 cmh1, [p2, #64];		nopb	;		nops	;		nopx	;		mov	p2, r18;		nopv	
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26]
.L_LEnd7:
	nopa	;		nopb	;		vst.flush.512.conv	 [p2, sf, r26];		nopxm	;		nopv	
// %bb.22:                              // %for.cond.cleanup148
                                        //   in Loop: Header=BB2_18 Depth=1
	nopa	;		nopb	;		nopx	;		mov	r18, p2
	jz	 r4, #.LBB2_26
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.23:                              // %if.then157
                                        //   in Loop: Header=BB2_18 Depth=1
	lshl	 r17, r21, r2
	mov	dj1, r17
	vldb.128	 wh2, [p0, dj1]
	nop	
	nop	
	movxm	ls, #.LBB2_24
	movxm	le, #.L_LEnd6
	movs	p2, p1;		mov	m2, r17
	padda	 [p2], m2;		nopb	;		nops	;		add	r17, r21, #8;		add.nc	lc, r20, #-1;		nopv	
.LBB2_24:                               // %for.body166
                                        //   Parent Loop BB2_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		vst.128	 wh2, [p2, dj0];		lshl	 r18, r17, r2;		nopm	;		nopv	
	nopa	;		nopb	;		movs	dj1, r18;		nopxm	;		nopv	
	nopa	;		vldb.128	 wh2, [p0, dj1];		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nopx	
	mov	p2, p1
	mov	m2, r18
.L_LEnd6:
	padda	 [p2], m2;		nopb	;		nops	;		add	r17, r17, #8;		nopm	;		nopv	
// %bb.25:                              //   in Loop: Header=BB2_18 Depth=1
	nopa	;		nopb	;		vst.128	 wh2, [p2, dj0];		nopxm	;		nopv	
.LBB2_26:                               // %if.end179
                                        //   in Loop: Header=BB2_18 Depth=1
	nopa	;		add	r16, r16, #1
	eq	 r17, r16, r20
	jz	 r17, #.LBB2_18
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	add	 r7, r7, r28                    //  Delay Slot 3
	add	 r6, r6, r23                    //  Delay Slot 2
	add	 r5, r5, r9                     //  Delay Slot 1
	j	#.LBB2_34
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB2_27:                               // %if.end183
	mova	r0, #4;		nopb	;		nopxm	
	ne	 r0, r13, r0
	jnz	 r0, #.LBB2_30
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.28:                              // %for.body190.preheader
	movxm	r15, #4182016
	mova	r13, #0;		movx	r10, #0;		mov	r14, #2
.LBB2_29:                               // %for.body190
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		lshl	 r0, r10, r14;		nopm	;		nops	
	movxm	p0, ##_ZL6acc_hi
	movxm	p2, ##serf_cd
	movs	p1, p6;		add	 r1, r0, r15;		mov	m0, r0
	padda	 [p1], m0;		jl	#_ZL10gelu_lut_nP8bfloat16iPKfS2_
	movs	m0, r1;		mov	r0, p1          //  Delay Slot 5
	padda	 [p0], m0;		movxm	p1, ##serf_ab //  Delay Slot 4
	ltu	 r27, r13, r8;		mov	r1, p0  //  Delay Slot 3
	sel.nez	 r1, r0, r1, r27                //  Delay Slot 2
	mova	r0, #16;		mov	p0, r1          //  Delay Slot 1
	nopa	;		add	r13, r13, #1;		nopm	
	ne	 r0, r13, r12
	jnz	 r0, #.LBB2_29
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	add	 r10, r10, r9                   //  Delay Slot 1
.LBB2_30:                               // %for.body201.preheader
	lda	 r23, [sp, #-64];		nopb	;		nopx	 // 4-byte Folded Reload
	mova	r4, #-1;		movx	r2, #2;		mov	r1, #576
	mova	r7, #0;		movx	r6, #0;		mov	r5, #1
	mova	r17, #0;		movx	r18, #0;		mov	r16, #0
	mova	r22, #0;		movxm	r0, #4608
	mova	r24, #288;		movxm	r3, #4182016
	movx	r28, #8;		mov	r25, #4
.LBB2_31:                               // %for.body201
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB2_32 Depth 2
	eq	 r27, r23, r25
	lshl	 r20, r6, r2
	movxm	p0, ##_ZL6acc_hi
	mul	r19, r17, r0
	mac	r19, r19, r18, r1
	add	 r21, r20, r3
	movs	p1, p6;		lshl	 r19, r19, r4;		mov	m0, r20
	padda	 [p1], m0;		movs	m0, r21;		sel.nez	 r19, r7, r19, r27
	padda	 [p0], m0;		ltu	 r27, r16, r8;		mov	r20, p1
	lshl	 r19, r19, r5;		mov	r21, p0
	sel.nez	 r20, r20, r21, r27
	movs	p2, p6;		lshl	 r19, r22, r5;		mov	m0, r19
	padda	 [p2], m0;		movs	p1, r20;		mov	m0, r19
	padda	 [p1], m0
	vlda.conv.fp32.bf16	 cml1, [p1, #0]
	vlda.conv.fp32.bf16	 cmh1, [p1, #64]
	nop	
	nop	
	add.nc	lc, r28, #-1
	movxm	ls, #.LBB2_32
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd9;		nopv	
	mova	r26, #0;		nopb	;		nops	;		add	 r19, r22, r11;		mov	p0, r20;		nopv	
.LBB2_32:                               // %for.body231
                                        //   Parent Loop BB2_31 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		nopxm	;		nopv	
	nopa	;		nopb	;		vst.flush.512.conv	 [p2, sf, r26];		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		lshl	 r20, r19, r5;		nopm	;		nopv	
	movs	p1, p0;		nopx	;		mov	m0, r20
	padda	 [p1], m0
	vlda.conv.fp32.bf16	 cmh1, [p1, #64]
	vlda.conv.fp32.bf16	 cml1, [p1, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
.L_LEnd9:
	nopa	;		nopb	;		nops	;		add	 r19, r19, r11;		nopm	;		nopv	
// %bb.33:                              //   in Loop: Header=BB2_31 Depth=1
	nopa	;		add	r18, r18, #1;		nopm	
	add	r16, r16, #1
	eq	 r19, r16, r12
	jz	 r19, #.LBB2_31
	add	 r7, r7, r24                    //  Delay Slot 5
	add	 r6, r6, r9                     //  Delay Slot 4
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		eq	 r27, r18, r8 //  Delay Slot 3
	vst.flush.512.conv	 [p2, sf, r26];		add	 r17, r17, r27 //  Delay Slot 2
	sel.nez	 r18, r22, r18, r27             //  Delay Slot 1
.LBB2_34:                               // %cleanup
	lda	 p7, [sp, #-20];		nopx	        // 4-byte Folded Reload
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
	paddxm	 [sp], #-832                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end2:
	.size	acc_epilogue, .Lfunc_end2-acc_epilogue
                                        // -- End function
	.section	.text._ZL9bias_passPKfP8bfloat16PKS1_ii,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL9bias_passPKfP8bfloat16PKS1_ii
	.type	_ZL9bias_passPKfP8bfloat16PKS1_ii,@function
_ZL9bias_passPKfP8bfloat16PKS1_ii:      // @_ZL9bias_passPKfP8bfloat16PKS1_ii
// %bb.0:                               // %entry
	nopa	;		vldb	 x0, [p2, #0];		nopxm	;		nops	
	nop	
	mova	m0, #512
	padda	 [p2], m0
	vldb	 x0, [p2], #-448
	vldb	 x0, [p2], #64
	add.nc	lc, r0, #-1
	vmov	lfh0, x0
	mova	r2, #1;		vldb	 x0, [p2], #64;		paddxm	 [sp], #512
	vldb	 x0, [p2], #64;		lshl	 r1, r1, r2;		vmov	x10, lfh0
	vlda	 bmll1, [p0], #64;		vldb	 x0, [p2], #64;		movxm	r2, #16256
	vlda	 bmlh1, [p0], #64;		vmov	lfh1, x0
	vlda	 bmll0, [p0], #64;		vldb	 x1, [p2], #64;		vmov	lfl0, x0
	vlda	 bmlh0, [p0], #64;		vldb	 x3, [p2], #64;		vbcst.16	 x8, r2
	vlda	 bmll2, [p0], #64;		vldb	 x5, [p2], #128;		movx	r0, #828;		vmov	x10, lfl0
	vlda	 bmlh2, [p0], #64;		vldb	 x7, [p2], #64;		vmov	lfl1, x0;		vmac.f	dm1, dm1, x10, x8, r0
	vlda	 bmll3, [p0], #64;		vldb	 x9, [p2], #64;		vmov	sfl, x0
	vlda	 bmlh3, [p0], #64;		vldb	 x11, [p2], #64;		vmov	sfh, x0
	vlda	 bmll4, [p0], #64;		vldb	 x0, [p2], #64;		vmov	x10, lfl1
	vlda	 bmlh4, [p0], #64;		vldb	 x2, [p2], #64;		vmov	x10, sfl;		vmac.f	dm1, dm0, x10, x8, r0
	vlda	 bmll0, [p0], #64;		vldb	 x4, [p2, #0];		vmov	x10, sfh;		vmac.f	dm0, dm2, x10, x8, r0
	vlda	 bmlh0, [p0], #64;		vldb	 x6, [p2, #64];		vst	 bmll1, [sp, #-256];		nopx	;		mov	p2, p1;		vmac.f	dm2, dm3, x10, x8, r0 // 64-byte Folded Spill
	vlda	 bmll4, [sp, #-256];		paddb	 [p1], #64;		vst	 bmlh1, [sp, #-192];		vmov	x10, lfh1 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh4, [sp, #-192];		vst	 bmhh1, [sp, #-64];		vmac.f	dm3, dm4, x10, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll1, [p0], #64;		vst	 bmhl1, [sp, #-128] // 64-byte Folded Spill
	vlda	 bmlh1, [p0], #64;		vst.conv.bf16.fp32	 cml1, [p1], #64;		vmac.f	dm0, dm0, x1, x8, r0
	vlda	 bmll2, [p0], #64;		vst.conv.bf16.fp32	 cml0, [p1], #64
	vlda	 bmlh2, [p0], #64;		vst.conv.bf16.fp32	 cml2, [p1], #64
	vlda	 bmll3, [p0], #64
	vlda	 bmlh3, [p0], #64;		vst.conv.bf16.fp32	 cml3, [p1], #64;		mov	m0, r1;		vmac.f	dm4, dm1, x3, x8, r0
	vlda	 bmll0, [p0], #64;		vst.conv.bf16.fp32	 cml4, [p2], m0
	vlda	 bmlh0, [p0], #64;		vst	 bmlh0, [sp, #-192];		vmac.f	dm2, dm2, x5, x8, r0 // 64-byte Folded Spill
	vlda	 bmlh0, [sp, #-192];		vst	 bmll0, [sp, #-256] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll0, [sp, #-256];		vst	 bmhl0, [sp, #-128];		vmac.f	dm3, dm3, x10, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll0, [p0], #64;		vst	 bmhh0, [sp, #-64] // 64-byte Folded Spill
	vlda	 bmlh0, [p0], #64;		vmac.f	dm1, dm0, x7, x8, r0
	vlda	 bmhl0, [sp, #-128]             // 64-byte Folded Reload
	vlda	 bmhh0, [sp, #-64]              // 64-byte Folded Reload
	vlda	 bmll1, [p0], #64
	vlda	 bmlh1, [p0], #64
	vlda	 bmll2, [p0], #64;		vst.conv.bf16.fp32	 cml0, [p1], #64
	vlda	 bmlh2, [p0], #64;		vst.conv.bf16.fp32	 cml4, [p1], #64;		vmac.f	dm0, dm0, x9, x8, r0
	vlda	 bmll3, [p0], #64;		vst.conv.bf16.fp32	 cml2, [p1], #64
	vlda	 bmlh3, [p0], #64;		vst.conv.bf16.fp32	 cml3, [p1], #64;		vmac.f	dm1, dm1, x11, x8, r0
	vlda	 bmll4, [p0], #64;		vst.conv.bf16.fp32	 cml1, [p1], #64
	vlda	 bmlh4, [p0], #64
	vlda	 bmll0, [p0], #64
	vlda	 bmlh0, [p0], #64;		vst	 bmlh0, [sp, #-192];		vmac.f	dm1, dm2, x0, x8, r0 // 64-byte Folded Spill
	vlda	 bmlh4, [sp, #-192];		vst	 bmll0, [sp, #-256];		vmac.f	dm2, dm3, x2, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll4, [sp, #-256];		vst	 bmlh1, [sp, #-448];		vmac.f	dm3, dm4, x4, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh4, [sp, #-448];		vst	 bmll1, [sp, #-512] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll4, [sp, #-512];		nopb	;		vst	 bmhh1, [sp, #-320];		movxm	ls, #.LBB3_1;		vmac.f	dm0, dm0, x6, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vst	 bmhl1, [sp, #-384];		movxm	le, #.L_LEnd10 // 64-byte Folded Spill
.LBB3_1:                                // %for.cond1.preheader
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		vst	 bmhl0, [sp, #-128];		nopx	;		vmov	x10, lfh0;		nopv	 // 64-byte Folded Spill
	vlda	 bmll1, [p0], #64;		nopb	;		nopx	;		vst	 bmhh0, [sp, #-64] // 64-byte Folded Spill
	vlda	 bmlh1, [p0], #64
	vlda	 bmll0, [p0], #64;		vst.conv.bf16.fp32	 cml4, [p1], #64
	vlda	 bmlh0, [p0], #64
	vlda	 bmll2, [p0], #64;		vst.conv.bf16.fp32	 cml4, [p1], #64;		vmov	x10, lfl0
	vlda	 bmlh2, [p0], #64;		vst.conv.bf16.fp32	 cml1, [p1], #64;		vmac.f	dm1, dm1, x10, x8, r0
	vlda	 bmll3, [p0], #64;		vst.conv.bf16.fp32	 cml2, [p1], #64
	vlda	 bmlh3, [p0], #64;		vst.conv.bf16.fp32	 cml3, [p1, #0]
	vlda	 bmll4, [p0], #64;		vst.conv.bf16.fp32	 cml0, [p1, #64];		vmov	x10, lfl1
	vlda	 bmlh4, [p0], #64;		vmov	x10, sfl;		vmac.f	dm1, dm0, x10, x8, r0
	vlda	 bmll0, [p0], #64;		vmov	x10, sfh;		vmac.f	dm0, dm2, x10, x8, r0
	vlda	 bmlh0, [p0], #64;		vst	 bmll1, [sp, #-256];		mov	p1, p2;		vmac.f	dm2, dm3, x10, x8, r0 // 64-byte Folded Spill
	vlda	 bmll4, [sp, #-256];		paddb	 [p1], #64;		vst	 bmlh1, [sp, #-192];		vmov	x10, lfh1 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh4, [sp, #-192];		vst	 bmhh1, [sp, #-64];		vmac.f	dm3, dm4, x10, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll1, [p0], #64;		vst	 bmhl1, [sp, #-128] // 64-byte Folded Spill
	vlda	 bmlh1, [p0], #64;		vst.conv.bf16.fp32	 cml1, [p1], #64;		vmac.f	dm0, dm0, x1, x8, r0
	vlda	 bmll2, [p0], #64;		vst.conv.bf16.fp32	 cml0, [p1], #64
	vlda	 bmlh2, [p0], #64;		vst.conv.bf16.fp32	 cml2, [p1], #64
	vlda	 bmll3, [p0], #64
	vlda	 bmlh3, [p0], #64;		vst.conv.bf16.fp32	 cml3, [p1], #64;		vmac.f	dm4, dm1, x3, x8, r0
	vlda	 bmll0, [p0], #64;		vst.conv.bf16.fp32	 cml4, [p2], m0
	vlda	 bmlh0, [p0], #64;		vst	 bmlh0, [sp, #-192];		vmac.f	dm2, dm2, x5, x8, r0 // 64-byte Folded Spill
	vlda	 bmlh0, [sp, #-192];		vst	 bmll0, [sp, #-256] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll0, [sp, #-256];		vst	 bmhl0, [sp, #-128];		vmac.f	dm3, dm3, x10, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll0, [p0], #64;		vst	 bmhh0, [sp, #-64] // 64-byte Folded Spill
	vlda	 bmlh0, [p0], #64;		vmac.f	dm1, dm0, x7, x8, r0
	vlda	 bmhl0, [sp, #-128]             // 64-byte Folded Reload
	vlda	 bmhh0, [sp, #-64]              // 64-byte Folded Reload
	vlda	 bmll1, [p0], #64
	vlda	 bmlh1, [p0], #64
	vlda	 bmll2, [p0], #64;		vst.conv.bf16.fp32	 cml0, [p1], #64
	vlda	 bmlh2, [p0], #64;		vst.conv.bf16.fp32	 cml4, [p1], #64;		vmac.f	dm0, dm0, x9, x8, r0
	vlda	 bmll3, [p0], #64;		vst.conv.bf16.fp32	 cml2, [p1], #64
	vlda	 bmlh3, [p0], #64;		vst.conv.bf16.fp32	 cml3, [p1], #64;		vmac.f	dm1, dm1, x11, x8, r0
	vlda	 bmll4, [p0], #64;		vst.conv.bf16.fp32	 cml1, [p1], #64
	vlda	 bmlh4, [p0], #64
	vlda	 bmll0, [p0], #64
	vlda	 bmlh0, [p0], #64;		vst	 bmlh0, [sp, #-192];		vmac.f	dm1, dm2, x0, x8, r0 // 64-byte Folded Spill
	vlda	 bmlh4, [sp, #-192];		vst	 bmll0, [sp, #-256];		vmac.f	dm2, dm3, x2, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll4, [sp, #-256];		vst	 bmlh1, [sp, #-448];		vmac.f	dm3, dm4, x4, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh4, [sp, #-448];		vst	 bmll1, [sp, #-512] // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll4, [sp, #-512];		vst	 bmhh1, [sp, #-320];		vmac.f	dm0, dm0, x6, x8, r0 // 64-byte Folded Reload64-byte Folded Spill
.L_LEnd10:
	nopa	;		nopb	;		vst	 bmhl1, [sp, #-384];		nopxm	;		nopv	 // 64-byte Folded Spill
// %bb.2:                               // %for.cond.cleanup
	nopa	;		nopb	;		vst	 bmhl0, [sp, #-128];		nopxm	;		nopv	 // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-64];		nopx	 // 64-byte Folded Spill
	paddxm	 [sp], #-512
	vst.conv.bf16.fp32	 cml4, [p1], #64
	nop	
	vst.conv.bf16.fp32	 cml4, [p1], #64
	vst.conv.bf16.fp32	 cml1, [p1], #64
	vst.conv.bf16.fp32	 cml2, [p1], #64;		ret	lr
	vst.conv.bf16.fp32	 cml3, [p1, #0] //  Delay Slot 5
	vst.conv.bf16.fp32	 cml0, [p1, #64] //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mov	p1, p2                          //  Delay Slot 1
.Lfunc_end3:
	.size	_ZL9bias_passPKfP8bfloat16PKS1_ii, .Lfunc_end3-_ZL9bias_passPKfP8bfloat16PKS1_ii
                                        // -- End function
	.section	.text._ZL10gelu_lut_nP8bfloat16iPKfS2_,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL10gelu_lut_nP8bfloat16iPKfS2_
	.type	_ZL10gelu_lut_nP8bfloat16iPKfS2_,@function
_ZL10gelu_lut_nP8bfloat16iPKfS2_:       // @_ZL10gelu_lut_nP8bfloat16iPKfS2_
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	r16, #52428;		nopv	
	movxm	r2, #-16512
	mov	r3, p1
	movxm	ls, #.LBB4_1
	movxm	le, #.L_LEnd11
	mova	r1, #512;		mov	s0, #7
	vbcst.32	 x6, r1
	movxm	r1, #16511
	vbcst.32	 x8, r3
	mov	r3, p2
	vbcst.16	 x0, r1
	movxm	r1, #-16256
	vbcst.32	 x10, r3
	vmov	bmll2, x0
	vbcst.16	 x0, r1
	vsel.32	 x8, x8, x10, r16
	mova	r1, #5;		vmov	bmll3, x0
	mova	r4, #-6;		lshl	 r0, r0, r1;		vbcst.16	 x0, r2
	mova	r2, #0;		add	r3, r0, #-1;		vmov	bmll4, x0
	mova	r0, #24;		lshl	 r4, r3, r4;		vadd.32	 x0, x8, x6
	mova	r3, #828;		add	r5, r4, #1;		vmov	bmlh2, x0
	mova	r5, #1;		movx	r4, #64;		add.nc	lc, r5, #0
.LBB4_1:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		lshl	 r6, r2, r5;		nopm	;		movs	p1, p0
	movs	m0, r6
	padda	 [p1], m0
	vlda	 x8, [p1, #0]
	nop	
	nop	
	nop	
	nop	
	vmov	x0, bmll2
	vmov	x2, bmll3
	vmin_ge.bf16	 x9, r16, x8, x0
	vmax_lt.bf16	 x10, r16, x9, x2
	nop	
	vlda	 x1, [p1, #64];		vfloor.s32.bf16	x7, wh10, s0
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
	vldb.4x64.hi	 wh4, wh4;		vshuffle	bmll1, x5, x7, r1
	vshuffle	x5, x5, x7, r0
	vshuffle	bmlh1, x9, x2, r1
	vshuffle	x2, x9, x2, r0
	vmov	wl2, wh5
	vshuffle	x2, x11, x0, r0
	vshuffle	bmlh0, x11, x0, r1;		vmac.f	dm1, dm1, x2, x10, r3
	vshuffle	x0, x6, x4, r0
	vmov	wl2, wh0
	nop	
	vmac.f	dm0, dm0, x2, x3, r3
	vshuffle	bmll0, x6, x4, r1
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml1;		vmov	x4, bmll4
	vconv.bf16.fp32	 x9, cml0
	vmac.f	dm1, dm1, x0, x4, r3
	vmac.f	dm0, dm0, x9, x4, r3
	nop	
	nop	
	nop	
	vmul.f	dm1, x8, x0, r3
	vconv.bf16.fp32	 x4, cml1;		vmul.f	dm0, x1, x9, r3
	vconv.bf16.fp32	 x0, cml0
	vmac.f	dm1, dm1, x8, x4, r3
	vmac.f	dm0, dm0, x1, x0, r3
	nop	
	nop	
	nop	
	nop	
	vst.conv.bf16.fp32	 cml1, [p1, #0]
.L_LEnd11:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml0, [p1, #64];		add	 r2, r2, r4;		nopm	;		nopv	
// %bb.2:                               // %for.cond.cleanup
	nopa	;		ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end4:
	.size	_ZL10gelu_lut_nP8bfloat16iPKfS2_, .Lfunc_end4-_ZL10gelu_lut_nP8bfloat16iPKfS2_
                                        // -- End function
	.section	.text.attn_begin,"ax",@progbits
	.globl	attn_begin                      // -- Begin function attn_begin
	.p2align	4
	.type	attn_begin,@function
attn_begin:                             // @attn_begin
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	r1, #-15744
	movxm	p4, ##_ZN2atL5g_srbE
	movxm	p2, ##_ZL6acc_hi
	movxm	p1, ##_ZN2atL3g_qE
	movxm	p3, ##_ZL6acc_hi
	movxm	m0, #12288
	movxm	m3, #15744
	movxm	m1, #15360
	movxm	m2, #21888
	movxm	r7, #1045504
	mov	crrnd, #12
	movs	p5, p0;		vbcst.16	 x0, r1
	padda	 [p2], m0;		paddb	 [p5], m1;		movxm	m1, #19200
	padda	 [p3], m3;		st	 p2, [p1, #0];		movxm	p2, ##_ZL6acc_hi
	st	 p3, [p1, #4];		movxm	p3, ##_ZN2atL5g_erbE
	mova	r4, #192;		movxm	p1, ##_ZL6acc_hi
	padda	 [p1], m2;		st	 p5, [p4, #0];		movxm	p4, ##_ZN2atL6g_osumE
	padda	 [p2], m1;		st	 p1, [p3, #4];		movxm	m1, #16128
	mova	r18, #96;		movx	r17, #95;		mov	p1, p0
	padda	 [p1], m1;		st	 p2, [p3, #0];		movxm	p2, ##_ZN2atL6g_maxbE
	mova	r16, #768;		st	 p1, [p2, #0];		movxm	p3, ##_ZN2atL3g_oE
	mova	r3, #127;		movx	r2, #432;		mov	p2, p0
	padda	 [p2], m0;		movxm	m0, #16640
	mova	r5, #64;		st	 p0, [p3, #0];		j	#.LBB5_2
	padda	 [p0], m0;		st	 p2, [p4, #0];		movxm	p4, ##_ZN2atL6g_onesE //  Delay Slot 5
	mova	r6, #8;		st	 p0, [p4, #0];		movxm	p4, ##_ZL6acc_hi //  Delay Slot 4
	mova	r19, #0;		st	 p4, [p3, #4];		movxm	p4, ##_ZN2atL8g_ntilesE //  Delay Slot 3
	mova	r1, #0;		st	 r0, [p4, #0];		movxm	p4, ##_ZN2atL7g_chunkE //  Delay Slot 2
	mova	r0, #2;		st	 r1, [p4, #0];		movx	r20, #0;		vbcst.32	 x2, r1 //  Delay Slot 1
.LBB5_1:                                // %for.body
                                        //   in Loop: Header=BB5_2 Depth=1
	nopa	;		nopb	;		ltu	 r21, r17, r20;		nopm	;		nops	
	lshl	 r21, r21, r0
	mov	dj0, r21
	lda	 p4, [p3, dj0]
	nop	
	ltu	 r27, r20, r18
	add	 r21, r19, r7
	sel.nez	 r21, r19, r21, r27
	lshl	 r21, r21, r0
	mov	m0, r21
	padda	 [p4], m0
	vst	 x2, [p4, #0];		add	r20, r20, #1
	vst	 x2, [p4, #64];		add	r19, r19, #32
.LBB5_2:                                // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		ne	 r21, r20, r4;		nopm	
	jnz	 r21, #.LBB5_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:
	j	#.LBB5_5
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r7, #0;		movx	r0, #2          //  Delay Slot 1
.LBB5_4:                                // %for.body11
                                        //   in Loop: Header=BB5_5 Depth=1
	nopa	;		lshl	 r17, r7, r0;		nopm	
	movs	p3, p2;		mov	m0, r17
	padda	 [p3], m0
	vst	 x2, [p3, #0]
	vst	 x2, [p3, #64];		add	r7, r7, #32
.LBB5_5:                                // %for.cond8
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		ne	 r17, r7, r16;		nopm	
	jnz	 r17, #.LBB5_4
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.6:
	j	#.LBB5_8
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r7, #0;		movx	r0, #1          //  Delay Slot 1
.LBB5_7:                                // %for.body24
                                        //   in Loop: Header=BB5_8 Depth=1
	nopa	;		lshl	 r16, r7, r0
	mov	dj0, r16
	vst	 x0, [p1, dj0];		add	r7, r7, #32
.LBB5_8:                                // %for.cond21
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		ne	 r16, r7, r4;		nopm	
	jnz	 r16, #.LBB5_7
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.9:
	j	#.LBB5_11
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r27, #0;		movx	r0, #0          //  Delay Slot 1
.LBB5_10:                               // %for.body37
                                        //   in Loop: Header=BB5_11 Depth=1
	nopa	;		nopb	;		nops	;		nopx	;		mov	dj0, r0;		nopv	
	st.s8	 r4, [p0, dj0];		nopx	
	nop	
	nop	
	nop	
	add	r0, r0, #1
	sel.eqz	 r4, r3, r5, r27
	add	r4, r27, #1
	eq	 r27, r27, r6
	sel.nez	 r27, r1, r4, r27
.LBB5_11:                               // %for.cond34
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		ne	 r4, r0, r2;		nopm	;		nopv	
	jnz	 r4, #.LBB5_10
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.12:                              // %for.cond.cleanup36
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end5:
	.size	attn_begin, .Lfunc_end5-attn_begin
                                        // -- End function
	.section	.text.attn_qcopy,"ax",@progbits
	.globl	attn_qcopy                      // -- Begin function attn_qcopy
	.p2align	4
	.type	attn_qcopy,@function
attn_qcopy:                             // @attn_qcopy
// %bb.0:                               // %entry
	nopa	;		ne	 r0, r0, r3;		nopm	
	jnz	 r0, #.LBB6_8
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
	jnz	 r4, #.LBB6_8
	padda	 [p3], m0;		movxm	m1, #15744 //  Delay Slot 5
	padda	 [p4], m1;		st	 p3, [p7, #0];		movxm	m2, #19200 //  Delay Slot 4
	padda	 [p6], m2;		st	 p4, [p7, #4];		movxm	m0, #21888 //  Delay Slot 3
	padda	 [p2], m0;		st	 p6, [p5, #0] //  Delay Slot 2
	st	 p2, [p5, #4]                   //  Delay Slot 1
// %bb.2:                               // %if.end7
	mov	p4, sp
	j	#.LBB6_6
	padda	 [p4], #-64                     //  Delay Slot 5
	movs	p2, p0;		sub	 r1, r1, r3;		mov	p0, p4 //  Delay Slot 4
	st	 p2, [p0], #4;		lshl	 r1, r1, r2 //  Delay Slot 3
	mova	r3, #288;		st	 r7, [p0, #0];		mov	dj0, r1 //  Delay Slot 2
	lda	 p3, [p7, dj0];		st	 r6, [p0, #4];		movx	r1, #0;		mov	r4, #0 //  Delay Slot 1
.LBB6_3:                                // %for.body13
                                        //   in Loop: Header=BB6_4 Depth=2
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
.LBB6_4:                                // %for.cond10
                                        //   Parent Loop BB6_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		ne	 r6, r5, r3
	jnz	 r6, #.LBB6_3
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.5:                               // %for.cond.cleanup12
                                        //   in Loop: Header=BB6_6 Depth=1
	add	r4, r4, #1
	add	 r1, r1, r3
.LBB6_6:                                // %for.cond
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_4 Depth 2
	nopa	;		nopb	;		eq	 r5, r4, r0;		nopm	;		nops	
	jnz	 r5, #.LBB6_8
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.7:                               // %for.cond10.preheader
                                        //   in Loop: Header=BB6_6 Depth=1
	j	#.LBB6_4
	nop	                                //  Delay Slot 5
	lshl	 r5, r4, r2                     //  Delay Slot 4
	lshl	 r5, r1, r2;		mov	dj0, r5 //  Delay Slot 3
	lda	 p0, [p4, dj0];		movs	p1, p3;		mov	m0, r5 //  Delay Slot 2
	padda	 [p1], m0;		movx	r5, #0  //  Delay Slot 1
.LBB6_8:                                // %cleanup.cont
	lda	 p7, [sp, #-48];		nopxm	        // 4-byte Folded Reload
	lda	 p6, [sp, #-52]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end6:
	.size	attn_qcopy, .Lfunc_end6-attn_qcopy
                                        // -- End function
	.section	.text.attn_chunk_k,"ax",@progbits
	.globl	attn_chunk_k                    // -- Begin function attn_chunk_k
	.p2align	4
	.type	attn_chunk_k,@function
attn_chunk_k:                           // @attn_chunk_k
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		paddxm	 [sp], #384;		nopv	
	st	 p7, [sp, #-40];		nopb	;		nopx	 // 4-byte Folded Spill
	st	 r8, [sp, #-60]                 // 4-byte Folded Spill
	st	 lr, [sp, #-64]                 // 4-byte Folded Spill
	st	 r9, [sp, #-56]                 // 4-byte Folded Spill
	st	 r10, [sp, #-52]                // 4-byte Folded Spill
	st	 r11, [sp, #-48];		movxm	p1, ##_ZN2atL7g_chunkE // 4-byte Folded Spill
	lda	 r0, [p1, #0];		st	 p6, [sp, #-44];		event	#0 // 4-byte Folded Spill
	nop	
	nop	
	nop	
	nop	
	mova	r1, #-48
	movxm	r2, #1072
	mac	r2, r2, r0, r1
	movx	r8, #0
	lt	 r27, r8, r2
	mova	r3, #47;		sel.nez	 r0, r2, r8, r27
	mova	r1, #-3;		lt	 r27, r3, r2
	mova	r1, #6;		lshl	 r0, r0, r1
	sel.nez	 r9, r1, r0, r27
	jz	 r9, #.LBB7_4
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	p7, sp                          //  Delay Slot 2
	padda	 [p7], #-384                    //  Delay Slot 1
// %bb.1:                               // %if.then
	movs	p6, p0;		movxm	p0, ##_ZN2atL8g_ntilesE
	movxm	r0, #1065353216
	vbcst.32	 x1, r0
	lda	 r10, [p0, #0];		movxm	r0, #1060372480
	movs	p0, p7;		j	#.LBB7_3
	vst	 x1, [p0], #128;		vbcst.32	 x2, r0 //  Delay Slot 5
	vst	 x1, [p7, #64];		movxm	r0, #16256 //  Delay Slot 4
	vst	 x2, [p0, #0];		vbcst.16	 x0, r0 //  Delay Slot 3
	vst	 x2, [p0, #64]                  //  Delay Slot 2
	vst	 x0, [p0, #128];		movx	r11, #2 //  Delay Slot 1
.LBB7_2:                                // %for.body
                                        //   in Loop: Header=BB7_3 Depth=1
	nopa	;		nopb	;		lshl	 r0, r8, r11
	movxm	p0, ##_ZN2atL3g_qE
	mov	dj0, r0
	lda	 p0, [p0, dj0]
	jl	#_ZN2atL7chunk_kEiPK9bfp16ebs8S2_iRKN3nk84ExpQE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mov	p1, p6                          //  Delay Slot 2
	movs	p2, p7;		or	 r1, r9, r9;		mov	r0, r8 //  Delay Slot 1
	nopa	;		nopb	;		nops	;		add	r8, r8, #1;		nopm	;		nopv	
.LBB7_3:                                // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		lt	 r0, r8, r10;		nopm	;		nopv	
	jnz	 r0, #.LBB7_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB7_4:                                // %if.end
	lda	 lr, [sp, #-64];		event	#1      // 4-byte Folded Reload
	lda	 p7, [sp, #-40]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-44]                 // 4-byte Folded Reload
	lda	 r11, [sp, #-48]                // 4-byte Folded Reload
	lda	 r10, [sp, #-52]                // 4-byte Folded Reload
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-384                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end7:
	.size	attn_chunk_k, .Lfunc_end7-attn_chunk_k
                                        // -- End function
	.section	.text._ZN2atL7chunk_kEiPK9bfp16ebs8S2_iRKN3nk84ExpQE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN2atL7chunk_kEiPK9bfp16ebs8S2_iRKN3nk84ExpQE
	.type	_ZN2atL7chunk_kEiPK9bfp16ebs8S2_iRKN3nk84ExpQE,@function
_ZN2atL7chunk_kEiPK9bfp16ebs8S2_iRKN3nk84ExpQE: // @_ZN2atL7chunk_kEiPK9bfp16ebs8S2_iRKN3nk84ExpQE
// %bb.0:                               // %entry
	nopa	;		nopb	;		paddxm	 [sp], #1088
	st	 r12, [sp, #-64]                // 4-byte Folded Spill
	st	 r13, [sp, #-60]                // 4-byte Folded Spill
	st	 r14, [sp, #-56]                // 4-byte Folded Spill
	st	 r9, [sp, #-76]                 // 4-byte Folded Spill
	st	 r10, [sp, #-72]                // 4-byte Folded Spill
	st	 r15, [sp, #-52]                // 4-byte Folded Spill
	st	 r8, [sp, #-80]                 // 4-byte Folded Spill
	st	 p6, [sp, #-48]                 // 4-byte Folded Spill
	st	 p7, [sp, #-44]                 // 4-byte Folded Spill
	st	 lr, [sp, #-84]                 // 4-byte Folded Spill
	st	 r11, [sp, #-68];		movxm	r3, #15872 // 4-byte Folded Spill
	st	 r0, [sp, #-96];		movxm	r4, #16640 // 4-byte Folded Spill
	mova	r2, #96;		st	 p0, [sp, #-108];		vbcst.16	 x2, r4 // 4-byte Folded Spill
	mova	r3, #2;		st	 p2, [sp, #-104];		mul	r18, r0, r2;		vbcst.16	 x0, r3 // 4-byte Folded Spill
	vst	 x2, [sp, #-704];		movxm	r2, #-16512 // 64-byte Folded Spill
	mova	r5, #6;		nopb	;		vst	 x0, [sp, #-512];		lshl	 r2, r0, r3;		vbcst.16	 x2, r2;		vclr	dm0 // 64-byte Folded Spill
	vst	 x0, [sp, #-448];		sub	 r0, r5, r1 // 64-byte Folded Spill
	st	 r0, [sp, #-92]                 // 4-byte Folded Spill
	st	 r2, [sp, #-100];		movxm	p4, ##_ZN2atL6g_maxbE // 4-byte Folded Spill
	vst	 x2, [sp, #-576];		lshl	 r0, r1, r5 // 64-byte Folded Spill
	lda	 p4, [p4, #0];		st	 r0, [sp, #-88];		movxm	p3, ##_ZN2atL5g_srbE // 4-byte Folded Spill
	lda	 p3, [p3, #0];		vst	 bmll0, [sp, #-384] // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-320]             // 64-byte Folded Spill
	mova	r4, #0;		vst	 bmhl0, [sp, #-256];		movxm	r7, #1152 // 64-byte Folded Spill
	mova	r0, #0;		vst	 bmhh0, [sp, #-192];		movx	r6, #576;		vbcst.16	 x2, r4 // 64-byte Folded Spill
	mova	r19, #16;		st	 r0, [sp, #-128];		movx	r17, #128;		mov	p7, p1 // 4-byte Folded Spill
	mova	r14, #828;		vst	 x2, [sp, #-640];		movx	r13, #60;		mov	r12, #780 // 64-byte Folded Spill
	lda	 p6, [p3, #0];		st	 p4, [sp, #-112];		movx	r10, #64;		mov	r9, r1 // 4-byte Folded Spill
	mova	r3, #0;		st	 p3, [sp, #-116];		movx	r8, #-128;		mov	r15, #1 // 4-byte Folded Spill
.LBB8_1:                                // %_ZN3aie41block_vector_restrict_input_buffer_streamI9bfp16ebs8Lj64EL15aie_dm_resource5EE3popEv.exit.7
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_4 Depth 2
                                        //     Child Loop BB8_11 Depth 2
                                        //     Child Loop BB8_13 Depth 2
	lda	 p0, [sp, #-108];		nopb	;		nops	;		nopxm	;		nopv	 // 4-byte Folded Reload
	nopx	
	nop	
	nop	
	nop	
	nop	
	mov	m0, r4
	padda	 [p0], m0;		movx	r24, #0
	vldb.fill.512	 [p0, lf0, r24]
	vldb.pop.576	 ex0, [p0, lf0, r24]
	nop	
	nop	
	vldb.pop.576	 ex0, [p0, lf0, r24]
	nop	
	nop	
	vldb.pop.576	 ex0, [p0, lf0, r24];		st	 r4, [sp, #-120] // 4-byte Folded Spill
	st	 r3, [sp, #-124]                // 4-byte Folded Spill
	vst	 x0, [sp, #-1088]               // 64-byte Folded Spill
	st	 el0, [sp, #-1024]              // 4-byte Folded Spill
	st	 eh0, [sp, #-1020]              // 4-byte Folded Spill
	vst	 x0, [sp, #-960];		j	#.LBB8_4 // 64-byte Folded Spill
	vldb.pop.576	 ex6, [p0, lf0, r24];		st	 el0, [sp, #-896] // 4-byte Folded Spill Delay Slot 5
	vldb.pop.576	 ex2, [p0, lf0, r24];		st	 eh0, [sp, #-892] // 4-byte Folded Spill Delay Slot 4
	vldb.pop.576	 ex3, [p0, lf0, r24];		vst	 x0, [sp, #-832] // 64-byte Folded Spill Delay Slot 3
	vldb.pop.576	 ex5, [p0, lf0, r24];		st	 el0, [sp, #-768] // 4-byte Folded Spill Delay Slot 2
	mova	r5, #0;		vldb.pop.576	 ex7, [p0, lf0, r24];		movx	r4, #0;		mov	r0, #64;		st	 eh0, [sp, #-764] // 4-byte Folded Spill Delay Slot 1
.LBB8_2:                                // %cond.false
                                        //   in Loop: Header=BB8_4 Depth=2
	vmov	x4, sfh
	vmov	x0, sfl
	vmax_lt.bf16	 x4, r16, x4, x8
	vmax_lt.bf16	 x9, r16, x0, x9
.LBB8_3:                                // %cond.end
                                        //   in Loop: Header=BB8_4 Depth=2
	nopa	;		nopb	;		add	r5, r5, #2
	ltu	 r1, r5, r9
	jz	 r1, #.LBB8_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	add	 r4, r3, r7                     //  Delay Slot 2
	add	 r0, r0, r17                    //  Delay Slot 1
.LBB8_4:                                // %if.end47.3
                                        //   Parent Loop BB8_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		nopx	;		mov	p1, p7
	mova	r2, #0;		movs	m0, r4;		add	 r1, r4, r6;		mov	p0, p7
	padda	 [p1], m0;		movs	m0, r1;		mov	r25, r2
	padda	 [p0], m0;		vldb.fill.512	 [p1, lf1, r25];		mov	r24, r2
	vlda	 x10, [sp, #-1088];		vldb.fill.512	 [p0, lf0, r24] // 64-byte Folded Reload
	lda	 el10, [sp, #-1024];		vldb.pop.576	 ex8, [p1, lf1, r25] // 4-byte Folded Reload
	lda	 eh10, [sp, #-1020];		vldb.pop.576	 ex8, [p0, lf0, r24] // 4-byte Folded Reload
	vldb.pop.576	 ex0, [p1, lf1, r25]
	vlda	 x8, [sp, #-832];		vldb.pop.576	 ex4, [p1, lf1, r25] // 64-byte Folded Reload
	lda	 el8, [sp, #-768];		vldb.pop.576	 ex9, [p0, lf0, r24] // 4-byte Folded Reload
	lda	 eh8, [sp, #-764];		vldb.pop.576	 ex4, [p0, lf0, r24] // 4-byte Folded Reload
	nop	
	vlda	 x4, [sp, #-960]                // 64-byte Folded Reload
	lda	 el4, [sp, #-896];		vmul.f	dm0, ex10, ex8, r12 // 4-byte Folded Reload
	lda	 eh4, [sp, #-892];		vmul.f	dm1, ex10, ex8, r12 // 4-byte Folded Reload
	vmov	sfh, x4
	vldb.pop.576	 ex0, [p1, lf1, r25];		vmov	sfl, x9
	vldb.pop.576	 ex0, [p0, lf0, r24];		vmac.f	dm0, dm0, ex8, ex4, r12
	vldb.pop.576	 ex0, [p1, lf1, r25];		vmac.f	dm2, dm1, ex8, ex4, r12
	vldb.pop.576	 ex0, [p0, lf0, r24]
	vldb.pop.576	 ex0, [p1, lf1, r25]
	vldb.pop.576	 ex0, [p0, lf0, r24];		vmul.f	dm1, ex4, ex0, r12
	vldb.pop.576	 ex0, [p1, lf1, r25];		vmul.f	dm1, ex4, ex9, r12
	vldb.pop.576	 ex8, [p1, lf1, r25]
	vmac.f	dm3, dm1, ex6, ex0, r12
	vldb.pop.576	 ex4, [p0, lf0, r24];		vmac.f	dm4, dm1, ex6, ex0, r12
	vmac.f	dm1, dm0, ex2, ex0, r12
	vldb.pop.576	 ex0, [p0, lf0, r24];		vmac.f	dm0, dm2, ex2, ex0, r12
	vmac.f	dm3, dm3, ex3, ex0, r12
	vmac.f	dm2, dm4, ex3, ex0, r12
	vmac.f	dm1, dm1, ex5, ex0, r12
	vmac.f	dm3, dm3, ex7, ex8, r12
	nop	
	vlda	 x1, [sp, #-448];		vmac.f	dm0, dm0, ex5, ex4, r12 // 64-byte Folded Reload
	vlda	 x0, [sp, #-512];		vadd.f	dm1, dm1, dm3, r13 // 64-byte Folded Reload
	vmac.f	dm2, dm2, ex7, ex0, r12
	nop	
	nop	
	vadd.f	dm0, dm0, dm2, r13
	nop	
	vconv.bf16.fp32	 x8, cml1
	vconv.bf16.fp32	 x9, cmh1
	nop	
	vmul.f	dm1, y4, y0, r14
	vconv.bf16.fp32	 x10, cml0
	vconv.bf16.fp32	 x11, cmh0
	lshl	 r1, r0, r15
	add	 r2, r1, r8;		vmul.f	dm0, y5, y0, r14
	mov	m0, r2
	vconv.bf16.fp32	 x0, cml1;		mov	p0, p6
	padda	 [p0], m0;		vconv.bf16.fp32	 x10, cmh1
	vst	 x0, [p0, #0];		jnz	 r5, #.LBB8_2
	vst	 x10, [p0, #64];		mov	m0, r1  //  Delay Slot 5
	vconv.bf16.fp32	 x4, cml0;		mov	p0, p6 //  Delay Slot 4
	padda	 [p0], m0;		vconv.bf16.fp32	 x0, cmh0 //  Delay Slot 3
	vst	 x4, [p0, #0];		vmax_lt.bf16	 x8, r16, x0, x4 //  Delay Slot 2
	vst	 x0, [p0, #64];		or	 r3, r4, r4;		vmax_lt.bf16	 x9, r16, x10, x0 //  Delay Slot 1
// %bb.5:                               //   in Loop: Header=BB8_4 Depth=2
	j	#.LBB8_3
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	vmov	x4, x8                          //  Delay Slot 1
.LBB8_6:                                // %for.cond.cleanup7
                                        //   in Loop: Header=BB8_1 Depth=1
	nopa	;		nopb	;		nops	;		lshl	 r0, r18, r15;		nopm	;		nopv	
	movs	p1, p4;		nopx	;		mov	m0, r0
	padda	 [p1], m0
	vldb	 wl2, [p1, #0]
	vlda	 x0, [sp, #-704]                // 64-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	vlda	 bmll2, [sp, #-384];		vconv.fp32.bf16	bmll0, wl2 // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-320];		vconv.fp32.bf16	bmll1, wl0;		vadd.f	dm1, dm2, dm1, r13 // 64-byte Folded Reload
	vlda	 bmhl2, [sp, #-256];		movs	dj0, r0;		movx	r0, #52;		vmov	bmll2, bmll0 // 64-byte Folded Reload
	vlda	 bmhh2, [sp, #-192];		movx	r0, #53;		vshuffle	x6, x4, x9, r0 // 64-byte Folded Reload
	vshuffle	x0, x4, x9, r0
	vmov	wl4, wh6
	vmax_lt.bf16	 x4, r16, x6, x4
	vconv.bf16.fp32	 wl8, bmll1;		vmov	wl6, wh0
	vmax_lt.bf16	 x0, r16, x0, x6;		vadd.f	dm1, dm1, dm2, r13
	vconv.fp32.bf16	cml1, x8
	vmax_lt.bf16	 x0, r16, x4, x0
	vmov	wh4, wl0
	vmov	wl4, wl0
	vshift	x4, x4, x0, r19
	vconv.bf16.fp32	 x6, cml1;		vmax_lt.bf16	 x0, r16, x0, x4
	vmax_lt.bf16	 x4, r16, x0, x2
	vlt.bf16	 r16, x6, x4
	nop	
	extend.u16	 r0, r16
	jz	 r0, #.LBB8_8
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	st	 r18, [sp, #-1088]              // 4-byte Folded Spill Delay Slot 1
// %bb.7:                               // %if.then82
                                        //   in Loop: Header=BB8_1 Depth=1
	vlda	 x0, [sp, #-576]                // 64-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	lda	 r0, [sp, #-96];		vmac.f	dm0, dm0, x4, x0, r14 // 4-byte Folded Reload
	lda	 r1, [sp, #-128]                // 4-byte Folded Reload
	jl	#_ZN2atL10rescale_rbEiiN3aie6vectorI8bfloat16Lj16EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst	 wl4, [p4, dj0]                 //  Delay Slot 3
	vconv.bf16.fp32	 wl0, bmll0             //  Delay Slot 2
	mov	r11, p1                         //  Delay Slot 1
	nopa	;		nopb	;		nopx	;		mov	p0, r11;		nops	
	lda.s16	 r3, [p0], #2;		j	#.LBB8_9
	lda.s16	 r2, [p0], #2                   //  Delay Slot 5
	lda.s16	 r1, [p0, #0]                   //  Delay Slot 4
	lda.s16	 r0, [p0, #2]                   //  Delay Slot 3
	nop	                                //  Delay Slot 2
	lda	 p3, [sp, #-116];		movx	r19, #16;		mov	p1, r11 // 4-byte Folded Reload Delay Slot 1
.LBB8_8:                                //   in Loop: Header=BB8_1 Depth=1
	vextract.16	 r3, x2, #0, vaddsign1
	vextract.16	 r2, x2, #1, vaddsign1
	vextract.16	 r1, x2, #2, vaddsign1
	vextract.16	 r0, x2, #3, vaddsign1
.LBB8_9:                                // %if.end87
                                        //   in Loop: Header=BB8_1 Depth=1
	mova	m0, #8;		nopx	
	padda	 [p1], m0;		vbcst.16	 x0, r3
	lda.s16	 r2, [p1], #2;		vbcst.16	 x2, r2
	lda.s16	 r1, [p1], #2;		vbcst.16	 x4, r1
	vbcst.16	 x6, r0
	lda.s16	 r0, [p1, #0];		vshift	x0, x0, x0, r19
	vshift	x0, x0, x2, r19
	lda.s16	 r3, [p1, #2];		vshift	x0, x0, x4, r19
	vshift	x0, x0, x6, r19
	vbcst.16	 x2, r2
	vbcst.16	 x4, r1
	vshift	x2, x0, x2, r19
	lda	 p1, [sp, #-104];		vbcst.16	 x6, r0 // 4-byte Folded Reload
	jl	#_ZN2atL6expd_nEP8bfloat16N3aie6vectorIS0_Lj64EEERKN3nk84ExpQE
	vbcst.16	 x8, r3                         //  Delay Slot 5
	vshift	x2, x2, x4, r19                 //  Delay Slot 4
	vshift	x2, x2, x6, r19                 //  Delay Slot 3
	mova	r0, #6;		vshift	x2, x2, x8, r19 //  Delay Slot 2
	movs	p0, p3;		geu	 r11, r9, r0    //  Delay Slot 1
	lda	 p3, [sp, #-116]                // 4-byte Folded Reload
	jnz	 r11, #.LBB8_12
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	vlda	 x2, [sp, #-640]                // 64-byte Folded Reload Delay Slot 1
// %bb.10:                              // %for.body.i.preheader
                                        //   in Loop: Header=BB8_1 Depth=1
	lda	 r0, [sp, #-92]                 // 4-byte Folded Reload
	nop	
	nop	
	lda	 r0, [sp, #-88]                 // 4-byte Folded Reload
	nop	
	movxm	ls, #.LBB8_11
	movxm	le, #.L_LEnd13
	nopa	;		nopb	;		nops	;		nopx	;		add.nc	lc, r0, #0;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
.LBB8_11:                               // %for.body.i
                                        //   Parent Loop BB8_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		nops	;		lshl	 r1, r0, r15;		nopm	;		nopv	
	nopa	;		nopb	;		movs	p0, p3;		nopx	;		mov	m0, r1;		nopv	
	padda	 [p0], m0;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		vst	 x2, [p0, #0];		nopxm	;		nopv	
.L_LEnd13:
	nopa	;		nopb	;		vst	 x2, [p0, #64];		add	 r0, r0, r10;		nopm	;		nopv	
.LBB8_12:                               // %for.cond.cleanup.i
                                        //   in Loop: Header=BB8_1 Depth=1
	lda	 dj0, [sp, #-100];		nopx	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	lda	 r3, [sp, #-124];		movxm	p0, ##_ZN2atL5g_erbE // 4-byte Folded Reload
	lda	 p2, [p0, dj0]
	nop	
	nop	
	nop	
	nop	
	movx	r26, #0
	lshl	 r0, r26, r15;		mov	m0, r3
	padda	 [p2], m0;		movs	p0, p3;		mov	m0, r0
	padda	 [p0], m0
	vlda.conv.fp32.bf16	 cml1, [p0, #0]
	vlda.conv.fp32.bf16	 cmh1, [p0, #64]
	nop	
	mova	r19, #16;		movx	r17, #128;		mov	r6, #576
	lda	 p4, [sp, #-112];		movxm	ls, #.LBB8_13 // 4-byte Folded Reload
	lda	 r18, [sp, #-1088];		movxm	le, #.L_LEnd12 // 4-byte Folded Reload
	mova	r2, #6;		movxm	r7, #1152
	lda	 r4, [sp, #-120];		nopb	;		nops	;		add	 r0, r26, r10;		add.nc	lc, r2, #-1;		nopv	 // 4-byte Folded Reload
.LBB8_13:                               // %for.body8.i
                                        //   Parent Loop BB8_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		nopxm	;		nopv	
	nopa	;		nopb	;		vst.flush.512.conv	 [p2, sf, r26];		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		lshl	 r1, r0, r15;		nopm	;		nopv	
	nopa	;		nopb	;		movs	p0, p3;		nopx	;		mov	m0, r1;		nopv	
	padda	 [p0], m0;		nopb	;		nopxm	;		nops	
	vlda.conv.fp32.bf16	 cmh1, [p0, #64]
	vlda.conv.fp32.bf16	 cml1, [p0, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
.L_LEnd12:
	nopa	;		nopb	;		nops	;		add	 r0, r0, r10;		nopm	;		nopv	
// %bb.14:                              //   in Loop: Header=BB8_1 Depth=1
	lda	 r1, [sp, #-128];		nopb	;		nops	;		nopxm	;		nopv	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	mova	r0, #448
	add	 r3, r3, r0
	add	r1, r1, #1
	eq	 r0, r1, r2
	jz	 r0, #.LBB8_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26] //  Delay Slot 3
	vst.flush.512.conv	 [p2, sf, r26];		add	 r4, r4, r6 //  Delay Slot 2
	st	 r1, [sp, #-128];		add	r18, r18, #16 // 4-byte Folded Spill Delay Slot 1
// %bb.15:                              // %for.cond.cleanup
	lda	 p7, [sp, #-44]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-48]                 // 4-byte Folded Reload
	lda	 r15, [sp, #-52]                // 4-byte Folded Reload
	lda	 r14, [sp, #-56]                // 4-byte Folded Reload
	lda	 lr, [sp, #-84]                 // 4-byte Folded Reload
	lda	 r13, [sp, #-60]                // 4-byte Folded Reload
	lda	 r12, [sp, #-64]                // 4-byte Folded Reload
	lda	 r11, [sp, #-68]                // 4-byte Folded Reload
	lda	 r10, [sp, #-72]                // 4-byte Folded Reload
	lda	 r9, [sp, #-76]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-80]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-1088                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end8:
	.size	_ZN2atL7chunk_kEiPK9bfp16ebs8S2_iRKN3nk84ExpQE, .Lfunc_end8-_ZN2atL7chunk_kEiPK9bfp16ebs8S2_iRKN3nk84ExpQE
                                        // -- End function
	.section	.text.attn_chunk_v,"ax",@progbits
	.globl	attn_chunk_v                    // -- Begin function attn_chunk_v
	.p2align	4
	.type	attn_chunk_v,@function
attn_chunk_v:                           // @attn_chunk_v
// %bb.0:                               // %entry
	paddxm	 [sp], #64
	st	 p7, [sp, #-40];		movxm	p7, ##_ZN2atL7g_chunkE // 4-byte Folded Spill
	lda	 r0, [p7, #0]
	nop	
	nop	
	nop	
	nop	
	mova	r1, #-48
	movxm	r2, #1072
	mac	r2, r2, r0, r1
	st	 r9, [sp, #-56];		movx	r9, #0  // 4-byte Folded Spill
	ge	 r1, r9, r2
	jnz	 r1, #.LBB9_5
	st	 r10, [sp, #-52]                // 4-byte Folded Spill Delay Slot 5
	st	 lr, [sp, #-64]                 // 4-byte Folded Spill Delay Slot 4
	st	 r8, [sp, #-60]                 // 4-byte Folded Spill Delay Slot 3
	st	 r11, [sp, #-48]                // 4-byte Folded Spill Delay Slot 2
	mova	r10, #24;		st	 p6, [sp, #-44] // 4-byte Folded Spill Delay Slot 1
// %bb.1:                               // %for.cond.preheader
	movs	p6, p0;		movxm	p0, ##_ZN2atL8g_ntilesE
	lda	 r11, [p0, #0]
	j	#.LBB9_3
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	movx	r8, #0                          //  Delay Slot 1
.LBB9_2:                                // %for.body
                                        //   in Loop: Header=BB9_3 Depth=1
	nopa	;		nopb	;		nops	;		jl	#_ZN2atL7chunk_vEiPK9bfp16ebs8;		nopv	
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	or	 r0, r8, r8;		mov	p0, p6  //  Delay Slot 1
	nopa	;		nopb	;		nops	;		add	r8, r8, #1;		nopm	;		nopv	
.LBB9_3:                                // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		ge	 r0, r8, r11;		nopm	
	jz	 r0, #.LBB9_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.4:                               // %if.end.loopexit
	lda	 r0, [p7, #0]
.LBB9_5:                                // %if.end
	lda	 lr, [sp, #-64];		nopb	;		nopxm	 // 4-byte Folded Reload
	lda	 p6, [sp, #-44]                 // 4-byte Folded Reload
	lda	 r11, [sp, #-48]                // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	lda	 r10, [sp, #-52]                // 4-byte Folded Reload
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 p7, [sp, #-40]                 // 4-byte Folded Reload
	ret	lr
	add	r0, r0, #1                      //  Delay Slot 5
	eq	 r27, r0, r10                   //  Delay Slot 4
	sel.nez	 r0, r9, r0, r27                //  Delay Slot 3
	st	 r0, [p7, #0];		paddxm	 [sp], #-64 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end9:
	.size	attn_chunk_v, .Lfunc_end9-attn_chunk_v
                                        // -- End function
	.section	.text._ZN2atL7chunk_vEiPK9bfp16ebs8,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN2atL7chunk_vEiPK9bfp16ebs8
	.type	_ZN2atL7chunk_vEiPK9bfp16ebs8,@function
_ZN2atL7chunk_vEiPK9bfp16ebs8:          // @_ZN2atL7chunk_vEiPK9bfp16ebs8
// %bb.0:                               // %entry
	mova	r3, #64;		nopb	;		nops	;		movxm	p2, ##_ZN2atL3g_oE;		nopv	
	mova	r1, #2;		nopb	;		nops	;		movxm	p1, ##_ZN2atL5g_erbE;		nopv	
	mova	r5, #780;		nopb	;		lshl	 r2, r0, r1;		mov	r4, #448
	mova	r7, #896;		movx	r6, #60;		mov	dj0, r2
	lda	 r22, [p1, dj0];		movxm	r18, #3072
	mova	r16, #128;		movxm	p3, ##_ZN2atL6g_osumE
	mova	r17, #512;		movxm	p5, ##_ZN2atL6g_onesE
	mova	r2, #384;		paddxm	 [sp], #64
	lda	 p3, [p3, #0];		st	 p6, [sp, #-64];		mul	r0, r0, r2;		mov	r19, #256 // 4-byte Folded Spill
	lda	 r23, [p5, #0];		st	 p7, [sp, #-60];		movx	r25, #0;		mov	r21, #0 // 4-byte Folded Spill
	lda	 p4, [p2, dj0];		movx	r2, #4;		mov	p2, p0
.LBB10_1:                               // %_ZN3aie41block_vector_restrict_input_buffer_streamI9bfp16ebs8Lj64EL15aie_dm_resource5EE3popEv.exit.5
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_2 Depth 2
	movs	p0, r22;		nopx	;		mov	m0, r21
	padda	 [p0], m0;		movx	r24, #0
	vldb.fill.512	 [p0, lf0, r24]
	vldb.pop.576	 ex10, [p0, lf0, r24];		mov	r26, r21
	vldb.pop.576	 ex8, [p0, lf0, r24];		movxm	ls, #.LBB10_2
	vldb.pop.576	 ex2, [p0, lf0, r24];		movxm	le, #.L_LEnd14
	vldb.pop.576	 ex0, [p0, lf0, r24];		lshl	 r20, r25, r1;		add.nc	lc, r2, #0
	movs	p5, p4;		vldb.pop.576	 ex4, [p0, lf0, r24];		or	 r27, r25, r25;		mov	m0, r20
	padda	 [p5], m0;		vldb.pop.576	 ex6, [p0, lf0, r24];		movx	r29, #0;		mov	r28, #0
.LBB10_2:                               // %for.inc57.2
                                        //   Parent Loop BB10_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nops	;		mov	p6, p5
	mov	p1, p2
	lshl	 r20, r28, r1;		mov	p0, p2
	movx	r24, #0;		mov	m0, r20
	padds	 [p6], m0;		add	 r21, r20, r19;		mov	r25, r24
	vlda	 bmll1, [p6, #0];		add	 r20, r29, r4;		mov	m0, r29
	padda	 [p1], m0;		mov	m0, r20
	vlda	 bmlh1, [p6, #64];		vldb.fill.512	 [p1, lf1, r25];		padds	 [p0], m0
	vlda	 bmhl1, [p6, #128];		vldb.fill.512	 [p0, lf0, r24];		mov	p7, p5
	vlda	 bmhh1, [p6, #192];		vldb.pop.576	 ex1, [p1, lf1, r25];		mov	m0, r21
	padda	 [p7], m0;		vldb.pop.576	 ex3, [p0, lf0, r24]
	vlda	 bmll2, [p7, #0];		vldb.pop.576	 ex5, [p1, lf1, r25]
	vlda	 bmlh2, [p7, #64];		vldb.pop.576	 ex7, [p0, lf0, r24]
	vlda	 bmhl2, [p7, #128];		vldb.pop.576	 ex9, [p1, lf1, r25]
	vlda	 bmhh2, [p7, #192];		vldb.pop.576	 ex1, [p1, lf1, r25]
	vldb.pop.576	 ex11, [p0, lf0, r24]
	vldb.pop.576	 ex5, [p1, lf1, r25]
	vldb.pop.576	 ex11, [p1, lf1, r25];		vmac.f	dm3, dm1, ex10, ex1, r5
	vldb.pop.576	 ex3, [p0, lf0, r24];		vmac.f	dm4, dm2, ex10, ex3, r5
	vldb.pop.576	 ex9, [p0, lf0, r24];		vmul.f	dm1, ex8, ex5, r5
	vmul.f	dm2, ex8, ex7, r5
	vldb.pop.576	 ex1, [p0, lf0, r24];		vmac.f	dm0, dm3, ex2, ex9, r5
	vmac.f	dm1, dm1, ex0, ex1, r5
	vmac.f	dm4, dm4, ex2, ex11, r5
	vmac.f	dm0, dm0, ex4, ex5, r5
	vmac.f	dm1, dm1, ex6, ex11, r5
	vmac.f	dm2, dm2, ex0, ex3, r5
	vmac.f	dm4, dm4, ex4, ex9, r5
	vadd.f	dm0, dm0, dm1, r6
	vmac.f	dm2, dm2, ex6, ex1, r5
	nop	
	nop	
	vadd.f	dm4, dm4, dm2, r6
	nop	
	vst	 bmll0, [p6, #0]
	vst	 bmlh0, [p6, #64]
	vst	 bmhl0, [p6, #128]
	vst	 bmhh0, [p6, #192]
	vst	 bmll4, [p7, #0]
	vst	 bmlh4, [p7, #64]
	vst	 bmhl4, [p7, #128];		add	 r28, r28, r16
.L_LEnd14:
	nopa	;		nopb	;		vst	 bmhh4, [p7, #192];		add	 r29, r29, r7;		nopm	;		nopv	
// %bb.3:                               // %for.inc106.2
                                        //   in Loop: Header=BB10_1 Depth=1
	mova	r25, #0;		nopb	;		lshl	 r20, r0, r1;		mov	p1, r23
	movs	p0, p3;		vldb.fill.512	 [p1, lf1, r25];		mov	m0, r20
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
	add	 r25, r27, r17
	eq	 r20, r25, r18
	jz	 r20, #.LBB10_1
	nop	                                //  Delay Slot 5
	vst	 bmll0, [p0, #0]                //  Delay Slot 4
	vst	 bmlh0, [p0, #64]               //  Delay Slot 3
	vst	 bmhl0, [p0, #128];		add	 r21, r26, r4 //  Delay Slot 2
	vst	 bmhh0, [p0, #192];		add	 r0, r0, r3 //  Delay Slot 1
// %bb.4:                               // %for.cond.cleanup
	lda	 p7, [sp, #-60]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-64]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end10:
	.size	_ZN2atL7chunk_vEiPK9bfp16ebs8, .Lfunc_end10-_ZN2atL7chunk_vEiPK9bfp16ebs8
                                        // -- End function
	.section	.text.attn_end,"ax",@progbits
	.globl	attn_end                        // -- Begin function attn_end
	.p2align	4
	.type	attn_end,@function
attn_end:                               // @attn_end
// %bb.0:                               // %entry
	movxm	p1, ##_ZN2atL8g_ntilesE
	movxm	p2, ##_ZN2atL6g_osumE
	movxm	r2, #3072
	movxm	r3, #-16512
	movxm	r4, #16256
	movxm	r7, #3456
	paddxm	 [sp], #192
	vbcst.16	 x4, r4
	lda	 r1, [p1, #0];		movxm	p1, ##_ZN2atL3g_oE
	vbcst.16	 x0, r3
	movxm	r3, #32499
	mova	r0, #0;		vbcst.16	 x2, r3
	mova	r5, #64;		movxm	r3, #1065353216
	mova	r6, #576;		j	#.LBB11_7
	movx	r16, #384;		vbcst.32	 x6, r3 //  Delay Slot 5
	mova	r18, #0;		movx	r17, #2;		vmov	bmll0, x6 //  Delay Slot 4
	mova	r4, #512;		st	 p6, [sp, #-64];		movx	r19, #0;		vmov	bmlh0, x6 // 4-byte Folded Spill Delay Slot 3
	lda	 p5, [p2, #0];		vst	 bmll0, [sp, #-192];		lt	 r27, r0, r1;		vmov	x7, x0 // 64-byte Folded Spill Delay Slot 2
	mova	r3, #828;		vst	 bmlh0, [sp, #-128];		sel.nez	 r1, r1, r0, r27;		vmov	x6, x0 // 64-byte Folded Spill Delay Slot 1
.LBB11_1:                               // %for.body21
                                        //   in Loop: Header=BB11_2 Depth=3
	nopa	;		nopb	;		lshl	 r24, r23, r17
	movs	p6, p4;		mov	m0, r24
	padda	 [p6], m0
	vlda	 bmll1, [p6], #64
	vlda	 bmlh1, [p6], #64
	vlda	 bmhl1, [p6, #0]
	vlda	 bmhh1, [p6, #64]
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x10, cml1
	vconv.bf16.fp32	 x11, cmh1
	nop	
	vmac.f	dm0, dm1, y5, y3, r3
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x10, cml0;		vmul.f	dm1, y5, y4, r3
	vconv.bf16.fp32	 x11, cmh0
	nop	
	vmac.f	dm0, dm1, y5, y4, r3
	nop	
	nop	
	nop	
	nop	
	nop	
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26]
	vst.flush.512.conv	 [p2, sf, r26]
	nop	
	add	 r23, r23, r5
.LBB11_2:                               // %for.cond18
                                        //   Parent Loop BB11_7 Depth=1
                                        //     Parent Loop BB11_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ne	 r24, r23, r4
	jnz	 r24, #.LBB11_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %for.cond.cleanup20
                                        //   in Loop: Header=BB11_4 Depth=2
	add	 r22, r22, r6
	add	 r21, r21, r5
	add	 r20, r20, r4
.LBB11_4:                               // %for.cond1
                                        //   Parent Loop BB11_7 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB11_2 Depth 3
	nopa	;		nopb	;		ne	 r23, r20, r2;		nopm	
	jz	 r23, #.LBB11_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.5:                               // %for.body4
                                        //   in Loop: Header=BB11_4 Depth=2
	lshl	 r23, r21, r17
	movs	p2, p5;		mov	m0, r23
	padda	 [p2], m0
	vlda	 bmll2, [p2], #64
	vlda	 bmlh2, [p2], #64
	vlda	 bmll1, [p2, #0]
	vlda	 bmlh1, [p2, #64]
	nop	
	nop	
	nop	
	nop	
	nop	
	vlda	 bmll0, [sp, #-192];		vconv.bf16.fp32	 x8, cml2 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-128];		vconv.bf16.fp32	 x10, cml1 // 64-byte Folded Reload
	vmac.f	dm2, dm2, x8, x0, r3
	vmac.f	dm3, dm1, x10, x0, r3
	nop	
	vsub.16	 x5, x2, x8
	vsub.16	 x9, x2, x10
	vmsc.f	dm2, dm1, x8, x5, r3
	vconv.bf16.fp32	 x1, cml2;		vmov	cml1, cml0;		vmsc.f	dm3, dm1, x10, x9, r3
	vconv.bf16.fp32	 x3, cml3
	vmsc.f	dm2, dm2, x1, x5, r3
	vmsc.f	dm3, dm3, x3, x9, r3
	nop	
	nop	
	nop	
	vmul.f	dm2, x5, x4, r3
	vconv.bf16.fp32	 x11, cml2;		vmul.f	dm3, x9, x4, r3
	vconv.bf16.fp32	 x5, cml3
	vmac.f	dm2, dm2, x5, x11, r3
	vmac.f	dm3, dm3, x9, x5, r3
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x5, cml2
	vconv.bf16.fp32	 x9, cml3
	vmac.f	dm4, dm2, x5, x0, r3
	vmac.f	dm4, dm3, x9, x0, r3
	nop	
	nop	
	nop	
	vmsc.f	dm0, dm1, x8, x5, r3
	vconv.bf16.fp32	 x11, cml4;		vmsc.f	dm1, dm1, x10, x9, r3
	vconv.bf16.fp32	 x8, cml4
	vmsc.f	dm0, dm0, x8, x11, r3
	vmsc.f	dm1, dm1, x10, x8, r3
	nop	
	vmsc.f	dm0, dm0, x1, x5, r3
	vmsc.f	dm1, dm1, x3, x9, r3
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x8, cml0
	vconv.bf16.fp32	 x10, cml1
	vmac.f	dm0, dm2, x5, x8, r3
	vmac.f	dm1, dm3, x9, x10, r3
	j	#.LBB11_2
	lshl	 r23, r20, r17                  //  Delay Slot 5
	movs	p4, p3;		mov	m0, r23         //  Delay Slot 4
	padda	 [p4], m0;		mov	m0, r22 //  Delay Slot 3
	vconv.bf16.fp32	 x8, cml0;		mov	p2, p0 //  Delay Slot 2
	padda	 [p2], m0;		vconv.bf16.fp32	 x9, cml1;		movx	r23, #0;		mov	r26, #0 //  Delay Slot 1
.LBB11_6:                               // %for.cond.cleanup3
                                        //   in Loop: Header=BB11_7 Depth=1
	nopa	;		nopb	;		add	r19, r19, #1
	add	 r18, r18, r7
	add	 r0, r0, r16
.LBB11_7:                               // %for.cond
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_4 Depth 2
                                        //       Child Loop BB11_2 Depth 3
	nopa	;		nopb	;		nops	;		eq	 r20, r1, r19;		nopm	;		nopv	
	jnz	 r20, #.LBB11_9
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.8:                               // %for.cond1.preheader
                                        //   in Loop: Header=BB11_7 Depth=1
	j	#.LBB11_4
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	lshl	 r20, r19, r17                  //  Delay Slot 3
	or	 r21, r0, r0;		mov	dj0, r20 //  Delay Slot 2
	lda	 p3, [p1, dj0];		movx	r20, #0;		mov	r22, r18 //  Delay Slot 1
.LBB11_9:                               // %for.cond.cleanup
	lda	 p6, [sp, #-64];		nopb	;		nopxm	;		nops	 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-192                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end11:
	.size	attn_end, .Lfunc_end11-attn_end
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
.Lfunc_end13:
	.size	_ZN2atL5rows8EPK8bfloat16, .Lfunc_end13-_ZN2atL5rows8EPK8bfloat16
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
.Lfunc_end14:
	.size	_ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_, .Lfunc_end14-_ZN3nk8L5exp32EN3aie6vectorI8bfloat16Lj32EEERKNS_4ExpCES3_
                                        // -- End function
	.section	.text._ZN2atL6expd_nEP8bfloat16N3aie6vectorIS0_Lj64EEERKN3nk84ExpQE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN2atL6expd_nEP8bfloat16N3aie6vectorIS0_Lj64EEERKN3nk84ExpQE
	.type	_ZN2atL6expd_nEP8bfloat16N3aie6vectorIS0_Lj64EEERKN3nk84ExpQE,@function
_ZN2atL6expd_nEP8bfloat16N3aie6vectorIS0_Lj64EEERKN3nk84ExpQE: // @_ZN2atL6expd_nEP8bfloat16N3aie6vectorIS0_Lj64EEERKN3nk84ExpQE
// %bb.0:                               // %entry
	nopa	;		movxm	r0, #16313
	mov	s0, #7
	mov	crsrsmode, #0
	movxm	ls, #.LBB15_1
	movxm	le, #.L_LEnd15
	mova	r4, #6;		mov	crupsmode, #0
	add.nc	lc, r4, #0
	vbcst.16	 x4, r0
	mova	r2, #0;		movxm	r0, #-17620
	mov	s1, r2
	vbcst.16	 x6, r0
	movxm	r0, #19264
	vbcst.16	 x8, r0
	movxm	r0, #-13504
	vbcst.16	 x10, r0
	movxm	r0, #-16512
	vmov	lfh0, x8
	vbcst.16	 x1, r0
	movxm	r0, #65416
	mova	r1, #2;		vbcst.16	 x3, r0
	mova	r3, #64;		movxm	r0, #15990
	mova	r0, #828;		movx	r4, #1;		vbcst.16	 x5, r0
.LBB15_1:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		lshl	 r5, r2, r4;		nopm	;		nops	
	movs	m0, r5;		mov	p2, p0
	paddb	 [p2], m0
	vldb	 x11, [p2, #0]
	nop	
	nop	
	nop	
	nop	
	vldb	 x8, [p2, #64]
	nop	
	vmul.f	dm1, x11, x4, r0
	nop	
	nop	
	vmac.f	dm3, dm1, x11, x6, r0
	nop	
	vldb	 x7, [p1, #256];		vmul.f	dm2, x8, x4, r0
	vmsc.f	dm4, dm3, x0, x4, r0
	nop	
	vmac.f	dm0, dm2, x8, x6, r0
	vmsc.f	dm4, dm4, x0, x6, r0
	vmov	x8, lfh0
	vmsc.f	dm0, dm0, x2, x4, r0
	vmac.f	dm1, dm4, x7, x8, r0
	nop	
	vmsc.f	dm0, dm0, x2, x6, r0
	vmac.f	dm2, dm1, x7, x10, r0
	nop	
	vmac.f	dm3, dm0, x7, x8, r0
	nop	
	nop	
	vmac.f	dm2, dm3, x7, x10, r0
	vconv.bf16.fp32	 x9, cml2
	nop	
	vmac.f	dm4, dm4, x9, x1, r0
	nop	
	nop	
	vlda	 bmlh1, [p1, #192];		vconv.bf16.fp32	 x11, cml2
	vlda	 bmll1, [p1, #128]
	vmac.f	dm0, dm0, x11, x1, r0
	vconv.bf16.fp32	 x8, cml4
	nop	
	vmac.f	dm2, dm1, x8, x5, r0
	vmov	x7, bmlh1
	vmov	x9, bmll1
	vlda	 bmlh4, [p1, #64];		vconv.bf16.fp32	 x11, cml0
	vlda	 bmll4, [p1, #0]
	vmac.f	dm0, dm1, x11, x5, r0
	vconv.bf16.fp32	 x7, cml2;		vshuffle	x9, x9, x7, r1
	vmax_lt.16	 x9, r16, x9, x3, vaddsign1
	vmov	x8, bmlh3;		vmac.f	dm1, dm4, x8, x7, r0
	vmov	x7, bmll3
	vshuffle	x7, x7, x8, r1
	vconv.bf16.fp32	 x9, cml0;		vmax_lt.16	 x7, r16, x7, x3, vaddsign1
	vups.2x	cmh3, x9, s0, upssign1
	vups.2x	cml3, x7, s0, upssign1;		vmac.f	dm2, dm4, x11, x9, r0
	vconv.bf16.fp32	 x8, cml1
	vsrs.2x	x11, cmh3, s1, srssign1
	vsrs.2x	x9, cml3, s1, srssign1
	nop	
	nop	
	vconv.bf16.fp32	 x7, cml2
	vadd.16	 x11, x11, x8
	vadd.16	 x9, x9, x7
	vst	 x11, [p2, #0]
.L_LEnd15:
	nopa	;		nopb	;		vst	 x9, [p2, #64];		add	 r2, r2, r3;		nopm	;		nopv	
// %bb.2:                               // %for.cond.cleanup
	nopa	;		ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end15:
	.size	_ZN2atL6expd_nEP8bfloat16N3aie6vectorIS0_Lj64EEERKN3nk84ExpQE, .Lfunc_end15-_ZN2atL6expd_nEP8bfloat16N3aie6vectorIS0_Lj64EEERKN3nk84ExpQE
                                        // -- End function
	.type	sgelu_ab,@object                // @sgelu_ab
	.section	.data.sgelu_ab,"aw",@progbits
	.globl	sgelu_ab
	.p2align	6, 0x0
sgelu_ab:
	.word	0x38f80000                      // float 1.18255615E-4
	.word	0x3a006f29                      // float 4.89937665E-4
	.word	0x39590000                      // float 2.06947327E-4
	.word	0x3a5a703c                      // float 8.33276426E-4
	.word	0x39b90000                      // float 3.52859497E-4
	.word	0x3ab4dbf6                      // float 0.00137984636
	.word	0x3a190000                      // float 5.83648681E-4
	.word	0x3b1133d7                      // float 0.00221561431
	.word	0x38f80000                      // float 1.18255615E-4
	.word	0x3a006f29                      // float 4.89937665E-4
	.word	0x39590000                      // float 2.06947327E-4
	.word	0x3a5a703c                      // float 8.33276426E-4
	.word	0x39b90000                      // float 3.52859497E-4
	.word	0x3ab4dbf6                      // float 0.00137984636
	.word	0x3a190000                      // float 5.83648681E-4
	.word	0x3b1133d7                      // float 0.00221561431
	.word	0x3a780000                      // float 9.46044921E-4
	.word	0x3b64402d                      // float 0.00348282908
	.word	0x3ac40000                      // float 0.00149536133
	.word	0x3baed268                      // float 0.00533514097
	.word	0x3b180000                      // float 0.00231933594
	.word	0x3c033e6d                      // float 0.00801048893
	.word	0x3b660000                      // float 0.00350952148
	.word	0x3c401e23                      // float 0.0117259352
	.word	0x3a780000                      // float 9.46044921E-4
	.word	0x3b64402d                      // float 0.00348282908
	.word	0x3ac40000                      // float 0.00149536133
	.word	0x3baed268                      // float 0.00533514097
	.word	0x3b180000                      // float 0.00231933594
	.word	0x3c033e6d                      // float 0.00801048893
	.word	0x3b660000                      // float 0.00350952148
	.word	0x3c401e23                      // float 0.0117259352
	.word	0x3baa0000                      // float 0.00518798828
	.word	0x3c8945c8                      // float 0.0167569071
	.word	0x3bf70000                      // float 0.0075378418
	.word	0x3cc092e1                      // float 0.0235075373
	.word	0x3c300000                      // float 0.0107421875
	.word	0x3d0459de                      // float 0.0323122665
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3d323fd7                      // float 0.0435179137
	.word	0x3baa0000                      // float 0.00518798828
	.word	0x3c8945c8                      // float 0.0167569071
	.word	0x3bf70000                      // float 0.0075378418
	.word	0x3cc092e1                      // float 0.0235075373
	.word	0x3c300000                      // float 0.0107421875
	.word	0x3d0459de                      // float 0.0323122665
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3d323fd7                      // float 0.0435179137
	.word	0x3ca90000                      // float 0.0206298828
	.word	0x3d6bb1c6                      // float 0.0575425848
	.word	0x3ce30000                      // float 0.0277099609
	.word	0x3d9842f1                      // float 0.0743464306
	.word	0x3d170000                      // float 0.0368652344
	.word	0x3dc26b28                      // float 0.0949309468
	.word	0x3d440000                      // float 0.0478515625
	.word	0x3df2348e                      // float 0.118264303
	.word	0x3ca90000                      // float 0.0206298828
	.word	0x3d6bb1c6                      // float 0.0575425848
	.word	0x3ce30000                      // float 0.0277099609
	.word	0x3d9842f1                      // float 0.0743464306
	.word	0x3d170000                      // float 0.0368652344
	.word	0x3dc26b28                      // float 0.0949309468
	.word	0x3d440000                      // float 0.0478515625
	.word	0x3df2348e                      // float 0.118264303
	.word	0x3d7c0000                      // float 0.0615234375
	.word	0x3e1517ad                      // float 0.145598128
	.word	0x3d9f0000                      // float 0.0776367188
	.word	0x3e340325                      // float 0.175793245
	.word	0x3dc60000                      // float 0.0966796875
	.word	0x3e561b66                      // float 0.209088892
	.word	0x3df20000                      // float 0.118164063
	.word	0x3e79d16e                      // float 0.243962973
	.word	0x3d7c0000                      // float 0.0615234375
	.word	0x3e1517ad                      // float 0.145598128
	.word	0x3d9f0000                      // float 0.0776367188
	.word	0x3e340325                      // float 0.175793245
	.word	0x3dc60000                      // float 0.0966796875
	.word	0x3e561b66                      // float 0.209088892
	.word	0x3df20000                      // float 0.118164063
	.word	0x3e79d16e                      // float 0.243962973
	.word	0x3e120000                      // float 0.142578125
	.word	0x3e8fa260                      // float 0.280535698
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3ea22b41                      // float 0.316736251
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3eb4461b                      // float 0.352097362
	.word	0x3e680000                      // float 0.2265625
	.word	0x3ec524ed                      // float 0.385047346
	.word	0x3e120000                      // float 0.142578125
	.word	0x3e8fa260                      // float 0.280535698
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3ea22b41                      // float 0.316736251
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3eb4461b                      // float 0.352097362
	.word	0x3e680000                      // float 0.2265625
	.word	0x3ec524ed                      // float 0.385047346
	.word	0x3e830000                      // float 0.255859375
	.word	0x3ed42e24                      // float 0.414414525
	.word	0x3e930000                      // float 0.287109375
	.word	0x3ee23223                      // float 0.441788763
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3eecb0ea                      // float 0.462287247
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3ef4d96e                      // float 0.478221357
	.word	0x3e830000                      // float 0.255859375
	.word	0x3ed42e24                      // float 0.414414525
	.word	0x3e930000                      // float 0.287109375
	.word	0x3ee23223                      // float 0.441788763
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3eecb0ea                      // float 0.462287247
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3ef4d96e                      // float 0.478221357
	.word	0x3eb90000                      // float 0.361328125
	.word	0x3efa6911                      // float 0.489082843
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3efddccf                      // float 0.495825261
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3effa358                      // float 0.499293089
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3f000013                      // float 0.500001132
	.word	0x3eb90000                      // float 0.361328125
	.word	0x3efa6911                      // float 0.489082843
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3efddccf                      // float 0.495825261
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3effa358                      // float 0.499293089
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3f000013                      // float 0.500001132
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3effffe2                      // float 0.499999106
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3f002e5b                      // float 0.500707328
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3f011193                      // float 0.504174411
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3f029378                      // float 0.510062695
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3effffe2                      // float 0.499999106
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3f002e5b                      // float 0.500707328
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3f011193                      // float 0.504174411
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3f029378                      // float 0.510062695
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3f059346                      // float 0.521778464
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3f09a78a                      // float 0.537712693
	.word	0x3e930000                      // float 0.287109375
	.word	0x3f0ee6f4                      // float 0.558211565
	.word	0x3e840000                      // float 0.2578125
	.word	0x3f1570ee                      // float 0.58375442
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3f059346                      // float 0.521778464
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3f09a78a                      // float 0.537712693
	.word	0x3e930000                      // float 0.287109375
	.word	0x3f0ee6f4                      // float 0.558211565
	.word	0x3e840000                      // float 0.2578125
	.word	0x3f1570ee                      // float 0.58375442
	.word	0x3e680000                      // float 0.2265625
	.word	0x3f1d6d88                      // float 0.614952564
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3f25dcf4                      // float 0.647902727
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3f2eea61                      // float 0.683263838
	.word	0x3e120000                      // float 0.142578125
	.word	0x3f382ed2                      // float 0.719464421
	.word	0x3e680000                      // float 0.2265625
	.word	0x3f1d6d88                      // float 0.614952564
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3f25dcf4                      // float 0.647902727
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3f2eea61                      // float 0.683263838
	.word	0x3e120000                      // float 0.142578125
	.word	0x3f382ed2                      // float 0.719464421
	.word	0x3df20000                      // float 0.118164063
	.word	0x3f418ba5                      // float 0.756037056
	.word	0x3dc60000                      // float 0.0966796875
	.word	0x3f4a7927                      // float 0.790911138
	.word	0x3d9f0000                      // float 0.0776367188
	.word	0x3f52ff36                      // float 0.824206709
	.word	0x3d7c0000                      // float 0.0615234375
	.word	0x3f5aba14                      // float 0.854401827
	.word	0x3df20000                      // float 0.118164063
	.word	0x3f418ba5                      // float 0.756037056
	.word	0x3dc60000                      // float 0.0966796875
	.word	0x3f4a7927                      // float 0.790911138
	.word	0x3d9f0000                      // float 0.0776367188
	.word	0x3f52ff36                      // float 0.824206709
	.word	0x3d7c0000                      // float 0.0615234375
	.word	0x3f5aba14                      // float 0.854401827
	.word	0x3d450000                      // float 0.0480957031
	.word	0x3f61986e                      // float 0.881232142
	.word	0x3d170000                      // float 0.0368652344
	.word	0x3f67b29b                      // float 0.905069053
	.word	0x3ce40000                      // float 0.0278320313
	.word	0x3f6ce522                      // float 0.925371289
	.word	0x3ca90000                      // float 0.0206298828
	.word	0x3f7144e4                      // float 0.942457438
	.word	0x3d450000                      // float 0.0480957031
	.word	0x3f61986e                      // float 0.881232142
	.word	0x3d170000                      // float 0.0368652344
	.word	0x3f67b29b                      // float 0.905069053
	.word	0x3ce40000                      // float 0.0278320313
	.word	0x3f6ce522                      // float 0.925371289
	.word	0x3ca90000                      // float 0.0206298828
	.word	0x3f7144e4                      // float 0.942457438
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3f74dc02                      // float 0.956482052
	.word	0x3c300000                      // float 0.0107421875
	.word	0x3f77ba62                      // float 0.967687726
	.word	0x3bf70000                      // float 0.0075378418
	.word	0x3f79fb69                      // float 0.976492464
	.word	0x3baa0000                      // float 0.00518798828
	.word	0x3f7bb5d2                      // float 0.983243107
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3f74dc02                      // float 0.956482052
	.word	0x3c300000                      // float 0.0107421875
	.word	0x3f77ba62                      // float 0.967687726
	.word	0x3bf70000                      // float 0.0075378418
	.word	0x3f79fb69                      // float 0.976492464
	.word	0x3baa0000                      // float 0.00518798828
	.word	0x3f7bb5d2                      // float 0.983243107
	.word	0x3b660000                      // float 0.00350952148
	.word	0x3f7cff87                      // float 0.988274037
	.word	0x3b180000                      // float 0.00231933594
	.word	0x3f7df306                      // float 0.991989493
	.word	0x3ac50000                      // float 0.00150299072
	.word	0x3f7ea0b3                      // float 0.994639575
	.word	0x3a780000                      // float 9.46044921E-4
	.word	0x3f7f1bc0                      // float 0.996517181
	.word	0x3b660000                      // float 0.00350952148
	.word	0x3f7cff87                      // float 0.988274037
	.word	0x3b180000                      // float 0.00231933594
	.word	0x3f7df306                      // float 0.991989493
	.word	0x3ac50000                      // float 0.00150299072
	.word	0x3f7ea0b3                      // float 0.994639575
	.word	0x3a780000                      // float 9.46044921E-4
	.word	0x3f7f1bc0                      // float 0.996517181
	.word	0x3a190000                      // float 5.83648681E-4
	.word	0x3f7f6ecc                      // float 0.997784376
	.word	0x39b90000                      // float 3.52859497E-4
	.word	0x3f7fa592                      // float 0.998620152
	.word	0x39590000                      // float 2.06947327E-4
	.word	0x3f7fc964                      // float 0.999166727
	.word	0x38f90000                      // float 1.18732452E-4
	.word	0x3f7fdfc5                      // float 0.999508202
	.word	0x3a190000                      // float 5.83648681E-4
	.word	0x3f7f6ecc                      // float 0.997784376
	.word	0x39b90000                      // float 3.52859497E-4
	.word	0x3f7fa592                      // float 0.998620152
	.word	0x39590000                      // float 2.06947327E-4
	.word	0x3f7fc964                      // float 0.999166727
	.word	0x38f90000                      // float 1.18732452E-4
	.word	0x3f7fdfc5                      // float 0.999508202
	.size	sgelu_ab, 1024

	.type	sgelu_cd,@object                // @sgelu_cd
	.section	.aie.bank3,"aw",@progbits
	.globl	sgelu_cd
	.p2align	6, 0x0
sgelu_cd:
	.word	0x38f80000                      // float 1.18255615E-4
	.word	0x3a006f29                      // float 4.89937665E-4
	.word	0x39590000                      // float 2.06947327E-4
	.word	0x3a5a703c                      // float 8.33276426E-4
	.word	0x39b90000                      // float 3.52859497E-4
	.word	0x3ab4dbf6                      // float 0.00137984636
	.word	0x3a190000                      // float 5.83648681E-4
	.word	0x3b1133d7                      // float 0.00221561431
	.word	0x38f80000                      // float 1.18255615E-4
	.word	0x3a006f29                      // float 4.89937665E-4
	.word	0x39590000                      // float 2.06947327E-4
	.word	0x3a5a703c                      // float 8.33276426E-4
	.word	0x39b90000                      // float 3.52859497E-4
	.word	0x3ab4dbf6                      // float 0.00137984636
	.word	0x3a190000                      // float 5.83648681E-4
	.word	0x3b1133d7                      // float 0.00221561431
	.word	0x3a780000                      // float 9.46044921E-4
	.word	0x3b64402d                      // float 0.00348282908
	.word	0x3ac40000                      // float 0.00149536133
	.word	0x3baed268                      // float 0.00533514097
	.word	0x3b180000                      // float 0.00231933594
	.word	0x3c033e6d                      // float 0.00801048893
	.word	0x3b660000                      // float 0.00350952148
	.word	0x3c401e23                      // float 0.0117259352
	.word	0x3a780000                      // float 9.46044921E-4
	.word	0x3b64402d                      // float 0.00348282908
	.word	0x3ac40000                      // float 0.00149536133
	.word	0x3baed268                      // float 0.00533514097
	.word	0x3b180000                      // float 0.00231933594
	.word	0x3c033e6d                      // float 0.00801048893
	.word	0x3b660000                      // float 0.00350952148
	.word	0x3c401e23                      // float 0.0117259352
	.word	0x3baa0000                      // float 0.00518798828
	.word	0x3c8945c8                      // float 0.0167569071
	.word	0x3bf70000                      // float 0.0075378418
	.word	0x3cc092e1                      // float 0.0235075373
	.word	0x3c300000                      // float 0.0107421875
	.word	0x3d0459de                      // float 0.0323122665
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3d323fd7                      // float 0.0435179137
	.word	0x3baa0000                      // float 0.00518798828
	.word	0x3c8945c8                      // float 0.0167569071
	.word	0x3bf70000                      // float 0.0075378418
	.word	0x3cc092e1                      // float 0.0235075373
	.word	0x3c300000                      // float 0.0107421875
	.word	0x3d0459de                      // float 0.0323122665
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3d323fd7                      // float 0.0435179137
	.word	0x3ca90000                      // float 0.0206298828
	.word	0x3d6bb1c6                      // float 0.0575425848
	.word	0x3ce30000                      // float 0.0277099609
	.word	0x3d9842f1                      // float 0.0743464306
	.word	0x3d170000                      // float 0.0368652344
	.word	0x3dc26b28                      // float 0.0949309468
	.word	0x3d440000                      // float 0.0478515625
	.word	0x3df2348e                      // float 0.118264303
	.word	0x3ca90000                      // float 0.0206298828
	.word	0x3d6bb1c6                      // float 0.0575425848
	.word	0x3ce30000                      // float 0.0277099609
	.word	0x3d9842f1                      // float 0.0743464306
	.word	0x3d170000                      // float 0.0368652344
	.word	0x3dc26b28                      // float 0.0949309468
	.word	0x3d440000                      // float 0.0478515625
	.word	0x3df2348e                      // float 0.118264303
	.word	0x3d7c0000                      // float 0.0615234375
	.word	0x3e1517ad                      // float 0.145598128
	.word	0x3d9f0000                      // float 0.0776367188
	.word	0x3e340325                      // float 0.175793245
	.word	0x3dc60000                      // float 0.0966796875
	.word	0x3e561b66                      // float 0.209088892
	.word	0x3df20000                      // float 0.118164063
	.word	0x3e79d16e                      // float 0.243962973
	.word	0x3d7c0000                      // float 0.0615234375
	.word	0x3e1517ad                      // float 0.145598128
	.word	0x3d9f0000                      // float 0.0776367188
	.word	0x3e340325                      // float 0.175793245
	.word	0x3dc60000                      // float 0.0966796875
	.word	0x3e561b66                      // float 0.209088892
	.word	0x3df20000                      // float 0.118164063
	.word	0x3e79d16e                      // float 0.243962973
	.word	0x3e120000                      // float 0.142578125
	.word	0x3e8fa260                      // float 0.280535698
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3ea22b41                      // float 0.316736251
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3eb4461b                      // float 0.352097362
	.word	0x3e680000                      // float 0.2265625
	.word	0x3ec524ed                      // float 0.385047346
	.word	0x3e120000                      // float 0.142578125
	.word	0x3e8fa260                      // float 0.280535698
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3ea22b41                      // float 0.316736251
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3eb4461b                      // float 0.352097362
	.word	0x3e680000                      // float 0.2265625
	.word	0x3ec524ed                      // float 0.385047346
	.word	0x3e830000                      // float 0.255859375
	.word	0x3ed42e24                      // float 0.414414525
	.word	0x3e930000                      // float 0.287109375
	.word	0x3ee23223                      // float 0.441788763
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3eecb0ea                      // float 0.462287247
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3ef4d96e                      // float 0.478221357
	.word	0x3e830000                      // float 0.255859375
	.word	0x3ed42e24                      // float 0.414414525
	.word	0x3e930000                      // float 0.287109375
	.word	0x3ee23223                      // float 0.441788763
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3eecb0ea                      // float 0.462287247
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3ef4d96e                      // float 0.478221357
	.word	0x3eb90000                      // float 0.361328125
	.word	0x3efa6911                      // float 0.489082843
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3efddccf                      // float 0.495825261
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3effa358                      // float 0.499293089
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3f000013                      // float 0.500001132
	.word	0x3eb90000                      // float 0.361328125
	.word	0x3efa6911                      // float 0.489082843
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3efddccf                      // float 0.495825261
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3effa358                      // float 0.499293089
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3f000013                      // float 0.500001132
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3effffe2                      // float 0.499999106
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3f002e5b                      // float 0.500707328
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3f011193                      // float 0.504174411
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3f029378                      // float 0.510062695
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3effffe2                      // float 0.499999106
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3f002e5b                      // float 0.500707328
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3f011193                      // float 0.504174411
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3f029378                      // float 0.510062695
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3f059346                      // float 0.521778464
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3f09a78a                      // float 0.537712693
	.word	0x3e930000                      // float 0.287109375
	.word	0x3f0ee6f4                      // float 0.558211565
	.word	0x3e840000                      // float 0.2578125
	.word	0x3f1570ee                      // float 0.58375442
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3f059346                      // float 0.521778464
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3f09a78a                      // float 0.537712693
	.word	0x3e930000                      // float 0.287109375
	.word	0x3f0ee6f4                      // float 0.558211565
	.word	0x3e840000                      // float 0.2578125
	.word	0x3f1570ee                      // float 0.58375442
	.word	0x3e680000                      // float 0.2265625
	.word	0x3f1d6d88                      // float 0.614952564
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3f25dcf4                      // float 0.647902727
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3f2eea61                      // float 0.683263838
	.word	0x3e120000                      // float 0.142578125
	.word	0x3f382ed2                      // float 0.719464421
	.word	0x3e680000                      // float 0.2265625
	.word	0x3f1d6d88                      // float 0.614952564
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3f25dcf4                      // float 0.647902727
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3f2eea61                      // float 0.683263838
	.word	0x3e120000                      // float 0.142578125
	.word	0x3f382ed2                      // float 0.719464421
	.word	0x3df20000                      // float 0.118164063
	.word	0x3f418ba5                      // float 0.756037056
	.word	0x3dc60000                      // float 0.0966796875
	.word	0x3f4a7927                      // float 0.790911138
	.word	0x3d9f0000                      // float 0.0776367188
	.word	0x3f52ff36                      // float 0.824206709
	.word	0x3d7c0000                      // float 0.0615234375
	.word	0x3f5aba14                      // float 0.854401827
	.word	0x3df20000                      // float 0.118164063
	.word	0x3f418ba5                      // float 0.756037056
	.word	0x3dc60000                      // float 0.0966796875
	.word	0x3f4a7927                      // float 0.790911138
	.word	0x3d9f0000                      // float 0.0776367188
	.word	0x3f52ff36                      // float 0.824206709
	.word	0x3d7c0000                      // float 0.0615234375
	.word	0x3f5aba14                      // float 0.854401827
	.word	0x3d450000                      // float 0.0480957031
	.word	0x3f61986e                      // float 0.881232142
	.word	0x3d170000                      // float 0.0368652344
	.word	0x3f67b29b                      // float 0.905069053
	.word	0x3ce40000                      // float 0.0278320313
	.word	0x3f6ce522                      // float 0.925371289
	.word	0x3ca90000                      // float 0.0206298828
	.word	0x3f7144e4                      // float 0.942457438
	.word	0x3d450000                      // float 0.0480957031
	.word	0x3f61986e                      // float 0.881232142
	.word	0x3d170000                      // float 0.0368652344
	.word	0x3f67b29b                      // float 0.905069053
	.word	0x3ce40000                      // float 0.0278320313
	.word	0x3f6ce522                      // float 0.925371289
	.word	0x3ca90000                      // float 0.0206298828
	.word	0x3f7144e4                      // float 0.942457438
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3f74dc02                      // float 0.956482052
	.word	0x3c300000                      // float 0.0107421875
	.word	0x3f77ba62                      // float 0.967687726
	.word	0x3bf70000                      // float 0.0075378418
	.word	0x3f79fb69                      // float 0.976492464
	.word	0x3baa0000                      // float 0.00518798828
	.word	0x3f7bb5d2                      // float 0.983243107
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3f74dc02                      // float 0.956482052
	.word	0x3c300000                      // float 0.0107421875
	.word	0x3f77ba62                      // float 0.967687726
	.word	0x3bf70000                      // float 0.0075378418
	.word	0x3f79fb69                      // float 0.976492464
	.word	0x3baa0000                      // float 0.00518798828
	.word	0x3f7bb5d2                      // float 0.983243107
	.word	0x3b660000                      // float 0.00350952148
	.word	0x3f7cff87                      // float 0.988274037
	.word	0x3b180000                      // float 0.00231933594
	.word	0x3f7df306                      // float 0.991989493
	.word	0x3ac50000                      // float 0.00150299072
	.word	0x3f7ea0b3                      // float 0.994639575
	.word	0x3a780000                      // float 9.46044921E-4
	.word	0x3f7f1bc0                      // float 0.996517181
	.word	0x3b660000                      // float 0.00350952148
	.word	0x3f7cff87                      // float 0.988274037
	.word	0x3b180000                      // float 0.00231933594
	.word	0x3f7df306                      // float 0.991989493
	.word	0x3ac50000                      // float 0.00150299072
	.word	0x3f7ea0b3                      // float 0.994639575
	.word	0x3a780000                      // float 9.46044921E-4
	.word	0x3f7f1bc0                      // float 0.996517181
	.word	0x3a190000                      // float 5.83648681E-4
	.word	0x3f7f6ecc                      // float 0.997784376
	.word	0x39b90000                      // float 3.52859497E-4
	.word	0x3f7fa592                      // float 0.998620152
	.word	0x39590000                      // float 2.06947327E-4
	.word	0x3f7fc964                      // float 0.999166727
	.word	0x38f90000                      // float 1.18732452E-4
	.word	0x3f7fdfc5                      // float 0.999508202
	.word	0x3a190000                      // float 5.83648681E-4
	.word	0x3f7f6ecc                      // float 0.997784376
	.word	0x39b90000                      // float 3.52859497E-4
	.word	0x3f7fa592                      // float 0.998620152
	.word	0x39590000                      // float 2.06947327E-4
	.word	0x3f7fc964                      // float 0.999166727
	.word	0x38f90000                      // float 1.18732452E-4
	.word	0x3f7fdfc5                      // float 0.999508202
	.size	sgelu_cd, 1024

	.type	serf_ab,@object                 // @serf_ab
	.section	.data.serf_ab,"aw",@progbits
	.globl	serf_ab
	.p2align	6, 0x0
serf_ab:
	.word	0x39350000                      // float 1.72615051E-4
	.word	0x3a3d1445                      // float 7.21279823E-4
	.word	0x39930000                      // float 2.80380249E-4
	.word	0x3a95373f                      // float 0.00113842625
	.word	0x39ea0000                      // float 4.4631958E-4
	.word	0x3ae6b366                      // float 0.00176010723
	.word	0x3a380000                      // float 7.01904296E-4
	.word	0x3b3004be                      // float 0.00268582953
	.word	0x39350000                      // float 1.72615051E-4
	.word	0x3a3d1445                      // float 7.21279823E-4
	.word	0x39930000                      // float 2.80380249E-4
	.word	0x3a95373f                      // float 0.00113842625
	.word	0x39ea0000                      // float 4.4631958E-4
	.word	0x3ae6b366                      // float 0.00176010723
	.word	0x3a380000                      // float 7.01904296E-4
	.word	0x3b3004be                      // float 0.00268582953
	.word	0x3a8f0000                      // float 0.00109100342
	.word	0x3b8496e8                      // float 0.00404630974
	.word	0x3ad90000                      // float 0.00165557861
	.word	0x3bc2f83e                      // float 0.00595000293
	.word	0x3b230000                      // float 0.00248718262
	.word	0x3c0dbb35                      // float 0.00865059067
	.word	0x3b710000                      // float 0.00367736816
	.word	0x3c4a9ddf                      // float 0.012366741
	.word	0x3a8f0000                      // float 0.00109100342
	.word	0x3b8496e8                      // float 0.00404630974
	.word	0x3ad90000                      // float 0.00165557861
	.word	0x3bc2f83e                      // float 0.00595000293
	.word	0x3b230000                      // float 0.00248718262
	.word	0x3c0dbb35                      // float 0.00865059067
	.word	0x3b710000                      // float 0.00367736816
	.word	0x3c4a9ddf                      // float 0.012366741
	.word	0x3baf0000                      // float 0.00534057617
	.word	0x3c8e26d6                      // float 0.0173525028
	.word	0x3bfb0000                      // float 0.00765991211
	.word	0x3cc4bbf6                      // float 0.024015408
	.word	0x3c310000                      // float 0.0108032227
	.word	0x3d05be17                      // float 0.0326519869
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3d32faae                      // float 0.043696098
	.word	0x3baf0000                      // float 0.00534057617
	.word	0x3c8e26d6                      // float 0.0173525028
	.word	0x3bfb0000                      // float 0.00765991211
	.word	0x3cc4bbf6                      // float 0.024015408
	.word	0x3c310000                      // float 0.0108032227
	.word	0x3d05be17                      // float 0.0326519869
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3d32faae                      // float 0.043696098
	.word	0x3ca80000                      // float 0.0205078125
	.word	0x3d6b2bdc                      // float 0.0574148744
	.word	0x3ce20000                      // float 0.0275878906
	.word	0x3d97fd42                      // float 0.0742135197
	.word	0x3d150000                      // float 0.0363769531
	.word	0x3dc07854                      // float 0.0939795076
	.word	0x3d430000                      // float 0.0476074219
	.word	0x3df15315                      // float 0.117834248
	.word	0x3ca80000                      // float 0.0205078125
	.word	0x3d6b2bdc                      // float 0.0574148744
	.word	0x3ce20000                      // float 0.0275878906
	.word	0x3d97fd42                      // float 0.0742135197
	.word	0x3d150000                      // float 0.0363769531
	.word	0x3dc07854                      // float 0.0939795076
	.word	0x3d430000                      // float 0.0476074219
	.word	0x3df15315                      // float 0.117834248
	.word	0x3d7a0000                      // float 0.0610351563
	.word	0x3e14259e                      // float 0.144674748
	.word	0x3d9e0000                      // float 0.0771484375
	.word	0x3e3312fa                      // float 0.174877077
	.word	0x3dc50000                      // float 0.0961914062
	.word	0x3e552db4                      // float 0.208182156
	.word	0x3df10000                      // float 0.117675781
	.word	0x3e78e82f                      // float 0.24307321
	.word	0x3d7a0000                      // float 0.0610351563
	.word	0x3e14259e                      // float 0.144674748
	.word	0x3d9e0000                      // float 0.0771484375
	.word	0x3e3312fa                      // float 0.174877077
	.word	0x3dc50000                      // float 0.0961914062
	.word	0x3e552db4                      // float 0.208182156
	.word	0x3df10000                      // float 0.117675781
	.word	0x3e78e82f                      // float 0.24307321
	.word	0x3e110000                      // float 0.141601563
	.word	0x3e8ed58b                      // float 0.278972954
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3ea2143a                      // float 0.316560566
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3eb42eed                      // float 0.351920515
	.word	0x3e680000                      // float 0.2265625
	.word	0x3ec50f7e                      // float 0.384883821
	.word	0x3e110000                      // float 0.141601563
	.word	0x3e8ed58b                      // float 0.278972954
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3ea2143a                      // float 0.316560566
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3eb42eed                      // float 0.351920515
	.word	0x3e680000                      // float 0.2265625
	.word	0x3ec50f7e                      // float 0.384883821
	.word	0x3e840000                      // float 0.2578125
	.word	0x3ed50bf9                      // float 0.416106969
	.word	0x3e930000                      // float 0.287109375
	.word	0x3ee223f2                      // float 0.441680491
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3eeca6f2                      // float 0.462211192
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3ef4d344                      // float 0.478174329
	.word	0x3e840000                      // float 0.2578125
	.word	0x3ed50bf9                      // float 0.416106969
	.word	0x3e930000                      // float 0.287109375
	.word	0x3ee223f2                      // float 0.441680491
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3eeca6f2                      // float 0.462211192
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3ef4d344                      // float 0.478174329
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3efad5ed                      // float 0.489913374
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3efddb89                      // float 0.495815545
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3effa309                      // float 0.499290735
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3f000011                      // float 0.500001013
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3efad5ed                      // float 0.489913374
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3efddb89                      // float 0.495815545
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3effa309                      // float 0.499290735
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3f000011                      // float 0.500001013
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3effffe7                      // float 0.499999255
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3f002e82                      // float 0.500709653
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3f011235                      // float 0.504184067
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3f029511                      // float 0.510087073
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3effffe7                      // float 0.499999255
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3f002e82                      // float 0.500709653
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3f011235                      // float 0.504184067
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3f029511                      // float 0.510087073
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3f059659                      // float 0.521825373
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3f09ac84                      // float 0.53778863
	.word	0x3e930000                      // float 0.287109375
	.word	0x3f0eee0a                      // float 0.558319688
	.word	0x3e840000                      // float 0.2578125
	.word	0x3f157a0a                      // float 0.583893418
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3f059659                      // float 0.521825373
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3f09ac84                      // float 0.53778863
	.word	0x3e930000                      // float 0.287109375
	.word	0x3f0eee0a                      // float 0.558319688
	.word	0x3e840000                      // float 0.2578125
	.word	0x3f157a0a                      // float 0.583893418
	.word	0x3e680000                      // float 0.2265625
	.word	0x3f1d783e                      // float 0.615116
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3f25e88a                      // float 0.648079515
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3f2ef5e5                      // float 0.683439552
	.word	0x3e110000                      // float 0.141601563
	.word	0x3f389537                      // float 0.721026837
	.word	0x3e680000                      // float 0.2265625
	.word	0x3f1d783e                      // float 0.615116
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3f25e88a                      // float 0.648079515
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3f2ef5e5                      // float 0.683439552
	.word	0x3e110000                      // float 0.141601563
	.word	0x3f389537                      // float 0.721026837
	.word	0x3df10000                      // float 0.117675781
	.word	0x3f41c5f3                      // float 0.756926715
	.word	0x3dc50000                      // float 0.0961914062
	.word	0x3f4ab493                      // float 0.791817843
	.word	0x3d9e0000                      // float 0.0771484375
	.word	0x3f533b40                      // float 0.825122833
	.word	0x3d7a0000                      // float 0.0610351563
	.word	0x3f5af697                      // float 0.855325162
	.word	0x3df10000                      // float 0.117675781
	.word	0x3f41c5f3                      // float 0.756926715
	.word	0x3dc50000                      // float 0.0961914062
	.word	0x3f4ab493                      // float 0.791817843
	.word	0x3d9e0000                      // float 0.0771484375
	.word	0x3f533b40                      // float 0.825122833
	.word	0x3d7a0000                      // float 0.0610351563
	.word	0x3f5af697                      // float 0.855325162
	.word	0x3d430000                      // float 0.0476074219
	.word	0x3f61d59d                      // float 0.882165729
	.word	0x3d160000                      // float 0.0366210938
	.word	0x3f67cdf5                      // float 0.905486404
	.word	0x3ce20000                      // float 0.0275878906
	.word	0x3f6d0058                      // float 0.925786495
	.word	0x3ca80000                      // float 0.0205078125
	.word	0x3f714d42                      // float 0.94258511
	.word	0x3d430000                      // float 0.0476074219
	.word	0x3f61d59d                      // float 0.882165729
	.word	0x3d160000                      // float 0.0366210938
	.word	0x3f67cdf5                      // float 0.905486404
	.word	0x3ce20000                      // float 0.0275878906
	.word	0x3f6d0058                      // float 0.925786495
	.word	0x3ca80000                      // float 0.0205078125
	.word	0x3f714d42                      // float 0.94258511
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3f74d055                      // float 0.956303894
	.word	0x3c310000                      // float 0.0108032227
	.word	0x3f77a41e                      // float 0.967347979
	.word	0x3bfb0000                      // float 0.00765991211
	.word	0x3f79da20                      // float 0.975984573
	.word	0x3baf0000                      // float 0.00534057617
	.word	0x3f7b8ec9                      // float 0.982647478
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3f74d055                      // float 0.956303894
	.word	0x3c310000                      // float 0.0108032227
	.word	0x3f77a41e                      // float 0.967347979
	.word	0x3bfb0000                      // float 0.00765991211
	.word	0x3f79da20                      // float 0.975984573
	.word	0x3baf0000                      // float 0.00534057617
	.word	0x3f7b8ec9                      // float 0.982647478
	.word	0x3b710000                      // float 0.00367736816
	.word	0x3f7cd588                      // float 0.987633228
	.word	0x3b230000                      // float 0.00248718262
	.word	0x3f7dc913                      // float 0.991349399
	.word	0x3ada0000                      // float 0.00166320801
	.word	0x3f7e7867                      // float 0.994024693
	.word	0x3a8f0000                      // float 0.00109100342
	.word	0x3f7ef6d2                      // float 0.995953679
	.word	0x3b710000                      // float 0.00367736816
	.word	0x3f7cd588                      // float 0.987633228
	.word	0x3b230000                      // float 0.00248718262
	.word	0x3f7dc913                      // float 0.991349399
	.word	0x3ada0000                      // float 0.00166320801
	.word	0x3f7e7867                      // float 0.994024693
	.word	0x3a8f0000                      // float 0.00109100342
	.word	0x3f7ef6d2                      // float 0.995953679
	.word	0x3a380000                      // float 7.01904296E-4
	.word	0x3f7f4ffb                      // float 0.997314155
	.word	0x39eb0000                      // float 4.48226929E-4
	.word	0x3f7f8c30                      // float 0.998232841
	.word	0x39930000                      // float 2.80380249E-4
	.word	0x3f7fb564                      // float 0.998861551
	.word	0x39350000                      // float 1.72615051E-4
	.word	0x3f7fd0bb                      // float 0.999278724
	.word	0x3a380000                      // float 7.01904296E-4
	.word	0x3f7f4ffb                      // float 0.997314155
	.word	0x39eb0000                      // float 4.48226929E-4
	.word	0x3f7f8c30                      // float 0.998232841
	.word	0x39930000                      // float 2.80380249E-4
	.word	0x3f7fb564                      // float 0.998861551
	.word	0x39350000                      // float 1.72615051E-4
	.word	0x3f7fd0bb                      // float 0.999278724
	.size	serf_ab, 1024

	.type	serf_cd,@object                 // @serf_cd
	.section	.aie.bank3,"aw",@progbits
	.globl	serf_cd
	.p2align	6, 0x0
serf_cd:
	.word	0x39350000                      // float 1.72615051E-4
	.word	0x3a3d1445                      // float 7.21279823E-4
	.word	0x39930000                      // float 2.80380249E-4
	.word	0x3a95373f                      // float 0.00113842625
	.word	0x39ea0000                      // float 4.4631958E-4
	.word	0x3ae6b366                      // float 0.00176010723
	.word	0x3a380000                      // float 7.01904296E-4
	.word	0x3b3004be                      // float 0.00268582953
	.word	0x39350000                      // float 1.72615051E-4
	.word	0x3a3d1445                      // float 7.21279823E-4
	.word	0x39930000                      // float 2.80380249E-4
	.word	0x3a95373f                      // float 0.00113842625
	.word	0x39ea0000                      // float 4.4631958E-4
	.word	0x3ae6b366                      // float 0.00176010723
	.word	0x3a380000                      // float 7.01904296E-4
	.word	0x3b3004be                      // float 0.00268582953
	.word	0x3a8f0000                      // float 0.00109100342
	.word	0x3b8496e8                      // float 0.00404630974
	.word	0x3ad90000                      // float 0.00165557861
	.word	0x3bc2f83e                      // float 0.00595000293
	.word	0x3b230000                      // float 0.00248718262
	.word	0x3c0dbb35                      // float 0.00865059067
	.word	0x3b710000                      // float 0.00367736816
	.word	0x3c4a9ddf                      // float 0.012366741
	.word	0x3a8f0000                      // float 0.00109100342
	.word	0x3b8496e8                      // float 0.00404630974
	.word	0x3ad90000                      // float 0.00165557861
	.word	0x3bc2f83e                      // float 0.00595000293
	.word	0x3b230000                      // float 0.00248718262
	.word	0x3c0dbb35                      // float 0.00865059067
	.word	0x3b710000                      // float 0.00367736816
	.word	0x3c4a9ddf                      // float 0.012366741
	.word	0x3baf0000                      // float 0.00534057617
	.word	0x3c8e26d6                      // float 0.0173525028
	.word	0x3bfb0000                      // float 0.00765991211
	.word	0x3cc4bbf6                      // float 0.024015408
	.word	0x3c310000                      // float 0.0108032227
	.word	0x3d05be17                      // float 0.0326519869
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3d32faae                      // float 0.043696098
	.word	0x3baf0000                      // float 0.00534057617
	.word	0x3c8e26d6                      // float 0.0173525028
	.word	0x3bfb0000                      // float 0.00765991211
	.word	0x3cc4bbf6                      // float 0.024015408
	.word	0x3c310000                      // float 0.0108032227
	.word	0x3d05be17                      // float 0.0326519869
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3d32faae                      // float 0.043696098
	.word	0x3ca80000                      // float 0.0205078125
	.word	0x3d6b2bdc                      // float 0.0574148744
	.word	0x3ce20000                      // float 0.0275878906
	.word	0x3d97fd42                      // float 0.0742135197
	.word	0x3d150000                      // float 0.0363769531
	.word	0x3dc07854                      // float 0.0939795076
	.word	0x3d430000                      // float 0.0476074219
	.word	0x3df15315                      // float 0.117834248
	.word	0x3ca80000                      // float 0.0205078125
	.word	0x3d6b2bdc                      // float 0.0574148744
	.word	0x3ce20000                      // float 0.0275878906
	.word	0x3d97fd42                      // float 0.0742135197
	.word	0x3d150000                      // float 0.0363769531
	.word	0x3dc07854                      // float 0.0939795076
	.word	0x3d430000                      // float 0.0476074219
	.word	0x3df15315                      // float 0.117834248
	.word	0x3d7a0000                      // float 0.0610351563
	.word	0x3e14259e                      // float 0.144674748
	.word	0x3d9e0000                      // float 0.0771484375
	.word	0x3e3312fa                      // float 0.174877077
	.word	0x3dc50000                      // float 0.0961914062
	.word	0x3e552db4                      // float 0.208182156
	.word	0x3df10000                      // float 0.117675781
	.word	0x3e78e82f                      // float 0.24307321
	.word	0x3d7a0000                      // float 0.0610351563
	.word	0x3e14259e                      // float 0.144674748
	.word	0x3d9e0000                      // float 0.0771484375
	.word	0x3e3312fa                      // float 0.174877077
	.word	0x3dc50000                      // float 0.0961914062
	.word	0x3e552db4                      // float 0.208182156
	.word	0x3df10000                      // float 0.117675781
	.word	0x3e78e82f                      // float 0.24307321
	.word	0x3e110000                      // float 0.141601563
	.word	0x3e8ed58b                      // float 0.278972954
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3ea2143a                      // float 0.316560566
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3eb42eed                      // float 0.351920515
	.word	0x3e680000                      // float 0.2265625
	.word	0x3ec50f7e                      // float 0.384883821
	.word	0x3e110000                      // float 0.141601563
	.word	0x3e8ed58b                      // float 0.278972954
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3ea2143a                      // float 0.316560566
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3eb42eed                      // float 0.351920515
	.word	0x3e680000                      // float 0.2265625
	.word	0x3ec50f7e                      // float 0.384883821
	.word	0x3e840000                      // float 0.2578125
	.word	0x3ed50bf9                      // float 0.416106969
	.word	0x3e930000                      // float 0.287109375
	.word	0x3ee223f2                      // float 0.441680491
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3eeca6f2                      // float 0.462211192
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3ef4d344                      // float 0.478174329
	.word	0x3e840000                      // float 0.2578125
	.word	0x3ed50bf9                      // float 0.416106969
	.word	0x3e930000                      // float 0.287109375
	.word	0x3ee223f2                      // float 0.441680491
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3eeca6f2                      // float 0.462211192
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3ef4d344                      // float 0.478174329
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3efad5ed                      // float 0.489913374
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3efddb89                      // float 0.495815545
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3effa309                      // float 0.499290735
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3f000011                      // float 0.500001013
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3efad5ed                      // float 0.489913374
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3efddb89                      // float 0.495815545
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3effa309                      // float 0.499290735
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3f000011                      // float 0.500001013
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3effffe7                      // float 0.499999255
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3f002e82                      // float 0.500709653
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3f011235                      // float 0.504184067
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3f029511                      // float 0.510087073
	.word	0x3ecc0000                      // float 0.3984375
	.word	0x3effffe7                      // float 0.499999255
	.word	0x3ec90000                      // float 0.392578125
	.word	0x3f002e82                      // float 0.500709653
	.word	0x3ec20000                      // float 0.37890625
	.word	0x3f011235                      // float 0.504184067
	.word	0x3eba0000                      // float 0.36328125
	.word	0x3f029511                      // float 0.510087073
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3f059659                      // float 0.521825373
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3f09ac84                      // float 0.53778863
	.word	0x3e930000                      // float 0.287109375
	.word	0x3f0eee0a                      // float 0.558319688
	.word	0x3e840000                      // float 0.2578125
	.word	0x3f157a0a                      // float 0.583893418
	.word	0x3eae0000                      // float 0.33984375
	.word	0x3f059659                      // float 0.521825373
	.word	0x3ea10000                      // float 0.314453125
	.word	0x3f09ac84                      // float 0.53778863
	.word	0x3e930000                      // float 0.287109375
	.word	0x3f0eee0a                      // float 0.558319688
	.word	0x3e840000                      // float 0.2578125
	.word	0x3f157a0a                      // float 0.583893418
	.word	0x3e680000                      // float 0.2265625
	.word	0x3f1d783e                      // float 0.615116
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3f25e88a                      // float 0.648079515
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3f2ef5e5                      // float 0.683439552
	.word	0x3e110000                      // float 0.141601563
	.word	0x3f389537                      // float 0.721026837
	.word	0x3e680000                      // float 0.2265625
	.word	0x3f1d783e                      // float 0.615116
	.word	0x3e4a0000                      // float 0.197265625
	.word	0x3f25e88a                      // float 0.648079515
	.word	0x3e2d0000                      // float 0.168945313
	.word	0x3f2ef5e5                      // float 0.683439552
	.word	0x3e110000                      // float 0.141601563
	.word	0x3f389537                      // float 0.721026837
	.word	0x3df10000                      // float 0.117675781
	.word	0x3f41c5f3                      // float 0.756926715
	.word	0x3dc50000                      // float 0.0961914062
	.word	0x3f4ab493                      // float 0.791817843
	.word	0x3d9e0000                      // float 0.0771484375
	.word	0x3f533b40                      // float 0.825122833
	.word	0x3d7a0000                      // float 0.0610351563
	.word	0x3f5af697                      // float 0.855325162
	.word	0x3df10000                      // float 0.117675781
	.word	0x3f41c5f3                      // float 0.756926715
	.word	0x3dc50000                      // float 0.0961914062
	.word	0x3f4ab493                      // float 0.791817843
	.word	0x3d9e0000                      // float 0.0771484375
	.word	0x3f533b40                      // float 0.825122833
	.word	0x3d7a0000                      // float 0.0610351563
	.word	0x3f5af697                      // float 0.855325162
	.word	0x3d430000                      // float 0.0476074219
	.word	0x3f61d59d                      // float 0.882165729
	.word	0x3d160000                      // float 0.0366210938
	.word	0x3f67cdf5                      // float 0.905486404
	.word	0x3ce20000                      // float 0.0275878906
	.word	0x3f6d0058                      // float 0.925786495
	.word	0x3ca80000                      // float 0.0205078125
	.word	0x3f714d42                      // float 0.94258511
	.word	0x3d430000                      // float 0.0476074219
	.word	0x3f61d59d                      // float 0.882165729
	.word	0x3d160000                      // float 0.0366210938
	.word	0x3f67cdf5                      // float 0.905486404
	.word	0x3ce20000                      // float 0.0275878906
	.word	0x3f6d0058                      // float 0.925786495
	.word	0x3ca80000                      // float 0.0205078125
	.word	0x3f714d42                      // float 0.94258511
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3f74d055                      // float 0.956303894
	.word	0x3c310000                      // float 0.0108032227
	.word	0x3f77a41e                      // float 0.967347979
	.word	0x3bfb0000                      // float 0.00765991211
	.word	0x3f79da20                      // float 0.975984573
	.word	0x3baf0000                      // float 0.00534057617
	.word	0x3f7b8ec9                      // float 0.982647478
	.word	0x3c760000                      // float 0.0150146484
	.word	0x3f74d055                      // float 0.956303894
	.word	0x3c310000                      // float 0.0108032227
	.word	0x3f77a41e                      // float 0.967347979
	.word	0x3bfb0000                      // float 0.00765991211
	.word	0x3f79da20                      // float 0.975984573
	.word	0x3baf0000                      // float 0.00534057617
	.word	0x3f7b8ec9                      // float 0.982647478
	.word	0x3b710000                      // float 0.00367736816
	.word	0x3f7cd588                      // float 0.987633228
	.word	0x3b230000                      // float 0.00248718262
	.word	0x3f7dc913                      // float 0.991349399
	.word	0x3ada0000                      // float 0.00166320801
	.word	0x3f7e7867                      // float 0.994024693
	.word	0x3a8f0000                      // float 0.00109100342
	.word	0x3f7ef6d2                      // float 0.995953679
	.word	0x3b710000                      // float 0.00367736816
	.word	0x3f7cd588                      // float 0.987633228
	.word	0x3b230000                      // float 0.00248718262
	.word	0x3f7dc913                      // float 0.991349399
	.word	0x3ada0000                      // float 0.00166320801
	.word	0x3f7e7867                      // float 0.994024693
	.word	0x3a8f0000                      // float 0.00109100342
	.word	0x3f7ef6d2                      // float 0.995953679
	.word	0x3a380000                      // float 7.01904296E-4
	.word	0x3f7f4ffb                      // float 0.997314155
	.word	0x39eb0000                      // float 4.48226929E-4
	.word	0x3f7f8c30                      // float 0.998232841
	.word	0x39930000                      // float 2.80380249E-4
	.word	0x3f7fb564                      // float 0.998861551
	.word	0x39350000                      // float 1.72615051E-4
	.word	0x3f7fd0bb                      // float 0.999278724
	.word	0x3a380000                      // float 7.01904296E-4
	.word	0x3f7f4ffb                      // float 0.997314155
	.word	0x39eb0000                      // float 4.48226929E-4
	.word	0x3f7f8c30                      // float 0.998232841
	.word	0x39930000                      // float 2.80380249E-4
	.word	0x3f7fb564                      // float 0.998861551
	.word	0x39350000                      // float 1.72615051E-4
	.word	0x3f7fd0bb                      // float 0.999278724
	.size	serf_cd, 1024

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
	.addrsig_sym sgelu_ab
	.addrsig_sym sgelu_cd
	.addrsig_sym serf_ab
	.addrsig_sym serf_cd
	.addrsig_sym _ZL6acc_hi
