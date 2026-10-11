	.file	"ew8.cc"
	.section	.text.ew_norm,"ax",@progbits
	.globl	ew_norm                         // -- Begin function ew_norm
	.p2align	4
	.type	ew_norm,@function
ew_norm:                                // @ew_norm
// %bb.0:                               // %entry
	nopa	;		nopb	;		paddxm	 [sp], #128
	st	 p7, [sp, #-24]                 // 4-byte Folded Spill
	st	 r13, [sp, #-36];		movxm	r1, #3840 // 4-byte Folded Spill
	st	 r14, [sp, #-32];		movxm	r2, #2304 // 4-byte Folded Spill
	st	 r9, [sp, #-52];		movxm	r9, #15359 // 4-byte Folded Spill
	st	 r10, [sp, #-48];		movxm	r10, #1033216 // 4-byte Folded Spill
	st	 r11, [sp, #-44];		movxm	r11, #15355 // 4-byte Folded Spill
	st	 r12, [sp, #-40];		jl	#_ZL7in_needPKji // 4-byte Folded Spill
	st	 p6, [sp, #-28];		movxm	r12, #1033220 // 4-byte Folded Spill Delay Slot 5
	mova	r3, #1;		st	 r8, [sp, #-56];		movxm	p6, ##_ZL7g_arena // 4-byte Folded Spill Delay Slot 4
	st	 lr, [sp, #-60];		lt	 r27, r3, r0;		mov	r8, r0 // 4-byte Folded Spill Delay Slot 3
	st	 p1, [sp, #-64];		sel.nez	 r0, r1, r2, r27;		mov	crrnd, #12 // 4-byte Folded Spill Delay Slot 2
	mova	r13, #4;		movs	p7, p0;		mov	r14, r27 //  Delay Slot 1
	nopa	;		nopb	;		nops	;		movxm	p0, ##_ZL6g_in_b;		nopv	
	lda	 r0, [p0, #0];		nopx	
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	lt	 r27, r11, r0
	sel.nez	 r1, r12, r13, r27
	add	 r1, r1, r0
	mov	dj0, r1
	lda.s16	 r3, [p7, dj0];		add	 r2, r0, r10
	lt	 r27, r9, r0
	sel.nez	 r0, r2, r0, r27
	mov	dj0, r0
	lda	 r9, [p7, dj0]
	jl	#_ZL7in_copyPKjPvii
	movxm	m0, #12288                      //  Delay Slot 5
	vbcst.16	 x0, r3                         //  Delay Slot 4
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 3
	padda	 [p1], m0;		vst	 x0, [sp, #-128];		movxm	r1, #1536 // 64-byte Folded Spill Delay Slot 2
	movx	r0, #768;		mov	p0, p7  //  Delay Slot 1
	nopa	;		nopb	;		jz	 r14, #.LBB0_2;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.1:                               // %if.then
	jl	#_ZL7in_copyPKjPvii
	nop	                                //  Delay Slot 5
	movxm	r0, #2304                       //  Delay Slot 4
	movxm	m0, #13824                      //  Delay Slot 3
	padda	 [p6], m0;		movxm	r1, #1536 //  Delay Slot 2
	mova	r10, #3;		movs	p0, p7;		mov	p1, p6 //  Delay Slot 1
	nopa	;		nopb	;		nops	;		jl	#_ZL7in_donePji;		nopv	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movxm	r0, #3840                       //  Delay Slot 2
	mov	p0, p7                          //  Delay Slot 1
	lda	 p1, [sp, #-64];		nopxm	        // 4-byte Folded Reload
	vlda	 x0, [sp, #-128]                // 64-byte Folded Reload
	lda	 p7, [sp, #-24]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-28]                 // 4-byte Folded Reload
	lda	 r14, [sp, #-32]                // 4-byte Folded Reload
	lda	 r13, [sp, #-36]                // 4-byte Folded Reload
	lda	 r12, [sp, #-40]                // 4-byte Folded Reload
	lda	 r11, [sp, #-44]                // 4-byte Folded Reload
	lda	 r10, [sp, #-48];		movs	p0, p7 // 4-byte Folded Reload
	lda	 r9, [sp, #-52]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-56]                 // 4-byte Folded Reload
	lda	 lr, [sp, #-60]                 // 4-byte Folded Reload
	j	#_ZL9norm_rowsILb1EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE
	nop	                                //  Delay Slot 5
	eq	 r1, r8, r10                    //  Delay Slot 4
	mov	r0, r9                          //  Delay Slot 3
	paddxm	 [sp], #-128                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB0_2:                                // %if.else.critedge
	nopa	;		nopb	;		nops	;		jl	#_ZL7in_donePji;		nopv	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movxm	r0, #2304                       //  Delay Slot 2
	mov	p0, p7                          //  Delay Slot 1
	lda	 p1, [sp, #-64];		nopb	;		nopxm	 // 4-byte Folded Reload
	vlda	 x0, [sp, #-128]                // 64-byte Folded Reload
	lda	 p7, [sp, #-24]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-28]                 // 4-byte Folded Reload
	lda	 r14, [sp, #-32]                // 4-byte Folded Reload
	lda	 r13, [sp, #-36]                // 4-byte Folded Reload
	lda	 r12, [sp, #-40]                // 4-byte Folded Reload
	lda	 r11, [sp, #-44]                // 4-byte Folded Reload
	lda	 r10, [sp, #-48];		movs	p0, p7 // 4-byte Folded Reload
	lda	 r9, [sp, #-52]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-56]                 // 4-byte Folded Reload
	lda	 lr, [sp, #-60]                 // 4-byte Folded Reload
	j	#_ZL9norm_rowsILb0EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	mov	r0, r9                          //  Delay Slot 3
	paddxm	 [sp], #-128                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end0:
	.size	ew_norm, .Lfunc_end0-ew_norm
                                        // -- End function
	.section	.text._ZL7in_needPKji,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL7in_needPKji
	.type	_ZL7in_needPKji,@function
_ZL7in_needPKji:                        // @_ZL7in_needPKji
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		paddxm	 [sp], #64;		nopv	
	nopa	;		st	 p6, [sp, #-44];		movxm	p6, ##_ZL10g_in_avail // 4-byte Folded Spill
	lda	 r8, [p6, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	st	 r8, [sp, #-60]                 // 4-byte Folded Spill
	ge	 r1, r8, r0
	jnz	 r1, #.LBB1_2
	nop	                                //  Delay Slot 5
	st	 lr, [sp, #-64]                 // 4-byte Folded Spill Delay Slot 4
	st	 r9, [sp, #-56]                 // 4-byte Folded Spill Delay Slot 3
	st	 r10, [sp, #-52]                // 4-byte Folded Spill Delay Slot 2
	st	 r11, [sp, #-48]                // 4-byte Folded Spill Delay Slot 1
// %bb.1:                               // %while.end
	mova	r9, #768;		movxm	r4, #-1431655765
	add	 r1, r8, r9
	lt	 r27, r1, r0
	jl	#__muldi3
	mova	r1, #-768;		sel.nez	 r0, r0, r1, r27 //  Delay Slot 5
	add	 r0, r0, r1                     //  Delay Slot 4
	ne	 r10, r0, r8                    //  Delay Slot 3
	mova	r3, #0;		add	 r1, r8, r10    //  Delay Slot 2
	mova	r11, #-1;		sub	 r2, r0, r1;		mov	r5, #0 //  Delay Slot 1
	mova	r0, #-9;		nopb	;		nopxm	;		nops	
	lshl	 r0, r1, r0
	add	 r0, r0, r10
	mac	r8, r8, r0, r9
	nop	
	add	 r1, r8, r9
	st	 r1, [p6, #0]
	nop	
	nop	
	xor	 r0, r0, r11
	acq	#49, r0
.LBB1_2:                                // %if.end
	lda	 lr, [sp, #-64];		nopb	;		nopx	 // 4-byte Folded Reload
	nop	
	lda	 p6, [sp, #-44]                 // 4-byte Folded Reload
	lda	 r11, [sp, #-48]                // 4-byte Folded Reload
	lda	 r10, [sp, #-52]                // 4-byte Folded Reload
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end1:
	.size	_ZL7in_needPKji, .Lfunc_end1-_ZL7in_needPKji
                                        // -- End function
	.section	.text._ZL7in_copyPKjPvii,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL7in_copyPKjPvii
	.type	_ZL7in_copyPKjPvii,@function
_ZL7in_copyPKjPvii:                     // @_ZL7in_copyPKjPvii
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	p2, ##_ZL6g_in_b;		nops	
	lda	 r4, [p2, #0]
	nop	
	j	#.LBB2_2
	movxm	r2, #15359                      //  Delay Slot 5
	mova	r5, #-6;		movxm	r3, #1033216    //  Delay Slot 4
	mova	r6, #5;		lshl	 r5, r1, r5     //  Delay Slot 3
	lshl	 r5, r5, r6;		mov	r1, #64 //  Delay Slot 2
	mova	r4, #0;		add	 r0, r0, r4;		mov	r6, #1 //  Delay Slot 1
.LBB2_1:                                // %for.body
                                        //   in Loop: Header=BB2_2 Depth=1
	nopa	;		nopb	;		add	 r7, r0, r3
	lt	 r27, r2, r0
	sel.nez	 r7, r7, r0, r27
	mov	dj0, r7
	vldb	 x0, [p0, dj0]
	nop	
	nop	
	nop	
	nop	
	lshl	 r7, r4, r6
	add	 r0, r0, r1;		mov	dj0, r7
	vst	 x0, [p1, dj0];		add	r4, r4, #32
.LBB2_2:                                // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		ne	 r7, r5, r4;		nopm	;		nopv	
	jnz	 r7, #.LBB2_1
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %for.cond.cleanup
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end2:
	.size	_ZL7in_copyPKjPvii, .Lfunc_end2-_ZL7in_copyPKjPvii
                                        // -- End function
	.section	.text._ZL7in_donePji,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL7in_donePji
	.type	_ZL7in_donePji,@function
_ZL7in_donePji:                         // @_ZL7in_donePji
// %bb.0:                               // %entry
	nopa	;		movxm	p0, ##_ZL6g_in_b
	lda	 r1, [p0, #0];		movxm	p1, ##_ZL10g_in_avail
	lda	 r2, [p1, #0]
	paddxm	 [sp], #64
	st	 p6, [sp, #-52];		movxm	p6, ##_ZL13g_in_consumed // 4-byte Folded Spill
	lda	 r4, [p6, #0]
	movxm	r3, #15359
	movxm	r5, #-15360
	add	 r1, r1, r0
	sub	 r2, r2, r0
	add	 r5, r1, r5
	lt	 r27, r3, r1
	st	 r8, [sp, #-60];		add	 r8, r4, r0 // 4-byte Folded Spill
	mova	r1, #767;		sel.nez	 r3, r5, r1, r27
	lt	 r0, r1, r8
	jz	 r0, #.LBB3_2
	nop	                                //  Delay Slot 5
	st	 lr, [sp, #-64]                 // 4-byte Folded Spill Delay Slot 4
	st	 r9, [sp, #-56]                 // 4-byte Folded Spill Delay Slot 3
	st	 r2, [p1, #0]                   //  Delay Slot 2
	st	 r3, [p0, #0]                   //  Delay Slot 1
// %bb.1:                               // %while.end
	movxm	r0, #1535
	movxm	r2, #-1535
	jl	#__muldi3
	movxm	r4, #-1431655765                //  Delay Slot 5
	add	 r2, r8, r2                     //  Delay Slot 4
	mova	r3, #0;		ltu	 r27, r8, r0    //  Delay Slot 3
	sel.nez	 r0, r3, r2, r27                //  Delay Slot 2
	mova	r9, #-768;		add	 r2, r0, r1;		mov	r5, #0 //  Delay Slot 1
	mova	r0, #-9;		nopb	;		nopxm	
	lshl	 r0, r1, r0
	mac	r8, r8, r0, r9
	nop	
	add	 r1, r8, r9
	j	#.LBB3_3
	st	 r1, [p6, #0]                   //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	add	r0, r0, #1                      //  Delay Slot 2
	rel	#48, r0                         //  Delay Slot 1
.LBB3_2:                                // %while.end.thread
	nopa	;		nopb	;		st	 r8, [p6, #0];		nopxm	;		nopv	
.LBB3_3:                                // %if.end6
	lda	 lr, [sp, #-64];		nopb	;		nopxm	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	lda	 p6, [sp, #-52]                 // 4-byte Folded Reload
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end3:
	.size	_ZL7in_donePji, .Lfunc_end3-_ZL7in_donePji
                                        // -- End function
	.section	.text._ZL9norm_rowsILb1EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL9norm_rowsILb1EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE
	.type	_ZL9norm_rowsILb1EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE,@function
_ZL9norm_rowsILb1EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE: // @_ZL9norm_rowsILb1EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	r3, #16256;		nopv	
	nopa	;		paddxm	 [sp], #2944
	st	 r8, [sp, #-128];		movxm	r6, #-16512 // 4-byte Folded Spill
	st	 r13, [sp, #-108];		movxm	m0, #-1664 // 4-byte Folded Spill
	st	 p6, [sp, #-96];		movxm	r13, #15359 // 4-byte Folded Spill
	movs	p6, p0;		mov	p0, sp
	padda	 [p0], m0;		st	 r14, [sp, #-104];		movxm	r14, #1033216 // 4-byte Folded Spill
	st	 p0, [sp, #-160];		movxm	m0, #-1600 // 4-byte Folded Spill
	st	 r11, [sp, #-116];		mov	p0, sp // 4-byte Folded Spill
	padda	 [p0], m0;		st	 r15, [sp, #-100];		movxm	r15, #9215 // 4-byte Folded Spill
	st	 p0, [sp, #-156];		movxm	m0, #-1536 // 4-byte Folded Spill
	st	 r12, [sp, #-112];		mov	p0, sp // 4-byte Folded Spill
	padda	 [p0], m0;		st	 p7, [sp, #-92];		vbcst.16	 x2, r3 // 4-byte Folded Spill
	st	 p0, [sp, #-152];		movxm	m0, #-1472 // 4-byte Folded Spill
	st	 r10, [sp, #-120];		mov	p0, sp // 4-byte Folded Spill
	padda	 [p0], m0;		vst	 x2, [sp, #-1408];		movx	r2, #10;		vbcst.16	 x2, r6 // 64-byte Folded Spill
	mova	r10, #1;		st	 p0, [sp, #-148];		movxm	m0, #-2944 // 4-byte Folded Spill
	mova	r7, #0;		st	 lr, [sp, #-132];		lt	 r27, r10, r0;		mov	p0, sp // 4-byte Folded Spill
	padda	 [p0], m0;		vst	 x2, [sp, #-1088];		lshl	 r4, r27, r2;		vbcst.32	 x2, r7 // 64-byte Folded Spill
	st	 p0, [sp, #-176];		movxm	m0, #-2816 // 4-byte Folded Spill
	st	 r4, [sp, #-188];		sel.nez	 r4, r3, r7, r27;		mov	p0, sp // 4-byte Folded Spill
	padda	 [p0], m0;		st	 r9, [sp, #-124];		vbcst.16	 x4, r4 // 4-byte Folded Spill
	mova	r5, #2;		st	 p0, [sp, #-144];		movxm	m0, #-2688 // 4-byte Folded Spill
	st	 r1, [sp, #-180];		lt	 r27, r5, r0;		mov	p0, sp // 4-byte Folded Spill
	padda	 [p0], m0;		vst	 x0, [sp, #-896];		lshl	 r0, r0, r2;		vmov	bmll0, x2 // 64-byte Folded Spill
	mova	r5, #11;		st	 p0, [sp, #-172];		movxm	m0, #-2560 // 4-byte Folded Spill
	st	 r0, [sp, #-192];		lshl	 r5, r27, r5;		mov	p0, sp // 4-byte Folded Spill
	padda	 [p0], m0;		st	 r5, [sp, #-184];		vmov	bmlh0, x2 // 4-byte Folded Spill
	st	 p0, [sp, #-140];		movxm	m0, #-2432 // 4-byte Folded Spill
	vst	 x4, [sp, #-1024];		sel.nez	 r3, r3, r7, r27;		mov	p0, sp // 64-byte Folded Spill
	padda	 [p0], m0;		vst	 bmll0, [sp, #-832];		vbcst.16	 x4, r3 // 64-byte Folded Spill
	st	 p0, [sp, #-168];		movxm	m0, #-2304 // 4-byte Folded Spill
	vst	 bmlh0, [sp, #-768];		mov	p0, sp // 64-byte Folded Spill
	padda	 [p0], m0;		vst	 x4, [sp, #-960];		movx	r8, #828;		mov	p7, p1 // 64-byte Folded Spill
	mova	r0, #18;		st	 p0, [sp, #-136];		movx	r12, #16;		mov	r11, #64 // 4-byte Folded Spill
.LBB4_1:                                // %for.cond10.preheader
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB4_2 Depth 2
                                        //       Child Loop BB4_3 Depth 3
                                        //     Child Loop BB4_6 Depth 2
                                        //       Child Loop BB4_7 Depth 3
                                        //       Child Loop BB4_9 Depth 3
                                        //       Child Loop BB4_12 Depth 3
	vlda	 bmll0, [sp, #-832];		nopx	 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-768]             // 64-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	st	 r0, [sp, #-164]                // 4-byte Folded Spill
	vst	 bmll0, [sp, #-1920];		vmov	cml1, cml0 // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-1856]            // 64-byte Folded Spill
	vst	 bmll1, [sp, #-704]             // 64-byte Folded Spill
	vst	 bmlh1, [sp, #-640]             // 64-byte Folded Spill
	vst	 bmll1, [sp, #-448]             // 64-byte Folded Spill
	vst	 bmlh1, [sp, #-384]             // 64-byte Folded Spill
	lda	 r9, [sp, #-192];		vst	 bmll1, [sp, #-2176] // 4-byte Folded Reload64-byte Folded Spill
	mova	r1, #0;		vst	 bmlh1, [sp, #-2112] // 64-byte Folded Spill
.LBB4_2:                                // %for.body13
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_3 Depth 3
	nopa	;		nopb	;		jl	#_ZL7in_needPKji;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	movs	p0, p6;		or	 r0, r9, r9;		mov	r9, r1 //  Delay Slot 1
	vlda	 x1, [sp, #-1408];		nopb	;		nops	;		nopxm	;		nopv	 // 64-byte Folded Reload
	vlda	 bmll1, [sp, #-1920];		nopx	 // 64-byte Folded Reload
	lda	 r1, [sp, #-184];		movxm	p0, ##_ZL6g_in_b // 4-byte Folded Reload
	lda	 r0, [p0, #0]
	lda	 r2, [sp, #-188]                // 4-byte Folded Reload
	vlda	 bmlh1, [sp, #-1856]            // 64-byte Folded Reload
	vlda	 bmhl1, [sp, #-1792]            // 64-byte Folded Reload
	vlda	 bmhh1, [sp, #-1728]            // 64-byte Folded Reload
	vlda	 bmlh3, [sp, #-2112]            // 64-byte Folded Reload
	vlda	 bmll0, [sp, #-704];		lshl	 r2, r9, r10 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-640];		add	 r1, r1, r0;		mov	m0, r2 // 64-byte Folded Reload
	mova	r4, #0;		add	 r2, r2, r0
	vlda	 bmhl0, [sp, #-576];		add	 r5, r0, r4 // 64-byte Folded Reload
	vlda	 bmhh0, [sp, #-512];		movxm	p0, ##_ZL7g_arena // 64-byte Folded Reload
	vlda	 bmhl3, [sp, #-2048];		add	 r6, r5, r14 // 64-byte Folded Reload
	vlda	 bmll3, [sp, #-2176];		lt	 r27, r13, r5;		mov	p1, p6 // 64-byte Folded Reload
	padda	 [p0], m0;		sel.nez	 r5, r6, r5, r27
	vlda	 bmhh3, [sp, #-1984];		add	 r6, r2, r4;		mov	m0, r5;		vmov.d	dm2, dm0 // 64-byte Folded Reload
	padda	 [p1], m0
	vlda	 x11, [p1, #0];		add	 r7, r6, r14
	vldb	 x0, [p1, #64];		lt	 r27, r13, r6
	sel.nez	 r5, r7, r6, r27;		mov	p1, p6
	mova	r16, #128;		add	 r6, r1, r4;		mov	m0, r5;		vmov.d	dm3, dm1
	padda	 [p1], m0;		vst	 bmll3, [sp, #-2176];		add	 r4, r4, r16 // 64-byte Folded Spill
	vlda	 x3, [sp, #-1024];		vldb	 x4, [p1, #64];		lt	 r27, r13, r6;		mov	r3, #8;		vst	 bmlh3, [sp, #-2112] // 64-byte Folded Reload64-byte Folded Spill
	vst	 bmhl3, [sp, #-2048];		vldb	 x2, [p1, #0];		add	 r5, r6, r14;		add.nc	lc, r3, #-1 // 64-byte Folded Spill
	mova	r3, #0;		nopb	;		vst	 bmhh3, [sp, #-1984];		sel.nez	 r5, r5, r6, r27;		mov	p1, p6;		vmul.f	dm0, x11, x1, r8 // 64-byte Folded Spill
	lshl	 r5, r3, r10;		mov	m0, r5;		vmul.f	dm1, x0, x1, r8
	paddb	 [p1], m0;		add	 r3, r3, r11
	vlda	 x5, [sp, #-960];		vldb	 x6, [p1, #0];		mov	m0, r5 // 64-byte Folded Reload
	vldb	 x8, [p1, #64];		mov	p1, p0
	vmac.f	dm1, dm1, x4, x3, r8
	paddb	 [p1], m0;		vmac.f	dm0, dm0, x2, x3, r8
	vlda	 bmll4, [sp, #-448]             // 64-byte Folded Reload
	vlda	 bmlh4, [sp, #-384]             // 64-byte Folded Reload
	vlda	 bmhl4, [sp, #-320]             // 64-byte Folded Reload
	vlda	 bmhh4, [sp, #-256];		vmac.f	dm0, dm0, x6, x5, r8 // 64-byte Folded Reload
	vlda	 bmll3, [sp, #-2176];		st	 r9, [sp, #-1344];		vmac.f	dm1, dm1, x8, x5, r8 // 64-byte Folded Reload4-byte Folded Spill
	vlda	 bmlh3, [sp, #-2112]            // 64-byte Folded Reload
	vlda	 bmhl3, [sp, #-2048];		vmov.d	dm1, dm3 // 64-byte Folded Reload
	vlda	 bmhh3, [sp, #-1984]            // 64-byte Folded Reload
	movxm	ls, #.LBB4_3
	vconv.bf16.fp32	 x10, cml0;		movxm	le, #.L_LEnd3
.LBB4_3:                                // %for.body18
                                        //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		vconv.bf16.fp32	 x7, cml1;		add	 r5, r0, r4;		nopm	;		nopv	
	nop	
	vst	 x10, [p1, #0];		add	 r6, r5, r14;		vmac.f	dm0, dm2, x10, x1, r8
	vst	 x7, [p1, #64];		lt	 r27, r13, r5;		mov	p1, p6;		vmac.f	dm4, dm4, x7, x1, r8
	sel.nez	 r5, r6, r5, r27;		vmac.f	dm3, dm3, x10, x10, r8
	add	 r6, r2, r4;		mov	m0, r5;		vmov.d	dm2, dm0
	padda	 [p1], m0;		vmac.f	dm1, dm1, x7, x7, r8
	vlda	 x11, [p1, #0];		add	 r7, r6, r14
	vldb	 x0, [p1, #64];		lt	 r27, r13, r6
	sel.nez	 r5, r7, r6, r27;		mov	p1, p6
	add	 r6, r1, r4;		mov	m0, r5;		vmov.d	dm3, dm1
	padda	 [p1], m0;		vst	 bmll3, [sp, #-2176];		add	 r4, r4, r16 // 64-byte Folded Spill
	vst	 bmlh3, [sp, #-2112];		vldb	 x4, [p1, #64];		lt	 r27, r13, r6 // 64-byte Folded Spill
	vst	 bmhl3, [sp, #-2048];		vldb	 x2, [p1, #0];		add	 r5, r6, r14 // 64-byte Folded Spill
	vst	 bmhh3, [sp, #-1984];		sel.nez	 r5, r5, r6, r27;		mov	p1, p6;		vmul.f	dm0, x11, x1, r8 // 64-byte Folded Spill
	lshl	 r5, r3, r10;		mov	m0, r5;		vmul.f	dm1, x0, x1, r8
	paddb	 [p1], m0;		add	 r3, r3, r11
	vldb	 x6, [p1, #0];		mov	m0, r5
	vldb	 x8, [p1, #64];		mov	p1, p0
	vmac.f	dm1, dm1, x4, x3, r8
	paddb	 [p1], m0;		vmac.f	dm0, dm0, x2, x3, r8
	nop	
	nop	
	nop	
	vmac.f	dm0, dm0, x6, x5, r8
	vlda	 bmll3, [sp, #-2176];		vmac.f	dm1, dm1, x8, x5, r8 // 64-byte Folded Reload
	vlda	 bmlh3, [sp, #-2112]            // 64-byte Folded Reload
	vlda	 bmhl3, [sp, #-2048];		vmov.d	dm1, dm3 // 64-byte Folded Reload
	vlda	 bmhh3, [sp, #-1984]            // 64-byte Folded Reload
	nop	
.L_LEnd3:
	nopa	;		nopb	;		vconv.bf16.fp32	 x10, cml0;		nopxm	;		nopv	
// %bb.4:                               // %for.cond.cleanup17
                                        //   in Loop: Header=BB4_2 Depth=2
	lda	 r9, [sp, #-192];		vconv.bf16.fp32	 x7, cml1 // 4-byte Folded Reload
	mov	p0, p6
	vst	 x10, [p1, #0];		vmac.f	dm0, dm2, x10, x1, r8
	vst	 x7, [p1, #64];		vmac.f	dm4, dm4, x7, x1, r8
	vmac.f	dm3, dm3, x10, x10, r8
	nop	
	vmac.f	dm1, dm1, x7, x7, r8
	mov	r0, r9
	vst	 bmll0, [sp, #-704]             // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-640]             // 64-byte Folded Spill
	vst	 bmhl0, [sp, #-576]             // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-512]             // 64-byte Folded Spill
	vst	 bmll3, [sp, #-2176]            // 64-byte Folded Spill
	vst	 bmlh3, [sp, #-2112]            // 64-byte Folded Spill
	vst	 bmhl3, [sp, #-2048]            // 64-byte Folded Spill
	vst	 bmhh3, [sp, #-1984]            // 64-byte Folded Spill
	vst	 bmll4, [sp, #-448]             // 64-byte Folded Spill
	vst	 bmlh4, [sp, #-384]             // 64-byte Folded Spill
	vst	 bmhl4, [sp, #-320]             // 64-byte Folded Spill
	vst	 bmhh4, [sp, #-256];		jl	#_ZL7in_donePji // 64-byte Folded Spill
	vst	 bmll1, [sp, #-1920]            // 64-byte Folded Spill Delay Slot 5
	vst	 bmlh1, [sp, #-1856]            // 64-byte Folded Spill Delay Slot 4
	vst	 bmhl1, [sp, #-1792]            // 64-byte Folded Spill Delay Slot 3
	vst	 bmhh1, [sp, #-1728]            // 64-byte Folded Spill Delay Slot 2
	nop	                                //  Delay Slot 1
	lda	 r1, [sp, #-1344];		nopb	;		nops	;		nopxm	;		nopv	 // 4-byte Folded Reload
	nopx	
	nop	
	nop	
	nop	
	nop	
	movx	r0, #512
	add	 r1, r1, r0
	movxm	r0, #6144
	eq	 r0, r1, r0
	jz	 r0, #.LBB4_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.5:                               // %for.cond.cleanup12
                                        //   in Loop: Header=BB4_1 Depth=1
	vlda	 bmll0, [sp, #-704]             // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-640]             // 64-byte Folded Reload
	nop	
	lda	 p0, [sp, #-172]                // 4-byte Folded Reload
	jl	#_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vmov	x4, bmll0                       //  Delay Slot 3
	vmov	x5, bmlh0                       //  Delay Slot 2
	nop	                                //  Delay Slot 1
	vlda	 bmll0, [sp, #-2176];		nopb	;		nops	;		nopxm	;		nopv	 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-2112]            // 64-byte Folded Reload
	lda	 r9, [sp, #-140]                // 4-byte Folded Reload
	nop	
	jl	#_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vmov	x4, bmll0                       //  Delay Slot 3
	vmov	x5, bmlh0                       //  Delay Slot 2
	mov	p0, r9                          //  Delay Slot 1
	lda	 p0, [sp, #-172];		nopb	;		nopx	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vlda	 x0, [sp, #-896];		vldb	 x5, [p0, #64] // 64-byte Folded Reload
	lda	 p1, [sp, #-152];		vldb	 x4, [p0, #0];		mov	p0, r9 // 4-byte Folded Reload
	lda	 p2, [sp, #-176];		vldb	 x7, [p0, #64] // 4-byte Folded Reload
	lda	 p0, [sp, #-160];		vldb	 x6, [p0, #0] // 4-byte Folded Reload
	jl	#_ZL8ln_statsN3aie5accumIu10__accfloatLj32EEES1_NS_6vectorI8bfloat16Lj32EEERS4_S5_RS1_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
	vlda	 bmll0, [sp, #-448];		nopb	;		nops	;		nopxm	;		nopv	 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-384];		nopx	 // 64-byte Folded Reload
	nop	
	lda	 p0, [sp, #-168]                // 4-byte Folded Reload
	jl	#_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vmov	x4, bmll0                       //  Delay Slot 3
	vmov	x5, bmlh0                       //  Delay Slot 2
	nop	                                //  Delay Slot 1
	vlda	 bmll0, [sp, #-1920];		nopb	;		nops	;		nopxm	;		nopv	 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-1856]            // 64-byte Folded Reload
	lda	 r9, [sp, #-136]                // 4-byte Folded Reload
	nop	
	jl	#_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vmov	x4, bmll0                       //  Delay Slot 3
	vmov	x5, bmlh0                       //  Delay Slot 2
	mov	p0, r9                          //  Delay Slot 1
	lda	 p0, [sp, #-168];		nopx	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vlda	 x0, [sp, #-896];		vldb	 x5, [p0, #64] // 64-byte Folded Reload
	lda	 p1, [sp, #-148];		vldb	 x4, [p0, #0];		mov	p0, r9 // 4-byte Folded Reload
	lda	 p2, [sp, #-144];		vldb	 x7, [p0, #64] // 4-byte Folded Reload
	lda	 p0, [sp, #-156];		vldb	 x6, [p0, #0] // 4-byte Folded Reload
	jl	#_ZL8ln_statsN3aie5accumIu10__accfloatLj32EEES1_NS_6vectorI8bfloat16Lj32EEERS4_S5_RS1_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mov	r9, p2                          //  Delay Slot 1
	lda	 p0, [sp, #-176];		nopb	;		nops	;		nopxm	;		nopv	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	vlda	 x0, [sp, #-1664]
	vlda	 bmll0, [p0, #0]
	vlda	 bmlh0, [p0, #64]
	vlda	 x0, [sp, #-1536];		mov	p0, r9
	vlda	 bmll0, [p0, #0]
	vlda	 bmlh0, [p0, #64]
	vlda	 x0, [sp, #-1600]
	vlda	 x0, [sp, #-1472];		vst	 x0, [sp, #-1344] // 64-byte Folded Spill
	vst	 bmll0, [sp, #-448]             // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-384]             // 64-byte Folded Spill
	vst	 x0, [sp, #-1280]               // 64-byte Folded Spill
	vst	 bmll0, [sp, #-704]             // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-640]             // 64-byte Folded Spill
	vst	 x0, [sp, #-1216]               // 64-byte Folded Spill
	mova	r9, #0;		vst	 x0, [sp, #-1152];		movx	r1, #0 // 64-byte Folded Spill
.LBB4_6:                                // %for.body67
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_7 Depth 3
                                        //       Child Loop BB4_9 Depth 3
                                        //       Child Loop BB4_12 Depth 3
	nopa	;		nopb	;		jl	#_ZL9out_beginPji;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	st	 r9, [sp, #-1920]               // 4-byte Folded Spill Delay Slot 2
	movs	p0, p7;		mov	r9, r1          //  Delay Slot 1
	vlda	 x10, [sp, #-1408];		nopb	;		nops	;		nopxm	;		nopv	 // 64-byte Folded Reload
	vlda	 x1, [sp, #-1088];		nopx	 // 64-byte Folded Reload
	vlda	 bmll3, [sp, #-448];		movxm	dj0, #-2176 // 64-byte Folded Reload
	vlda	 bmlh3, [sp, #-384];		movxm	p1, ##_ZL7g_arena // 64-byte Folded Reload
	vlda	 bmll4, [sp, #-704];		movxm	ls, #.LBB4_7 // 64-byte Folded Reload
	vlda	 bmlh4, [sp, #-640];		movxm	le, #.L_LEnd2 // 64-byte Folded Reload
	vlda	 x3, [sp, #-1344];		movxm	p2, ##_ZL8g_panels // 64-byte Folded Reload
	lda	 r9, [sp, #-1920];		movxm	r6, #1039360 // 4-byte Folded Reload
	vlda	 x5, [sp, #-1280];		movxm	p0, ##_ZL7g_out_b // 64-byte Folded Reload
	lda	 r0, [p0, #0];		movxm	r7, #1039424
	mova	r26, #0;		movxm	dj1, #12288
	mova	r16, #128;		movxm	dj2, #13824
	mova	r3, #8;		lshl	 r1, r9, r10;		mov	p0, sp
	vlda	 x7, [sp, #-1216];		st	 r9, [p0, dj0];		movxm	p0, ##_ZL7g_arena // 64-byte Folded Reload4-byte Folded Spill
	vlda	 x9, [sp, #-1152];		movs	m0, r1;		lshl	 r2, r9, r10;		add.nc	lc, r3, #0 // 64-byte Folded Reload
	padda	 [p0], m0;		movx	r1, #0;		mov	m0, r2
	padda	 [p1], m0;		movx	r2, #0;		mov	r3, r0
.LBB4_7:                                // %for.body76
                                        //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lshl	 r4, r1, r10
	movs	m0, r4;		add	 r5, r3, r7;		mov	p3, p0
	padda	 [p3], m0;		add	 r4, r3, r11
	vldb	 x0, [p3, #0];		lt	 r27, r15, r4
	sel.nez	 r4, r5, r4, r27
	vldb	 x2, [p3, #64];		add	 r5, r3, r6
	lt	 r27, r15, r3
	sel.nez	 r5, r5, r3, r27
	lshl	 r5, r2, r10;		mov	dj0, r5
	movs	m0, r5
	vst	 x0, [p7, dj0]
	movs	dj0, r4;		mov	p4, p1
	paddb	 [p4], m0;		vst	 x2, [p7, dj0]
	vlda.128	 wl4, [p4, dj1];		vmac.f	dm0, dm2, x0, x3, r8
	vlda.128	 wl6, [p4, dj2];		vmov	cml2, cml3;		vmac.f	dm2, dm2, x2, x7, r8
	vmov	cml2, cml4
	vmac.f	dm1, dm0, x0, x5, r8
	vmac.f	dm2, dm2, x2, x9, r8
	nop	
	nop	
	vshift	x11, x0, x4, r12
	vshift	x0, x0, x6, r12
	vconv.bf16.fp32	 x8, cml1;		vshift	x0, x0, x6, r12
	vconv.bf16.fp32	 x2, cml2;		vshift	x0, x0, x6, r12
	vshift	x0, x0, x6, r12;		vmac.f	dm0, dm1, x8, x1, r8
	vshift	x11, x11, x4, r12;		vmac.f	dm1, dm2, x2, x1, r8
	vshift	x11, x11, x4, r12;		vmul.f	dm2, x0, x10, r8
	vshift	x11, x11, x4, r12
	nop	
	vmac.f	dm0, dm2, x8, x11, r8
	vconv.bf16.fp32	 x4, cml0;		vmac.f	dm2, dm2, x2, x11, r8
	vconv.bf16.fp32	 x6, cml1
	vmac.f	dm0, dm0, x4, x11, r8
	vmac.f	dm2, dm2, x6, x11, r8
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x8, cml0
	vconv.bf16.fp32	 x0, cml2
	vconv.fp32.bf16	cml1, x8
	vst	 x8, [p3, #0];		vconv.fp32.bf16	cmh1, x0
	vst	 x0, [p3, #64];		add	 r1, r1, r11
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		add	 r3, r3, r16
.L_LEnd2:
	nopa	;		nopb	;		vst.flush.512.conv	 [p2, sf, r26];		add	r2, r2, #8;		nopm	;		nopv	
// %bb.8:                               // %for.cond130.preheader
                                        //   in Loop: Header=BB4_6 Depth=2
	movxm	r1, #1040384
	movxm	ls, #.LBB4_9
	mova	r2, #9;		movxm	le, #.L_LEnd1
	mova	r3, #0;		movxm	p1, ##_ZL8g_panels
	nopa	;		nopb	;		nops	;		lshl	 r2, r3, r10;		add.nc	lc, r2, #-1;		nopv	
	nopa	;		nopb	;		nops	;		nopx	;		mov	dj0, r2;		nopv	
	nopa	;		vldb	 x2, [p1, dj0];		nops	;		movxm	r5, #-1039360;		nopv	
	nopa	;		nopb	;		nops	;		add	 r1, r0, r1;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		add	r2, r3, #32;		nopm	;		nopv	
.LBB4_9:                                // %for.body133
                                        //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		nops	;		add	 r3, r1, r5;		nopm	;		nopv	
	nopa	;		lt	 r27, r15, r3
	sel.nez	 r3, r1, r3, r27
	lshl	 r4, r2, r10;		mov	dj0, r3
	vst	 x2, [p7, dj0];		mov	dj0, r4
	vldb	 x2, [p1, dj0]
	add	r2, r2, #32
.L_LEnd1:
	nopa	;		nopb	;		nops	;		add	 r1, r1, r11;		nopm	;		nopv	
// %bb.10:                              //   in Loop: Header=BB4_6 Depth=2
	lda	 r1, [sp, #-180];		nopb	;		nopxm	;		nops	 // 4-byte Folded Reload
	nop	
	nop	
	add	 r2, r1, r5
	lt	 r27, r15, r2
	sel.nez	 r1, r1, r2, r27
	mov	dj0, r1
	jz	 r1, #.LBB4_14
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	vst	 x2, [p7, dj0]                  //  Delay Slot 1
// %bb.11:                              // %for.body149.preheader
                                        //   in Loop: Header=BB4_6 Depth=2
	mova	r2, #0
	lshl	 r1, r2, r10
	mov	dj0, r1
	vldb	 x2, [p0, dj0];		movxm	ls, #.LBB4_12
	movxm	le, #.L_LEnd0
	nopa	;		nopb	;		nops	;		add	r1, r2, #32;		add.nc	lc, r12, #-1;		nopv	
.LBB4_12:                               // %for.body149
                                        //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		nops	;		add	 r2, r0, r6;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		lt	 r27, r15, r0;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		sel.nez	 r2, r2, r0, r27;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		lshl	 r3, r1, r10;		mov	dj0, r2;		nopv	
	nopa	;		nopb	;		vst	 x2, [p7, dj0];		nopx	;		mov	dj0, r3;		nopv	
	nopa	;		vldb	 x2, [p0, dj0];		nopxm	
	add	r1, r1, #32
.L_LEnd0:
	nopa	;		nopb	;		nops	;		add	 r0, r0, r11;		nopm	;		nopv	
// %bb.13:                              //   in Loop: Header=BB4_6 Depth=2
	nopa	;		nopb	;		nops	;		add	 r1, r0, r6;		nopm	;		nopv	
	lt	 r27, r15, r0
	sel.nez	 r0, r1, r0, r27
	mov	dj0, r0
	vst	 x2, [p7, dj0]
.LBB4_14:                               // %if.end
                                        //   in Loop: Header=BB4_6 Depth=2
	nopa	;		nopb	;		nops	;		jl	#_ZL7out_endPji;		nopv	
	nopa	;		nopx	                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mov	p0, p7                          //  Delay Slot 1
	nopa	;		add	 r9, r9, r11;		mov	p0, sp
	movxm	dj0, #-2176
	lda	 r2, [p0, dj0];		movx	r0, #768 // 4-byte Folded Reload
	eq	 r0, r9, r0
	jz	 r0, #.LBB4_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movx	r1, #512                        //  Delay Slot 2
	add	 r1, r2, r1                     //  Delay Slot 1
// %bb.15:                              // %for.cond.cleanup66
                                        //   in Loop: Header=BB4_1 Depth=1
	lda	 r0, [sp, #-164]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	add	r0, r0, #-1
	jnz	 r0, #.LBB4_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.16:                              // %for.cond.cleanup
	lda	 p7, [sp, #-92]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-96]                 // 4-byte Folded Reload
	lda	 r15, [sp, #-100]               // 4-byte Folded Reload
	lda	 r14, [sp, #-104]               // 4-byte Folded Reload
	lda	 lr, [sp, #-132]                // 4-byte Folded Reload
	lda	 r13, [sp, #-108]               // 4-byte Folded Reload
	lda	 r12, [sp, #-112]               // 4-byte Folded Reload
	lda	 r11, [sp, #-116]               // 4-byte Folded Reload
	lda	 r10, [sp, #-120]               // 4-byte Folded Reload
	lda	 r9, [sp, #-124]                // 4-byte Folded Reload
	lda	 r8, [sp, #-128]                // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-2944                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end4:
	.size	_ZL9norm_rowsILb1EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE, .Lfunc_end4-_ZL9norm_rowsILb1EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE
                                        // -- End function
	.section	.text._ZL9norm_rowsILb0EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL9norm_rowsILb0EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE
	.type	_ZL9norm_rowsILb0EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE,@function
_ZL9norm_rowsILb0EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE: // @_ZL9norm_rowsILb0EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE
// %bb.0:                               // %entry
	nopa	;		nopb	;		paddxm	 [sp], #1536
	st	 r11, [sp, #-144]               // 4-byte Folded Spill
	st	 r14, [sp, #-132]               // 4-byte Folded Spill
	st	 r12, [sp, #-140]               // 4-byte Folded Spill
	st	 r13, [sp, #-136]               // 4-byte Folded Spill
	st	 r15, [sp, #-128]               // 4-byte Folded Spill
	st	 r10, [sp, #-148]               // 4-byte Folded Spill
	st	 p7, [sp, #-120];		movxm	r2, #16256 // 4-byte Folded Spill
	st	 lr, [sp, #-160];		movxm	r12, #15359 // 4-byte Folded Spill
	mova	r1, #10;		st	 p6, [sp, #-124];		movxm	r13, #1033216 // 4-byte Folded Spill
	mova	r9, #1;		st	 r9, [sp, #-152];		movxm	r15, #9215 // 4-byte Folded Spill
	mova	m0, #-768;		movs	p6, p0;		lt	 r27, r9, r0;		mov	p0, sp
	padda	 [p0], m0;		st	 r8, [sp, #-156];		movxm	m0, #-1536 // 4-byte Folded Spill
	mova	r6, #0;		vst	 x0, [sp, #-384];		lshl	 r3, r27, r1;		vbcst.16	 x2, r2 // 64-byte Folded Spill
	mova	r4, #2;		st	 r3, [sp, #-188];		sel.nez	 r3, r2, r6, r27 // 4-byte Folded Spill
	st	 p0, [sp, #-176];		lt	 r27, r4, r0;		vbcst.32	 x2, r6 // 4-byte Folded Spill
	vst	 x2, [sp, #-576];		lshl	 r0, r0, r1;		mov	p0, sp // 64-byte Folded Spill
	mova	r4, #11;		paddb	 [p0], m0;		st	 r0, [sp, #-192];		vmov	bmll0, x2 // 4-byte Folded Spill
	st	 p0, [sp, #-172];		lshl	 r4, r27, r4;		vmov	bmlh0, x2 // 4-byte Folded Spill
	st	 r4, [sp, #-184];		vbcst.16	 x4, r3 // 4-byte Folded Spill
	mova	m0, #-768;		vst	 bmll0, [sp, #-320];		sel.nez	 r5, r2, r6, r27;		mov	p0, sp // 64-byte Folded Spill
	padda	 [p0], m0;		vst	 bmlh0, [sp, #-256];		movxm	m0, #-1408 // 64-byte Folded Spill
	st	 p0, [sp, #-168];		vbcst.16	 x4, r5 // 4-byte Folded Spill
	vst	 x4, [sp, #-512];		mov	p0, sp // 64-byte Folded Spill
	padda	 [p0], m0;		vst	 x4, [sp, #-448];		movx	r11, #828;		mov	p7, p1 // 64-byte Folded Spill
	mova	r0, #18;		st	 p0, [sp, #-164];		movx	r10, #16;		mov	r14, #64 // 4-byte Folded Spill
.LBB5_1:                                // %for.cond10.preheader
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_2 Depth 2
                                        //       Child Loop BB5_3 Depth 3
                                        //     Child Loop BB5_6 Depth 2
                                        //       Child Loop BB5_7 Depth 3
                                        //       Child Loop BB5_9 Depth 3
	vlda	 bmll0, [sp, #-320];		nopx	 // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-256]             // 64-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	st	 r0, [sp, #-180]                // 4-byte Folded Spill
	vst	 bmll0, [sp, #-1280];		vmov	cml1, cml0 // 64-byte Folded Spill
	vst	 bmlh0, [sp, #-1216]            // 64-byte Folded Spill
	lda	 r8, [sp, #-192];		vst	 bmll1, [sp, #-1024] // 4-byte Folded Reload64-byte Folded Spill
	mova	r1, #0;		vst	 bmlh1, [sp, #-960] // 64-byte Folded Spill
.LBB5_2:                                // %for.body13
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_3 Depth 3
	nopa	;		nopb	;		jl	#_ZL7in_needPKji;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	movs	p0, p6;		or	 r0, r8, r8;		mov	r8, r1 //  Delay Slot 1
	lda	 r1, [sp, #-184];		nopb	;		movxm	p0, ##_ZL6g_in_b // 4-byte Folded Reload
	lda	 r0, [p0, #0]
	lda	 r2, [sp, #-188]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	lshl	 r2, r8, r9
	movxm	p0, ##_ZL7g_arena
	add	 r1, r1, r0;		mov	m0, r2
	mova	r4, #0;		add	 r2, r2, r0
	add	 r5, r0, r4;		mov	p2, p6
	vlda	 bmll2, [sp, #-1280];		add	 r6, r5, r13 // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-1216];		lt	 r27, r12, r5 // 64-byte Folded Reload
	padda	 [p0], m0;		sel.nez	 r5, r6, r5, r27
	vlda	 bmhl2, [sp, #-1152];		movs	m0, r5;		add	 r6, r2, r4 // 64-byte Folded Reload
	mova	r3, #8;		add	 r7, r6, r13
	vlda	 bmhh2, [sp, #-1088];		lt	 r27, r12, r6;		add.nc	lc, r3, #-1 // 64-byte Folded Reload
	vlda	 x10, [sp, #-576];		sel.nez	 r5, r7, r6, r27;		mov	r16, #128 // 64-byte Folded Reload
	vlda	 x1, [sp, #-512];		add	 r6, r1, r4;		mov	p1, p6 // 64-byte Folded Reload
	padda	 [p1], m0;		add	 r5, r6, r13;		mov	m0, r5
	vlda	 x2, [p1, #0];		lt	 r27, r12, r6;		mov	r3, #0
	vlda	 x4, [p1, #64];		sel.nez	 r5, r5, r6, r27;		mov	p1, p6
	padda	 [p2], m0;		lshl	 r5, r3, r9;		mov	m0, r5
	vlda	 x6, [p2, #0];		paddb	 [p1], m0;		add	 r4, r4, r16;		mov	m0, r5
	vlda	 bmll3, [sp, #-1024];		vldb	 x8, [p2, #64];		add	 r3, r3, r14 // 64-byte Folded Reload
	vlda	 bmlh3, [sp, #-960];		vldb	 x5, [p1, #0] // 64-byte Folded Reload
	vlda	 x3, [sp, #-448];		vldb	 x7, [p1, #64];		mov	p1, p0 // 64-byte Folded Reload
	vlda	 bmhl3, [sp, #-896];		paddb	 [p1], m0;		vmul.f	dm4, x2, x10, r11 // 64-byte Folded Reload
	vlda	 bmhh3, [sp, #-832];		st	 r8, [sp, #-704];		vmul.f	dm0, x4, x10, r11 // 64-byte Folded Reload4-byte Folded Spill
	nop	
	vmac.f	dm1, dm4, x6, x1, r11
	vmac.f	dm0, dm0, x8, x1, r11
	movxm	ls, #.LBB5_3
	movxm	le, #.L_LEnd6;		vmac.f	dm1, dm1, x5, x3, r11
.LBB5_3:                                // %for.body18
                                        //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		nops	;		add	 r5, r0, r4;		mov	p2, p6;		vmac.f	dm0, dm0, x7, x3, r11
	nopa	;		nopb	;		add	 r6, r5, r13;		nopm	
	lt	 r27, r12, r5
	sel.nez	 r5, r6, r5, r27
	movs	m0, r5;		add	 r6, r2, r4
	vconv.bf16.fp32	 x9, cml1;		add	 r7, r6, r13
	vconv.bf16.fp32	 x11, cml0;		lt	 r27, r12, r6
	vst	 x9, [p1, #0];		sel.nez	 r5, r7, r6, r27;		vmul.f	dm4, x9, x9, r11
	vst	 x11, [p1, #64];		add	 r6, r1, r4;		mov	p1, p6;		vmul.f	dm1, x11, x11, r11
	padda	 [p1], m0;		add	 r5, r6, r13;		mov	m0, r5
	vlda	 x2, [p1, #0];		lt	 r27, r12, r6
	vlda	 x4, [p1, #64];		sel.nez	 r5, r5, r6, r27;		mov	p1, p6
	padda	 [p2], m0;		lshl	 r5, r3, r9;		mov	m0, r5
	vlda	 x6, [p2, #0];		paddb	 [p1], m0;		add	 r4, r4, r16;		mov	m0, r5;		vconv.bf16.fp32	 x0, cml4
	vconv.bf16.fp32	 x2, cml1;		vldb	 x8, [p2, #64];		add	 r3, r3, r14
	vldb	 x5, [p1, #0];		vmac.f	dm3, dm3, x0, x10, r11
	vldb	 x7, [p1, #64];		mov	p1, p0;		vmac.f	dm2, dm2, x2, x10, r11
	paddb	 [p1], m0;		vmul.f	dm4, x2, x10, r11
	vmul.f	dm0, x4, x10, r11
	nop	
	vmac.f	dm1, dm4, x6, x1, r11
	vmac.f	dm0, dm0, x8, x1, r11
	nop	
.L_LEnd6:
	nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm1, dm1, x5, x3, r11
// %bb.4:                               // %for.cond.cleanup17
                                        //   in Loop: Header=BB5_2 Depth=2
	lda	 r8, [sp, #-192];		nopb	;		movs	p0, p6;		nopxm	;		vmac.f	dm0, dm0, x7, x3, r11 // 4-byte Folded Reload
	nopa	;		nopb	;		nopx	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cml1
	vconv.bf16.fp32	 x11, cml0
	vst	 x9, [p1, #0];		mov	r0, r8;		vmul.f	dm4, x9, x9, r11
	vst	 x11, [p1, #64];		vmul.f	dm1, x11, x11, r11
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml4
	vconv.bf16.fp32	 x2, cml1
	vmac.f	dm3, dm3, x0, x10, r11
	vmac.f	dm2, dm2, x2, x10, r11
	nop	
	nop	
	nop	
	nop	
	vst	 bmll3, [sp, #-1024]            // 64-byte Folded Spill
	vst	 bmlh3, [sp, #-960]             // 64-byte Folded Spill
	vst	 bmhl3, [sp, #-896]             // 64-byte Folded Spill
	vst	 bmhh3, [sp, #-832];		jl	#_ZL7in_donePji // 64-byte Folded Spill
	vst	 bmll2, [sp, #-1280]            // 64-byte Folded Spill Delay Slot 5
	vst	 bmlh2, [sp, #-1216]            // 64-byte Folded Spill Delay Slot 4
	vst	 bmhl2, [sp, #-1152]            // 64-byte Folded Spill Delay Slot 3
	vst	 bmhh2, [sp, #-1088]            // 64-byte Folded Spill Delay Slot 2
	nop	                                //  Delay Slot 1
	lda	 r1, [sp, #-704];		nopb	;		nops	;		nopxm	;		nopv	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	movx	r0, #512
	add	 r1, r1, r0
	movxm	r0, #6144
	eq	 r0, r1, r0
	jz	 r0, #.LBB5_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.5:                               // %for.cond.cleanup12
                                        //   in Loop: Header=BB5_1 Depth=1
	vlda	 bmll0, [sp, #-1024]            // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-960]             // 64-byte Folded Reload
	lda	 r8, [sp, #-172]                // 4-byte Folded Reload
	nop	
	jl	#_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vmov	x4, bmll0                       //  Delay Slot 3
	vmov	x5, bmlh0                       //  Delay Slot 2
	mov	p0, r8                          //  Delay Slot 1
	mov	p0, r8
	vlda	 x0, [sp, #-384];		vldb	 x5, [p0, #64] // 64-byte Folded Reload
	lda	 p0, [sp, #-176];		vldb	 x4, [p0, #0] // 4-byte Folded Reload
	jl	#_ZL8rms_rstdN3aie5accumIu10__accfloatLj32EEENS_6vectorI8bfloat16Lj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
	vlda	 x0, [sp, #-768];		nopb	;		nopx	
	vlda	 bmll0, [sp, #-1280]            // 64-byte Folded Reload
	vlda	 bmlh0, [sp, #-1216]            // 64-byte Folded Reload
	lda	 r8, [sp, #-164]                // 4-byte Folded Reload
	nop	
	jl	#_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vmov	x4, bmll0                       //  Delay Slot 3
	vst	 x0, [sp, #-704];		vmov	x5, bmlh0 // 64-byte Folded Spill Delay Slot 2
	mov	p0, r8                          //  Delay Slot 1
	mov	p0, r8
	vlda	 x0, [sp, #-384];		vldb	 x5, [p0, #64] // 64-byte Folded Reload
	lda	 p0, [sp, #-168];		vldb	 x4, [p0, #0] // 4-byte Folded Reload
	jl	#_ZL8rms_rstdN3aie5accumIu10__accfloatLj32EEENS_6vectorI8bfloat16Lj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
	vlda	 x0, [sp, #-768];		nopxm	
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	mova	r8, #0;		vst	 x0, [sp, #-640];		movx	r1, #0 // 64-byte Folded Spill
.LBB5_6:                                // %for.body67
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_7 Depth 3
                                        //       Child Loop BB5_9 Depth 3
	nopa	;		nopb	;		st	 r8, [sp, #-1024];		jl	#_ZL9out_beginPji;		nopv	 // 4-byte Folded Spill
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	movs	p0, p7;		mov	r8, r1          //  Delay Slot 1
	lda	 r8, [sp, #-1024];		nopb	;		nops	;		movxm	p0, ##_ZL7g_out_b;		nopv	 // 4-byte Folded Reload
	lda	 r0, [p0, #0];		nopb	;		lshl	 r1, r8, r9
	movxm	p1, ##_ZL7g_arena
	movxm	r6, #1039360
	movxm	r7, #1039424
	mova	r3, #8;		movxm	dj1, #12288
	st	 r8, [sp, #-1280];		movxm	p0, ##_ZL7g_arena // 4-byte Folded Spill
	movs	m0, r1;		lshl	 r2, r8, r9;		add.nc	lc, r3, #-1
	padda	 [p0], m0;		movs	m0, r2;		movx	r1, #0;		mov	r3, r0
	padda	 [p1], m0;		lshl	 r4, r1, r9;		mov	p3, p0
	movs	m0, r4;		add	 r4, r3, r14
	padda	 [p3], m0;		add	 r5, r3, r7
	vlda	 x6, [sp, #-704];		vldb	 x2, [p3, #0];		lt	 r27, r15, r4 // 64-byte Folded Reload
	vlda	 x8, [sp, #-640];		vldb	 x4, [p3, #64];		sel.nez	 r4, r5, r4, r27;		movs	p3, p1 // 64-byte Folded Reload
	add	 r5, r3, r6
	lt	 r27, r15, r3
	mova	r2, #0;		sel.nez	 r5, r5, r3, r27
	lshl	 r5, r2, r9;		mov	dj0, r5
	mova	r16, #128;		add	r2, r2, #8;		mov	m0, r5
	padda	 [p3], m0;		nopb	;		vst	 x2, [p7, dj0];		add	 r1, r1, r14;		mov	dj0, r4;		vmul.f	dm0, x2, x6, r11
	vst	 x4, [p7, dj0];		add	 r3, r3, r16;		vmul.f	dm1, x4, x8, r11
	vlda.128	 wl10, [p3, dj1]
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cml0
	nop	
	vconv.bf16.fp32	 x11, cml1
	vshift	x1, x0, x10, r10
	vshift	x3, x1, x10, r10
	vshift	x5, x3, x10, r10
	vshift	x7, x5, x10, r10
	nop	
	vmul.f	dm2, x9, x7, r11
	vmul.f	dm3, x11, x7, r11
	nop	
	nop	
	movxm	ls, #.LBB5_7
	movxm	le, #.L_LEnd5
	mova	r26, #0;		vconv.bf16.fp32	 x0, cml2;		movxm	p2, ##_ZL8g_panels
.LBB5_7:                                // %for.body76
                                        //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		vconv.bf16.fp32	 x2, cml3;		lshl	 r4, r1, r9;		mov	p3, p0;		nopv	
	nopa	;		movs	m0, r4;		add	 r4, r3, r14;		vconv.fp32.bf16	cml4, x0
	padda	 [p3], m0;		add	 r5, r3, r7;		vconv.fp32.bf16	cmh4, x2
	vldb	 x2, [p3, #0];		lt	 r27, r15, r4
	movs	p3, p1;		vldb	 x4, [p3, #64];		sel.nez	 r4, r5, r4, r27
	vst.push.576.conv.bfp16ebs8.fp32	 dm4, [p2, sf, r26];		add	 r5, r3, r6
	vst.flush.512.conv	 [p2, sf, r26];		lt	 r27, r15, r3
	sel.nez	 r5, r5, r3, r27
	lshl	 r5, r2, r9;		mov	dj0, r5
	add	r2, r2, #8;		mov	m0, r5
	padda	 [p3], m0;		nopb	;		vst	 x2, [p7, dj0];		add	 r1, r1, r14;		mov	dj0, r4;		vmul.f	dm0, x2, x6, r11
	vst	 x4, [p7, dj0];		add	 r3, r3, r16;		vmul.f	dm1, x4, x8, r11
	vlda.128	 wl10, [p3, dj1]
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cml0
	nop	
	vconv.bf16.fp32	 x11, cml1
	vshift	x1, x0, x10, r10
	vshift	x3, x1, x10, r10
	vshift	x5, x3, x10, r10
	vshift	x7, x5, x10, r10
	nop	
	vmul.f	dm2, x9, x7, r11
	vmul.f	dm3, x11, x7, r11
	nop	
	nop	
	nop	
	nop	
.L_LEnd5:
	nopa	;		nopb	;		vconv.bf16.fp32	 x0, cml2;		nopxm	;		nopv	
// %bb.8:                               // %for.cond113.preheader
                                        //   in Loop: Header=BB5_6 Depth=2
	mova	r2, #0;		nopb	;		vconv.bf16.fp32	 x2, cml3;		movxm	r1, #1040384;		nopv	
	mova	r1, #9;		nopb	;		nops	;		add	 r0, r0, r1;		vconv.fp32.bf16	cml4, x0;		nopv	
	nopx	;		vconv.fp32.bf16	cmh4, x2
	lshl	 r1, r2, r9;		add.nc	lc, r1, #-1
	add	r1, r2, #32;		mov	dj0, r1
	vst.push.576.conv.bfp16ebs8.fp32	 dm4, [p2, sf, r26];		movxm	p0, ##_ZL8g_panels
	vst.flush.512.conv	 [p2, sf, r26];		movxm	ls, #.LBB5_9
	nop	
	nop	
	nop	
	vldb	 x2, [p0, dj0]
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd4;		nopv	
	nopa	;		nopb	;		nops	;		movxm	r4, #-1039360;		nopv	
.LBB5_9:                                // %for.body116
                                        //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		nops	;		add	 r2, r0, r4;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		lt	 r27, r15, r2;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		sel.nez	 r2, r0, r2, r27;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		lshl	 r3, r1, r9;		mov	dj0, r2;		nopv	
	vst	 x2, [p7, dj0];		mov	dj0, r3
	vldb	 x2, [p0, dj0]
	add	r1, r1, #32
.L_LEnd4:
	nopa	;		nopb	;		nops	;		add	 r0, r0, r14;		nopm	;		nopv	
// %bb.10:                              //   in Loop: Header=BB5_6 Depth=2
	nopa	;		add	 r1, r0, r4
	jl	#_ZL7out_endPji
	lt	 r27, r15, r1                   //  Delay Slot 5
	sel.nez	 r0, r0, r1, r27                //  Delay Slot 4
	mov	dj0, r0                         //  Delay Slot 3
	vst	 x2, [p7, dj0]                  //  Delay Slot 2
	mov	p0, p7                          //  Delay Slot 1
	nopa	;		nopb	;		nops	;		add	 r8, r8, r14;		nopm	;		nopv	
	lda	 r2, [sp, #-1280];		movx	r0, #768 // 4-byte Folded Reload
	eq	 r0, r8, r0
	jz	 r0, #.LBB5_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movx	r1, #512                        //  Delay Slot 2
	add	 r1, r2, r1                     //  Delay Slot 1
// %bb.11:                              // %for.cond.cleanup66
                                        //   in Loop: Header=BB5_1 Depth=1
	lda	 r0, [sp, #-180]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	add	r0, r0, #-1
	jnz	 r0, #.LBB5_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.12:                              // %for.cond.cleanup
	lda	 p7, [sp, #-120]                // 4-byte Folded Reload
	lda	 p6, [sp, #-124]                // 4-byte Folded Reload
	lda	 r15, [sp, #-128]               // 4-byte Folded Reload
	lda	 r14, [sp, #-132]               // 4-byte Folded Reload
	lda	 lr, [sp, #-160]                // 4-byte Folded Reload
	lda	 r13, [sp, #-136]               // 4-byte Folded Reload
	lda	 r12, [sp, #-140]               // 4-byte Folded Reload
	lda	 r11, [sp, #-144]               // 4-byte Folded Reload
	lda	 r10, [sp, #-148]               // 4-byte Folded Reload
	lda	 r9, [sp, #-152]                // 4-byte Folded Reload
	lda	 r8, [sp, #-156]                // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-1536                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end5:
	.size	_ZL9norm_rowsILb0EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE, .Lfunc_end5-_ZL9norm_rowsILb0EEvPjS0_iiN3aie6vectorI8bfloat16Lj32EEE
                                        // -- End function
	.section	.text.ew_patch,"ax",@progbits
	.globl	ew_patch                        // -- Begin function ew_patch
	.p2align	4
	.type	ew_patch,@function
ew_patch:                               // @ew_patch
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	p2, ##_ZL7g_arena;		nops	
	paddxm	 [sp], #1088
	st	 r11, [sp, #-48];		movxm	m0, #13824 // 4-byte Folded Spill
	st	 p6, [sp, #-28];		mov	crrnd, #12 // 4-byte Folded Spill
	st	 r12, [sp, #-44];		mov	p6, p0 // 4-byte Folded Spill
	st	 r8, [sp, #-60];		movxm	r8, #3072 // 4-byte Folded Spill
	st	 r9, [sp, #-56];		movxm	r9, #15359 // 4-byte Folded Spill
	st	 r10, [sp, #-52];		j	#.LBB6_7 // 4-byte Folded Spill
	st	 r15, [sp, #-32];		mov	r15, p1 // 4-byte Folded Spill Delay Slot 5
	st	 p7, [sp, #-24];		movxm	r10, #1033216 // 4-byte Folded Spill Delay Slot 4
	mova	r11, #12;		st	 lr, [sp, #-64];		mov	p7, sp // 4-byte Folded Spill Delay Slot 3
	padda	 [p2], m0;		st	 r13, [sp, #-40];		movxm	m0, #-1088 // 4-byte Folded Spill Delay Slot 2
	padda	 [p7], m0;		st	 r14, [sp, #-36];		movx	r12, #128;		mov	r14, p2 // 4-byte Folded Spill Delay Slot 1
.LBB6_1:                                // %for.body8
                                        //   in Loop: Header=BB6_2 Depth=3
	nopa	;		add	 r2, r0, r1
	add	 r3, r2, r10
	lt	 r27, r9, r2
	sel.nez	 r2, r3, r2, r27
	movs	p0, p6;		mov	m0, r2
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
	nop	
	add	 r1, r1, r12
.LBB6_2:                                // %for.cond5
                                        //   Parent Loop BB6_7 Depth=1
                                        //     Parent Loop BB6_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		ne	 r2, r1, r8;		nopm	
	jnz	 r2, #.LBB6_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %for.cond.cleanup7
                                        //   in Loop: Header=BB6_4 Depth=2
	jl	#_ZL7in_donePji
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movxm	r0, #3072                       //  Delay Slot 2
	mov	p0, p6                          //  Delay Slot 1
	nopa	;		nopb	;		nops	;		jl	#_ZL5gemm8PK9bfp16ebs8S1_iPKN3aie6vectorI8bfloat16Lj64EEEPS4_i;		nopv	
	nopa	;		nopx	                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mov	p0, p7                          //  Delay Slot 1
	nopa	;		nopb	;		nops	;		jl	#_ZL9out_bytesPjPKvi;		nopv	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	movs	p0, r15;		mov	p1, p7          //  Delay Slot 1
	nopa	;		nopb	;		nops	;		add	r13, r13, #-1;		nopm	;		nopv	
.LBB6_4:                                // %for.cond1
                                        //   Parent Loop BB6_7 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_2 Depth 3
	nopa	;		nopb	;		nops	;		jz	 r13, #.LBB6_6;		nopv	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.5:                               // %for.body4
                                        //   in Loop: Header=BB6_4 Depth=2
	jl	#_ZL7in_needPKji
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movxm	r0, #3072                       //  Delay Slot 2
	mov	p0, p6                          //  Delay Slot 1
	j	#.LBB6_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mova	r1, #0;		movxm	p0, ##_ZL6g_in_b //  Delay Slot 2
	lda	 r0, [p0, #0];		movx	r26, #0;		mov	p2, r14 //  Delay Slot 1
.LBB6_6:                                // %for.cond.cleanup3
                                        //   in Loop: Header=BB6_7 Depth=1
	nopa	;		nopb	;		nops	;		add	r11, r11, #-1;		nopm	;		nopv	
.LBB6_7:                                // %for.cond
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_4 Depth 2
                                        //       Child Loop BB6_2 Depth 3
	nopa	;		nopb	;		nops	;		jz	 r11, #.LBB6_9;		nopv	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.8:                               // %for.body
                                        //   in Loop: Header=BB6_7 Depth=1
	jl	#_ZL7in_needPKji
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movxm	r0, #14592                      //  Delay Slot 2
	mov	p0, p6                          //  Delay Slot 1
	nopa	;		jl	#_ZL7in_copyPKjPvii
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 3
	movxm	r1, #13824                      //  Delay Slot 2
	mova	r0, #0;		mov	p0, p6          //  Delay Slot 1
	nopa	;		nopb	;		jl	#_ZL7in_copyPKjPvii;		nops	
	nop	                                //  Delay Slot 5
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 4
	movxm	m0, #15552                      //  Delay Slot 3
	padda	 [p1], m0;		movxm	r0, #13824 //  Delay Slot 2
	mova	r1, #128;		movs	p0, p6;		mov	r13, p1 //  Delay Slot 1
	nopa	;		nopb	;		nops	;		jl	#_ZL7in_donePji;		nopv	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movxm	r0, #14592                      //  Delay Slot 2
	mov	p0, p6                          //  Delay Slot 1
	nopa	;		jl	#_ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE
	nop	                                //  Delay Slot 5
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 4
	movxm	m0, #15680                      //  Delay Slot 3
	padda	 [p1], m0                       //  Delay Slot 2
	mov	p0, r13                         //  Delay Slot 1
	nopa	;		nopb	;		nops	;		j	#.LBB6_4;		nopv	
	nopa	;		nopx	                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r13, #18                        //  Delay Slot 1
.LBB6_9:                                // %for.cond.cleanup
	lda	 p7, [sp, #-24];		nopx	        // 4-byte Folded Reload
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
	paddxm	 [sp], #-1088                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end6:
	.size	ew_patch, .Lfunc_end6-ew_patch
                                        // -- End function
	.section	.text._ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE
	.type	_ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE,@function
_ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE: // @_ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE
// %bb.0:                               // %entry
	j	#.LBB7_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	mova	r2, #16;		movx	r1, #0;		mov	r0, #8 //  Delay Slot 2
	mova	r5, #0;		movx	r4, #7;		mov	r3, #1 //  Delay Slot 1
.LBB7_1:                                // %for.body
                                        //   in Loop: Header=BB7_2 Depth=1
	nopa	;		nopb	;		nops	;		lshl	 r6, r1, r3;		nopm	;		nopv	
	nopx	;		mov	dj0, r6
	vldb.128	 wl0, [p0, dj0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vshift	x2, x0, x0, r2
	vshift	x2, x2, x0, r2
	movs	p2, p1;		lshl	 r6, r5, r4;		vshift	x2, x2, x0, r2
	movs	m0, r6;		vshift	x0, x2, x0, r2
	padda	 [p2], m0
	vst	 x0, [p2, #0];		add	r1, r1, #8
	vst	 x0, [p2, #64];		add	r5, r5, #1
.LBB7_2:                                // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		ne	 r6, r5, r0;		nopm	;		nopv	
	jnz	 r6, #.LBB7_1
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %for.cond.cleanup
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end7:
	.size	_ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE, .Lfunc_end7-_ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE
                                        // -- End function
	.section	.text._ZL5gemm8PK9bfp16ebs8S1_iPKN3aie6vectorI8bfloat16Lj64EEEPS4_i,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL5gemm8PK9bfp16ebs8S1_iPKN3aie6vectorI8bfloat16Lj64EEEPS4_i
	.type	_ZL5gemm8PK9bfp16ebs8S1_iPKN3aie6vectorI8bfloat16Lj64EEEPS4_i,@function
_ZL5gemm8PK9bfp16ebs8S1_iPKN3aie6vectorI8bfloat16Lj64EEEPS4_i: // @_ZL5gemm8PK9bfp16ebs8S1_iPKN3aie6vectorI8bfloat16Lj64EEEPS4_i
// %bb.0:                               // %entry
	mova	r0, #1;		nopxm	
	mova	r1, #7;		movxm	p3, ##_ZL7g_arena
	mova	r3, #0;		movxm	r2, #4608
	mova	r4, #780;		movxm	r6, #16256
	mova	r5, #8;		movxm	r7, #1152
	mova	r16, #2;		movxm	m0, #13824
	mova	r17, #828;		movxm	dj0, #15680
	mova	r18, #1;		movxm	dj1, #15744
	mova	r19, #0;		paddxm	 [sp], #192
	mova	r21, #0;		movx	r20, #0;		vbcst.16	 x0, r6
	padda	 [p3], m0;		movs	p4, p0;		movx	r6, #3;		vmov	x1, x0
.LBB8_1:                                // %for.cond1.preheader
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_3 Depth 2
                                        //       Child Loop BB8_6 Depth 3
                                        //     Child Loop BB8_9 Depth 2
	j	#.LBB8_3
	nopx	                                //  Delay Slot 5
	movxm	p2, ##_ZL7g_arena               //  Delay Slot 4
	movs	m0, r20;		mov	r24, r3         //  Delay Slot 3
	mova	r23, #0;		st	 r24, [sp, #-64];		movx	r22, #0;		mov	p0, p3 // 4-byte Folded Spill Delay Slot 2
	padda	 [p2], m0;		st	 p0, [sp, #-60];		movx	r29, #0;		mov	r26, #781 // 4-byte Folded Spill Delay Slot 1
.LBB8_2:                                // %for.cond.cleanup10
                                        //   in Loop: Header=BB8_3 Depth=2
	nopa	;		add	r23, r23, #1;		nopm	
	eq	 r25, r23, r6
	jnz	 r25, #.LBB8_8
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	add	 r22, r22, r2                   //  Delay Slot 1
.LBB8_3:                                // %for.body4
                                        //   Parent Loop BB8_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB8_6 Depth 3
	j	#.LBB8_6
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movs	p0, p2;		mov	m0, r22         //  Delay Slot 2
	padda	 [p0], m0;		movx	r27, #0;		mov	r24, r3 //  Delay Slot 1
.LBB8_4:                                // %if.else
                                        //   in Loop: Header=BB8_6 Depth=3
	nopa	;		nopb	;		nopx	;		vmac.f	dm1, dm1, ex2, ex4, r26
	vmac.f	dm0, dm0, ex2, ex6, r26
.LBB8_5:                                // %if.end
                                        //   in Loop: Header=BB8_6 Depth=3
	nopa	;		add	r27, r27, #1
	eq	 r28, r27, r5
	jnz	 r28, #.LBB8_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	add	r25, r29, #1                    //  Delay Slot 2
	mova	r26, #780;		and	 r29, r25, r1 //  Delay Slot 1
.LBB8_6:                                // %_ZN3aie41block_vector_restrict_input_buffer_streamI9bfp16ebs8Lj64EL15aie_dm_resource5EE3popEv.exit
                                        //   Parent Loop BB8_1 Depth=1
                                        //     Parent Loop BB8_3 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lda	 r25, [sp, #-64];		nopxm	 // 4-byte Folded Reload
	lda	 p1, [sp, #-60]                 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	vlda	 lfl1, [sp, #-192]              // 64-byte Folded Reload
	vlda	 lfh1, [sp, #-128]              // 64-byte Folded Reload
	vlda.fill.512	 [p0, lf0, r24];		vldb.fill.512	 [p1, lf1, r25]
	vlda.pop.576	 ex4, [p0, lf0, r24];		vldb.pop.576	 ex2, [p1, lf1, r25]
	vldb.fill.512	 [p0, lf0, r24]
	vldb.pop.576	 ex6, [p0, lf0, r24];		or	 r28, r27, r23
	jnz	 r28, #.LBB8_4
	nop	                                //  Delay Slot 5
	st	 r25, [sp, #-64]                // 4-byte Folded Spill Delay Slot 4
	st	 p1, [sp, #-60]                 // 4-byte Folded Spill Delay Slot 3
	vst	 lfl1, [sp, #-192]              // 64-byte Folded Spill Delay Slot 2
	vst	 lfh1, [sp, #-128]              // 64-byte Folded Spill Delay Slot 1
// %bb.7:                               // %if.then
                                        //   in Loop: Header=BB8_6 Depth=3
	j	#.LBB8_5
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vmul.f	dm1, ex2, ex4, r4               //  Delay Slot 2
	vmul.f	dm0, ex2, ex6, r4               //  Delay Slot 1
.LBB8_8:                                // %for.cond23.preheader
                                        //   in Loop: Header=BB8_1 Depth=1
	nopx	;		add.nc	lc, r16, #-1
	vmov	x10, x0
	vmov	x11, x1
	vmov	x6, bmll1
	vmov	x7, bmll0
	vmov	x2, bmlh1
	add	 r23, r21, r3;		vmov	x8, bmlh0
	movxm	p1, ##_ZL7g_arena
	lshl	 r23, r23, r1;		vmov	x4, bmhl1
	mov	m0, r23
	padda	 [p1], m0;		vmov	x3, bmhl0
	vldb	 x0, [p1, dj0];		vmov	x5, bmhh1
	vldb	 x1, [p1, dj1];		add	r22, r18, #-1;		vmov	x9, bmhh0
	vsel.32	 x0, x6, x7, r22
	vmov	bmll1, x0
	vsel.32	 x0, x4, x3, r22
	vsel.32	 x1, x2, x8, r22
	vmov	bmhl1, x0
	vmov	bmlh1, x1
	vsel.32	 x9, x5, x9, r22;		vmac.f	dm4, dm1, y0, y5, r17
	vmov	bmhh1, x9
	movxm	ls, #.LBB8_9
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd7;		nopv	
.LBB8_9:                                // %for.body26
                                        //   Parent Loop BB8_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		nopx	;		mov	p1, p4;		nops	
	padda	 [p1], m0;		add	 r23, r21, r0
	vst.conv.bf16.fp32	 cmh4, [p1, #64];		lshl	 r23, r23, r1
	vst.conv.bf16.fp32	 cml4, [p1, #0];		movxm	p1, ##_ZL7g_arena
	mov	m0, r23
	padda	 [p1], m0
	vldb	 x1, [p1, dj1]
	vldb	 x0, [p1, dj0];		add	r22, r19, #-1;		vmov	x1, bmhh0
	vsel.32	 x9, x6, x7, r22
	vsel.32	 x0, x2, x8, r22
	vsel.32	 x1, x5, x1, r22
	vmov	bmll3, x9
	vmov	bmhh3, x1
	vmov	bmlh3, x0
	vsel.32	 x9, x4, x3, r22;		vmac.f	dm4, dm3, y0, y5, r17
	vmov	bmhl3, x9
	nop	
.L_LEnd7:
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
// %bb.10:                              //   in Loop: Header=BB8_1 Depth=1
	nopa	;		nopb	;		add	r21, r21, #2
	eq	 r22, r21, r5
	jz	 r22, #.LBB8_1
	nop	                                //  Delay Slot 5
	mov	p1, p4                          //  Delay Slot 4
	padda	 [p1], m0                       //  Delay Slot 3
	vst.conv.bf16.fp32	 cml4, [p1, #0];		vmov	x0, x10 //  Delay Slot 2
	vst.conv.bf16.fp32	 cmh4, [p1, #64];		add	 r20, r20, r7;		vmov	x1, x11 //  Delay Slot 1
// %bb.11:                              // %for.cond.cleanup
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-192                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end8:
	.size	_ZL5gemm8PK9bfp16ebs8S1_iPKN3aie6vectorI8bfloat16Lj64EEEPS4_i, .Lfunc_end8-_ZL5gemm8PK9bfp16ebs8S1_iPKN3aie6vectorI8bfloat16Lj64EEEPS4_i
                                        // -- End function
	.section	.text._ZL9out_bytesPjPKvi,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL9out_bytesPjPKvi
	.type	_ZL9out_bytesPjPKvi,@function
_ZL9out_bytesPjPKvi:                    // @_ZL9out_bytesPjPKvi
// %bb.0:                               // %entry
	paddxm	 [sp], #64
	st	 p6, [sp, #-48];		nopx	        // 4-byte Folded Spill
	st	 p7, [sp, #-44]                 // 4-byte Folded Spill
	st	 r8, [sp, #-60]                 // 4-byte Folded Spill
	st	 r9, [sp, #-56];		mov	p6, p0  // 4-byte Folded Spill
	st	 r10, [sp, #-52];		mov	p7, p1 // 4-byte Folded Spill
	mova	r10, #1;		st	 lr, [sp, #-64];		movx	r9, #0;		mov	r8, #512 // 4-byte Folded Spill
.LBB9_1:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		lshl	 r0, r9, r10
	mov	dj0, r0
	vldb	 x0, [p7, dj0]
	jl	#_ZL5out64PjN3aie6vectorI8bfloat16Lj32EEE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mov	p0, p6                          //  Delay Slot 1
	nopa	;		nopb	;		nops	;		add	r9, r9, #32;		nopm	;		nopv	
	eq	 r0, r9, r8
	jz	 r0, #.LBB9_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.2:                               // %for.cond.cleanup
	lda	 lr, [sp, #-64]                 // 4-byte Folded Reload
	nop	
	lda	 p7, [sp, #-44]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-48]                 // 4-byte Folded Reload
	lda	 r10, [sp, #-52]                // 4-byte Folded Reload
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end9:
	.size	_ZL9out_bytesPjPKvi, .Lfunc_end9-_ZL9out_bytesPjPKvi
                                        // -- End function
	.section	.text.ew_ssm,"ax",@progbits
	.globl	ew_ssm                          // -- Begin function ew_ssm
	.p2align	4
	.type	ew_ssm,@function
ew_ssm:                                 // @ew_ssm
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	r0, #1065353216;		nopv	
	movxm	m2, #17472
	movxm	p2, ##_ZL7g_arena
	movxm	m0, #13376
	movxm	r1, #-16512
	movxm	p3, ##_ZL7g_arena
	paddxm	 [sp], #896
	st	 r11, [sp, #-100];		movxm	p4, ##_ZL7g_arena // 4-byte Folded Spill
	st	 r14, [sp, #-88];		movxm	m4, #12352 // 4-byte Folded Spill
	st	 p6, [sp, #-80];		movxm	m3, #22592 // 4-byte Folded Spill
	st	 lr, [sp, #-116];		movxm	m1, #16448 // 4-byte Folded Spill
	st	 r8, [sp, #-112];		vbcst.32	 x1, r0 // 4-byte Folded Spill
	st	 r9, [sp, #-108];		movxm	r0, #16256 // 4-byte Folded Spill
	st	 p7, [sp, #-76];		movxm	p7, ##_ZL7g_arena // 4-byte Folded Spill
	padda	 [p2], m2;		st	 r10, [sp, #-104];		movxm	m2, #10752 // 4-byte Folded Spill
	st	 r12, [sp, #-96];		vbcst.16	 x8, r0 // 4-byte Folded Spill
	padda	 [p3], m4;		st	 p2, [sp, #-148];		movxm	r0, #16177 // 4-byte Folded Spill
	st	 p3, [sp, #-192];		movxm	p2, ##_ZL7g_out_b // 4-byte Folded Spill
	padda	 [p4], m3;		st	 r13, [sp, #-92];		mov	p3, sp // 4-byte Folded Spill
	st	 p4, [sp, #-140];		movxm	p4, ##_ZL7g_arena // 4-byte Folded Spill
	st	 r15, [sp, #-84];		vbcst.16	 x6, r0 // 4-byte Folded Spill
	st	 p1, [sp, #-124];		movxm	r0, #15990 // 4-byte Folded Spill
	padda	 [p7], m0;		st	 p0, [sp, #-128];		vbcst.16	 x4, r0 // 4-byte Folded Spill
	mova	m0, #-896;		st	 p7, [sp, #-144];		movxm	r0, #15715 // 4-byte Folded Spill
	padda	 [p4], m2;		vst	 x8, [sp, #-512];		vbcst.16	 x2, r0 // 64-byte Folded Spill
	padda	 [p3], m0;		st	 p4, [sp, #-132];		movxm	r0, #-16128 // 4-byte Folded Spill
	st	 p3, [sp, #-184];		vbcst.16	 x10, r0 // 4-byte Folded Spill
	lda	 r1, [p2, #0];		vst	 x1, [p3, #64];		movxm	p2, ##_ZL7g_arena
	padda	 [p2], m1;		vst	 x10, [sp, #-320];		vbcst.16	 x10, r1 // 64-byte Folded Spill
	st	 p2, [sp, #-136]                // 4-byte Folded Spill
	vst	 x10, [sp, #-256];		mov	p2, p3 // 64-byte Folded Spill
	vst	 x1, [p2], #128
	vst	 x8, [p2], #64
	vst	 x6, [p2], #64;		movxm	m5, #11328
	vst	 x4, [p2, #0];		mov	crrnd, #12
	vst	 x2, [p2, #64];		movxm	p6, ##_ZL7g_arena
	mova	r0, #12;		paddb	 [p6], m5;		movx	r14, #1;		mov	r11, #828;		st	 r1, [sp, #-188] // 4-byte Folded Spill
.LBB10_1:                               // %for.body
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_2 Depth 2
                                        //       Child Loop BB10_3 Depth 3
                                        //       Child Loop BB10_5 Depth 3
                                        //       Child Loop BB10_7 Depth 3
                                        //       Child Loop BB10_9 Depth 3
                                        //       Child Loop BB10_11 Depth 3
	nopa	;		nopb	;		jl	#_ZL7in_needPKji
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	st	 r0, [sp, #-120];		movxm	r0, #10752 // 4-byte Folded Spill Delay Slot 2
	mov	r8, p0                          //  Delay Slot 1
	nopa	;		nopb	;		movxm	p0, ##_ZL6g_in_b;		nops	
	lda	 r0, [p0, #0]
	nop	
	nop	
	nop	
	nop	
	movxm	r1, #1033216
	movxm	r2, #15359
	add	 r1, r0, r1
	lt	 r27, r2, r0
	sel.nez	 r0, r1, r0, r27
	movs	p0, r8;		mov	dj0, r0
	lda	 r9, [p0, dj0]
	jl	#_ZL7in_copyPKjPvii
	nop	                                //  Delay Slot 5
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 4
	movxm	m0, #9216                       //  Delay Slot 3
	padda	 [p1], m0;		movxm	r1, #1536 //  Delay Slot 2
	movx	r0, #0                          //  Delay Slot 1
	jl	#_ZL7in_copyPKjPvii
	nopx	                                //  Delay Slot 5
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 4
	movxm	r0, #1536                       //  Delay Slot 3
	movxm	r1, #4608                       //  Delay Slot 2
	mov	p0, r8                          //  Delay Slot 1
	nopa	;		nopb	;		jl	#_ZL7in_copyPKjPvii
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 5
	movxm	m0, #4608                       //  Delay Slot 4
	padda	 [p1], m0;		movxm	r0, #6144 //  Delay Slot 3
	st	 p1, [sp, #-152];		movxm	r1, #4608 // 4-byte Folded Spill Delay Slot 2
	mov	p0, r8                          //  Delay Slot 1
	nopa	;		nopb	;		nops	;		jl	#_ZL7in_donePji;		nopv	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movxm	r0, #10752                      //  Delay Slot 2
	mov	p0, r8                          //  Delay Slot 1
	nopa	;		movxm	r1, #16256
	mova	r2, #0;		mov	r27, r9
	sel.eqz	 r0, r2, r1, r27
	sel.eqz	 r1, r1, r2, r27
	jl	#_ZL10ssm_tablesv
	vbcst.16	 x0, r0                         //  Delay Slot 5
	vbcst.16	 x0, r1                         //  Delay Slot 4
	vst	 x0, [sp, #-448]                // 64-byte Folded Spill Delay Slot 3
	vst	 x0, [sp, #-384]                // 64-byte Folded Spill Delay Slot 2
	nop	                                //  Delay Slot 1
	mova	r13, #8;		nopb	;		nops	;		movx	r9, #18;		nopm	;		nopv	
.LBB10_2:                               // %for.body7
                                        //   Parent Loop BB10_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_3 Depth 3
                                        //       Child Loop BB10_5 Depth 3
                                        //       Child Loop BB10_7 Depth 3
                                        //       Child Loop BB10_9 Depth 3
                                        //       Child Loop BB10_11 Depth 3
	jl	#_ZL7in_needPKji
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	st	 r9, [sp, #-180];		movxm	r0, #7168 // 4-byte Folded Spill Delay Slot 2
	movs	p7, r8;		mov	p0, r8          //  Delay Slot 1
	nopa	;		nopb	;		nops	;		jl	#_ZL7in_partPKji;		nopv	
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r0, #0;		mov	p1, r8          //  Delay Slot 1
	nopa	;		nopb	;		jl	#_ZL7in_partPKji
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	st	 p0, [sp, #-172];		movxm	r0, #1024 // 4-byte Folded Spill Delay Slot 2
	mov	p1, r8                          //  Delay Slot 1
	nopa	;		nopb	;		jl	#_ZL7in_partPKji;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	movxm	r0, #2048                       //  Delay Slot 3
	mov	r9, p0                          //  Delay Slot 2
	mov	p1, r8                          //  Delay Slot 1
	nopa	;		nopb	;		jl	#_ZL7in_partPKji;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	movxm	r0, #3072                       //  Delay Slot 3
	mov	r12, p0                         //  Delay Slot 2
	mov	p1, r8                          //  Delay Slot 1
	nopa	;		nopb	;		jl	#_ZL7in_partPKji
	nop	                                //  Delay Slot 5
	movxm	r0, #4096                       //  Delay Slot 4
	mov	r15, p6                         //  Delay Slot 3
	mov	r8, p0                          //  Delay Slot 2
	mov	p1, p7                          //  Delay Slot 1
	nopa	;		nopb	;		jl	#_ZL7in_partPKji;		nops	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	movxm	r0, #5120                       //  Delay Slot 3
	mov	r10, p0                         //  Delay Slot 2
	mov	p1, p7                          //  Delay Slot 1
	nopa	;		nopb	;		jl	#_ZL7in_partPKji
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	st	 p0, [sp, #-168];		movxm	r0, #6144 // 4-byte Folded Spill Delay Slot 2
	mov	p1, p7                          //  Delay Slot 1
	lda	 r19, [sp, #-148];		nopb	;		movx	r0, #32;		add.nc	lc, r13, #0 // 4-byte Folded Reload
	mova	r1, #8;		movxm	ls, #.LBB10_3
	mova	r7, #8;		movxm	le, #.L_LEnd12
	mova	r16, #-512;		movxm	r6, #1024
	mova	r17, #512;		movxm	dj1, #19520
	vlda	 x1, [sp, #-512];		st	 p0, [sp, #-164];		movx	r18, #64;		mov	r13, #6 // 64-byte Folded Reload4-byte Folded Spill
.LBB10_3:                               // %for.cond17.preheader
                                        //   Parent Loop BB10_1 Depth=1
                                        //     Parent Loop BB10_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		lshl	 r2, r1, r13;		nopm	;		nops	
	lshl	 r4, r0, r14;		mov	p1, r19
	add	 r3, r2, r16;		mov	p0, r10
	add	r3, r4, #-64;		mov	m0, r3
	paddb	 [p1], m0;		mov	dj0, r3
	vlda	 x5, [p1, #0];		vldb	 x3, [p0, dj0];		mov	p2, r19
	mov	m1, r2
	padda	 [p2], m1;		mov	p7, r8
	vlda	 x7, [p7, dj0];		vldb	 x9, [p2, #0];		add	 r3, r2, r17
	movs	m1, r3;		mov	p0, r19
	paddb	 [p0], m1;		movs	p4, r12
	vlda	 x11, [p4, dj0];		vldb	 x0, [p0, #0];		add	 r5, r2, r6;		mov	p6, r9
	movs	p3, r19;		mov	m1, r5;		vmul.f	dm0, x3, x5, r11
	padda	 [p3], m1;		vldb	 x2, [p6, dj0]
	vldb	 x4, [p3, #0]
	vmul.f	dm1, x7, x9, r11
	nop	
	nop	
	vconv.bf16.fp32	 x6, cml0;		vmul.f	dm2, x11, x0, r11
	nop	
	vmul.f	dm3, x6, x1, r11
	vconv.bf16.fp32	 x8, cml1;		vmul.f	dm4, x2, x4, r11
	nop	
	vmac.f	dm3, dm3, x8, x1, r11
	vconv.bf16.fp32	 x10, cml2
	nop	
	vmac.f	dm3, dm3, x10, x1, r11
	vconv.bf16.fp32	 x3, cml4
	nop	
	vmac.f	dm3, dm3, x3, x1, r11
	nop	
	movxm	p4, ##_ZL7g_arena
	padda	 [p4], m0
	vlda	 x5, [p4, dj1]
	nop	
	vconv.bf16.fp32	 x7, cml3
	nop	
	vmul.f	dm0, x7, x1, r11
	nop	
	nop	
	vmac.f	dm0, dm0, x5, x1, r11
	nop	
	nop	
	nop	
	nop	
	mov	p5, r15
	vst.conv.bf16.fp32	 cml0, [p5, dj0];		mov	dj0, r4
	vldb	 x9, [p1, #0];		mov	p1, r10
	vldb	 x11, [p1, dj0]
	nop	
	nop	
	vlda	 x0, [p2, #0];		vldb	 x2, [p7, dj0]
	nop	
	vlda	 x4, [p0, #0];		mov	p0, r12
	vldb	 x6, [p0, dj0]
	vlda	 x8, [p3, #0];		vmul.f	dm1, x11, x9, r11
	nop	
	vlda	 x10, [p6, dj0]
	vmul.f	dm2, x2, x0, r11
	nop	
	nop	
	vconv.bf16.fp32	 x3, cml1;		vmul.f	dm4, x6, x4, r11
	nop	
	vmul.f	dm3, x3, x1, r11
	vconv.bf16.fp32	 x5, cml2;		vmul.f	dm0, x10, x8, r11
	nop	
	vmac.f	dm3, dm3, x5, x1, r11
	vconv.bf16.fp32	 x7, cml4
	nop	
	vmac.f	dm3, dm3, x7, x1, r11
	vconv.bf16.fp32	 x9, cml0
	nop	
	vmac.f	dm3, dm3, x9, x1, r11
	nop	
	nop	
	nop	
	vlda	 x11, [p4, dj1]
	nop	
	vconv.bf16.fp32	 x0, cml3
	nop	
	vmul.f	dm1, x0, x1, r11
	nop	
	nop	
	vmac.f	dm1, dm1, x11, x1, r11
	nop	
	nop	
	nop	
	nop	
	add	r1, r1, #1
.L_LEnd12:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml1, [p5, dj0];		add	 r0, r0, r18;		nopm	;		nopv	
// %bb.4:                               // %for.body73.preheader
                                        //   in Loop: Header=BB10_2 Depth=2
	mova	r1, #0;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		lshl	 r0, r1, r14
	lda	 p2, [sp, #-132];		movs	p0, r15;		mov	m0, r0 // 4-byte Folded Reload
	padda	 [p0], m0
	vlda.conv.fp32.bf16	 cml1, [p0, #0]
	vlda.conv.fp32.bf16	 cmh1, [p0, #64]
	nop	
	nop	
	add.nc	lc, r7, #-1
	mova	r26, #0;		st	 r12, [sp, #-160];		movxm	ls, #.LBB10_5 // 4-byte Folded Spill
	mova	r12, #2;		nopb	;		st	 r8, [sp, #-176];		movxm	le, #.L_LEnd11;		nopv	 // 4-byte Folded Spill
	lda	 r9, [sp, #-144];		nopb	;		st	 r9, [sp, #-156];		add	 r0, r1, r18;		mov	r8, #-1;		nopv	 // 4-byte Folded Reload4-byte Folded Spill
.LBB10_5:                               // %for.body73
                                        //   Parent Loop BB10_1 Depth=1
                                        //     Parent Loop BB10_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		nopxm	;		nopv	
	nopa	;		nopb	;		vst.flush.512.conv	 [p2, sf, r26];		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		lshl	 r1, r0, r14;		nopm	;		nopv	
	movs	p0, r15;		nopx	;		mov	m0, r1
	padda	 [p0], m0
	vlda.conv.fp32.bf16	 cmh1, [p0, #64]
	vlda.conv.fp32.bf16	 cml1, [p0, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
.L_LEnd11:
	nopa	;		nopb	;		nops	;		add	 r0, r0, r18;		nopm	;		nopv	
// %bb.6:                               //   in Loop: Header=BB10_2 Depth=2
	lda	 p2, [sp, #-192];		nopxm	 // 4-byte Folded Reload
	vst.push.576.conv.bfp16ebs8.fp32	 dm1, [p2, sf, r26];		jl	#_ZL9gate_gemmPK9bfp16ebs8S1_PKN3aie6vectorI8bfloat16Lj64EEEPS4_
	vst.flush.512.conv	 [p2, sf, r26]  //  Delay Slot 5
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 4
	movxm	m0, #20544                      //  Delay Slot 3
	padda	 [p1], m0;		movxm	p0, ##_ZL7g_arena //  Delay Slot 2
	nop	                                //  Delay Slot 1
	lda	 p0, [sp, #-152]                // 4-byte Folded Reload
	jl	#_ZL9gate_gemmPK9bfp16ebs8S1_PKN3aie6vectorI8bfloat16Lj64EEEPS4_
	nop	                                //  Delay Slot 5
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 4
	movxm	m0, #21568                      //  Delay Slot 3
	padda	 [p1], m0                       //  Delay Slot 2
	movs	p2, r9                          //  Delay Slot 1
	lda	 p0, [sp, #-184];		nopb	;		nops	;		nopxm	;		nopv	 // 4-byte Folded Reload
	jl	#_ZN3nk8L9sigmoid_nEP8bfloat16iRKNS_4ExpCE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
	mova	r1, #0;		nopb	;		movx	r0, #16
	lshl	 r0, r1, r14;		add.nc	lc, r0, #-1
	movs	p0, r9;		mov	m0, r0
	padda	 [p0], m0
	vlda	 x4, [sp, #-320];		vldb	 x0, [p0, #0] // 64-byte Folded Reload
	nop	
	nop	
	movxm	p1, ##_ZL7g_arena
	lshl	 r0, r1, r8
	lshl	 r0, r0, r13
	mov	m0, r0
	padda	 [p1], m0;		movxm	dj0, #20032;		vmul.f	dm0, x0, x4, r11
	vldb	 x0, [p1, dj0]
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml0
	nop	
	vmul.f	dm3, x2, x0, r11
	nop	
	movxm	ls, #.LBB10_7
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd10;		nopv	
	nopa	;		nopb	;		add	r0, r1, #1;		nopm	
	add	r1, r1, #32
.LBB10_7:                               // %for.body86
                                        //   Parent Loop BB10_1 Depth=1
                                        //     Parent Loop BB10_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml3, [p0, #0];		lshl	 r2, r1, r14;		nopm	;		nopv	
	movs	p0, r9;		nopx	;		mov	m0, r2
	padda	 [p0], m0
	vldb	 x8, [p0, #0]
	nop	
	nop	
	movxm	p1, ##_ZL7g_arena
	lshl	 r2, r0, r8
	lshl	 r2, r2, r13
	mov	m0, r2
	padda	 [p1], m0;		vmul.f	dm2, x8, x4, r11
	vldb	 x10, [p1, dj0]
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x1, cml2
	nop	
	vmul.f	dm3, x1, x10, r11
	nop	
	nop	
	nop	
	add	r1, r1, #32
.L_LEnd10:
	nopa	;		nopb	;		nops	;		add	r0, r0, #1;		nopm	;		nopv	
// %bb.8:                               //   in Loop: Header=BB10_2 Depth=2
	lda	 p0, [sp, #-184];		nopb	;		nopxm	;		nops	 // 4-byte Folded Reload
	jl	#_ZN3nk8L8exp_a2_nEP8bfloat16S1_iRKNS_4ExpCE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml3, [p0, #0] //  Delay Slot 2
	nop	                                //  Delay Slot 1
	jl	#_ZN3nk8L6sqrt_nEP8bfloat16iRKNS_4ExpCE
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
	vlda	 x8, [sp, #-512];		nopb	;		nopx	 // 64-byte Folded Reload
	vlda	 x10, [sp, #-256]               // 64-byte Folded Reload
	lda	 p3, [sp, #-140]                // 4-byte Folded Reload
	lda	 p4, [sp, #-136]                // 4-byte Folded Reload
	lda	 p2, [sp, #-192];		movx	r0, #0 // 4-byte Folded Reload
	lda	 p5, [sp, #-168];		movx	r1, #0;		mov	m2, #960 // 4-byte Folded Reload
	vlda	 x1, [sp, #-448];		movxm	ls, #.LBB10_9 // 64-byte Folded Reload
	vlda	 x3, [sp, #-384];		movxm	le, #.L_LEnd9 // 64-byte Folded Reload
	mova	r4, #8;		movs	p7, r15;		movxm	m1, #14400
	lda	 r6, [sp, #-164];		movs	p6, r9;		movx	r5, #64;		add.nc	lc, r4, #0 // 4-byte Folded Reload
.LBB10_9:                               // %for.cond108.preheader
                                        //   Parent Loop BB10_1 Depth=1
                                        //     Parent Loop BB10_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		lshl	 r2, r0, r14;		nopm	;		nops	
	add	 r3, r2, r5;		mov	dj0, r2
	vldb	 x5, [p7, dj0];		lshl	 r2, r1, r12;		mov	p0, p5
	vlda	 x7, [p2, dj0];		mov	m0, r2
	padda	 [p0], m0
	vlda	 bmll2, [p0, #0]
	vlda	 bmlh2, [p0, #64]
	nop	
	nop	
	nop	
	vmul.f	dm3, x5, x7, r11
	vldb	 x9, [p3, dj0]
	nop	
	vlda	 x0, [p6, dj0];		vconv.bf16.fp32	 x11, cml2
	nop	
	vmac.f	dm4, dm2, x11, x10, r11
	vconv.bf16.fp32	 x2, cml3
	nop	
	vmul.f	dm0, x2, x9, r11
	nop	
	vmul.f	dm1, x0, x11, r11
	vconv.bf16.fp32	 x4, cml4
	nop	
	vmac.f	dm1, dm1, x0, x4, r11
	vconv.bf16.fp32	 x6, cml0
	nop	
	vmac.f	dm1, dm1, x6, x8, r11
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x5, cml1
	nop	
	vmac.f	dm2, dm1, x5, x10, r11
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm3, x5, x3, r11
	vconv.bf16.fp32	 x7, cml2
	nop	
	vmac.f	dm3, dm3, x7, x3, r11
	nop	
	nop	
	vmac.f	dm3, dm3, x2, x1, r11
	nop	
	nop	
	movxm	p0, ##_ZL7g_arena
	padda	 [p0], m0
	paddb	 [p0], m1
	vst	 bmll3, [p0], #64
	vst	 bmlh3, [p0], m2
	vst.conv.bf16.fp32	 cml3, [p4, dj0];		mov	dj0, r3
	vldb	 x9, [p7, dj0]
	vldb	 x11, [p2, dj0];		mov	p1, r6
	paddb	 [p1], m0
	vlda	 bmll4, [p1, #0]
	vlda	 bmlh4, [p1, #64]
	nop	
	nop	
	nop	
	vmul.f	dm0, x9, x11, r11
	vldb	 x0, [p3, dj0]
	nop	
	vldb	 x4, [p6, dj0];		vconv.bf16.fp32	 x2, cml4
	nop	
	vmac.f	dm1, dm4, x2, x10, r11
	vconv.bf16.fp32	 x6, cml0
	nop	
	vmul.f	dm2, x6, x0, r11
	nop	
	vmul.f	dm3, x4, x2, r11
	vconv.bf16.fp32	 x5, cml1
	nop	
	vmac.f	dm3, dm3, x4, x5, r11
	vconv.bf16.fp32	 x7, cml2
	nop	
	vmac.f	dm3, dm3, x7, x8, r11
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cml3
	nop	
	vmac.f	dm4, dm3, x9, x10, r11
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm0, x9, x3, r11
	vconv.bf16.fp32	 x11, cml4
	nop	
	vmac.f	dm0, dm0, x11, x3, r11
	nop	
	nop	
	vmac.f	dm0, dm0, x6, x1, r11
	nop	
	nop	
	nop	
	nop	
	nop	
	vst	 bmll0, [p0, #0]
	vst	 bmlh0, [p0, #64];		add	 r0, r0, r5
.L_LEnd9:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml0, [p4, dj0];		add	r1, r1, #32;		nopm	;		nopv	
// %bb.10:                              // %for.body163.preheader
                                        //   in Loop: Header=BB10_2 Depth=2
	lda	 p3, [sp, #-172];		nopb	;		nopxm	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	mova	r1, #0
	lshl	 r0, r1, r14
	movs	m0, r0;		mov	p0, p3
	padda	 [p0], m0
	vldb	 x0, [p0, #0]
	vldb	 x1, [p0, #64];		mov	p0, p4
	padda	 [p0], m0
	vldb	 x2, [p0, #0]
	vldb	 x3, [p0, #64]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm0, y1, y0, r11
	nop	
	nop	
	nop	
	add.nc	lc, r4, #-1
	movxm	ls, #.LBB10_11
	vconv.bf16.fp32	 x2, cml0;		movxm	le, #.L_LEnd8
	mova	r26, #0;		vconv.bf16.fp32	 x8, cmh0;		movxm	p2, ##_ZL8g_panels
	lda	 p1, [sp, #-156];		movs	p6, r15;		add	 r0, r1, r5;		vconv.fp32.bf16	cml3, x2 // 4-byte Folded Reload
.LBB10_11:                              // %for.body163
                                        //   Parent Loop BB10_1 Depth=1
                                        //     Parent Loop BB10_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		nops	;		nopx	;		vconv.fp32.bf16	cmh0, x8;		nopv	
	lshl	 r1, r0, r14;		vmov	cml0, cml3
	mov	m0, r1
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26];		mov	p0, p4
	padda	 [p0], m0;		vst.flush.512.conv	 [p2, sf, r26]
	vldb	 x2, [p0, #0]
	vldb	 x3, [p0, #64];		mov	p0, p3
	padda	 [p0], m0
	vldb	 x1, [p0, #64]
	vldb	 x0, [p0, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm4, y1, y0, r11
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x10, cml4
	vconv.bf16.fp32	 x8, cmh4
.L_LEnd8:
	nopa	;		nopb	;		nops	;		add	 r0, r0, r5;		vconv.fp32.bf16	cml3, x10;		nopv	
// %bb.12:                              //   in Loop: Header=BB10_2 Depth=2
	nopa	;		nopb	;		nops	;		nopx	;		vconv.fp32.bf16	cmh0, x8;		nopv	
	nopx	;		vmov	cml0, cml3
	nop	
	vst.push.576.conv.bfp16ebs8.fp32	 dm0, [p2, sf, r26]
	vst.flush.512.conv	 [p2, sf, r26]
	nop	
	nop	
	nop	
	nop	
	nop	
	lda	 r9, [sp, #-180];		movx	r2, #-8 // 4-byte Folded Reload
	lda	 r8, [sp, #-124];		acq	#50, r2 // 4-byte Folded Reload
	nop	
	jl	#_ZL8out_copyPjiPK8bfloat16i
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r1, #16;		movx	r0, #0;		mov	p0, r8 //  Delay Slot 1
	lda	 p1, [sp, #-160];		nopx	 // 4-byte Folded Reload
	jl	#_ZL8out_copyPjiPK8bfloat16i
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r1, #16;		movs	p0, r8;		movxm	r0, #1024 //  Delay Slot 1
	lda	 p1, [sp, #-176];		nopx	 // 4-byte Folded Reload
	jl	#_ZL8out_copyPjiPK8bfloat16i
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r1, #16;		movs	p0, r8;		movxm	r0, #2048 //  Delay Slot 1
	nopa	;		nopb	;		nops	;		jl	#_ZL8out_copyPjiPK8bfloat16i;		nopv	
	nopx	                                //  Delay Slot 5
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 4
	movxm	m0, #14400                      //  Delay Slot 3
	padda	 [p1], m0;		movxm	r0, #3072 //  Delay Slot 2
	mova	r1, #32;		mov	p0, r8          //  Delay Slot 1
	nopa	;		jl	#_ZL8out_copyPjiPK8bfloat16i
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	movxm	r0, #5120                       //  Delay Slot 3
	movxm	p1, ##_ZL8g_panels              //  Delay Slot 2
	mova	r1, #16;		mov	p0, r8          //  Delay Slot 1
	lda	 r2, [sp, #-188]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	movxm	r0, #3071
	mova	r13, #8;		movxm	r1, #-3072
	rel	#51, r13
	lda	 p0, [sp, #-128];		lt	 r27, r0, r2 // 4-byte Folded Reload
	movxm	r0, #6144
	jl	#_ZL7in_donePji
	sel.nez	 r0, r1, r0, r27                //  Delay Slot 5
	add	 r2, r0, r2                     //  Delay Slot 4
	st	 r2, [sp, #-188];		movxm	p0, ##_ZL7g_out_b // 4-byte Folded Spill Delay Slot 3
	st	 r2, [p0, #0];		movxm	r0, #7168 //  Delay Slot 2
	mov	r8, p0                          //  Delay Slot 1
	nopa	;		nopb	;		add	r9, r9, #-1;		nopm	
	jnz	 r9, #.LBB10_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.13:                              // %for.cond.cleanup6
                                        //   in Loop: Header=BB10_1 Depth=1
	lda	 r0, [sp, #-120]                // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	add	r0, r0, #-1
	jnz	 r0, #.LBB10_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mov	p0, r8                          //  Delay Slot 1
// %bb.14:                              // %for.cond.cleanup
	lda	 p7, [sp, #-76]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-80]                 // 4-byte Folded Reload
	lda	 r15, [sp, #-84]                // 4-byte Folded Reload
	lda	 r14, [sp, #-88]                // 4-byte Folded Reload
	lda	 lr, [sp, #-116]                // 4-byte Folded Reload
	lda	 r13, [sp, #-92]                // 4-byte Folded Reload
	lda	 r12, [sp, #-96]                // 4-byte Folded Reload
	lda	 r11, [sp, #-100]               // 4-byte Folded Reload
	lda	 r10, [sp, #-104]               // 4-byte Folded Reload
	lda	 r9, [sp, #-108]                // 4-byte Folded Reload
	lda	 r8, [sp, #-112]                // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-896                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end10:
	.size	ew_ssm, .Lfunc_end10-ew_ssm
                                        // -- End function
	.section	.text._ZL10ssm_tablesv,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL10ssm_tablesv
	.type	_ZL10ssm_tablesv,@function
_ZL10ssm_tablesv:                       // @_ZL10ssm_tablesv
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	m0, #9472
	mova	r0, #8;		movxm	m1, #19520
	mova	r1, #256;		movxm	m2, #9344
	mova	r2, #0;		j	#.LBB11_4
	mova	r3, #16;		movxm	m3, #20032      //  Delay Slot 5
	mova	r4, #64;		movxm	dj0, #9600      //  Delay Slot 4
	mova	r5, #1;		paddxm	 [sp], #64      //  Delay Slot 3
	mova	r6, #6;		st	 p6, [sp, #-60];		movxm	dj1, #17472 // 4-byte Folded Spill Delay Slot 2
	mova	r7, #0;		st	 lr, [sp, #-64];		movxm	p6, ##_ZL7g_arena // 4-byte Folded Spill Delay Slot 1
.LBB11_1:                               // %for.body4
                                        //   in Loop: Header=BB11_2 Depth=2
	nopa	;		nopb	;		lshl	 r19, r18, r5;		nopm	;		nops	
	movs	p1, p0;		mov	m4, r19
	padda	 [p1], m4
	vldb.128	 wl0, [p1, dj0]
	nop	
	nop	
	nop	
	nop	
	nop	
	movxm	p1, ##_ZL7g_arena
	vshift	x2, x0, x0, r3
	vshift	x2, x2, x0, r3
	lshl	 r19, r17, r6;		vshift	x2, x2, x0, r3
	movs	m4, r19;		vshift	x0, x2, x0, r3
	padda	 [p1], m4;		add	 r18, r18, r4
	vst	 x0, [p1, dj1];		add	r17, r17, #8
.LBB11_2:                               // %for.cond1
                                        //   Parent Loop BB11_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		ne	 r19, r18, r1;		nopm	;		nops	
	jnz	 r19, #.LBB11_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %for.cond.cleanup3
                                        //   in Loop: Header=BB11_4 Depth=1
	movxm	p0, ##_ZL7g_arena
	padda	 [p0], m0;		mov	dj2, r16
	vldb.128	 wl0, [p0, dj2]
	nop	
	nop	
	nop	
	nop	
	movxm	p1, ##_ZL7g_arena
	movxm	p0, ##_ZL7g_arena
	vshift	x2, x0, x0, r3
	vshift	x2, x2, x0, r3
	vshift	x2, x2, x0, r3
	lshl	 r16, r7, r6;		vshift	x0, x2, x0, r3
	padda	 [p0], m1;		mov	dj3, r16
	padda	 [p1], m2;		vst	 x0, [p0, dj3]
	vldb.128	 wl0, [p1, dj2]
	nop	
	nop	
	nop	
	nop	
	nop	
	movxm	p0, ##_ZL7g_arena
	vshift	x2, x0, x0, r3
	vshift	x2, x2, x0, r3
	vshift	x2, x2, x0, r3
	vshift	x0, x2, x0, r3
	padda	 [p0], m3;		add	r7, r7, #1
	vst	 x0, [p0, dj3];		add	r2, r2, #8
.LBB11_4:                               // %for.cond
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_2 Depth 2
	nopa	;		nopb	;		nops	;		eq	 r16, r7, r0;		nopm	;		nopv	
	jnz	 r16, #.LBB11_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.5:                               // %for.cond1.preheader
                                        //   in Loop: Header=BB11_4 Depth=1
	j	#.LBB11_2
	nop	                                //  Delay Slot 5
	lshl	 r16, r2, r5                    //  Delay Slot 4
	movxm	p0, ##_ZL7g_arena               //  Delay Slot 3
	mov	m4, r16                         //  Delay Slot 2
	padda	 [p0], m4;		movx	r18, #0;		mov	r17, r7 //  Delay Slot 1
.LBB11_6:                               // %for.cond.cleanup
	nopa	;		nopb	;		nops	;		movxm	m0, #10112;		nopv	
	jl	#_ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE
	movxm	p0, ##_ZL7g_arena               //  Delay Slot 5
	movxm	m1, #20544                      //  Delay Slot 4
	movxm	p1, ##_ZL7g_arena               //  Delay Slot 3
	padda	 [p1], m1;		paddb	 [p0], m0 //  Delay Slot 2
	nop	                                //  Delay Slot 1
	lda	 lr, [sp, #-64]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-60]                 // 4-byte Folded Reload
	j	#_ZL8bias_tabPK8bfloat16PN3aie6vectorIS_Lj64EEE
	movxm	m0, #10240                      //  Delay Slot 5
	movxm	p0, ##_ZL7g_arena               //  Delay Slot 4
	movxm	m1, #21568                      //  Delay Slot 3
	padds	 [p0], m0;		paddb	 [p6], m1;		paddxm	 [sp], #-64 //  Delay Slot 2
	movs	p1, p6                          //  Delay Slot 1
.Lfunc_end11:
	.size	_ZL10ssm_tablesv, .Lfunc_end11-_ZL10ssm_tablesv
                                        // -- End function
	.section	.text._ZL7in_partPKji,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL7in_partPKji
	.type	_ZL7in_partPKji,@function
_ZL7in_partPKji:                        // @_ZL7in_partPKji
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	p0, ##_ZL6g_in_b;		nops	
	lda	 r1, [p0, #0]
	nop	
	nop	
	nop	
	nop	
	movxm	r2, #15359
	movxm	r3, #-15360
	add	 r0, r1, r0
	add	 r1, r0, r3
	lt	 r27, r2, r0
	sel.nez	 r0, r1, r0, r27
	movxm	r1, #14337
	lt	 r1, r0, r1
	jnz	 r1, #.LBB12_7
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movs	p0, p1;		mov	m0, r0          //  Delay Slot 2
	padda	 [p0], m0                       //  Delay Slot 1
// %bb.1:                               // %if.end4
	movxm	r1, #15360
	movxm	r2, #15297
	movxm	p2, ##_ZL6g_ptmp
	geu	 r5, r0, r2
	jnz	 r5, #.LBB12_4
	mova	r3, #-31;		sub	 r1, r1, r0 //  Delay Slot 5
	mova	r4, #-26;		ashl	 r3, r1, r3 //  Delay Slot 4
	lshl	 r3, r3, r4                     //  Delay Slot 3
	mova	r4, #-6;		add	 r3, r1, r3;		mov	r2, #5 //  Delay Slot 2
	mova	r1, #16;		ashl	 r3, r3, r4;		mov	r0, #0 //  Delay Slot 1
// %bb.2:                               // %for.body.preheader
	movxm	ls, #.LBB12_3
	mova	r4, #1;		movxm	le, #.L_LEnd14
	lt	 r27, r4, r3
	sel.nez	 r4, r3, r4, r27
	mova	r4, #1;		nopb	;		nops	;		movx	r5, #0;		add.nc	lc, r4, #0;		nopv	
.LBB12_3:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		lshl	 r6, r5, r4;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		nopx	;		mov	dj0, r6;		nopv	
	nopa	;		vldb	 x2, [p0, dj0];		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopxm	
	nop	
	nop	
	nop	
.L_LEnd14:
	nopa	;		nopb	;		vst	 x2, [p2, dj0];		add	r5, r5, #32;		nopm	;		nopv	
.LBB12_4:                               // %for.body16.preheader
	sub	 r4, r1, r3
	movxm	ls, #.LBB12_5
	movxm	le, #.L_LEnd13
	mova	r2, #1;		nopb	;		nops	;		lshl	 r1, r3, r2;		add.nc	lc, r4, #0;		nopv	
.LBB12_5:                               // %for.body16
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		lshl	 r3, r0, r2;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		nopx	;		mov	dj0, r3;		nopv	
	nopa	;		vldb	 x2, [p1, dj0];		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
	nopx	
	add	 r3, r1, r0
	lshl	 r3, r3, r2
	mov	dj0, r3
.L_LEnd13:
	nopa	;		nopb	;		vst	 x2, [p2, dj0];		add	r0, r0, #32;		nopm	;		nopv	
// %bb.6:
	nopa	;		nopb	;		nops	;		movxm	p0, ##_ZL6g_ptmp;		nopv	
.LBB12_7:                               // %cleanup
	nopa	;		ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end12:
	.size	_ZL7in_partPKji, .Lfunc_end12-_ZL7in_partPKji
                                        // -- End function
	.section	.text._ZL9gate_gemmPK9bfp16ebs8S1_PKN3aie6vectorI8bfloat16Lj64EEEPS4_,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL9gate_gemmPK9bfp16ebs8S1_PKN3aie6vectorI8bfloat16Lj64EEEPS4_
	.type	_ZL9gate_gemmPK9bfp16ebs8S1_PKN3aie6vectorI8bfloat16Lj64EEEPS4_,@function
_ZL9gate_gemmPK9bfp16ebs8S1_PKN3aie6vectorI8bfloat16Lj64EEEPS4_: // @_ZL9gate_gemmPK9bfp16ebs8S1_PKN3aie6vectorI8bfloat16Lj64EEEPS4_
// %bb.0:                               // %entry
	nopa	;		nopb	;		movs	p3, p1;		movxm	p1, ##_ZL7g_arena;		nopv	
	movxm	m0, #10752
	padda	 [p1], m0;		movx	r25, #0
	vldb.fill.512	 [p1, lf1, r25]
	vldb.pop.576	 ex0, [p1, lf1, r25]
	nop	
	nop	
	vldb.pop.576	 ex0, [p1, lf1, r25]
	nop	
	nop	
	movxm	r1, #16256
	paddxm	 [sp], #256
	mova	r0, #780;		vldb.pop.576	 ex0, [p1, lf1, r25];		movxm	r2, #1152;		vst	 x0, [sp, #-256] // 64-byte Folded Spill
	mova	r3, #128;		vldb.pop.576	 ex11, [p1, lf1, r25];		movxm	ls, #.LBB13_1;		st	 el0, [sp, #-192] // 4-byte Folded Spill
	mova	r4, #1;		vldb.pop.576	 ex2, [p1, lf1, r25];		movxm	le, #.L_LEnd15;		st	 eh0, [sp, #-188] // 4-byte Folded Spill
	mova	r6, #4;		vldb.pop.576	 ex10, [p1, lf1, r25];		movx	r5, #7;		vbcst.16	 x4, r1;		vst	 x0, [sp, #-128] // 64-byte Folded Spill
	mova	r24, #0;		vldb.pop.576	 ex1, [p1, lf1, r25];		movx	r16, #0;		add.nc	lc, r6, #0;		st	 el0, [sp, #-64] // 4-byte Folded Spill
	mova	r1, #828;		vldb.pop.576	 ex3, [p1, lf1, r25];		movx	r17, #0;		vmov	x5, x4;		st	 eh0, [sp, #-60] // 4-byte Folded Spill
.LBB13_1:                               // %if.end.7
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopx	;		mov	m0, r24
	vlda	 x8, [sp, #-256];		mov	p1, p0 // 64-byte Folded Reload
	lda	 el8, [sp, #-192];		padds	 [p1], m0;		mov	r25, #0 // 4-byte Folded Reload
	lda	 eh8, [sp, #-188];		vldb.fill.512	 [p1, lf1, r25] // 4-byte Folded Reload
	vlda.pop.576	 ex9, [p1, lf1, r25]
	vldb.pop.576	 ex6, [p1, lf1, r25]
	vlda.pop.576	 ex7, [p1, lf1, r25]
	nop	
	vlda	 x8, [sp, #-128]                // 64-byte Folded Reload
	lda	 el8, [sp, #-64]                // 4-byte Folded Reload
	lda	 eh8, [sp, #-60]                // 4-byte Folded Reload
	vldb.pop.576	 ex9, [p1, lf1, r25]
	vlda.pop.576	 ex8, [p1, lf1, r25]
	vmul.f	dm0, ex8, ex9, r0
	vldb.pop.576	 ex6, [p1, lf1, r25];		vmul.f	dm1, ex8, ex6, r0
	vlda.pop.576	 ex7, [p1, lf1, r25]
	vldb.pop.576	 ex9, [p1, lf1, r25]
	vlda.fill.512	 [p1, lf1, r25];		vmac.f	dm2, dm0, ex8, ex7, r0
	vlda.pop.576	 ex8, [p1, lf1, r25]
	vmac.f	dm3, dm1, ex8, ex9, r0
	vldb.pop.576	 ex6, [p1, lf1, r25];		vmac.f	dm4, dm2, ex0, ex8, r0
	vldb.pop.576	 ex7, [p1, lf1, r25]
	vmac.f	dm3, dm3, ex0, ex6, r0
	vlda.pop.576	 ex9, [p1, lf1, r25];		vmac.f	dm4, dm4, ex11, ex7, r0
	vlda.pop.576	 ex8, [p1, lf1, r25]
	vmac.f	dm3, dm3, ex11, ex9, r0
	vldb.pop.576	 ex6, [p1, lf1, r25];		vmac.f	dm4, dm4, ex2, ex8, r0
	vldb.pop.576	 ex7, [p1, lf1, r25]
	vmac.f	dm3, dm3, ex2, ex6, r0
	vldb.pop.576	 ex9, [p1, lf1, r25];		vmac.f	dm4, dm4, ex10, ex7, r0
	nop	
	vmac.f	dm3, dm3, ex10, ex9, r0
	vmac.f	dm4, dm4, ex1, ex8, r0
	nop	
	lshl	 r6, r17, r5;		vmac.f	dm3, dm3, ex1, ex6, r0
	mov	m0, r6;		vmac.f	dm4, dm4, ex3, ex7, r0
	mov	p1, p3
	padds	 [p1], m0;		add	 r7, r6, r3;		vmac.f	dm3, dm3, ex3, ex9, r0
	vldb	 x6, [p1, #0];		mov	m0, r7
	vldb	 x7, [p1, #64];		mov	p1, p3
	padds	 [p1], m0
	vldb	 x8, [p1, #0];		vconv.bf16.fp32	 x6, cml4
	vldb	 x9, [p1, #64];		vconv.bf16.fp32	 x7, cmh4
	vconv.bf16.fp32	 x8, cml3
	vconv.bf16.fp32	 x9, cmh3;		vmul.f	dm0, y3, y2, r1
	nop	
	vmul.f	dm1, y4, y2, r1
	vmac.f	dm0, dm0, y3, y2, r1
	nop	
	lshl	 r6, r16, r4;		vmac.f	dm1, dm1, y4, y2, r1
	mov	m0, r6
	mov	p1, p2
	padds	 [p1], m0
	vst.conv.bf16.fp32	 cml0, [p1], #64
	vst.conv.bf16.fp32	 cmh0, [p1], #64;		add	r17, r17, #2
	vst.conv.bf16.fp32	 cml1, [p1, #0];		add	 r24, r24, r2
.L_LEnd15:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cmh1, [p1, #64];		add	 r16, r16, r3;		nopm	;		nopv	
// %bb.2:                               // %for.cond.cleanup
	nopa	;		nopb	;		nops	;		ret	lr;		nopm	;		nopv	
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-256                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end13:
	.size	_ZL9gate_gemmPK9bfp16ebs8S1_PKN3aie6vectorI8bfloat16Lj64EEEPS4_, .Lfunc_end13-_ZL9gate_gemmPK9bfp16ebs8S1_PKN3aie6vectorI8bfloat16Lj64EEEPS4_
                                        // -- End function
	.section	.text._ZN3nk8L9sigmoid_nEP8bfloat16iRKNS_4ExpCE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN3nk8L9sigmoid_nEP8bfloat16iRKNS_4ExpCE
	.type	_ZN3nk8L9sigmoid_nEP8bfloat16iRKNS_4ExpCE,@function
_ZN3nk8L9sigmoid_nEP8bfloat16iRKNS_4ExpCE: // @_ZN3nk8L9sigmoid_nEP8bfloat16iRKNS_4ExpCE
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	r0, #-16512;		nopv	
	movxm	r1, #19264
	movxm	dj0, #12352
	mov	s0, #7
	movxm	dj1, #12416
	mov	crsrsmode, #0
	movxm	ls, #.LBB14_1
	movxm	le, #.L_LEnd16
	mova	r4, #16;		mov	crupsmode, #0
	add.nc	lc, r4, #0
	vbcst.16	 x0, r0
	movxm	r0, #-15760
	vbcst.16	 x2, r0
	movxm	r0, #17008
	vmov	lfh0, x2
	vbcst.16	 x2, r0
	movxm	r0, #16313
	vbcst.16	 x6, r0
	movxm	r0, #-17620
	vmov	lfh1, x2
	vbcst.16	 x2, r0
	mova	r0, #0;		vmov	lfl0, x2
	vbcst.16	 x2, r0
	mov	s1, r0
	vmov	lfl1, x2
	vbcst.16	 x2, r1
	movxm	r1, #-13504
	vmov	sfl, x2
	vbcst.16	 x2, r1
	movxm	r1, #16256
	vmov	sfh, x2
	vbcst.16	 x2, r1
	movxm	r1, #32499
	mova	r3, #64;		movx	r2, #2;		vbcst.16	 x7, r1
	mova	r1, #828;		movx	r4, #1;		vmov	lfe, x2
.LBB14_1:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		lshl	 r5, r0, r4;		nopm	;		nops	
	movs	m0, r5;		movxm	p1, ##_ZL7g_arena
	padda	 [p1], m0
	vldb	 x2, [p1, dj0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm0, x2, x0, r1
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml0;		vmov	x4, lfh0
	vmov	x8, lfh1
	vldb	 x2, [p1, dj1];		vmax_lt.bf16	 x2, r16, x2, x4
	vmin_ge.bf16	 x2, r16, x2, x8
	nop	
	vmul.f	dm0, x2, x6, r1
	vmov	x10, lfl0
	mov	p2, p0
	paddb	 [p2], #128;		vmac.f	dm0, dm0, x2, x10, r1
	vldb	 x9, [p2], #64;		vmul.f	dm1, x2, x0, r1
	nop	
	vmov	x1, lfl1
	nop	
	vmac.f	dm0, dm0, x1, x6, r1
	vmov	x3, sfl
	vconv.bf16.fp32	 x2, cml1
	vmov	x5, sfh;		vmac.f	dm2, dm0, x9, x3, r1
	vmax_lt.bf16	 x2, r16, x2, x4
	vmin_ge.bf16	 x2, r16, x2, x8
	vmac.f	dm1, dm2, x9, x5, r1
	vmul.f	dm1, x2, x6, r1
	nop	
	nop	
	vmac.f	dm1, dm1, x2, x10, r1
	nop	
	vconv.bf16.fp32	 x4, cml1
	vmac.f	dm3, dm1, x1, x6, r1
	nop	
	nop	
	vmac.f	dm1, dm3, x9, x3, r1
	nop	
	vmac.f	dm0, dm0, x4, x0, r1
	vmac.f	dm0, dm1, x9, x5, r1
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x8, cml0
	vconv.bf16.fp32	 x2, cml0
	vmul.f	dm0, x8, x8, r1
	vmac.f	dm4, dm3, x2, x0, r1
	nop	
	vlda	 x11, [p2], #64
	vldb	 x2, [p2, #0]
	vlda	 bmlh0, [p0, #64]
	vlda	 bmll0, [p0, #0];		vconv.bf16.fp32	 x10, cml0;		vmov	x1, bmll2
	vconv.bf16.fp32	 x1, cml4;		vmov	x4, bmlh2
	vlda	 x8, [p2, #64];		vshuffle	x4, x1, x4, r2;		vmul.f	dm2, x10, x8, r1
	vmul.f	dm4, x1, x1, r1
	vmac.f	dm3, dm0, x8, x11, r1
	nop	
	nop	
	vmac.f	dm3, dm3, x10, x2, r1
	vconv.bf16.fp32	 x10, cml2
	vconv.bf16.fp32	 x10, cml4
	vmac.f	dm2, dm3, x10, x8, r1
	vmul.f	dm1, x10, x1, r1
	nop	
	vmac.f	dm3, dm0, x1, x11, r1
	vmov	x3, bmlh1
	vmov	x5, bmll1
	vshuffle	x3, x5, x3, r2;		vmac.f	dm3, dm3, x10, x2, r1
	vconv.bf16.fp32	 x1, cml1;		vups.2x	cml1, x4, s0, upssign1
	vups.2x	cml1, x3, s0, upssign1
	vmac.f	dm3, dm3, x1, x8, r1
	vsrs.2x	x2, cml1, s1, srssign1
	vsrs.2x	x8, cml1, s1, srssign1
	nop	
	vmov	x1, lfe
	vconv.bf16.fp32	 x4, cml2;		vadd.16	 x2, x2, x1
	vconv.bf16.fp32	 x10, cml3;		vadd.16	 x2, x8, x1
	vmul.f	dm1, x4, x2, r1
	vmul.f	dm1, x10, x2, r1
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x4, cml1
	vconv.bf16.fp32	 x2, cml1
	vmac.f	dm1, dm0, x4, x9, r1
	vmac.f	dm2, dm0, x2, x9, r1
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml1
	vconv.bf16.fp32	 x11, cml2
	vmac.f	dm1, dm1, x2, x0, r1
	vmac.f	dm2, dm2, x11, x0, r1
	nop	
	vsub.16	 x10, x7, x2
	vsub.16	 x1, x7, x11
	vmsc.f	dm1, dm0, x2, x10, r1
	vconv.bf16.fp32	 x4, cml1;		vmsc.f	dm2, dm0, x11, x1, r1
	vconv.bf16.fp32	 x8, cml2
	vmsc.f	dm1, dm1, x4, x10, r1
	vmsc.f	dm2, dm2, x8, x1, r1
	nop	
	nop	
	nop	
	vmul.f	dm1, x10, x9, r1
	vconv.bf16.fp32	 x3, cml1;		vmul.f	dm2, x1, x9, r1
	vconv.bf16.fp32	 x10, cml2
	vmac.f	dm1, dm1, x10, x3, r1
	vmac.f	dm2, dm2, x1, x10, r1
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cml1
	vconv.bf16.fp32	 x1, cml2
	vmac.f	dm3, dm1, x9, x0, r1
	vmac.f	dm3, dm2, x1, x0, r1
	nop	
	nop	
	nop	
	vmsc.f	dm4, dm0, x2, x9, r1
	vconv.bf16.fp32	 x3, cml3;		vmsc.f	dm0, dm0, x11, x1, r1
	vconv.bf16.fp32	 x2, cml3
	vmsc.f	dm4, dm4, x2, x3, r1
	vmsc.f	dm0, dm0, x11, x2, r1
	nop	
	vmsc.f	dm3, dm4, x4, x9, r1
	vmsc.f	dm0, dm0, x8, x1, r1
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml3
	vconv.bf16.fp32	 x9, cml0
	vmac.f	dm1, dm1, x9, x2, r1
	vmac.f	dm0, dm2, x1, x9, r1
	nop	
	nop	
	nop	
	nop	
	vst.conv.bf16.fp32	 cml1, [p1, dj0]
.L_LEnd16:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml0, [p1, dj1];		add	 r0, r0, r3;		nopm	;		nopv	
// %bb.2:                               // %for.cond.cleanup
	nopa	;		ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end14:
	.size	_ZN3nk8L9sigmoid_nEP8bfloat16iRKNS_4ExpCE, .Lfunc_end14-_ZN3nk8L9sigmoid_nEP8bfloat16iRKNS_4ExpCE
                                        // -- End function
	.section	.text._ZN3nk8L8exp_a2_nEP8bfloat16S1_iRKNS_4ExpCE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN3nk8L8exp_a2_nEP8bfloat16S1_iRKNS_4ExpCE
	.type	_ZN3nk8L8exp_a2_nEP8bfloat16S1_iRKNS_4ExpCE,@function
_ZN3nk8L8exp_a2_nEP8bfloat16S1_iRKNS_4ExpCE: // @_ZN3nk8L8exp_a2_nEP8bfloat16S1_iRKNS_4ExpCE
// %bb.0:                               // %entry
	nopa	;		movxm	r0, #-15760
	movxm	m0, #13376
	mov	s0, #7
	movxm	m1, #22592
	mov	crsrsmode, #0
	movxm	ls, #.LBB15_1
	movxm	le, #.L_LEnd17
	mov	crupsmode, #0
	mova	r4, #8;		paddxm	 [sp], #256
	add.nc	lc, r4, #0
	vbcst.16	 x0, r0
	mova	r2, #0;		movxm	r0, #17008
	mov	s1, r2
	vmov	lfh0, x0
	vbcst.16	 x0, r0
	movxm	r0, #16313
	vmov	lfh1, x0
	vbcst.16	 x0, r0
	movxm	r0, #-17620
	vbcst.16	 x6, r0
	movxm	r0, #19264
	vmov	lfl0, x0
	vbcst.16	 x8, r0
	movxm	r0, #-13504
	vbcst.16	 x10, r0
	movxm	r0, #-16512
	vbcst.16	 x1, r0
	movxm	r0, #16256
	mova	r1, #2;		vbcst.16	 x3, r0
	mova	r3, #64;		movxm	r0, #16384
	mova	r0, #828;		movx	r4, #1;		vbcst.16	 x5, r0
.LBB15_1:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		lshl	 r5, r2, r4;		nopm	
	movxm	p1, ##_ZL7g_arena
	paddb	 [p1], m0;		movs	m2, r5
	paddb	 [p1], m2
	vldb	 x7, [p1, #0]
	nop	
	nop	
	mov	p2, p0
	vldb	 x9, [p1, #64];		vmov	x0, lfh0
	vmov	x2, lfh1
	vmov	x4, lfl0
	vmax_lt.bf16	 x7, r16, x7, x0
	padda	 [p2], #128;		vmin_ge.bf16	 x7, r16, x7, x2
	vldb	 x7, [p2], #64
	vmul.f	dm0, x7, x4, r0
	vmax_lt.bf16	 x9, r16, x9, x0
	vmin_ge.bf16	 x9, r16, x9, x2
	vmac.f	dm0, dm0, x7, x6, r0
	vmul.f	dm3, x9, x4, r0
	nop	
	vmac.f	dm1, dm0, x7, x8, r0
	vmac.f	dm3, dm3, x9, x6, r0
	nop	
	vmac.f	dm2, dm1, x7, x10, r0
	vmac.f	dm2, dm3, x7, x8, r0
	nop	
	nop	
	vmac.f	dm4, dm2, x7, x10, r0
	nop	
	vconv.bf16.fp32	 x9, cml2
	nop	
	nop	
	vmac.f	dm0, dm0, x9, x1, r0
	vldb	 x11, [p2], #64;		vconv.bf16.fp32	 x7, cml4
	nop	
	vlda	 bmlh3, [p0, #64];		vmac.f	dm3, dm3, x7, x1, r0
	vlda	 bmll3, [p0, #0]
	nop	
	vconv.bf16.fp32	 x9, cml0
	nop	
	vmac.f	dm0, dm3, x9, x11, r0
	vconv.bf16.fp32	 x7, cml3
	vmul.f	dm4, x9, x9, r0
	nop	
	nop	
	nop	
	vst	 bmhl0, [sp, #-128]             // 64-byte Folded Spill
	vst	 bmhh0, [sp, #-64]              // 64-byte Folded Spill
	vlda	 bmhl3, [sp, #-128];		vldb	 x0, [p2, #0];		vconv.bf16.fp32	 x11, cml4;		vmul.f	dm3, x7, x7, r0 // 64-byte Folded Reload
	vlda	 bmhh3, [sp, #-64];		vst	 bmll0, [sp, #-256];		vmac.f	dm0, dm3, x7, x11, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmll3, [sp, #-256];		vst	 bmlh0, [sp, #-192];		vmul.f	dm1, x11, x9, r0 // 64-byte Folded Reload64-byte Folded Spill
	vlda	 bmlh3, [sp, #-192];		vldb	 x2, [p2, #64] // 64-byte Folded Reload
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml3;		vmov	x4, bmll1
	vmov	x0, bmlh1;		vmac.f	dm3, dm3, x11, x0, r0
	vconv.bf16.fp32	 x7, cml1;		vmac.f	dm0, dm0, x2, x0, r0
	vmul.f	dm2, x2, x7, r0
	vmac.f	dm1, dm3, x7, x2, r0
	vmov	x9, bmll2
	vshuffle	x0, x4, x0, r1
	vmov	x4, bmlh2
	vups.2x	cml2, x0, s0, upssign1
	vconv.bf16.fp32	 x7, cml2;		vshuffle	x4, x9, x4, r1
	vconv.bf16.fp32	 x2, cml1
	vsrs.2x	x0, cml2, s1, srssign1;		vups.2x	cml2, x4, s0, upssign1;		vmac.f	dm0, dm0, x7, x2, r0
	vmac.f	dm1, dm1, x2, x1, r0
	nop	
	vsrs.2x	x4, cml2, s1, srssign1
	vadd.16	 x0, x0, x3
	nop	
	vconv.bf16.fp32	 x2, cml0;		vmul.f	dm3, x2, x0, r0
	vconv.bf16.fp32	 x7, cml1
	vmac.f	dm0, dm0, x2, x1, r0
	vmac.f	dm1, dm3, x7, x0, r0
	nop	
	vadd.16	 x4, x4, x3
	nop	
	vmul.f	dm0, x2, x4, r0
	vconv.bf16.fp32	 x0, cml0
	vconv.bf16.fp32	 x7, cml1
	vmac.f	dm0, dm0, x0, x4, r0
	vmac.f	dm1, dm1, x7, x1, r0
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cml0
	vconv.bf16.fp32	 x0, cml1
	vmac.f	dm0, dm0, x9, x1, r0
	vmul.f	dm1, x0, x5, r0
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0;		vmul.f	dm0, x7, x7, r0
	vconv.bf16.fp32	 x0, cml1
	vst	 x7, [p1, #0];		vmul.f	dm2, x0, x5, r0
	vst	 x9, [p1, #64];		vmac.f	dm0, dm0, x7, x0, r0
	vldb	 x0, [p0, #128]
	nop	
	vlda	 bmlh0, [p0, #64]
	vlda	 bmll0, [p0, #0];		vmul.f	dm1, x9, x9, r0
	vconv.bf16.fp32	 x0, cml2
	vconv.bf16.fp32	 x2, cml0
	vmac.f	dm1, dm1, x9, x0, r0
	vmsc.f	dm0, dm0, x2, x0, r0
	nop	
	nop	
	movxm	p1, ##_ZL7g_arena
	paddb	 [p1], m1
	padda	 [p1], m2
	vst.conv.bf16.fp32	 cml0, [p1, #0]
	vldb	 x2, [p0, #128]
	nop	
	vlda	 bmlh0, [p0, #64]
	vlda	 bmll0, [p0, #0]
	nop	
	vconv.bf16.fp32	 x0, cml1
	nop	
	vmsc.f	dm0, dm0, x0, x2, r0
	nop	
	nop	
	nop	
	nop	
	nop	
.L_LEnd17:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml0, [p1, #64];		add	 r2, r2, r3;		nopm	;		nopv	
// %bb.2:                               // %for.cond.cleanup
	nopa	;		nopb	;		nops	;		ret	lr;		nopm	;		nopv	
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-256                    //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end15:
	.size	_ZN3nk8L8exp_a2_nEP8bfloat16S1_iRKNS_4ExpCE, .Lfunc_end15-_ZN3nk8L8exp_a2_nEP8bfloat16S1_iRKNS_4ExpCE
                                        // -- End function
	.section	.text._ZN3nk8L6sqrt_nEP8bfloat16iRKNS_4ExpCE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZN3nk8L6sqrt_nEP8bfloat16iRKNS_4ExpCE
	.type	_ZN3nk8L6sqrt_nEP8bfloat16iRKNS_4ExpCE,@function
_ZN3nk8L6sqrt_nEP8bfloat16iRKNS_4ExpCE: // @_ZN3nk8L6sqrt_nEP8bfloat16iRKNS_4ExpCE
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	r0, #3490
	movxm	dj0, #22592
	movxm	dj1, #22656
	mov	crsrsmode, #0
	movxm	ls, #.LBB16_1
	movxm	le, #.L_LEnd18
	mov	crupsmode, #0
	vbcst.16	 x0, r0
	mova	r4, #8;		movxm	r0, #24375
	mova	r1, #0;		add.nc	lc, r4, #0
	mov	s0, r1
	vbcst.16	 x2, r0
	movxm	r0, #16128
	vbcst.16	 x4, r0
	movxm	r0, #1069547520
	mova	r0, #1;		vbcst.32	 x6, r0
	mov	s1, r0
	vmov	bmll0, x6
	mova	r3, #64;		movx	r2, #828;		vmov	bmlh0, x6
.LBB16_1:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		lshl	 r4, r1, r0;		nopm	
	movs	m0, r4;		movxm	p0, ##_ZL7g_arena
	padda	 [p0], m0
	vldb	 x3, [p0, dj0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	vmax_lt.bf16	 x5, r16, x3, x0
	nop	
	vups.2x	cmh0, x5, s0, upssign1
	mov	r4, crrnd
	mov	crrnd, #0
	vsrs.2x	x7, cmh0, s1, srssign1
	nop	
	nop	
	nop	
	vsub.16	 x9, x2, x7
	vmul.f	dm2, x5, x4, r2
	vmul.f	dm3, x9, x9, r2
	nop	
	nop	
	nop	
	mov	crrnd, r4
	vconv.bf16.fp32	 x11, cml2
	vconv.bf16.fp32	 x6, cml3
	nop	
	vmsc.f	dm4, dm0, x11, x6, r2
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x8, cml4
	nop	
	vmul.f	dm1, x9, x8, r2
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x10, cml1
	nop	
	vmul.f	dm2, x10, x10, r2
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x1, cml2
	nop	
	vmsc.f	dm3, dm0, x11, x1, r2
	nop	
	nop	
	vlda	 x5, [p0, dj1]
	nop	
	nop	
	vconv.bf16.fp32	 x7, cml3
	nop	
	vmul.f	dm4, x10, x7, r2
	nop	
	vmax_lt.bf16	 x9, r16, x5, x0
	nop	
	vups.2x	cmh0, x9, s0, upssign1
	nop	
	vconv.bf16.fp32	 x11, cml4;		mov	crrnd, #0
	vsrs.2x	x6, cmh0, s1, srssign1
	nop	
	nop	
	nop	
	vsub.16	 x6, x2, x6
	vmul.f	dm1, x9, x4, r2
	vmul.f	dm2, x6, x6, r2
	nop	
	nop	
	nop	
	mov	crrnd, r4
	vconv.bf16.fp32	 x8, cml1
	vconv.bf16.fp32	 x10, cml2
	nop	
	vmsc.f	dm3, dm0, x8, x10, r2
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x1, cml3
	nop	
	vmul.f	dm4, x6, x1, r2
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x7, cml4
	nop	
	vmul.f	dm1, x7, x7, r2
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x9, cml1
	nop	
	vmsc.f	dm2, dm0, x8, x9, r2
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x6, cml2
	nop	
	vmul.f	dm3, x7, x6, r2
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x3, cml3
	vmul.f	dm4, x3, x11, r2
	vmul.f	dm1, x5, x3, r2
	nop	
	nop	
	nop	
	nop	
	vst.conv.bf16.fp32	 cml4, [p0, dj0]
.L_LEnd18:
	nopa	;		nopb	;		vst.conv.bf16.fp32	 cml1, [p0, dj1];		add	 r1, r1, r3;		nopm	;		nopv	
// %bb.2:                               // %for.cond.cleanup
	nopa	;		ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end16:
	.size	_ZN3nk8L6sqrt_nEP8bfloat16iRKNS_4ExpCE, .Lfunc_end16-_ZN3nk8L6sqrt_nEP8bfloat16iRKNS_4ExpCE
                                        // -- End function
	.section	.text._ZL8out_copyPjiPK8bfloat16i,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL8out_copyPjiPK8bfloat16i
	.type	_ZL8out_copyPjiPK8bfloat16i,@function
_ZL8out_copyPjiPK8bfloat16i:            // @_ZL8out_copyPjiPK8bfloat16i
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	p2, ##_ZL7g_out_b
	lda	 r2, [p2, #0]
	nop	
	nop	
	nop	
	nop	
	movxm	r3, #9215
	movxm	r4, #-9216
	add	 r0, r2, r0
	lt	 r27, r3, r0
	add	 r2, r0, r4
	sel.nez	 r0, r2, r0, r27
	movxm	r2, #9216
	mova	r3, #-6;		sub	 r2, r2, r0
	lshl	 r4, r2, r3
	ltu	 r27, r4, r1
	mova	r3, #2;		sel.nez	 r2, r4, r1, r27
	ltu	 r27, r3, r2
	sel.nez	 r3, r2, r3, r27
	mova	r5, #-1;		add	r3, r3, #-1
	lshl	 r3, r3, r5
	add	r5, r3, #1
	mova	r5, #1;		movx	r3, #0;		add.nc	lc, r5, #-1
	movs	p2, p0;		lshl	 r6, r3, r5;		mov	m0, r0
	padda	 [p2], m0;		movs	p3, p1;		mov	m0, r6
	padda	 [p3], m0
	vldb	 x8, [p3, #64];		movxm	ls, #.LBB17_1
	nopa	;		vldb	 x10, [p3, #0];		nops	;		movxm	le, #.L_LEnd20;		nopv	
	mova	r0, #64;		nopb	;		nops	;		nopx	;		mov	p3, p2;		nopv	
	padda	 [p3], m0;		nopb	;		nops	;		add	 r6, r3, r0;		nopm	;		nopv	
.LBB17_1:                               // %for.body.i
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		lshl	 r7, r6, r5;		nopm	;		nopv	
	nopa	;		nopb	;		movs	p4, p1;		nopx	;		mov	m0, r7;		nopv	
	padda	 [p4], m0;		nopb	;		nopx	
	vldb	 x8, [p4, #64];		vmov	x4, x8
	vldb	 x10, [p4, #0];		vmov	x6, x10
	vst	 x4, [p3, #64]
	vst	 x6, [p3, #0];		mov	p3, p2
.L_LEnd20:
	padda	 [p3], m0;		nopb	;		nops	;		add	 r6, r6, r0;		nopm	;		nopv	
// %bb.2:
	nopa	;		nopb	;		geu	 r4, r4, r1;		nopm	;		nops	
	jnz	 r4, #.LBB17_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst	 x10, [p3, #0]                  //  Delay Slot 2
	vst	 x8, [p3, #64]                  //  Delay Slot 1
// %bb.3:                               // %if.then6
	mova	r4, #6;		add	r1, r1, #-1
	lshl	 r4, r2, r4
	mova	r2, #-1;		sub	 r1, r1, r2
	lshl	 r1, r1, r2
	add	r1, r1, #1
	mova	r1, #1;		movs	m0, r4;		add.nc	lc, r1, #-1
	padda	 [p1], m0;		lshl	 r2, r3, r1
	movs	p2, p1;		mov	m0, r2
	padda	 [p2], m0
	vldb	 x8, [p2, #64];		movxm	ls, #.LBB17_4
	nopa	;		vldb	 x10, [p2, #0];		nops	;		movxm	le, #.L_LEnd19;		nopv	
	nopa	;		nopb	;		nops	;		nopx	;		mov	p2, p0;		nopv	
	padda	 [p2], m0;		nopb	;		nops	;		add	 r2, r3, r0;		nopm	;		nopv	
.LBB17_4:                               // %for.body.i23
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		lshl	 r3, r2, r1;		nopm	;		nopv	
	nopa	;		nopb	;		movs	p3, p1;		nopx	;		mov	m0, r3;		nopv	
	padda	 [p3], m0;		nopb	;		nopx	
	vldb	 x8, [p3, #64];		vmov	x4, x8
	vldb	 x10, [p3, #0];		vmov	x6, x10
	vst	 x4, [p2, #64]
	vst	 x6, [p2, #0];		mov	p2, p0
.L_LEnd19:
	padda	 [p2], m0;		nopb	;		nops	;		add	 r2, r2, r0;		nopm	;		nopv	
// %bb.5:
	nopx	
	nop	
	nop	
	vst	 x10, [p2, #0]
	vst	 x8, [p2, #64]
.LBB17_6:                               // %if.end10
	nopa	;		ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end17:
	.size	_ZL8out_copyPjiPK8bfloat16i, .Lfunc_end17-_ZL8out_copyPjiPK8bfloat16i
                                        // -- End function
	.section	.text.ew_op,"ax",@progbits
	.globl	ew_op                           // -- Begin function ew_op
	.p2align	4
	.type	ew_op,@function
ew_op:                                  // @ew_op
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		paddxm	 [sp], #64;		nopv	
	st	 p7, [sp, #-40];		nopb	;		nopx	 // 4-byte Folded Spill
	st	 p6, [sp, #-44]                 // 4-byte Folded Spill
	st	 r8, [sp, #-60];		jl	#_ZL7in_needPKji // 4-byte Folded Spill
	st	 r9, [sp, #-56]                 // 4-byte Folded Spill Delay Slot 5
	st	 r10, [sp, #-52]                // 4-byte Folded Spill Delay Slot 4
	st	 lr, [sp, #-64];		movxm	r8, #15347 // 4-byte Folded Spill Delay Slot 3
	mova	r0, #16;		st	 r11, [sp, #-48];		movxm	r9, #1033228 // 4-byte Folded Spill Delay Slot 2
	mova	r11, #4;		movs	p7, p0;		movx	r10, #12;		mov	p6, p1 //  Delay Slot 1
	movxm	p1, ##_ZL6g_in_b
	lda	 r0, [p1, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	lt	 r27, r8, r0
	sel.nez	 r1, r9, r10, r27
	add	 r0, r1, r0
	mov	dj0, r0
	lda	 r0, [p7, dj0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	ne	 r1, r0, r11
	jnz	 r1, #.LBB18_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	movs	p0, p7;		mov	p1, p6          //  Delay Slot 1
// %bb.1:                               // %if.then
	lda	 p7, [sp, #-40]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-44]                 // 4-byte Folded Reload
	lda	 r11, [sp, #-48]                // 4-byte Folded Reload
	lda	 r10, [sp, #-52]                // 4-byte Folded Reload
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	lda	 lr, [sp, #-64]                 // 4-byte Folded Reload
	j	#ew_ssm
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB18_2:                               // %if.else
	lda	 p7, [sp, #-40]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-44]                 // 4-byte Folded Reload
	lda	 r11, [sp, #-48]                // 4-byte Folded Reload
	lda	 r10, [sp, #-52]                // 4-byte Folded Reload
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	lda	 lr, [sp, #-64]                 // 4-byte Folded Reload
	j	#ew_norm
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end18:
	.size	ew_op, .Lfunc_end18-ew_op
                                        // -- End function
	.section	.text._ZL5out64PjN3aie6vectorI8bfloat16Lj32EEE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL5out64PjN3aie6vectorI8bfloat16Lj32EEE
	.type	_ZL5out64PjN3aie6vectorI8bfloat16Lj32EEE,@function
_ZL5out64PjN3aie6vectorI8bfloat16Lj32EEE: // @_ZL5out64PjN3aie6vectorI8bfloat16Lj32EEE
// %bb.0:                               // %entry
	nopa	;		movxm	p2, ##_ZL12g_out_slot_b
	lda	 r0, [p2, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	jnz	 r0, #.LBB19_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	movxm	p1, ##_ZL7g_out_b               //  Delay Slot 2
	mova	r1, #768;		movx	r3, #64 //  Delay Slot 1
// %bb.1:                               // %if.then
	mova	r2, #-1
	acq	#50, r2
	nop	
	nop	
	nop	
.LBB19_2:                               // %if.end
	lda	 r4, [p1, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	add	 r2, r4, r3
	add	 r3, r0, r3
	ne	 r4, r3, r1;		mov	dj0, r4
	jz	 r4, #.LBB19_5
	nop	                                //  Delay Slot 5
	vst	 x0, [p0, dj0]                  //  Delay Slot 4
	st	 r2, [p1, #0]                   //  Delay Slot 3
	st	 r3, [p2, #0]                   //  Delay Slot 2
	mova	r0, #0;		movxm	r1, #9216       //  Delay Slot 1
// %bb.3:                               // %if.end4
	ne	 r1, r2, r1
	jz	 r1, #.LBB19_6
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB19_4:                               // %if.end7
	nopa	;		ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.LBB19_5:                               // %if.then3
	nopa	;		nopb	;		ne	 r1, r2, r1
	mova	r3, #1;		jnz	 r1, #.LBB19_4
	rel	#51, r3                         //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	st	 r0, [p2, #0]                   //  Delay Slot 1
.LBB19_6:                               // %if.then6
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	st	 r0, [p1, #0]                   //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end19:
	.size	_ZL5out64PjN3aie6vectorI8bfloat16Lj32EEE, .Lfunc_end19-_ZL5out64PjN3aie6vectorI8bfloat16Lj32EEE
                                        // -- End function
	.section	.text._ZL8ln_statsN3aie5accumIu10__accfloatLj32EEES1_NS_6vectorI8bfloat16Lj32EEERS4_S5_RS1_,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL8ln_statsN3aie5accumIu10__accfloatLj32EEES1_NS_6vectorI8bfloat16Lj32EEERS4_S5_RS1_
	.type	_ZL8ln_statsN3aie5accumIu10__accfloatLj32EEES1_NS_6vectorI8bfloat16Lj32EEERS4_S5_RS1_,@function
_ZL8ln_statsN3aie5accumIu10__accfloatLj32EEES1_NS_6vectorI8bfloat16Lj32EEERS4_S5_RS1_: // @_ZL8ln_statsN3aie5accumIu10__accfloatLj32EEES1_NS_6vectorI8bfloat16Lj32EEERS4_S5_RS1_
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	r0, #-16512;		nopv	
	nops	;		vmov	bmll2, x4
	paddxm	 [sp], #1024
	vmov	bmll1, x6
	vmov	bmlh1, x7
	st	 r8, [sp, #-124];		vmov	bmlh2, x5 // 4-byte Folded Spill
	vconv.bf16.fp32	 x6, cml1;		vbcst.16	 x1, r0
	mova	r8, #828;		vconv.bf16.fp32	 x2, cml2;		vmov	bmll0, x6
	vmov	bmlh0, x7;		vmac.f	dm0, dm0, x6, x1, r8
	vmov	bmll1, x4;		vmac.f	dm1, dm1, x2, x1, r8
	vmov	bmlh1, x5
	mova	m0, #-1024;		mov	p3, sp
	padda	 [p3], m0
	vst	 x4, [p3, #0]
	vconv.bf16.fp32	 x4, cml0
	vconv.bf16.fp32	 x10, cml1;		movxm	r0, #17472
	vbcst.16	 x8, r0;		vmac.f	dm0, dm0, x4, x1, r8
	vmac.f	dm1, dm1, x10, x1, r8
	vmul.f	dm2, x6, x8, r8
	movxm	r0, #16384
	vbcst.16	 x6, r0
	vmac.f	dm2, dm2, x4, x8, r8
	vconv.bf16.fp32	 x4, cml0;		vmul.f	dm1, x10, x6, r8
	vconv.bf16.fp32	 x4, cml1
	vmac.f	dm0, dm2, x4, x8, r8
	vmul.f	dm1, x4, x6, r8
	nop	
	vmsc.f	dm0, dm0, x2, x2, r8
	vconv.bf16.fp32	 x8, cml1
	nop	
	st	 r12, [sp, #-108];		vmsc.f	dm0, dm0, x2, x8, r8 // 4-byte Folded Spill
	vconv.bf16.fp32	 x4, cml1;		movxm	r3, #16256
	st	 r11, [sp, #-112];		mov	r12, p0 // 4-byte Folded Spill
	st	 r10, [sp, #-116];		mov	r11, p1;		vmsc.f	dm0, dm0, x2, x4, r8 // 4-byte Folded Spill
	st	 p6, [sp, #-104];		movxm	r1, #-20124 // 4-byte Folded Spill
	st	 p7, [sp, #-100];		movxm	r2, #11492 // 4-byte Folded Spill
	st	 lr, [sp, #-128];		mov	r10, p2;		vmsc.f	dm0, dm0, x10, x10, r8 // 4-byte Folded Spill
	st	 r9, [sp, #-120];		vbcst.16	 x0, r3 // 4-byte Folded Spill
	vst	 x0, [sp, #-192];		mov	r9, p3 // 64-byte Folded Spill
	vst	 x1, [sp, #-256];		jl	#_ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_ // 64-byte Folded Spill
	mova	m0, #-896;		vst	 x5, [p3, #64];		mov	p3, sp //  Delay Slot 5
	mova	m0, #-768;		paddb	 [p3], m0;		vst	 x0, [sp, #-512];		mov	p7, sp // 64-byte Folded Spill Delay Slot 4
	mova	m0, #-640;		paddb	 [p7], m0;		vst	 bmlh0, [p3, #64];		mov	p6, sp //  Delay Slot 3
	padda	 [p6], m0;		vst	 bmll0, [p3, #0];		movxm	r0, #13796 //  Delay Slot 2
	movs	p1, p3;		mov	p0, p7          //  Delay Slot 1
	vlda	 x0, [sp, #-192];		nopxm	 // 64-byte Folded Reload
	vlda	 x2, [sp, #-512]                // 64-byte Folded Reload
	nop	
	vlda	 bmlh0, [p7, #64]
	vlda	 bmll0, [p7, #0]
	nop	
	nop	
	nop	
	vlda	 x2, [sp, #-256];		vmac.f	dm0, dm0, x0, x2, r8 // 64-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm0, x0, x2, r8
	nop	
	jl	#_ZL7rsqrt_iN3aie6vectorI8bfloat16Lj32EEES2_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vconv.bf16.fp32	 x2, cml0               //  Delay Slot 2
	mov	p0, p7                          //  Delay Slot 1
	vlda	 bmlh0, [p7, #64];		nopb	;		nopxm	
	vlda	 bmll0, [p7, #0]
	vlda	 x0, [sp, #-256]                // 64-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml0
	nop	
	vmac.f	dm0, dm0, x2, x0, r8
	nop	
	nop	
	mov	p0, r12
	vst	 x2, [p0, #0]
	vst	 x2, [sp, #-192];		mov	p0, r11 // 64-byte Folded Spill
	vst.conv.bf16.fp32	 cml0, [p0, #0];		jl	#_ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_
	vst	 bmll0, [sp, #-512]             // 64-byte Folded Spill Delay Slot 5
	vst	 bmlh0, [sp, #-448];		movxm	r0, #15019 // 64-byte Folded Spill Delay Slot 4
	vst	 bmhl0, [sp, #-384];		movxm	r1, #-18901 // 64-byte Folded Spill Delay Slot 3
	vst	 bmhh0, [sp, #-320];		movxm	r2, #12715 // 64-byte Folded Spill Delay Slot 2
	movs	p1, r9;		mov	p0, p6          //  Delay Slot 1
	vlda	 bmlh0, [p6, #64];		nopb	;		nopxm	
	vlda	 bmll0, [p6, #0]
	vlda	 x8, [sp, #-256]                // 64-byte Folded Reload
	vlda	 bmll2, [sp, #-512]             // 64-byte Folded Reload
	vlda	 bmlh2, [sp, #-448]             // 64-byte Folded Reload
	vlda	 x10, [sp, #-192]               // 64-byte Folded Reload
	nop	
	nop	
	vlda	 bmhl2, [sp, #-384];		vconv.bf16.fp32	 x0, cml0 // 64-byte Folded Reload
	vlda	 bmhh2, [sp, #-320]             // 64-byte Folded Reload
	vmac.f	dm0, dm0, x0, x8, r8
	vconv.bf16.fp32	 x2, cml2
	vnegmul.f	dm1, x0, x10, r8
	vmac.f	dm2, dm2, x2, x8, r8
	nop	
	vmsc.f	dm1, dm1, x0, x2, r8
	vconv.bf16.fp32	 x4, cml0
	nop	
	vmsc.f	dm1, dm1, x4, x10, r8
	vconv.bf16.fp32	 x6, cml2;		vmac.f	dm0, dm0, x4, x8, r8
	nop	
	vmsc.f	dm1, dm1, x0, x6, r8
	lda	 p7, [sp, #-100]                // 4-byte Folded Reload
	lda	 lr, [sp, #-128]                // 4-byte Folded Reload
	lda	 p6, [sp, #-104];		vmsc.f	dm1, dm1, x4, x2, r8 // 4-byte Folded Reload
	lda	 r12, [sp, #-108];		vconv.bf16.fp32	 x0, cml0 // 4-byte Folded Reload
	lda	 r11, [sp, #-112]               // 4-byte Folded Reload
	lda	 r10, [sp, #-116];		vmsc.f	dm0, dm1, x0, x10, r8 // 4-byte Folded Reload
	lda	 r9, [sp, #-120]                // 4-byte Folded Reload
	lda	 r8, [sp, #-124]                // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	mov	p0, r10                         //  Delay Slot 4
	vst	 bmlh0, [p0, #64]               //  Delay Slot 3
	vst	 bmll0, [p0, #0];		paddxm	 [sp], #-1024 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end20:
	.size	_ZL8ln_statsN3aie5accumIu10__accfloatLj32EEES1_NS_6vectorI8bfloat16Lj32EEERS4_S5_RS1_, .Lfunc_end20-_ZL8ln_statsN3aie5accumIu10__accfloatLj32EEES1_NS_6vectorI8bfloat16Lj32EEERS4_S5_RS1_
                                        // -- End function
	.section	.text._ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE
	.type	_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE,@function
_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE: // @_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		j	#.LBB21_2;		nopv	
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	mova	r1, #67;		movx	r0, #1          //  Delay Slot 3
	mova	r3, #60;		movx	r2, #66;		vmov	bmll0, x4 //  Delay Slot 2
	mova	r5, #-1;		movx	r4, #4;		vmov	bmlh0, x5 //  Delay Slot 1
.LBB21_1:                               // %_ZN3aie6detail24interleave_bits_zip_implILj32EfLj16EE3runERKNS_6vectorIfLj16EEES6_RKNS0_12shuffle_modeILj32ELj16EEE.exit.i.i.i
                                        //   in Loop: Header=BB21_2 Depth=1
	nopa	;		nopb	;		nops	;		clz	 r6, r4;		nopm	;		nopv	
	lshl	 r6, r6, r0;		vmov	x0, bmlh0
	sub	 r7, r1, r6;		vmov	x2, bmll0
	sub	 r7, r2, r6;		vshuffle	x4, x2, x0, r7
	add	r7, r6, #-45;		vshuffle	x0, x2, x0, r7
	add	r6, r6, #-46;		vshuffle	bmlh1, x4, x0, r7;		vadd.f	dm0, dm0, dm1, r3
	vshuffle	bmll1, x4, x0, r6
	nop	
	nop	
	nop	
	lshl	 r4, r4, r5
.LBB21_2:                               // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		jnz	 r4, #.LBB21_1;		nopv	
	nopa	;		nopx	                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %for.cond.cleanup
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst	 bmlh0, [p0, #64]               //  Delay Slot 3
	vst	 bmll0, [p0, #0]                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end21:
	.size	_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE, .Lfunc_end21-_ZL7rowsum8N3aie5accumIu10__accfloatLj32EEE
                                        // -- End function
	.section	.text._ZL9out_beginPji,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL9out_beginPji
	.type	_ZL9out_beginPji,@function
_ZL9out_beginPji:                       // @_ZL9out_beginPji
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	p0, ##_ZL12g_out_slot_b;		nopv	
	lda	 r27, [p0, #0];		nopb	;		nopx	
	nop	
	nop	
	nop	
	nop	
	nop	
	movx	r0, #768
	mova	r1, #0;		sub	 r0, r0, r27
	sel.eqz	 r0, r1, r0, r27
	movxm	r1, #1600
	ge	 r1, r0, r1
	jnz	 r1, #.LBB22_2
	nop	                                //  Delay Slot 5
	paddxm	 [sp], #64                      //  Delay Slot 4
	st	 lr, [sp, #-64]                 // 4-byte Folded Spill Delay Slot 3
	st	 r8, [sp, #-60]                 // 4-byte Folded Spill Delay Slot 2
	st	 r9, [sp, #-56]                 // 4-byte Folded Spill Delay Slot 1
// %bb.1:                               // %while.end
	movxm	r4, #-1431655765
	mova	r1, #832;		jl	#__muldi3
	lt	 r27, r1, r0                    //  Delay Slot 5
	lt	 r8, r0, r1                     //  Delay Slot 4
	sel.nez	 r2, r0, r1, r27                //  Delay Slot 3
	mova	r3, #0;		add	 r0, r0, r8     //  Delay Slot 2
	mova	r9, #-1;		sub	 r2, r2, r0;		mov	r5, #0 //  Delay Slot 1
	mova	r0, #-9;		nopb	;		nops	;		nopxm	;		nopv	
	lshl	 r0, r1, r0
	add	 r0, r0, r8
	xor	 r0, r0, r9
	acq	#50, r0
.LBB22_2:                               // %if.end
	lda	 lr, [sp, #-64];		nopb	;		nopxm	;		nops	 // 4-byte Folded Reload
	nop	
	nop	
	nop	
	nop	
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end22:
	.size	_ZL9out_beginPji, .Lfunc_end22-_ZL9out_beginPji
                                        // -- End function
	.section	.text._ZL7out_endPji,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL7out_endPji
	.type	_ZL7out_endPji,@function
_ZL7out_endPji:                         // @_ZL7out_endPji
// %bb.0:                               // %entry
	nopa	;		nopb	;		nops	;		movxm	p0, ##_ZL7g_out_b;		nopv	
	paddxm	 [sp], #64
	lda	 r0, [p0, #0];		nopx	
	st	 p6, [sp, #-48];		movxm	p6, ##_ZL12g_out_slot_b // 4-byte Folded Spill
	lda	 r8, [p6, #0]
	nop	
	movxm	r1, #7615
	movxm	r2, #-7616
	movxm	r3, #1600
	lt	 r27, r1, r0
	st	 r8, [sp, #-60];		sel.nez	 r1, r2, r3, r27 // 4-byte Folded Spill
	mova	r2, #-833;		add	 r0, r1, r0
	lt	 r1, r2, r8
	jz	 r1, #.LBB23_2
	nop	                                //  Delay Slot 5
	st	 lr, [sp, #-64]                 // 4-byte Folded Spill Delay Slot 4
	st	 r9, [sp, #-56]                 // 4-byte Folded Spill Delay Slot 3
	st	 r10, [sp, #-52]                // 4-byte Folded Spill Delay Slot 2
	st	 r0, [p0, #0];		add	 r0, r8, r3 //  Delay Slot 1
// %bb.1:                               // %while.end
	movxm	r1, #1535
	movxm	r4, #-1431655765
	jl	#__muldi3
	lt	 r27, r0, r1                    //  Delay Slot 5
	sel.nez	 r0, r0, r1, r27                //  Delay Slot 4
	movxm	r1, #2367                       //  Delay Slot 3
	mova	r5, #0;		add	 r1, r8, r1;		mov	r3, #0 //  Delay Slot 2
	mova	r10, #-768;		sub	 r2, r1, r0;		mov	r9, #832 //  Delay Slot 1
	mova	r0, #-9;		nopb	;		nopxm	
	lshl	 r0, r1, r0
	mac	r8, r8, r0, r10
	nop	
	add	 r1, r8, r9
	j	#.LBB23_3
	st	 r1, [p6, #0]                   //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	add	r0, r0, #1                      //  Delay Slot 2
	rel	#51, r0                         //  Delay Slot 1
.LBB23_2:                               // %while.end.thread
	nopa	;		nopb	;		st	 r0, [p6, #0];		nopxm	;		nopv	
.LBB23_3:                               // %if.end5
	lda	 lr, [sp, #-64];		nopxm	        // 4-byte Folded Reload
	nop	
	nop	
	lda	 p6, [sp, #-48]                 // 4-byte Folded Reload
	lda	 r10, [sp, #-52]                // 4-byte Folded Reload
	lda	 r9, [sp, #-56]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end23:
	.size	_ZL7out_endPji, .Lfunc_end23-_ZL7out_endPji
                                        // -- End function
	.section	.text._ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_
	.type	_ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_,@function
_ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_: // @_ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_
// %bb.0:                               // %entry
	vlda	 bmlh0, [p1, #64];		nopb	;		nopx	
	vlda	 bmll0, [p1, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	movxm	r3, #-16512
	vconv.bf16.fp32	 x2, cml0;		vbcst.16	 x0, r3
	mova	r3, #828
	vbcst.16	 x4, r0;		vmac.f	dm0, dm0, x2, x0, r3
	nop	
	vmul.f	dm1, x2, x4, r3
	vbcst.16	 x6, r1
	nop	
	vmac.f	dm1, dm1, x2, x6, r3
	vconv.bf16.fp32	 x8, cml0
	nop	
	vmac.f	dm1, dm1, x8, x4, r3
	vbcst.16	 x0, r2;		vmac.f	dm0, dm0, x8, x0, r3
	nop	
	vmac.f	dm1, dm1, x2, x0, r3
	nop	
	nop	
	vmac.f	dm1, dm1, x8, x6, r3
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmac.f	dm0, dm1, x0, x4, r3
	nop	
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst	 bmlh0, [p0, #64]               //  Delay Slot 3
	vst	 bmll0, [p0, #0]                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end24:
	.size	_ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_, .Lfunc_end24-_ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_
                                        // -- End function
	.section	.text._ZL7rsqrt_iN3aie6vectorI8bfloat16Lj32EEES2_,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL7rsqrt_iN3aie6vectorI8bfloat16Lj32EEES2_
	.type	_ZL7rsqrt_iN3aie6vectorI8bfloat16Lj32EEES2_,@function
_ZL7rsqrt_iN3aie6vectorI8bfloat16Lj32EEES2_: // @_ZL7rsqrt_iN3aie6vectorI8bfloat16Lj32EEES2_
// %bb.0:                               // %entry
	nopa	;		nopb	;		movxm	r0, #3490
	mov	crupsmode, #0
	mov	r1, crrnd
	mov	crrnd, #0
	mov	crsrsmode, #0
	vbcst.16	 x6, r0
	mova	r0, #0;		vmax_lt.bf16	 x0, r16, x0, x6
	mov	s0, r0
	vups.2x	cml0, x0, s0, upssign1
	vbcst.16	 x4, r0
	mov	s0, #1
	vsrs.2x	x8, cml0, s0, srssign1;		movxm	r0, #24375
	vbcst.16	 x10, r0
	movxm	r0, #16128
	vbcst.16	 x6, r0
	mova	r0, #828;		vsub.16	 x8, x10, x8
	vmul.f	dm0, x0, x6, r0
	vmul.f	dm1, x8, x8, r0
	nop	
	nop	
	nop	
	mov	crrnd, r1
	vconv.bf16.fp32	 x10, cml0
	vconv.bf16.fp32	 x3, cml1;		movxm	r2, #1069547520
	vbcst.32	 x1, r2
	vmov	bmll0, x1;		vmsc.f	dm0, dm0, x10, x3, r0
	vmov	bmlh0, x1
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x10, cml0;		movxm	r1, #16384
	vbcst.16	 x8, r1
	movxm	r1, #-16512;		vmul.f	dm0, x8, x10, r0
	vbcst.16	 x10, r1
	movxm	r1, #1065353216
	j	#.LBB25_2
	vbcst.32	 x3, r1                         //  Delay Slot 5
	movxm	r1, #16256                      //  Delay Slot 4
	vmov	bmll0, x3                       //  Delay Slot 3
	vconv.bf16.fp32	 x1, cml0;		vmov	bmlh0, x3 //  Delay Slot 2
	mova	r1, #2;		vbcst.16	 x3, r1         //  Delay Slot 1
.LBB25_1:                               // %for.body
                                        //   in Loop: Header=BB25_2 Depth=1
	vmul.f	dm1, x4, x8, r0
	nop	
	nop	
	nop	
	nop	
	vmul.f	dm2, x1, x1, r0
	vconv.bf16.fp32	 x5, cml1
	nop	
	vmac.f	dm1, dm2, x1, x5, r0
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x5, cml1
	nop	
	vmac.f	dm1, dm1, x5, x10, r0
	nop	
	nop	
	nop	
	nop	
	vmsc.f	dm2, dm2, x0, x5, r0
	vconv.bf16.fp32	 x7, cml1;		vmov	cml2, cml0
	nop	
	vmsc.f	dm1, dm2, x0, x7, r0
	nop	
	nop	
	vmsc.f	dm1, dm1, x2, x5, r0
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x5, cml1
	nop	
	vmul.f	dm1, x5, x6, r0
	nop	
	vmul.f	dm2, x1, x3, r0
	nop	
	nop	
	vmac.f	dm2, dm2, x4, x3, r0
	vconv.bf16.fp32	 x4, cml1
	nop	
	vmac.f	dm1, dm2, x1, x4, r0
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x1, cml1
	nop	
	vmac.f	dm1, dm1, x1, x10, r0
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x4, cml1;		add	r1, r1, #-1
.LBB25_2:                               // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		nops	;		jnz	 r1, #.LBB25_1;		nopv	
	nopa	;		nopx	                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.3:                               // %for.cond.cleanup
	vmul.f	dm0, x1, x3, r0
	nop	
	nop	
	vmac.f	dm0, dm0, x4, x3, r0
	nop	
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst	 bmlh0, [p0, #64]               //  Delay Slot 3
	vst	 bmll0, [p0, #0]                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end25:
	.size	_ZL7rsqrt_iN3aie6vectorI8bfloat16Lj32EEES2_, .Lfunc_end25-_ZL7rsqrt_iN3aie6vectorI8bfloat16Lj32EEES2_
                                        // -- End function
	.section	.text._ZL8rms_rstdN3aie5accumIu10__accfloatLj32EEENS_6vectorI8bfloat16Lj32EEE,"ax",@progbits
	.p2align	4                               // -- Begin function _ZL8rms_rstdN3aie5accumIu10__accfloatLj32EEENS_6vectorI8bfloat16Lj32EEE
	.type	_ZL8rms_rstdN3aie5accumIu10__accfloatLj32EEENS_6vectorI8bfloat16Lj32EEE,@function
_ZL8rms_rstdN3aie5accumIu10__accfloatLj32EEENS_6vectorI8bfloat16Lj32EEE: // @_ZL8rms_rstdN3aie5accumIu10__accfloatLj32EEENS_6vectorI8bfloat16Lj32EEE
// %bb.0:                               // %entry
	nopa	;		nopb	;		paddxm	 [sp], #512;		nops	
	st	 p6, [sp, #-56];		movxm	r1, #-18901 // 4-byte Folded Spill
	st	 r8, [sp, #-60];		movxm	r0, #16256 // 4-byte Folded Spill
	st	 p7, [sp, #-52];		vbcst.16	 x2, r0 // 4-byte Folded Spill
	vst	 x0, [sp, #-128];		mov	p1, sp // 64-byte Folded Spill
	mova	r3, #0;		st	 lr, [sp, #-64];		jl	#_ZL4mul3RKN3aie5accumIu10__accfloatLj32EEE8bfloat16S4_S4_ // 4-byte Folded Spill
	padda	 [p1], #-512;		vst	 x2, [sp, #-256];		vbcst.16	 x0, r3 // 64-byte Folded Spill Delay Slot 5
	vst	 x5, [p1, #64];		movxm	r2, #12715 //  Delay Slot 4
	vst	 x0, [sp, #-192];		mov	p7, sp // 64-byte Folded Spill Delay Slot 3
	padda	 [p7], #-384;		vst	 x4, [p1, #0];		movxm	r0, #15019 //  Delay Slot 2
	mova	r8, #828;		movs	p6, p0;		mov	p0, p7 //  Delay Slot 1
	vlda	 bmlh0, [p7, #64];		nopb	;		nops	;		nopxm	;		nopv	
	vlda	 bmll0, [p7, #0]
	vlda	 x2, [sp, #-256]                // 64-byte Folded Reload
	nop	
	nop	
	nop	
	vlda	 x0, [sp, #-128]                // 64-byte Folded Reload
	nop	
	vconv.bf16.fp32	 x0, cml0
	nop	
	vmul.f	dm0, x0, x2, r8
	nop	
	nop	
	vlda	 x2, [sp, #-192];		vmac.f	dm0, dm0, x0, x2, r8 // 64-byte Folded Reload
	nop	
	jl	#_ZL7rsqrt_iN3aie6vectorI8bfloat16Lj32EEES2_
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vconv.bf16.fp32	 x0, cml0               //  Delay Slot 2
	mov	p0, p7                          //  Delay Slot 1
	lda	 lr, [sp, #-64]                 // 4-byte Folded Reload
	nop	
	vlda	 bmlh0, [p7, #64]
	vlda	 bmll0, [p7, #0]
	lda	 p7, [sp, #-52]                 // 4-byte Folded Reload
	lda	 r8, [sp, #-60]                 // 4-byte Folded Reload
	lda	 p6, [sp, #-56]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml0, [p6, #0];		paddxm	 [sp], #-512 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end26:
	.size	_ZL8rms_rstdN3aie5accumIu10__accfloatLj32EEENS_6vectorI8bfloat16Lj32EEE, .Lfunc_end26-_ZL8rms_rstdN3aie5accumIu10__accfloatLj32EEENS_6vectorI8bfloat16Lj32EEE
                                        // -- End function
	.type	_ZL7g_arena,@object             // @_ZL7g_arena
	.section	.bss._ZL7g_arena,"aw",@nobits
	.p2align	6, 0x0
_ZL7g_arena:
	.space	24576
	.size	_ZL7g_arena, 24576

	.type	_ZL8g_panels,@object            // @_ZL8g_panels
	.section	.bss._ZL8g_panels,"aw",@nobits
	.p2align	6, 0x0
_ZL8g_panels:
	.space	1024
	.size	_ZL8g_panels, 1024

	.type	_ZL7g_out_b,@object             // @_ZL7g_out_b
	.section	.bss._ZL7g_out_b,"aw",@nobits
	.p2align	2, 0x0
_ZL7g_out_b:
	.word	0                               // 0x0
	.size	_ZL7g_out_b, 4

	.type	_ZL10g_in_avail,@object         // @_ZL10g_in_avail
	.section	.bss._ZL10g_in_avail,"aw",@nobits
	.p2align	2, 0x0
_ZL10g_in_avail:
	.word	0                               // 0x0
	.size	_ZL10g_in_avail, 4

	.type	_ZL6g_in_b,@object              // @_ZL6g_in_b
	.section	.bss._ZL6g_in_b,"aw",@nobits
	.p2align	2, 0x0
_ZL6g_in_b:
	.word	0                               // 0x0
	.size	_ZL6g_in_b, 4

	.type	_ZL13g_in_consumed,@object      // @_ZL13g_in_consumed
	.section	.bss._ZL13g_in_consumed,"aw",@nobits
	.p2align	2, 0x0
_ZL13g_in_consumed:
	.word	0                               // 0x0
	.size	_ZL13g_in_consumed, 4

	.type	_ZL12g_out_slot_b,@object       // @_ZL12g_out_slot_b
	.section	.bss._ZL12g_out_slot_b,"aw",@nobits
	.p2align	2, 0x0
_ZL12g_out_slot_b:
	.word	0                               // 0x0
	.size	_ZL12g_out_slot_b, 4

	.type	_ZL6g_ptmp,@object              // @_ZL6g_ptmp
	.section	.bss._ZL6g_ptmp,"aw",@nobits
	.p2align	6, 0x0
_ZL6g_ptmp:
	.space	1024
	.size	_ZL6g_ptmp, 1024

	.ident	"clang version 22.0.0 (https://github.com/Xilinx/llvm-aie 0006955eb0bd0a9033185a5be72b621dce63ce21)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZL7g_arena
	.addrsig_sym _ZL8g_panels
	.addrsig_sym _ZL6g_ptmp
