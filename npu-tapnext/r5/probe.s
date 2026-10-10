	.file	"probe.cc"
	.section	.text.probe_rowsum,"ax",@progbits
	.globl	probe_rowsum                    // -- Begin function probe_rowsum
	.p2align	4
	.type	probe_rowsum,@function
probe_rowsum:                           // @probe_rowsum
// %bb.0:                               // %entry
	vlda	 bmll0, [p0, #0]
	vlda	 bmlh0, [p0, #64]
	nop	
	nop	
	movxm	ls, #.LBB0_1
	mova	r4, #3;		movxm	le, #.L_LEnd0
	mova	r5, #4;		add.nc	lc, r4, #-1
	mova	r0, #1;		clz	 r6, r5;		vmov	x0, bmll0
	mova	r2, #67;		lshl	 r6, r6, r0;		vmov	x2, bmlh0
	mova	r3, #66;		sub	 r7, r2, r6;		vmov	cml4, cml0
	sub	 r7, r3, r6;		vshuffle	x4, x0, x2, r7
	add	r7, r6, #-46;		vshuffle	x0, x0, x2, r7
	mova	r4, #-1;		add	r6, r6, #-45;		vshuffle	bmll2, x4, x0, r7
	mova	r1, #60;		lshl	 r5, r5, r4;		vshuffle	bmlh2, x4, x0, r6
.LBB0_1:                                // %_ZN3aie6detail26interleave_bits_unzip_implILj32EfLj16EE3runERKNS_6vectorIfLj16EEES6_RKNS0_12shuffle_modeILj32ELj16EEE.exit.i.i.i.i
                                        // =>This Inner Loop Header: Depth=1
	vadd.f	dm4, dm4, dm3, r1
	vmov	cml3, cml2
	nop	
	nop	
	nop	
	clz	 r6, r5
	lshl	 r6, r6, r0;		vmov	x8, bmll4
	sub	 r7, r2, r6;		vmov	x10, bmlh4
	sub	 r7, r3, r6;		vshuffle	x1, x8, x10, r7
	add	r7, r6, #-46;		vshuffle	x3, x8, x10, r7
	add	r6, r6, #-45;		vshuffle	bmll2, x1, x3, r7
.L_LEnd0:
	nopa	;		nopb	;		nops	;		lshl	 r5, r5, r4;		vshuffle	bmlh2, x1, x3, r6;		nopv	
// %bb.2:
	nopx	;		vadd.f	dm0, dm4, dm1, r1
	vmov	cml1, cml2
	nop	
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst	 bmll0, [p1, #0]                //  Delay Slot 3
	vst	 bmlh0, [p1, #64]               //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end0:
	.size	probe_rowsum, .Lfunc_end0-probe_rowsum
                                        // -- End function
	.section	.text.probe_rowsum_b,"ax",@progbits
	.globl	probe_rowsum_b                  // -- Begin function probe_rowsum_b
	.p2align	4
	.type	probe_rowsum_b,@function
probe_rowsum_b:                         // @probe_rowsum_b
// %bb.0:                               // %entry
	vlda	 bmll0, [p0, #0];		nopb	;		nops	;		nopxm	;		nopv	
	vlda	 bmlh0, [p0, #64];		movxm	r0, #-16512
	movxm	r3, #16256
	movxm	ls, #.LBB1_1
	movxm	le, #.L_LEnd1
	mova	r1, #1;		movx	r4, #3;		vbcst.16	 x0, r0
	mova	r0, #828;		movx	r2, #64;		vbcst.16	 x2, r3
	mova	r4, #-1;		nopb	;		nops	;		movx	r3, #4;		add.nc	lc, r4, #0;		nopv	
.LBB1_1:                                // %for.body.i
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		nopb	;		vconv.bf16.fp32	 x9, cml0;		clz	 r5, r3;		nopm	;		nopv	
	nopa	;		nopb	;		nops	;		lshl	 r5, r5, r1;		nopm	;		nopv	
	sub	 r6, r2, r5;		vmac.f	dm3, dm0, x9, x0, r0
	or	 r7, r6, r1;		vshuffle	x11, x9, x9, r6
	add	r5, r5, #-44;		vshuffle	x6, x9, x9, r7
	lshl	 r3, r3, r4
	nop	
	vshuffle	x8, x6, x11, r5
	vconv.bf16.fp32	 x4, cml3
	vmac.f	dm4, dm0, x8, x2, r0
	vshuffle	x10, x4, x4, r7
	vshuffle	x1, x4, x4, r6
	vshuffle	x3, x10, x1, r5
	nop	
	vmac.f	dm0, dm4, x3, x2, r0
	nop	
	nop	
	nop	
	nop	
.L_LEnd1:
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
// %bb.2:                               // %_ZL8rowsum8bN3aie5accumIu10__accfloatLj32EEE.exit
	nopa	;		nopb	;		nops	;		ret	lr;		nopm	;		nopv	
	nopx	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst	 bmll0, [p1, #0]                //  Delay Slot 3
	vst	 bmlh0, [p1, #64]               //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end1:
	.size	probe_rowsum_b, .Lfunc_end1-probe_rowsum_b
                                        // -- End function
	.ident	"clang version 22.0.0 (https://github.com/Xilinx/llvm-aie 0006955eb0bd0a9033185a5be72b621dce63ce21)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
