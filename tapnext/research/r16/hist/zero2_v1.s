// RECONSTRUCTED (research rev 11): r14/zero2.s with the six full-form nop bundles after add.nc lc replaced by six 2-byte nops, the round-14 version that ran its body once (NOTES 19.3)
// round 14: acc_zero's two zero fills as ONE zero-overhead loop, one 64 B store per bundle: per trip 64 B of the C object's
// fp32 rows (12,288 B) and 128 B of acc_hi (24,576 B), 192 trips. Peano's C++ stored 2 per 3 bundles in 336 B of text.
//     extern "C" void zero2(float *a /*p0*/, float *b /*p1*/, int trips /*r0*/)    a: 64 B x trips, b: 128 B x trips
// Measured (r13/attkb.py --zero): with 6 two-byte nops between `add.nc lc` and the loop start the loop ran ONCE; 6 full-form
// (16 B) nop bundles work.
	.text
	.section	.text.zero2,"ax",@progbits
	.globl	zero2
	.p2align	4
	.type	zero2,@function
zero2:
	mov	r7, r0
	mova	r2, #0
	vbcst.32	x0, r2
	movxm	ls, #zero2_a
	movxm	le, #zero2_a_last
	add.nc	lc, r7, #0
	nop
	nop
	nop
	nop
	nop
	nop
	.p2align	4
zero2_a:
	nopa;		nopb;		vst	x0, [p0], #64;		nopx;		nopm;		nopv
	nopa;		nopb;		vst	x0, [p1], #64;		nopx;		nopm;		nopv
zero2_a_last:
	nopa;		nopb;		vst	x0, [p1], #64;		nopx;		nopm;		nopv
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_zero2:
	.size	zero2, .Lfunc_end_zero2-zero2
