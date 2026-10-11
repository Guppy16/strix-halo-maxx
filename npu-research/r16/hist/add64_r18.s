	.text
	.section	.text.v2t_pre,"ax",@progbits
	.globl	v2t_pre
	.p2align	4
	.type	v2t_pre,@function
// round 18 V2T (RECONSTRUCTED): add takes a 7-bit signed immediate; #64 assembled silently as #-64
v2t_pre:
	add r1, r1, #64
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_v2t_pre:
	.size	v2t_pre, .Lfunc_end_v2t_pre-v2t_pre
