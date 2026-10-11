// pass 19 RELC: rel.cond #52 (lock 4, the A ring's free lock) with the condition register r26 = the argument (r0).
// void relc(int32_t cond)
	.text
	.globl	relc
	.p2align	4
	.type	relc,@function
relc:
	mov	r26, r0
	mova	r5, #1
	nop
	nop
	nop
	nop
	rel.cond	#52, r5, r26
	nop
	nop
	nop
	nop
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_relc:
	.size	relc, .Lfunc_end_relc-relc
