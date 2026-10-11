	.text
	.section	.text.probe0,"ax",@progbits
	.globl	probe0
	.p2align	4
	.type	probe0,@function
probe0:
	movxm	m2, #-16
	mova	dn2, #4
	mova	dj2, #4
	mova	dc2, #0
	nop
	nop
	nop
	nop
	nop
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	r5, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe0:
	.size	probe0, .Lfunc_end_probe0-probe0
	.section	.text.probe1,"ax",@progbits
	.globl	probe1
	.p2align	4
	.type	probe1,@function
probe1:
	movxm	m2, #-16
	mova	dn2, #4
	mova	dj2, #4
	mova	dc2, #0
	nop
	nop
	nop
	nop
	nop
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	p4, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe1:
	.size	probe1, .Lfunc_end_probe1-probe1
	.section	.text.probe2,"ax",@progbits
	.globl	probe2
	.p2align	4
	.type	probe2,@function
probe2:
	movxm	m2, #-16
	mova	dn2, #4
	mova	dj2, #4
	mova	dc2, #0
	nop
	nop
	nop
	nop
	nop
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda.2d	m0, [p0], d2
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe2:
	.size	probe2, .Lfunc_end_probe2-probe2
	.section	.text.probe3,"ax",@progbits
	.globl	probe3
	.p2align	4
	.type	probe3,@function
probe3:
	nop
	nop
	nop
	nop
	nop
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	p4, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, p4
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe3:
	.size	probe3, .Lfunc_end_probe3-probe3
	.section	.text.probe4,"ax",@progbits
	.globl	probe4
	.p2align	4
	.type	probe4,@function
probe4:
	nop
	nop
	nop
	nop
	nop
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	lda	m0, [p0], #4
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	r5, m0
	nop
	nop
	nop
	st	r5, [p1], #4
	nop
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe4:
	.size	probe4, .Lfunc_end_probe4-probe4
