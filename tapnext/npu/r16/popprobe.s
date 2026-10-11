// round 16: what 8 vlda.pop.576 from a fill return (x part + el / eh words of each ex register), A-style restart
// void probe0(const bfloat16 *in /*p0*/, bfloat16 *out /*p1*/, int n /*r0*/, const float *ab, const float *cd): out = 8 x (64 B x, 8 B e)
	.text
	.section	.text.probe0,"ax",@progbits
	.globl	probe0
	.p2align	4
	.type	probe0,@function
probe0:
	mova	r24, #0
	nop
	nop
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex0, [p0, lf0, r24]
	nop
	vlda.pop.576	ex1, [p0, lf0, r24]
	nop
	vlda.pop.576	ex2, [p0, lf0, r24]
	nop
	vlda.pop.576	ex3, [p0, lf0, r24]
	nop
	vlda.pop.576	ex4, [p0, lf0, r24]
	nop
	vlda.pop.576	ex5, [p0, lf0, r24]
	nop
	vlda.pop.576	ex6, [p0, lf0, r24]
	nop
	vlda.pop.576	ex7, [p0, lf0, r24]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	x0, [p1], #64
	vst	x1, [p1], #64
	vst	x2, [p1], #64
	vst	x3, [p1], #64
	vst	x4, [p1], #64
	vst	x5, [p1], #64
	vst	x6, [p1], #64
	vst	x7, [p1], #64
	st	el0, [p1], #4
	st	eh0, [p1], #4
	st	el1, [p1], #4
	st	eh1, [p1], #4
	st	el2, [p1], #4
	st	eh2, [p1], #4
	st	el3, [p1], #4
	st	eh3, [p1], #4
	st	el4, [p1], #4
	st	eh4, [p1], #4
	st	el5, [p1], #4
	st	eh5, [p1], #4
	st	el6, [p1], #4
	st	eh6, [p1], #4
	st	el7, [p1], #4
	st	eh7, [p1], #4
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
	mova	r24, #0
	nop
	nop
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex0, [p0, lf0, r24]
	nop
	vlda.pop.576	ex1, [p0, lf0, r24]
	nop
	vlda.pop.576	ex2, [p0, lf0, r24]
	nop
	vlda.pop.576	ex3, [p0, lf0, r24]
	nop
	vlda.pop.576	ex4, [p0, lf0, r24];		vconv.bf16.fp32	x9, cml0
	nop
	vlda.pop.576	ex5, [p0, lf0, r24];		vconv.bf16.fp32	x10, cmh0
	nop
	vlda.pop.576	ex6, [p0, lf0, r24]
	nop
	vlda.pop.576	ex7, [p0, lf0, r24]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	x0, [p1], #64
	vst	x1, [p1], #64
	vst	x2, [p1], #64
	vst	x3, [p1], #64
	vst	x4, [p1], #64
	vst	x5, [p1], #64
	vst	x6, [p1], #64
	vst	x7, [p1], #64
	st	el0, [p1], #4
	st	eh0, [p1], #4
	st	el1, [p1], #4
	st	eh1, [p1], #4
	st	el2, [p1], #4
	st	eh2, [p1], #4
	st	el3, [p1], #4
	st	eh3, [p1], #4
	st	el4, [p1], #4
	st	eh4, [p1], #4
	st	el5, [p1], #4
	st	eh5, [p1], #4
	st	el6, [p1], #4
	st	eh6, [p1], #4
	st	el7, [p1], #4
	st	eh7, [p1], #4
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
	mov	p3, p1
	mov	p1, p0
	padda	[p1], #448
	padda	[p1], #192
	mova	r25, #0
	mova	r24, #0
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex0, [p0, lf0, r24];		vldb.pop.576	ex8, [p1, lf1, r25]
	nop
	vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.pop.576	ex1, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex2, [p0, lf0, r24]
	vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex8, [p1, lf1, r25]
	nop
	vlda.pop.576	ex3, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex4, [p0, lf0, r24];		vldb.pop.576	ex8, [p1, lf1, r25]
	nop
	vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.pop.576	ex5, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex6, [p0, lf0, r24];		vldb.pop.576	ex8, [p1, lf1, r25]
	vlda.pop.576	ex7, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	x0, [p3], #64
	vst	x1, [p3], #64
	vst	x2, [p3], #64
	vst	x3, [p3], #64
	vst	x4, [p3], #64
	vst	x5, [p3], #64
	vst	x6, [p3], #64
	vst	x7, [p3], #64
	st	el0, [p3], #4
	st	eh0, [p3], #4
	st	el1, [p3], #4
	st	eh1, [p3], #4
	st	el2, [p3], #4
	st	eh2, [p3], #4
	st	el3, [p3], #4
	st	eh3, [p3], #4
	st	el4, [p3], #4
	st	eh4, [p3], #4
	st	el5, [p3], #4
	st	eh5, [p3], #4
	st	el6, [p3], #4
	st	eh6, [p3], #4
	st	el7, [p3], #4
	st	eh7, [p3], #4
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
	mov	p3, p1
	mov	p5, p0
	mov	p1, p0
	padda	[p1], #448
	padda	[p1], #192
	mova	r25, #0
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	p0, p5
	nop
	mov	r24, #0
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	nop
	vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	mov	p0, p5
	vldb.pop.576	ex11, [p1, lf1, r25]
	mov	r24, #0;		vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex0, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex1, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.pop.576	ex2, [p0, lf0, r24]
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	nop
	vlda.pop.576	ex3, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex4, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex5, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	nop
	vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.pop.576	ex6, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex7, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	nop
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	x0, [p3], #64
	vst	x1, [p3], #64
	vst	x2, [p3], #64
	vst	x3, [p3], #64
	vst	x4, [p3], #64
	vst	x5, [p3], #64
	vst	x6, [p3], #64
	vst	x7, [p3], #64
	st	el0, [p3], #4
	st	eh0, [p3], #4
	st	el1, [p3], #4
	st	eh1, [p3], #4
	st	el2, [p3], #4
	st	eh2, [p3], #4
	st	el3, [p3], #4
	st	eh3, [p3], #4
	st	el4, [p3], #4
	st	eh4, [p3], #4
	st	el5, [p3], #4
	st	eh5, [p3], #4
	st	el6, [p3], #4
	st	eh6, [p3], #4
	st	el7, [p3], #4
	st	eh7, [p3], #4
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
	mov	p3, p1
	mov	p5, p0
	mov	p1, p0
	padda	[p1], #448
	padda	[p1], #192
	mova	r25, #0
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	p0, p5
	nop
	mov	r24, #0
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	nop
	vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	mov	p0, p5
	vldb.pop.576	ex11, [p1, lf1, r25]
	mov	r24, #0;		vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex0, [p1, lf1, r25]
	nop
	vldb.pop.576	ex1, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex2, [p1, lf1, r25]
	nop
	vldb.pop.576	ex3, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex4, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	x0, [p3], #64
	vst	x1, [p3], #64
	vst	x2, [p3], #64
	vst	x3, [p3], #64
	vst	x4, [p3], #64
	vst	x5, [p3], #64
	vst	x6, [p3], #64
	vst	x7, [p3], #64
	st	el0, [p3], #4
	st	eh0, [p3], #4
	st	el1, [p3], #4
	st	eh1, [p3], #4
	st	el2, [p3], #4
	st	eh2, [p3], #4
	st	el3, [p3], #4
	st	eh3, [p3], #4
	st	el4, [p3], #4
	st	eh4, [p3], #4
	st	el5, [p3], #4
	st	eh5, [p3], #4
	st	el6, [p3], #4
	st	eh6, [p3], #4
	st	el7, [p3], #4
	st	eh7, [p3], #4
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe4:
	.size	probe4, .Lfunc_end_probe4-probe4
	.section	.text.probe5,"ax",@progbits
	.globl	probe5
	.p2align	4
	.type	probe5,@function
probe5:
	mov	p3, p1
	mov	p5, p0
	mov	p1, p0
	padda	[p1], #448
	padda	[p1], #192
	mov	p4, p1
	movxm	m3, #1152
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov	p1, p4
	mov	r25, #0
	mov	p0, p5;		vldb.fill.512	[p1, lf1, r25]
	padda	[p4], m3
	mov	r24, #0
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	nop
	vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25];		mov	p1, p4
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25];		mov	r25, #0
	mov	p0, p5;		vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25];		padda	[p4], m3
	mov	r24, #0;		vldb.pop.576	ex9, [p1, lf1, r25]
	vlda.fill.512	[p0, lf0, r24]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	vldb.pop.576	ex10, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex0, [p1, lf1, r25]
	nop
	vldb.pop.576	ex1, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex2, [p1, lf1, r25]
	nop
	vldb.pop.576	ex3, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex4, [p1, lf1, r25]
	vlda.pop.576	ex8, [p0, lf0, r24];		vldb.pop.576	ex10, [p1, lf1, r25]
	nop
	vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	x0, [p3], #64
	vst	x1, [p3], #64
	vst	x2, [p3], #64
	vst	x3, [p3], #64
	vst	x4, [p3], #64
	vst	x5, [p3], #64
	vst	x6, [p3], #64
	vst	x7, [p3], #64
	st	el0, [p3], #4
	st	eh0, [p3], #4
	st	el1, [p3], #4
	st	eh1, [p3], #4
	st	el2, [p3], #4
	st	eh2, [p3], #4
	st	el3, [p3], #4
	st	eh3, [p3], #4
	st	el4, [p3], #4
	st	eh4, [p3], #4
	st	el5, [p3], #4
	st	eh5, [p3], #4
	st	el6, [p3], #4
	st	eh6, [p3], #4
	st	el7, [p3], #4
	st	eh7, [p3], #4
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe5:
	.size	probe5, .Lfunc_end_probe5-probe5
	.section	.text.probe6,"ax",@progbits
	.globl	probe6
	.p2align	4
	.type	probe6,@function
probe6:
	mov	p3, p1
	mov	p1, p0
	padda	[p1], #448
	padda	[p1], #192
	mova	r25, #0
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex8, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex8, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
	vldb.pop.576	ex8, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex8, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	x0, [p3], #64
	vst	x1, [p3], #64
	vst	x2, [p3], #64
	vst	x3, [p3], #64
	vst	x4, [p3], #64
	vst	x5, [p3], #64
	vst	x6, [p3], #64
	vst	x7, [p3], #64
	st	el0, [p3], #4
	st	eh0, [p3], #4
	st	el1, [p3], #4
	st	eh1, [p3], #4
	st	el2, [p3], #4
	st	eh2, [p3], #4
	st	el3, [p3], #4
	st	eh3, [p3], #4
	st	el4, [p3], #4
	st	eh4, [p3], #4
	st	el5, [p3], #4
	st	eh5, [p3], #4
	st	el6, [p3], #4
	st	eh6, [p3], #4
	st	el7, [p3], #4
	st	eh7, [p3], #4
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe6:
	.size	probe6, .Lfunc_end_probe6-probe6
	.section	.text.probe7,"ax",@progbits
	.globl	probe7
	.p2align	4
	.type	probe7,@function
probe7:
	mov	p3, p1
	mov	p1, p0
	padda	[p1], #448
	padda	[p1], #192
	mova	r25, #0
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex8, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex8, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex8, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	vldb.pop.576	ex8, [p1, lf1, r25]
	vldb.pop.576	ex9, [p1, lf1, r25]
	vldb.pop.576	ex10, [p1, lf1, r25]
	vldb.pop.576	ex11, [p1, lf1, r25]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	x0, [p3], #64
	vst	x1, [p3], #64
	vst	x2, [p3], #64
	vst	x3, [p3], #64
	vst	x4, [p3], #64
	vst	x5, [p3], #64
	vst	x6, [p3], #64
	vst	x7, [p3], #64
	st	el0, [p3], #4
	st	eh0, [p3], #4
	st	el1, [p3], #4
	st	eh1, [p3], #4
	st	el2, [p3], #4
	st	eh2, [p3], #4
	st	el3, [p3], #4
	st	eh3, [p3], #4
	st	el4, [p3], #4
	st	eh4, [p3], #4
	st	el5, [p3], #4
	st	eh5, [p3], #4
	st	el6, [p3], #4
	st	eh6, [p3], #4
	st	el7, [p3], #4
	st	eh7, [p3], #4
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe7:
	.size	probe7, .Lfunc_end_probe7-probe7
