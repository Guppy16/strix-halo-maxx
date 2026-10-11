	.text
	.section	.text.probe0,"ax",@progbits
	.globl	probe0
	.p2align	4
	.type	probe0,@function
probe0:
	mov	p3, p1
	mov	p4, p1
	mov	p1, p0
	mov	p5, p0
	mova	r25, #0
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #320
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	mov	p1, p5
	mova	r25, #0
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
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
	vst	x8, [p3], #64
	st	el8, [p4], #4
	st	eh8, [p4], #4
	vst	x9, [p3], #64
	st	el9, [p4], #4
	st	eh9, [p4], #4
	vst	x10, [p3], #64
	st	el10, [p4], #4
	st	eh10, [p4], #4
	vst	x11, [p3], #64
	st	el11, [p4], #4
	st	eh11, [p4], #4
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
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
	mov	p3, p1
	mov	p4, p1
	mov	p1, p0
	mov	p5, p0
	mova	r25, #0
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #320
	movxm	m2, #-4544
	mova	dn2, #0
	mova	dj2, #64
	mova	dc2, #0
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576.2d	ex7, [p1, lf1, r25, d2];	mov	r25, #0
	vldb.fill.512	[p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vst	x8, [p3], #64
	st	el8, [p4], #4
	st	eh8, [p4], #4
	vst	x9, [p3], #64
	st	el9, [p4], #4
	st	eh9, [p4], #4
	vst	x10, [p3], #64
	st	el10, [p4], #4
	st	eh10, [p4], #4
	vst	x11, [p3], #64
	st	el11, [p4], #4
	st	eh11, [p4], #4
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
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
	mov	p4, p1
	mov	p1, p0
	mov	p5, p0
	mova	r25, #0
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #320
	movxm	m2, #-4608
	mova	dn2, #0
	mova	dj2, #64
	mova	dc2, #0
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576.2d	ex7, [p1, lf1, r25, d2];	mov	r25, #0
	vldb.fill.512	[p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vst	x8, [p3], #64
	st	el8, [p4], #4
	st	eh8, [p4], #4
	vst	x9, [p3], #64
	st	el9, [p4], #4
	st	eh9, [p4], #4
	vst	x10, [p3], #64
	st	el10, [p4], #4
	st	eh10, [p4], #4
	vst	x11, [p3], #64
	st	el11, [p4], #4
	st	eh11, [p4], #4
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
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
	mov	p4, p1
	mov	p1, p0
	mov	p5, p0
	mova	r25, #0
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #320
	movxm	m2, #-4608
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25];	mov	r25, #0
	padda	[p1], m2
	vldb.fill.512	[p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vst	x8, [p3], #64
	st	el8, [p4], #4
	st	eh8, [p4], #4
	vst	x9, [p3], #64
	st	el9, [p4], #4
	st	eh9, [p4], #4
	vst	x10, [p3], #64
	st	el10, [p4], #4
	st	eh10, [p4], #4
	vst	x11, [p3], #64
	st	el11, [p4], #4
	st	eh11, [p4], #4
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
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
	mov	p4, p1
	mov	p1, p0
	mov	p5, p0
	mova	r25, #0
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #448
	padda	[p4], #320
	movxm	m2, #-4608
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25];	mov	r25, #0
	padda	[p1], m2;	vldb.fill.512	[p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vst	x8, [p3], #64
	st	el8, [p4], #4
	st	eh8, [p4], #4
	vst	x9, [p3], #64
	st	el9, [p4], #4
	st	eh9, [p4], #4
	vst	x10, [p3], #64
	st	el10, [p4], #4
	st	eh10, [p4], #4
	vst	x11, [p3], #64
	st	el11, [p4], #4
	st	eh11, [p4], #4
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	vldb.fill.512	[p1, lf1, r25]
	vldb.pop.576	ex0, [p1, lf1, r25]
	vldb.pop.576	ex1, [p1, lf1, r25]
	vldb.pop.576	ex2, [p1, lf1, r25]
	vldb.pop.576	ex3, [p1, lf1, r25]
	vldb.pop.576	ex4, [p1, lf1, r25]
	vldb.pop.576	ex5, [p1, lf1, r25]
	vldb.pop.576	ex6, [p1, lf1, r25]
	vldb.pop.576	ex7, [p1, lf1, r25]
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
	st	el0, [p4], #4
	st	eh0, [p4], #4
	vst	x1, [p3], #64
	st	el1, [p4], #4
	st	eh1, [p4], #4
	vst	x2, [p3], #64
	st	el2, [p4], #4
	st	eh2, [p4], #4
	vst	x3, [p3], #64
	st	el3, [p4], #4
	st	eh3, [p4], #4
	vst	x4, [p3], #64
	st	el4, [p4], #4
	st	eh4, [p4], #4
	vst	x5, [p3], #64
	st	el5, [p4], #4
	st	eh5, [p4], #4
	vst	x6, [p3], #64
	st	el6, [p4], #4
	st	eh6, [p4], #4
	vst	x7, [p3], #64
	st	el7, [p4], #4
	st	eh7, [p4], #4
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe4:
	.size	probe4, .Lfunc_end_probe4-probe4
