	.text
	.globl	mm_rot
	.p2align	4
	.type	mm_rot,@function
mm_rot:
	movs p3, p2
	movs p4, p2
	movs p5, p1
	mova r3, #780
	mova r25, #0
	mova r24, #0
	mova r4, #-1
	mova r5, #1
	movs p2, p0
	movxm ls, #.Lmm_rot_body
	movxm le, #.Lmm_rot_end
	mova r0, #18
	add.nc lc, r0, #0
	movxm m2, #-4608
	.p2align	4
.Lmm_rot_pro:
	lda m0, [p2], #4;		nopb;		nops;		nopx;		nopm;		nopv
	lda r26, [p2], #4
	nop
	lda p0, [p2], #4
	lda m1, [p2], #4
	nop
	nop
	nop
	padds [p3], m0
	acq.cond #53, r4, r26
	nop
	vlda bmll0, [p3], #64;		vldb.fill.512 [p1, lf1, r25]
	vlda bmlh0, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vlda bmhl0, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vlda bmhh0, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		mov r24, #0
	vlda.fill.512 [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex9, [p1, lf1, r25]
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda bmll1, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda bmlh1, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vlda bmhl1, [p3], #64;		vldb.fill.512 [p1, lf1, r25]
	vlda bmhh1, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex9, [p1, lf1, r25]
	vlda bmll2, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		vmac.f dm0, dm0, ex0, ex5, r3
	vlda bmlh2, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm1, dm1, ex0, ex5, r3
	vlda bmhl2, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vlda bmhh2, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vmac.f dm0, dm0, ex1, ex7, r3
	vlda bmll3, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda bmlh3, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		vmac.f dm1, dm1, ex1, ex7, r3
	vlda bmhl3, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vmac.f dm0, dm0, ex2, ex9, r3
	vlda bmhh3, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm2, dm2, ex0, ex5, r3
	vldb.pop.576 ex7, [p1, lf1, r25];		vmac.f dm1, dm1, ex2, ex9, r3
	.p2align	4
.Lmm_rot_body:
mm_rot_start:
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex11, [p1, lf1, r25];		padds [p3], m1;		nopx;		nopm;		vmac.f dm0, dm0, ex3, ex11, r3
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex1, ex7, r3
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex0, ex5, r3
	lda r26, [p2], #4;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex3, ex11, r3
	vlda.pop.576 ex8, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex2, ex9, r3
	vlda.pop.576 ex10, [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex1, ex7, r3
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		nopv
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex3, ex11, r3
	nopa;		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex4, ex5, r3
	nopa;		vldb.pop.576 ex11, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex2, ex9, r3
	nopa;		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex4, ex5, r3
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex6, ex7, r3
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex3, ex11, r3
	nopa;		vldb.pop.576 ex11, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex4, ex5, r3
	nopa;		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex8, ex9, r3
	nopa;		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex6, ex7, r3
	nopa;		vldb.pop.576 ex11, [p1, lf1, r25];		padds [p4], m0;		nopx;		nopm;		vmac.f dm2, dm2, ex6, ex7, r3
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex10, ex11, r3
	vlda bmll0, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex8, ex9, r3
	vlda bmlh0, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex4, ex5, r3
	vlda bmhl0, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex8, ex9, r3
	vlda bmhh0, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex10, ex11, r3
	vlda bmll1, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex6, ex7, r3
	vlda bmlh1, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		vst bmll0, [p4], #64;		nopx;		nopm;		nopv
	vlda bmhl1, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmlh0, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex10, ex11, r3
	vlda bmhh1, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmhl0, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex0, ex5, r3
	vlda bmll2, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmhh0, [p4], #64;		nopx;		nopm;		vmac.f dm3, dm3, ex8, ex9, r3
	vlda bmlh2, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmll1, [p4], #64;		nopx;		nopm;		nopv
	vlda bmhl2, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		vst bmlh1, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex1, ex7, r3
	vlda bmhh2, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmhl1, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex0, ex5, r3
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		vst bmhh1, [p4], #64;		nopx;		nopm;		vmac.f dm3, dm3, ex10, ex11, r3
	nopa;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmll2, [p4], #64;		rel.cond #52, r5, r26;		nopm;		vmac.f dm0, dm0, ex2, ex9, r3
	vlda bmll3, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		vst bmlh2, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex1, ex7, r3
	nopa;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmhl2, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex0, ex5, r3
	nopa;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmhh2, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex3, ex11, r3
	vlda bmlh3, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex2, ex9, r3
	vlda bmhl3, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex1, ex7, r3
	vlda bmhh3, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmll3, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex4, ex5, r3
	nopa;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmlh3, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex3, ex11, r3
	nopa;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmhl3, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex2, ex9, r3
	lda m0, [p2], #4;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex6, ex7, r3
	lda r26, [p2], #4;		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex0, ex5, r3
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmhh3, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex3, ex11, r3
	lda p0, [p2], #4;		vldb.pop.576 ex11, [p1, lf1, r25];		padds [p4], m1;		nopx;		nopm;		vmac.f dm0, dm0, ex8, ex9, r3
	lda m1, [p2], #4;		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex4, ex5, r3
	nopa;		vldb.pop.576 ex11, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex1, ex7, r3
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex10, ex11, r3
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex6, ex7, r3
	nopa;		vldb.pop.576 ex11, [p1, lf1, r25];		padds [p3], m0;		nopx;		nopm;		vmac.f dm3, dm3, ex2, ex9, r3
	nopa;		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		acq.cond #53, r4, r26;		mov r25, #0;		vmac.f dm2, dm2, ex4, ex5, r3
	padda [p1], m2;		nopb;		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex8, ex9, r3
	vlda bmll0, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex3, ex11, r3
	vlda bmlh0, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmll0, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex6, ex7, r3
	vlda bmhl0, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmlh0, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex10, ex11, r3
	vlda bmhh0, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		vst bmhl0, [p4], #64;		nopx;		mov r24, #0;		vmac.f dm3, dm3, ex4, ex5, r3
	vlda.fill.512 [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		nopv
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex8, ex9, r3
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex6, ex7, r3
	vlda bmll1, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		vst bmhh0, [p4], #64;		nopx;		nopm;		nopv
	vlda bmlh1, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmll1, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex10, ex11, r3
	vlda bmhl1, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		vst bmlh1, [p4], #64;		nopx;		nopm;		vmac.f dm3, dm3, ex8, ex9, r3
	vlda bmhh1, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmhl1, [p4], #64;		nopx;		nopm;		nopv
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		nopv
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex10, ex11, r3
	vlda bmll2, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		vst bmhh1, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex0, ex5, r3
	vlda bmlh2, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmll2, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex0, ex5, r3
	vlda bmhl2, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmlh2, [p4], #64;		nopx;		nopm;		nopv
	vlda bmhh2, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmhl2, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex1, ex7, r3
	vlda bmll3, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		vst bmhh2, [p4], #64;		nopx;		nopm;		nopv
	vlda bmlh3, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		vst bmll3, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex1, ex7, r3
	vlda bmhl3, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmlh3, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex2, ex9, r3
	vlda bmhh3, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmhl3, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex0, ex5, r3
	.p2align	4
.Lmm_rot_end:
mm_rot_last:
	nopa;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmhh3, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex2, ex9, r3
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_mm_rot:
	.size	mm_rot, .Lfunc_end_mm_rot-mm_rot
