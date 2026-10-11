	.text
	.globl	mm_e7k_nl
	.p2align	4
	.type	mm_e7k_nl,@function
mm_e7k_nl:
	movs p5, p2
	paddxm [sp], #64
	nop
	st r8, [sp, #-64]
	st r9, [sp, #-60]
	st r10, [sp, #-56]
	st r11, [sp, #-52]
	mova r8, #0
	mova r9, #0
	mova r10, #0
	mova r11, #0
	mov r6, r0
	mov r27, r1
	mov r16, r2
	mov r19, r3
	movxm r20, #1728
	movxm r21, #2880
	nop
	add r17, r16, r20
	add r18, r16, r21
	movxm r20, #12416
	movxm r21, #16256
	movxm r22, #12416
	movxm r23, #1600
	movs p3, p1
	movs p4, p1
	movs p2, p0
	mova r3, #780
	mova r25, #0
	mova r24, #0
	mova r4, #-1
	mova r5, #1
	movxm m2, #-428
	mova dn2, #107
	mova dj2, #4
	mova dc2, #0
	movxm ls, #.Lmm_e7k_nl_body
	movxm le, #.Lmm_e7k_nl_end
	add.nc lc, r27, #0
	.p2align	4
.Lmm_e7k_nl_pro:
	lda r26, [p2], #4
	lda m0, [p2], #4
	nop
	nop
	nop
	nop
	nop
	nop
	lda p0, [p2], #4;		mul r26, r26, r27
	padds [p3], m0
	add r8, r8, r26
	nop
	vlda bmll0, [p3], #64
	vlda bmlh0, [p3], #64
	vlda bmhl0, [p3], #64
	vlda bmhh0, [p3], #64;		mov r24, #0
	lda m1, [p2], #4;		vldb.fill.512 [p0, lf0, r24];		movs p1, r18
	vlda bmll1, [p3], #64;		mov r25, #0
	vlda.fill.512 [p1, lf1, r25];		vldb.pop.576 ex0, [p0, lf0, r24]
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda bmlh1, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda bmhl1, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vlda bmhh1, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex9, [p1, lf1, r25]
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex9, [p1, lf1, r25]
	vlda bmll2, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		vmac.f dm0, dm0, ex0, ex5, r6
	vlda.pop.576 ex8, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		vmac.f dm1, dm1, ex0, ex5, r6
	vlda bmlh2, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vlda bmhl2, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		vmac.f dm0, dm0, ex1, ex7, r3
	vlda bmhh2, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vmac.f dm1, dm1, ex1, ex7, r3
	vlda.pop.576 ex10, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		mov r24, #0
	vlda bmll3, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		movs p0, r19;		nopx;		nopm;		vmac.f dm0, dm0, ex2, ex9, r3
	vlda bmlh3, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm1, dm1, ex2, ex9, r3
	vlda.fill.512 [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		mov r25, #0;		vmac.f dm2, dm2, ex0, ex5, r6
	vldb.pop.576 ex11, [p0, lf0, r24];		movs p1, r17;		vmac.f dm0, dm0, ex3, ex11, r3
	vldb.pop.576 ex7, [p0, lf0, r24];		vmac.f dm1, dm1, ex3, ex11, r3
	vlda bmhl3, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		add r30, r27, #-1;		vmac.f dm2, dm2, ex1, ex7, r3
	vlda bmhh3, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vmac.f dm0, dm0, ex4, ex5, r3
	vldb.pop.576 ex11, [p1, lf1, r25];		vmac.f dm1, dm1, ex4, ex5, r3
	vldb.pop.576 ex9, [p1, lf1, r25];		vmac.f dm2, dm2, ex2, ex9, r3
	.p2align	4
.Lmm_e7k_nl_body:
mm_e7k_nl_start:
	nopa;		vldb.pop.576 ex5, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex6, ex7, r3
	vlda.pop.576 ex9, [p0, lf0, r24];		vldb.pop.576 ex11, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex0, ex5, r6
	nopa;		vldb.pop.576 ex9, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex3, ex11, r3
	vlda.pop.576 ex11, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex8, ex9, r3
	vlda.pop.576 ex5, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		padds [p3], m1;		nopx;		nopm;		vmac.f dm3, dm3, ex1, ex7, r3
	nopa;		vldb.pop.576 ex11, [p0, lf0, r24];		padds [p4], m0;		nopx;		mov r24, #0;		vmac.f dm1, dm1, ex6, ex7, r3
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		movs p0, r16;		nopx;		nopm;		vmac.f dm0, dm0, ex10, ex11, r3
	vlda bmll0, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex2, ex9, r3
	vlda bmlh0, [p3], #64;		vldb.fill.512 [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex4, ex5, r3
	vlda bmhl0, [p3], #64;		vldb.pop.576 ex7, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex8, ex9, r3
	vlda bmhh0, [p3], #64;		vldb.pop.576 ex7, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex3, ex11, r3
	lda.2d r26, [p2], d2;		vldb.pop.576 ex11, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex6, ex7, r3
	nopa;		vldb.pop.576 ex9, [p0, lf0, r24];		vst bmll0, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex10, ex11, r3
	vlda bmll1, [p3], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		vst bmlh0, [p4], #64;		nopx;		nopm;		vmac.f dm3, dm3, ex4, ex5, r3
	vlda bmlh1, [p3], #64;		vldb.pop.576 ex5, [p0, lf0, r24];		vst bmhl0, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex0, ex5, r6
	vlda bmhl1, [p3], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		vst bmhh0, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex8, ex9, r3
	vlda bmhh1, [p3], #64;		vldb.pop.576 ex11, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex6, ex7, r3
	lda.2d r29, [p2], d2;		vldb.fill.512 [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex1, ex7, r3
	nopa;		vldb.pop.576 ex7, [p0, lf0, r24];		vst bmll1, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex10, ex11, r3
	nopa;		vldb.pop.576 ex5, [p0, lf0, r24];		vst bmlh1, [p4], #64;		nopx;		nopm;		nopv
	vlda bmll2, [p3], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		vst bmhl1, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex2, ex9, r3
	nopa;		vldb.pop.576 ex5, [p0, lf0, r24];		vst bmhh1, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex0, ex5, r6
	vlda bmlh2, [p3], #64;		vldb.pop.576 ex11, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex8, ex9, r3
	vlda bmhl2, [p3], #64;		vldb.pop.576 ex7, [p0, lf0, r24];		nops;		add r9, r9, r26;		nopm;		vmac.f dm0, dm0, ex3, ex11, r3
	vlda bmhh2, [p3], #64;		vldb.pop.576 ex11, [p0, lf0, r24];		vst bmll2, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex1, ex7, r3
	nopa;		vldb.pop.576 ex9, [p0, lf0, r24];		vst bmlh2, [p4], #64;		mul r26, r29, r30;		nopm;		vmac.f dm3, dm3, ex10, ex11, r3
	nopa;		vldb.fill.512 [p0, lf0, r24];		vst bmhl2, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex4, ex5, r3
	vlda bmll3, [p3], #64;		vldb.pop.576 ex5, [p0, lf0, r24];		vst bmhh2, [p4], #64;		add r10, r10, r26;		nopm;		vmac.f dm1, dm1, ex2, ex9, r3
	lda.2d r26, [p2], d2;		vldb.pop.576 ex7, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex0, ex5, r6
	lda.2d m0, [p2], d2;		vldb.fill.512 [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm0, dm0, ex6, ex7, r3
	vlda bmlh3, [p3], #64;		vldb.pop.576 ex11, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex3, ex11, r3
	vlda bmhl3, [p3], #64;		vldb.pop.576 ex7, [p0, lf0, r24];		vst bmll3, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex1, ex7, r3
	vlda bmhh3, [p3], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		nops;		add r27, r27, #-1;		nopm;		vmac.f dm0, dm0, ex8, ex9, r3
	nopa;		vldb.pop.576 ex5, [p0, lf0, r24];		vst bmlh3, [p4], #64;		nopx;		nopm;		vmac.f dm1, dm1, ex4, ex5, r3
	nopa;		vldb.pop.576 ex9, [p0, lf0, r24];		vst bmhl3, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex2, ex9, r3
	nopa;		vldb.pop.576 ex11, [p0, lf0, r24];		vst bmhh3, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex10, ex11, r3
	lda.2d p0, [p2], d2;		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		mul r26, r26, r27;		nopm;		vmac.f dm3, dm3, ex0, ex5, r6
	nopa;		vldb.pop.576 ex11, [p1, lf1, r25];		padds [p3], m0;		nopx;		nopm;		vmac.f dm1, dm1, ex6, ex7, r3
	nopa;		vldb.pop.576 ex5, [p1, lf1, r25];		padds [p4], m1;		add r8, r8, r26;		nopm;		vmac.f dm2, dm2, ex3, ex11, r3
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex1, ex7, r3
	vlda bmll0, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex8, ex9, r3
	vlda bmlh0, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmll0, [p4], #64;		mul r7, r29, r22;		nopm;		vmac.f dm2, dm2, ex4, ex5, r3
	vlda bmhl0, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmlh0, [p4], #64;		nopx;		nopm;		vmac.f dm3, dm3, ex2, ex9, r3
	vlda bmhh0, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmhl0, [p4], #64;		xor r18, r18, r7;		mov r24, #0;		vmac.f dm1, dm1, ex10, ex11, r3
	lda.2d m1, [p2], d2;		vldb.fill.512 [p0, lf0, r24];		movs p1, r18;		nopx;		nopm;		vmac.f dm2, dm2, ex6, ex7, r3
	vlda bmll1, [p3], #64;		nopb;		vst bmhh0, [p4], #64;		nopx;		mov r25, #0;		vmac.f dm3, dm3, ex3, ex11, r3
	vlda.fill.512 [p1, lf1, r25];		vldb.pop.576 ex0, [p0, lf0, r24];		nops;		nopx;		nopm;		nopv
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm2, dm2, ex8, ex9, r3
	vlda.pop.576 ex2, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex4, ex5, r3
	vlda bmlh1, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmll1, [p4], #64;		nopx;		nopm;		nopv
	vlda bmhl1, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmlh1, [p4], #64;		nopx;		nopm;		vmac.f dm2, dm2, ex10, ex11, r3
	vlda bmhh1, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmhl1, [p4], #64;		xor r31, r29, r4;		nopm;		vmac.f dm3, dm3, ex6, ex7, r3
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		nopv
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		and r6, r6, r31;		nopm;		nopv
	vlda.pop.576 ex6, [p0, lf0, r24];		vldb.pop.576 ex9, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm3, dm3, ex8, ex9, r3
	vlda bmll2, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		vst bmhh1, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex0, ex5, r6
	vlda.pop.576 ex8, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex0, ex5, r6
	vlda bmlh2, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmll2, [p4], #64;		mul r1, r29, r23;		nopm;		vmac.f dm3, dm3, ex10, ex11, r3
	vlda bmhl2, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		vst bmlh2, [p4], #64;		nopx;		nopm;		vmac.f dm0, dm0, ex1, ex7, r3
	vlda bmhh2, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmhl2, [p4], #64;		xor r19, r19, r1;		nopm;		vmac.f dm1, dm1, ex1, ex7, r3
	vlda.pop.576 ex10, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		mul r0, r29, r21;		mov r24, #0;		nopv
	vlda bmll3, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		movs p0, r19;		nopx;		nopm;		vmac.f dm0, dm0, ex2, ex9, r3
	vlda bmlh3, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmhh2, [p4], #64;		xor r17, r17, r0;		nopm;		vmac.f dm1, dm1, ex2, ex9, r3
	vlda.fill.512 [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25];		nops;		nopx;		mov r25, #0;		vmac.f dm2, dm2, ex0, ex5, r6
	nopa;		vldb.pop.576 ex11, [p0, lf0, r24];		movs p1, r17;		nopx;		nopm;		vmac.f dm0, dm0, ex3, ex11, r3
	nopa;		vldb.pop.576 ex7, [p0, lf0, r24];		nops;		nopx;		nopm;		vmac.f dm1, dm1, ex3, ex11, r3
	vlda bmhl3, [p3], #64;		vldb.fill.512 [p1, lf1, r25];		vst bmll3, [p4], #64;		add r30, r27, #-1;		mov r26, r29;		vmac.f dm2, dm2, ex1, ex7, r3
	vlda bmhh3, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25];		vst bmlh3, [p4], #64;		mul r2, r29, r20;		nopm;		vmac.f dm0, dm0, ex4, ex5, r3
	nopa;		vldb.pop.576 ex11, [p1, lf1, r25];		vst bmhl3, [p4], #64;		add r11, r11, r26;		nopm;		vmac.f dm1, dm1, ex4, ex5, r3
	.p2align	4
.Lmm_e7k_nl_end:
mm_e7k_nl_last:
	nopa;		vldb.pop.576 ex9, [p1, lf1, r25];		vst bmhh3, [p4], #64;		xor r16, r16, r2;		nopm;		vmac.f dm2, dm2, ex2, ex9, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	st r8, [p5], #4
	st r9, [p5], #4
	st r10, [p5], #4
	st r11, [p5], #4
	st r16, [p5], #4
	st r17, [p5], #4
	st r18, [p5], #4
	st r19, [p5], #4
	st r27, [p5], #4
	st r6, [p5], #4
	mov r0, p2
	nop
	st r0, [p5], #4
	lda r8, [sp, #-64]
	lda r9, [sp, #-60]
	lda r10, [sp, #-56]
	lda r11, [sp, #-52]
	paddxm [sp], #-64
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_mm_e7k_nl:
	.size	mm_e7k_nl, .Lfunc_end_mm_e7k_nl-mm_e7k_nl
