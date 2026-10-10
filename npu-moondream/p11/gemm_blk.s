	.text
	.globl	mm_tile_asm
	.p2align	4
	.type	mm_tile_asm,@function
mm_tile_asm:
	mova r3, #780;		movs p3, p2;		mov r24, #0
	mov r25, #0;		movs p4, p2
	movs p5, p1
	vlda bmll0, [p3], #64
	vlda bmlh0, [p3], #64
	vlda bmhl0, [p3], #64
	vlda bmhh0, [p3], #64
	vlda bmll1, [p3], #64
	vlda bmlh1, [p3], #64
	vlda bmhl1, [p3], #64
	vlda bmhh1, [p3], #64
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda.pop.576 ex2, [p0, lf0, r24]
	vlda.pop.576 ex3, [p0, lf0, r24];		vldb.pop.576 ex9, [p1, lf1, r25]
	vlda.pop.576 ex4, [p0, lf0, r24];		vldb.pop.576 ex11, [p1, lf1, r25]
	vlda.pop.576 ex6, [p0, lf0, r24]
	vlda.pop.576 ex8, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.pop.576 ex10, [p0, lf0, r24];		vldb.pop.576 ex7, [p1, lf1, r25]
	vlda bmll2, [p3], #64;		vmac.f dm0, dm0, ex0, ex5, r3
	vlda bmlh2, [p3], #64;		vmac.f dm1, dm1, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vlda bmhl2, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vlda bmhh2, [p3], #64;		vmac.f dm0, dm0, ex1, ex9, r3
	vlda bmll3, [p3], #64;		vmac.f dm1, dm1, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vlda bmlh3, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda bmhl3, [p3], #64;		vmac.f dm0, dm0, ex2, ex5, r3
	vlda bmhh3, [p3], #64;		vmac.f dm1, dm1, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm0, dm0, ex3, ex9, r3
	vmac.f dm1, dm1, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm1, ex4, ex7, r3
	vmac.f dm0, dm0, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vldb.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm1, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm0, dm0, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	nop
	vmac.f dm1, dm1, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm0, dm0, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	nop
	vmac.f dm1, dm1, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm2, dm2, ex0, ex5, r3
	vmac.f dm3, dm3, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm2, ex1, ex9, r3
	vst bmll0, [p4], #64;		vmac.f dm3, dm3, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh0, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl0, [p4], #64;		vmac.f dm2, dm2, ex2, ex5, r3
	vst bmhh0, [p4], #64;		vmac.f dm3, dm3, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll1, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmlh1, [p4], #64;		vmac.f dm2, dm2, ex3, ex9, r3
	vst bmhl1, [p4], #64;		vmac.f dm3, dm3, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh1, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm2, dm2, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm3, dm3, ex4, ex7, r3
	vmac.f dm2, dm2, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmll0, [p3], #64
	vldb.fill.512 [p1, lf1, r25];		vlda bmlh0, [p3], #64
	vmac.f dm3, dm3, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl0, [p3], #64
	vmac.f dm2, dm2, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh0, [p3], #64
	vlda bmll1, [p3], #64
	vmac.f dm3, dm3, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh1, [p3], #64
	vmac.f dm2, dm2, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl1, [p3], #64
	vlda bmhh1, [p3], #64
	vmac.f dm3, dm3, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm0, ex0, ex5, r3
	vmac.f dm1, dm1, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm0, dm0, ex1, ex9, r3
	vst bmll2, [p4], #64;		vmac.f dm1, dm1, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh2, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl2, [p4], #64;		vmac.f dm0, dm0, ex2, ex5, r3
	vst bmhh2, [p4], #64;		vmac.f dm1, dm1, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll3, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmlh3, [p4], #64;		vmac.f dm0, dm0, ex3, ex9, r3
	vst bmhl3, [p4], #64;		vmac.f dm1, dm1, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh3, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm1, ex4, ex7, r3
	vmac.f dm0, dm0, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmll2, [p3], #64
	vldb.fill.512 [p1, lf1, r25];		vlda bmlh2, [p3], #64
	vmac.f dm1, dm1, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl2, [p3], #64
	vmac.f dm0, dm0, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh2, [p3], #64
	vlda bmll3, [p3], #64
	vmac.f dm1, dm1, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh3, [p3], #64
	vmac.f dm0, dm0, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl3, [p3], #64
	vlda bmhh3, [p3], #64
	vmac.f dm1, dm1, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm2, dm2, ex0, ex5, r3
	vmac.f dm3, dm3, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm2, ex1, ex9, r3
	vst bmll0, [p4], #64;		vmac.f dm3, dm3, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh0, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl0, [p4], #64;		vmac.f dm2, dm2, ex2, ex5, r3
	vst bmhh0, [p4], #64;		vmac.f dm3, dm3, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll1, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmlh1, [p4], #64;		vmac.f dm2, dm2, ex3, ex9, r3
	vst bmhl1, [p4], #64;		vmac.f dm3, dm3, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh1, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm2, dm2, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm3, dm3, ex4, ex7, r3
	vmac.f dm2, dm2, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmll0, [p3], #64
	vldb.fill.512 [p1, lf1, r25];		vlda bmlh0, [p3], #64
	vmac.f dm3, dm3, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl0, [p3], #64
	vmac.f dm2, dm2, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh0, [p3], #64
	vlda bmll1, [p3], #64
	vmac.f dm3, dm3, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh1, [p3], #64
	vmac.f dm2, dm2, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl1, [p3], #64
	vlda bmhh1, [p3], #64
	vmac.f dm3, dm3, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm0, ex0, ex5, r3
	vmac.f dm1, dm1, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm0, dm0, ex1, ex9, r3
	vst bmll2, [p4], #64;		vmac.f dm1, dm1, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh2, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl2, [p4], #64;		vmac.f dm0, dm0, ex2, ex5, r3
	vst bmhh2, [p4], #64;		vmac.f dm1, dm1, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll3, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmlh3, [p4], #64;		vmac.f dm0, dm0, ex3, ex9, r3
	vst bmhl3, [p4], #64;		vmac.f dm1, dm1, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh3, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm1, ex4, ex7, r3
	vmac.f dm0, dm0, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmll2, [p3], #64
	vldb.fill.512 [p1, lf1, r25];		vlda bmlh2, [p3], #64
	vmac.f dm1, dm1, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl2, [p3], #64
	vmac.f dm0, dm0, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh2, [p3], #64
	vlda bmll3, [p3], #64
	vmac.f dm1, dm1, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh3, [p3], #64
	vmac.f dm0, dm0, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl3, [p3], #64
	vlda bmhh3, [p3], #64;		mov r24, #0
	vmac.f dm1, dm1, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex7, [p1, lf1, r25];		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm2, ex0, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm3, dm3, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda.pop.576 ex2, [p0, lf0, r24]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm2, ex1, ex9, r3;		vlda.pop.576 ex3, [p0, lf0, r24]
	vst bmll0, [p4], #64;		vmac.f dm3, dm3, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh0, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl0, [p4], #64;		vmac.f dm2, dm2, ex2, ex5, r3
	vst bmhh0, [p4], #64;		vmac.f dm3, dm3, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll1, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda.pop.576 ex4, [p0, lf0, r24]
	vst bmlh1, [p4], #64;		vmac.f dm2, dm2, ex3, ex9, r3
	vst bmhl1, [p4], #64;		vmac.f dm3, dm3, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh1, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda.pop.576 ex6, [p0, lf0, r24]
	nop
	vmac.f dm2, dm2, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25];		vlda.pop.576 ex8, [p0, lf0, r24]
	vmac.f dm3, dm3, ex4, ex7, r3
	vmac.f dm2, dm2, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		mov r25, #0
	vlda.pop.576 ex10, [p0, lf0, r24];		mov p1, p5
	vmac.f dm3, dm3, ex6, ex11, r3;		vlda bmll0, [p3], #64
	vmac.f dm2, dm2, ex8, ex5, r3;		vlda bmlh0, [p3], #64;		vldb.fill.512 [p1, lf1, r25]
	vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl0, [p3], #64
	vmac.f dm3, dm3, ex8, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh0, [p3], #64
	vmac.f dm2, dm2, ex10, ex9, r3;		vlda bmll1, [p3], #64
	vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh1, [p3], #64
	vmac.f dm3, dm3, ex10, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl1, [p3], #64
	vlda bmhh1, [p3], #64
	vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm0, ex0, ex5, r3
	vmac.f dm1, dm1, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmll2, [p4], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmlh2, [p4], #64;		vmac.f dm0, dm0, ex1, ex9, r3
	vst bmhl2, [p4], #64;		vmac.f dm1, dm1, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmhh2, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmll3, [p4], #64;		vmac.f dm0, dm0, ex2, ex5, r3
	vst bmlh3, [p4], #64;		vmac.f dm1, dm1, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmhl3, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmhh3, [p4], #64;		vmac.f dm0, dm0, ex3, ex9, r3
	vmac.f dm1, dm1, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm1, ex4, ex7, r3
	vmac.f dm0, dm0, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmll2, [p3], #64
	vldb.fill.512 [p1, lf1, r25];		vlda bmlh2, [p3], #64
	vmac.f dm1, dm1, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl2, [p3], #64
	vmac.f dm0, dm0, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh2, [p3], #64
	vlda bmll3, [p3], #64
	vmac.f dm1, dm1, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh3, [p3], #64
	vmac.f dm0, dm0, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl3, [p3], #64
	vlda bmhh3, [p3], #64
	vmac.f dm1, dm1, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm2, dm2, ex0, ex5, r3
	vmac.f dm3, dm3, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm2, ex1, ex9, r3
	vst bmll0, [p4], #64;		vmac.f dm3, dm3, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh0, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl0, [p4], #64;		vmac.f dm2, dm2, ex2, ex5, r3
	vst bmhh0, [p4], #64;		vmac.f dm3, dm3, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll1, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmlh1, [p4], #64;		vmac.f dm2, dm2, ex3, ex9, r3
	vst bmhl1, [p4], #64;		vmac.f dm3, dm3, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh1, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm2, dm2, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm3, dm3, ex4, ex7, r3
	vmac.f dm2, dm2, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmll0, [p3], #64
	vldb.fill.512 [p1, lf1, r25];		vlda bmlh0, [p3], #64
	vmac.f dm3, dm3, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl0, [p3], #64
	vmac.f dm2, dm2, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh0, [p3], #64
	vlda bmll1, [p3], #64
	vmac.f dm3, dm3, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh1, [p3], #64
	vmac.f dm2, dm2, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl1, [p3], #64
	vlda bmhh1, [p3], #64
	vmac.f dm3, dm3, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm0, ex0, ex5, r3
	vmac.f dm1, dm1, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm0, dm0, ex1, ex9, r3
	vst bmll2, [p4], #64;		vmac.f dm1, dm1, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh2, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl2, [p4], #64;		vmac.f dm0, dm0, ex2, ex5, r3
	vst bmhh2, [p4], #64;		vmac.f dm1, dm1, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll3, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmlh3, [p4], #64;		vmac.f dm0, dm0, ex3, ex9, r3
	vst bmhl3, [p4], #64;		vmac.f dm1, dm1, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh3, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm1, ex4, ex7, r3
	vmac.f dm0, dm0, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmll2, [p3], #64
	vldb.fill.512 [p1, lf1, r25];		vlda bmlh2, [p3], #64
	vmac.f dm1, dm1, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl2, [p3], #64
	vmac.f dm0, dm0, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh2, [p3], #64
	vlda bmll3, [p3], #64
	vmac.f dm1, dm1, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh3, [p3], #64
	vmac.f dm0, dm0, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl3, [p3], #64
	vlda bmhh3, [p3], #64
	vmac.f dm1, dm1, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm2, dm2, ex0, ex5, r3
	vmac.f dm3, dm3, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm2, ex1, ex9, r3
	vst bmll0, [p4], #64;		vmac.f dm3, dm3, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh0, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl0, [p4], #64;		vmac.f dm2, dm2, ex2, ex5, r3
	vst bmhh0, [p4], #64;		vmac.f dm3, dm3, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll1, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmlh1, [p4], #64;		vmac.f dm2, dm2, ex3, ex9, r3
	vst bmhl1, [p4], #64;		vmac.f dm3, dm3, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh1, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm2, dm2, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm3, dm3, ex4, ex7, r3
	vmac.f dm2, dm2, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmll0, [p3], #64
	vldb.fill.512 [p1, lf1, r25];		vlda bmlh0, [p3], #64
	vmac.f dm3, dm3, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl0, [p3], #64
	vmac.f dm2, dm2, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh0, [p3], #64
	vlda bmll1, [p3], #64
	vmac.f dm3, dm3, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh1, [p3], #64
	vmac.f dm2, dm2, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl1, [p3], #64
	vlda bmhh1, [p3], #64
	vmac.f dm3, dm3, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm0, ex0, ex5, r3
	vmac.f dm1, dm1, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm0, dm0, ex1, ex9, r3
	vst bmll2, [p4], #64;		vmac.f dm1, dm1, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh2, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl2, [p4], #64;		vmac.f dm0, dm0, ex2, ex5, r3
	vst bmhh2, [p4], #64;		vmac.f dm1, dm1, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll3, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmlh3, [p4], #64;		vmac.f dm0, dm0, ex3, ex9, r3
	vst bmhl3, [p4], #64;		vmac.f dm1, dm1, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh3, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm1, ex4, ex7, r3
	vmac.f dm0, dm0, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmll2, [p3], #64
	vldb.fill.512 [p1, lf1, r25];		vlda bmlh2, [p3], #64
	vmac.f dm1, dm1, ex6, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25];		vlda bmhl2, [p3], #64
	vmac.f dm0, dm0, ex8, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25];		vlda bmhh2, [p3], #64
	vlda bmll3, [p3], #64
	vmac.f dm1, dm1, ex8, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25];		vlda bmlh3, [p3], #64
	vmac.f dm0, dm0, ex10, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25];		vlda bmhl3, [p3], #64
	vlda bmhh3, [p3], #64
	vmac.f dm1, dm1, ex10, ex11, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm2, dm2, ex0, ex5, r3
	vmac.f dm3, dm3, ex0, ex7, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm2, ex1, ex9, r3
	vst bmll0, [p4], #64;		vmac.f dm3, dm3, ex1, ex11, r3;		vldb.fill.512 [p1, lf1, r25]
	vst bmlh0, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vst bmhl0, [p4], #64;		vmac.f dm2, dm2, ex2, ex5, r3
	vst bmhh0, [p4], #64;		vmac.f dm3, dm3, ex2, ex7, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vst bmll1, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vst bmlh1, [p4], #64;		vmac.f dm2, dm2, ex3, ex9, r3
	vst bmhl1, [p4], #64;		vmac.f dm3, dm3, ex3, ex11, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vst bmhh1, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm2, dm2, ex4, ex5, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm3, dm3, ex4, ex7, r3
	vmac.f dm2, dm2, ex6, ex9, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	nop
	vmac.f dm3, dm3, ex6, ex11, r3
	vmac.f dm2, dm2, ex8, ex5, r3
	nop
	vmac.f dm3, dm3, ex8, ex7, r3
	vmac.f dm2, dm2, ex10, ex9, r3
	nop
	vmac.f dm3, dm3, ex10, ex11, r3
	nop
	nop
	nop
	nop
	nop
	vst bmll2, [p4], #64
	vst bmlh2, [p4], #64
	vst bmhl2, [p4], #64
	vst bmhh2, [p4], #64
	vst bmll3, [p4], #64
	vst bmlh3, [p4], #64
	vst bmhl3, [p4], #64
	vst bmhh3, [p4], #64
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_mm:
	.size	mm_tile_asm, .Lfunc_end_mm-mm_tile_asm
