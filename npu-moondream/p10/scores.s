	.text
	.globl	attn_scores_asm
	.p2align	4
	.type	attn_scores_asm,@function
attn_scores_asm:
	mova r3, #780;		mov r4, #828
	mova r16, #0;		movs p4, p0
	movs p5, p2
	padda [p5], #448
	movs p0, p4;		mov r24, #0
	mov r25, #0
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	nop
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmul.f dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		movs p0, p4;		mov r24, #0
	mov r25, #0
	vmac.f dm0, dm0, ex1, ex5, r3;		vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vmac.f dm0, dm0, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda x6, [p3, #0]
	vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vlda x7, [p3, #64]
	vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	nop
	vmul.f dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vconv.bf16.fp32 x8, cml0;		vmac.f dm1, dm1, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vconv.bf16.fp32 x9, cmh0
	vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vmul.f dm2, y4, y3, r4
	vmac.f dm1, dm1, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		movs p0, p4;		mov r24, #0
	vlda x10, [p3, #128];		mov r25, #0
	vmac.f dm1, dm1, ex1, ex5, r3;		vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vconv.bf16.fp32 x8, cml2;		vlda x11, [p3, #192]
	vconv.bf16.fp32 x9, cmh2;		vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vst x8, [p2, #0]
	vst x9, [p2, #64];		vmac.f dm1, dm1, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda x6, [p3, #0]
	vmax_lt.bf16 x10, r16, x10, x8;		vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vmax_lt.bf16 x11, r16, x11, x9;		vlda x7, [p3, #64]
	vst x10, [p3, #128];		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vst x11, [p3, #192]
	vmul.f dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vconv.bf16.fp32 x8, cml1;		vmac.f dm0, dm0, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vconv.bf16.fp32 x9, cmh1
	vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vmul.f dm3, y4, y3, r4
	vmac.f dm0, dm0, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		movs p0, p4;		mov r24, #0
	vlda x10, [p3, #128];		mov r25, #0
	vmac.f dm0, dm0, ex1, ex5, r3;		vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vconv.bf16.fp32 x8, cml3;		vlda x11, [p3, #192]
	vconv.bf16.fp32 x9, cmh3;		vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vst x8, [p2, #128]
	vst x9, [p2, #192];		vmac.f dm0, dm0, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda x6, [p3, #0]
	vmax_lt.bf16 x10, r16, x10, x8;		vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vmax_lt.bf16 x11, r16, x11, x9;		vlda x7, [p3, #64]
	vst x10, [p3, #128];		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vst x11, [p3, #192]
	vmul.f dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vconv.bf16.fp32 x8, cml0;		vmac.f dm1, dm1, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vconv.bf16.fp32 x9, cmh0
	vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vmul.f dm2, y4, y3, r4
	vmac.f dm1, dm1, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		movs p0, p4;		mov r24, #0
	vlda x10, [p3, #128];		mov r25, #0
	vmac.f dm1, dm1, ex1, ex5, r3;		vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vconv.bf16.fp32 x8, cml2;		vlda x11, [p3, #192]
	vconv.bf16.fp32 x9, cmh2;		vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vst x8, [p2, #256]
	vst x9, [p2, #320];		vmac.f dm1, dm1, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda x6, [p3, #0]
	vmax_lt.bf16 x10, r16, x10, x8;		vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vmax_lt.bf16 x11, r16, x11, x9;		vlda x7, [p3, #64]
	vst x10, [p3, #128];		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vst x11, [p3, #192]
	vmul.f dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vconv.bf16.fp32 x8, cml1;		vmac.f dm0, dm0, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vconv.bf16.fp32 x9, cmh1
	vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vmul.f dm3, y4, y3, r4
	vmac.f dm0, dm0, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25];		movs p0, p4;		mov r24, #0
	vlda x10, [p3, #128];		mov r25, #0
	vmac.f dm0, dm0, ex1, ex5, r3;		vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vconv.bf16.fp32 x8, cml3;		vlda x11, [p3, #192]
	vconv.bf16.fp32 x9, cmh3;		vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vst x8, [p2, #384]
	vst x9, [p2, #448];		vmac.f dm0, dm0, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda x6, [p3, #0]
	vmax_lt.bf16 x10, r16, x10, x8;		vmac.f dm0, dm0, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vmax_lt.bf16 x11, r16, x11, x9;		vlda x7, [p3, #64]
	vst x10, [p3, #128];		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vst x11, [p3, #192]
	vmul.f dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	nop
	vconv.bf16.fp32 x8, cml0;		vmac.f dm1, dm1, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vconv.bf16.fp32 x9, cmh0
	vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vmul.f dm2, y4, y3, r4
	vmac.f dm1, dm1, ex1, ex5, r3;		vlda.pop.576 ex1, [p0, lf0, r24];		vldb.pop.576 ex5, [p1, lf1, r25]
	vlda.fill.512 [p0, lf0, r24];		vldb.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm1, ex0, ex4, r3;		vlda.pop.576 ex0, [p0, lf0, r24];		vldb.pop.576 ex4, [p1, lf1, r25]
	vlda x10, [p3, #128]
	vmac.f dm1, dm1, ex1, ex5, r3
	vconv.bf16.fp32 x8, cml2;		vlda x11, [p3, #192]
	vconv.bf16.fp32 x9, cmh2;		vmac.f dm1, dm1, ex0, ex4, r3
	vst x8, [p5, #64]
	vst x9, [p5, #128];		vmac.f dm1, dm1, ex1, ex5, r3
	vlda x6, [p3, #0]
	vmax_lt.bf16 x10, r16, x10, x8;		vmac.f dm1, dm1, ex0, ex4, r3
	vmax_lt.bf16 x11, r16, x11, x9;		vlda x7, [p3, #64]
	vst x10, [p3, #128]
	vst x11, [p3, #192]
	nop
	nop
	vconv.bf16.fp32 x8, cml1
	vconv.bf16.fp32 x9, cmh1
	nop
	vmul.f dm3, y4, y3, r4
	nop
	nop
	nop
	vlda x10, [p3, #128]
	nop
	vconv.bf16.fp32 x8, cml3;		vlda x11, [p3, #192]
	vconv.bf16.fp32 x9, cmh3
	vst x8, [p5, #192]
	vst x9, [p5, #256]
	nop
	vmax_lt.bf16 x10, r16, x10, x8
	vmax_lt.bf16 x11, r16, x11, x9
	vst x10, [p3, #128]
	vst x11, [p3, #192]
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_scores:
	.size	attn_scores_asm, .Lfunc_end_scores-attn_scores_asm
