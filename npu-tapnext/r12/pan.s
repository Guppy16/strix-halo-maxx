	.text
	.section	.text.pan,"ax",@progbits
	.globl	pan
	.p2align	4
	.type	pan,@function
pan:
	mov	m4, r1
	mov	dn4, r2
	mova	dj4, #0
	mova	dc4, #0
	mova	dn0, #7
	mova	dj0, #64
	movxm	m0, #576
	mova	dc0, #0
	mova	dn1, #7
	mova	dj1, #64
	movxm	m1, #576
	mova	dc1, #0
	mova	r26, #0
	movxm	ls, #pan_start
	movxm	le, #pan_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 16, U 1, stages 2, ops 25
	vlda.2d.conv.fp32.bf16 cml0, [p0], d0
	nop
	nop
	nop
	nop
	nop
	vlda.2d.conv.fp32.bf16 cmh0, [p0], d0
	vlda.2d.conv.fp32.bf16 cml1, [p0], d0
	vlda.2d.conv.fp32.bf16 cmh1, [p0], d0
	vlda.2d.conv.fp32.bf16 cml2, [p0], d0
	vlda.2d.conv.fp32.bf16 cmh2, [p0], d0
	vlda.2d.conv.fp32.bf16 cml0, [p0], d0
	vlda.2d.conv.fp32.bf16 cml1, [p1], d1
	vlda.2d.conv.fp32.bf16 cmh0, [p0], d0
	vlda.2d.conv.fp32.bf16 cmh1, [p1], d1
	vlda.2d.conv.fp32.bf16 cml2, [p1], d1
	.p2align	4
pan_start:
	vlda.2d.conv.fp32.bf16 cml0, [p0], d0;		nopb;		vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26];		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cmh2, [p1], d1;		nopb;		vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26];		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cml1, [p1], d1;		nopb;		vst.push.576.conv.bfp16ebs8.fp32 dm2, [p2, sf, r26];		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cmh1, [p1], d1;		nopb;		nops;		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cml2, [p1], d1;		nopb;		vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26];		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cmh2, [p1], d1;		nopb;		vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26];		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cmh0, [p0], d0;		nopb;		nops;		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cml1, [p0], d0;		nopb;		nops;		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cmh1, [p0], d0;		nopb;		vst.push.576.conv.bfp16ebs8.fp32 dm2, [p2, sf, r26];		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cml2, [p0], d0;		nopb;		nops;		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cmh2, [p0], d0;		nopb;		vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26];		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cml0, [p0], d0;		nopb;		nops;		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cml1, [p1], d1;		nopb;		vst.push.576.conv.bfp16ebs8.fp32 dm2, [p2, sf, r26];		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cmh0, [p0], d0;		nopb;		vst.flush.512.conv.2d [p2, sf, r26, d4];		nopx;		nopm;		nopv
	vlda.2d.conv.fp32.bf16 cmh1, [p1], d1;		nopb;		nops;		nopx;		nopm;		nopv
pan_last:
	vlda.2d.conv.fp32.bf16 cml2, [p1], d1;		nopb;		nops;		nopx;		nopm;		nopv
	nop
	nop
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
.Lfunc_end_pan:
	.size	pan, .Lfunc_end_pan-pan
