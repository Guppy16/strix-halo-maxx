	.text
	.section	.text.probe0,"ax",@progbits
	.globl	probe0
	.p2align	4
	.type	probe0,@function
probe0:
	mov	p2, p1
	mova	r26, #0
	mova	dn0, #3
	mova	dj0, #0
	movxm	m0, #32
	mova	dc0, #0
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
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
.Lfunc_end_probe0:
	.size	probe0, .Lfunc_end_probe0-probe0
	.section	.text.probe1,"ax",@progbits
	.globl	probe1
	.p2align	4
	.type	probe1,@function
probe1:
	mov	p2, p1
	mova	r26, #0
	mova	dn0, #3
	mova	dj0, #0
	movxm	m0, #288
	mova	dc0, #0
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
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
.Lfunc_end_probe1:
	.size	probe1, .Lfunc_end_probe1-probe1
	.section	.text.probe2,"ax",@progbits
	.globl	probe2
	.p2align	4
	.type	probe2,@function
probe2:
	mov	p2, p1
	mova	r26, #0
	mova	dn0, #3
	mova	dj0, #0
	movxm	m0, #0
	mova	dc0, #0
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
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
.Lfunc_end_probe2:
	.size	probe2, .Lfunc_end_probe2-probe2
	.section	.text.probe3,"ax",@progbits
	.globl	probe3
	.p2align	4
	.type	probe3,@function
probe3:
	mov	p2, p1
	mova	r26, #0
	mova	dn0, #1
	mova	dj0, #0
	movxm	m0, #64
	mova	dc0, #0
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.push.576.conv.bfp16ebs8.fp32	dm1, [p2, sf, r26]
	vst.flush.512.conv.2d	[p2, sf, r26, d0]
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
.Lfunc_end_probe3:
	.size	probe3, .Lfunc_end_probe3-probe3
