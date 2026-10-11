	.text
	.section	.text.probe0,"ax",@progbits
	.globl	probe0
	.p2align	4
	.type	probe0,@function
probe0:
	mov	p2, p1
	mova	r26, #0
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	nop
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
	nop
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
	nop
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
	nop
	nop
	nop
	nop
	mov	p2, p1
	movxm	r1, #1024
	mov	m0, r1
	nop
	padda	[p2], m0
	mova	r26, #0
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	.section	.text.probe4,"ax",@progbits
	.globl	probe4
	.p2align	4
	.type	probe4,@function
probe4:
	mov	p2, p1
	mova	r26, #0
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
	nop
	nop
	nop
	nop
	mova	r1, #32
	mov	m0, r1
	nop
	padda	[p2], m0
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
	vst.flush.512.conv	[p2, sf, r26]
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
.Lfunc_end_probe4:
	.size	probe4, .Lfunc_end_probe4-probe4
	.section	.text.probe5,"ax",@progbits
	.globl	probe5
	.p2align	4
	.type	probe5,@function
probe5:
	mov	p2, p1
	mova	r26, #0
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml2, [p0, #0]
	vlda.conv.fp32.bf16	cmh2, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml3, [p0, #0]
	vlda.conv.fp32.bf16	cmh3, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml4, [p0, #0]
	vlda.conv.fp32.bf16	cmh4, [p0, #64]
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
	vst.flush.512.conv	[p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm2, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm3, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm4, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
	nop
	nop
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml2, [p0, #0]
	vlda.conv.fp32.bf16	cmh2, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml3, [p0, #0]
	vlda.conv.fp32.bf16	cmh3, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml4, [p0, #0]
	vlda.conv.fp32.bf16	cmh4, [p0, #64]
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
	vst.flush.512.conv	[p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm2, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm3, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm4, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
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
.Lfunc_end_probe5:
	.size	probe5, .Lfunc_end_probe5-probe5
	.section	.text.probe6,"ax",@progbits
	.globl	probe6
	.p2align	4
	.type	probe6,@function
probe6:
	mov	p2, p1
	mova	r26, #0
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml2, [p0, #0]
	vlda.conv.fp32.bf16	cmh2, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml3, [p0, #0]
	vlda.conv.fp32.bf16	cmh3, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml4, [p0, #0]
	vlda.conv.fp32.bf16	cmh4, [p0, #64]
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
	vst.push.576.conv.bfp16ebs8.fp32	dm2, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm3, [p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm4, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
	nop
	nop
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16	cml1, [p0, #0]
	vlda.conv.fp32.bf16	cmh1, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml2, [p0, #0]
	vlda.conv.fp32.bf16	cmh2, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml3, [p0, #0]
	vlda.conv.fp32.bf16	cmh3, [p0, #64]
	padda	[p0], #128
	vlda.conv.fp32.bf16	cml4, [p0, #0]
	vlda.conv.fp32.bf16	cmh4, [p0, #64]
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
	vst.push.576.conv.bfp16ebs8.fp32	dm2, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm3, [p2, sf, r26]
	vst.push.576.conv.bfp16ebs8.fp32	dm4, [p2, sf, r26]
	vst.flush.512.conv	[p2, sf, r26]
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
.Lfunc_end_probe6:
	.size	probe6, .Lfunc_end_probe6-probe6
	.section	.text.probe7,"ax",@progbits
	.globl	probe7
	.p2align	4
	.type	probe7,@function
probe7:
	mov	p2, p1
	mova	r26, #0
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
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
	nop
	nop
	nop
	vst.flush.512.conv	[p2, sf, r26]
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
.Lfunc_end_probe7:
	.size	probe7, .Lfunc_end_probe7-probe7
