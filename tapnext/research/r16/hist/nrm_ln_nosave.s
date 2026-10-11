// RECONSTRUCTED (research rev 11): the deployed nrm_ln.s with its p6 / p7 save / restore removed (the round-15 rule)
	.text
	.section	.text.nrm_ln,"ax",@progbits
	.globl	nrm_ln
	.p2align	4
	.type	nrm_ln,@function
nrm_ln:
	vlda	x10, [p3, #0]
	vlda	x9, [p3, #64]
	vlda	x8, [p3, #128]
	vlda	x7, [p3, #192]
	paddxm	[sp], #64
	mov	p6, p0
	mov	p7, p0
	movxm	r0, #16256
	vbcst.16	x11, r0
	mova	r1, #828
	mova	r3, #16
	mova	r26, #0
	movxm	ls, #nrm_ln_start
	movxm	le, #nrm_ln_last
	mova	r0, #4
	add.nc	lc, r0, #0
	// modulo schedule: II 13, U 2, stages 4, ops 43
	vlda x3, [p0], #64
	vlda bmlh2, [p3, #320]
	vlda bmll2, [p3, #256]
	nop
	nop
	nop
	vlda x5, [p0], #64
	vlda bmll2, [p3, #384];		vmac.f dm3, dm2, x3, x10, r1
	vlda bmlh2, [p3, #448]
	nop
	vldb.128 wl1, [p4], #16;		vmac.f dm0, dm3, x3, x8, r1
	vldb.128 wl2, [p1], #16
	nop
	vlda x3, [p0], #64
	vlda bmlh2, [p3, #320];		vmac.f dm4, dm2, x5, x9, r1
	vlda bmll2, [p3, #256]
	nop
	vshift x5, x1, x1, r3;		vmac.f dm1, dm4, x5, x7, r1
	vshift x4, x2, x2, r3
	vlda x5, [p0], #64;		vshift x5, x5, x1, r3
	vlda bmll2, [p3, #384];		vmac.f dm3, dm2, x3, x10, r1
	vlda bmlh2, [p3, #448];		vshift x0, x4, x2, r3
	vconv.bf16.fp32 x4, cml0;		vshift x5, x5, x1, r3
	vldb.128 wl1, [p4], #16;		vconv.bf16.fp32 x3, cml1;		vshift x0, x0, x2, r3;		vmac.f dm0, dm3, x3, x8, r1
	vldb.128 wl2, [p1], #16;		vshift x1, x5, x1, r3;		vmsc.f dm4, dm0, x4, x11, r1
	vshift x0, x0, x2, r3;		vmsc.f dm3, dm1, x3, x11, r1
	vlda x3, [p0], #64;		vmul.f dm2, x1, x11, r1
	vlda bmlh2, [p3, #320];		vmac.f dm4, dm2, x5, x9, r1
	vlda bmll2, [p3, #256];		vmul.f dm3, x1, x11, r1
	vlda x6, [p7], #64;		vmac.f dm4, dm2, x4, x0, r1
	vconv.bf16.fp32 x6, cml4;		vshift x5, x1, x1, r3;		vmac.f dm1, dm4, x5, x7, r1
	vshift x4, x2, x2, r3;		vmac.f dm3, dm3, x3, x0, r1
	vlda x5, [p0], #64;		vconv.bf16.fp32 x6, cml3;		vshift x5, x5, x1, r3;		vmac.f dm2, dm4, x6, x0, r1
	vlda bmll2, [p3, #384];		vmac.f dm3, dm2, x3, x10, r1
	vlda bmlh2, [p3, #448];		vshift x0, x4, x2, r3;		vmac.f dm4, dm3, x6, x0, r1
	vconv.bf16.fp32 x4, cml0;		vshift x5, x5, x1, r3
	vldb.128 wl1, [p4], #16;		vconv.bf16.fp32 x3, cml1;		vshift x0, x0, x2, r3;		vmac.f dm0, dm3, x3, x8, r1
	vlda x4, [p7], #64;		vldb.128 wl2, [p1], #16;		vshift x1, x5, x1, r3;		vmsc.f dm4, dm0, x4, x11, r1
	vconv.bf16.fp32 x2, cml2;		vshift x0, x0, x2, r3;		vmsc.f dm3, dm1, x3, x11, r1
	.p2align	4
nrm_ln_start:
	vlda x3, [p0], #64;		nopb;		vst x6, [p5], #64;		nopx;		nopm;		vmul.f dm2, x1, x11, r1
	vlda bmlh2, [p3, #320];		vconv.bf16.fp32 x6, cml4;		vmac.f dm4, dm2, x5, x9, r1
	vlda bmll2, [p3, #256];		nopb;		vst x2, [p6], #64;		nopx;		vconv.fp32.bf16 cml1, x2;		vmul.f dm3, x1, x11, r1
	vlda x6, [p7], #64;		nopb;		vst x6, [p6], #64;		nopx;		vconv.fp32.bf16 cmh1, x6;		vmac.f dm4, dm2, x4, x0, r1
	vconv.bf16.fp32 x6, cml4;		vshift x5, x1, x1, r3;		vmac.f dm1, dm4, x5, x7, r1
	vst x4, [p5], #64;		vshift x4, x2, x2, r3;		vmac.f dm3, dm3, x3, x0, r1
	vlda x5, [p0], #64;		vconv.bf16.fp32 x6, cml3;		vshift x5, x5, x1, r3;		vmac.f dm2, dm4, x6, x0, r1
	vlda bmll2, [p3, #384];		vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26];		vmac.f dm3, dm2, x3, x10, r1
	vlda bmlh2, [p3, #448];		vshift x0, x4, x2, r3;		vmac.f dm4, dm3, x6, x0, r1
	vconv.bf16.fp32 x4, cml0;		vshift x5, x5, x1, r3
	vldb.128 wl1, [p4], #16;		vconv.bf16.fp32 x3, cml1;		vshift x0, x0, x2, r3;		vmac.f dm0, dm3, x3, x8, r1
	vlda x4, [p7], #64;		vldb.128 wl2, [p1], #16;		vshift x1, x5, x1, r3;		vmsc.f dm4, dm0, x4, x11, r1
	vconv.bf16.fp32 x2, cml2;		vshift x0, x0, x2, r3;		vmsc.f dm3, dm1, x3, x11, r1
	vlda x3, [p0], #64;		vst x6, [p5], #64;		vmul.f dm2, x1, x11, r1
	vlda bmlh2, [p3, #320];		vconv.bf16.fp32 x6, cml4;		vmac.f dm4, dm2, x5, x9, r1
	vlda bmll2, [p3, #256];		vst x2, [p6], #64;		vconv.fp32.bf16 cml1, x2;		vmul.f dm3, x1, x11, r1
	vlda x6, [p7], #64;		vst x6, [p6], #64;		vconv.fp32.bf16 cmh1, x6;		vmac.f dm4, dm2, x4, x0, r1
	vconv.bf16.fp32 x6, cml4;		vshift x5, x1, x1, r3;		vmac.f dm1, dm4, x5, x7, r1
	vst x4, [p5], #64;		vshift x4, x2, x2, r3;		vmac.f dm3, dm3, x3, x0, r1
	vlda x5, [p0], #64;		vconv.bf16.fp32 x6, cml3;		vshift x5, x5, x1, r3;		vmac.f dm2, dm4, x6, x0, r1
	vlda bmll2, [p3, #384];		vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26];		vmac.f dm3, dm2, x3, x10, r1
	vlda bmlh2, [p3, #448];		vshift x0, x4, x2, r3;		vmac.f dm4, dm3, x6, x0, r1
	vconv.bf16.fp32 x4, cml0;		vshift x5, x5, x1, r3
	vldb.128 wl1, [p4], #16;		vconv.bf16.fp32 x3, cml1;		vshift x0, x0, x2, r3;		vmac.f dm0, dm3, x3, x8, r1
	vlda x4, [p7], #64;		vldb.128 wl2, [p1], #16;		vshift x1, x5, x1, r3;		vmsc.f dm4, dm0, x4, x11, r1
nrm_ln_last:
	nopa;		nopb;		vconv.bf16.fp32 x2, cml2;		nopx;		vshift x0, x0, x2, r3;		vmsc.f dm3, dm1, x3, x11, r1
	vst.flush.512.conv	[p2, sf, r26]
	nop
	nop
	nop
	nop
	nop
	paddxm	[sp], #-64
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_nrm_ln:
	.size	nrm_ln, .Lfunc_end_nrm_ln-nrm_ln
