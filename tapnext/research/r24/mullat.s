	.text
	.section	.text.probe0,"ax",@progbits
	.globl	probe0
	.p2align	4
	.type	probe0,@function
probe0:
	mova	r20, #5
	mova	r21, #7
	mova	r22, #1
	movxm	ls, #.Lprobe0_body
	movxm	le, #.Lprobe0_end
	mova	r0, #2
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	.p2align	4
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe0_body:
	nopa;		nopb;		nops;		nopx;		mov r26, r22;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		mul r26, r20, r21;		nopm;		nopv
	nopa;		nopb;		st r26, [p1], #4;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe0_end:
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
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
	mova	r20, #5
	mova	r21, #7
	mova	r22, #1
	movxm	ls, #.Lprobe1_body
	movxm	le, #.Lprobe1_end
	mova	r0, #2
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	.p2align	4
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe1_body:
	nopa;		nopb;		nops;		nopx;		mov r26, r22;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		mul r26, r20, r21;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		st r26, [p1], #4;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe1_end:
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
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
	mova	r20, #5
	mova	r21, #7
	mova	r22, #1
	movxm	ls, #.Lprobe2_body
	movxm	le, #.Lprobe2_end
	mova	r0, #2
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	.p2align	4
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe2_body:
	nopa;		nopb;		nops;		nopx;		mov r26, r22;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		mul r26, r20, r21;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		st r26, [p1], #4;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe2_end:
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
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
	mova	r20, #5
	mova	r21, #7
	mova	r22, #1
	mov	r26, r22
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mul r26, r20, r21
	st	r26, [p1], #4
	nop
	nop
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
.Lfunc_end_probe3:
	.size	probe3, .Lfunc_end_probe3-probe3
	.section	.text.probe4,"ax",@progbits
	.globl	probe4
	.p2align	4
	.type	probe4,@function
probe4:
	mova	r20, #5
	mova	r21, #7
	mova	r22, #1
	mov	r26, r22
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mul r26, r20, r21
	nop
	st	r26, [p1], #4
	nop
	nop
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
.Lfunc_end_probe4:
	.size	probe4, .Lfunc_end_probe4-probe4
	.section	.text.probe5,"ax",@progbits
	.globl	probe5
	.p2align	4
	.type	probe5,@function
probe5:
	mova	r20, #5
	mova	r21, #7
	mova	r22, #1
	mov	r26, r22
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mul r26, r20, r21
	nop
	nop
	st	r26, [p1], #4
	nop
	nop
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
	mova	r20, #5
	mova	r21, #7
	mova	r22, #1
	movxm	ls, #.Lprobe6_body
	movxm	le, #.Lprobe6_end
	mova	r0, #2
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	.p2align	4
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe6_body:
	nopa;		nopb;		nops;		nopx;		mov r26, r22;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		add r26, r20, #1;		nopm;		nopv
	nopa;		nopb;		st r26, [p1], #4;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe6_end:
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
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
	mova	r20, #5
	mova	r21, #7
	mova	r22, #1
	movxm	ls, #.Lprobe7_body
	movxm	le, #.Lprobe7_end
	mova	r0, #2
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	.p2align	4
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe7_body:
	nopa;		nopb;		nops;		nopx;		mov r26, r22;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		add r26, r20, #1;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		st r26, [p1], #4;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	.p2align	4
.Lprobe7_end:
	nopa;		nopb;		nops;		nopx;		nopm;		nopv
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_probe7:
	.size	probe7, .Lfunc_end_probe7-probe7
