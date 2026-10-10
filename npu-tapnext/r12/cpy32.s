	.text
	.section	.text.cpy32,"ax",@progbits
	.globl	cpy32
	.p2align	4
	.type	cpy32,@function
cpy32:
	mov	dn0, r3
	mov	dn1, r3
	mova	dj0, #32
	mova	dj1, #32
	mova	dc0, #0
	mova	dc1, #0
	mov	m0, r1
	mov	m1, r2
	movxm	ls, #cpy32_start
	movxm	le, #cpy32_last
	add.nc	lc, r0, #0
	nop
	nop
	nop
	nop
	nop
	nop
	// modulo schedule: II 1, U 2, stages 8, ops 2
	vlda.2d wl1, [p0], d0
	vlda.2d wl0, [p0], d0
	vlda.2d wl1, [p0], d0
	vlda.2d wl0, [p0], d0
	vlda.2d wl1, [p0], d0
	vlda.2d wl0, [p0], d0
	vlda.2d wl1, [p0], d0
	.p2align	4
cpy32_start:
	vlda.2d wl0, [p0], d0;		nopb;		vst.2d wl1, [p1], d1;		nopx;		nopm;		nopv
cpy32_last:
	vlda.2d wl1, [p0], d0;		nopb;		vst.2d wl0, [p1], d1;		nopx;		nopm;		nopv
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
.Lfunc_end_cpy32:
	.size	cpy32, .Lfunc_end_cpy32-cpy32
