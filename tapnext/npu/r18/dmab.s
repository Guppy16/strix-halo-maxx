// round 18 DMAB (r18/dmab.py): zero-overhead loops of 4 full-form bundles with one 64 B access each at a fixed address
	.text
	.section	.text.dmab_ld,"ax",@progbits
	.globl	dmab_ld
	.p2align	4
	.type	dmab_ld,@function
dmab_ld:
	movxm	ls, #.Lld_s
	movxm	le, #.Lld_e
	add.nc	lc, r0, #0
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	.p2align	4
.Lld_s:
	vlda	x0, [p0, #0];		nopb	;		nops	;		nopx	;		nopm	;		nopv
	vlda	x1, [p0, #0];		nopb	;		nops	;		nopx	;		nopm	;		nopv
	vlda	x2, [p0, #0];		nopb	;		nops	;		nopx	;		nopm	;		nopv
.Lld_e:
	vlda	x3, [p0, #0];		nopb	;		nops	;		nopx	;		nopm	;		nopv
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
.Lfe_ld:
	.size	dmab_ld, .Lfe_ld-dmab_ld

	.section	.text.dmab_st,"ax",@progbits
	.globl	dmab_st
	.p2align	4
	.type	dmab_st,@function
dmab_st:
	movxm	ls, #.Lst_s
	movxm	le, #.Lst_e
	add.nc	lc, r0, #0
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	.p2align	4
.Lst_s:
	nopa	;		nopb	;		vst	x0, [p0, #0];		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		vst	x0, [p0, #0];		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		vst	x0, [p0, #0];		nopx	;		nopm	;		nopv
.Lst_e:
	nopa	;		nopb	;		vst	x0, [p0, #0];		nopx	;		nopm	;		nopv
	nop
	nop
	nop
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfe_st:
	.size	dmab_st, .Lfe_st-dmab_st

	.section	.text.dmab_nop,"ax",@progbits
	.globl	dmab_nop
	.p2align	4
	.type	dmab_nop,@function
dmab_nop:
	movxm	ls, #.Lnp_s
	movxm	le, #.Lnp_e
	add.nc	lc, r0, #0
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	.p2align	4
.Lnp_s:
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
.Lnp_e:
	nopa	;		nopb	;		nops	;		nopx	;		nopm	;		nopv
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfe_np:
	.size	dmab_nop, .Lfe_np-dmab_nop
