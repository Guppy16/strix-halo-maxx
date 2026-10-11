#!/bin/bash
# round 24: r23/try23.sh (full12r23z) + the round-24 kept changes:
#   STATS64 (the norm statistics of both 32-row halves as one 64-lane chain, LN and RMS; STATS64=0 without);
#   OPPRE2 (out_proj's A loaded per attention head group during pass 1, its sends before the waits; OPPRE2=0 without);
#   SSMPRE (the SSM's head-0 params fills before the yx wait: the columns start ~0.55 instead of ~1.6 us apart; SSMPRE=0 without);
#   CPPRE (post-wait shim fills configured before the wait, only pushed after: SSM head 0, the norms' inputs; CPPRE=0 without;
#   CPPRE_OP=1 also OPPRE2's parts: static BD peak 20 > 16, never dispatched).
# Dead / off by default (env switches kept in design8 / plan8 / ew8): MSDEFER (out's pass sends after yx's drains, with MSPLIT incl.
# out), YXH (yx steps head-major; drains in 4 groups), ZDROP (SSM h-only ring under SRES). Timing builds: --ew-add -DSSMIN (r24/ssmin.py),
# MSPLIT= ... --k-add -DTPROF (r20/tprof20.py + r24/tpk.py).
# usage: r24/try24.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V] [--args ..]   (NOREG / NOTIME / MSPLIT / YXSSM / .. as try23)
name=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
R24=()
[ "${STATS64:-1}" = 1 ] && R24+=(--ew-add -DSTATS64)
[ "${OPPRE2:-1}" = 1 ] && R24+=(--env OPPRE2=1)  # out_proj's A per attention head group (OPPRE2_SENDS=0: its sends after the waits, v1)
[ "${SSMPRE:-1}" = 1 ] && R24+=(--env SSMPRE=1)
[ "${CPPRE:-1}" = 1 ] && R24+=(--env CPPRE=1)
exec r23/try23.sh "$name" "${R24[@]}" "$@"
