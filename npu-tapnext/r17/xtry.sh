#!/bin/bash
# round 17 (X1 full rotation): build NAME with XROT=N frame phases, i.e. try17.sh (phase 0, lint) + the phases 1 .. N-1 of every
# depth (the same docker command with XPH=k and the name full<d>NAME_p<k>), then regress (4 frames: N <= 4 includes a wrap frame)
# and time (per-phase medians). usage: r17/xtry.sh NAME N [try.sh options, e.g. --sweep --ew-add ..]
name=$1; N=$2; shift 2
cd ~/git/app/.scratch/npu-tapnext || exit 1
SW=0; for a in "$@"; do [ "$a" = --sweep ] && SW=1; done
args=(); for a in "$@"; do [ "$a" != --sweep ] && args+=("$a"); done
if [ "${SKIP0:-0}" != 1 ]; then  # (try.sh exits 1 without a regress: judge the build by its log)
  r17/try17.sh "$name" --env XROT=$N --ew-add -DXROT --no-regress --no-time "${args[@]}"
  tail -1 build/full12$name.log | grep -q rc=0 || exit 1
fi
depths="12"; [ $SW = 1 ] && depths="12 10 8 6"
cmd=$(grep -m1 "^+ docker run --rm --name try-build-full12$name " build/$name.try.log | sed 's/^+ //')
[ -n "$cmd" ] || { echo "no build command"; exit 1; }
for d in $depths; do
  k0=1; [ $d != 12 ] && k0=0  # (try.sh built phase 0 of depth 12 only; the depth is in the name)
  for k in $(seq $k0 $((N - 1))); do
    n=full$d$name; [ $k != 0 ] && n=${n}_p$k
    [ -e build/$n.xclbin ] && tail -1 build/$n.log | grep -q rc=0 && continue
    c=${cmd//full12$name/$n}
    c=${c/docker run --rm/docker run --rm -e XPH=$k}
    bash -c "$c" >> build/$name.try.log 2>&1
    tail -1 build/$n.log | grep -q rc=0 || { echo "BUILD FAILED $n"; exit 1; }
  done
done
echo "$name: phases built"
[ "${NOREG:-0}" = 1 ] || { r16/nrun.sh ${name}_reg python3 step8.py regress --build full12$name --depth 12 --ref full12r9a | grep "^REGRESS BIT\|^REGRESS DIFF\|^REGRESS frame 3\|rc="; }
[ "${NOTIME:-0}" = 1 ] || { r16/nrun.sh ${name}_time python3 step8.py time --build full12$name --depth 12 --iters 60 --warmup 20 | grep "^TIME\|^PHASES\|rc="; }
[ $SW = 1 ] && { TMO=1500 r16/nrun.sh ${name}_sweep python3 step8.py sweep --build full12$name --iters 60 --warmup 20 | grep "^SWEEP\|^PHASES\|rc="; }
