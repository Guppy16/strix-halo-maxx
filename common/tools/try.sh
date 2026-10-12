#!/bin/bash
# try.sh - one-command experiment for the TAPNext++ step (research rev 11; run on the host side).
#
#   try.sh NAME [options]
#     builds full12NAME (+ full10/8/6NAME with --sweep) from the deployed r15m configuration plus your changes, lints the
#     elfs (asmlint), checks numerics (step8 regress vs full12r9a), times depth 12 (median of 60 after 20 warm-up, gpu_busy
#     checked before and recorded), and prints a <= 6-line summary. Every build gets a FRESH name: IRON's jit cache ignores
#     source files, so a reused name silently reuses the old build.
#
#   options (repeatable):
#     --ew-add "-DFOO"      --ew-del "-DFOO"        EW_DEFS (base: r15m's)
#     --k-add "-DBAR"       --k-del "-DBAR"         --kdefs  (base: r15m's)
#     --env KEY=VALUE                               design / plan env (P60G, A_BANKS, ASM_FIT, ...; base r15m's)
#     --args "--l2-depth 12"                        extra design8 arguments (base: --lookahead 2 --l2-depth 8)
#     --sweep               also build depths 10 / 8 / 6 and run step8 sweep
#     --no-regress  --no-time  --no-lint
#     --base-ms 41.52       baseline dispatch ms for the delta (default: full12r15m's 41.52)
#     --allow-busy          time even when gpu_busy > 5 % (labelled CONTENDED)
#     --print-only          print every command, run nothing (no NPU, no docker)
#
#   Output: build/NAME.try.log (everything), the summary on stdout. Exit 0 = built, lint clean, BIT-IDENTICAL.
#   Builds run sequentially (two design builds in one build dir race on the generated asm: tools/asmfit_tmp.patch).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SHM_SCRATCH=$(cd "${SHM_SCRATCH:-$REPO/../.scratch}" && pwd)   # scratch / working dir (docker mounts it as /work)
set -u

ROOT=${TRY_ROOT:-$SHM_SCRATCH}     # the docker mount root (one variable: move the repo -> change this line)
WS=npu-tapnext                                  # workspace inside ROOT (design8 / plan8 / step8 / build/)
REF=full12r9a
EW_BASE="-DSSM_LUT -DSSM_EXP3 -DSIGK -DSSX -DCPY64 -DCONVK -DRECK -DNRMK -DZRING"
K_BASE="-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS,-DGELU_MS,-DGUP,-DCPY32,-DPAN,-DGELU_FLAT,-DBGELU,-DATTKF,-DATTKE,-DATTKB,-DKFOLD,-DZASM,-DATTV,-DVTK,-DRESCK"
ENV_BASE=(P60G=1 P43=1 A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 ASM_FIT=1 EW_OPT=-O2 AIECC_EXTRA=--reclaim-runtime-bds)
ARGS_BASE="--lookahead 2 --l2-depth 8"

[ $# -ge 1 ] || { sed -n '2,26p' "$0"; exit 2; }
NAME=$1; shift
EW=" $EW_BASE "; K=",$K_BASE,"; ENVS=("${ENV_BASE[@]}"); ARGS=$ARGS_BASE
SWEEP=0; REGRESS=1; TIME=1; LINT=1; BASE_MS=41.52; ALLOW_BUSY=0; PRINT=0
while [ $# -gt 0 ]; do
  case $1 in
    --ew-add) EW="$EW $2 "; shift ;;
    --ew-del) EW=${EW// $2 / }; shift ;;
    --k-add) K="$K$2,"; shift ;;
    --k-del) K=${K//,$2,/,}; shift ;;
    --env) ENVS=("${ENVS[@]/#${2%%=*}=*/}" "$2"); shift ;;
    --args) ARGS="$ARGS $2"; shift ;;
    --sweep) SWEEP=1 ;;
    --no-regress) REGRESS=0 ;;
    --no-time) TIME=0 ;;
    --no-lint) LINT=0 ;;
    --base-ms) BASE_MS=$2; shift ;;
    --allow-busy) ALLOW_BUSY=1 ;;
    --print-only) PRINT=1 ;;
    *) echo "try.sh: unknown option $1"; exit 2 ;;
  esac
  shift
done
EW=$(echo $EW); K=$(echo "$K" | sed 's/^,*//; s/,*$//; s/,,*/,/g')
B=$ROOT/$WS/build
LOG=$B/$NAME.try.log
DEPTHS="12"; [ $SWEEP = 1 ] && DEPTHS="12 10 8 6"

run() {  # print, and run unless --print-only; output appended to the log
  echo "+ $*" | tee -a "${LOG:-/dev/null}" >&2
  [ $PRINT = 1 ] && return 0
  bash -c "$*" >> "$LOG" 2>&1
}
envflags() { local e; for e in "${ENVS[@]}"; do [ -n "$e" ] && printf -- "-e %q " "$e"; done; printf -- "-e EW_DEFS=%q " "$EW"; }

if [ $PRINT = 0 ]; then
  for d in $DEPTHS; do
    [ -e "$B/full$d$NAME.xclbin" ] && { echo "try.sh: build/full$d$NAME exists - pick a new name (the jit cache ignores sources)"; exit 2; }
  done
  : > "$LOG"
fi

# 1. builds (sequential), the same steps as bb.sh with the mount root in one variable
for d in $DEPTHS; do
  n=full$d$NAME
  run "docker run --rm --name try-build-$n $(envflags) -v $ROOT:/work -v $REPO:$REPO npu-iron bash -c \"cd /work/$WS && python3 run8.py sched $n && DESIGN=design8 bash b.sh $n --sched /work/$WS/build/$n.sched $ARGS --kdefs=$K && python3 r11/stackaudit.py build/$n.prj\""
  if [ $PRINT = 0 ] && ! tail -1 "$B/$n.log" 2>/dev/null | grep -q "rc=0"; then
    echo "$NAME: BUILD FAILED ($n: $(tail -1 "$B/$n.log" 2>/dev/null)); see build/$n.log"; exit 1
  fi
done

# 2. lint + program memory (CPU container, no device)
SUM_LINT="lint skipped"
if [ $LINT = 1 ]; then
  run "docker run --rm --network none --name try-lint-$NAME -v $ROOT:/work -v $REPO:$REPO npu-iron bash -c \"python3 $REPO/common/tools/asmlint.py --prj /work/$WS/build/full12$NAME.prj --json > /work/$WS/build/$NAME.lint.json; P=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin; for c in 0_2 0_5; do \\\$P/llvm-size -A /work/$WS/build/full12$NAME.prj/elfs_main_core_\\\$c/elfs_main_core_\\\$c.elf | awk '/^.text/{print \\\"text_\\\" \\\"'\\\$c'\\\" \\\" \\\" \\\$2}'; done > /work/$WS/build/$NAME.size\""
  if [ $PRINT = 0 ]; then
    SUM_LINT=$(python3 -c "import json,sys; j=json.load(open('$B/$NAME.lint.json')); c=j['counts']; f=[x for x in j['findings'] if x['severity']=='error'][:2]; print(f\"lint {c.get('error',0)} error / {c.get('warning',0)} warning\" + ''.join(f\"; {x['rule']} {x['func']}@{x['addr']:#x}\" for x in f))" 2>&1)
    SUM_LINT="$SUM_LINT; text $(tr '\n' ' ' < "$B/$NAME.size" | sed 's/text_0_2/GEMM/; s/text_0_5/EW/')(of 16384)"
  fi
fi

S8="docker run --rm --device /dev/accel/accel0 --ulimit memlock=-1:-1 -v $ROOT:/work -v $REPO:$REPO -v ${TAPNEXT_SRC:?path to the TAPNext torch model.py}:/tapnext/model.py:ro -v ${TAPNEXT_WEIGHTS:?path to tapnextpp_256.safetensors}:/tapnext/tapnextpp_256.safetensors:ro npu-iron-torch bash -c"
# 3. numerics: bit-level regress vs the round-9 reference (4 frames of clip30)
SUM_REG="regress skipped"
if [ $REGRESS = 1 ]; then
  run "$S8 \"cd /work/$WS && timeout 900 python3 step8.py regress --build full12$NAME --depth 12 --ref $REF > build/$NAME.regress.out 2>&1; echo rc=\\\$? >> build/$NAME.regress.out\""
  [ $PRINT = 0 ] && SUM_REG=$(grep -h "^REGRESS BIT\|^REGRESS DIFF\|^REGRESS frame 3" "$B/$NAME.regress.out" | tail -2 | tr '\n' ' ' | cut -c1-160)
  [ $PRINT = 0 ] && [ -z "$SUM_REG" ] && SUM_REG="regress FAILED: $(tail -2 "$B/$NAME.regress.out" | tr '\n' ' ')"
fi

# 4. timing at depth 12 (gpu_busy checked first; the TIME line records it again)
SUM_TIME="time skipped"
if [ $TIME = 1 ]; then
  BUSY=$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo "?")
  if [ $PRINT = 0 ] && [ "$BUSY" != "?" ] && [ "$BUSY" -gt 5 ] && [ $ALLOW_BUSY = 0 ]; then
    SUM_TIME="time NOT RUN: gpu_busy $BUSY % (use --allow-busy for a CONTENDED number)"
  else
    run "$S8 \"cd /work/$WS && timeout 900 python3 step8.py time --build full12$NAME --depth 12 --iters 60 --warmup 20 > build/$NAME.time.out 2>&1; echo rc=\\\$? >> build/$NAME.time.out\""
    if [ $PRINT = 0 ]; then
      T=$(grep -h "^TIME" "$B/$NAME.time.out" | tail -1)
      MED=$(echo "$T" | sed -n 's/.*dispatch median \([0-9.]*\).*/\1/p')
      [ -n "$MED" ] && SUM_TIME=$(echo "$T" | sed 's/^TIME [^:]*: //' | cut -c1-120)"; delta $(python3 -c "print(f'{$MED - $BASE_MS:+.2f}')") ms vs $BASE_MS (pre-check gpu_busy $BUSY %)" \
                    || SUM_TIME="time FAILED: $(tail -2 "$B/$NAME.time.out" | tr '\n' ' ')"
    fi
  fi
fi

# 5. optional depth sweep
SUM_SWEEP=""
if [ $SWEEP = 1 ]; then
  run "$S8 \"cd /work/$WS && timeout 1500 python3 step8.py sweep --build full12$NAME --iters 60 --warmup 20 > build/$NAME.sweep.out 2>&1\""
  [ $PRINT = 0 ] && SUM_SWEEP=$(grep -h "^SWEEP" "$B/$NAME.sweep.out" | sed -n 's/.*depth *\([0-9]*\): dispatch median \([0-9.]*\).*/d\1 \2/p' | tr '\n' ' ')
fi

[ $PRINT = 1 ] && { echo "(print-only: nothing ran)"; exit 0; }
echo "$NAME: built $(echo $DEPTHS | sed 's/ /,/g') | EW_DEFS +[$(comm -13 <(echo $EW_BASE | tr ' ' '\n' | sort) <(echo $EW | tr ' ' '\n' | sort) | tr '\n' ' ')] -[$(comm -23 <(echo $EW_BASE | tr ' ' '\n' | sort) <(echo $EW | tr ' ' '\n' | sort) | tr '\n' ' ')] kdefs +[$(comm -13 <(echo $K_BASE | tr ',' '\n' | sort) <(echo $K | tr ',' '\n' | sort) | tr '\n' ' ')]"
echo "  $SUM_LINT"
echo "  $SUM_REG"
echo "  $SUM_TIME"
[ -n "$SUM_SWEEP" ] && echo "  sweep: $SUM_SWEEP"
echo "  log: build/$NAME.try.log"
case "$SUM_LINT$SUM_REG" in *"0 error"*BIT-IDENTICAL*) exit 0 ;; *"lint skipped"*BIT-IDENTICAL*) exit 0 ;; *) exit 1 ;; esac
