#!/bin/bash
# usage (host): r6/ccs.sh SRC TAG [flags]   compile SRC like the GEMM/EW design (build/asm/cc.sh) to r6/TAG.o and r6/TAG.s,
# print the function sizes and the asmstat loop table for FUNCS (env, space-separated)
cd ~/git/app/.scratch/npu-tapnext
src=$1; tag=$2; shift 2
docker run --rm --name r6-cc-$tag -v $HOME/git/app/.scratch:/work -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx npu-iron bash -c "cd /work/npu-tapnext && N=${N:-8} bash build/asm/cc.sh $src r6/$tag.o $* && bash build/asm/cc.sh $src r6/$tag.s -S $* >/dev/null 2>&1; chmod a+rw r6/$tag.*"
[ -n "$FUNCS" ] && python3 r6/asmstat.py r6/$tag.s $FUNCS
