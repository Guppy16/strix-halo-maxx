#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
# round 24: asmlint (common/tools/asmlint.py --prj: GEMM core 0_2 + EW core 0_5) + elf text sizes of a build; usage: r24/lint.sh BUILD
b=$1
cd "$SHM_SCRATCH/npu-tapnext" || exit 1
docker run --rm --network none --name lint-$b -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron bash -c "
cd /work/npu-tapnext; python3 $SHM_REPO/common/tools/asmlint.py --prj build/$b.prj > build/$b.lint.txt 2>&1; echo lint rc=\$?
P=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
for c in 0_2 0_5; do \$P/llvm-size -A build/$b.prj/elfs_main_core_\$c/elfs_main_core_\$c.elf | awk -v c=\$c '/^.text/{print \"text_\" c, \$2}'; done
chmod a+rw build/$b.lint.txt"
grep -i "error\|summary\|total" build/$b.lint.txt | tail -4
