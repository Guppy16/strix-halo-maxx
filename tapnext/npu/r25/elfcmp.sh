#!/bin/bash
# round 25: are two builds' core elfs (patched csplit prj and the IRON prj) and CDOs byte-identical? usage: r25/elfcmp.sh A B
cd ~/git/app/.scratch/npu-tapnext || exit 1
for n in "$1" "$2"; do
  a=$(cd build/r19/cs_$n/prj && find . -name "*.elf" | sort | xargs md5sum | md5sum | cut -c1-12)
  b=$(cd build/full12$n.prj && find . -name "*.elf" | sort | xargs md5sum | md5sum | cut -c1-12)
  c=$(cd build/r19/cs_$n/prj && find . -name "*cdo*.bin" | sort | xargs md5sum | md5sum | cut -c1-12)
  ne=$(cd build/r19/cs_$n/prj && find . -name "*.elf" | wc -l)
  echo "$n: patched elfs $a ($ne)  iron elfs $b  cdo $c"
done
