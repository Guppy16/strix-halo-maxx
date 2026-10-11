#!/bin/bash
# usage: loopii.sh NAME : bundle count of each inner loop (label .. L_LEnd) in mm_acc
f=/work/p2b/build/$1.mm.dis
awk '/<\.LBB1_[0-9]+>:/{lab=$2; n=0; inl=1; next} /<\.L_LEnd/{if(inl) print lab, n; inl=0; next} inl && /^ +[0-9a-f]+:/{n++}' $f
echo "vmac $(grep -c vmac $f) sp $(grep -c '\[sp' $f)"
