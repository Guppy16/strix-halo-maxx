#!/bin/bash
# usage (host): crop-study/run.sh NAME crop_study.py-args...   CPU only, nice 19, 8 threads; log in out/NAME.log
name=$1; shift
D=$HOME/git/app/strix-halo-maxx/crop-study
mkdir -p $D/out
docker run --rm --name crop-$name -e OMP_NUM_THREADS=8 -v $HOME/git/app/.scratch:/work \
  -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx -v $HOME/git/app:/repo:ro \
  npu-iron-torch nice -n 19 python3 $D/crop_study.py "$@" > $D/out/$name.log 2>&1
echo "rc=$?" >> $D/out/$name.log
chmod -R a+rwX $D/out $D/*.json 2>/dev/null
