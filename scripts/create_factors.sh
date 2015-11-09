#!/bin/bash 

indir=$1
outdir=$2
src_lang=$3
tgt_lang=$4

mkdir -p $outdir

d='/home/development/anoop/experiments/unsupervised_transliterator/src/transliterator/scripts'

python $d/create_factors.py $indir/train.$src_lang $outdir/train.$src_lang $src_lang
python $d/create_factors.py $indir/train.$tgt_lang $outdir/train.$tgt_lang $tgt_lang
python $d/create_factors.py $indir/tun.$src_lang $outdir/tun.$src_lang $src_lang
cp $indir/tun.$tgt_lang $outdir/tun.$tgt_lang
#python create_factors.py $indir/test.$src_lang $outdir/test.$src_lang $src_lang
#cp $indir/test.$tgt_lang $outdir/test.$tgt_lang
