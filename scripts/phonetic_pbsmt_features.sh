# expdir should contain a file called phrase_table and moses.ini (changes for new features and pointing to the new phrase table called phrase_table_out)

exp_dir=/home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb/12_13/bn-hi/evaluation/pbsmt_rerank/phonetic_features
parallel_corpus=/home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb/12_13/bn-hi/evaluation/parallel_corpus_rerank

mkdir $exp_dir/{tuning,evaluation}
python phonetic_sim.py $exp_dir/phrase_table $exp_dir/phrase_table_out bn hi

SCRIPTS_ROOTDIR=/usr/local/bin/smt/mosesdecoder-2.1.1/scripts
src_lang=bn
tgt_lang=hi

$SCRIPTS_ROOTDIR/training/mert-moses.pl \
            "$parallel_corpus/tun.$src_lang" \
            "$parallel_corpus/tun.$tgt_lang" \
            "/usr/local/bin/smt/mosesdecoder-2.1.1/bin/moses"  \
            "$exp_dir/moses.ini" \
             --working-dir "$exp_dir/tuning/" \
             --rootdir "$SCRIPTS_ROOTDIR" \
            --decoder-flags='-threads 20 -distortion-limit 0'

/usr/local/bin/smt/mosesdecoder-2.1.1/bin/moses \
-f $exp_dir/tuning//moses.ini \
-n-best-list "$exp_dir/evaluation/test.nbest.hi" 10 distinct \
-threads 16 -distortion-limit 0 \
< /home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/bn-hi/test.bn \
> $exp_dir/evaluation/test.hi


# evaluate
XLIT_HOME=/home/development/anoop/experiments/unsupervised_transliterator/src/transliterator
src_lang=bn
tgt_lang=hi
data_dir=/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/bn-hi
ref_xml=$data_dir/test.xml
workspace_dir=$exp_dir


systemtype=pb_pe
dataset=nonparallel

python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
        "$data_dir/test.id" \
        "$ref_xml" \
        "$workspace_dir/evaluation/test.nbest.$tgt_lang" \
        "$workspace_dir/evaluation/test.nbest.$tgt_lang.xml" \
        "$systemtype" "$dataset" "$src_lang" "$tgt_lang"

# run evaluation 
python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
        -t $ref_xml \
        -i $workspace_dir/evaluation/test.nbest.$tgt_lang.xml \
        -o $workspace_dir/evaluation/test.nbest.$tgt_lang.detaileval.csv \
        > $workspace_dir/evaluation/test.nbest.$tgt_lang.eval


