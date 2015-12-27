#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:$XLIT_HOME/src

. $1

fcorpus_fname="$data_dir/train.$src_lang"
ecorpus_fname="$data_dir/train.$tgt_lang"
test_fcorpus_fname="$data_dir/test.$src_lang"
test_ecorpus_fname="$data_dir/test.$tgt_lang"

#training_corpus_dir=""
#test_corpus_dir=""

## unsupervised model dir
unsup_model="$workspace_dir/model/translit.model"
lm_2g_fname=$lm_fname 
lm_ng_fname=$lm_ng_fname 
lm_n_order="5"
topk="10"

##  output directory
output_dir="$2"

### 
#vtuning_set=1000
#pbsmt_conf_template_fname=""

### make the required directories 
mkdir -p $output_dir/results
mkdir -p $output_dir/corpus
mkdir -p $output_dir/conf
mkdir -p $output_dir/log

### decode the source side of the training 
# test

## bigram lm used 
time python $XLIT_HOME/src/cfilt/transliteration/transliterate_cli.py transliterate_topn \
            $unsup_model \
            $lm_2g_fname \
            $fcorpus_fname \
            $output_dir/corpus/all.nbest.top50.$tgt_lang \
            50 \
            $decoder_params_fname > $output_dir/log/create_initial_train.log >&1 

### rerank the output with a higher order LM 
time python $XLIT_HOME/src/cfilt/transliteration/reranker.py rerank_file \
            $output_dir/corpus/all.nbest.top50.$tgt_lang \
            $output_dir/corpus/all.nbest.reranked.$tgt_lang \
            $topk  \
            $lm_ng_fname \
            $lm_n_order > $output_dir/log/rerank_initial_train.log >&1 

cp $fcorpus_fname $output_dir/corpus/all.$src_lang
