src_lang=""
tgt_lang=""
training_corpus_dir=""
test_corpus_dir=""

## unsupervised model dir
unsup_model=""
decoder_params_fname=""
lm_2g_fname=""
lm_ng_fname=""
lm_n_order=""
topk=""

##  output directory
output_dir=""

## 
vtuning_set=1000
pbsmt_conf_template_fname=""

### make the required directories 
mkdir -p $output_dir/results
mkdir -p $output_dir/corpus
mkdir -p $output_dir/conf

### decode the source side of the training 
# test

## bigram lm used 
time python $XLIT_HOME/src/cfilt/transliteration/transliterate_cli.py transliterate_topn \
            $unsup_model \
            $lm_2g_fname \
            $training_corpus_dir/train.$src_lang \
            $output_dir/corpus/all.nbest.unreranked.$tgt_lang \
            $topk \
            $decoder_params_fname > $workspace_dir/log/decode.log 2>&1 

### rerank the output with a higher order LM 
time python $XLIT_HOME/src/cfilt/transliteration/reranker.py rerank_file \
            $output_dir/corpus/all.nbest.unranked.$tgt_lang \
            $output_dir/corpus/all.nbest.reranked.$tgt_lang \
            $topk  \
            $lm_ng_fname \
            $lm_n_order

decoded_vcorpus=$output_dir/corpus/all.nbest.reranked.$tgt_lang


### create multiple parallel corpora using the top-k outputs from the  unsup decoding (split into train and tun sets) 
### the tun sets would be common across all splits, coming from the top-1 results 
time python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_synthetic_corpus\
            $training_corpus_dir/train.$src_lang \
            $decoded_vcorpus \
            $output_dir/corpus \
            $src_lang \
            $tgt_lang \
            $topk \
            $vtuning_set 

### train systems for each split 
## create moses.ini files
for i in `seq 0 $((topk-1))`
do 
    conf_fname=$output_dir/conf/pbmst_$i.conf
    workspace_dir=$output_dir/results/gen1/$i
    parallel_corpus=$output_dir/corpus/$i
    time python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_moses_run_params \
            $pbsmt_conf_template_fname\
            $conf_fname\
            $workspace_dir\
            $parallel_corpus \
            $lm_ng_fname\
            $src_lang \
            $tgt_lang
    
    /usr/local/bin/smt/moses_job_scripts/moses_run.sh $conf_fname notun        
done 

### combine the systems if required 

### tune the final system

### decode the test set with the final system model 

### run evaluation metrics 
