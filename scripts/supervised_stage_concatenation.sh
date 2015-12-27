
#### Inputs 
###
# contains the following files: 
#   all.$src_lang all.nbest.top50.$tgt_lang all.nbest.reranked.$tgt_lang
corpus_dir=""
lm_fname=""
src_lang=""
tgt_lang=""
test_data_dir=""

##  output directory
output_dir=""

####### Fixed paramters for now  
vtuning_set=1000
topk=10
niter=1
pbsmt_conf_template_fname="$XLIT_HOME/scripts/supervised_stage2_moses_template.conf"
MOSES_CMD="/usr/local/bin/smt/mosesdecoder-3.0/bin/moses"
JOBSCRIPT="/usr/local/bin/smt/moses_job_scripts/moses_run.sh"


##### Initialize for the first iteration 
src_corpus=$corpus_dir/all.$src_lang  
tgt_corpus=$corpus_dir/all.nbest.reranked.$tgt_lang  

while 1 
do

#iter_dir=$output_dir/$itern    
#mkdir -p $iter_dir/{corpus,evaluation,workspace,log}
#
#### create the corpus for training PBSMT system     
#time python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_synthetic_corpus_concatenated \
#            $src_corpus  \
#            $tgt_corpus  \
#            $iter_dir/corpus \
#            $src_lang \
#            $tgt_lang \
#            $topk \
#            $vtuning_set 
#
###### create the configuration file 
#python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_moses_run_params_concat_sys \
#        $pbsmt_conf_template_fname\
#        $iter_dir/pbmst.conf \
#        $iter_dir/workspace \
#        $iter_dir/corpus \
#        $lm_fname  \
#        $src_lang \
#        $tgt_lang
#
##### train the PBSMT model 
#nohup $JOBSCRIPT $iter_dir/pbmst.conf > $iter_dir/log/pbsmt_train.log  2>&1 & 
#
###### evaluate on test set
#$MOSES_CMD -f $iter_dir/workspace/tuning/moses.ini < $test_data_dir/test.$src_lang > $iter_dir/evaluation/test.nbest.$tgt_lang 2> $iter_dir/log/pbsmt_evaluate.log 
#ref_xml=$test_data_dir/test.xml
#
## generate NEWS 2015 evaluation format output file 
#python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#        "$test_data_dir/test.id" \
#        "$ref_xml" \
#        "$iter_dir/evaluation/test.nbest.$tgt_lang" \
#        "$iter_dir/evaluation/test.nbest.$tgt_lang.xml" \
#        "pbsmt" "testset" "$src_lang" "$tgt_lang"  
#
## run evaluation 
#python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#        -t $ref_xml \
#        -i $iter_dir/evaluation/test.nbest.$tgt_lang.xml \
#        -o $iter_dir/evaluation/test.nbest.$tgt_lang.detaileval.csv \
#        > $iter_dir/evaluation/test.nbest.$tgt_lang.eval 

#### break if all iterations done
if $citer -eq $niter
then 
    break 
fi 

####### create virtual corpus for next iteration 
#
#$output_dir/$itern/corpus 
#mkdir 
#
### update the variables to the new file 

done 


