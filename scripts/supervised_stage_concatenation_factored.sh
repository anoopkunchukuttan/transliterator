#!/bin/bash 

## read config file 
. $1 


# for training 
mkdir -p  $output_dir
cp $1 $output_dir/stage2.conf

### for evaluation 
#mkdir -p  $eval_output_dir
#cp $1 $eval_output_dir/stage2.conf

######  FIX the paramters 
MOSES_CMD="/usr/local/bin/smt/mosesdecoder-latest-25Dec2015/bin/moses"
JOBSCRIPT="/usr/local/bin/smt/moses_job_scripts/moses_run.sh"

##### Initialize for the first iteration 

# for starting with first iteration 
src_corpus=$corpus_dir/all.$src_lang  
tgt_corpus=$corpus_dir/all.nbest.reranked.$tgt_lang  
citer=1

##### for starting with any other iteration, the previous iteration must have generated its virtual corpus
#src_corpus=$corpus_dir/all.$src_lang  
#tgt_corpus=$output_dir/1/next_iter_decoded/all.nbest.reranked.$tgt_lang  
#citer=2

## create factored version of source corpus 
src_factored_corpus=$output_dir/all.factored.$src_lang
python $XLIT_HOME/scripts/create_factors.py create_factor_file $src_corpus $src_factored_corpus $src_lang

while true 
do

    iter_dir=$output_dir/$citer

    ###### TRAIN 
    mkdir -p $iter_dir/{corpus,evaluation,workspace,next_iter_decoded,log}
    
    ### create the corpus for training PBSMT system     
    time python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_synthetic_corpus_concatenated \
                $src_corpus  \
                $tgt_corpus  \
                $iter_dir/corpus \
                $src_lang \
                $tgt_lang \
                $topk \
                $vtuning_set 
   
    mv $iter_dir/corpus $iter_dir/corpus_pb 
    mkdir $iter_dir/corpus 

    # create factored corpus 
    python $XLIT_HOME/scripts/create_factors.py create_factor_file $iter_dir/corpus_pb/train.$src_lang $iter_dir/corpus/train.$src_lang $src_lang 
    python $XLIT_HOME/scripts/create_factors.py create_factor_file $iter_dir/corpus_pb/train.$tgt_lang $iter_dir/corpus/train.$tgt_lang $tgt_lang 
    python $XLIT_HOME/scripts/create_factors.py create_factor_file $iter_dir/corpus_pb/tun.$src_lang $iter_dir/corpus/tun.$src_lang $src_lang 
    cp $iter_dir/corpus_pb/tun.$tgt_lang $iter_dir/corpus/tun.$tgt_lang

    ##### create the configuration file 
    ## TODO: 
    python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_moses_factored_run_params \
            $pbsmt_conf_template_fname\
            $iter_dir/pbmst.conf \
            $iter_dir/workspace \
            $iter_dir/corpus \
            $lm_fname  \
            $factored_lm_dir \
            $src_lang \
            $tgt_lang
    
    #### train the PBSMT model 
    nohup $JOBSCRIPT $iter_dir/pbmst.conf > $iter_dir/log/pbsmt_train.log  2>&1  
  
    eval_output_dir=$output_dir

    ######## EVALUATION 

    ref_xml=$test_data_dir/test.xml
    eval_iter_dir=$eval_output_dir/$citer
    mkdir -p $eval_iter_dir/{evaluation,log}

    ##### evaluate on test set
    $MOSES_CMD -n-best-list $eval_iter_dir/evaluation/test.nbest.reranked.$tgt_lang 10 distinct -f $iter_dir/workspace/tuning/moses.ini \
        < $test_data_dir/test.$src_lang > $eval_iter_dir/evaluation/test.$tgt_lang 2> $eval_iter_dir/log/pbsmt_evaluate.log 
    
    
    # generate NEWS 2015 evaluation format output file 
    python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
            "$test_data_dir/test.id" \
            "$ref_xml" \
            "$eval_iter_dir/evaluation/test.nbest.reranked.$tgt_lang" \
            "$eval_iter_dir/evaluation/test.nbest.reranked.$tgt_lang.xml" \
            "pbsmt" "testset" "$src_lang" "$tgt_lang"  
    
    # run evaluation 
    python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
            -t $ref_xml \
            -i $eval_iter_dir/evaluation/test.nbest.reranked.$tgt_lang.xml \
            -o $eval_iter_dir/evaluation/test.nbest.reranked.$tgt_lang.detaileval.csv \
             > $eval_iter_dir/evaluation/test.nbest.reranked.$tgt_lang.eval 
    
    #### break if all iterations done
    echo Completed iteration $citer of $niter 
    
    if [ $citer -eq $niter ]
    then 
        break 
    fi 
   
    ### PREPARING FOR NEXT ITERATION
    ### update the variables to the new file 
    src_corpus=$corpus_dir/all.$src_lang  
    tgt_corpus=$iter_dir/next_iter_decoded/all.nbest.reranked.$tgt_lang  
   
    ####### create virtual corpus for next iteration 
    mkdir -p $iter_dir/next_iter_decoded
    $MOSES_CMD -n-best-list $iter_dir/next_iter_decoded/all.nbest.reranked.$tgt_lang 10 distinct -f $iter_dir/workspace/tuning/moses.ini \
        < $src_factored_corpus  > $iter_dir/next_iter_decoded/all.$tgt_lang 2> $iter_dir/log/next_iter_decoded.log 

    ### update citer 
    citer=$((citer+1))

done 


