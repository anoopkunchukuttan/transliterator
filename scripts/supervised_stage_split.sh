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

## environment variables 
MOSES_DIR="/usr/local/bin/smt/mosesdecoder-latest-25Dec2015"
MOSES_CMD="$MOSES_DIR/bin/moses"
SCRIPTS_ROOTDIR="$MOSES_DIR/scripts"
JOBSCRIPT="/usr/local/bin/smt/moses_job_scripts/moses_run.sh"

## template files 
pbsmt_conf_template_fname="$XLIT_HOME/scripts/supervised_stage2_moses_template.conf"
fillup_tun_template_fname="$XLIT_HOME/scripts/fillup_tun_template_moses.ini"
mdp_either_tun_template_fname="$XLIT_HOME/scripts/mdp_either_tun_template_moses.ini"
mdp_union_tun_template_fname="$XLIT_HOME/scripts/mdp_union_tun_template_moses.ini"
dyn_backoff_tun_template_fname="$XLIT_HOME/scripts/dyn_backoff_tun_template_moses.ini"

########## FUNCTIONS ############

function top1() {
    iter_dir=$1
    combmethod=$2

    working_dir="$iter_dir/combinations/$combmethod/tuning/" 
    start_ini_fname="$iter_dir/workspace/0/moses_data/model/moses.ini" 
    final_ini_fname="$working_dir/moses.ini" 
}

function fillup() {
    iter_dir=$1
    combmethod=$2

    cmdline=""
    for corpusno in `seq 0 $((topk-1))`
    do 
        cmdline="$cmdline $iter_dir/workspace/$corpusno/moses_data/model/phrase-table"
    done 
    $MOSES_DIR/contrib/combine-ptables/combine-ptables.pl --mode=fillup $cmdline > $iter_dir/combinations/$combmethod/phrase-table ;

    #### create moses.ini file for the combined table 
    python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_moses_ini_params \
            $fillup_tun_template_fname \
            $iter_dir/combinations/$combmethod/moses_$combmethod.ini \
            13 \
            $iter_dir/combinations/$combmethod/phrase-table \
            $lm_fname  \
            5 

    working_dir="$iter_dir/combinations/$combmethod/tuning/" 
    start_ini_fname="$iter_dir/combinations/$combmethod/moses_$combmethod.ini" 
    final_ini_fname="$working_dir/moses.ini" 
}

function dynamic_backoff() {
    iter_dir=$1
    combmethod=$2

    #### create moses.ini file for the combined table 
    python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_moses_ini_params \
            $dyn_backoff_tun_template_fname \
            $iter_dir/combinations/$combmethod/moses_$combmethod.ini \
            4 \
            $iter_dir/workspace \
            $lm_fname  \
            5 
    
    working_dir="$iter_dir/combinations/$combmethod/tuning/" 
    start_ini_fname="$iter_dir/combinations/$combmethod/moses_$combmethod.ini" 
    final_ini_fname="$working_dir/moses.ini" 
}

function dynamic_selection_indecoder() {
    iter_dir=$1
    combmethod=$2

    ##### create moses.ini file for the combined table 
    python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_moses_ini_params \
            $mdp_either_tun_template_fname \
            $iter_dir/combinations/$combmethod/moses_$combmethod.ini \
            4 \
            $iter_dir/workspace \
            $lm_fname  \
            5 
    
    working_dir="$iter_dir/combinations/$combmethod/tuning/" 
    start_ini_fname="$iter_dir/combinations/$combmethod/moses_$combmethod.ini" 
    final_ini_fname="$working_dir/moses.ini" 
}

function dynamic_combination_indecoder() {
    iter_dir=$1
    combmethod=$2

    #### create moses.ini file for the combined table 
    python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_moses_ini_params \
            $mdp_union_tun_template_fname \
            $iter_dir/combinations/$combmethod/moses_$combmethod.ini \
            40 \
            $iter_dir/workspace \
            $lm_fname  \
            5 
    
    working_dir="$iter_dir/combinations/$combmethod/tuning/" 
    start_ini_fname="$iter_dir/combinations/$combmethod/moses_$combmethod.ini" 
    final_ini_fname="$working_dir/moses.ini" 
}

########## FUNCTIONS END ############


#### Initialize for the first iteration 

## for starting with first iteration 
#src_corpus=$corpus_dir/all.$src_lang  
#tgt_corpus=$corpus_dir/all.nbest.reranked.$tgt_lang  
#citer=1

#### for starting with any other iteration, the previous iteration must have generated its virtual corpus
src_corpus=$corpus_dir/all.$src_lang  
tgt_corpus=$output_dir/1/next_iter_decoded/dynamic_combination_indecoder/all.nbest.reranked.$tgt_lang  
citer=2

#combmethod_list="top1 fillup  dynamic_backoff dynamic_selection_indecoder dynamic_combination_indecoder"
combmethod_list="dynamic_combination_indecoder"

###### Iterations start ######
while true 
do

    ## iteration base dir 
    iter_dir=$output_dir/$citer
   
    ######## TRAIN 
    ### tuning files - the last 1000 sentences in the first split
    src_tune_ref="$iter_dir/corpus/0/tun.$src_lang" 
    tgt_tune_ref="$iter_dir/corpus/0/tun.$tgt_lang" 
    
    ## make required directories 
    mkdir -p $iter_dir/{corpus,evaluation,workspace,log,conf}
    mkdir -p $iter_dir/combinations/{top1,fillup,dynamic_selection_indecoder,dynamic_combination_indecoder,dynamic_combination_postdecoder,dynamic_backoff}
    mkdir -p $iter_dir/evaluation/{top1,fillup,dynamic_selection_indecoder,dynamic_combination_indecoder,dynamic_combination_postdecoder,dynamic_backoff}
    
    ### create the corpus for training PBSMT system     
    time python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_synthetic_corpus_split \
                $src_corpus  \
                $tgt_corpus  \
                $iter_dir/corpus \
                $src_lang \
                $tgt_lang \
                $topk \
                $vtuning_set 
   
    ### for each split 
    for corpusno in `seq 0 $((topk-1))`
    do 
        ##### create the configuration file 
        
        python $XLIT_HOME/src/cfilt/transliteration/phrase_based.py create_moses_run_params \
                $pbsmt_conf_template_fname\
                $iter_dir/conf/pbmst_$corpusno.conf \
                $iter_dir/workspace/$corpusno/ \
                $iter_dir/corpus/$corpusno \
                $lm_fname  \
                $src_lang \
                $tgt_lang \
        
        ### train the PBSMT model
        #NOTE: this can be marked as notun
        nohup $JOBSCRIPT $iter_dir/conf/pbmst_$corpusno.conf notune > $iter_dir/log/pbsmt_train_$corpusno.log  2>&1  
    
        gunzip -c $iter_dir/workspace/$corpusno/moses_data/model/phrase-table.gz >  $iter_dir/workspace/$corpusno/moses_data/model/phrase-table
        echo ;
    done 
    
    
    for combmethod in `echo $combmethod_list`
    do
        #### COMBINE MODELS 
        case $combmethod in 
            top1) top1 $iter_dir $combmethod 
                ;;
            fillup) fillup $iter_dir $combmethod 
                ;;
            dynamic_backoff) dynamic_backoff $iter_dir $combmethod 
                ;;
            dynamic_selection_indecoder) dynamic_selection_indecoder $iter_dir $combmethod 
                ;;
            dynamic_combination_indecoder) dynamic_combination_indecoder $iter_dir $combmethod 
                ;;
            *) echo "Unknown command: $combmethod" 
                ;;
        esac 
    
        ####### this code is same to all the combination methods 
        #### TUNE 
        ### set working_dir and start_ini_fname variables 
        $SCRIPTS_ROOTDIR/training/mert-moses.pl \
                $src_tune_ref \
                $tgt_tune_ref \
                "$MOSES_CMD"  \
                $start_ini_fname \
                --working-dir $working_dir \
                --batch-mira \
                 --rootdir "$SCRIPTS_ROOTDIR" \
                --decoder-flags='-threads 20 -distortion-limit 0'
      
        eval_output_dir=$output_dir  

        ###### EVALUATION 
        working_dir="$iter_dir/combinations/$combmethod/tuning/" 
        final_ini_fname="$working_dir/moses.ini" 

        ref_xml=$test_data_dir/test.xml
        eval_iter_dir=$eval_output_dir/$citer
        result_dir=$eval_iter_dir/evaluation/$combmethod 

        echo $ref_xml

        mkdir -p $eval_iter_dir/{evaluation,log}
        mkdir -p $eval_iter_dir/evaluation/{top1,fillup,dynamic_selection_indecoder,dynamic_combination_indecoder,dynamic_combination_postdecoder,dynamic_backoff}

        ##### evaluate on test set
        # set final_ini_fname and evaluation_dir variables 
        $MOSES_CMD -n-best-list $result_dir/test.nbest.reranked.$tgt_lang 10 distinct -f $final_ini_fname \
            < $test_data_dir/test.$src_lang > $result_dir/test.$tgt_lang 2> $eval_iter_dir/log/pbsmt_evaluate_$combmethod.log 
        
        # generate NEWS 2015 evaluation format output file 
        python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
                "$test_data_dir/test.id" \
                "$ref_xml" \
                "$result_dir/test.nbest.reranked.$tgt_lang" \
                "$result_dir/test.nbest.reranked.$tgt_lang.xml" \
                "pbsmt" "testset" "$src_lang" "$tgt_lang"  
        
        # run evaluation 
        python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
                -t $ref_xml \
                -i $result_dir/test.nbest.reranked.$tgt_lang.xml \
                -o $result_dir/test.nbest.reranked.$tgt_lang.detaileval.csv \
                >  $result_dir/test.nbest.reranked.$tgt_lang.eval 
    
    done 
    
    #### break if all iterations done
    echo Completed iteration $citer of $niter 
    
    if [ $citer -eq $niter ]
    then 
        break 
    fi 
    
    ### PREPARING FOR NEXT ITERATION
    ### update the variables to the new file 
    src_corpus=$corpus_dir/all.$src_lang  
    tgt_corpus=$iter_dir/next_iter_decoded/dynamic_combination_indecoder/all.nbest.reranked.$tgt_lang  
    for combmethod in `echo dynamic_combination_indecoder`
    do
        
        ####### create virtual corpus for next iteration 
        working_dir="$iter_dir/combinations/$combmethod/tuning/" 
        final_ini_fname="$working_dir/moses.ini" 
        mkdir -p $iter_dir/next_iter_decoded/$combmethod 
        $MOSES_CMD -n-best-list $iter_dir/next_iter_decoded/$combmethod/all.nbest.reranked.$tgt_lang 10 distinct -f $final_ini_fname \
            < $src_corpus  > $iter_dir/next_iter_decoded/$combmethod/all.$tgt_lang 2> $iter_dir/log/next_iter_decoded_$combmethod.log 
    done 

    ### update citer 
    citer=$((citer+1))
done 


