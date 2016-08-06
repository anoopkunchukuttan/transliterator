#!/bin/bash 

## read config file 
. $1 

MOSES_CMD="/usr/local/bin/smt/mosesdecoder-latest-25Dec2015/bin/moses"

## create the directory to store the final results 
mkdir -p $base_dir/final/{input,evaluation,log}

### decode src into bridge  
workspace_dir=$base_dir/$src_lang-$bridge_lang
$MOSES_CMD -n-best-list $workspace_dir/evaluation/test.nbest.reranked.$bridge_lang \
        10 distinct -f $workspace_dir/tuning/moses.ini \
    < $data_dir/test.$src_lang > $workspace_dir/evaluation/test.$bridge_lang 2> $workspace_dir/log/pbsmt_evaluate.log 

### get the input for next stage 
sed  's/ ||| /|/g;s/ |/|/g' $workspace_dir/evaluation/test.nbest.reranked.$bridge_lang | \
    cut -d'|' -f2  > $workspace_dir/evaluation/input_second_stage.$bridge_lang 
output_1=$workspace_dir/evaluation/test.nbest.reranked.$bridge_lang 
input_2nd_fname=$workspace_dir/evaluation/input_second_stage.$bridge_lang

#### decode brdige into target
workspace_dir=$base_dir/$bridge_lang-$tgt_lang
$MOSES_CMD -n-best-list $workspace_dir/evaluation/test.nbest.reranked.$tgt_lang \
        10 distinct -f $workspace_dir/tuning/moses.ini \
    < $input_2nd_fname > $workspace_dir/evaluation/test.$tgt_lang 2> $workspace_dir/log/pbsmt_evaluate.log 
output_2=$workspace_dir/evaluation/test.nbest.reranked.$tgt_lang

### creating the final output file 
workspace_dir=$base_dir/final
final_output=$workspace_dir/evaluation/test.nbest.reranked.$tgt_lang
python compute_bridge_sys_output.py $output_1 $output_2 $final_output

# generate NEWS 2015 evaluation format output file 
python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
        "$data_dir/test.id" \
        "$ref_xml" \
        "$workspace_dir/evaluation/test.nbest.reranked.$tgt_lang" \
        "$workspace_dir/evaluation/test.nbest.reranked.$tgt_lang.xml" \
        "$systemtype" "$dataset" "$src_lang" "$tgt_lang"  

# run evaluation 
python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
        -t $ref_xml \
        -i $workspace_dir/evaluation/test.nbest.reranked.$tgt_lang.xml \
        -o $workspace_dir/evaluation/test.nbest.reranked.$tgt_lang.detaileval.csv \
         > $workspace_dir/evaluation/test.nbest.reranked.$tgt_lang.eval 
