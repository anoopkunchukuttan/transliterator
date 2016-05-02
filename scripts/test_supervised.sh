#!/bin/bash 

## read config file 
. $1 

MOSES_CMD="/usr/local/bin/smt/mosesdecoder-latest-25Dec2015/bin/moses"

##### evaluate on test set
$MOSES_CMD -n-best-list $workspace_dir/evaluation/test.nbest.reranked.$tgt_lang 10 distinct -f $workspace_dir/tuning/moses.ini \
    < $data_dir/test.$src_lang > $workspace_dir/evaluation/test.$tgt_lang 2> $workspace_dir/log/pbsmt_evaluate.log 

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
