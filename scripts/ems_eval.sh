#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:$XLIT_HOME/src

. $1

fcorpus_fname="$data_dir/train.$src_lang"
ecorpus_fname="$data_dir/train.$tgt_lang"
test_fcorpus_fname="$data_dir/test.$src_lang"
test_ecorpus_fname="$data_dir/test.$tgt_lang"

## generate NEWS 2015 evaluation format output file 
#python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#        "$data_dir/test.id" \
#        "$ref_xml" \
#        "$workspace_dir/evaluation/test.nbest.$tgt_lang" \
#        "$workspace_dir/evaluation/test.nbest.$tgt_lang.xml" \
#        "$systemtype" "$dataset" "$src_lang" "$tgt_lang"  
#
## run evaluation 
#python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#        -t $ref_xml \
#        -i $workspace_dir/evaluation/test.nbest.$tgt_lang.xml \
#        -o $workspace_dir/evaluation/test.nbest.$tgt_lang.detaileval.csv \
#        > $workspace_dir/evaluation/test.nbest.$tgt_lang.eval 

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

