#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:$XLIT_HOME/src

. $1

### char LM reranking
python $XLIT_HOME/src/cfilt/transliteration/reranker.py rerank_file \
    $workspace_dir/evaluation/test.50best.$tgt_lang \
    $workspace_dir/evaluation/test.nbest.reranked.$tgt_lang \
    10 $lm_fname 5

#### word LM reranking
#python $XLIT_HOME/src/cfilt/transliteration/reranker.py rerank_wordlm_file \
#    $workspace_dir/evaluation/test.50best.$tgt_lang \
#    $workspace_dir/evaluation/test.nbest.reranked.$tgt_lang \
#    10 $lm_fname $word_lm_fname 5

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

