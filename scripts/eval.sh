#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:$XLIT_HOME/src

. $1

test_fcorpus_fname="$data_dir/test.$src_lang"
test_ecorpus_fname="$data_dir/test.$tgt_lang"

if [ -d $workspace_dir ]
then     
    rm -rf $workspace_dir
fi 

mkdir -p $workspace_dir/{evaluation,log}

cp $1 $workspace_dir/train.conf

# unsupervised 
# test
time python $XLIT_HOME/src/cfilt/transliteration/transliterate_cli.py transliterate \
    $model_fname $lm_fname $test_fcorpus_fname $workspace_dir/evaluation/test.$tgt_lang > $workspace_dir/log/decode.log 2>&1 

# convert to nbest format 
python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py  convert_to_nbest_format  \
    $workspace_dir/evaluation/test.$tgt_lang $workspace_dir/evaluation/test.nbest.$tgt_lang

# generate NEWS 2015 evaluation format output file 
python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
        $data_dir/test.id \
        $ref_xml \
        $workspace_dir/evaluation/test.nbest.$tgt_lang \
        $workspace_dir/evaluation/test.nbest.$tgt_lang.xml \
        $systemtype $dataset $src_lang $tgt_lang  

# run evaluation 
python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
        -t $ref_xml \
        -i $workspace_dir/evaluation/test.nbest.$tgt_lang.xml \
        -o $workspace_dir/evaluation/test.nbest.$tgt_lang.detaileval.csv \
        > $workspace_dir/evaluation/test.nbest.$tgt_lang.eval 
