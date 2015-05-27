#!/bin/bash

## creating language model
#for lang in `echo bn kn hi ta`
#do 
#        corpus=/home/anoop/experiments/news_2015/news_2015_match_brahminet/data/processed/en-$lang/train.$lang
#        lm_file=/home/anoop/experiments/news_2015/news_2015_match_brahminet/data/lm/5-gram/$lang.lm
#        ngram-count -order 5 -interpolate -wbdiscount -text $corpus -lm $lm_file &
#done 

## select corpus
#for lang in `echo bn hi ta`
#do 
#        incorpus=/home/anoop/experiments/news_2015/news_2015/data/processed/en-$lang/
#        outcorpus=/home/anoop/experiments/news_2015/news_2015_match_brahminet/data/processed/en-$lang/
#        python utilities.py randomize_and_select   $incorpus/train.en       \
#                                                   $incorpus/train.$lang    \
#                                                   $outcorpus/train.en      \
#                                                   $outcorpus/train.$lang   \
#                                                   `wc -l /home/anoop/experiments/news_2015/brahminet/data/processed/en-$lang/train.en | cut -f 1 -d ' '`
#done 

## merge corpus
#for lang in `echo bn hi ta`
#do 
#        incorpus1=/home/anoop/experiments/news_2015/news_2015/data/processed/en-$lang/
#        incorpus2=/home/anoop/experiments/news_2015/brahminet/data/processed/en-$lang/
#        outcorpus=/home/anoop/experiments/news_2015/news_2015_indic+brahminet/data/processed/en-$lang/
#        mkdir -p $outcorpus
#        cat  $incorpus1/train.en $incorpus2/train.en  > $outcorpus/train.en
#        cat  $incorpus1/train.$lang $incorpus2/train.$lang  > $outcorpus/train.$lang
#done 

## create ngram corpus
#basedata=/home/anoop/experiments/news_2015/news_2015_indic/data/processed
#order=3
#overlap='True'
#outdir=/home/anoop/experiments/news_2015/news_2015_indic/data/${order}_gram/$overlap
#
#for lang in `echo bn hi ta kn`
#do 
#    mkdir -p $outdir/en-$lang
#    python utilities.py create_ngram  $basedata/en-$lang/train.$lang $outdir/en-$lang/train.$lang $order $overlap
#    python utilities.py create_ngram  $basedata/en-$lang/test.$lang $outdir/en-$lang/test.$lang $order $overlap
#    python utilities.py create_ngram  $basedata/en-$lang/train.en $outdir/en-$lang/train.en $order $overlap
#    python utilities.py create_ngram  $basedata/en-$lang/test.en $outdir/en-$lang/test.en $order $overlap
#done 

### convert output to format required for evaluation
#
## pb
#python utilities.py gen_news_output \
#                    ../../brahminet_length_ratio_std_1/data/processed/en-hi/test.id  \
#                    ../../brahminet_length_ratio_std_1/data/processed/en-hi/test.en \
#                    ../../brahminet_length_ratio_std_1/results/pb/en-hi/evaluation/test_no_tun.hi  \
#                    ../../brahminet_length_ratio_std_1/results/pb/en-hi/evaluation/test_no_tun.hi.xml
#
#python evaluation_script/news_evaluation.py \
#                -i ../../brahminet_length_ratio_std_1/results/pb/en-hi/evaluation/test_no_tun.hi.xml \
#                -t ../../brahminet_length_ratio_std_1/data/raw/NEWS15_dev_EnHi_997.xml \
#                -o ../../brahminet_length_ratio_std_1/results/pb/en-hi/evaluation/evaluation_details_no_tun.csv
#


## 2 gram - False 
#python utilities.py postedit_ngram_output \
#                                          ../../brahminet/results/2_gram_False/en-hi/evaluation/test_no_tun.hi \
#                                          ../../brahminet/results/2_gram_False/en-hi/evaluation/test_no_tun.hi.deseg \
#                                          2 False  
#
#python utilities.py gen_news_output \
#                    ../../brahminet/data/processed/en-hi/test.id \
#                    ../../brahminet/data/processed/en-hi/test.en \
#                    ../../brahminet/results/2_gram_False/en-hi/evaluation/test_no_tun.hi.deseg  \
#                    ../../brahminet/results/2_gram_False/en-hi/evaluation/test_no_tun.hi.xml

#python evaluation_script/news_evaluation.py \
#                -i ../../brahminet/results/2_gram_False/en-hi/evaluation/test_no_tun.hi.xml \
#                -t ../../brahminet/data/raw/NEWS15_dev_EnHi_997.xml \
#                -o ../../brahminet/results/2_gram_False/en-hi/evaluation/evaluation_details_no_tun.csv


## 2 gram - True 
#python utilities.py postedit_ngram_output \
#                                          ../../brahminet/results/2_gram_True/en-hi/evaluation/test_no_tun.hi \
#                                          ../../brahminet/results/2_gram_True/en-hi/evaluation/test_no_tun.hi.deseg \
#                                          2 True
#
#python utilities.py gen_news_output \
#                    ../../brahminet/data/processed/en-hi/test.id \
#                    ../../brahminet/data/processed/en-hi/test.en \
#                    ../../brahminet/results/2_gram_True/en-hi/evaluation/test_no_tun.hi.deseg  \
#                    ../../brahminet/results/2_gram_True/en-hi/evaluation/test_no_tun.hi.xml

#python evaluation_script/news_evaluation.py \
#                -i ../../brahminet/results/2_gram_True/en-hi/evaluation/test_no_tun.hi.xml \
#                -t ../../brahminet/data/raw/NEWS15_dev_EnHi_997.xml \
#                -o ../../brahminet/results/2_gram_True/en-hi/evaluation/evaluation_details_no_tun.csv


## 3 gram - True 
#python utilities.py postedit_ngram_output \
#                                          ../../news_2015_indic/results/3_gram_True/en-hi/evaluation/test_no_tun.hi \
#                                          ../../news_2015_indic/results/3_gram_True/en-hi/evaluation/test_no_tun.hi.deseg \
#                                          3 True
#
#python utilities.py gen_news_output \
#                    ../../news_2015_indic/data/processed/en-hi/test.id \
#                    ../../news_2015_indic/data/processed/en-hi/test.en \
#                    ../../news_2015_indic/results/3_gram_True/en-hi/evaluation/test_no_tun.hi.deseg  \
#                    ../../news_2015_indic/results/3_gram_True/en-hi/evaluation/test_no_tun.hi.xml
#
#python evaluation_script/news_evaluation.py \
#                -i ../../news_2015_indic/results/3_gram_True/en-hi/evaluation/test_no_tun.hi.xml \
#                -t ../../news_2015_indic/data/raw/NEWS15_dev_EnHi_997.xml \
#                -o ../../news_2015_indic/results/3_gram_True/en-hi/evaluation/evaluation_details_no_tun.csv



#### Prepare Brahminet corpus 
#
## creating language model
#brahminet_dir=/home/ratish/mined_pairs
#for lang in `echo bn hi ta`
#do
#        outdir=/home/anoop/experiments/news_2015/brahminet/data/processed/en-$lang
#        python utilities.py prepare_brahminet_training_corpus $brahminet_dir/en-$lang.csv $outdir/train.en $outdir/train.$lang
#done 
#



## pb
#python utilities.py gen_news_output \
#                    /home/anoop/experiments/news_2015/tmp/tuning/data/en-hi/test.id  \
#                    /home/anoop/experiments/news_2015/tmp/tuning/data/en-hi/test.en \
#                    /home/anoop/experiments/news_2015/tmp/tuning/results/en-hi/evaluation/test_no_tun.hi  \
#                    /home/anoop/experiments/news_2015/tmp/tuning/results/en-hi/evaluation/test_no_tun.hi.xml
#
#python evaluation_script/news_evaluation.py \
#                -i /home/anoop/experiments/news_2015/tmp/tuning/results/en-hi/evaluation/test_no_tun.hi.xml \
#                -t /home/anoop/experiments/news_2015/tmp/tuning/data/raw/NEWS15_dev_EnHi_997.xml \
#                -o /home/anoop/experiments/news_2015/tmp/tuning/results/en-hi/evaluation/evaluation_details_no_tun.csv


### 

#src='en'
#tgt='hi'
#outdir="/home/anoop/experiments/news_2015/tmp/tuning/results/2_gram_True/${src}-${tgt}"
#refdir="/home/anoop/experiments/news_2015/news_2015_indic/data/processed/${src}-${tgt}"
#testxml="/home/anoop/experiments/news_2015/tmp/tuning/data/raw/NEWS15_dev_EnHi_997.xml"
#
#cp $outdir/evaluation/test_no_tun.${tgt} $outdir/evaluation/test.${tgt}.bak
#
## post-edit for ngrams 
#python utilities.py postedit_ngram_output $outdir/evaluation/test.${tgt}
#
## remove markers
#python utilities.py remove_markers $outdir/evaluation/test.${tgt}
#
## generate NEWS output
#python utilities.py gen_news_output \
#                    $refdir/test.id \
#                    $refdir/test.${src} \
#                    $outdir/evaluation/test.${tgt}  \
#                    $outdir/evaluation/test.${tgt}.xml
#
## run evaluation 
#python evaluation_script/news_evaluation.py \
#                -i $outdir/evaluation/test.$tgt.xml \
#                -t $testxml \
#                -o $outdir/evaluation/evaluation_details.csv
#
#

DATASET='news_2015'

####  create training and tuning set 
#
#datadir="/home/development/anoop/experiments/news_2015/$DATASET/data/pb"
#echo $datadir
#
#while read line
#do 
#    src_lang=`echo $line | cut -d '-' -f '1'`
#    tgt_lang=`echo $line | cut -d '-' -f '2'`
#
#    python utilities.py create_train_tun  \
#                        $datadir/$src_lang-$tgt_lang \
#                        $src_lang $tgt_lang 
#
#done 


### create marker corpus 
#datadir="/home/development/anoop/experiments/news_2015/$DATASET/data/marker"
#
#while read line
#do 
#    src_lang=`echo $line | cut -d '-' -f '1'`
#    tgt_lang=`echo $line | cut -d '-' -f '2'`
#
#    python utilities.py add_markers_corpus \
#                        $datadir/$src_lang-$tgt_lang \
#                        $src_lang $tgt_lang 
#
#done 

#### create 2 gram corpus
#datadir="/home/development/anoop/experiments/news_2015/$DATASET/data/2gram"
#
#while read line
#do 
#    src_lang=`echo $line | cut -d '-' -f '1'`
#    tgt_lang=`echo $line | cut -d '-' -f '2'`
#
#    python utilities.py create_ngram_corpus \
#                        $datadir/$src_lang-$tgt_lang \
#                        $src_lang $tgt_lang 
#
#done 

#### create marker+2 gram corpus  
#datadir="/home/development/anoop/experiments/news_2015/$DATASET/data/marker+2gram"
#
#while read line
#do 
#    src_lang=`echo $line | cut -d '-' -f '1'`
#    tgt_lang=`echo $line | cut -d '-' -f '2'`
#
#    python utilities.py create_ngram_corpus \
#                        $datadir/$src_lang-$tgt_lang \
#                        $src_lang $tgt_lang 
#
#done 

## train the MT systems 
#for systemtype in `echo 2gram  marker  marker+2gram  pb`
#do     
#    mkdir -p /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype
#echo $systemtype  started   
#/usr/local/bin/smt/moses_job_scripts/batch_train_scripts/run_for_lang_pairs.sh \
#    /home/development/anoop/experiments/news_2015/$DATASET/data/lang_pairs.txt \
#    /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype \
#    $PWD/confs/$systemtype.conf  > $systemtype.log 2>&1 
#echo $systemtype  ended
#done 
#
## tune the systems
#for systemtype in `echo 2gram  marker  marker+2gram  pb`
#do 
#cat /home/development/anoop/experiments/news_2015/$DATASET/data/lang_pairs.txt | parallel --gnu --colsep '-' -j 8 --joblog $systemtype.joblog "/usr/local/bin/smt/moses_job_scripts/moses_run.sh /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/run_params.conf notrain > $systemtype.tun.log"
#done 

#### generate n-best output for untuned system
#
#for systemtype in `echo 2gram  marker  marker+2gram  pb`
#do 
#cat /home/development/anoop/experiments/news_2015/$DATASET/data/lang_pairs.txt | parallel  --gnu --colsep '-' -j 8 --joblog $systemtype.joblog "python utilities.py \
#    convert_to_nbest_format \
#    /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.{2} \
#    /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.nbest.{2} "
#done 

# backup 

#for systemtype in `echo 2gram  marker  marker+2gram  pb`
#do 
### for tuned output
##cat /home/development/anoop/experiments/news_2015/$DATASET/data/lang_pairs.txt | parallel  --gnu --colsep '-' -j 8 --joblog $systemtype.joblog "cp /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test.nbest.{2} /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test.nbest.{2}.bak"
#
## for untuned output
#cat /home/development/anoop/experiments/news_2015/$DATASET/data/lang_pairs.txt | parallel  --gnu --colsep '-' -j 8 --joblog $systemtype.joblog "cp /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.nbest.{2} /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.nbest.{2}.bak"
#done 



### post edit output 

#systemtype='marker+2gram'
#
### for tuned system
##cat /home/development/anoop/experiments/news_2015/$DATASET/data/lang_pairs.txt | parallel  --gnu --colsep '-' -j 8 --joblog $systemtype.joblog "python utilities.py postprocess_nbest_list /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test.nbest.{2} $systemtype > $systemtype.tun.log"
#
##  for untuned system
#cat /home/development/anoop/experiments/news_2015/$DATASET/data/lang_pairs.txt | parallel  --gnu --colsep '-' -j 8 --joblog $systemtype.joblog "python utilities.py postprocess_nbest_list /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.nbest.{2} $systemtype > $systemtype.tun.log"
#


##### generate output xml 
#for systemtype in `echo 2gram  marker  marker+2gram  pb`
#do 
#    ## for tuned system 
#    #cat /home/development/anoop/experiments/news_2015/$DATASET/data/file_info.txt | parallel  --gnu --colsep '\|' -j 8 --joblog $systemtype.joblog "python utilities.py gen_news_output /home/development/anoop/experiments/news_2015/news_2015/data/pb/{1}-{2}/test.id  \
#    #/home/development/anoop/experiments/news_2015/news_2015/data/raw/{4} \
#    #/home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test.nbest.{2} \
#    #/home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test.nbest.{2}.xml \
#    #$systemtype $DATASET {1} {2} "
#
#    # for untuned system 
#    cat /home/development/anoop/experiments/news_2015/$DATASET/data/file_info.txt | parallel  --gnu --colsep '\|' -j 8 --joblog $systemtype.joblog "python utilities.py gen_news_output /home/development/anoop/experiments/news_2015/news_2015/data/pb/{1}-{2}/test.id  \
#    /home/development/anoop/experiments/news_2015/news_2015/data/raw/{4} \
#    /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.nbest.{2} \
#    /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.nbest.{2}.xml \
#    $systemtype $DATASET {1} {2} "
#
#done
#
#
### evaluate output 
#
#for systemtype in `echo 2gram  marker  marker+2gram  pb`
#do 
#    ## for tuned system 
#    #cat /home/development/anoop/experiments/news_2015/$DATASET/data/file_info.txt | parallel --gnu --colsep '\|' -j 8 --joblog $systemtype.joblog "python  evaluation_script/news_evaluation.py \
#    #-t /home/development/anoop/experiments/news_2015/news_2015/data/raw/{4} \
#    #-i /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test.nbest.{2}.xml \
#    #-o /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test.nbest.{2}.detaileval.csv \
#    #> /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test.nbest.{2}.eval "
#
#    # for untuned system 
#    cat /home/development/anoop/experiments/news_2015/$DATASET/data/file_info.txt | parallel --gnu --colsep '\|' -j 8 --joblog $systemtype.joblog "python  evaluation_script/news_evaluation.py \
#    -t /home/development/anoop/experiments/news_2015/news_2015/data/raw/{4} \
#    -i /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.nbest.{2}.xml \
#    -o /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.nbest.{2}.detaileval.csv \
#    > /home/development/anoop/experiments/news_2015/$DATASET/results/$systemtype/{1}-{2}/evaluation/test_no_tun.nbest.{2}.eval "
#
#done


#### gather evaluation output 
#datadir="/home/development/anoop/experiments/news_2015/$DATASET/"
#python utilities.py gather_evaluation_results \
#            $datadir/results/ \
#            $datadir/data/lang_pairs.txt \
#            $datadir/eval.csv


#####################################################################################


## create new15+brahminet dataset
# tuning and test from news15, training is concatenation of the two
### create 2 gram corpus
#cp -r /home/development/anoop/experiments/news_2015/news_2015/data /home/development/anoop/experiments/news_2015/news_2015+brahminet/

#for systemtype in `echo 2gram  marker  marker+2gram  pb`
#do 
#    while read line
#    do 
#        src_lang=`echo $line | cut -d '-' -f '1'`
#        tgt_lang=`echo $line | cut -d '-' -f '2'`
#   
#        cat /home/development/anoop/experiments/news_2015/news_2015/data/$systemtype/$src_lang-$tgt_lang/train.$src_lang \
#            /home/development/anoop/experiments/news_2015/brahminet/data/$systemtype/$src_lang-$tgt_lang/train.$src_lang \
#            /home/development/anoop/experiments/news_2015/brahminet/data/$systemtype/$src_lang-$tgt_lang/tun.$src_lang \
#            > /home/development/anoop/experiments/news_2015/news_2015+brahminet/data/$systemtype/$src_lang-$tgt_lang/train.$src_lang  
#
#        cat /home/development/anoop/experiments/news_2015/news_2015/data/$systemtype/$src_lang-$tgt_lang/train.$tgt_lang \
#            /home/development/anoop/experiments/news_2015/brahminet/data/$systemtype/$src_lang-$tgt_lang/train.$tgt_lang \
#            /home/development/anoop/experiments/news_2015/brahminet/data/$systemtype/$src_lang-$tgt_lang/tun.$tgt_lang \
#            > /home/development/anoop/experiments/news_2015/news_2015+brahminet/data/$systemtype/$src_lang-$tgt_lang/train.$tgt_lang  
#
#
#    done <  /home/development/anoop/experiments/news_2015/brahminet/data/lang_pairs.txt
#done 


##### preparing official test corpus
#rawdir='/home/development/anoop/experiments/news_2015/official_test_set_news12/data/raw'
#outdir='/home/development/anoop/experiments/news_2015/official_test_set_news12/data/processed'
#
#for f in `ls $rawdir`
#do 
#    python utilities.py parse $rawdir/$f $outdir "test" $f  $f
#done 

#### decoding the test sets 
##for model in `echo brahminet  news_2015+brahminet news_2015`
#for model in `echo brahminet  news_2015+brahminet`
##for model in `echo news_2015`
#do     
#    for systemtype in `echo 2gram  marker  marker+2gram  pb`
#    do 
#
#        #cat /home/development/anoop/experiments/news_2015/$model/data/file_info.txt | parallel --gnu --colsep '\|' -j 8 --joblog $systemtype.joblog "mkdir -p /home/development/anoop/experiments/news_2015/$DATASET/results/$model/$systemtype/{1}-{2}/evaluation"
#        #cat /home/development/anoop/experiments/news_2015/$model/data/file_info.txt | parallel --gnu --colsep '\|' -j 8 --joblog $systemtype.joblog "/usr/local/bin/smt/mosesdecoder-2.1.1/bin/moses \
#        #    -f /home/development/anoop/experiments/news_2015/$model/results/$systemtype/{1}-{2}/tuning/moses.ini \
#        #    -n-best-list /home/development/anoop/experiments/news_2015/$DATASET/results/$model/$systemtype/{1}-{2}/evaluation/test.nbest.{2} 10 distinct \
#        #    < /home/development/anoop/experiments/news_2015/$DATASET/data/$systemtype/{1}-{2}/test.{1} \
#        #    > /home/development/anoop/experiments/news_2015/$DATASET/results/$model/$systemtype/{1}-{2}/evaluation/test.{2}"
#
#    
#        ## backup 
#        #cat /home/development/anoop/experiments/news_2015/$model/data/file_info.txt | parallel --gnu --colsep '\|' -j 8 --joblog $systemtype.joblog "cp \
#        #    /home/development/anoop/experiments/news_2015/$DATASET/results/$model/$systemtype/{1}-{2}/evaluation/test.nbest.{2} \
#        #    /home/development/anoop/experiments/news_2015/$DATASET/results/$model/$systemtype/{1}-{2}/evaluation/test.nbest.{2}.bak"
#
#        ### postprocess
#        #cat /home/development/anoop/experiments/news_2015/$model/data/file_info.txt | parallel --gnu --colsep '\|' -j 8 --joblog $systemtype.joblog " python \
#        #    utilities.py postprocess_nbest_list \
#        #    /home/development/anoop/experiments/news_2015/$DATASET/results/$model/$systemtype/{1}-{2}/evaluation/test.nbest.{2} $systemtype"
#
#
#        cat /home/development/anoop/experiments/news_2015/$DATASET/data/file_info2.txt | parallel --gnu --colsep '\|' -j 8 --joblog $systemtype.joblog "python utilities.py gen_news_output \
#        /home/development/anoop/experiments/news_2015/$DATASET/data/pb/{1}-{2}/test.id  \
#        /home/development/anoop/experiments/news_2015/$DATASET/data/raw/{4} \
#        /home/development/anoop/experiments/news_2015/$DATASET/results/$model/$systemtype/{1}-{2}/evaluation/test.nbest.{2} \
#        /home/development/anoop/experiments/news_2015/$DATASET/results/$model/$systemtype/{1}-{2}/evaluation/test.nbest.{2}.xml \
#        $systemtype $DATASET {1} {2} "
#    done
#done


########### Collecting together the files for submission
#
#ds=official_test_set_news12
#
## Run 1: marker for all languages trained on NEWS dataset : Standard Submission (Primary)
#langinfo_fname=/home/development/anoop/experiments/news_2015/$ds/data/file_info.txt
#resultdir=/home/development/anoop/experiments/news_2015/$ds/results/news_2015/marker
#outdir='news_2015_output'
#run="1"
#
#python utilities.py copy_output $langinfo_fname $resultdir $outdir $run
#
## Run 2: marker+2gram for all languages trained on NEWS dataset : Standard Submission 
#langinfo_fname=/home/development/anoop/experiments/news_2015/$ds/data/file_info.txt
#resultdir=/home/development/anoop/experiments/news_2015/$ds/results/news_2015/marker+2gram
#outdir='news_2015_output'
#run="2"
#
#python utilities.py copy_output $langinfo_fname $resultdir $outdir $run
#
## Run 3: marker for 3 languages trained on NEWS+Brahminet dataset : Non-Standard Submission 
#langinfo_fname=/home/development/anoop/experiments/news_2015/$ds/data/file_info2.txt
#resultdir=/home/development/anoop/experiments/news_2015/$ds/results/news_2015+brahminet/marker
#outdir='news_2015_output'
#run="3"
#
#python utilities.py copy_output $langinfo_fname $resultdir $outdir $run
#
## Run 4: marker+2gram for 3 languages trained on NEWS+Brahminet dataset : Non-Standard Submission 
#langinfo_fname=/home/development/anoop/experiments/news_2015/$ds/data/file_info2.txt
#resultdir=/home/development/anoop/experiments/news_2015/$ds/results/news_2015+brahminet/marker+2gram
#outdir='news_2015_output'
#run="4"
#
#python utilities.py copy_output $langinfo_fname $resultdir $outdir $run
#
## Run 5: marker for 3 languages trained on Brahminet dataset : Non-Standard Submission 
#langinfo_fname=/home/development/anoop/experiments/news_2015/$ds/data/file_info2.txt
#resultdir=/home/development/anoop/experiments/news_2015/$ds/results/brahminet/marker
#outdir='news_2015_output'
#run="5"
#
#python utilities.py copy_output $langinfo_fname $resultdir $outdir $run
#
## Run 6: marker+2gram for 3 languages trained on Brahminet dataset : Non-Standard Submission 
#langinfo_fname=/home/development/anoop/experiments/news_2015/$ds/data/file_info2.txt
#resultdir=/home/development/anoop/experiments/news_2015/$ds/results/brahminet/marker+2gram
#outdir='news_2015_output'
#run="6"
#
#python utilities.py copy_output $langinfo_fname $resultdir $outdir $run
#

### comparing brahibet and news2015
#model_file='/home/development/anoop/experiments/news_2015/news_2015/results/pb/En-Hi/tuning/moses.ini'
#WORKSPACE_DIR="/home/development/anoop/experiments/news_2015/compare_brahminet_news_2015/compare_arjun_corpus/results/news_2015/"
#
#parallel_corpus="/home/development/anoop/experiments/news_2015/compare_brahminet_news_2015/compare_arjun_corpus/data/hi-en"
#SRC_LANG='en'
#TGT_LANG='hi'
#
#/usr/local/bin/smt/mosesdecoder-2.1.1/bin/moses -config "$model_file" \
#               -input-file "$parallel_corpus/test.$SRC_LANG" \
#	           -alignment-output-file "$WORKSPACE_DIR/evaluation/test.align.$TGT_LANG" \
#	           -n-best-list "$WORKSPACE_DIR/evaluation/test.nbest.$TGT_LANG" 10 distinct \
#	           -output-unknowns "$WORKSPACE_DIR/evaluation/test.oov.$TGT_LANG" \
#               -threads 25  > \
#               "$WORKSPACE_DIR/evaluation/test.$TGT_LANG" 2> $WORKSPACE_DIR/log



##### generate output xml 
#python utilities.py gen_news_output $parallel_corpus/test.id  \
# $parallel_corpus/test.xml \
# $WORKSPACE_DIR/evaluation/test.nbest.$TGT_LANG \
# $WORKSPACE_DIR/evaluation/test.nbest.$TGT_LANG.xml \
#pb brahminet $SRC_LANG $TGT_LANG
#
### evaluate output 
#python  evaluation_script/news_evaluation.py \
# -t $parallel_corpus/test.xml \
# -i $WORKSPACE_DIR/evaluation/test.nbest.$TGT_LANG.xml \
# -o $WORKSPACE_DIR/evaluation/test.nbest.$TGT_LANG.detaileval.csv \
# >  $WORKSPACE_DIR/evaluation/test.nbest.$TGT_LANG.eval
#
#
