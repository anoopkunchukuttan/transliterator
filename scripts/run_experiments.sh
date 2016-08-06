#!/bin/bash 

#########  Just a compilation of commands for experiments #### 
##### You can ignore #######

#### phonetic features related 

### create the phrase tables
### WARNING: make sure that code is setup correctly  to use the correct metrics 
#expname='sim1+cos'    
#base_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic"
#
#for langpair in  `echo kn-hi hi-kn`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    mkdir -p $base_dir/pb+phonetic_feat/$expname/$src_lang-$tgt_lang/moses_data/model
#    mkdir -p $base_dir/pb+phonetic_feat/$expname/$src_lang-$tgt_lang/{log,log,evaluation,tuning}
#
#    gunzip -c $base_dir/pb/$src_lang-$tgt_lang/moses_data/model/phrase-table.gz > $base_dir/pb/$src_lang-$tgt_lang/moses_data/model/phrase-table
#
#    python $XLIT_HOME/src/cfilt/transliteration/phonetic_sim.py \
#        $base_dir/pb/$src_lang-$tgt_lang/moses_data/model/phrase-table \
#        $base_dir/pb+phonetic_feat/$expname/$src_lang-$tgt_lang/moses_data/model/phrase-table \
#        $src_lang \
#        $tgt_lang
#
#    #cp  \
#    #    $base_dir/pb/$src_lang-$tgt_lang/run_params.conf \
#    #    $base_dir/pb+phonetic_feat/$expname/$src_lang-$tgt_lang/ \
#
#    #cp  \
#    #    $base_dir/pb/$src_lang-$tgt_lang/moses_data/model/moses.ini \
#    #    $base_dir/pb+phonetic_feat/$expname/$src_lang-$tgt_lang/moses_data/model/ 
#
#done     

### tune the new phrase tables 
#JOBSCRIPT="/usr/local/bin/smt/moses_job_scripts/moses_run.sh"
#base_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic"
#
#for expname in `echo sim1 cos sim1+cos`
#do 
#    for langpair in  `echo kn-hi hi-kn`
#    do 
#        src_lang=`echo $langpair | cut -f 1 -d '-'` 
#        tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#        nohup $JOBSCRIPT $base_dir/pb+phonetic_feat/$expname/$src_lang-$tgt_lang/run_params.conf notrain > log/$expname_$src_lang-$tgt_lang.log 2>&1  
#        nohup ./test_supervised.sh $base_dir/$expname-$src_lang-$tgt_lang.config  > log/$expname_$src_lang-$tgt_lang.test.log 2>&1  
#    done 
#done 

#### FACTOR CORPUS related ###

### list of properties 

PV_PROP='basic_type vowel_length vowel_strength vowel_status consonant_type articulation_place aspiration voicing nasalization vowel_horizontal vowel_vertical vowel_roundness'   

## data 
#lm_corpora_dir=/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/lm_corpora
#lm_corpora_factored_dir=/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/lm_corpora_factored
#lm_factored_dir=/home/development/anoop/experiments/unsupervised_transliterator/data/lm/nonparallel/factored/5g/

#lm_corpora_dir=/home/development/anoop/experiments/phonetic_transliteration/brahminet_transliteration/data/lm_corpora
#lm_corpora_factored_dir=/home/development/anoop/experiments/phonetic_transliteration/brahminet_transliteration/data/lm_corpora_factored
#lm_factored_dir=/home/development/anoop/experiments/phonetic_transliteration/brahminet_transliteration/data/lm/factored/5g/

#### create factored monolingual corpus
#for lang in  `echo bn hi kn ta`
#do 
#    mkdir -p $lm_corpora_factored_dir/$lang
#    python create_factors.py $lm_corpora_dir/train.$lang $lm_corpora_factored_dir/$lang $lang
#done     
#
### create LM for factors 
#for lang in  `echo bn hi kn ta`
#do 
#    for prop in `echo $PV_PROP`
#    do         
#        mkdir -p $lm_factored_dir/$lang
#        ngram-count -text $lm_corpora_factored_dir/$lang/$prop.$lang \
#                -lm $lm_factored_dir/$lang/$prop.$lang.lm \
#                -order 5 -wbdiscount -interpolate
#    done             
#done            

### create factored parallel corpus 
#parallel_corpus='/home/development/anoop/experiments/unsupervised_transliterator/data/parallel_news_2015_indic/pb'
#parallel_factored_corpus='/home/development/anoop/experiments/unsupervised_transliterator/data/parallel_news_2015_indic/factored'
##parallel_corpus='/home/development/anoop/experiments/phonetic_transliteration/brahminet_transliteration/data/pb'
##parallel_factored_corpus='/home/development/anoop/experiments/phonetic_transliteration/brahminet_transliteration/data/factored'

#for langpair in  `echo bn-hi hi-kn kn-hi`
##for langpair in  `echo hi-ta ta-hi ta-kn kn-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    mkdir -p $parallel_factored_corpus/$src_lang-$tgt_lang
#
#    python create_factors.py create_factor_file $parallel_corpus/$src_lang-$tgt_lang/train.$src_lang $parallel_factored_corpus/$src_lang-$tgt_lang/train.$src_lang $src_lang
#    python create_factors.py create_factor_file $parallel_corpus/$src_lang-$tgt_lang/train.$tgt_lang $parallel_factored_corpus/$src_lang-$tgt_lang/train.$tgt_lang $tgt_lang
#
#    python create_factors.py create_factor_file $parallel_corpus/$src_lang-$tgt_lang/tun.$src_lang $parallel_factored_corpus/$src_lang-$tgt_lang/tun.$src_lang $src_lang
#    ## don't add factors for target side of tuning set 
#    cp $parallel_corpus/$src_lang-$tgt_lang/tun.$tgt_lang $parallel_factored_corpus/$src_lang-$tgt_lang/tun.$tgt_lang 
#done     
#
### create factored corpus for test set 
#parallel_corpus='/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb'
#parallel_factored_corpus='/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/factored'
#
#for langpair in  `echo bn-hi hi-kn kn-hi`
##for langpair in  `echo hi-ta ta-hi ta-kn kn-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    #mkdir -p $parallel_factored_corpus/$src_lang-$tgt_lang
#
#    python create_factors.py create_factor_file $parallel_corpus/$src_lang-$tgt_lang/test.$src_lang $parallel_factored_corpus/$src_lang-$tgt_lang/test.$src_lang $src_lang
#    python create_factors.py create_factor_file $parallel_corpus/$src_lang-$tgt_lang/test.$tgt_lang $parallel_factored_corpus/$src_lang-$tgt_lang/test.$tgt_lang $tgt_lang
#
#done     

## FACTORED 

## factored 
#template_fpath="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/factored/conf/template_f_lm.conf"
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/factored/conf"

##- create moses conf file
##for langpair in  `echo bn-hi hi-kn kn-hi`
#for langpair in  `echo hi-ta ta-hi ta-kn kn-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    propno=1    
#    for prop in `echo $PV_PROP`
#    do         
#        python create_factors.py create_single_factor_moses_conf $template_fpath $conf_dir/f_lm_${src_lang}-${tgt_lang}.$propno.$prop.conf $src_lang $tgt_lang $prop $propno 
#        propno=$((propno+1))
#    done         
#done     

### training - single factors 
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/factored/conf"
#
##for langpair in  `echo bn-hi hi-kn kn-hi`
#for langpair in  `echo hi-ta ta-hi ta-kn kn-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    propno=1    
#    for prop in `echo $PV_PROP`
#    do         
#        /usr/local/bin/smt/moses_job_scripts/moses_run.sh $conf_dir/f_lm_${src_lang}-${tgt_lang}.$propno.$prop.conf > log/f_lm_${src_lang}-${tgt_lang}.$propno.$prop.log 2>&1 
#        propno=$((propno+1))
#    done         
#done     

### create moses test ems file 
#template_fpath="/home/development/anoop/experiments/unsupervised_transliterator/src/transliterator/scripts/test_supervised.config"
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/factored/conf"

##for langpair in  `echo bn-hi hi-kn kn-hi`
#for langpair in  `echo hi-ta ta-hi ta-kn kn-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    propno=1    
#    for prop in `echo $PV_PROP`
#    do         
#        python create_factors.py create_test_moses_ems_conf $template_fpath $conf_dir/f_lm_${src_lang}-${tgt_lang}.$propno.$prop.testems.config $src_lang $tgt_lang $prop $propno 
#        propno=$((propno+1))
#    done         
#
#done     

###decoide with moses and evaluate
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/factored/conf"
#
#for langpair in  `echo hi-ta ta-hi ta-kn kn-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    propno=1    
#    for prop in `echo $PV_PROP`
#    do         
#        ./test_supervised.sh $conf_dir/f_lm_${src_lang}-${tgt_lang}.$propno.$prop.testems.config  > log/f_lm_${src_lang}-${tgt_lang}.$propno.$prop.log 2>&1
#        propno=$((propno+1))
#    done         
#
#done     

## PHRASE-BASED 
#
#template_fpath="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/pb/conf/template_pb.conf"
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/pb/conf"
#
##- create moses conf file
##for langpair in  `echo bn-hi hi-kn kn-hi`
#for langpair in  `echo hi-ta ta-hi ta-kn kn-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    python create_factors.py create_single_factor_moses_conf $template_fpath $conf_dir/pb_${src_lang}-${tgt_lang}.conf $src_lang $tgt_lang "prop" "propno" 
#done     

#### training - phrase based 
##conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/pb/conf"
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_brahminet/pb/conf"
#
#for langpair in  `echo bn-hi`
##for langpair in  `echo bn-hi hi-kn kn-hi`
##for langpair in  `echo hi-ta ta-hi ta-kn kn-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    /usr/local/bin/smt/moses_job_scripts/moses_run.sh $conf_dir/pb_${src_lang}-${tgt_lang}.conf > log/pb_${src_lang}-${tgt_lang}.log 2>&1 
#done     


### create moses test ems file 
#template_fpath="/home/development/anoop/experiments/unsupervised_transliterator/src/transliterator/scripts/test_supervised.config"
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/pb/conf"
#
##for langpair in  `echo bn-hi hi-kn kn-hi`
#for langpair in  `echo hi-ta ta-hi ta-kn kn-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    python create_factors.py create_test_moses_ems_conf $template_fpath $conf_dir/pb_${src_lang}-${tgt_lang}.testems.config $src_lang $tgt_lang  "prop" "propno" 
#
#done     

###decoide with moses and evaluate
##conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/pb/conf"
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_brahminet/pb/conf"
#
#for langpair in  `echo bn-hi`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'` 
#
#    ./test_supervised.sh $conf_dir/pb_${src_lang}-${tgt_lang}.testems.config  > log/pb_${src_lang}-${tgt_lang}.log 2>&1 & 
#
#done     


######### STAGE 2 ####################
#nohup ./supervised_stage_split.sh c_bn-hi_20_4_4.conf > split_20_4_4_bn-hi.log   2>&1 
#nohup ./supervised_stage_split.sh c_hi-kn_20_4_4.conf > split_20_4_4_hi-kn.log   2>&1 
#nohup ./supervised_stage_split.sh c_kn-hi_20_4_4.conf > split_20_4_4_kn-hi.log   2>&1 
#
#nohup ./supervised_stage_split.sh c_bn-hi_20_9_2.conf > split_20_9_2_bn-hi.log   2>&1 
#nohup ./supervised_stage_split.sh c_hi-kn_20_9_2.conf > split_20_9_2_hi-kn.log   2>&1
#nohup ./supervised_stage_split.sh c_kn-hi_20_9_2.conf > split_20_9_2_kn-hi.log   2>&1 
#
#nohup ./supervised_stage_split.sh c_bn-hi_20_9.conf   > split_20_9_bn-hi.log   2>&1 
#nohup ./supervised_stage_split.sh c_hi-kn_20_9.conf   > split_20_9_hi-kn.log   2>&1
#nohup ./supervised_stage_split.sh c_kn-hi_20_9.conf   > split_20_9_kn-hi.log   2>&1 

#nohup ./supervised_stage_concatenation.sh c_bn-hi_20_9.conf   > concat_20_9_bn-hi.log   2>&1 
#nohup ./supervised_stage_concatenation.sh c_hi-kn_20_9.conf   > concat_20_9_hi-kn.log   2>&1
#nohup ./supervised_stage_concatenation.sh c_kn-hi_20_9.conf   > concat_20_9_kn-hi.log   2>&1 


#nohup ./supervised_stage_concatenation.sh c_en-hi_20_3.conf     > log/concat_20_3_en-hi.log   2>&1 

nohup ./supervised_stage_concatenation.sh stage2_conf/c_ta-kn_20_3.conf   > log/concat_20_3_ta-kn.log   2>&1 
#nohup ./supervised_stage_concatenation.sh stage2_conf/c_ta-kn_20_4_4.conf   > log/concat_20_4_4_ta-kn.log   2>&1 
#nohup ./supervised_stage_concatenation.sh stage2_conf/c_ta-kn_20_9.conf     > log/concat_20_9_ta-kn.log   2>&1 
#nohup ./supervised_stage_concatenation.sh stage2_conf/c_ta-kn_20_9_2.conf     > log/concat_20_9_2_ta-kn.log   2>&1 


#nohup ./supervised_stage_concatenation.sh stage2_conf/c_hi-kn_20_4_4.conf   > log/concat_20_4_4_hi-kn.log   2>&1 
#nohup ./supervised_stage_concatenation.sh stage2_conf/c_kn-hi_20_4_4.conf   > log/concat_20_4_4_kn-hi.log   2>&1 
#nohup ./supervised_stage_concatenation.sh stage2_conf/c_bn-hi_20_4_4.conf   > log/concat_20_4_4_bn-hi.log   2>&1 

#nohup ./supervised_stage_concatenation_factored.sh factored_conf/c_bn-hi_20_4_4.conf > log/concat_20_4_4_bn-hi.log   2>&1 
#nohup ./supervised_stage_concatenation_factored.sh factored_conf/c_hi-kn_20_4_4.conf > log/concat_20_4_4_hi-kn.log   2>&1 
#nohup ./supervised_stage_concatenation_factored.sh factored_conf/c_kn-hi_20_4_4.conf > log/concat_20_4_4_kn-hi.log   2>&1 
#nohup ./supervised_stage_concatenation_factored.sh factored_conf/c_bn-hi_20_9.conf   > log/concat_20_9_bn-hi.log   2>&1 
#nohup ./supervised_stage_concatenation_factored.sh factored_conf/c_hi-kn_20_9.conf   > log/concat_20_9_hi-kn.log   2>&1
#nohup ./supervised_stage_concatenation_factored.sh factored_conf/c_kn-hi_20_9.conf   > log/concat_20_9_kn-hi.log   2>&1 



############# STAGE 1 ##############

# data langauge 
config_dir=/home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/config/newsindic2015_test
#config_dir=/home/development/anoop/experiments/unsupervised_transliterator/experiments/test_sets/indowordnet/config
#config_dir=/home/development/anoop/experiments/unsupervised_transliterator/experiments/test_sets/arjun/config

####### TRAINING DATA

#while read lpair
#do  
#    for expn in `echo 0_a`
#    do 
#        echo "======= Training $lpair for experiment $expn ====== "  
#        nohup ./ems.sh $config_dir/$lpair/${expn}_ems.config  > log/${lpair}_${expn}.log 2>&1 
#    done 
#done <  /home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/10lang_pairs.txt 

########  `echo 20_3 20_3_2 20_3_3 20_4_3 20_4_4  20_7 20_7_2 20_9 20_9_2`
########  `echo kn-hi hi-kn kn-ta ta-kn bn-hi ta-hi hi-ta`

#for lpair in `echo en-hi`
#do  
#    for expn in `echo 20_9_2 20_4_4 20_3`
#    do 
#        echo "======= Training $lpair for experiment $expn ====== "  
#        nohup ./ems.sh $config_dir/$lpair/${expn}_ems.config  > log/${lpair}_${expn}.log 2>&1 
#    done 
#done
#
#echo 'Stage 1 done'
#
######## RERANKING
#for lpair in `echo en-hi`
#do  
#    for expn in `echo 20_9  20_9_2 20_4_4 20_3`
#    do 
#        echo "======= Training $lpair for experiment $expn ====== "  
#        nohup ./rerank.sh $config_dir/$lpair/${expn}_ems.config  > log/${lpair}_${expn}.log 2>&1 
#    done 
#done
#
#echo 'Reranking done'

##########  DECODE-TRAINING
#stage_2_corpus="/home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/stage_2_corpus/"
#for lpair in `echo ta-kn`
#do  
#    for expn in `echo 20_3`
#    do 
#        echo "======= Training $lpair for experiment $expn ====== "  
#        nohup ./decode_training.sh   $config_dir/$lpair/${expn}_ems.config $stage_2_corpus/${lpair}/${expn}  > log/decode_training_${lpair}_${expn}.log 2>&1 
#    done 
#done
#
#echo 'Prepared stage 2 corpus'

###### run evaluation on ARJUN test set
#model_dir_base="/home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb"
#for lpair in `echo bn-hi kn-hi hi-kn` #kn-ta ta-kn ta-hi hi-ta`
#do  
#    for expn in `echo 20_3 20_3_2 20_3_3 20_4_3 20_4_4  20_7 20_7_2 20_9 20_9_2`
#    do 
#        echo "======= Training $lpair for experiment $expn ====== "  
#        nohup ./ems_decode_testset.sh $config_dir/$lpair/${expn}_ems.config $model_dir_base/$expn/$lpair/model/translit.model  > log/decode_testset_${lpair}_${expn}.log 2>&1 
#    done 
#done

######### rule based mapping
#for lpair in `echo kn-hi hi-kn kn-ta ta-kn bn-hi ta-hi hi-ta`
#do  
#    for expn in `echo 20_0`
#    do 
#        echo "======= Training $lpair for experiment $expn ====== "  
#        nohup ./ems_rule.sh $config_dir/$lpair/${expn}_ems.config  > log/${lpair}_${expn}.log 2>&1 
#    done 
#done

#### run evaluation on NEWS 2015 corpus for models trained on Arjun IITB 
#config_dir=/home/development/anoop/experiments/unsupervised_transliterator/experiments/news_2015_indic/config
#
#for lpair in `echo hi-ta ta-hi bn-hi`
#do  
#    for expn in `echo 0_a`
#    do 
#        echo "======= Training $lpair for experiment $expn ====== "  
#        nohup ./ems.sh $config_dir/$lpair/${expn}_ems.config  > log/${lpair}_${expn}.log 2>&1 
#    done 
#done



####### datasize study #######
#config_dir=/home/development/anoop/experiments/unsupervised_transliterator/experiments/datasize_study/config
#
#for ds in `echo 5k 15k 30k`
#do  
#        echo "======= Training bn-hi for experiment 7_b for datasize $ds ====== "  
#        nohup ./ems.sh $config_dir/bn-hi/$ds/7_b_ems.config  > log/ds-$ds.log 2>&1 
#done 
#

####  Prepare data for en-hi
#data_dir=/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/
#src_lang=en
#tgt_lang=hi
#split='train'
#
#python en-hi_utilities.py filter_for_g2p  \
#    $data_dir/$src_lang-$tgt_lang/$split.$src_lang.word.unfiltered \
#    $data_dir/$src_lang-$tgt_lang/$split.$src_lang.word
#
#python en-hi_utilities.py convert_to_phonetisaurus_input_format \
#    $data_dir/$src_lang-$tgt_lang/$split.$src_lang.word \
#    $data_dir/$src_lang-$tgt_lang/$split.$src_lang.phonetisaurus.word 
#
#phonetisaurus-g2p --model=$G2PMODEL \
#    --input=$data_dir/$src_lang-$tgt_lang/$split.$src_lang.phonetisaurus.word \
#    --isfile=true > \
#    $data_dir/$src_lang-$tgt_lang/$split.$src_lang.phonetisaurus.phoneme  2> err
#
#python en-hi_utilities.py convert_phonetisaurus_to_xlit_format \
# $data_dir/$src_lang-$tgt_lang/$split.$src_lang.phonetisaurus.phoneme \
# $data_dir/$src_lang-$tgt_lang/$split.$src_lang.phoneme \
# $data_dir/$src_lang-$tgt_lang/$split.$src_lang
#
