
########## corpus creation for bridge work 
#for langpair in `echo Ba-Hi  Ba-Ka  Ka-Hi  Ta-Ba  Ta-Hi  Ta-Ka`
#do 
#
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#    python news2015_utilities.py extract_exclusive_msr_corpus \
#        /home/development/anoop/experiments/news_2015/news_2015/data/pb/En-$src_lang/ \
#        /home/development/anoop/experiments/news_2015/news_2015/data/pb/En-$tgt_lang/ \
#        $src_lang $tgt_lang \
#        /home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel_old/tmp/create_test_corpus/NEWS_2015_exclusive/$src_lang-$tgt_lang
#done 

##########   create the corpus in required format 
#w=/home/development/anoop/experiments/unsupervised_transliterator/data/bridge/pb/w
#
#
#for langpair in `echo Ba-Hi  Ka-Hi `
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#    outdir=$w/$src_lang-$tgt_lang
#
#    mkdir -p $outdir/En-$src_lang
#    python test.py    /home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel_old/tmp/create_test_corpus/NEWS_2015_exclusive/$src_lang-$tgt_lang/train.En-$src_lang $outdir/En-$src_lang/train.En $outdir/En-$src_lang/train.$src_lang
#
#    mkdir -p $outdir/En-$tgt_lang
#    python test.py    /home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel_old/tmp/create_test_corpus/NEWS_2015_exclusive/$src_lang-$tgt_lang/train.En-$tgt_lang $outdir/En-$tgt_lang/train.En $outdir/En-$tgt_lang/train.$tgt_lang
#done


######### create moses configuration file 
#template_fpath="/home/development/anoop/experiments/unsupervised_transliterator/experiments/bridge/pb/conf/template_pb.conf"
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/bridge/pb/conf"
#
#for langpair in `echo bn-hi hi-kn kn-hi`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#
#    python create_factors.py create_single_factor_moses_conf $template_fpath $conf_dir/pb_bridge-${src_lang}-${tgt_lang}--${src_lang}-en.conf \
#        $src_lang-$tgt_lang $src_lang en "prop" "propno" 
#
#    python create_factors.py create_single_factor_moses_conf $template_fpath $conf_dir/pb_bridge-${src_lang}-${tgt_lang}--en-${tgt_lang}.conf \
#        $src_lang-$tgt_lang en $tgt_lang "prop" "propno" 
#done     



##### run moses training 
#conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/bridge/pb/conf"
#for langpair in `echo bn-hi hi-kn kn-hi`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'` 
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#
#    /usr/local/bin/smt/moses_job_scripts/moses_run.sh $conf_dir/pb_bridge-${src_lang}-${tgt_lang}--${src_lang}-en.conf > log/pb_bridge-${src_lang}-${tgt_lang}--${src_lang}-en.log 2>&1 
#
#    /usr/local/bin/smt/moses_job_scripts/moses_run.sh $conf_dir/pb_bridge-${src_lang}-${tgt_lang}--en-${tgt_lang}.conf > log/pb_bridge-${src_lang}-${tgt_lang}--en-${tgt_lang}.log 2>&1 
#
#done     


### create bridge ems file 
conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/bridge/pb/conf"
template_fpath="/home/development/anoop/experiments/unsupervised_transliterator/src/transliterator/scripts/test_bridge.config"
for langpair in `echo kn-hi hi-kn` 
do 
    src_lang=`echo $langpair | cut -f 1 -d '-'` 
    tgt_lang=`echo $langpair | cut -f 2 -d '-'`

    python create_factors.py create_test_moses_ems_conf $template_fpath $conf_dir/bridge_decode_${src_lang}-${tgt_lang}.testems.config $src_lang $tgt_lang  'en'  "prop" "propno" 

done     

### bridge decode 
conf_dir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/bridge/pb/conf"

for langpair in `echo kn-hi hi-kn` 
do 
    src_lang=`echo $langpair | cut -f 1 -d '-'` 
    tgt_lang=`echo $langpair | cut -f 2 -d '-'`

    ./bridge_decode.sh $conf_dir/bridge_decode_${src_lang}-${tgt_lang}.testems.config > log/bridge_decode_${src_lang}-${tgt_lang}.log 2>&1

done     
