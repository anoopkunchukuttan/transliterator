#### Some script to extract common msr corpus

datadir=/home/development/anoop/experiments/news_2015/news_2015/data/pb

c0=Ta

c0=Ba
c1=Hi
mkdir common/$c0-$c1
python news2015_utilities.py extract_common_msr_corpus $datadir/En-$c0/ $datadir/En-$c1/ $c0 $c1 common/$c0-$c1

#python ~/installs/indic_nlp_library/src/indicnlp/transliterate/unicode_transliterate.py common/$c0-$c1/train.$c0 common/$c0-$c1/train.$c0.xlit bn hi
#cp common/$c0-$c1/train.$c1 common/$c0-$c1/train.$c1.xlit


c0=Ka
c1=Hi
mkdir common/$c0-$c1
python news2015_utilities.py extract_common_msr_corpus $datadir/En-$c0/ $datadir/En-$c1/ $c0 $c1 common/$c0-$c1
#python ~/installs/indic_nlp_library/src/indicnlp/transliterate/unicode_transliterate.py common/$c0-$c1/train.$c0 common/$c0-$c1/train.$c0.xlit kn hi
#cp common/$c0-$c1/train.$c1 common/$c0-$c1/train.$c1.xlit

c0=Ta
c1=Ba
mkdir common/$c0-$c1
python news2015_utilities.py extract_common_msr_corpus $datadir/En-$c0/ $datadir/En-$c1/ $c0 $c1 common/$c0-$c1
#python ~/installs/indic_nlp_library/src/indicnlp/transliterate/unicode_transliterate.py common/$c0-$c1/train.$c0 common/$c0-$c1/train.$c0.xlit ta hi
#python ~/installs/indic_nlp_library/src/indicnlp/transliterate/unicode_transliterate.py common/$c0-$c1/train.$c1 common/$c0-$c1/train.$c1.xlit bn hi

c0=Ta
c1=Ka
mkdir common/$c0-$c1
python news2015_utilities.py extract_common_msr_corpus $datadir/En-$c0/ $datadir/En-$c1/ $c0 $c1 common/$c0-$c1
#python ~/installs/indic_nlp_library/src/indicnlp/transliterate/unicode_transliterate.py common/$c0-$c1/train.$c0 common/$c0-$c1/train.$c0.xlit ta hi
#python ~/installs/indic_nlp_library/src/indicnlp/transliterate/unicode_transliterate.py common/$c0-$c1/train.$c1 common/$c0-$c1/train.$c1.xlit kn hi

c0=Ta
c1=Hi
mkdir common/$c0-$c1
python news2015_utilities.py extract_common_msr_corpus $datadir/En-$c0/ $datadir/En-$c1/ $c0 $c1 common/$c0-$c1
#python ~/installs/indic_nlp_library/src/indicnlp/transliterate/unicode_transliterate.py common/$c0-$c1/train.$c0 common/$c0-$c1/train.$c0.xlit ta hi
#cp common/$c0-$c1/train.$c1 common/$c0-$c1/train.$c1.xlit

c0=Ba
c1=Ka
mkdir common/$c0-$c1
python news2015_utilities.py extract_common_msr_corpus $datadir/En-$c0/ $datadir/En-$c1/ $c0 $c1 common/$c0-$c1
#python ~/installs/indic_nlp_library/src/indicnlp/transliterate/unicode_transliterate.py common/$c0-$c1/train.$c0 common/$c0-$c1/train.$c0.xlit bn hi
#python ~/installs/indic_nlp_library/src/indicnlp/transliterate/unicode_transliterate.py common/$c0-$c1/train.$c1 common/$c0-$c1/train.$c1.xlit kn hi

#mv common/Ta-Hi/train.Ta common/Ta-Hi/test.ta
#mv common/Ta-Hi/train.Hi common/Ta-Hi/test.hi
#mv common/Ta-Hi common/ta-hi
#
#mv common/Ba-Hi/train.Ba common/Ba-Hi/test.bn
#mv common/Ba-Hi/train.Hi common/Ba-Hi/test.hi
#mv common/Ba-Hi common/bn-hi
