import sys, os, codecs
import yaml
from cfilt.transliteration import phonetic_sim 
from indicnlp import loader

loader.load()

### GLOBALS
moses_decoder_dir="/usr/local/bin/smt/mosesdecoder-3.0"
SCRIPTS_ROOTDIR="{}/scripts".format(moses_decoder_dir)

## script starts here 
#'/home/development/anoop/experiments/unsupervised_transliterator/src/transliterator/src/cfilt/transliteration/test.yaml'
f=open(sys.argv[1])
p=yaml.load(f)
f.close()

phrase_table_fname=p['phrase_table_fname']
parallel_corpus=p['parallel_corpus']
template_fname=p['template_fname']
src_lang=p['src_lang']
tgt_lang=p['tgt_lang']

sim_metrics=p['sim_metrics']
numfeatures=p['numfeatures']
basedir=p['basedir']


#### Paramters - for experiment using only additional features 
#phrase_table_fname="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/results/pb/bn-hi/moses_data/model/phrase-table"
#parallel_corpus="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/data/pb/bn-hi"
#template_fname="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/conf/sim_equal_moses_template.ini"
#src_lang="bn"
#tgt_lang="hi"
#sim_metrics=[equal]
#
#numfeatures=5
#basedir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/results/sim_equal"


##### Decide best values for hyper-parameters related to phonetic alignment

### values for mismatch penalty
#mismatch_penalties=[-0.05,-0.1,-0.15,-0.2,-0.3]
mismatch_penalties=[0.0]

### values of gap penalty
#gap_penalties=[-0.05,-0.1,-0.15]
gap_penalties=[0.0]

os.mkdir(basedir)
os.system("cp {} {}/".format(sys.argv[1],basedir))

for mismatch in mismatch_penalties:
    for gap in gap_penalties: 
        
        outdir='{}/m-{}_g-{}'.format(basedir,mismatch,gap)
        moses_ini_fname='{}/moses.ini'.format(outdir)

        os.mkdir(outdir)

        ########### Generating new phrase table 
        ### adding feature to phrase table 
        phonetic_sim.add_phonetic_alignment_phrase_table(phrase_table_fname,outdir+'/phrase-table',
                src_lang,tgt_lang,sim_metrics,mismatch,gap) 

        ### create moses.ini file from template
        template=None
        with codecs.open(template_fname,'r','utf-8') as template_file: 
            template=template_file.read()
        moses_ini_contents=template.format(numfeatures=numfeatures,phrasetable=outdir+'/phrase-table',initfeatvalues=u' '.join(['0.2']*numfeatures))

        with codecs.open(moses_ini_fname,'w','utf-8') as moses_ini_file: 
            moses_ini_file.write(moses_ini_contents)

        ######### tuning 
        tuning_cmd="{0}/training/mert-moses.pl \
            {1}/tun.{2} \
            {1}/tun.{3} \
            /usr/local/bin/smt/mosesdecoder-3.0/bin/moses  \
            {4} \
            --working-dir {5}/tuning \
            --rootdir {0} \
            --decoder-flags='-threads 20 -distortion-limit 0' > {5}/tuning.out 2> {5}/tuning.err".format(SCRIPTS_ROOTDIR,parallel_corpus,src_lang,tgt_lang,moses_ini_fname,outdir)

        os.system(tuning_cmd)

        ## decoding the tuning set 
        decode_cmd="{0}/bin/moses \
                -f {4}/tuning/moses.ini \
                -n-best-list {4}/decode.nbest.{3} 10 distinct < {1}/tun.{2} \
                > {4}/decode.{3} 2> {4}/decode.err".format(moses_decoder_dir,parallel_corpus,src_lang,tgt_lang,outdir)

        os.system(decode_cmd)


        ## run evaluation 
        eval_cmd1="python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
                {0}/tun.id \
                {0}/tun.xml \
                {1}/decode.nbest.{3} \
                {1}/decode.nbest.{3}.xml \
                tuning dataset {2} {3}".format(parallel_corpus,outdir,src_lang,tgt_lang)  

        eval_cmd2="python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
                        -t {0}/tun.xml \
                        -i {1}/decode.nbest.{3}.xml \
                        -o {1}/decode.nbest.{3}.detaileval.csv \
                        >  {1}/decode.nbest.{3}.eval".format(parallel_corpus,outdir,src_lang,tgt_lang) 

        os.system(eval_cmd1)
        os.system(eval_cmd2)


        ######## testing
        os.mkdir('{}/evaluation'.format(outdir))
        ## decoding the test set 
        test_cmd="{0}/bin/moses \
                -f {4}/tuning/moses.ini \
                -n-best-list {4}/evaluation/test.nbest.{3} 10 distinct < {1}/test.{2} \
                > {4}/evaluation/test.{3} 2> {4}/test.err".format(moses_decoder_dir,parallel_corpus,src_lang,tgt_lang,outdir)

        os.system(test_cmd)

        ## run evaluation 
        teval_cmd1="python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
                {0}/test.id \
                {0}/test.xml \
                {1}/evaluation/test.nbest.{3} \
                {1}/evaluation/test.nbest.{3}.xml \
                test dataset {2} {3}".format(parallel_corpus,outdir,src_lang,tgt_lang)  

        teval_cmd2="python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
                        -t {0}/test.xml \
                        -i {1}/evaluation/test.nbest.{3}.xml \
                        -o {1}/evaluation/test.nbest.{3}.detaileval.csv \
                        >  {1}/evaluation/test.nbest.{3}.eval".format(parallel_corpus,outdir,src_lang,tgt_lang) 

        os.system(teval_cmd1)
        os.system(teval_cmd2)



##### Paramters - for experiment using factors + additional features 
#phrase_table_fname="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/results/f_lm_all_c2/bn-hi/moses_data/model/phrase-table.0-0"
#parallel_corpus="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/data/factored/bn-hi"
#template_fname="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/conf/f_lm_all_c2+sim_cos_moses_template.ini"
#src_lang="bn"
#tgt_lang="hi"
#
#numfeatures=5
#basedir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/results/f_lm_all_c2+sim_cos"

##### Paramters - for experiment using factors + additional features 
#phrase_table_fname="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/results/f_lm2/bn-hi/moses_data/model/phrase-table.0-0"
#parallel_corpus="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/data/factored/bn-hi"
#template_fname="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/conf/f_lm2+sim_cos_moses_template.ini"
#src_lang="bn"
#tgt_lang="hi"
#
#numfeatures=5
#basedir="/home/development/anoop/experiments/unsupervised_transliterator/experiments/brahminet/factored/results/f_lm2+sim_cos"
