from cfilt.transliteration.decoder import *
from cfilt.transliteration.parallel_decoder import *
from cfilt.transliteration.utilities import *

import os, scipy
import numpy as np 
import sys
import itertools as it

def read_lines(corpus_fname): 
    with codecs.open(corpus_fname,'r','utf-8') as infile:
            return infile.readlines()

def compute_av_entropy(model_params):

    logz_func=np.vectorize(log_z,otypes=[np.float])

    return np.average( np.sum( -1.0*model_params*logz_func(model_params),
                axis=1)
        )

def get_iterations(log_dir): 
    return sorted(map(int,filter(lambda x:os.path.isdir(log_dir+'/'+x),os.listdir(log_dir))))

def av_entropy_generator(log_dir):
    for itern in get_iterations(log_dir): 
        model=TransliterationModel.load_translit_model( '{}/{}/{}'.format(log_dir,itern,'translit.model') )
        yield compute_av_entropy(model.param_values)

def model_size(log_dir):
    for itern in get_iterations(log_dir): 
        model=TransliterationModel.load_translit_model( '{}/{}/{}'.format(log_dir,itern,'translit.model') )
        print model.param_values.shape

def words_never_changed(log_dir): 
    change_status=[]
    for word_list in it.izip(*[read_lines('{}/{}/{}'.format(log_dir,itern,'transliterations.txt'))   for itern in get_iterations(log_dir)[1:]]): 
        changed=False
        for i in xrange(len(word_list)-1):
            if word_list[i]!=word_list[i+1]:
                changed=True
                break

        change_status.append(changed)

    print 'No of words changed at least once: {}'.format(len(filter(lambda x: x==True,change_status)))

def likelihood(log_dir): 

    lm_fname='/home/development/anoop/experiments/unsupervised_transliterator/data/lm/nonparallel/pb/2g/hi.lm'
    fcorpus_fname='/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/bn-hi/train.bn'

    for itern in get_iterations(log_dir)[1:]: 

        likelihood=parallel_likelihood(TransliterationModel.load_translit_model( '{}/{}/{}'.format(log_dir,itern,'translit.model') ), 
                                    load_lm_model(lm_fname),
                                    read_parallel_corpus(fcorpus_fname,'{}/{}/{}'.format(log_dir,itern,'transliterations.txt'))
                                  )

        print likelihood    

def equal_sentences(): 
    for i in xrange(1,10): 
        bitvec=[len(f)==len(e) for f, e in it.izip(read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/bn-hi/train.bn'),read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb/7_b_again/bn-hi/log/{}/transliterations.txt'.format(i) ))]
        print sum(bitvec) 

def debug_training(log_dir): 
   #print list(av_entropy_generator(log_dir))
   #equal_sentences()
   #words_never_changed(log_dir)
   #model_size(log_dir)
   likelihood(log_dir)

if __name__=='__main__': 
   debug_training(sys.argv[1]) 
   #equal_sentences()
