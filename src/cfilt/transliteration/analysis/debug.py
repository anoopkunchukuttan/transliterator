from cfilt.transliteration.decoder import *
from cfilt.transliteration.parallel_decoder import *
from cfilt.transliteration.utilities import *
from indicnlp.transliterate import unicode_transliterate

import os, scipy
import numpy as np 
import sys
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def read_lines(corpus_fname): 
    with codecs.open(corpus_fname,'r','utf-8') as infile:
            return infile.readlines()

def read_transliterations(log_dir,itern): 
    """
    Read decoded transliterations from iteration
    """

    ecorpus_fname='{}/{}/{}'.format(log_dir,itern,'transliterations.txt')
    triplets_fname='{}/{}/{}'.format(log_dir,itern,'word_triplets.pickle')

    transliterations=None

    if os.path.exists(ecorpus_fname):
        transliterations=read_monolingual_corpus(ecorpus_fname)
    else: 
        with open(triplets_fname,'r') as infile: 
            triplets=pickle.load(infile)
            transliterations=[ tgt[0][0] for src,tgt,prev in triplets]

    return transliterations 

def compute_av_entropy(model_params):

    logz_func=np.vectorize(log_z,otypes=[np.float])

    return np.average( np.sum( -1.0*model_params*logz_func(model_params),
                axis=1)
        )

def get_iterations(log_dir): 
    """
    Get the list of iteration numbers 
    """
    return sorted(map(int,filter(lambda x:os.path.isdir(log_dir+'/'+x),os.listdir(log_dir))))

def av_entropy_generator(log_dir):
    """
    Get list of average transliteration entropy for iterations sorted by iteration number starting at iteration 0
    """
    for itern in get_iterations(log_dir): 
        model=TransliterationModel.load_translit_model( '{}/{}/{}'.format(log_dir,itern,'translit.model') )
        yield compute_av_entropy(model.param_values)

def model_size_generator(log_dir):
    """
    Get list of model sizes (dimensions)
    """

    for itern in get_iterations(log_dir): 
        model=TransliterationModel.load_translit_model( '{}/{}/{}'.format(log_dir,itern,'translit.model') )
        print model.param_values.shape

def words_changed_atleast_once(log_dir): 
    """
    Number of words whose transliterations have changed at least once during iterations
    """
    change_status=[]
    #for word_list in it.izip(*[read_lines('{}/{}/{}'.format(log_dir,itern,'transliterations.txt'))   for itern in get_iterations(log_dir)[1:]]): 
    for wno, word_list in enumerate(it.izip(*[read_transliterations(log_dir,itern)   for itern in get_iterations(log_dir)[1:]])): 
        changed=False
        for i in xrange(len(word_list)-1):
            if word_list[i]!=word_list[i+1]:
                changed=True
                print '{} {}'.format(wno,i+1)
                break

        change_status.append(changed)
    
    changed_count=len(filter(lambda x: x==True,change_status))
    print 'No of words changed at least once: {}'.format(changed_count)

def likelihood_generator(log_dir,
               lm_fname='/home/development/anoop/experiments/unsupervised_transliterator/data/lm/nonparallel/pb/2g/hi.lm',
               fcorpus_fname='/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/bn-hi/train.bn'): 

    for itern in get_iterations(log_dir)[1:]: 
        #likelihood=parallel_likelihood_unsupervised(TransliterationModel.load_translit_model( '{}/{}/{}'.format(log_dir,itern,'translit.model') ), 
        #                            load_lm_model(lm_fname),
        #                            it.izip(read_monolingual_corpus(fcorpus_fname),read_transliterations(log_dir,itern))
        #                          )

        wpairs_aligns=None
        wpairs_weights=None
        wpairs_eword_weights=None 

        with open('{}/{}/wpairs_aligns.pickle'.format(log_dir,itern) , 'r' ) as ifile:
            wpairs_aligns=pickle.load(ifile)

        with open('{}/{}/wpairs_weights.pickle'.format(log_dir,itern) , 'r' ) as ifile:
            wpairs_weights=pickle.load(ifile)

        with open('{}/{}/wpairs_eword_weights.pickle'.format(log_dir,itern) , 'r' ) as ifile:
            wpairs_eword_weights=pickle.load(ifile)

        decoder=TransliterationDecoder(TransliterationModel.load_translit_model( '{}/{}/{}'.format(log_dir,itern,'translit.model') ), 
                                    load_lm_model(lm_fname))
        likelihood=decoder.compute_log_likelihood_unsupervised(
                                    it.izip(read_monolingual_corpus(fcorpus_fname),read_transliterations(log_dir,itern)),
                                    wpairs_aligns,wpairs_weights,wpairs_eword_weights
                                  )
        yield likelihood    

#def equal_sentences(): 
#    #for i in xrange(1,10): 
#    #    bitvec=[len(f)==len(e) for f, e in it.izip(read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/bn-hi/train.bn'),read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb/7_b_again/bn-hi/log/{}/transliterations.txt'.format(i) ))]
#    #    print sum(bitvec) 
#
#    bitvec=[len(f)>=len(e) for f, e in it.izip(read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/data/parallel/pb/bn-hi/train.bn'),read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/data/parallel/pb/bn-hi/train.hi' ))]
#    print sum(bitvec) 


def plot_confusion_matrix(confusion_mat_fname,tgt='hi'): 

    matplotlibcfont.rcParams['font.family']='Lohit Devanagari'

    confusion_df=pd.read_pickle(confusion_mat_fname)
    
    schar=list(confusion_df.index)
    tchar=list(confusion_df.columns)
    i=0
    for c in schar: 
        if c in tchar: 
            confusion_df.ix[c,c]=0.0
    
    data=confusion_df.as_matrix()
    
    # # normalize along row
    # sums=np.sum(data,axis=1)
    # data=data.T/sums
    # data=data.T
    
    # # normalize along column
    # sums=np.sum(data,axis=0)
    # data=data/sums
    
    s=np.sum(data)
    data=data/s
    
    columns=list(confusion_df.columns)
    col_names=[ x if tgt=='hi' else unicode_transliterate.UnicodeIndicTransliterator.transliterate(x,lcode_map[tgt],'hi') for x in columns]

    rows=list(confusion_df.index)
    row_names=[ x if tgt=='hi' else unicode_transliterate.UnicodeIndicTransliterator.transliterate(x,lcode_map[tgt],'hi') for x in rows]
    
    #plt.pcolor(data,cmap=plt.cm.gray_r,edgecolors='k')
    plt.pcolor(data,cmap=plt.cm.hot_r,edgecolors='k')
    
    #plt.pcolor(data,edgecolors='k')
    plt.colorbar()
    plt.xticks(np.arange(0,len(col_names))+0.5,col_names)
    plt.yticks(np.arange(0,len(row_names))+0.5,row_names)
    
    plt.show()
    plt.close()

def debug_training(log_dir): 
   #print list(av_entropy_generator(log_dir))
   #equal_sentences()
   #words_never_changed(log_dir)
   #model_size(log_dir)
   #likelihood(log_dir)
   pass 

if __name__=='__main__': 
    plot_confusion_matrix(sys.argv[1])
    pass 
