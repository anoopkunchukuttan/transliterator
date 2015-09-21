from cfilt.transliteration.decoder import *
from cfilt.transliteration.parallel_decoder import *
from cfilt.transliteration.utilities import *
from indicnlp.transliterate import unicode_transliterate

from indicnlp import loader
from indicnlp import common
from indicnlp.script import indic_scripts

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

#def read_synthesized_parallel_corpus(parallel_dir,src_l,tgt_l,n_xlit=10,n_tun=1000):
#    """
#    Read the final synthesizede parallel corpus from the final iteration 
#    nxlit: number of transliterations to extract 
#    """
#
#    last_iter_no=get_iterations(log_dir)[-1]
#
#    triplets_fname='{}/{}/{}'.format(log_dir,last_iter_no,'word_triplets.pickle')
#    triplets=None
#    with open(triplets_fname,'r') as infile: 
#        triplets=pickle.load(infile)
#
#    if not os.path.isdir(parallel_dir): 
#        os.mkdir(parallel_dir)
#
#    ### training data 
#    train_src_file=codecs.open('{}/train.{}'.format(parallel_dir,src_l),'w','utf-8')
#    train_tgt_file=codecs.open('{}/train.{}'.format(parallel_dir,tgt_l),'w','utf-8')
#    train_score_file=codecs.open('{}/train.score'.format(parallel_dir),'w','utf-8')
#
#    for src,tgt,prev in triplets[:-n_tun]: 
#        for xlit, score in tgt[:min(len(tgt),n_xlit)]:
#            train_src_file.write(u' '.join(src)+u'\n')
#            train_tgt_file.write(u' '.join(xlit)+u'\n')
#            train_score_file.write(str(score)+u'\n')
#
#    train_src_file.close()
#    train_tgt_file.close()
#    train_score_file.close()
#
#    ### tuning data 
#    tun_src_file=codecs.open('{}/tun.{}'.format(parallel_dir,src_l),'w','utf-8')
#    tun_tgt_file=codecs.open('{}/tun.{}'.format(parallel_dir,tgt_l),'w','utf-8')
#    tun_score_file=codecs.open('{}/tun.score'.format(parallel_dir),'w','utf-8')
#
#    for src,tgt,prev in triplets[-n_tun:]: 
#        for xlit, score in tgt[:min(len(tgt),n_xlit)]:
#            tun_src_file.write(u' '.join(src)+u'\n')
#            tun_tgt_file.write(u' '.join(xlit)+u'\n')
#            tun_score_file.write(str(score)+u'\n')
#
#    tun_src_file.close()
#    tun_tgt_file.close()
#    tun_score_file.close()

def read_synthesized_parallel_corpus_from_moses(src_fname,parallel_dir,src_l,tgt_l,n_xlit=10,n_tun=1000):
    """
    Read the final synthesizede parallel corpus obtained from decoding with final paramters, which is in moses format  
    nxlit: number of transliterations to extract 
    """

    ### Methods for parsing n-best lists
    def parse_line(line):
        """
            line in n-best file 
            return list of fields
        """
        fields=[ x.strip() for x in  line.strip().split('|||') ]
        fields[0]=int(fields[0])
        fields[3]=float(fields[3])
        return fields
    
    def iterate_nbest_list(nbest_fname): 
        """
            nbest_fname: moses format nbest file name 
            return iterator over tuple of sent_no, list of n-best candidates
    
        """
    
        infile=codecs.open(nbest_fname,'r','utf-8')
        
        for sent_no, lines in it.groupby(iter(infile),key=lambda x:parse_line(x)[0]):
            parsed_lines = [ parse_line(line) for line in lines ]
            yield((sent_no,parsed_lines))
    
        infile.close()

    all_src_lines=None
    with codecs.open(src_fname,'r','utf-8') as src_file: 
        src_lines=[x.strip() for x in src_file.readlines()]

    data=list(it.izip(src_lines,iterate_nbest_list('{}/all.topn.reranked.{}'.format(parallel_dir,tgt_l))))

    ### training data 
    train_src_file=codecs.open('{}/train.{}'.format(parallel_dir,src_l),'w','utf-8')
    train_tgt_file=codecs.open('{}/train.{}'.format(parallel_dir,tgt_l),'w','utf-8')
    train_score_file=codecs.open('{}/train.score'.format(parallel_dir),'w','utf-8')

    for src,(i,tgt) in data[:-n_tun]: 
        total=sum( [ score for j, xlit, _, score in tgt[:min(len(tgt),n_xlit)]] )
        for j, xlit, _, score in tgt[:min(len(tgt),n_xlit)]:
            train_src_file.write(u' '.join(src)+u'\n')
            train_tgt_file.write(u' '.join(xlit)+u'\n')
            train_score_file.write(str(score/total)+u'\n')

    train_src_file.close()
    train_tgt_file.close()
    train_score_file.close()

    ### tuning data 
    tun_src_file=codecs.open('{}/tun.{}'.format(parallel_dir,src_l),'w','utf-8')
    tun_tgt_file=codecs.open('{}/tun.{}'.format(parallel_dir,tgt_l),'w','utf-8')
    tun_score_file=codecs.open('{}/tun.score'.format(parallel_dir),'w','utf-8')

    for src,(i,tgt) in data[-n_tun:]: 
        total=sum( [ score for j, xlit, _, score in tgt[:min(len(tgt),n_xlit)]] )
        for j, xlit, _, score in tgt[:min(len(tgt),n_xlit)]:
            tun_src_file.write(u' '.join(src)+u'\n')
            tun_tgt_file.write(u' '.join(xlit)+u'\n')
            tun_score_file.write(str(score/total)+u'\n')

    tun_src_file.close()
    tun_tgt_file.close()
    tun_score_file.close()

def read_synthesized_parallel_corpus_from_moses_2(src_fname,parallel_dir,src_l,tgt_l,n_xlit=10,n_tun=1000):
    """
    Read the final synthesizede parallel corpus obtained from decoding with final paramters, which is in moses format  
    nxlit: number of transliterations to extract 
    """

    ### Methods for parsing n-best lists
    def parse_line(line):
        """
            line in n-best file 
            return list of fields
        """
        fields=[ x.strip() for x in  line.strip().split('|||') ]
        fields[0]=int(fields[0])
        fields[3]=float(fields[3])
        return fields
    
    def iterate_nbest_list(nbest_fname): 
        """
            nbest_fname: moses format nbest file name 
            return iterator over tuple of sent_no, list of n-best candidates
    
        """
    
        infile=codecs.open(nbest_fname,'r','utf-8')
        
        for sent_no, lines in it.groupby(iter(infile),key=lambda x:parse_line(x)[0]):
            parsed_lines = [ parse_line(line) for line in lines ]
            yield((sent_no,parsed_lines))
    
        infile.close()

    ## get the output from the transliteration system's training set decoding 
    all_src_lines=None
    with codecs.open(src_fname,'r','utf-8') as src_file: 
        src_lines=[x.strip() for x in src_file.readlines()]

    data=list(it.izip(src_lines,iterate_nbest_list('{}/all.topn.reranked.{}'.format(parallel_dir,tgt_l))))

    ## make the parallel corpus directories 
    for i in xrange(n_xlit):
        os.mkdir('{}/{}'.format(parallel_dir,i))

    ### training data 
    train_src_files=[ codecs.open('{}/{}/train.{}'.format(parallel_dir,i,src_l),'w','utf-8') for i in xrange(n_xlit) ]
    train_tgt_files=[ codecs.open('{}/{}/train.{}'.format(parallel_dir,i,tgt_l),'w','utf-8') for i in xrange(n_xlit) ]
    train_score_files=[ codecs.open('{}/{}/train.score'.format(parallel_dir,i),'w','utf-8') for i in xrange(n_xlit) ]

    for src,(sno,tgt) in data[:-n_tun]: 
        for i,(j, xlit, _, score) in enumerate(tgt[:n_xlit]):
            train_src_files[i].write(u' '.join(src)+u'\n')
            train_tgt_files[i].write(u' '.join(xlit)+u'\n')
            train_score_files[i].write(str(score)+u'\n')

    for i in xrange(n_xlit):
        train_src_files[i].close()
        train_tgt_files[i].close()
        train_score_files[i].close()

    ### tuning data 
    tun_src_files=[ codecs.open('{}/{}/tun.{}'.format(parallel_dir,i,src_l),'w','utf-8')  for i in xrange(n_xlit) ]
    tun_tgt_files=[ codecs.open('{}/{}/tun.{}'.format(parallel_dir,i,tgt_l),'w','utf-8')  for i in xrange(n_xlit) ]

    ### take the best output for tuning 
    for src,(sno,tgt) in data[-n_tun:]: 
        j, xlit, _, score=tgt[0]
        for i in xrange(n_xlit):
            tun_src_files[i].write(u' '.join(src)+u'\n')
            tun_tgt_files[i].write(u' '.join(xlit)+u'\n')

    for i in xrange(n_xlit):
        tun_src_files[i].close()
        tun_tgt_files[i].close()

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

#def likelihood_generator(log_dir,
#               lm_fname='/home/development/anoop/experiments/unsupervised_transliterator/data/lm/nonparallel/pb/2g/hi.lm',
#               fcorpus_fname='/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/bn-hi/train.bn'): 
#
#    for itern in get_iterations(log_dir)[1:]: 
#        #likelihood=parallel_likelihood_unsupervised(TransliterationModel.load_translit_model( '{}/{}/{}'.format(log_dir,itern,'translit.model') ), 
#        #                            load_lm_model(lm_fname),
#        #                            it.izip(read_monolingual_corpus(fcorpus_fname),read_transliterations(log_dir,itern))
#        #                          )
#
#        wpairs_aligns=None
#        wpairs_weights=None
#        wpairs_eword_weights=None 
#
#        with open('{}/{}/wpairs_aligns.pickle'.format(log_dir,itern) , 'r' ) as ifile:
#            wpairs_aligns=pickle.load(ifile)
#
#        with open('{}/{}/wpairs_weights.pickle'.format(log_dir,itern) , 'r' ) as ifile:
#            wpairs_weights=pickle.load(ifile)
#
#        with open('{}/{}/wpairs_eword_weights.pickle'.format(log_dir,itern) , 'r' ) as ifile:
#            wpairs_eword_weights=pickle.load(ifile)
#
#        decoder=TransliterationDecoder(TransliterationModel.load_translit_model( '{}/{}/{}'.format(log_dir,itern,'translit.model') ), 
#                                    load_lm_model(lm_fname))
#        likelihood=decoder.compute_log_likelihood_unsupervised(
#                                    it.izip(read_monolingual_corpus(fcorpus_fname),read_transliterations(log_dir,itern)),
#                                    wpairs_aligns,wpairs_weights,wpairs_eword_weights
#                                  )
#        yield likelihood    

#def equal_sentences(): 
#    #for i in xrange(1,10): 
#    #    bitvec=[len(f)==len(e) for f, e in it.izip(read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/bn-hi/train.bn'),read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb/7_b_again/bn-hi/log/{}/transliterations.txt'.format(i) ))]
#    #    print sum(bitvec) 
#
#    bitvec=[len(f)>=len(e) for f, e in it.izip(read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/data/parallel/pb/bn-hi/train.bn'),read_monolingual_corpus('/home/development/anoop/experiments/unsupervised_transliterator/data/parallel/pb/bn-hi/train.hi' ))]
#    print sum(bitvec) 

def plot_confusion_matrix(confusion_mat_fname,tgt='hi'): 
    """
    Plots a heat map of the confusion matrix of character alignments. 
    
    confusion_mat_fname: The input is a confusion matrix generated by the align script
    tgt: target language of the transliteration
    
    The heat map shows characters in Devanagiri irrespective of target language for readability
    Needs 'Lohit Devanagari' font to be installed 
    """
    
    matplotlib.rc('font', family='Lohit Devanagari') 

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
    col_names=[ x if tgt=='hi' else unicode_transliterate.UnicodeIndicTransliterator.transliterate(x,tgt,'hi') for x in columns]

    rows=list(confusion_df.index)
    row_names=[ x if tgt=='hi' else unicode_transliterate.UnicodeIndicTransliterator.transliterate(x,tgt,'hi') for x in rows]
    
    plt.figure(figsize=(20,10))

    #plt.pcolor(data,cmap=plt.cm.gray_r,edgecolors='k')
    plt.pcolor(data,cmap=plt.cm.hot_r,edgecolors='k')
    
    #plt.pcolor(data,edgecolors='k')
    plt.colorbar()
    plt.xticks(np.arange(0,len(col_names))+0.5,col_names)
    plt.yticks(np.arange(0,len(row_names))+0.5,row_names)
    
    plt.show()
    plt.close()
    
def find_top_errors(confusion_mat_fname,nerr=5):
    """
    Displays the top `nerr` character level errors, sorted in decreasing order by error count  
    
    confusion_mat_fname: The input is a confusion matrix generated by the align script
    """
    
    confusion_df=pd.read_pickle(confusion_mat_fname)
    
    schar=list(confusion_df.index)
    tchar=list(confusion_df.columns)
    
    err_list=[]
    for s in schar:
        for t in tchar:
            if not s==t: 
                err_list.append(((s,t),confusion_df.ix[s,t]))
    total=sum(map(lambda x: x[1]  ,err_list))                    
    err_list=map(lambda x: (x[0],x[1],float(x[1])/float(total))  ,err_list)
    err_list.sort(key=lambda x:x[1],reverse=True)
    
    return err_list[:min(nerr,len(err_list))]

def find_phrase_errors(alignments_fname,nerr=20):     
    """
    Displays the top `nerr` phrase level errors, sorted in decreasing order by error count 
    
    alignments_fname: The input is the alignment file generated by the align script
    """
    
    with open(alignments_fname,'r') as align_file: 
        err_dict={}
        align_dbase=pickle.load(align_file)
        for src_aln, tgt_aln in align_dbase: 
            start=-1
            for i, (src_c,tgt_c) in enumerate(zip(src_aln,tgt_aln)):
                if start==-1 and src_c!=tgt_c:
                    start=i
                elif start>=0 and src_c==tgt_c:
                    cseq_pair=u''.join(src_aln[start:i])+u' '+u''.join(tgt_aln[start:i])
                    err_dict[cseq_pair]=err_dict.get(cseq_pair,0)+1
                    start=-1
            if start>=0:
                cseq_pair=u''.join(src_aln[start])+u' '+u''.join(tgt_aln[start])
                err_dict[cseq_pair]=err_dict.get(cseq_pair,0)+1
             
    err_list=err_dict.items()  
    err_list.sort(key=lambda x:x[1],reverse=True)

    return err_list

def print_errors(err_list):
    """
    Print the error list returned by the `find_phrase_errors` and `find_top_errors` functions
    err_list: error list 
    """
    refc=map(lambda x: x[0][0]  ,err_list)
    hypc=map(lambda x: x[0][1]  ,err_list)
    count=map(lambda x: x[1]  ,err_list)
    
    err_df=pd.DataFrame({'refc':refc,'hypc':hypc,'count':count})
    err_df=err_df.reindex_axis([u'refc', u'hypc', u'count'], axis=1)
    return err_df

#     for err in err_list: 
#         print u'{} {} {} {:.2%}'.format(err[0][0],err[0][1],err[1],err[2])
#         print u'{} {} {}'.format(err[0][0],err[0][1],err[1],err[2])


def find_examples_for_charpairs(alignments_fname,inchar_pair): 
    """
    Find examples for character level alignments in the alignment file. 
    
    alignments_fname: The input is the alignment file generated by the align script
    inchar_pair: char pair to search for 

    """
    with open(alignments_fname,'r') as align_file: 
        examples_list=[]
        align_dbase=pickle.load(align_file)
        for src_aln, tgt_aln in align_dbase: 
            if inchar_pair in zip(src_aln,tgt_aln):
                examples_list.append((u''.join(src_aln), u''.join(tgt_aln)))
        return examples_list
                        
def compute_phonetic_diff_vector(alignments_fname,lang): 
    """
	Gathers statistics on phonetic differences between languages
    	alignments_fname: The input is the alignment file generated by the align script
	lang: lang of the data in the alignment file
    """
    with open(alignments_fname,'r') as align_file: 
        diff_vector=np.array([0]*indic_scripts.PHONETIC_VECTOR_LENGTH)
        align_dbase=pickle.load(align_file)
        for src_aln, tgt_aln in align_dbase: 
            for src_c,tgt_c in zip(src_aln,tgt_aln):
                src_v=indic_scripts.get_phonetic_feature_vector(src_c,lang)
                tgt_v=indic_scripts.get_phonetic_feature_vector(tgt_c,lang)
                diff_vector=diff_vector + indic_scripts.xor_vectors(src_v,tgt_v)
        return diff_vector

def debug_training(log_dir): 
   #print list(av_entropy_generator(log_dir))
   #equal_sentences()
   #words_never_changed(log_dir)
   #model_size(log_dir)
   #likelihood(log_dir)
   pass 

if __name__=='__main__': 
    ### INDIC_NLP_RESOURCES environment variable must be set
    #loader.load()

    #plot_confusion_matrix(sys.argv[1])
    read_synthesized_parallel_corpus_from_moses_2(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],n_xlit=int(sys.argv[5]),n_tun=int(sys.argv[6]))
    pass 
