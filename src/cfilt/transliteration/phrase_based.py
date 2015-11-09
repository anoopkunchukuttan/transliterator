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

#def read_synthesized_parallel_corpus(parallel_dir,src_l,tgt_l,n_xlit=10,n_tun=1000):
#    """
#    Read the final synthesizede parallel corpus from the final iteration 
#    nxlit: number of transliterations to extract 
#       reads word triplets 
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

#def read_synthesized_parallel_corpus_from_moses(src_fname,parallel_dir,src_l,tgt_l,n_xlit=10,n_tun=1000):
#    """
#    Read the final synthesizede parallel corpus obtained from decoding with final paramters, which is in moses format  
#    nxlit: number of transliterations to extract 
#    """
#
#    ### Methods for parsing n-best lists
#    def parse_line(line):
#        """
#            line in n-best file 
#            return list of fields
#        """
#        fields=[ x.strip() for x in  line.strip().split('|||') ]
#        fields[0]=int(fields[0])
#        fields[3]=float(fields[3])
#        return fields
#    
#    def iterate_nbest_list(nbest_fname): 
#        """
#            nbest_fname: moses format nbest file name 
#            return iterator over tuple of sent_no, list of n-best candidates
#    
#        """
#    
#        infile=codecs.open(nbest_fname,'r','utf-8')
#        
#        for sent_no, lines in it.groupby(iter(infile),key=lambda x:parse_line(x)[0]):
#            parsed_lines = [ parse_line(line) for line in lines ]
#            yield((sent_no,parsed_lines))
#    
#        infile.close()
#
#    all_src_lines=None
#    with codecs.open(src_fname,'r','utf-8') as src_file: 
#        src_lines=[x.strip() for x in src_file.readlines()]
#
#    data=list(it.izip(src_lines,iterate_nbest_list('{}/all.topn.reranked.{}'.format(parallel_dir,tgt_l))))
#
#    ### training data 
#    train_src_file=codecs.open('{}/train.{}'.format(parallel_dir,src_l),'w','utf-8')
#    train_tgt_file=codecs.open('{}/train.{}'.format(parallel_dir,tgt_l),'w','utf-8')
#    train_score_file=codecs.open('{}/train.score'.format(parallel_dir),'w','utf-8')
#
#    for src,(i,tgt) in data[:-n_tun]: 
#        total=sum( [ score for j, xlit, _, score in tgt[:min(len(tgt),n_xlit)]] )
#        for j, xlit, _, score in tgt[:min(len(tgt),n_xlit)]:
#            train_src_file.write(u' '.join(src)+u'\n')
#            train_tgt_file.write(u' '.join(xlit)+u'\n')
#            train_score_file.write(str(score/total)+u'\n')
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
#    for src,(i,tgt) in data[-n_tun:]: 
#        total=sum( [ score for j, xlit, _, score in tgt[:min(len(tgt),n_xlit)]] )
#        for j, xlit, _, score in tgt[:min(len(tgt),n_xlit)]:
#            tun_src_file.write(u' '.join(src)+u'\n')
#            tun_tgt_file.write(u' '.join(xlit)+u'\n')
#            tun_score_file.write(str(score/total)+u'\n')
#
#    tun_src_file.close()
#    tun_tgt_file.close()
#    tun_score_file.close()

def create_synthetic_corpus(src_fname,tgt_fname,parallel_dir,src_l,tgt_l,n_xlit=10,n_tun=1000):
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

    data=list(it.izip(src_lines,iterate_nbest_list(tgt_fname)))

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

def create_moses_run_params(conf_template_fname,conf_fname,workspace_dir,parallel_corpus,lm_file,src_lang,tgt_lang): 

    with codecs.open(conf_fname,'w','utf-8') as conf_file: 
        conf_template=read_lines(conf_template_fname)
        conf=conf_template.format(workspace_dir=workspace_dir,parallel_corpus=parallel_corpus,lm_file=lm_file,src_lang=src_lang,tgt_lang=tgt_lang)
        conf_file.write(conf)


if __name__=='__main__': 
    ### INDIC_NLP_RESOURCES environment variable must be set
    loader.load()

    command=sys.argv[1]
    if command=='create_synthetic_corpus': 
        create_synthetic_corpus(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],n_xlit=int(sys.argv[7]),n_tun=int(sys.argv[8]))
