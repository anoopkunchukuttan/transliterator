#Copyright Anoop Kunchukuttan 2015 - present
# 
#This file is part of the IITB Unsupervised Transliterator 
#
#IITB Unsupervised Transliterator is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#IITB Unsupervised Transliterator  is distributed in the hope that it will be useful, 
#but WITHOUT ANY WARRANTY; without even the implied warranty of 
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
#GNU General Public License for more details. 
#
#You should have received a copy of the GNU General Public License 
#along with IITB Unsupervised Transliterator.   If not, see <http://www.gnu.org/licenses/>.

import itertools as it
import codecs
import math
import numpy as np
import yaml

import srilm

ZERO_LOG_PROB_REPLACEMENT=-700.0

def log_z(x): 
    return  math.log(x) if x>0.0 else ZERO_LOG_PROB_REPLACEMENT

numpy_log_z=np.vectorize(log_z)

LOG_E_BASE10=np.log10(np.e)

def load_lm_model(lm_fname, order=2): 
    """
    function to load language model 
    """
    lm_model=srilm.initLM(order)
    srilm.readLM(lm_model,lm_fname)    
    return lm_model

def read_yaml_file(fname):
    """
        read yaml configuration file 
    """
    cf=open(fname,'r')
    config=yaml.load(cf.read())
    cf.close()
    return config 

### Corpus reading utilities 

def read_parallel_corpus(fcorpus_fname,ecorpus_fname): 
    with codecs.open(fcorpus_fname,'r','utf-8') as ffile:
        with codecs.open(ecorpus_fname,'r','utf-8') as efile:
            return [ ( f.strip().split() , e.strip().split()  )  for f,e in it.izip( iter(ffile)  , iter(efile)  ) ] 

def read_monolingual_corpus(corpus_fname): 
    with codecs.open(corpus_fname,'r','utf-8') as infile:
            return [ w.strip().split()  for w in infile ] 

def write_monolingual_corpus(corpus_fname,output_list): 
    with codecs.open(corpus_fname,'w','utf-8') as outfile:
        for output in output_list: 
            outfile.write(u' '.join(output) + '\n')

def generate_char_set(fname):
    char_set=set()
    for input_word in read_monolingual_corpus(fname): 
        char_set.update(input_word)
    return list(char_set)

