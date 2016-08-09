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

from multiprocessing import Pool
import multiprocessing
import functools
import itertools as it

from cfilt.transliteration.decoder import *
from cfilt.transliteration.utilities import *

decoder=None

# initializer
def initdecoder(translit_model,lm_model,decoder_params={}):
    global decoder
    decoder=TransliterationDecoder(translit_model,lm_model,decoder_params)

# task functions 
def task_decode(src):
    return decoder.decode(src)

def task_loglikelihood_unsupervised(wpair):
    return decoder.compute_log_likelihood_unsupervised([wpair])

# Convenience functions for parallel decoding
def parallel_decode(translit_model,lm_model,word_list, decoder_params={},
                     n_processes=None,
                   ): 

    pool = Pool(processes=n_processes,initializer=initdecoder,initargs=[translit_model,lm_model,decoder_params]) 

    output=pool.map(task_decode,word_list)
    pool.close()
    pool.join()

    return output

# Convenience functions for parallel decoding
def parallel_likelihood_unsupervised(translit_model, word_pair_list, decoder_params={},
                     n_processes=None,
                   ): 

    pool = Pool(processes=n_processes,initializer=initdecoder,initargs=[translit_model,lm_model,decoder_params]) 

    wll_list=pool.map(task_loglikelihood_unsupervised,word_pair_list)
    pool.close()
    pool.join()

    return sum(wll_list)

def parallel_evaluate(translit_model,lm_model,word_pairs,decoder_params={},
                     n_processes=None
                     ): 

    f_input_words=[ w[0] for w in word_pairs ]

    best_outputs=parallel_decode(translit_model,lm_model,f_input_words, decoder_params,
                     n_processes,
                     ) 
    for (f_input_word, e_output_word), best_output  in it.izip(word_pairs, best_outputs): 
        print u'Input: {}'.format(''.join(f_input_word)).encode('utf-8')
        print u'{} {} {}'.format(*map(lambda x:''.join(x), [f_input_word,e_output_word,best_output]) ).encode('utf-8')


# task functions 
def task_decode_topn(char_list,topn):
    return decoder.decode_topn(char_list, topn)

# Convenience functions for parallel decoding
def parallel_decode_topn(translit_model, lm_model, word_list, topn, decoder_params={},
                     n_processes=None,
                     ): 

    pool = Pool(processes=n_processes,initializer=initdecoder,initargs=[translit_model,lm_model,decoder_params]) 

    p_task_decode_topn =  functools.partial(task_decode_topn,topn=topn)
    output=pool.map(p_task_decode_topn,word_list)
    pool.close()
    pool.join()

    return output

## Seems main in needed for some reason for using multiprocessor in this fashion
if __name__ == '__main__': 
    pass 
