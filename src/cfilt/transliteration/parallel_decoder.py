from cfilt.transliteration.unsupervised import * 
from multiprocessing import Pool
import multiprocessing
import random
import itertools as it

from cfilt.transliteration.decoder import *

#class Test: 
#    def __init__(self,x,y):
#        self._rn=x+y

# Global decoder object, one for each process. 
# Decoder is not thread safe
decoder=None

# initializer
def initdecoder(translit_model,lm_model):
    global decoder
    decoder=TransliterationDecoder(translit_model,lm_model)

# task functions 
def task_decode(src):
    return decoder.decode(src)

def task_decode_char_list(char_list):
    return decoder._decode_internal(char_list)

# Convenience functions for parallel decoding
def parallel_decode_base(translit_model,lm_model,word_list,
                     n_processes=None,
                     decoder_func=task_decode,
                     ): 

    pool = Pool(processes=n_processes,initializer=initdecoder,initargs=[translit_model,lm_model]) 

    output=pool.map(decoder_func,word_list)
    pool.close()
    pool.join()

    return output


def parallel_decode(translit_model,lm_model,word_list,
                     n_processes=None,
                   ): 

    return parallel_decode_base(translit_model,lm_model,word_list,
                     n_processes,
                     task_decode,
                     ) 


def parallel_decode_char_id(translit_model,lm_model,word_list,
                     n_processes=None,
                     ): 

    return parallel_decode_base(translit_model,lm_model,word_list,
                     n_processes,
                     task_decode_list,
                     ) 

def parallel_evaluate(translit_model,lm_model,word_pairs,
                     n_processes=None,
                     ): 

    f_input_words=[ w[0] for w in word_pairs ]

    best_outputs=parallel_decode_base(translit_model,lm_model,f_input_words,
                     n_processes,
                     task_decode_char_list,
                     ) 
    for (f_input_word, e_output_word), best_output  in it.izip(word_pairs, best_outputs): 
        print u'Input: {}'.format(''.join(f_input_word)).encode('utf-8')
        print u'{} {} {}'.format(*map(lambda x:''.join(x), [f_input_word,e_output_word,best_output]) ).encode('utf-8')

if __name__ == '__main__': 
    pass 