from multiprocessing import Pool
import multiprocessing
import random, functools
import itertools as it

from cfilt.transliteration.decoder import *
from cfilt.transliteration.utilities import *

decoder=None

# initializer
def initdecoder(translit_model,lm_model):
    global decoder
    decoder=TransliterationDecoder(translit_model,lm_model)

# task functions 
def task_decode(src):
    return decoder.decode(src)

# Convenience functions for parallel decoding
def parallel_decode(translit_model,lm_model,word_list,
                     n_processes=None,
                   ): 

    pool = Pool(processes=n_processes,initializer=initdecoder,initargs=[translit_model,lm_model]) 

    output=pool.map(task_decode,word_list)
    pool.close()
    pool.join()

    return output

def parallel_evaluate(translit_model,lm_model,word_pairs,
                     n_processes=None
                     ): 

    f_input_words=[ w[0] for w in word_pairs ]

    best_outputs=parallel_decode(translit_model,lm_model,f_input_words,
                     n_processes,
                     ) 
    for (f_input_word, e_output_word), best_output  in it.izip(word_pairs, best_outputs): 
        print u'Input: {}'.format(''.join(f_input_word)).encode('utf-8')
        print u'{} {} {}'.format(*map(lambda x:''.join(x), [f_input_word,e_output_word,best_output]) ).encode('utf-8')


# task functions 
def task_decode_topn(char_list,topn):
    return decoder.decode_topn(char_list, topn)

# Convenience functions for parallel decoding
def parallel_decode_topn(translit_model,lm_model,word_list, topn,
                     n_processes=None,
                     ): 

    pool = Pool(processes=n_processes,initializer=initdecoder,initargs=[translit_model,lm_model]) 

    p_task_decode_topn =  functools.partial(task_decode_topn,topn=topn)
    output=pool.map(p_task_decode_topn,word_list)
    pool.close()
    pool.join()

    return output

## Seems main in needed for some reason for using multiprocessor in this fashion
if __name__ == '__main__': 
    pass 
