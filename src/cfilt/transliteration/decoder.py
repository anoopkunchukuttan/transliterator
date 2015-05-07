import itertools as it
import codecs, sys, pickle
import pprint 
from collections import defaultdict
import numpy as np, math
import pandas as pd
import random

import srilm

from cfilt.utilities import timer
from cfilt.transliteration.utilities import *

class TransliterationModel:
    """
        The transliteration Model 
    """

    @staticmethod
    def construct_transliteration_model(fcfname,ecfname,alignmentfname): 
        tm=TransliterationModel()

        #with open(ecfname,'r') as ecfile: 
        #    for l in ecfile: 
        #        print unicode(l,'utf-8','strict').encode('utf-8')
        #        #print hex(ord(l.strip()))

        with codecs.open(fcfname,'r','utf-8') as fcfile: 
            fchars=[x.strip() for x in fcfile.readlines()]
               
            for i,c in enumerate(fchars): 
                tm.f_sym_id_map[c]=i
                tm.f_id_sym_map[i]=c

        with codecs.open(ecfname,'r','utf-8') as ecfile: 

            echars=[x.strip() for x in ecfile.readlines()]

            for i,c in enumerate(echars): 
                tm.e_sym_id_map[c]=i
                tm.e_id_sym_map[i]=c

        with codecs.open(alignmentfname,'r','utf-8') as afile: 
            
            tm.param_values=np.zeros([len(tm.e_sym_id_map),len(tm.f_sym_id_map)])
            for e_id, line in enumerate(iter(afile)): 
                fields=[ float(x) for x in line.strip().split('\t') ]

                for f_id, v in enumerate(fields): 
                    tm.param_values[e_id,f_id]=v
                
                norm_factor=np.sum(tm.param_values[e_id,:])
                tm.param_values[e_id,:]=tm.param_values[e_id,:]/norm_factor if norm_factor>0.0 else 0.0
                print tm.param_values[e_id,:]

        return tm

    @staticmethod
    def load_translit_model(translit_model_fname): 
        with open(translit_model_fname,'r') as translit_model_file: 
            return pickle.load(translit_model_file)

    @staticmethod
    def save_translit_model(model_obj, translit_model_fname): 
        with open(translit_model_fname,'w') as translit_model_file: 
            pickle.dump(model_obj,translit_model_file)

    def __init__(self):

        self.f_sym_id_map={}
        self.e_sym_id_map={}

        self.f_id_sym_map={}
        self.e_id_sym_map={}

        # Dictionary of parameter values (point alignment) to its probability
        self.param_values={}

class TransliterationDecoder: 

    def __init__(self,translit_model, lm_model, order=2): 
        self._lm_model=lm_model
        self._lm_order=order
        self._translit_model=translit_model

        e_vocabsize=len(self._translit_model.e_id_sym_map)
        self._lm_cache=np.ones(  (e_vocabsize,e_vocabsize)  )*-1.0

    #def _bigram_score(self,hist_id,cur_id):
    #    """
    #    """
    #    bigram=u'{} {}'.format( self._translit_model.e_id_sym_map[hist_id] if hist_id>=0 else u'<s>',
    #                            self._translit_model.e_id_sym_map[cur_id])
    #    return math.pow( 10 , srilm.getBigramProb(self._lm_model,bigram.encode('utf-8')) )

    def _bigram_score(self,hist_id,cur_id):
        """
        """
        if self._lm_cache[hist_id,cur_id]==-1:
            bigram=u'{} {}'.format( self._translit_model.e_id_sym_map[hist_id] if hist_id>=0 else u'<s>',
                                    self._translit_model.e_id_sym_map[cur_id])
            self._lm_cache[hist_id,cur_id]=math.pow( 10 , srilm.getBigramProb(self._lm_model,bigram.encode('utf-8')) )

        return self._lm_cache[hist_id,cur_id]

    def _get_param_value( self, e_id, f_input_chars ): 
        if self._translit_model.f_sym_id_map.has_key(f_input_chars): 
            return self._translit_model.param_values[  e_id , self._translit_model.f_sym_id_map[f_input_chars] ]
        else:
            return 0.0

    #@profile
    def _decode_internal(self,f_input_word): 
        """
            bigram language model 
        """

        sm_shape=(len(f_input_word), len(self._translit_model.e_sym_id_map.keys()) )


        #print 'Parameter Space: ',
        #print self._translit_model.param_values.shape

        #print 'Scoring Matrix dimensions: ',
        #print sm_shape

        # score matrix
        score_matrix=np.zeros( sm_shape ) 
        # backtracking matrices 
        max_row_matrix=np.zeros( sm_shape ,dtype=int ) 
        max_char_matrix=np.zeros( sm_shape,dtype=int ) 

        # initialization
        for k in xrange(sm_shape[1]):
            #print u'{} {}'.format( self._translit_model.e_id_sym_map[k] ,self._bigram_score(-1,k)).encode('utf-8')
            #print u'{} {} {}'.format( self._translit_model.e_id_sym_map[k] , f_input_word[0], self._get_param_value(  k , f_input_word[0])).encode('utf-8')
            max_row_matrix[0,k]=-1                       
            max_char_matrix[0,k]=-1                       
            score_matrix[0,k]= sum ( map ( log_z  ,
                                  #    LM score                    translition matrix score
                               [ self._bigram_score(-1,k) , self._get_param_value(  k , f_input_word[0]) ] 
                         ) 
                       )

        ### modified  viterbi decoding
        # j- input position
        # k- output char id (current)
        # m- output char id (prev)


        # Compute the scoring matrix and create the backtracking vector 

        for j in xrange(1,np.shape(score_matrix)[0]):
            #print 'Row: {}'.format(j)
            for k in xrange(np.shape(score_matrix)[1]):
                #print 'Char: {}'.format(k)
                # Case 1: A new output character is generated by input(j) alone
                max_val_1=float('-inf')
                max_char_1=-1

                # evaluate all (j-1) candidates
                for m in xrange(np.shape(score_matrix)[1]):
                    v=score_matrix[j-1,m] + sum ( map ( log_z ,
                               # LM score                    transliteration matrix score
                               [ self._bigram_score(m,k) , self._get_param_value(  k , f_input_word[j] )  ] 
                         ) 
                       )  
                            
                    if v>max_val_1:
                        max_val_1=v
                        max_char_1=m
                        #print '({} {})'.format(max_val_1,max_char_1)
                        #print 'DDD: {} {} {}'.format( score_matrix[j-1,m] , self._bigram_score(m,k) , self._get_param_value(  k , f_input_word[j] )  ) 
                         
                # Case 2: A new output character is generated by input(j) and input(j-1)
                max_val_2=float('-inf')
                max_char_2=-1

                if j>=2:
                    # evaluate all (j-2) candidates
                    for m in xrange(np.shape(score_matrix)[1]):
                        v=score_matrix[j-2,m] +  sum ( map ( log_z ,
                                   # LM score                    transliteration matrix score
                                   [ self._bigram_score(m,k) , self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j]) ] 
                             ) 
                           )  

                        if v>max_val_2:
                            max_val_2=v
                            max_char_2=m
                            #print '({} {})'.format(max_val_2,max_char_2)
                            #print 'DDD: {} {} {}'.format( score_matrix[j-2,m] , self._bigram_score(m,k) , self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j])  ) 

                # Max of Case 1 and Case 2
                if max_val_2 > max_val_1 : 
                    score_matrix[j,k] = max_val_2
                    max_row_matrix[j,k] = j-2
                    max_char_matrix[j,k] = max_char_2
                else: 
                    score_matrix[j,k] = max_val_1
                    max_row_matrix[j,k] = j-1
                    max_char_matrix[j,k] = max_char_1

                #print 'Final: ({} {} {})'.format(score_matrix[j,k],max_row_matrix[j,k],max_char_matrix[j,k])

        #pd.DataFrame(score_matrix).to_csv(''.join(f_input_word)+'.csv')
        #pd.DataFrame(max_row_matrix).to_csv(''.join(f_input_word)+'_row.csv')
        #pd.DataFrame(max_char_matrix).to_csv(''.join(f_input_word)+'_char.csv')

        # Backtrack and compute the best output sequence
        decoded_output=[]

        cur_j=sm_shape[0]-1
        cur_k=np.argmax(score_matrix[cur_j,:])

        while cur_j>=0:
            #print u'{} {} {}'.format(cur_j,cur_k,self._translit_model.e_id_sym_map[cur_k]).encode('utf-8')
            decoded_output.append( self._translit_model.e_id_sym_map[cur_k] )
            new_j=max_row_matrix[cur_j,cur_k]
            new_k=max_char_matrix[cur_j,cur_k] 
            cur_j= new_j
            cur_k = new_k

            #cur_j=cur_j-1
            #cur_k=np.argmax(score_matrix[cur_j,:])

        return list(reversed(decoded_output))
 
    def decode(self,f_input_str):
        return ''.join( self._decode_internal( map(lambda x:x,f_input_str)  ) )

    def evaluate(self,word_pairs):
        for f_input_word, e_output_word in word_pairs: 
            print u'Input: {}'.format(''.join(f_input_word)).encode('utf-8')
            best_output=''.join(self._decode_internal(f_input_word))

            print u'{} {} {}'.format(*map(lambda x:''.join(x), [f_input_word,e_output_word,best_output]) ).encode('utf-8')
