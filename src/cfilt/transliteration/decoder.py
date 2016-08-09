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
import codecs, sys, pickle
import pprint 
from collections import defaultdict
import numpy as np, math
import pandas as pd
import random
import heapq, operator


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

    NULL_CHAR=u'`' # have used the backquotes character, which is unlikely to be found in text

    def __init__(self):

        self.f_sym_id_map={}
        self.e_sym_id_map={}

        self.f_id_sym_map={}
        self.e_id_sym_map={}

        # This is a dummy symbol to denote a null symbol in an n-gram - for representation of k-grams k<n, the order of the LM
        self.add_e_sym(TransliterationModel.NULL_CHAR)

        # Dictionary of parameter values (point alignment) to its probability
        self.param_values={}

    def add_f_sym(self,sym): 
        if sym not in self.f_sym_id_map: 
            self.f_sym_id_map[sym]=len(self.f_sym_id_map)
            self.f_id_sym_map[self.f_sym_id_map[sym]]=sym

    def add_e_sym(self,sym): 
        if sym not in self.e_sym_id_map: 
            self.e_sym_id_map[sym]=len(self.e_sym_id_map)
            self.e_id_sym_map[self.e_sym_id_map[sym]]=sym

class TransliterationDecoder: 

    """
    Default value for decoder parameters 
    """
    DEFAULT_LM_ORDER=2
    DEFAULT_PROB_STRETCH_POWER=1.0
    DEFAULT_LENGTH_CONTROL_ADD=0.0 
    DEFAULT_BEAM_PRUNE_SIZE=1000

    """
    Value of the max_beam_prune_size paramter so that no pruning takes place. Never use it !!!
    """
    NO_BEAM_PRUNE_SIZE=-1

    """
    Indices of the fields in the hypothesis list 
    """
    HYP_STATE =0
    HYP_PREV_POSN=1
    HYP_PREV_IDX=2
    HYP_SCORE =3
    HYP_LENGTH =4


    def __init__(self,translit_model, lm_model, params={}): 

        # input    
        self._translit_model=translit_model
        self._lm_model=lm_model

        # decoder parameters 
        self._lm_order=params.get('lm_order',TransliterationDecoder.DEFAULT_LM_ORDER)
        self._stretch =params.get('prob_stretch_power',TransliterationDecoder.DEFAULT_PROB_STRETCH_POWER)
        self._length_control_param=params.get('length_control_add',TransliterationDecoder.DEFAULT_LENGTH_CONTROL_ADD)
        self._max_beam_prune_size=params.get('max_beam_prune_size',TransliterationDecoder.DEFAULT_BEAM_PRUNE_SIZE)

        self._esize=len(self._translit_model.e_id_sym_map)

        ### local cache for speedup
        # bigram cache
        #TODO: initialization with -1.0 seems dicey - Check 
        self._lm_cache=np.ones(  ( int(np.power(self._esize,self._lm_order-1)), self._esize )  )*-1.0

        # ngram cache 
        self._lm_cache_ngram={}
        #self._lm_cache_ngram=np.ones(  [self._esize] * self._lm_order  )

        # transliteration probabilities pre-processed: log and cubing 
        self._param_values=self._stretch*numpy_log_z(self._translit_model.param_values)

    def _word_log_score(self,word):
        return srilm.getSentenceProb(self._lm_model,u' '.join(word).encode('utf-8'),len(word))

    #@profile
    def _get_param_value( self, e_id, f_input_chars):
        if self._translit_model.f_sym_id_map.has_key(f_input_chars): 
            return self._param_values[  e_id , self._translit_model.f_sym_id_map[f_input_chars] ]
        else: 
            return log_z(0.0)

    def _bigram_log_score(self,hist_id,cur_id):
        """
        """
        if self._lm_cache[hist_id,cur_id]==-1:
            bigram=u'{} {}'.format( self._translit_model.e_id_sym_map[hist_id] if hist_id>=0 else u'<s>',
                                    self._translit_model.e_id_sym_map[cur_id])
            self._lm_cache[hist_id,cur_id]=srilm.getBigramProb(self._lm_model,bigram.encode('utf-8'))/LOG_E_BASE10

        return self._lm_cache[hist_id,cur_id]

    #@profile
    def decode(self,f_input_word): 
        """
            bigram language model 
        """

        sm_shape=(len(f_input_word), len(self._translit_model.e_sym_id_map.keys()) )

        # score matrix
        score_matrix=np.zeros( sm_shape ) 
        # backtracking matrices 
        max_row_matrix=np.zeros( sm_shape ,dtype=int ) 
        max_char_matrix=np.zeros( sm_shape,dtype=int ) 

        # initialization
        #print 'Initialization'
        for k in xrange(sm_shape[1]):

            # for first input character generating first output character
            max_row_matrix[0,k]=-1                       
            max_char_matrix[0,k]=-1                       
                                  #    LM score                    translition matrix score
            score_matrix[0,k]= self._bigram_log_score(-1,k) + self._get_param_value(  k , f_input_word[0]) 
            #print 'Candidates for {} {}: {}'.format(0,k,score_matrix[0,k])

        ### modified  viterbi decoding
        # j- input position
        # k- output char id (current)
        # m- output char id (prev)


        # Compute the scoring matrix and create the backtracking vector 

        for j in xrange(1,np.shape(score_matrix)[0]):
            #print 'Input: {}'.format(j)
            for k in xrange(np.shape(score_matrix)[1]):
                #print 'Cur Char: {}'.format(k)
                # Case 1: A new output character is generated by input(j) alone
                max_val_1=float('-inf')
                max_char_1=-1

                # evaluate all (j-1) candidates
                for m in xrange(np.shape(score_matrix)[1]):
                    #print 'Prev Char: {}'.format(m)
                    #                               LM score                    transliteration matrix score
                    v=score_matrix[j-1,m] + self._bigram_log_score(m,k)  + self._get_param_value(  k , f_input_word[j] )  
                            
                    if v>max_val_1:
                        max_val_1=v
                        max_char_1=m
                        #print '({} {})'.format(max_val_1,max_char_1)
                        #print 'DDD: {} {} {}'.format( score_matrix[j-1,m] , self._bigram_score(m,k) , self._get_param_value(  k , f_input_word[j] )  ) 
                         
                # Case 2: A new output character is generated by input(j) and input(j-1)
                max_val_2=float('-inf')
                max_char_2=-1
                
                if j==1:
                    ## first 2 input characters generate the first output
                                    # LM score                    transliteration matrix score
                    max_val_2=self._bigram_log_score(-1,k) +  self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j])  

                if j>=2:
                    # evaluate all (j-2) candidates
                    for m in xrange(np.shape(score_matrix)[1]):
                        #print 'Prev Char: {}'.format(m)
                                   # LM score                    transliteration matrix score
                        v=score_matrix[j-2,m] + self._bigram_log_score(m,k) + self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j])  

                        if v>max_val_2:
                            max_val_2=v
                            max_char_2=m
                            #print '({} {})'.format(max_val_2,max_char_2)
                            #print 'DDD: {} {} {}'.format( score_matrix[j-2,m] , self._bigram_score(m,k) , self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j])  ) 

                # Max of Case 1 and Case 2
                #print 'Candidates for {} {}:'.format(j,k)
                #print '{} {}'.format(j-1,max_char_1)
                #print '{} {}'.format(j-2,max_char_2)

                if max_val_2 > max_val_1 : 
                    score_matrix[j,k] = max_val_2 
                    max_row_matrix[j,k] = j-2
                    max_char_matrix[j,k] = max_char_2
                else: 
                    score_matrix[j,k] = max_val_1
                    max_row_matrix[j,k] = j-1
                    max_char_matrix[j,k] = max_char_1
                score_matrix[j,k]+=self._length_control_param

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

        return list(reversed(decoded_output))
 
    def evaluate(self,word_pairs):
        for f_input_word, e_output_word in word_pairs: 
            print u'Input: {}'.format(''.join(f_input_word)).encode('utf-8')
            best_output=''.join(self.decode(f_input_word))

            print u'{} {} {}'.format(*map(lambda x:''.join(x), [f_input_word,e_output_word,best_output]) ).encode('utf-8')

    def _backtrack_nbest_output(self,max_row_matrix,max_char_matrix,max_nbest_matrix,j,k,n): 

        decoded_output=[]

        cur_j, cur_k, cur_n = j, k, n

        #print '==========='
        while cur_j>=0:
            #print u'{} {} {} {}'.format(cur_j,cur_k, cur_n, self._translit_model.e_id_sym_map[cur_k]).encode('utf-8')
            decoded_output.append( self._translit_model.e_id_sym_map[cur_k] )

            cur_j, cur_k, cur_n = [   
                    max_row_matrix[cur_j,cur_k,cur_n],
                    max_char_matrix[cur_j,cur_k,cur_n], 
                    max_nbest_matrix[cur_j,cur_k,cur_n], 
                ]                

        return list(reversed(decoded_output))

    def decode_topn(self,f_input_word,topn=1): 
        """
            bigram language model 
        """

        sm_shape=(len(f_input_word), len(self._translit_model.e_sym_id_map.keys()), topn )

        # score matrix
        score_matrix=np.zeros( sm_shape ) 
        # backtracking matrices 
        max_row_matrix=np.zeros( sm_shape ,dtype=int ) 
        max_char_matrix=np.zeros( sm_shape,dtype=int ) 
        max_nbest_matrix=np.zeros( sm_shape,dtype=int ) 

        # initialization
        #print 'Initialization'
        for k in xrange(sm_shape[1]):
            for n in xrange(sm_shape[2]):
                max_row_matrix[0,k,n]=-1                       
                max_char_matrix[0,k,n]=-1                       
                max_nbest_matrix[0,k,n]=-1
                                      #    LM score                    translition matrix score
                score_matrix[0,k,n]= self._bigram_log_score(-1,k) + self._get_param_value(  k , f_input_word[0]) 
            #print 'Candidates for {} {}: {}'.format(0,k,score_matrix[0,k,0])

        ### modified  viterbi decoding
        # j- input position
        # k- output char id (current)
        # m- output char id (prev)


        # Compute the scoring matrix and create the backtracking vector 

        for j in xrange(1,np.shape(score_matrix)[0]):
            #print 'Input posn: {}'.format(j)
            for k in xrange(np.shape(score_matrix)[1]):
                #print 'Cur Char: {}'.format(k)

                entry_scores=[]

                ## Case 1: A new output character is generated by input(j) alone
                for m in xrange(np.shape(score_matrix)[1]):
                    #print 'Prev Char: {}'.format(m)
                    for n in xrange(  np.shape(score_matrix)[2]  if j>1 else 1  ):  # for the first input candidate, only consider 1 candidate from initialization
                                   # LM score                    transliteration matrix score
                        v=score_matrix[j-1,m,n] + self._bigram_log_score(m,k) + self._get_param_value(  k , f_input_word[j] )
                        entry_scores.append( ( ( j-1, m, n )  , v  ) )


                # Case 2: A new output character is generated by input(j) and input(j-1)
                if j==1: 
                    ## first 2 input characters generate the first output
                               # LM score                    transliteration matrix score
                    v= self._bigram_log_score(-1,k)  + self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j])  
                    entry_scores.append( ( ( -1, -1, -1 )  , v  ) )

                if j>=2:
                    # evaluate all (j-2) candidates
                    for m in xrange(np.shape(score_matrix)[1]):
                        #print 'Prev Char: {}'.format(m)
                        for n in xrange(np.shape(score_matrix)[2] if j>2 else 1):
                                    # LM score                    transliteration matrix score
                            v=score_matrix[j-2,m,n] +  self._bigram_log_score(m,k) + self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j]) 
                            entry_scores.append( ( ( j-2, m, n )  , v  ) )

                top_n_candidates=heapq.nlargest(topn,entry_scores,key=operator.itemgetter(1))
                #print 'Candidates for {} {}:'.format(j,k)
                for n in xrange(len(top_n_candidates)): 
                    score_matrix[j,k,n]=top_n_candidates[n][1]+self._length_control_param
                    max_row_matrix[j,k,n]=top_n_candidates[n][0][0]
                    max_char_matrix[j,k,n]=top_n_candidates[n][0][1]
                    max_nbest_matrix[j,k,n]=top_n_candidates[n][0][2]
                    #print '{} {} {}: {}'.format(top_n_candidates[n][0][0],top_n_candidates[n][0][1],top_n_candidates[n][0][2],top_n_candidates[n][1])

        # find starting points for backtracking top-n outputs 
        final_top_n_candidates=heapq.nlargest(topn,
                it.product([sm_shape[0]-1],xrange(sm_shape[1]),xrange(sm_shape[2])),
                key=lambda ia: score_matrix[ ia[0], ia[1], ia[2] ] )

        nbest_list= [
                        (
                            self._backtrack_nbest_output(max_row_matrix,max_char_matrix,max_nbest_matrix,*candidate),
                            score_matrix[candidate[0], candidate[1], candidate[2]]
                        )    for candidate in final_top_n_candidates 

                    ]                             

        return nbest_list

    ##################################
    # N-gram  decoder with beam pruning 
    ##################################

    #@profile
    def _generate_ngram(self,hist_list,cur_id): 
        """
        Generates ngram corresponding to provided history/context and the current symbol 

        hist_list: the list of word ids in the history. Can be empty if no history

        cur_id: id of the current symbol. 
        """
        context_size=self._lm_order-1

        context_v=[]
        if len(hist_list)<context_size: 
            context_v.append(u'<s>')

        context_v.extend( [self._translit_model.e_id_sym_map[x] for x in hist_list ]  )
        context_v.append(self._translit_model.e_id_sym_map[cur_id])

        return (u' '.join(context_v), len(context_v))

    def _generate_index_representation(self,k_list): 
        """
        Using Horner's rule 
        Input has to be a valid state
        """

        k=0
        for i in k_list[:-1]:
            k+=i*self._esize
        return k+k_list[-1]

    #@profile
    def _ngram_log_score(self,hist_list,cur_id):
        """
        Generate n-gram probability score


        hist_list: the list of word ids in the history. Can be empty if no history

        cur_id: id of the current symbol. 
       """
        ngram_id=self._generate_index_representation(hist_list+[cur_id])
        #ngram_id= u'_'.join([str(x) for x in  hist_list+[cur_id]])

        if ngram_id not in self._lm_cache_ngram:
            ngram, len_ngram=self._generate_ngram(hist_list,cur_id)
            self._lm_cache_ngram[ngram_id]= srilm.getNgramProb(self._lm_model,ngram.encode('utf-8'),len_ngram)/LOG_E_BASE10 

        return self._lm_cache_ngram[ngram_id]

    #@profile
    #def _ngram_log_score(self,hist_list,cur_id):
    #    """
    #    Generate n-gram probability score


    #    hist_list: the list of word ids in the history. Can be empty if no history

    #    cur_id: id of the current symbol. 
    #   """
    #    ngram_id=tuple([0]*(self._lm_order-len(hist_list)-1)+hist_list+[cur_id])

    #    val=self._lm_cache_ngram.item(ngram_id)
    #    if val==1:
    #        ngram, len_ngram=self._generate_ngram(hist_list,cur_id)
    #        val=srilm.getNgramProb(self._lm_model,ngram.encode('utf-8'),len_ngram)/LOG_E_BASE10
    #        self._lm_cache_ngram.itemset(  ngram_id, val )

    #    return val


    #@profile
    def _backtrack_nbest_output_ngram(self,hypothesis_collection,candidate_hypothesis): 
        """
        Finds the best output for the candidates hypothesis, by backtracking
        """
        decoded_output=[]

        #### Initialize
        decoded_output.append(self._translit_model.e_id_sym_map[candidate_hypothesis[TransliterationDecoder.HYP_STATE][-1]]) 
        prev_j, prev_idx=candidate_hypothesis[TransliterationDecoder.HYP_PREV_POSN],candidate_hypothesis[TransliterationDecoder.HYP_PREV_IDX]

        ### Extract the remaining characters by extracting MSB from each of the states in the backtracking path
        while prev_j>=0:
            prev_hypo=hypothesis_collection[prev_j][prev_idx]
            decoded_output.append( self._translit_model.e_id_sym_map[ prev_hypo[TransliterationDecoder.HYP_STATE][-1] ] )
            prev_j, prev_idx=prev_hypo[TransliterationDecoder.HYP_PREV_POSN],prev_hypo[TransliterationDecoder.HYP_PREV_IDX]

        return list(reversed(decoded_output))

    #@profile
    def _generate_hypotheses(self,hypothesis_collection,f_input_word,j,prev_j): 
        hypotheses=[]

        for prev_idx, prev_hypothesis in enumerate(hypothesis_collection[prev_j]): 
          
            ### Code block 1 - used for profiling     
            #states= [   
            #                        (prev_hypothesis[TransliterationDecoder.HYP_STATE][1:] \
            #                                if len(prev_hypothesis[TransliterationDecoder.HYP_STATE])==self._lm_order-1 \
            #                            else  prev_hypothesis[TransliterationDecoder.HYP_STATE]
            #                         ) + [k] 
            #                for k in xrange(1,self._esize) ]

            #prev_js=[prev_j]*(self._esize-1)
            #prev_idxs=[prev_idx]*(self._esize-1)

            #scores= [   
            #                        prev_hypothesis[TransliterationDecoder.HYP_SCORE] + \
            #                            self._ngram_log_score(prev_hypothesis[TransliterationDecoder.HYP_STATE],k) + \
            #                            self._get_param_value(  k , u''.join(f_input_word[prev_j+1:j+1]) ) + \
            #                            self._length_control_param
            #                for k in xrange(1,self._esize) ]

            #lengths=[prev_hypothesis[TransliterationDecoder.HYP_LENGTH]+1]*(self._esize-1)

            #temp_hypotheses=list(it.izip(states,prev_js,prev_idxs,scores,lengths))

            ### Code block 2: same as Code block 1, represented as list comprehension 
            temp_hypotheses= [  [ 
                                    (prev_hypothesis[TransliterationDecoder.HYP_STATE][1:] \
                                            if len(prev_hypothesis[TransliterationDecoder.HYP_STATE])==self._lm_order-1 \
                                        else  prev_hypothesis[TransliterationDecoder.HYP_STATE]
                                     ) + [k] ,

                                    prev_j,
                                    prev_idx,

                                    prev_hypothesis[TransliterationDecoder.HYP_SCORE] + \
                                        self._ngram_log_score(prev_hypothesis[TransliterationDecoder.HYP_STATE],k) + \
                                        self._get_param_value(  k , u''.join(f_input_word[prev_j+1:j+1]) ) + \
                                        self._length_control_param ,

                                    prev_hypothesis[TransliterationDecoder.HYP_LENGTH]+1
                                ]                                    
                            for k in xrange(1,self._esize) ]
           
            hypotheses.extend(temp_hypotheses)

        return hypotheses

    #@profile
    def decode_topn_ngram(self,f_input_word,topn=1):
        """
            decoding with an ngram language model withg beam pruning and efficient computations 
        """

        print 'Start decoding'
        ## This is a list of list of hypothesis. The outer list is the list of hypothesis for each input position
        # Each of these hypothesis lists maintain a maximum of `max_beam_size` hypothesis at each input position
        hypothesis_collection=[]

        # initialization
        print 'Initialization'
        # for input position 0
        init_hypotheses=[]
        for k in xrange(1,self._esize): 
                                        #    LM score                    translition matrix score
            score=self._ngram_log_score([],k) + self._get_param_value(  k , f_input_word[0]) + self._length_control_param  
            ## hypothesis
            ## Tuple of (stat_list_representation, prev_input_posn, index_in_prev_hyp_list, score, length_of_output)
            h=[[k],-1,-1,score,1]
            init_hypotheses.append(h)

        ## top selected candidates by score
        top_beam_candidates=[]
        if self._max_beam_prune_size!=TransliterationDecoder.NO_BEAM_PRUNE_SIZE: 
            top_beam_candidates=heapq.nlargest(self._max_beam_prune_size,init_hypotheses,key=lambda x:x[TransliterationDecoder.HYP_SCORE])
        else: 
            top_beam_candidates=init_hypotheses
        hypothesis_collection.append(top_beam_candidates)

        print 'Initialized' 
        ### modified  viterbi decoding with beam pruning
        # j- input position

        # Compute the scoring matrix and create the backtracking vector 

        print 'Computing costs'             
        for j in xrange(1,len(f_input_word)):

            print 'Processing character: {}'.format(j) 

            current_hypotheses=[]

            # prev state from postn (j-1)
            current_hypotheses.extend(self._generate_hypotheses(hypothesis_collection,f_input_word,j,j-1))

            # prev state from postn (j-2)
            if j>1: 
                current_hypotheses.extend(self._generate_hypotheses(hypothesis_collection,f_input_word,j,j-2))
            else: 
                # for input position 1
                for k in xrange(1,self._esize): 
                                                #    LM score                    translition matrix score
                    score=self._ngram_log_score([],k) + self._get_param_value(  k , f_input_word[0] + f_input_word[1])  + self._length_control_param
                    h=[[k],-1,-1,score,1]
                    current_hypotheses.append(h)

            top_beam_candidates=[]
            if self._max_beam_prune_size!=TransliterationDecoder.NO_BEAM_PRUNE_SIZE: 
                top_beam_candidates=heapq.nlargest(self._max_beam_prune_size,current_hypotheses,key=lambda x:x[TransliterationDecoder.HYP_SCORE])
            else: 
                top_beam_candidates=current_hypotheses

            hypothesis_collection.append(top_beam_candidates)

        print 'Find best candidates' 
        # find starting points for backtracking top-n outputs 
        final_top_n_candidates=heapq.nlargest( topn,
                hypothesis_collection[-1],
                key=lambda x:x[TransliterationDecoder.HYP_SCORE] )
       
        print 'Do backtracking' 
        nbest_list= [
                        (
                            self._backtrack_nbest_output_ngram(hypothesis_collection,candidate_hypothesis),
                            candidate_hypothesis[TransliterationDecoder.HYP_SCORE]
                        )    for candidate_hypothesis in final_top_n_candidates 

                    ]                             

        return nbest_list

    #######  Likelihood Computation routines ##########
    #### Compute likelihood for candidates 
    #def _generate_alignments(self,f_word,e_word):
    #    """
    #        src_word: list of src characters
    #        tgt_word: list of tgt characters 
    #
    #        return: wpair_info for this pair
    #    """
    #    nf=len(f_word)
    #    ne=len(e_word)
    #    allowed_mappings=[1,2]
    #
    #    alignment_preseqs=list(it.ifilter( lambda x: sum(x)==nf, 
    #                            it.product(allowed_mappings ,repeat=ne)
    #                      ))
    #
    #    wpairs_aligns=[]
    #
    #    for ap in alignment_preseqs:
    #
    #        # compute the end offsets on 'f' side for each charseq_pair by cumulative addition 
    #        align_offsets=list(ap)
    #        for i in range(1,len(align_offsets)):
    #            align_offsets[i]=align_offsets[i]+align_offsets[i-1]

    #        # insert the first starting offset             
    #        align_offsets.insert(0,0)            
    #
    #        # extract the charseq_pair
    #        align=[]
    #        for i in range(len(align_offsets)-1): 
    #            #charseq_pair=u''.join( f_word[align_offsets[i]:align_offsets[i+1]] ) + u'|' + e_word[i]
    #            fs=u''.join( f_word[align_offsets[i]:align_offsets[i+1]] )
    #            es=e_word[i]

    #            # create 'f' sym to id mapping
    #            fs_id=  self._translit_model.f_sym_id_map[fs]  if   self._translit_model.f_sym_id_map.has_key(fs) else len(self._translit_model.f_sym_id_map)
    #            if fs_id==len(self._translit_model.f_sym_id_map):
    #                print 'Warning: unknown character'
    #                self._translit_model.f_sym_id_map[fs]=fs_id

    #            # create 'e' sym to id mapping
    #            es_id=  self._translit_model.e_sym_id_map[es]  if   self._translit_model.e_sym_id_map.has_key(es) else len(self._translit_model.e_sym_id_map)
    #            if es_id==len(self._translit_model.e_sym_id_map):
    #                print 'Warning: unknown character'
    #                self._translit_model.e_sym_id_map[es]=es_id

    #            align.append( [es_id,fs_id] )
    #       
    #        wpairs_aligns.append(align)
    #
    #    return wpairs_aligns
    #
    #
    #def _create_alignment_database(self,word_pairs):
    #    """
    #      every character sequence in the corpus can be uniquely addressed as: 
    #        corpus[0][word_pair_idx][align_idx][aln_point_idx]
    #
    #      every weight can be indexed as: 
    #        corpus[1][word_pair_idx][align_idx]
    #
    #    """
    #    #TODO: merge _create_alignment_database and _initialize_parameter_structures  into a single function _prepare_corpus_supervised

    #    wpairs_aligns=[]
    #    wpairs_weights=[]
    #    wpairs_eword_weights=[]
    #
    #    for f,e in word_pairs: 
    #        alignments=self._generate_alignments(f,e)
    #        if len(alignments)>0:
    #            wpairs_aligns.append(alignments)
    #            wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )
    #            wpairs_eword_weights.append(1.0)
    #        else: 
    #            #msg=u"No alignments from word pair: {} {}".format(u''.join(f),u''.join(e))
    #            #msg=u"No alignments from word pair: " + u''.join(f) + u" " + u''.join(e)
    #            #print msg.encode('utf-8') 
    #            pass 

    #    ## create inverse id-symbol mappings
    #    ## for f
    #    #for s,i in self._translit_model.f_sym_id_map.iteritems():
    #    #    self._translit_model.f_id_sym_map[i]=s
    #
    #    ## for e
    #    #for s,i in self._translit_model.e_sym_id_map.iteritems():
    #    #    self._translit_model.e_id_sym_map[i]=s

    #    return it.izip(wpairs_aligns, wpairs_weights, wpairs_eword_weights)

    #def compute_log_likelihood_unsupervised(self, word_pairs, wpairs_aligns=None, wpairs_weights=None,wpairs_eword_weights=None): 

    #    alignment_info=None

    #    if wpairs_aligns is None: 
    #        alignment_info=self._create_alignment_database(word_pairs)
    #    else:
    #        alignment_info=it.izip(wpairs_aligns,wpairs_weights,wpairs_eword_weights)

    #    ll=0.0
    #    i=0
    #    for (wpair_aligns, wpair_weights, wpair_eword_weight), (f, e) in it.izip(alignment_info, word_pairs): 
    #        wll=0.0
    #        for wpair_align, wpair_weight in it.izip(wpair_aligns,wpair_weights): 
    #            for e_id, f_id in wpair_align: 
    #                wll+=log_z(self._translit_model.param_values[e_id, f_id]) 
    #            wll*=wpair_weight                    
    #        wll+=self._word_log_score(e)
    #        ll+=wll

    #        #print '========= {} {} ==============='.format(i,len(wpair_aligns)) 
    #        i=i+1

    #    return ll             

    #def compute_log_likelihood_supervised(self, wpairs_aligns, wpairs_weights): 

    #    ll=0.0
    #    i=0
    #    for wpair_aligns, wpair_weights in it.izip(wpairs_aligns, wpairs_weights): 
    #        wll=0.0
    #        for wpair_align, wpair_weight in it.izip(wpair_aligns,wpair_weights): 
    #            for e_id, f_id in wpair_align: 
    #                wll+=log_z(self._translit_model.param_values[e_id, f_id]) 
    #            wll*=wpair_weight                    
    #        ll+=wll

    #        #print '========= {} {} ==============='.format(i,len(wpair_aligns)) 
    #        i=i+1

    #    return ll             

    ###################################
    ## N-gram Viterbi Decoder (exact i.e without pruning)
    ###################################

    #def _generate_ngram(self,hist_list,cur_id): 
    #    """
    #    Generates ngram corresponding to provided history/context and the current symbol 

    #    hist_list: the list of word ids in the history. Can be empty if no history

    #    cur_id: id of the current symbol. 
    #    """
    #    context_size=self._lm_order-1

    #    context_v=[]
    #    if len(hist_list)<context_size: 
    #        context_v.append(u'<s>')

    #    context_v.extend( [self._translit_model.e_id_sym_map[x] for x in hist_list ]  )
    #    context_v.append(self._translit_model.e_id_sym_map[cur_id])

    #    return (u' '.join(context_v), len(context_v))

    ##@profile
    #def _ngram_log_score(self,hist_id,hist_list,cur_id):
    #    """
    #    Generate n-gram probability score


    #    hist_list: the list of word ids in the history. Can be empty if no history

    #    cur_id: id of the current symbol. 
    #   """
    #    ngram_id=hist_id*self._esize+cur_id

    #    if ngram_id not in self._lm_cache_ngram:
    #        ngram, len_ngram=self._generate_ngram(hist_list,cur_id)
    #        self._lm_cache_ngram[ngram_id]= srilm.getNgramProb(self._lm_model,ngram.encode('utf-8'),len_ngram)/LOG_E_BASE10 

    #    return self._lm_cache_ngram[ngram_id]

    #def _backtrack_nbest_output_ngram(self,max_row_matrix,max_char_matrix,max_nbest_matrix,j,k,n): 
    #    """
    #    Finds the best output, ending with the character identified by the triplet (j,k,n), where: 

    #    j: index position of source character
    #    k: ending state/output sequence - k 
    #    n: nth best candidates for history/state k

    #    The length of the output is 'sl' 
    #
    #    The following datastructures are used to backtrack through the decoder generated trellis

    #    max_row_matrix: 
    #    max_char_matrix: 
    #    max_nbest_matrix:  

    #    """
    #    context_size=self._lm_order-1

    #    decoded_output=[]

    #    #### Initialize
    #    # Extract the last (order-2) or sl characters, whichever is less
    #    cur_j, cur_k, cur_n = j, k, n 
    #    cur_k_list=self._generate_list_representation_if_valid(cur_k)

    #    decoded_output.extend([self._translit_model.e_id_sym_map[x] for x in reversed(cur_k_list[1:])])
    #    
    #    if len(cur_k_list)<context_size:
    #        decoded_output.append(self._translit_model.e_id_sym_map[cur_k_list[0]])

    #    ### Extract the remaining characters by extracting MSB from each of the states in the backtracking path
    #    while len(cur_k_list)==context_size:
    #        decoded_output.append( self._translit_model.e_id_sym_map[cur_k_list[0]] )

    #        cur_j, cur_k, cur_n = [   
    #                max_row_matrix[cur_j,cur_k,cur_n],
    #                max_char_matrix[cur_j,cur_k,cur_n], 
    #                max_nbest_matrix[cur_j,cur_k,cur_n], 
    #            ]                
    #        cur_k_list=self._generate_list_representation_if_valid(cur_k)

    #    return list(reversed(decoded_output))

    #def _generate_list_representation_if_valid(self,k): 
    #    """
    #    Generate a list representation from index representation of state
    #    List is empty is k is not a valid state. This happens for: 
    #        - k=0
    #        - any k where base(self._esize) representation has '0' sandwiched between other word ids

    #    """

    #    l=[]
    #    null_encountered=False

    #    while k>0: 
    #        # get LSB
    #        d=k%self._esize

    #        # check if valid state: invalid if '0' is sandwiched between other ids
    #        if d==0:
    #            null_encountered=True 
    #        if null_encountered and d!=0:
    #            return []

    #        l.append(d)

    #        k=k/self._esize 

    #    
    #    return filter(lambda x:x!=0,reversed(l))            

    #def _generate_index_representation(self,k_list): 
    #    """
    #    Using Horner's rule 
    #    Input has to be a valid state
    #    """

    #    k=0
    #    for i in k_list[:-1]:
    #        k+=i*self._esize
    #    return k+k_list[-1]

    #def _generate_string_representation(self,k_list): 
    #    return ' '.join( [str(x) for x in k_list] )

    #def _generate_all_states(self): 
    #    """
    #    Generate all valid states 
    #    """
    #    context_size=self._lm_order-1

    #    for k in xrange(int(np.power(self._esize,context_size))):
    #        k_list=self._generate_list_representation_if_valid(k)

    #        if len(k_list)>0:
    #            yield (k,k_list)

    #def _generate_related_history(self,k,k_list): 
    #    context_size=self._lm_order-1

    #    rel_his=[k_list[:-1]]
    #    if len(k_list)==context_size: 
    #        rel_his.extend([ [c]+rel_his[0]  for c in xrange(1,self._esize)])

    #    return ( ( self._generate_index_representation(x), x) for x in rel_his if len(x)>0)

    ##@profile
    #def decode_topn_ngram(self,f_input_word,topn=1):
    #    """
    #        decoding with an ngram language model 
    #    """

    #    context_size=self._lm_order-1

    #    sm_shape=(len(f_input_word), int(np.power(self._esize,context_size)), topn )

    #    # score matrix
    #    ## initializing to 1.0, which denotes an invalid entry. Useful for keeping track of length of n-best list
    #    score_matrix=np.ones(sm_shape) 
    #    # backtracking matrices 
    #    max_row_matrix=np.ones(sm_shape ,dtype=int) * -1
    #    max_char_matrix=np.ones(sm_shape,dtype=int) * -1
    #    max_nbest_matrix=np.ones(sm_shape,dtype=int) * -1 

    #    # initialization
    #    #print 'Initialization'
    #    for k, k_list in self._generate_all_states():
    #        if len(k_list)==1:
    #                                  #    LM score                    translition matrix score
    #            score_matrix[0,k,0]=self._ngram_log_score(0,[],k_list[-1]) + self._get_param_value(  k , f_input_word[0])   
    #    #        print 'Candidates for {}, {}: {}'.format(0,self._generate_string_representation(k_list),score_matrix[0,k,0])

    #    ### modified  viterbi decoding
    #    # j- input position
    #    # k- output history id (current)
    #    # m- output history id (prev)


    #    # Compute the scoring matrix and create the backtracking vector 

    #    for j in xrange(1,sm_shape[0]):
    #        #print 'Input posn: {}'.format(j)
    #        for k, k_list in self._generate_all_states():
    #            #print 'Cur Char: {}'.format(self._generate_string_representation(k_list))

    #            entry_scores=[]

    #            ## Case 1: A new output character is generated by input(j) alone
    #            for m, m_list in self._generate_related_history(k,k_list):
    #                #print 'Prev Char: {}'.format(self._generate_string_representation(m_list))
    #                for n in xrange(sm_shape[2]):  # for the first input candidate, only consider 1 candidate from initialization
    #                    if score_matrix[j-1,m,n]>0.0:
    #                        break
    #                    v=score_matrix[j-1,m,n] + self._ngram_log_score(m,m_list,k_list[-1]) + self._get_param_value(  k_list[-1] , f_input_word[j] )
    #                    entry_scores.append( ( ( j-1, m, n )  , v  ) )


    #            #Case 2: A new output character is generated by input(j) and input(j-1)
    #            if j==1 and len(k_list)==1: 
    #                ## first 2 input characters generate the first output
    #                v= self._ngram_log_score(0,[],k_list[-1]) + self._get_param_value(  k_list[-1] ,  f_input_word[j-1] +  f_input_word[j])  
    #                     
    #                entry_scores.append( ( ( -1, -1, -1 )  , v  ) )
    #            
    #            #print '=='

    #            if j>=2:
    #                # evaluate all (j-2) candidates
    #                for m, m_list in self._generate_related_history(k,k_list):
    #                    #print 'Prev Char: {}'.format(self._generate_string_representation(m_list))
    #                    for n in xrange(sm_shape[2]):
    #                        if score_matrix[j-2,m,n]>0.0:
    #                            break
    #                        v=score_matrix[j-2,m,n] + \
    #                                    self._ngram_log_score(m,m_list,k_list[-1]) + \
    #                                       self._get_param_value( k_list[-1] ,  f_input_word[j-1] +  f_input_word[j]) 

    #                        entry_scores.append( ( ( j-2, m, n )  , v  ) )

    #            top_n_candidates=heapq.nlargest(topn,entry_scores,key=operator.itemgetter(1))
    #            #print 'Candidates for {}, {}:'.format(j,self._generate_string_representation(k_list))
    #            for n in xrange(len(top_n_candidates)): 
    #                score_matrix[j,k,n]=top_n_candidates[n][1]+self._length_control_param
    #                max_row_matrix[j,k,n]=top_n_candidates[n][0][0]
    #                max_char_matrix[j,k,n]=top_n_candidates[n][0][1]
    #                max_nbest_matrix[j,k,n]=top_n_candidates[n][0][2]
    #                #print '{}, {}, {}: {}'.format(top_n_candidates[n][0][0],
    #                #                        self._generate_string_representation(self._generate_list_representation_if_valid(top_n_candidates[n][0][1])),
    #                #                        top_n_candidates[n][0][2],
    #                #                        top_n_candidates[n][1])

    #    # find starting points for backtracking top-n outputs 
    #    def generate_final_candidates():
    #        for k, k_list in self._generate_all_states():
    #            for n in xrange(sm_shape[2]):  # for the first input candidate, only consider 1 candidate from initialization
    #                if score_matrix[sm_shape[0]-1,k,n]>0.0:
    #                    break

    #                yield [sm_shape[0]-1,k,n]

    #    final_top_n_candidates=heapq.nlargest(topn,
    #            generate_final_candidates(),
    #            key=lambda ia: score_matrix[ ia[0], ia[1], ia[2] ] )

    #    nbest_list= [
    #                    (
    #                        self._backtrack_nbest_output_ngram(max_row_matrix,max_char_matrix,max_nbest_matrix,*candidate),
    #                        score_matrix[candidate[0], candidate[1], candidate[2]]
    #                    )    for candidate in final_top_n_candidates 

    #                ]                             

    #    return nbest_list


