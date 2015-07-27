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

    def __init__(self):

        self.f_sym_id_map={}
        self.e_sym_id_map={}

        self.f_id_sym_map={}
        self.e_id_sym_map={}

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

    def __init__(self,translit_model, lm_model, order=2): 
        self._lm_model=lm_model
        self._lm_order=order
        self._translit_model=translit_model

        e_vocabsize=len(self._translit_model.e_id_sym_map)
        self._lm_cache=np.ones(  ( int(np.power(e_vocabsize,self._lm_order-1)), e_vocabsize )  )*-1.0

    def _bigram_score(self,hist_id,cur_id):
        """
        """
        if self._lm_cache[hist_id,cur_id]==-1:
            bigram=u'{} {}'.format( self._translit_model.e_id_sym_map[hist_id] if hist_id>=0 else u'<s>',
                                    self._translit_model.e_id_sym_map[cur_id])
            self._lm_cache[hist_id,cur_id]=math.pow( 10 , srilm.getBigramProb(self._lm_model,bigram.encode('utf-8')) )

        return self._lm_cache[hist_id,cur_id]

    def _word_log_score(self,word):
        return srilm.getSentenceProb(self._lm_model,u' '.join(word).encode('utf-8'),len(word))

    def _get_param_value( self, e_id, f_input_chars, stretch=3.0 ): 
        if self._translit_model.f_sym_id_map.has_key(f_input_chars): 
            return np.power(self._translit_model.param_values[  e_id , self._translit_model.f_sym_id_map[f_input_chars] ],stretch)
        else:
            return 0.0

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
        for k in xrange(sm_shape[1]):
            #print u'{} {}'.format( self._translit_model.e_id_sym_map[k] ,self._bigram_score(-1,k)).encode('utf-8')
            #print u'{} {} {}'.format( self._translit_model.e_id_sym_map[k] , f_input_word[0], self._get_param_value(  k , f_input_word[0])).encode('utf-8')

            # for first input character generating first output character
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
                
                if j==1:
                    ## first 2 input characters generate the first output
                    max_val_2=sum ( map ( log_z ,
                               # LM score                    transliteration matrix score
                               [ self._bigram_score(-1,k) , self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j]) ] 
                         ) 
                       )  

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
        for k in xrange(sm_shape[1]):
            #print 'Candidates for {} {}:'.format(0,k)
            for n in xrange(sm_shape[2]):
                #print u'{} {}'.format( self._translit_model.e_id_sym_map[k] ,self._bigram_score(-1,k)).encode('utf-8')
                #print u'{} {} {}'.format( self._translit_model.e_id_sym_map[k] , f_input_word[0], self._get_param_value(  k , f_input_word[0])).encode('utf-8')
                max_row_matrix[0,k,n]=-1                       
                max_char_matrix[0,k,n]=-1                       
                max_nbest_matrix[0,k,n]=-1
                score_matrix[0,k,n]= sum ( map ( log_z  ,
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
                ##print 'Char: {}'.format(k)

                entry_scores=[]

                ## Case 1: A new output character is generated by input(j) alone
                for m in xrange(np.shape(score_matrix)[1]):
                    for n in xrange(  np.shape(score_matrix)[2]  if j>1 else 1  ):  # for the first input candidate, only consider 1 candidate from initialization
                    #for n in xrange(  np.shape(score_matrix)[2]):
                        v=score_matrix[j-1,m,n] + sum ( map ( log_z ,
                                   # LM score                    transliteration matrix score
                                   [ self._bigram_score(m,k) , self._get_param_value(  k , f_input_word[j] )  ] 
                             ) 
                           )  
                        entry_scores.append( ( ( j-1, m, n )  , v  ) )


                # Case 2: A new output character is generated by input(j) and input(j-1)
                if j==1: 
                    ## first 2 input characters generate the first output
                    v=sum ( map ( log_z ,
                               # LM score                    transliteration matrix score
                               [ self._bigram_score(-1,k) , self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j]) ] 
                         ) 
                       )  
                    entry_scores.append( ( ( -1, -1, -1 )  , v  ) )

                if j>=2:
                    # evaluate all (j-2) candidates
                    for m in xrange(np.shape(score_matrix)[1]):
                        for n in xrange(np.shape(score_matrix)[2] if j>2 else 1):
                        #for n in xrange(np.shape(score_matrix)[2]):
                            v=score_matrix[j-2,m,n] +  sum ( map ( log_z ,
                                       # LM score                    transliteration matrix score
                                       [ self._bigram_score(m,k) , self._get_param_value(  k ,  f_input_word[j-1] +  f_input_word[j]) ] 
                                 ) 
                               )  
                            entry_scores.append( ( ( j-2, m, n )  , v  ) )

                top_n_candidates=heapq.nlargest(topn,entry_scores,key=operator.itemgetter(1))
                #print 'Candidates for {} {}:'.format(j,k)
                for n in xrange(len(top_n_candidates)): 
                    score_matrix[j,k,n]=top_n_candidates[n][1]
                    max_row_matrix[j,k,n]=top_n_candidates[n][0][0]
                    max_char_matrix[j,k,n]=top_n_candidates[n][0][1]
                    max_nbest_matrix[j,k,n]=top_n_candidates[n][0][2]
                    #print '({} {} {} {})'.format(score_matrix[j,k,n],max_row_matrix[j,k,n],max_char_matrix[j,k,n],max_nbest_matrix[j,k,n])

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

    #def _generate_ngram(self,hist_id,cur_id,hist_len): 
    #    """
    #    Generates ngram corresponding to provided history/context and the current symbol 

    #    hist_id: the id of the historical state. For a state-space with k symbols and history window of n, this is number between 0 and k^^n. 
    #             This can be interpreted as an n-digit number in base-k notation. This interpretation is used in the computations that follow. 
    #             hist_id=-1 is invalid/no history. Equivalent condition: hist_len=0

    #    cur_id: id of the current symbol. integer between 0 and k(exclusive)

    #    hist_len: length of the history indicated by hist_id. This information is required to identify histories less than length n. 
    #              For such candidates, only the 'hist_len' LSB radix-k digits are used to generate history/context 

    #    """
    #    e_size=len(self._translit_model.e_sym_id_map.keys())
    #    context_size=self._lm_order-1

    #    # initialize with current char id
    #    context_v=[self._translit_model.e_id_sym_map[cur_id]]

    #    # identify all characters from history
    #    if hist_len>0:
    #        v=hist_id
    #        l=0
    #        while v>0 or l<hist_len: 
    #            context_v.append(self._translit_model.e_id_sym_map[v%e_size])
    #            v=v/e_size
    #            l=l+1

    #    # for context less than ngram_order-1, choose only available context and add start of sentence marker
    #    if hist_len<context_size: 
    #        #context_v=context_v[:hist_len+1]
    #        context_v.append(u'<s>')

    #    return u' '.join( reversed(context_v) )

    #def _ngram_score(self,hist_id,cur_id,hist_len):
    #    """
    #    Generate n-gram probability score

    #    hist_id: the id of the historical state. For a state-space with k symbols and history window of n, this is number between 0 and k^^n. 
    #             This can be interpreted as an n-digit number in base-k notation. This interpretation is used in the computations that follow. 
    #             hist_id=-1 is invalid/no history. Equivalent condition: hist_len=0

    #    cur_id: id of the current symbol. integer between 0 and k(exclusive)

    #    hist_len: length of the history indicated by hist_id. This information is required to identify histories less than length n. 
    #              For such candidates, only the 'hist_len' LSB radix-k digits are used to generate history/context 
    #    """
    #    if self._lm_cache[hist_id,cur_id]==-1:
    #        ngram=_generate_ngram(hist_id,cur_id)
    #        self._lm_cache[hist_id,cur_id]=math.pow( 10 , srilm.getNgramProb(self._lm_model,ngram.encode('utf-8'),self._lm_order) )
    #   ##                                                                                                              ??           

    #    return self._lm_cache[hist_id,cur_id]

    #def _backtrack_nbest_output_ngram(self,max_row_matrix,max_char_matrix,max_nbest_matrix,sol_length_matrix,j,k,n,sl): 
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
    #    sol_length_matrix: 

    #    """
    #    e_size=len(self._translit_model.e_sym_id_map.keys())
    #    context_size=self._lm_order-1

    #    decoded_output=[]

    #    #### Initialize
    #    # Extract the last (order-2) or sl characters, whichever is less
    #    cur_j, cur_k, cur_n, cur_len = j, k, n, sl

    #    v=cur_k 
    #    l=cur_len
    #    while v>e_size and l>0: 
    #        ## first condition ensures that radix-(order-1) MSB is not extracted

    #        decoded_output.append(self._translit_model.e_id_sym_map[v%e_size])
    #        v=v/e_size
    #        l-=1


    #    ### Extract the remaining characters by extracting MSB from each of the states in the backtracking path
    #    while cur_len>=context_size:
    #        #print u'{} {} {} {}'.format(cur_j,cur_k, cur_n, self._translit_model.e_id_sym_map[cur_k]).encode('utf-8')
    #        ch=cur_k/int(np.pow(e_size,context_size-1))
    #        decoded_output.append( self._translit_model.e_id_sym_map[ch] )

    #        cur_j, cur_k, cur_n, cur_len = [   
    #                max_row_matrix[cur_j,cur_k,cur_n],
    #                max_char_matrix[cur_j,cur_k,cur_n], 
    #                max_nbest_matrix[cur_j,cur_k,cur_n], 
    #                sol_length_matrix[cur_j,cur_k,cur_n],
    #            ]                

    #    return list(reversed(decoded_output))

    #def decode_topn_ngram(self,f_input_word,topn=1):
    #    """
    #        decoding with an ngram language model 
    #    """

    #    e_size=len(self._translit_model.e_sym_id_map.keys())
    #    context_size=self._lm_order-1
    #    sm_shape=(len(f_input_word), int(np.pow(e_size,context_size)), topn )

    #    # score matrix
    #    score_matrix=np.zeros( sm_shape ) 
    #    # backtracking matrices 
    #    max_row_matrix=np.zeros( sm_shape ,dtype=int ) 
    #    max_char_matrix=np.zeros( sm_shape,dtype=int ) 
    #    max_nbest_matrix=np.zeros( sm_shape,dtype=int ) 
    #    # matrix to keep track of output sequence length. used to keep track of what history to use
    #    sol_length_matrix=np.zeros( sm_shape,dtype=int ) 

    #    # initialization
    #    for k in xrange(sm_shape[1]):
    #        #print 'Candidates for {} {}:'.format(0,k)
    #        for n in xrange(sm_shape[2]):
    #            #print u'{} {}'.format( self._translit_model.e_id_sym_map[k] ,self._bigram_score(-1,k)).encode('utf-8')
    #            #print u'{} {} {}'.format( self._translit_model.e_id_sym_map[k] , f_input_word[0], self._get_param_value(  k , f_input_word[0])).encode('utf-8')
    #            max_row_matrix[0,k,n]=-1                       
    #            max_char_matrix[0,k,n]=-1                       
    #            max_nbest_matrix[0,k,n]=-1
    #            sol_length_matrix[0,k,n]=1
    #            score_matrix[0,k,n]= sum ( map ( log_z  ,
    #                                  #    LM score                    translition matrix score
    #                               [ self._ngram_score(-1,k,0) , self._get_param_value(  k , f_input_word[0]) ] 
    #                         ) 
    #                       )

    #    ### modified  viterbi decoding
    #    # j- input position
    #    # k- output history id (current)
    #    # m- output history id (prev)


    #    # Compute the scoring matrix and create the backtracking vector 

    #    for j in xrange(1,sm_shape[0]):
    #        #print 'Row: {}'.format(j)
    #        for k in xrange(sm_shape[1]):
    #            ##print 'Char: {}'.format(k)

    #            entry_scores=[]
    #            context_first=k/e_size
    #            cur_char_id=k%e_size

    #            ## Case 1: A new output character is generated by input(j) alone
    #            for m in xrange(context_first,sm_shape[1],int(sm_shape[1]/e_size)):
    #                for n in xrange(  sm_shape[2]  if j>1 else 1  ):  # for the first input candidate, only consider 1 candidate from initialization
    #                    v=score_matrix[j-1,m,n] + sum ( map ( log_z ,
    #                               # LM score                    transliteration matrix score
    #                               [ self._ngram_score(m,k,sol_length_matrix[j-1,m,n]) , self._get_param_value(  cur_char_id , f_input_word[j] )  ] 
    #                         ) 
    #                       )  
    #                    entry_scores.append( ( ( j-1, m, n )  , v  ) )


    #            # Case 2: A new output character is generated by input(j) and input(j-1)
    #            if j>=2:
    #                # evaluate all (j-2) candidates
    #                for m in xrange(context_first,sm_shape[1],sm_shape[1]/e_size):
    #                    for n in xrange(sm_shape[2] if j>2 else 1):
    #                        v=score_matrix[j-2,m,n] +  sum ( map ( log_z ,
    #                                   # LM score                    transliteration matrix score
    #                                   [ self._ngram_score(m,k,sol_length_matrix[j-2,m,n]) , self._get_param_value(  cur_char_id ,  f_input_word[j-1] +  f_input_word[j]) ] 
    #                             ) 
    #                           )  

    #                        entry_scores.append( ( ( j-2, m, n )  , v  ) )

    #            top_n_candidates=heapq.nlargest(topn,entry_scores,key=operator.itemgetter(1))
    #            #print 'Candidates for {} {}:'.format(j,k)
    #            for n in xrange(len(top_n_candidates)): 
    #                score_matrix[j,k,n]=top_n_candidates[n][1]
    #                max_row_matrix[j,k,n]=top_n_candidates[n][0][0]
    #                max_char_matrix[j,k,n]=top_n_candidates[n][0][1]
    #                max_nbest_matrix[j,k,n]=top_n_candidates[n][0][2]
    #                sol_length_matrix[j,k,n]=sol_length_matrix[ max_row_matrix[j,k,n], max_char_matrix[j,k,n], max_nbest_matrix[j,k,n] ] + 1
    #                #print '({} {} {} {})'.format(score_matrix[j,k,n],max_row_matrix[j,k,n],max_char_matrix[j,k,n],max_nbest_matrix[j,k,n])

    #    # find starting points for backtracking top-n outputs 
    #    final_top_n_candidates=heapq.nlargest(topn,
    #            it.product([sm_shape[0]-1],xrange(sm_shape[1]),xrange(sm_shape[2])),
    #            key=lambda ia: score_matrix[ ia[0], ia[1], ia[2] ] )

    #    nbest_list= [
    #                    (
    #                        self._backtrack_ngram_nbest_output(max_row_matrix,max_char_matrix,max_nbest_matrix,*candidate),
    #                        score_matrix[candidate[0], candidate[1], candidate[2]]
    #                    )    for candidate in final_top_n_candidates 

    #                ]                             

    #    return nbest_list




    ### Compute likelihood for 
    def _generate_alignments(self,f_word,e_word):
        """
            src_word: list of src characters
            tgt_word: list of tgt characters 
    
            return: wpair_info for this pair
        """
        nf=len(f_word)
        ne=len(e_word)
        allowed_mappings=[1,2]
    
        alignment_preseqs=list(it.ifilter( lambda x: sum(x)==nf, 
                                it.product(allowed_mappings ,repeat=ne)
                          ))
    
        wpairs_aligns=[]
    
        for ap in alignment_preseqs:
    
            # compute the end offsets on 'f' side for each charseq_pair by cumulative addition 
            align_offsets=list(ap)
            for i in range(1,len(align_offsets)):
                align_offsets[i]=align_offsets[i]+align_offsets[i-1]

            # insert the first starting offset             
            align_offsets.insert(0,0)            
    
            # extract the charseq_pair
            align=[]
            for i in range(len(align_offsets)-1): 
                #charseq_pair=u''.join( f_word[align_offsets[i]:align_offsets[i+1]] ) + u'|' + e_word[i]
                fs=u''.join( f_word[align_offsets[i]:align_offsets[i+1]] )
                es=e_word[i]

                # create 'f' sym to id mapping
                # TODO: check if this works correctly in the unsupervised case
                fs_id=  self._translit_model.f_sym_id_map[fs]  if   self._translit_model.f_sym_id_map.has_key(fs) else len(self._translit_model.f_sym_id_map)
                if fs_id==len(self._translit_model.f_sym_id_map):
                    print 'Warning: unknown character'
                    self._translit_model.f_sym_id_map[fs]=fs_id

                # create 'e' sym to id mapping
                # TODO: check if this works correctly in the unsupervised case
                es_id=  self._translit_model.e_sym_id_map[es]  if   self._translit_model.e_sym_id_map.has_key(es) else len(self._translit_model.e_sym_id_map)
                if es_id==len(self._translit_model.e_sym_id_map):
                    print 'Warning: unknown character'
                    self._translit_model.e_sym_id_map[es]=es_id

                align.append( [es_id,fs_id] )
           
            wpairs_aligns.append(align)
    
        return wpairs_aligns
    
    
    def _create_alignment_database(self,word_pairs):
        """
          every character sequence in the corpus can be uniquely addressed as: 
            corpus[0][word_pair_idx][align_idx][aln_point_idx]
    
          every weight can be indexed as: 
            corpus[1][word_pair_idx][align_idx]
    
        """
        #TODO: merge _create_alignment_database and _initialize_parameter_structures  into a single function _prepare_corpus_supervised

        wpairs_aligns=[]
        wpairs_weights=[]
        wpairs_eword_weights=[]
    
        for f,e in word_pairs: 
            alignments=self._generate_alignments(f,e)
            if len(alignments)>0:
                wpairs_aligns.append(alignments)
                wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )
                wpairs_eword_weights.append(1.0)
            else: 
                #msg=u"No alignments from word pair: {} {}".format(u''.join(f),u''.join(e))
                #msg=u"No alignments from word pair: " + u''.join(f) + u" " + u''.join(e)
                #print msg.encode('utf-8') 
                pass 

        ## create inverse id-symbol mappings
        ## for f
        #for s,i in self._translit_model.f_sym_id_map.iteritems():
        #    self._translit_model.f_id_sym_map[i]=s
    
        ## for e
        #for s,i in self._translit_model.e_sym_id_map.iteritems():
        #    self._translit_model.e_id_sym_map[i]=s

        return it.izip(wpairs_aligns, wpairs_weights, wpairs_eword_weights)

    def compute_log_likelihood_unsupervised(self, word_pairs, wpairs_aligns=None, wpairs_weights=None,wpairs_eword_weights=None): 

        alignment_info=None

        if wpairs_aligns is None: 
            alignment_info=self._create_alignment_database(word_pairs)
        else:
            alignment_info=it.izip(wpairs_aligns,wpairs_weights,wpairs_eword_weights)

        ll=0.0
        i=0
        for (wpair_aligns, wpair_weights, wpair_eword_weight), (f, e) in it.izip(alignment_info, word_pairs): 
            wll=0.0
            for wpair_align, wpair_weight in it.izip(wpair_aligns,wpair_weights): 
                for e_id, f_id in wpair_align: 
                    wll+=log_z(self._translit_model.param_values[e_id, f_id]) 
                wll*=wpair_weight                    
            wll+=self._word_log_score(e)
            ll+=wll

            #print '========= {} {} ==============='.format(i,len(wpair_aligns)) 
            i=i+1

        return ll             

    def compute_log_likelihood_supervised(self, wpairs_aligns, wpairs_weights): 

        ll=0.0
        i=0
        for wpair_aligns, wpair_weights in it.izip(wpairs_aligns, wpairs_weights): 
            wll=0.0
            for wpair_align, wpair_weight in it.izip(wpair_aligns,wpair_weights): 
                for e_id, f_id in wpair_align: 
                    wll+=log_z(self._translit_model.param_values[e_id, f_id]) 
                wll*=wpair_weight                    
            ll+=wll

            #print '========= {} {} ==============='.format(i,len(wpair_aligns)) 
            i=i+1

        return ll             

