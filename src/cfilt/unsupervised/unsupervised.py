import itertools as it
import codecs, sys, pickle
import pprint 
from collections import defaultdict
import numpy as np, math
import pandas as pd
import random

import srilm

from cfilt.utilities import timer


#class SimpleLanguageModel: 
#
#    def __init__(c_id_map, lm_fname, order=2): 
#        with codecs.open(lm_fname,'r','utf-8') as lmfile:
#            codecs.readline()
#            codecs.readline()
#
#            # read up first header line
#            [ codecs.readline() for _ in xrange(2) ]
#                
#            section_lengths=[ int(codecs.readline().strip().split(u'=')[1]) for _ in xrange(order) ]
#
#            for o, section_length in enumerate(section_lengths,1): 
#                for 


def load_lm_model(lm_fname, order=2): 
    """
    function to load language model 
    """
    lm_model=srilm.initLM(order)
    srilm.readLM(lm_model,lm_fname)    
    return lm_model

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
        print 'Decoding word'

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

# Alignment is a list of 'charseq_pairs'
# represented as 'src_seq|tgtseq'
# Associated with every alignment is a weight
# Every word pair generates a list of tuples [alignment,weight]
# The corpus is a list of word pair
# Each alignment point in the corpus can be uniquely identified by (word_pair_idx,align_idx,aln_point_idx)
#
#       corpus =>  [wpairs_aligns,wpairs_weights]
#       wpairs_aligns=> list( alignments )
#       wpairs_weights=> list( weights )
#       alignment => list(charseq_pair)
#       weights => list(float)
#       charseq_pair => 'src_seq|tgt_seq'

###
# NOTES
# nf>ne required for allowed mapping of at least 1 on e side


####
    
ZERO_LOG_PROB_REPLACEMENT=-700.0
MAX_ITER=100
EPSILON=0.005

def log_z(x): 
    return  math.log(x) if x>0.0 else ZERO_LOG_PROB_REPLACEMENT

class UnsupervisedTransliteratorTrainer: 

    ## ALLOWED mappings
    ALLOWED_MAPPINGS=[1,2]
   
    def __init__(self,lm_model):

        self._translit_model=TransliterationModel()

        ####  corpus information
        # Each word-pair is indexed by its position in the input corpus (starting at 0)

        # list of all possible alignments for each word pair 
        self.wpairs_aligns=[]

        # list of weights for all possible alignments for a word pair 
        self.wpairs_weights=[]

        ##### parameter information

        # for each parameter (which corresponds to a point alignment), the list of occurences in of this point alignment all possible (wordpair,alignment) locations in the corpus
        self.param_occurence_info=defaultdict(lambda :defaultdict(list))

        # Parameter values in the previous iteration 
        self.prev_param_values={}

        #### Language model 
        self._lm_model=lm_model

    def print_obj(self): 

        print("xxx Printing EM instance xxx")
        print("Symbol mappings for F: ")
        for f_id in range(len(self._translit_model.f_id_sym_map)): 
            print u'{} {}'.format(f_id,self._translit_model.f_id_sym_map[f_id]).encode('utf-8')

        print("Symbol mappings for E: ")
        for e_id in range(len(self._translit_model.e_id_sym_map)): 
            print u'{} {}'.format(e_id,self._translit_model.e_id_sym_map[e_id]).encode('utf-8')

        #print("Param Occurence Info: ")
        #for eid in xrange(len(self._translit_model.e_sym_id_map)): 
        #    for fid in xrange(len(self._translit_model.f_sym_id_map)): 
        #        l=self.param_occurence_info[eid][fid]
        #        if len(l)>0:
        #            print 'eid={} fid={}'.format(eid,fid)
        #            pprint.pprint(l)

        #print("Alignments: ")
        ## gather transliteration occurrence info
        #for wp_idx,alignments in enumerate(self.wpairs_aligns): 
        #    for aln_idx,align in enumerate(alignments): 
        #        print 'wp_idx={} aln_idx={}'.format(wp_idx,aln_idx)
        #        pprint.pprint(align)
        #        #for charseq_pair in align: 
        #        #    print[charseq_pair[0]][charseq_pair[1]].append([wp_idx,aln_idx])

    def print_params(self): 
        #print("Alignment Weights: ")
        #pprint.pprint(self.wpairs_weights)
       
        print("Transliteration Probabilities")
        #for e_id in range(len(self._translit_model.e_sym_id_map)): 
        #    for f_id in range(len(self._translit_model.f_sym_id_map)): 
        #        #print u"P({}|{})={}".format(self._translit_model.f_id_sym_map[f_id],self._translit_model.e_id_sym_map[e_id],self._translit_model.param_values[e_id,f_id]).encode('utf-8') 
        #        print u"P({}|{})={}".format(f_id,e_id,self._translit_model.param_values[e_id,f_id]).encode('utf-8') 
        #pprint.pprint(self._translit_model.param_values)
        

    def _generate_alignments(self,f_word,e_word):
        """
            src_word: list of src characters
            tgt_word: list of tgt characters 
    
            return: wpair_info for this pair
        """
        nf=len(f_word)
        ne=len(e_word)
    
        alignment_preseqs=list(it.ifilter( lambda x: sum(x)==nf, 
                                it.product(UnsupervisedTransliteratorTrainer.ALLOWED_MAPPINGS,repeat=ne)
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
                    self._translit_model.f_sym_id_map[fs]=fs_id

                # create 'e' sym to id mapping
                # TODO: check if this works correctly in the unsupervised case
                es_id=  self._translit_model.e_sym_id_map[es]  if   self._translit_model.e_sym_id_map.has_key(es) else len(self._translit_model.e_sym_id_map)
                if es_id==len(self._translit_model.e_sym_id_map):
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

        self.wpairs_aligns=[]
        self.wpairs_weights=[]
    
        for f,e in word_pairs: 
            alignments=self._generate_alignments(f,e)
            if len(alignments)>0:
                self.wpairs_aligns.append(alignments)
                self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )
            else: 
                print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 

        # create inverse id-symbol mappings
        # for f
        for s,i in self._translit_model.f_sym_id_map.iteritems():
            self._translit_model.f_id_sym_map[i]=s
    
        # for e
        for s,i in self._translit_model.e_sym_id_map.iteritems():
            self._translit_model.e_id_sym_map[i]=s
    
    def _initialize_parameter_structures(self): 
        """
    
        """
        # gather transliteration occurrence info
        self.param_occurence_info=defaultdict(lambda :defaultdict(list))
        for wp_idx,alignments in enumerate(self.wpairs_aligns): 
            for aln_idx,align in enumerate(alignments): 
                for charseq_pair in align: 
                    self.param_occurence_info[charseq_pair[0]][charseq_pair[1]].append([wp_idx,aln_idx])
    
        # initialize transliteration probabilities 
        self._translit_model.param_values=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])
        self.prev_param_values=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])
    
    def _m_step(self): 
        
        # accumulating counts 
        for e_id in range(len(self._translit_model.e_sym_id_map)): 
            for f_id in range(len(self._translit_model.f_sym_id_map)): 
                self.prev_param_values[e_id,f_id]=self._translit_model.param_values[e_id,f_id]
                self._translit_model.param_values[e_id,f_id] = float(sum( [self.wpairs_weights[w_id][a_id] for w_id, a_id in  self.param_occurence_info[e_id][f_id] ] ))
       
        # normalizing
        for e_id in range(len(self._translit_model.e_sym_id_map)): 
            norm_factor=np.sum(self._translit_model.param_values[e_id,:])
            self._translit_model.param_values[e_id,:]=self._translit_model.param_values[e_id,:]/norm_factor if norm_factor>0.0 else 0.0
   
    def _e_step(self): 
        for wp_idx in xrange(len(self.wpairs_aligns)): 
            aln_probs=[]
            for aln_idx in xrange(len(self.wpairs_aligns[wp_idx])): 
                try: 
                    aln_probs.append(
                            math.exp(
                                    # TODO: what to do about negative log probability
                                    sum([ math.log(self._translit_model.param_values[x[0],x[1]]) if self._translit_model.param_values[x[0],x[1]]!=0.0  else 0.0 for x in self.wpairs_aligns[wp_idx][aln_idx] ])
                            )
                        )
                except ValueError as e: 
                    print "Exception"
                    print e.message
           
            norm_factor=sum(aln_probs)
            for aln_idx in xrange(len(self.wpairs_aligns[wp_idx])): 
                self.wpairs_weights[wp_idx][aln_idx]=aln_probs[aln_idx]/norm_factor


    def _has_converged(self):
        """
            check if parameter values between successive iterations are within a threshold 
        """
        converged=True

        for e_id in range(len(self._translit_model.e_sym_id_map)): 
            for f_id in range(len(self._translit_model.f_sym_id_map)): 
                if math.fabs(self._translit_model.param_values[e_id,f_id]-self.prev_param_values[e_id,f_id]) >= EPSILON:
                    converged=False
                    break;

        return converged

    def em_supervised_train(self,word_pairs): 
        """
        """
    
        self._create_alignment_database(word_pairs)
        self._initialize_parameter_structures()
    
        MAX_ITER=100
        niter=0
        while(True):
            # M-step
            self._m_step()
            niter+=1

            #self.print_params()
            print '=== Iteration {} completed ==='.format(niter)

            # check for end of process 
            if niter>=MAX_ITER or self._has_converged():
                break
    
            # E-step 
            self._e_step()

        #self.print_obj()
        #self.print_params()
        #print "Final parameters"
        #for e_id in range(len(self._translit_model.e_sym_id_map)): 
        #    for f_id in range(len(self._translit_model.f_sym_id_map)): 
        #        #print u"P({}|{})={}".format(self._translit_model.f_id_sym_map[f_id],self._translit_model.e_id_sym_map[e_id],self._translit_model.param_values[e_id,f_id]).encode('utf-8') 
        #        print u"P({}|{})={}".format(f_id,e_id,self._translit_model.param_values[e_id,f_id]).encode('utf-8') 

    ##########################
    ### Decoding function #####
    ##########################

    ############################################
    ### Unsupervised training functions ########
    ############################################

    def _smooth_alignment_parameters(self):
        pass 

    def _initialize_unsupervised_training(self,f_input_words, e_char_set): 

        # create symbol to id mappings for f (from data)
        for f_input_word in f_input_words: 
            for c in f_input_word: 
                if not self._translit_model.f_sym_id_map.has_key(c): 
                    self._translit_model.f_sym_id_map[c]=len(self._translit_model.f_sym_id_map)

        # create symbol to id mappings for e (from given list of characters)
        for i, c in enumerate(e_char_set):
            self._translit_model.e_sym_id_map[c]=i

        # create inverse id-symbol mappings
        # for f
        for s,i in self._translit_model.f_sym_id_map.iteritems():
            self._translit_model.f_id_sym_map[i]=s
    
        # for e
        for s,i in self._translit_model.e_sym_id_map.iteritems():
            self._translit_model.e_id_sym_map[i]=s
    
        # initialize transliteration probabilities 
        #self._translit_model.param_values=np.ones([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)]) * 1.0/len(self._translit_model.f_sym_id_map)
        self._translit_model.param_values=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])
        for i in xrange(self._translit_model.param_values.shape[0]):
            t=1.0
            for j in xrange(self._translit_model.param_values.shape[1]-1):
                v=random.random()*t
                self._translit_model.param_values[i,j]=v
                t=t-v
            self._translit_model.param_values[i,-1]=t                
            
        self.prev_param_values=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])

    def _prepare_corpus_unsupervised(self,word_pairs): 
        """
          symbol mappings have already been created using '_initialize_unsupervised_training' 

          every character sequence in the corpus can be uniquely addressed as: 
          corpus[0][word_pair_idx][align_idx][aln_point_idx]
    
          every weight can be indexed as: 
          corpus[1][word_pair_idx][align_idx]
    
        """
        self.wpairs_aligns=[]
        self.wpairs_weights=[]
    
        for f,e in word_pairs: 
            alignments=self._generate_alignments(f,e)
            if len(alignments)>0:
                self.wpairs_aligns.append(alignments)
                self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )
            else: 
                print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 

        self.param_occurence_info=defaultdict(lambda :defaultdict(list))
   
        # gather transliteration occurrence info
        for wp_idx,alignments in enumerate(self.wpairs_aligns): 
            for aln_idx,align in enumerate(alignments): 
                for charseq_pair in align: 
                    self.param_occurence_info[charseq_pair[0]][charseq_pair[1]].append([wp_idx,aln_idx])
    
    def em_unsupervised_train(self,f_input_words,e_char_set): 
        """
        """
        print "Initializing unsupervised learning" 
        self._initialize_unsupervised_training(f_input_words,e_char_set)

        niter=0
        while(True):
            ##### M-step #####
            #print "Intermediate parameters"
            #for e_id in range(len(self._translit_model.e_sym_id_map)): 
            #    for f_id in range(len(self._translit_model.f_sym_id_map)): 
            #        print u"P({}|{})={}".format(self._translit_model.f_id_sym_map[f_id],self._translit_model.e_id_sym_map[e_id],self._translit_model.param_values[e_id,f_id]).encode('utf-8') 

            # decode: approximate marginalization over all e strings by maximization
            print "Decoding for EM"
            decoder=TransliterationDecoder(self._translit_model,self._lm_model)
            word_pairs=list(it.izip( f_input_words, it.imap( decoder._decode_internal,f_input_words)  ))

            ## the best candidates after decoding    
            #for x,y in word_pairs: 
            #    print u'{}: {}: {}'.format(''.join(x),' '.join(y),len(y)).encode('utf-8')

            # initialize the EM training
            print "Preparing corpus"
            self._prepare_corpus_unsupervised(word_pairs)

            # estimate alignment parameters
            print "Estimating parameters"
            self._m_step()

            # smooth alignment parameters 
            self._smooth_alignment_parameters()
            
            niter+=1
            #self.print_params()
            print '=== Iteration {} completed ==='.format(niter)

            # check for end of process 
            if niter>=MAX_ITER or self._has_converged():
                break
    
            ######  E-step ####
            print "Computing alignment weights" 
            self._e_step()

        self.print_obj()
        print "Final parameters"
        for e_id in range(len(self._translit_model.e_sym_id_map)): 
            for f_id in range(len(self._translit_model.f_sym_id_map)): 
                print u"P({}|{})={}".format(self._translit_model.f_id_sym_map[f_id],self._translit_model.e_id_sym_map[e_id],self._translit_model.param_values[e_id,f_id]).encode('utf-8') 

def read_parallel_corpus(fcorpus_fname,ecorpus_fname): 
    with codecs.open(fcorpus_fname,'r','utf-8') as ffile:
        with codecs.open(ecorpus_fname,'r','utf-8') as efile:
            return [ ( f.strip().split() , e.strip().split()  )  for f,e in it.izip( iter(ffile)  , iter(efile)  ) ] 

def read_monolingual_corpus(corpus_fname): 
    with codecs.open(corpus_fname,'r','utf-8') as infile:
            return [ w.strip().split()  for w in infile ] 

def generate_char_set(fname):
    char_set=set()
    for input_word in read_monolingual_corpus(fname): 
        char_set.update(input_word)
    return list(char_set)

if __name__=='__main__': 

    
    #####  model information
    ##  F: source  E: target

    ## file listing set of characters in the source
    fcfname='kannada/En-Ka-News_EnglishLabels_KannadaRows_EnglishColumns_linear_'
    ## file listing set of characters in the target
    ecfname='kannada/En-Ka-News_KannadaLabels_KannadaRows_EnglishColumns_linear_'
    ## file listing alignment from source to target
    ##  target is along the rows, source is along the columns
    alignmentfname='kannada/En-Ka-News_CrossEntropy_AlignmentMatrix_KannadaRows_EnglishColumns_linear_'
    ### bigram- target language model in APRA  model. Note: decoding currently supports only bigram models
    lm_fname='kannada/Ka-2g.lm'
    
    ##### testset information 
    test_fcorpus_fname='kannada/test.En'
    test_ecorpus_fname='kannada/test.Ka'

    tm_model=TransliterationModel.construct_transliteration_model(fcfname,ecfname,alignmentfname)
    lm_model=load_lm_model(lm_fname)

    decoder=TransliterationDecoder(tm_model,lm_model)
    decoder.evaluate(read_parallel_corpus(test_fcorpus_fname,test_ecorpus_fname))

    ##fcorpus_fname=sys.argv[1]
    ##ecorpus_fname=sys.argv[2]
    ##model_dir=sys.argv[3]
    ##lm_fname=sys.argv[4]
    ##test_fcorpus_fname=sys.argv[5]
    ##test_ecorpus_fname=sys.argv[6]

    ###########  Supervised training
    #data_dir='/home/development/anoop/experiments/unsupervised_transliterator/data'
    #parallel_dir=data_dir+'/'+'en-hi'

    #fcorpus_fname=parallel_dir+'/'+'train.en'
    #ecorpus_fname=parallel_dir+'/'+'train.hi'
    #lm_fname=data_dir+'/'+'hi-2g.lm'
    ##test_fcorpus_fname=parallel_dir+'/'+'test.en'
    ##test_ecorpus_fname=parallel_dir+'/'+'test.hi'
    #test_fcorpus_fname='test.en'
    #test_ecorpus_fname='test.hi'

    #lm_model=load_lm_model(lm_fname)

    ##em=UnsupervisedTransliteratorTrainer(lm_model)
    ##em.em_supervised_train(read_parallel_corpus(fcorpus_fname,ecorpus_fname))
    ##TransliterationModel.save_translit_model(em._translit_model,'translit.model')

    #
    ##decoder=TransliterationDecoder(em._translit_model,em._lm_model)
    #decoder=TransliterationDecoder(TransliterationModel.load_translit_model('translit.model'),load_lm_model(lm_fname))
    #with timer.Timer(True) as t: 
    #    decoder.evaluate(read_parallel_corpus(test_fcorpus_fname,test_ecorpus_fname))
    #print 'Time for decoding: '.format(t.secs)
        

    #########  Unsupervised training
    #data_dir='/home/development/anoop/experiments/unsupervised_transliterator/data'
    ##parallel_dir=data_dir+'/'+'en-hi'

    ##fcorpus_fname=parallel_dir+'/'+'train.en'
    ##ecorpus_fname=parallel_dir+'/'+'train.hi'
    ##test_fcorpus_fname=parallel_dir+'/'+'test.en'
    ##test_ecorpus_fname=parallel_dir+'/'+'test.hi'

    #fcorpus_fname='10.en'
    #ecorpus_fname='10.hi'
    #test_fcorpus_fname='10.en'
    #test_ecorpus_fname='10.hi'

    #lm_fname=data_dir+'/'+'hi-2g.lm'

    #lm_model=load_lm_model(lm_fname)

    #em=UnsupervisedTransliteratorTrainer(lm_model)
    #em.em_unsupervised_train(read_monolingual_corpus(fcorpus_fname),generate_char_set(ecorpus_fname))

    #decoder=TransliterationDecoder(em._translit_model,em._lm_model)
    #decoder.evaluate(read_parallel_corpus(test_fcorpus_fname,test_ecorpus_fname))


