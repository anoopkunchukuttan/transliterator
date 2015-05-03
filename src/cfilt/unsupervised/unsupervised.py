import itertools as it
import codecs, sys
import pprint 
from collections import defaultdict
import numpy as np, math
import pandas as pd
import random

import srilm

class TransliterationModel:
    def __init__(self):

        self.f_sym_id_map={}
        self.e_sym_id_map={}

        self.f_id_sym_map={}
        self.e_id_sym_map={}

        # Dictionary of parameter values (point alignment) to its probability
        self.param_values={}


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
   
    def _init_lm(self,lm_fname): 
        self.lm_model=srilm.initLM(2)
        srilm.readLM(self.lm_model,lm_fname)    

    def __init__(self,lm_fname):

        self._model=TransliterationModel()

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

        ####  Decoder members 
        # bigram language model 
        self.lm_model=None
        self._init_lm(lm_fname)

    def print_obj(self): 

        print("xxx Printing EM instance xxx")
        print("Symbol mappings for F: ")
        for f_id in range(len(self._model.f_id_sym_map)): 
            print u'{} {}'.format(f_id,self._model.f_id_sym_map[f_id]).encode('utf-8')

        print("Symbol mappings for E: ")
        for e_id in range(len(self._model.e_id_sym_map)): 
            print u'{} {}'.format(e_id,self._model.e_id_sym_map[e_id]).encode('utf-8')

        #print("Param Occurence Info: ")
        #for eid in xrange(len(self._model.e_sym_id_map)): 
        #    for fid in xrange(len(self._model.f_sym_id_map)): 
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
        #for e_id in range(len(self._model.e_sym_id_map)): 
        #    for f_id in range(len(self._model.f_sym_id_map)): 
        #        #print u"P({}|{})={}".format(self._model.f_id_sym_map[f_id],self._model.e_id_sym_map[e_id],self._model.param_values[e_id,f_id]).encode('utf-8') 
        #        print u"P({}|{})={}".format(f_id,e_id,self._model.param_values[e_id,f_id]).encode('utf-8') 
        #pprint.pprint(self._model.param_values)
        

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
                fs_id=  self._model.f_sym_id_map[fs]  if   self._model.f_sym_id_map.has_key(fs) else len(self._model.f_sym_id_map)
                if fs_id==len(self._model.f_sym_id_map):
                    self._model.f_sym_id_map[fs]=fs_id

                # create 'e' sym to id mapping
                # TODO: check if this works correctly in the unsupervised case
                es_id=  self._model.e_sym_id_map[es]  if   self._model.e_sym_id_map.has_key(es) else len(self._model.e_sym_id_map)
                if es_id==len(self._model.e_sym_id_map):
                    self._model.e_sym_id_map[es]=es_id

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
        for s,i in self._model.f_sym_id_map.iteritems():
            self._model.f_id_sym_map[i]=s
    
        # for e
        for s,i in self._model.e_sym_id_map.iteritems():
            self._model.e_id_sym_map[i]=s
    
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
        self._model.param_values=np.zeros([len(self._model.e_sym_id_map),len(self._model.f_sym_id_map)])
        self.prev_param_values=np.zeros([len(self._model.e_sym_id_map),len(self._model.f_sym_id_map)])
    
    def _m_step(self): 
        
        # accumulating counts 
        for e_id in range(len(self._model.e_sym_id_map)): 
            for f_id in range(len(self._model.f_sym_id_map)): 
                self.prev_param_values[e_id,f_id]=self._model.param_values[e_id,f_id]
                self._model.param_values[e_id,f_id] = float(sum( [self.wpairs_weights[w_id][a_id] for w_id, a_id in  self.param_occurence_info[e_id][f_id] ] ))
       
        # normalizing
        for e_id in range(len(self._model.e_sym_id_map)): 
            norm_factor=np.sum(self._model.param_values[e_id,:])
            self._model.param_values[e_id,:]=self._model.param_values[e_id,:]/norm_factor if norm_factor>0.0 else 0.0
   
    def _e_step(self): 
        for wp_idx in xrange(len(self.wpairs_aligns)): 
            aln_probs=[]
            for aln_idx in xrange(len(self.wpairs_aligns[wp_idx])): 
                try: 
                    aln_probs.append(
                            math.exp(
                                    # TODO: what to do about negative log probability
                                    sum([ math.log(self._model.param_values[x[0],x[1]]) if self._model.param_values[x[0],x[1]]!=0.0  else 0.0 for x in self.wpairs_aligns[wp_idx][aln_idx] ])
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

        for e_id in range(len(self._model.e_sym_id_map)): 
            for f_id in range(len(self._model.f_sym_id_map)): 
                if math.fabs(self._model.param_values[e_id,f_id]-self.prev_param_values[e_id,f_id]) >= EPSILON:
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
        #for e_id in range(len(self._model.e_sym_id_map)): 
        #    for f_id in range(len(self._model.f_sym_id_map)): 
        #        #print u"P({}|{})={}".format(self._model.f_id_sym_map[f_id],self._model.e_id_sym_map[e_id],self._model.param_values[e_id,f_id]).encode('utf-8') 
        #        print u"P({}|{})={}".format(f_id,e_id,self._model.param_values[e_id,f_id]).encode('utf-8') 

    ##########################
    ### Decoding function #####
    ##########################
    def _bigram_score(self,hist_id,cur_id):
        """
        """
        bigram=u'{} {}'.format( self._model.e_id_sym_map[hist_id] if hist_id>=0 else u'<s>',
                                self._model.e_id_sym_map[cur_id])
        #bigram=u'{} {}'.format( self._model.e_id_sym_map[cur_id], self._model.e_id_sym_map[hist_id] if hist_id>=0 else u'<s>')
        return math.pow( 10 , srilm.getBigramProb(self.lm_model,bigram.encode('utf-8')) )

    def _get_param_value( self, e_id, f_input_chars ): 
        if self._model.f_sym_id_map.has_key(f_input_chars): 
            return self._model.param_values[  e_id , self._model.f_sym_id_map[f_input_chars] ]
        else:
            return 0.0

    def _decode_internal(self,f_input_word): 
        """
            bigram language model 
        """
        print 'Decoding word'

        sm_shape=(len(f_input_word), len(self._model.e_sym_id_map.keys()) )


        #print 'Parameter Space: ',
        #print self._model.param_values.shape

        #print 'Scoring Matrix dimensions: ',
        #print sm_shape

        # score matrix
        score_matrix=np.zeros( sm_shape ) 
        # backtracking matrices 
        max_row_matrix=np.zeros( sm_shape ,dtype=int ) 
        max_char_matrix=np.zeros( sm_shape,dtype=int ) 

        # initialization
        for k in xrange(sm_shape[1]):
            #print u'{} {}'.format( self._model.e_id_sym_map[k] ,self._bigram_score(-1,k)).encode('utf-8')
            #print u'{} {} {}'.format( self._model.e_id_sym_map[k] , f_input_word[0], self._get_param_value(  k , f_input_word[0])).encode('utf-8')
            max_row_matrix[0,k]=-1                       
            max_char_matrix[0,k]=-1                       
            score_matrix[0,k]= sum ( map ( log_z  ,
                                  #    LM score                    translition matrix score
                               [ self._bigram_score(-1,k) , self._get_param_value(  k , f_input_word[0]) ] 
                               #[  self._get_param_value(  k , f_input_word[0]) ] 
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
            #print u'{} {} {}'.format(cur_j,cur_k,self._model.e_id_sym_map[cur_k]).encode('utf-8')
            decoded_output.append( self._model.e_id_sym_map[cur_k] )
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

    ############################################
    ### Unsupervised training functions ########
    ############################################

    def _smooth_alignment_parameters(self):
        pass 

    def _initialize_unsupervised_training(self,f_input_words, e_char_set): 

        # create symbol to id mappings for f (from data)
        for f_input_word in f_input_words: 
            for c in f_input_word: 
                if not self._model.f_sym_id_map.has_key(c): 
                    self._model.f_sym_id_map[c]=len(self._model.f_sym_id_map)

        # create symbol to id mappings for e (from given list of characters)
        for i, c in enumerate(e_char_set):
            self._model.e_sym_id_map[c]=i

        # create inverse id-symbol mappings
        # for f
        for s,i in self._model.f_sym_id_map.iteritems():
            self._model.f_id_sym_map[i]=s
    
        # for e
        for s,i in self._model.e_sym_id_map.iteritems():
            self._model.e_id_sym_map[i]=s
    
        # initialize transliteration probabilities 
        #self._model.param_values=np.ones([len(self._model.e_sym_id_map),len(self._model.f_sym_id_map)]) * 1.0/len(self._model.f_sym_id_map)
        self._model.param_values=np.zeros([len(self._model.e_sym_id_map),len(self._model.f_sym_id_map)])
        for i in xrange(self._model.param_values.shape[0]):
            t=1.0
            for j in xrange(self._model.param_values.shape[1]-1):
                v=random.random()*t
                self._model.param_values[i,j]=v
                t=t-v
            self._model.param_values[i,-1]=t                
            
        self.prev_param_values=np.zeros([len(self._model.e_sym_id_map),len(self._model.f_sym_id_map)])

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
            #for e_id in range(len(self._model.e_sym_id_map)): 
            #    for f_id in range(len(self._model.f_sym_id_map)): 
            #        print u"P({}|{})={}".format(self._model.f_id_sym_map[f_id],self._model.e_id_sym_map[e_id],self._model.param_values[e_id,f_id]).encode('utf-8') 

            # decode: approximate marginalization over all e strings by maximization
            print "Decoding for EM"
            word_pairs=list(it.izip( f_input_words, it.imap( self._decode_internal,f_input_words)  ))

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
        for e_id in range(len(self._model.e_sym_id_map)): 
            for f_id in range(len(self._model.f_sym_id_map)): 
                print u"P({}|{})={}".format(self._model.f_id_sym_map[f_id],self._model.e_id_sym_map[e_id],self._model.param_values[e_id,f_id]).encode('utf-8') 

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
    
    #fcorpus_fname=sys.argv[1]
    #ecorpus_fname=sys.argv[2]
    #model_dir=sys.argv[3]
    #lm_fname=sys.argv[4]
    #test_fcorpus_fname=sys.argv[5]
    #test_ecorpus_fname=sys.argv[6]

    #########  Supervised training
    #data_dir='/home/development/anoop/experiments/unsupervised_transliterator/data'
    #parallel_dir=data_dir+'/'+'en-hi'

    #fcorpus_fname=parallel_dir+'/'+'train.en'
    #ecorpus_fname=parallel_dir+'/'+'train.hi'
    #lm_fname=data_dir+'/'+'hi-2g.lm'
    ##test_fcorpus_fname=parallel_dir+'/'+'test.en'
    ##test_ecorpus_fname=parallel_dir+'/'+'test.hi'
    #test_fcorpus_fname='test.en'
    #test_ecorpus_fname='test.hi'

    #em=UnsupervisedTransliteratorTrainer(lm_fname)
    #em.em_supervised_train(read_parallel_corpus(fcorpus_fname,ecorpus_fname))

    #em.evaluate(read_parallel_corpus(test_fcorpus_fname,test_ecorpus_fname))

        

    ########  Unsupervised training
    data_dir='/home/development/anoop/experiments/unsupervised_transliterator/data'
    #parallel_dir=data_dir+'/'+'en-hi'

    #fcorpus_fname=parallel_dir+'/'+'train.en'
    #ecorpus_fname=parallel_dir+'/'+'train.hi'
    #test_fcorpus_fname=parallel_dir+'/'+'test.en'
    #test_ecorpus_fname=parallel_dir+'/'+'test.hi'

    fcorpus_fname='10.en'
    ecorpus_fname='10.hi'
    test_fcorpus_fname='10.en'
    test_ecorpus_fname='10.hi'

    lm_fname=data_dir+'/'+'hi-2g.lm'

    em=UnsupervisedTransliteratorTrainer(lm_fname)
    em.em_unsupervised_train(read_monolingual_corpus(fcorpus_fname),generate_char_set(ecorpus_fname))

    em.evaluate(read_parallel_corpus(test_fcorpus_fname,test_ecorpus_fname))


