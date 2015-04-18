import itertools as it
import codecs, sys
import pprint 
from collections import defaultdict
import numpy as np, math

import srilm


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

class UnsupervisedTransliteratorTrainer: 

    ## ALLOWED mappings
    ALLOWED_MAPPINGS=[1,2]
    EPSILON=0.005
   
    def _init_lm(self,lm_fname): 
        self.lm_model=srilm.initLM(2)
        srilm.readLM(self.lm_model,lm_fname)    

    def __init__(self,lm_fname):
        self.wpairs_aligns=[]
        self.wpairs_weights=[]

        self.f_sym_id_map={}
        self.e_sym_id_map={}

        self.f_id_sym_map={}
        self.e_id_sym_map={}

        self.param_occurence_info=defaultdict(lambda :defaultdict(list))
        self.param_values={}
        self.prev_param_values={}

        # Decoder members 
        # bigram language model 
        self.lm_model=None
        self._init_lm(lm_fname)

    def print_obj(self): 

        print("xxx Printing EM instance xxx")
        print("Symbol mappings for F: ")
        pprint.pprint(self.f_sym_id_map)

        print("Symbol mappings for E: ")
        pprint.pprint(self.e_sym_id_map)

        print("Param Occurence Info: ")
        for eid in xrange(len(self.e_sym_id_map)): 
            for fid in xrange(len(self.f_sym_id_map)): 
                l=self.param_occurence_info[eid][fid]
                if len(l)>0:
                    print 'eid={} fid={}'.format(eid,fid)
                    pprint.pprint(l)

        print("Alignments: ")
        # gather transliteration occurrence info
        for wp_idx,alignments in enumerate(self.wpairs_aligns): 
            for aln_idx,align in enumerate(alignments): 
                print 'wp_idx={} aln_idx={}'.format(wp_idx,aln_idx)
                pprint.pprint(align)
                #for charseq_pair in align: 
                #    print[charseq_pair[0]][charseq_pair[1]].append([wp_idx,aln_idx])

    def print_params(self): 
        print("Alignment Weights: ")
        pprint.pprint(self.wpairs_weights)
       
        #print("Transliteration Probabilities")
        #pprint.pprint(self.param_values)
        

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
                fs_id=  self.f_sym_id_map[fs]  if   self.f_sym_id_map.has_key(fs) else len(self.f_sym_id_map)
                if fs_id==len(self.f_sym_id_map):
                    self.f_sym_id_map[fs]=fs_id

                # create 'e' sym to id mapping
                es_id=  self.e_sym_id_map[es]  if   self.e_sym_id_map.has_key(es) else len(self.e_sym_id_map)
                if es_id==len(self.e_sym_id_map):
                    self.e_sym_id_map[es]=es_id

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
        for s,i in self.f_sym_id_map.iteritems():
            self.f_id_sym_map[i]=s
    
        # for e
        for s,i in self.e_sym_id_map.iteritems():
            self.e_id_sym_map[i]=s
    
    def _initialize_parameter_structures(self): 
        """
    
        """
        self.param_occurence_info=defaultdict(lambda :defaultdict(list))
        self.param_values={}
   
        # gather transliteration occurrence info
        for wp_idx,alignments in enumerate(self.wpairs_aligns): 
            for aln_idx,align in enumerate(alignments): 
                for charseq_pair in align: 
                    self.param_occurence_info[charseq_pair[0]][charseq_pair[1]].append([wp_idx,aln_idx])
    
        # initialize transliteration probabilities 
        self.param_values=np.zeros([len(self.e_sym_id_map),len(self.f_sym_id_map)])
        self.prev_param_values=np.zeros([len(self.e_sym_id_map),len(self.f_sym_id_map)])
    
    def _m_step(self): 
        
        # accumulating counts 
        for e_id in range(len(self.e_sym_id_map)): 
            for f_id in range(len(self.f_sym_id_map)): 
                self.prev_param_values[e_id,f_id]=self.param_values[e_id,f_id]
                self.param_values[e_id,f_id] = float(sum( [self.wpairs_weights[w_id][a_id] for w_id, a_id in  self.param_occurence_info[e_id][f_id] ] ))
       
        # normalizing
        for e_id in range(len(self.e_sym_id_map)): 
            norm_factor=np.sum(self.param_values[e_id,:])
            self.param_values[e_id,:]=self.param_values[e_id,:]/norm_factor 
   
    def _e_step(self): 
        for wp_idx in xrange(len(self.wpairs_aligns)): 
            aln_probs=[]
            for aln_idx in xrange(len(self.wpairs_aligns[wp_idx])): 
                try: 
                    aln_probs.append(
                            math.exp(
                                    sum([ math.log(self.param_values[x[0],x[1]]) if self.param_values[x[0],x[1]]!=0.0  else 0.0 for x in self.wpairs_aligns[wp_idx][aln_idx] ])
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

        for e_id in range(len(self.e_sym_id_map)): 
            for f_id in range(len(self.f_sym_id_map)): 
                if math.fabs(self.param_values[e_id,f_id]-self.prev_param_values[e_id,f_id]) >= self.EPSILON:
                    converged=False
                    break;

        return converged

    def em_train(self,word_pairs): 
        """
        """
    
        self._create_alignment_database(word_pairs)
        self._initialize_parameter_structures()
    
        print self.param_occurence_info 

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


        print "Final paramters"
        for e_id in range(len(self.e_sym_id_map)): 
            for f_id in range(len(self.f_sym_id_map)): 
                print u"P({}|{})={}".format(self.f_id_sym_map[f_id],self.e_id_sym_map[e_id],self.param_values[e_id,f_id]).encode('utf-8') 

    def _bigram_score(self,hist_id,cur_id):
        """
        """
        bigram=u'{} {}'.format( self.e_id_sym_map(hist_id) if hist_id>=0 else u'<s>',
                                self.e_id_sym_map(cur_id))
        return math.pow( 10 , srilm.getBigramProb(self.lm_model,bigram.encode('utf-8')) )

    def decode(self,f_input_word): 
        """
            bigram language model 
        """

        sm_shape=(len(f_input_word), len(self.e_sym_id_map.keys()) )

        # score matrix
        score_matrix=np.zeros( sm_shape ) 
        # backtracking matrices 
        max_row_matrix=np.zeros( sm_shape ,dtype=int ) 
        max_char_matrix=np.zeros( sm_shape,dtype=int ) 

        # initialization
        for k in xrange(sm_shape[1]):
            score_matrix[0,k]= sum ( map ( math.log ,
                                  #    LM score                    translition matrix score
                               [ self._bigram_score(-1,k) , self.param_values[  k , self.f_sym_id_map[f_input_word[0]] ] ] 
                         ) 
                       )

            max_row_matrix[0,k]=-1                       
            max_char_matrix[0,k]=-1                       

        ### modified  viterbi decoding
        # j- input position
        # k- output char id (current)
        # m- output char id (prev)


        # Compute the scoring matrix and create the backtracking vector 

        for j in xrange(np.shape(score_matrix)[0]):
            for k in xrange(np.shape(score_matrix)[1]):

                # Case 1: A new output character is generated by input(j) alone
                max_val_1=float('-inf')
                max_char_1=-1

                # evaluate all (j-1) candidates
                for m in xrange(np.shape(score_matrix)[1]):
                    v=sum ( map ( math.log ,
                               # prev row score         LM score                    transliteration matrix score
                               [ score_matrix[j-1,m] , self._bigram_score(m,k) , self.param_values[  k , self.f_sym_id_map[f_input_word[j]] ] ] 
                         ) 
                       )  

                    if v>=max_val_1:
                        max_val_1=v
                        max_char_1=m
                         
                # Case 2: A new output character is generated by input(j) and input(j-1)
                max_val_2=float('-inf')
                max_char_2=-1

                # evaluate all (j-2) candidates
                for m in xrange(np.shape(score_matrix)[1]):
                    v=sum ( map ( math.log ,
                               # prev row score         LM score                    transliteration matrix score
                               [ score_matrix[j-2,m] , self._bigram_score(m,k) , self.param_values[  k , self.f_sym_id_map[ f_input_word[j-1] +  f_input_word[j] ] ] ] 
                         ) 
                       )  

                    if v>=max_val_2:
                        max_val_2=v
                        max_char_2=m

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
            decoded_output.append( self.e_id_sym_map[cur_k] )
            cur_j, cur_k = ( max_row_matrix[cur_j,cur_k] ,  max_char_matrix[cur_j,cur_k] )

        return reversed(u''.join(decoded_output))
   
    def evaluate(self,word_pairs):
        for f_input_word, e_output_word in word_pairs: 
            best_output=self.decode(f_input_word)
            print u'{} {} {}'.format(f_input_word,e_output_word,best_output).encode('utf-8')

def read_parallel_corpus(fcorpus_fname,ecorpus_fname): 
    with codecs.open(fcorpus_fname,'r','utf-8') as ffile:
        with codecs.open(ecorpus_fname,'r','utf-8') as efile:
            return [ ( f.split() , e.split()  )  for f,e in it.izip( iter(ffile)  , iter(efile)  ) ] 


if __name__=='__main__': 
    
    #fcorpus_fname=sys.argv[1]
    #ecorpus_fname=sys.argv[2]
    #model_dir=sys.argv[3]
    #lm_fname=sys.argv[4]
    #test_fcorpus_fname=sys.argv[5]
    #test_ecorpus_fname=sys.argv[6]

    fcorpus_fname=sys.argv[1]
    ecorpus_fname=sys.argv[2]
    model_dir=sys.argv[3]
    lm_fname=sys.argv[4]
    test_fcorpus_fname=sys.argv[5]
    test_ecorpus_fname=sys.argv[6]

    em=UnsupervisedTransliteratorTrainer()
    em.em_train(read_parallel_corpus(fcorpus_fname,ecorpus_fname))


    em.evaluate(read_parallel_corpus(test_fcorpus_fname,test_ecorpus_fname))
