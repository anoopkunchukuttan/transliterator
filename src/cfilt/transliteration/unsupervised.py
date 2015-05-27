import itertools as it
import codecs, sys, pickle
import pprint 
from collections import defaultdict
import numpy as np, math
import pandas as pd
import random
import yaml 

import srilm

from cfilt.transliteration.decoder import *
from cfilt.transliteration.parallel_decoder import *
from cfilt.transliteration.utilities import *

from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator 
from indicnlp import langinfo 


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

    def __init__(self, config_params, lm_model=None):
        
        ############ load config file 
        self._config=config_params

        ########### transliteration model 
        self._translit_model=TransliterationModel()

        ############  corpus information
        # Each word-pair is indexed by its position in the input corpus (starting at 0)

        # list of all possible alignments for each word pair 
        self.wpairs_aligns=[]

        # list of weights for all possible alignments for a word pair 
        self.wpairs_weights=[]

        ############   parameter information

        # for each parameter (which corresponds to a point alignment), the list of occurences in of this point alignment all possible (wordpair,alignment) locations in the corpus
        self.param_occurence_info=defaultdict(lambda :defaultdict(list))

        # Parameter values in the previous iteration 
        self.prev_param_values=None

        ############# Language model 
        self._lm_model=lm_model

        #############  Hyper parameters 
        # initialize hyper parameters 
        # Each of the P(f|e) distributions is governed by a Dirichlet Prior alpha[e,f] for each value for e and f

        # scale paramter     
        self._scale=1.0
        self._alpha=None

        if len(config_params['prior_config'].keys())>1: 
            raise Exception('More than one configuration for prior specified')
        elif len(config_params['prior_config'].keys())<1: 
            raise Exception('No prior specified')
        
        self._priormethod,self._priorparams=config_params['prior_config'].iteritems().next()


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
                                it.product(self._config['allowed_mappings'] ,repeat=ne)
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
   
    def _init_dirichlet_priors(self): 
        print 'Prior Method: {}'.format(self._priormethod)

        if self._priormethod=='random':
            self._scale=self._priorparams['scale']
            self._alpha=np.random.rand(len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map))
            self._alpha*=self._priorparams['scale']
            print 'Scale: {}'.format(self._scale)

        elif self._priormethod=='zero':
            self._alpha=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])

        elif self._priormethod=='indic_mapping':
            self._alpha=np.random.rand(len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map))
            src=self._priorparams['src']
            tgt=self._priorparams['tgt']

            for e_id, e_sym in self._translit_model.e_id_sym_map.iteritems(): 
                offset=ord(e_sym)-langinfo.SCRIPT_RANGES[src][0]
                if offset >=langinfo.COORDINATED_RANGE_START_INCLUSIVE and offset <= langinfo.COORDINATED_RANGE_END_INCLUSIVE:
                    f_sym_x=UnicodeIndicTransliterator.transliterate(e_sym,tgt,src)
                    if f_sym_x in self._translit_model.f_sym_id_map: 
                        self._alpha[e_id,self._translit_model.f_sym_id_map[f_sym_x]]=self._priorparams['base_measure_mapping_exists']
                        self._alpha[e_id,:]*=self._priorparams['scale_factor_mapping_exists']

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

        # initialize the Dirichlet Prior values 
        self._init_dirichlet_priors()

    def _m_step(self): 
        
        # accumulating counts 
        for e_id in range(len(self._translit_model.e_sym_id_map)): 
            for f_id in range(len(self._translit_model.f_sym_id_map)): 
                self.prev_param_values[e_id,f_id]=self._translit_model.param_values[e_id,f_id]
                self._translit_model.param_values[e_id,f_id] = float(sum( [self.wpairs_weights[w_id][a_id] for w_id, a_id in  self.param_occurence_info[e_id][f_id] ] )) + self._alpha[e_id,f_id]  
                        
       
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

        n=0
        for e_id in range(len(self._translit_model.e_sym_id_map)): 
            for f_id in range(len(self._translit_model.f_sym_id_map)): 
                if math.fabs(self._translit_model.param_values[e_id,f_id]-self.prev_param_values[e_id,f_id]) >= self._config['conv_epsilon']:
                    converged=False
                    break;
                n=n+1

        print 'Checked {} parameters for convergence'.format(n)
        return converged

    #def _debug_log(self): 
    #    if self._

    def em_supervised_train(self,word_pairs): 
        """
        """
    
        self._create_alignment_database(word_pairs)
        self._initialize_parameter_structures()
    
        niter=0
        while(True):
            # M-step
            self._m_step()
            niter+=1

            #self.print_params()
            #self._debug_log()
            print '=== Iteration {} completed ==='.format(niter)

            # check for end of process 
            if niter>=self._config['train_iteration'] or self._has_converged():
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

        # initialize hyper parameters 
        self._init_dirichlet_priors()

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

            ## decode: approximate marginalization over all e strings by maximization
            #print "Decoding for EM"
            #decoder=TransliterationDecoder(self._translit_model,self._lm_model)
            #word_pairs=list(it.izip( f_input_words, it.imap( decoder._decode_internal,f_input_words)  ))

            print "Parallel Decoding for EM"
            word_pairs=list(it.izip( f_input_words , parallel_decode_char_string(self._translit_model, self._lm_model, f_input_words) ) )

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
            if niter>= self._config['train_iteration']  or self._has_converged():
                break
    
            ######  E-step ####
            print "Computing alignment weights" 
            self._e_step()

        self.print_obj()
        print "Final parameters"
        for e_id in range(len(self._translit_model.e_sym_id_map)): 
            for f_id in range(len(self._translit_model.f_sym_id_map)): 
                print u"P({}|{})={}".format(self._translit_model.f_id_sym_map[f_id],self._translit_model.e_id_sym_map[e_id],self._translit_model.param_values[e_id,f_id]).encode('utf-8') 

if __name__=='__main__': 
    ##### Parallel Decoding
    #data_dir='/home/development/anoop/experiments/unsupervised_transliterator/data'
    ##parallel_dir=data_dir+'/'+'en-hi'

    ##fcorpus_fname=parallel_dir+'/'+'train.en'
    ##ecorpus_fname=parallel_dir+'/'+'train.hi'
    #lm_fname=data_dir+'/'+'hi-2g.lm'
    ###test_fcorpus_fname=parallel_dir+'/'+'test.en'
    ###test_ecorpus_fname=parallel_dir+'/'+'test.hi'
    ##test_fcorpus_fname='test.en'
    ##test_ecorpus_fname='test.hi'
    ##with timer.Timer(True) as t: 
    ##    parallel_evaluate(TransliterationModel.load_translit_model('translit.model'),
    ##                        load_lm_model(lm_fname),
    ##                        read_parallel_corpus(test_fcorpus_fname,test_ecorpus_fname)
    ##                     )

    #print parallel_decode_char_string(TransliterationModel.load_translit_model('translit.model'), load_lm_model(lm_fname), read_monolingual_string_corpus('test.en')) 
    ##print 'Time for decoding: '.format(t.secs)

    #
    #######  model information
    ####  F: source  E: target

    #### file listing set of characters in the source
    ##fcfname='kannada/En-Ka-News_EnglishLabels_KannadaRows_EnglishColumns_linear_'
    #### file listing set of characters in the target
    ##ecfname='kannada/En-Ka-News_KannadaLabels_KannadaRows_EnglishColumns_linear_'
    #### file listing alignment from source to target
    ####  target is along the rows, source is along the columns
    ##alignmentfname='kannada/En-Ka-News_CrossEntropy_AlignmentMatrix_KannadaRows_EnglishColumns_linear_'
    ##### bigram- target language model in APRA  model. Note: decoding currently supports only bigram models
    ##lm_fname='kannada/Ka-2g.lm'
    ##
    ####### testset information 
    ##test_fcorpus_fname='kannada/test.En'
    ##test_ecorpus_fname='kannada/test.Ka'

    ##tm_model=TransliterationModel.construct_transliteration_model(fcfname,ecfname,alignmentfname)
    ##lm_model=load_lm_model(lm_fname)

    ##decoder=TransliterationDecoder(tm_model,lm_model)
    ##decoder.evaluate(read_parallel_corpus(test_fcorpus_fname,test_ecorpus_fname))

    ###fcorpus_fname=sys.argv[1]
    ###ecorpus_fname=sys.argv[2]
    ###model_dir=sys.argv[3]
    ###lm_fname=sys.argv[4]
    ###test_fcorpus_fname=sys.argv[5]
    ###test_ecorpus_fname=sys.argv[6]

    #############  Supervised training
    #data_dir='/home/development/anoop/experiments/unsupervised_transliterator/data'
    #parallel_dir=data_dir+'/'+'en-hi'

    #fcorpus_fname=parallel_dir+'/'+'train.en'
    #ecorpus_fname=parallel_dir+'/'+'train.hi'
    #lm_fname=data_dir+'/'+'hi-2g.lm'
    ###test_fcorpus_fname=parallel_dir+'/'+'test.en'
    ###test_ecorpus_fname=parallel_dir+'/'+'test.hi'
    ##test_fcorpus_fname='test.en'
    ##test_ecorpus_fname='test.hi'

    #lm_model=load_lm_model(lm_fname)

    #em=UnsupervisedTransliteratorTrainer(lm_model)
    #em.em_supervised_train(read_parallel_string_corpus(fcorpus_fname,ecorpus_fname))
    ###TransliterationModel.save_translit_model(em._translit_model,'translit.model')

    #
    #decoder=TransliterationDecoder(em._translit_model,em._lm_model)
    ##decoder=TransliterationDecoder(TransliterationModel.load_translit_model('translit.model'),load_lm_model(lm_fname))
    #with timer.Timer(True) as t: 
    #    decoder.evaluate(read_parallel_string_corpus(test_fcorpus_fname,test_ecorpus_fname))
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
    ##TransliterationModel.save_translit_model(em._translit_model, 'noprior.model'): 

    ##decoder=TransliterationDecoder(em._translit_model,em._lm_model)
    ##decoder.evaluate(read_parallel_corpus(test_fcorpus_fname,test_ecorpus_fname))

    #with timer.Timer(True) as t: 
    #    parallel_evaluate(em._translit_model,em._lm_model,
    #                        read_parallel_string_corpus(test_fcorpus_fname,test_ecorpus_fname)
    #                     )
    #print 'Time for decoding: '.format(t.secs)

    pass 
