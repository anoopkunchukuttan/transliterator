import itertools as it
import codecs, sys, pickle, os
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
from cfilt.transliteration.char_mappings import *

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

        # list of weights for each word-pair. These are required for unsupervised training only
        self.wpairs_eword_weights=[]

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

        # Final alpha paramters for the Dirchlet distribution of the prior     
        self._alpha=None

        if 'prior_config' in  self._config:
            if len(self._config['prior_config'].keys())>1: 
                raise Exception('More than one configuration for prior specified')
            elif len(self._config['prior_config'].keys())<1: 
                raise Exception('No prior specified')
        
            self._priormethod,self._priorparams=config_params['prior_config'].iteritems().next()

        ##### parameter initialization
        if 'initialization' in  self._config:
            if len(self._config['initialization'].keys())>1: 
                raise Exception('More than one configuration for initialization specified')
            elif len(self._config['initialization'].keys())<1: 
                raise Exception('No initialization method specified')
        
            self._initmethod,self._initparams=config_params['initialization'].iteritems().next()

#    def print_obj(self): 
#
#        print("xxx Printing EM instance xxx")
#        print("Symbol mappings for F: ")
#        for f_id in range(len(self._translit_model.f_id_sym_map)): 
#            print u'{} {}'.format(f_id,self._translit_model.f_id_sym_map[f_id]).encode('utf-8')
#
#        print("Symbol mappings for E: ")
#        for e_id in range(len(self._translit_model.e_id_sym_map)): 
#            print u'{} {}'.format(e_id,self._translit_model.e_id_sym_map[e_id]).encode('utf-8')
#
#        #print("Param Occurence Info: ")
#        #for eid in xrange(len(self._translit_model.e_sym_id_map)): 
#        #    for fid in xrange(len(self._translit_model.f_sym_id_map)): 
#        #        l=self.param_occurence_info[eid][fid]
#        #        if len(l)>0:
#        #            print 'eid={} fid={}'.format(eid,fid)
#        #            pprint.pprint(l)
#
#        #print("Alignments: ")
#        ## gather transliteration occurrence info
#        #for wp_idx,alignments in enumerate(self.wpairs_aligns): 
#        #    for aln_idx,align in enumerate(alignments): 
#        #        print 'wp_idx={} aln_idx={}'.format(wp_idx,aln_idx)
#        #        pprint.pprint(align)
#        #        #for charseq_pair in align: 
#        #        #    print[charseq_pair[0]][charseq_pair[1]].append([wp_idx,aln_idx])
#
#    def print_params(self): 
#        #print("Alignment Weights: ")
#        #pprint.pprint(self.wpairs_weights)
#       
#        print("Transliteration Probabilities")
#        #for e_id in range(len(self._translit_model.e_sym_id_map)): 
#        #    for f_id in range(len(self._translit_model.f_sym_id_map)): 
#        #        #print u"P({}|{})={}".format(self._translit_model.f_id_sym_map[f_id],self._translit_model.e_id_sym_map[e_id],self._translit_model.param_values[e_id,f_id]).encode('utf-8') 
#        #        print u"P({}|{})={}".format(f_id,e_id,self._translit_model.param_values[e_id,f_id]).encode('utf-8') 
#        #pprint.pprint(self._translit_model.param_values)
        

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
        self.wpairs_eword_weights=[]
    
        for f,e in word_pairs: 
            alignments=self._generate_alignments(f,e)
            if len(alignments)>0:
                self.wpairs_aligns.append(alignments)
                self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )
                self.wpairs_eword_weights.append(1.0)
            else: 
                print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 

        # create inverse id-symbol mappings
        # for f
        for s,i in self._translit_model.f_sym_id_map.iteritems():
            self._translit_model.f_id_sym_map[i]=s
    
        # for e
        for s,i in self._translit_model.e_sym_id_map.iteritems():
            self._translit_model.e_id_sym_map[i]=s
  
    def _generate_en_indic_hyperparams(self,params):

        alpha=np.ones((len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)))

        ## get mapping rules 
        mapping_rules=filter_by_val_len(remove_constraints(en_il_rules))
        if 'upper_case' in params:
            mapping_rules=conv_upper(mapping_rules)

        src=params['src']
        tgt=params['tgt']

        for e_id, e_sym in self._translit_model.e_id_sym_map.iteritems(): 
            offset=ord(e_sym)-langinfo.SCRIPT_RANGES[tgt][0]
            if offset in mapping_rules: 
                for f_sym_x in mapping_rules[offset]:
                    if f_sym_x in self._translit_model.f_sym_id_map: 
                        alpha[e_id,self._translit_model.f_sym_id_map[f_sym_x]]=params['base_measure_mapping_exists']
                        alpha[e_id,:]*=params['scale_factor_mapping_exists']

        return alpha

    def _init_param_values(self):

        if  not hasattr(self,'_initmethod') or self._initmethod=='random':
            self._translit_model.param_values=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])
            for i in xrange(self._translit_model.param_values.shape[0]):
                t=1.0
                for j in xrange(self._translit_model.param_values.shape[1]-1):
                    v=random.random()*t
                    self._translit_model.param_values[i,j]=v
                    t=t-v
                self._translit_model.param_values[i,-1]=t                

        elif  self._initmethod=='uniform':
            self._translit_model.param_values=np.ones([len(self._translit_model.e_sym_id_map),
                                len(self._translit_model.f_sym_id_map)]) * 1.0/len(self._translit_model.f_sym_id_map)

        elif  self._initmethod=='indic_mapping':

            alpha=np.ones((len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)))
            src=self._initparams['src']
            tgt=self._initparams['tgt']

            for e_id, e_sym in self._translit_model.e_id_sym_map.iteritems(): 
                offset=ord(e_sym)-langinfo.SCRIPT_RANGES[tgt][0]
                if offset >=langinfo.COORDINATED_RANGE_START_INCLUSIVE and offset <= langinfo.COORDINATED_RANGE_END_INCLUSIVE:
                    f_sym_x=UnicodeIndicTransliterator.transliterate(e_sym,tgt,src)
                    if f_sym_x in self._translit_model.f_sym_id_map: 
                        alpha[e_id,self._translit_model.f_sym_id_map[f_sym_x]]=self._initparams['base_measure_mapping_exists']
                        alpha[e_id,:]*=self._initparams['scale_factor_mapping_exists']

            alpha_sums=np.sum(alpha, axis=1)
            self._translit_model.param_values=(alpha.transpose()/alpha_sums).transpose()

        elif  self._initmethod=='en_il_mapping':
            alpha=self._generate_en_indic_hyperparams(self._initparams)
            alpha_sums=np.sum(alpha, axis=1)
            self._translit_model.param_values=(alpha.transpose()/alpha_sums).transpose()

            df=pd.DataFrame(self._translit_model.param_values)
            df.to_csv('paramvalues.csv')


    def _init_dirichlet_priors(self): 

        self._alpha=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])

        if hasattr(self,'_priormethod'):
            if self._priormethod=='random':
                self._alpha=np.random.rand(len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map))
                self._alpha*=self._priorparams['scale']
                print 'Scale: {}'.format(self._priorparams['scale'] )

            elif self._priormethod=='indic_mapping':
                self._alpha=np.ones((len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)))
                src=self._priorparams['src']
                tgt=self._priorparams['tgt']

                for e_id, e_sym in self._translit_model.e_id_sym_map.iteritems(): 
                    offset=ord(e_sym)-langinfo.SCRIPT_RANGES[tgt][0]
                    if offset >=langinfo.COORDINATED_RANGE_START_INCLUSIVE and offset <= langinfo.COORDINATED_RANGE_END_INCLUSIVE:
                        f_sym_x=UnicodeIndicTransliterator.transliterate(e_sym,tgt,src)
                        if f_sym_x in self._translit_model.f_sym_id_map: 
                            self._alpha[e_id,self._translit_model.f_sym_id_map[f_sym_x]]=self._priorparams['base_measure_mapping_exists']
                            self._alpha[e_id,:]*=self._priorparams['scale_factor_mapping_exists']

            elif  self._initmethod=='en_il_mapping':
                self._alpha=self._generate_en_indic_hyperparams(self._initparams)
                df=pd.DataFrame(self._alpha)
                df.to_csv('alpha.csv')

            elif self._priormethod=='add_one_smoothing':
                self._alpha=np.ones((len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)))

    def _initialize_parameter_structures(self): 
        """
    
        """
        # gather transliteration occurrence info
        self.param_occurence_info=defaultdict(lambda :defaultdict(list))
        for wp_idx,alignments in enumerate(self.wpairs_aligns): 
            for aln_idx,align in enumerate(alignments): 
                for es_id, fs_id in align: 
                    self.param_occurence_info[es_id][fs_id].append([wp_idx,aln_idx])
    
        # initialize the Dirichlet Prior values 
        self._init_dirichlet_priors()

        # initialize transliteration probabilities 
        self._init_param_values()

        self.prev_param_values=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])

    def _m_step(self): 
        
        # accumulating counts 
        for e_id in range(len(self._translit_model.e_sym_id_map)): 
            for f_id in range(len(self._translit_model.f_sym_id_map)): 
                self.prev_param_values[e_id,f_id]=self._translit_model.param_values[e_id,f_id]
                #self._translit_model.param_values[e_id,f_id] = float(sum( [self.wpairs_weights[w_id][a_id] for w_id, a_id in  self.param_occurence_info[e_id][f_id] ] )) + self._alpha[e_id,f_id]  
                self._translit_model.param_values[e_id,f_id] = float(sum( [self.wpairs_eword_weights[w_id]*self.wpairs_weights[w_id][a_id] for w_id, a_id in  self.param_occurence_info[e_id][f_id] ] )) + self._alpha[e_id,f_id]  
       
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

    def _debug_log(self,iter_no,word_triplets=None): 
        # log if this parameter exists
        if 'log_dir' in self._config: 
            if iter_no==0:
                # create the log dir
                if not os.path.isdir(self._config['log_dir']):
                    os.mkdir(self._config['log_dir'])

                with open( '{}/alpha.pickle'.format(self._config['log_dir'],iter_no) , 'w' ) as pfile:
                    pickle.dump( self._alpha, pfile )

            os.mkdir('{}/{}'.format(self._config['log_dir'],iter_no))
            TransliterationModel.save_translit_model( self._translit_model  , '{}/{}/translit.model'.format(self._config['log_dir'],iter_no) )

            # write intemediate transliterations 
            if word_triplets is not None:
                with codecs.open('{}/{}/transliterations.txt'.format(self._config['log_dir'],iter_no) , 'w',  'utf-8' ) as ofile:

                    for src_w, tgt_w, prev_tgt_w in word_triplets:
                        ofile.write(' '.join(tgt_w)+'\n')

                    #for f,e_cands in word_triplets: 
                    #    for e, score in e_cands: 
                    #        ofile.write(' '.join(e)+'\n')

    def em_supervised_train(self,word_pairs): 
        """
        """
    
        self._create_alignment_database(word_pairs)
        self._initialize_parameter_structures()
    
        niter=0
        self._debug_log(niter)

        while(True):
            # M-step
            self._m_step()

            niter+=1
            self._debug_log(niter)

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
    
        # initialize hyper parameters 
        self._init_dirichlet_priors()

        #####  initialize transliteration probabilities #####
        self._init_param_values()

        #####  previous parameter values  #####
        self.prev_param_values=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])

    def _prepare_corpus_unsupervised(self,word_triplets, append=True): 
        """
          symbol mappings have already been created using '_initialize_unsupervised_training' 

          every character sequence in the corpus can be uniquely addressed as: 
          corpus[0][word_pair_idx][align_idx][aln_point_idx]
    
          every weight can be indexed as: 
          corpus[1][word_pair_idx][align_idx]
    
        """
        self.wpairs_aligns=[]
        self.wpairs_weights=[]
    
        for f,e,e_prev in word_triplets: 
            alignments=self._generate_alignments(f,e)
            if len(alignments)>0:
                self.wpairs_aligns.append(alignments)
                self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )
            else: 
                print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 

        self.wpairs_eword_weights=[1.0]*len(self.wpairs_aligns)

        self.param_occurence_info=defaultdict(lambda :defaultdict(list))
   
        # gather transliteration occurrence info
        for wp_idx,alignments in enumerate(self.wpairs_aligns): 
            for aln_idx,align in enumerate(alignments): 
                for es_id, fs_id in align: 
                    self.param_occurence_info[es_id][fs_id].append([wp_idx,aln_idx])
    
    #def _prepare_corpus_unsupervised(self,word_triplets,append): 
    #    """
    #      symbol mappings have already been created using '_initialize_unsupervised_training' 

    #      every character sequence in the corpus can be uniquely addressed as: 
    #      corpus[0][word_pair_idx][align_idx][aln_point_idx]
    #
    #      every weight can be indexed as: 
    #      corpus[1][word_pair_idx][align_idx]
    #
    #    """
    #    if append:
    #        for widx, (f,e,e_prev) in enumerate(word_triplets): 
    #            alignments=self._generate_alignments(f,e)
    #            if len(alignments)>0:
    #                self.wpairs_aligns.append(alignments)
    #                self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments) )  
    #            else: 
    #                print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 
    #    else:                         
    #        for widx, (f,e,e_prev) in enumerate(word_triplets): 
    #            if e!=e_prev:
    #                alignments=self._generate_alignments(f,e)
    #                if len(alignments)>0:
    #                    self.wpairs_aligns[widx]=alignments
    #                    self.wpairs_weights[widx]=[1.0/float( len(alignments) )] *  len(alignments)  
    #                else: 
    #                    print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 

    #    self.wpairs_eword_weights=[1.0]*len(self.wpairs_aligns)

    #    self.param_occurence_info=defaultdict(lambda :defaultdict(list))
   
    #    # gather transliteration occurrence info
    #    for wp_idx,alignments in enumerate(self.wpairs_aligns): 
    #        for aln_idx,align in enumerate(alignments): 
    #            for es_id, fs_id in align: 
    #                self.param_occurence_info[es_id][fs_id].append([wp_idx,aln_idx])

    def _normalize_topn_scores(self,translation_output): 
       
        e_scores=[ score for e, score in translation_output ]
        e_words=[ e for e, score in translation_output ]
        N=-min(e_scores)+1.0
        e_scores=[ (N+score)/(len(e_scores)*N+sum(e_scores)) for score in e_scores ]
        return zip(e_words,e_scores)

    def _prepare_corpus_unsupervised_topn(self,word_triplets, topn, append=True): 
        """
    
        """
        self.wpairs_aligns=[]
        self.wpairs_weights=[]
        self.wpairs_eword_weights=[]
   
        # normalize top-k scores 
        input_words= [ f for f, e_cands in word_triplets ]
        translation_outputs= [ e_cands for f, e_cands in word_triplets ]
        translation_outputs=[ self._normalize_topn_scores(x) for x in translation_outputs]
        word_triplets=list(it.izip(input_words,translation_outputs))

        # generate alignment information
        for f,e_cands in word_triplets: 
            for e, score in e_cands: 
                alignments=self._generate_alignments(f,e)
                if len(alignments)>0:
                    self.wpairs_aligns.append(alignments)
                    self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )
                    self.wpairs_eword_weights.append(score)
                else: 
                    print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 

        # gather transliteration occurrence info
        self.param_occurence_info=defaultdict(lambda :defaultdict(list))

        for wp_idx,alignments in enumerate(self.wpairs_aligns): 
            for aln_idx,align in enumerate(alignments): 
                for es_id, fs_id in align: 
                    self.param_occurence_info[es_id][fs_id].append([wp_idx,aln_idx])
    
    def em_unsupervised_train(self,f_input_words,e_char_set): 
        """
        """
        print "Initializing unsupervised learning" 
        topn=5
        self._initialize_unsupervised_training(f_input_words,e_char_set)

        niter=0
        output_words=None
        append=True
        

        self._debug_log(niter)

        while(True):
            ##### M-step #####

            ## decode: approximate marginalization over all e strings by maximization

            ### >>> Simple 1-best candidate based training (1)
            print "Parallel Decoding for EM"
            prev_outputs=output_words if output_words is not None else ['']*len(f_input_words)
            output_words=parallel_decode(self._translit_model, self._lm_model, f_input_words)
            word_triplets=list(it.izip( f_input_words , output_words, prev_outputs ) )

            # initialize the EM training
            print "Preparing corpus"
            self._prepare_corpus_unsupervised(word_triplets,append)
            ## >>>

            #### >>> top-k candidate based training (without updating alignment probabilities ie not using e-step)
            #print "Parallel Decoding for EM"
            ## translation outputs: list of tuples (source word, list of tuples(target, probability)  )
            #word_triplets=list(it.izip( f_input_words , parallel_decode_topn(self._translit_model, self._lm_model, f_input_words, topn) ) )

            ## initialize the EM training
            #print "Preparing corpus"
            #self._prepare_corpus_unsupervised_topn(word_triplets, topn)
            #### >>>

            # estimate alignment parameters
            print "Estimating parameters"
            self._m_step()

            niter+=1
            append=False
            self._debug_log(niter,word_triplets)
            print '=== Iteration {} completed ==='.format(niter)

            # check for end of process 
            if niter>= self._config['train_iteration']  or self._has_converged():
                break
    
            ######  E-step ####
            #print "Computing alignment weights" 
            #self._e_step()

