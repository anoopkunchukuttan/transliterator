import itertools as it
import pickle, os
from collections import defaultdict
import numpy as np, math
import random
import copy

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

    def __init__(self, config_params):
        
        ############ load config file 
        self._config=config_params

        ########### transliteration model 
        self._translit_model=TransliterationModel()

        ############  corpus information
        # Each word-pair is indexed by its position in the input corpus (starting at 0)

        # list of all possible alignments for each word pair 
        # each element is a tuple [e_id,f_id], the ids are the fundamental character units
        self.wpairs_aligns=[]

        # list of weights for all possible alignments for a word pair 
        self.wpairs_weights=[]

        # list of weights for each word-pair. These are required for unsupervised training only
        self.wpairs_eword_weights=[]

        ############   parameter information

        # for each parameter (which corresponds to a point alignment), 
        # the list of occurences in of this point alignment all possible (wordpair_id,alignment_id) locations in the corpus
        # the ids are indices into the wpair_aligns list and the alignment lists that consitute it
        self.param_occurence_info=defaultdict(lambda :defaultdict(list))

        # Parameter values in the previous iteration 
        self.prev_param_values=None

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
                if fs not in self._translit_model.f_sym_id_map: 
                    self._translit_model.add_f_sym(fs)
                fs_id=  self._translit_model.f_sym_id_map[fs] 

                # create 'e' sym to id mapping
                # TODO: check if this works correctly in the unsupervised case
                if es not in self._translit_model.e_sym_id_map: 
                    self._translit_model.add_e_sym(es)
                es_id=  self._translit_model.e_sym_id_map[es] 

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

    def _generate_indic_hyperparams(self,params):

        src=params['src']
        tgt=params['tgt']

        ###### add bigram parameters dynamically

        ### add new mappings
        #for e_id, e_sym in self._translit_model.e_id_sym_map.iteritems(): 
        #    offset= langinfo.get_offset(e_sym,tgt)
        #    if offset >=langinfo.COORDINATED_RANGE_START_INCLUSIVE and offset <= langinfo.COORDINATED_RANGE_END_INCLUSIVE:
        #        f_sym_x=UnicodeIndicTransliterator.transliterate(e_sym,tgt,src)
        #        if f_sym_x in self._translit_model.f_sym_id_map: 

        #            ## add 1-2 mappings for consonants
        #            f_offset=langinfo.get_offset(f_sym_x,src)
        #            if f_offset >=0x15 and f_offset <= 0x39:  ## if consonant
        #                # consonant with aa ki maatra 
        #                f_with_aa=f_sym_x+langinfo.offset_to_char(0x3e,src)
        #                self._translit_model.add_f_sym(f_with_aa)

        #                # consonant with halant 
        #                f_with_halant=f_sym_x+langinfo.offset_to_char(0x4d,src)
        #                self._translit_model.add_f_sym(f_with_halant)

        ## initialize hyperparams 
        alpha=np.ones((len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)))

        for e_id, e_sym in self._translit_model.e_id_sym_map.iteritems(): 
            offset= langinfo.get_offset(e_sym,tgt)
            if offset >=langinfo.COORDINATED_RANGE_START_INCLUSIVE and offset <= langinfo.COORDINATED_RANGE_END_INCLUSIVE:
                f_sym_x=UnicodeIndicTransliterator.transliterate(e_sym,tgt,src)
                if f_sym_x in self._translit_model.f_sym_id_map: 
                    alpha[e_id,self._translit_model.f_sym_id_map[f_sym_x]]=params['base_measure_mapping_exists']

                    ## add 1-2 mappings for consonants
                    f_offset=langinfo.get_offset(f_sym_x,src)
                    if f_offset >=0x15 and f_offset <= 0x39:  ## if consonant
                        # consonant with aa ki maatra 
                        f_with_aa=f_sym_x+langinfo.offset_to_char(0x3e,src)
                        if f_with_aa in self._translit_model.f_sym_id_map: 
                            alpha[e_id,self._translit_model.f_sym_id_map[f_with_aa]]=params['base_measure_mapping_exists']

                        # consonant with halant 
                        f_with_halant=f_sym_x+langinfo.offset_to_char(0x4d,src)
                        if f_with_halant in self._translit_model.f_sym_id_map: 
                            alpha[e_id,self._translit_model.f_sym_id_map[f_with_halant]]=params['base_measure_mapping_exists']

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
            alpha=self._generate_indic_hyperparams(self._initparams)
            alpha_sums=np.sum(alpha, axis=1)
            self._translit_model.param_values=(alpha.transpose()/alpha_sums).transpose()

        elif  self._initmethod=='en_il_mapping':
            alpha=self._generate_en_indic_hyperparams(self._initparams)
            alpha_sums=np.sum(alpha, axis=1)
            self._translit_model.param_values=(alpha.transpose()/alpha_sums).transpose()

    def _make_sparse_prior(self):
        #if hasattr(self,'_priormethod') and 'sparse_prior' in self._priorparams:
        max_vals=np.apply_along_axis(np.max,1,self._alpha)
        t=self._alpha.T/max_vals
        self._alpha=t.T

    def _init_dirichlet_priors(self): 

        self._alpha=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])

        if hasattr(self,'_priormethod'):
            if self._priormethod=='random':
                self._alpha=np.random.rand(len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map))
                self._alpha*=self._priorparams['scale']
                print 'Scale: {}'.format(self._priorparams['scale'] )

            elif self._priormethod=='indic_mapping':
                self._alpha=self._generate_indic_hyperparams(self._priorparams)

            elif  self._initmethod=='en_il_mapping':
                self._alpha=self._generate_en_indic_hyperparams(self._priorparams)

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
        #self._make_sparse_prior()

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

            with open('{}/{}/wpairs_aligns.pickle'.format(self._config['log_dir'],iter_no) , 'w' ) as ofile:
                pickle.dump(self.wpairs_aligns,ofile)

            with open('{}/{}/wpairs_weights.pickle'.format(self._config['log_dir'],iter_no) , 'w' ) as ofile:
                pickle.dump(self.wpairs_weights,ofile)

            with open('{}/{}/wpairs_eword_weights.pickle'.format(self._config['log_dir'],iter_no) , 'w' ) as ofile:
                pickle.dump(self.wpairs_eword_weights,ofile)

            if word_triplets is not None:
                with open('{}/{}/word_triplets.pickle'.format(self._config['log_dir'],iter_no) , 'w' ) as ofile:
                    pickle.dump(word_triplets,ofile)

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


    ############################################
    ### Unsupervised training functions ########
    ############################################

    def _initialize_unsupervised_training(self,f_input_words, e_char_set): 

        # create symbol to id mappings for f (from data)
        for f_input_word in f_input_words: 
            for i,c in enumerate(f_input_word): 
                self._translit_model.add_f_sym(c)

                ## add bigram parameters statically
                if i<len(f_input_word)-1:    
                    self._translit_model.add_f_sym(c+f_input_word[i+1])

        # create symbol to id mappings for e (from given list of characters)
        for c in e_char_set:
            self._translit_model.add_e_sym(c)

        # initialize hyper parameters 
        self._init_dirichlet_priors()

        #####  initialize transliteration probabilities #####
        self._init_param_values()

        #####  previous parameter values  #####
        self.prev_param_values=np.zeros([len(self._translit_model.e_sym_id_map),len(self._translit_model.f_sym_id_map)])

    def _update_param_info(self): 
        n_new_f_chars=len(self._translit_model.f_sym_id_map)-self._translit_model.param_values.shape[1]
        if n_new_f_chars>0:
            self._translit_model.param_values=np.concatenate(   (self._translit_model.param_values,
                                                                np.zeros((self._translit_model.param_values.shape[0],n_new_f_chars))),
                                                                axis=1)

            self.prev_param_values=np.concatenate(   (self.prev_param_values,
                                                                np.zeros((self.prev_param_values.shape[0],n_new_f_chars))),
                                                                axis=1)

            self._alpha=np.concatenate(   (self._alpha,
                                            np.ones((self._alpha.shape[0],n_new_f_chars))  ),
                                        axis=1)

    #def _prepare_corpus_unsupervised(self,word_triplets, append=True): 
    #    """
    #       Does not reuse weights from previous iteration
    #    """
    #    self.wpairs_aligns=[]
    #    self.wpairs_weights=[]
    #
    #    for f,e_cands,e_prev_cands in word_triplets: 
    #        e, escore=e_cands[0]
    #        alignments=self._generate_alignments(f,e)
    #        if len(alignments)>0:
    #            self.wpairs_aligns.append(alignments)
    #            self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )
    #        else: 
    #            print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 

    #    self.wpairs_eword_weights=[1.0]*len(self.wpairs_aligns)

    #    self.param_occurence_info=defaultdict(lambda :defaultdict(list))
   
    #    # gather transliteration occurrence info
    #    for wp_idx,alignments in enumerate(self.wpairs_aligns): 
    #        for aln_idx,align in enumerate(alignments): 
    #            for es_id, fs_id in align: 
    #                self.param_occurence_info[es_id][fs_id].append([wp_idx,aln_idx])
    
    def _prepare_corpus_unsupervised(self,word_triplets,append): 
        """
           reuses weights from previous iteration
        """
        if append:
            for widx, (f,e_cands,e_prev_cands) in enumerate(word_triplets): 
                e, escore=e_cands[0]
                alignments=self._generate_alignments(f,e)
                if len(alignments)>0:
                    self.wpairs_aligns.append(alignments)
                    self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments) )  
                else: 
                    print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 
        else:                         
            for widx, (f,e_cands,e_prev_cands) in enumerate(word_triplets): 
                e,escore=e_cands[0]
                e_prev, eprevscore=e_prev_cands[0]
                if e!=e_prev:
                    alignments=self._generate_alignments(f,e)
                    if len(alignments)>0:
                        self.wpairs_aligns[widx]=alignments
                        self.wpairs_weights[widx]=[1.0/float( len(alignments) )] *  len(alignments)  
                    else: 
                        print u"No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 
                else: 
                    pass 

        self.wpairs_eword_weights=[1.0]*len(self.wpairs_aligns)

        self.param_occurence_info=defaultdict(lambda :defaultdict(list))
   
        # gather transliteration occurrence info
        for wp_idx,alignments in enumerate(self.wpairs_aligns): 
            for aln_idx,align in enumerate(alignments): 
                for es_id, fs_id in align: 
                    self.param_occurence_info[es_id][fs_id].append([wp_idx,aln_idx])

    def _normalize_topn_scores(self,translation_output): 
       
        e_scores=[ score for e, score in translation_output ]
        e_words=[ e for e, score in translation_output ]
        N=-min(e_scores)+1.0
        e_scores=[ (N+score)/(len(e_scores)*N+sum(e_scores)) for score in e_scores ]
        return zip(e_words,e_scores)
    
    def _prepare_corpus_unsupervised_topn(self,word_triplets, topn, append=True): 
        """
    
        """
        # generate alignment information
        if append:
            ## first time
            for widx, (f, e_cands, e_prev_cands) in enumerate(word_triplets): 
                for e, score in e_cands: 
                    alignments=self._generate_alignments(f,e)
                    if len(alignments)>0:
                        self.wpairs_aligns.append(alignments)
                        self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )
                        self.wpairs_eword_weights.append(score)
                    else: 
                        print u"Warning: No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 
        else: 
            ## subsequently, check for possibility of reuse 
            wpairs_aligns_prev=copy.deepcopy(self.wpairs_aligns)
            wpairs_weights_prev=copy.deepcopy(self.wpairs_weights)

            for widx, (f, e_cands, e_prev_cands) in enumerate(word_triplets): 
                e_prev_candwords=[ w for w,s in e_prev_cands]
                for cidx, (e, score) in enumerate(e_cands): 
                    if e in e_prev_candwords: 
                        ## reuse alignments & weights 
                        cidx_prev=e_prev_candwords.index(e)
                        self.wpairs_aligns[widx*topn+cidx]=wpairs_aligns_prev[widx*topn+cidx_prev]
                        self.wpairs_weights[widx*topn+cidx]=wpairs_weights_prev[widx*topn+cidx_prev]
                    else:
                        ## regenerate alignments  and weights 
                        alignments=self._generate_alignments(f,e)
                        if len(alignments)>0:
                            self.wpairs_aligns[widx*topn+cidx]=alignments
                            self.wpairs_weights[widx*topn+cidx]=[1.0/float( len(alignments) )] *  len(alignments)  
                        else: 
                            print u"Warning: No alignments from word pair: {} {}".format(''.join(f),''.join(e)).encode('utf-8') 

                    self.wpairs_eword_weights[widx*topn+cidx]=score

        # gather transliteration occurrence info
        self.param_occurence_info=defaultdict(lambda :defaultdict(list))

        for wp_idx,alignments in enumerate(self.wpairs_aligns): 
            for aln_idx,align in enumerate(alignments): 
                for es_id, fs_id in align: 
                    self.param_occurence_info[es_id][fs_id].append([wp_idx,aln_idx])
    
    def em_unsupervised_train(self,f_input_words,e_char_set,lm_model): 
        """
        """
        print "Initializing unsupervised learning" 
        decoder_params=self._config.get('decoder_params',{})
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
            prev_outputs=output_words if output_words is not None else [ [ ('',1.0)  ] ]*len(f_input_words)
            output_words=parallel_decode(self._translit_model, lm_model, f_input_words, decoder_params)
            output_words=[ [(x,1.0)] for x in output_words ]
            word_triplets=list(it.izip( f_input_words , output_words, prev_outputs ) )

            #print "Parallel Decoding for EM"
            #prev_outputs=output_words if output_words is not None else [ [ ('',1.0)  ] ]*len(f_input_words)
            #output_words=parallel_decode_topn(self._translit_model, lm_model, f_input_words, 1, decoder_params)
            #output_words=[ self._normalize_topn_scores(x) for x in output_words]
            #word_triplets=list(it.izip( f_input_words , output_words, prev_outputs ) )

            print "Preparing corpus"
            self._prepare_corpus_unsupervised(word_triplets,append)
            ## >>>

            #### >>> top-k candidate based training (without updating alignment probabilities ie not using e-step)
            #print "Parallel Decoding for EM"
            #prev_outputs=output_words if output_words is not None else [ [ ('',1.0/float(topn))  ]*topn ]*len(f_input_words)
            #output_words=parallel_decode_topn(self._translit_model, lm_model, f_input_words, topn, decoder_params)
            #output_words=[ self._normalize_topn_scores(x) for x in output_words]
            #word_triplets=list(it.izip( f_input_words , output_words, prev_outputs ) )

            ## initialize the EM training
            #print "Preparing corpus"
            #self._prepare_corpus_unsupervised_topn(word_triplets, topn, append)
            #### >>>

            ## Updating param info to account for new parameters 
            #print "Updating param info" 
            #self._update_param_info()

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
            print "Computing alignment weights" 
            self._e_step()

