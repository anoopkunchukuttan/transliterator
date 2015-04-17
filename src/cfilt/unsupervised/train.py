import itertools as it
import pprint 
from collections import defaultdict
import numpy as np, math


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
    
    def __init__(self):
        self.wpairs_aligns=[]
        self.wpairs_weights=[]

        self.f_sym_id_map={}
        self.e_sym_id_map={}

        self.f_id_sym_map={}
        self.e_id_sym_map={}

        self.param_occurence_info=defaultdict(lambda :defaultdict(list))
        self.param_values={}


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
       
        print("Transliteration Probabilities")
        pprint.pprint(self.param_values)
        

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
            print ap
            align_offsets=list(ap)
            for i in range(1,len(align_offsets)):
                align_offsets[i]=align_offsets[i]+align_offsets[i-1]

            # insert the first starting offset             
            align_offsets.insert(0,0)            
            print align_offsets
    
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
            print align
    
            print 
        pprint.pprint(wpairs_aligns)        
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
            self.wpairs_aligns.append(alignments)
            self.wpairs_weights.append( [1.0/float( len(alignments) )] *  len(alignments)  )

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
    
    def _m_step(self): 
        
        # accumulating counts 
        for e_id in range(len(self.e_sym_id_map)): 
            for f_id in range(len(self.f_sym_id_map)): 
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
                    print self.wpairs_aligns[wp_idx][aln_idx]
                    for x in self.wpairs_aligns[wp_idx][aln_idx]:
                        print self.param_values[x[0],x[1]]

                    aln_probs.append(math.exp(sum([ math.log(self.param_values[x[0],x[1]]) if self.param_values[x[0],x[1]]==0.0  else 0.0 for x in self.wpairs_aligns[wp_idx][aln_idx] ])))
                except ValueError as e: 
                    #print "(((((( Exception Occurred )))))))))"
                    print e.message
                    #print e.args
           
            norm_factor=sum(aln_probs)
            for aln_idx in xrange(len(self.wpairs_aligns[wp_idx])): 
                self.wpairs_weights[wp_idx][aln_idx]=aln_probs[aln_idx]/norm_factor

    def em_train(self,word_pairs): 
        """
        """
    
        self._create_alignment_database(word_pairs)
        self._initialize_parameter_structures()
    
        print self.param_occurence_info 

        MAX_ITER=10
        niter=0
        while(True):
            # M-step
            self._m_step()
            niter+=1

            self.print_params()
            print '=== Iteration {} completed ==='.format(niter)

            # check for end of process 
            if niter>=MAX_ITER:
                break
    
            # E-step 
            self._e_step()

        print "Final paramters"
        for e_id in range(len(self.e_sym_id_map)): 
            for f_id in range(len(self.f_sym_id_map)): 
                print "P({}|{})={}".format(self.f_id_sym_map[f_id],self.e_id_sym_map[e_id],self.param_values[e_id,f_id]) 































