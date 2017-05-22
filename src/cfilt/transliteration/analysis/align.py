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

import sys, codecs, string, itertools, re, operator, os

from collections import defaultdict
import pandas as pd
import numpy as np
import pickle

from indicnlp import langinfo 
from indicnlp.transliterate import itrans_transliterator
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from indicnlp.script import indic_scripts as isc

from cfilt.transliteration.utilities import *
import slavic_characters


def align_nw(refw,outw): 
    """
    Implementation of Needleman-Wunch alignment algorithm
    """

    ## GAP symbol 
    GAP_SYM=u'~'

    ## Scores 
    MATCH_SCORE=1
    MISMATCH_SCORE=-1
    GAP_SCORE=-1

    ## operations 
    MATCH_OP=0
    DEL_OP=1
    INS_OP=2

    ## char to index mappings 
    c2imap=defaultdict(lambda:len(c2imap))
    i2cmap={}

    ## make vocab
    c2imap[GAP_SYM]
    refa=[-1] + [ c2imap[c] for c in refw ]        
    outa=[-1] + [ c2imap[c] for c in outw ]        
    for c,i in c2imap.iteritems(): 
        i2cmap[i]=c

    ## data structures 
    score_mat=np.zeros((len(refa),len(outa)))
    op_mat=np.zeros_like(score_mat)

    ## initialization 

    ## deletion
    for i in range(0,score_mat.shape[0]):
        score_mat[i,0]=i*GAP_SCORE
        op_mat[i,0]=DEL_OP

    ## insertion
    for j in range(0,score_mat.shape[1]):
        score_mat[0,j]=j*GAP_SCORE
        op_mat[0,j]=INS_OP

    ## fill the table 
    for i in range(1,score_mat.shape[0]):
        for j in range(1,score_mat.shape[1]):
            sarr = [  ## NOTE: order of elements in this array is important 
                    score_mat[i-1,j-1] + (MATCH_SCORE if refa[i]==outa[j] else MISMATCH_SCORE),  ## match (1) or substitute (0)
                    score_mat[i-1,j]+GAP_SCORE,   ## deletion  
                    score_mat[i,j-1]+GAP_SCORE,   ## insertion
                   ]
            score_mat[i,j]=np.max(sarr)
            op_mat[i,j]=np.argmax(sarr)
                
    ## backtrack 
    
    i=score_mat.shape[0]-1
    j=score_mat.shape[1]-1
    
    rev_refa_aln=[]
    rev_outa_aln=[]

    while( i>=0 and j>=0 ): 
        if op_mat[i,j]==MATCH_OP: 
            rev_refa_aln.append(refa[i])
            rev_outa_aln.append(outa[j])
            i-=1
            j-=1
        elif op_mat[i,j]==DEL_OP:             
            rev_refa_aln.append(refa[i])
            rev_outa_aln.append(c2imap[GAP_SYM])
            i-=1
        elif op_mat[i,j]==INS_OP:           
            rev_refa_aln.append(c2imap[GAP_SYM])
            rev_outa_aln.append(outa[j])
            j-=1

    ref_aln= [ i2cmap[i] for i in reversed(rev_refa_aln[:-1]) ]
    out_aln= [ i2cmap[i] for i in reversed(rev_outa_aln[:-1]) ]

    #print 'Score Mat'
    #print score_mat
    #print 
    #print 'Op Mat'
    #print op_mat
    #print 
    #print 'Score: {}'.format(score_mat[-1,-1])
    #print 
    #print u' '.join(ref_aln)
    #print u' '.join(out_aln)

    return (ref_aln,out_aln)

def align_transliterations(src_wordlist,tgt_wordlist,lang):

    for srcw,tgtw in itertools.izip(src_wordlist,tgt_wordlist): 
        
        # use global alignment 
        src_aln,tgt_aln=align_nw(srcw,tgtw)

        yield (src_aln,tgt_aln)

def create_confusion_matrix(alignments):
    conf_dict=defaultdict(dict)

    for src_aln, tgt_aln in alignments: 
        for s,t in zip(src_aln,tgt_aln): 
            conf_dict[s][t]=conf_dict[s].get(t,0)+1

    conf_df=pd.DataFrame(conf_dict,dtype=float).T.fillna(0.0)
    return  conf_df 

def gather_alignment_info(alignments):
    conf_dict=defaultdict(dict)
    alignment_counts=[]

    for src_aln, tgt_aln in alignments: 
        for s,t in zip(src_aln,tgt_aln): 
            conf_dict[s][t]=conf_dict[s].get(t,0)+1

    for s in conf_dict.keys(): 
        for t in conf_dict[s].keys(): 
            alignment_counts.append([s,t,conf_dict[s][t]])
    
    alnc_df=pd.DataFrame(alignment_counts,columns=['ref_char','out_char','count'],dtype=float)
    conf_df=pd.DataFrame(conf_dict,dtype=float).T.fillna(0.0)

    return  alnc_df,conf_df 

def save_analysis_artifacts(reffname, outfname, tgtlang, outdir):

    if not os.path.exists(outdir): 
        os.mkdir(outdir)

    # align words
    alignments=list(align_transliterations(read_monolingual_corpus(reffname),read_monolingual_corpus(outfname),tgtlang))

    # create confusion matrix 
    aligncount_df,confusion_df=gather_alignment_info(alignments)

    ## save artificats 

    ## per sequence alignments 
    with open(outdir+'/alignments.pickle','w') as align_file: 
        pickle.dump(alignments,align_file)

    ## confusion matric 
    confusion_df.to_csv(outdir+'/confusion_mat.csv',encoding='utf-8')
    confusion_df.to_pickle(outdir+'/confusion_mat.pickle')

    ## alignment pair counts         
    aligncount_df.to_csv(outdir+'/alignment_count.csv',encoding='utf-8')

################  ERROR ANALYSIS ######

GAP_SYM=u'~'

class CharClassIdentifier(object): 

    def __init__(self): 
        
        self.vowel_set={}
        self.consonant_set={}

        ## vowel set 
        self.vowel_set['en']=set(['A','E','I','O','U'])
        for lang in [ 'pl', 'cs', 'sl', 'sk' ]:
            self.vowel_set[lang]=set(slavic_characters.latin_vowels)  

        ##consonant set 
        for lang in [ 'en']:
            self.consonant_set['en']=set([ unichr(i) for i in range(ord('A'),ord('Z')) ]) \
                                        - self.vowel_set['en']
        for lang in [ 'pl', 'cs', 'sl', 'sk' ]:
            self.consonant_set[lang]=set(slavic_characters.latin_consonants)  

    def is_supported_language(self,lang): 
        return isc.is_supported_language(lang) or lang in self.vowel_set

    def is_vowel(self,c,lang): 
        if isc.is_supported_language(lang): 
            return isc.is_vowel(isc.get_phonetic_feature_vector(c,lang))
        elif lang in self.vowel_set: 
            return c in self.vowel_set[lang]
        else: 
            raise Exception('Language no supported. Add list of vowels for this language')

    def is_consonant(self,c,lang): 
        if isc.is_supported_language(lang): 
            return isc.is_consonant(isc.get_phonetic_feature_vector(c,lang))
        elif lang in self.consonant_set: 
            return c in self.consonant_set[lang]
        else: 
            raise Exception('Language no supported. Add list of consonants for this language')

    def get_char_type(self,c,lang):         

        if self.is_vowel(c,lang): 
            return 'V'
        elif self.is_consonant(c,lang): 
            return 'C'
        else: 
            return 'O'

cci=CharClassIdentifier()

def read_align_count_file(align_count_fname):
    return pd.read_csv(align_count_fname,header=0,index_col=0,sep=',',encoding='utf-8')

def char_error_rate(a_df): 
    """
     a_df: align count dataframe
    """
    return a_df[a_df.ref_char!=a_df.out_char]['count'].sum()/a_df['count'].sum()

def char_error_count(a_df): 
    """
     a_df: align count dataframe
    """
    return a_df[a_df.ref_char!=a_df.out_char]['count'].sum()

def ins_error_rate(a_df): 
    """
    rate of unnecessary insertion 
    a_df: align count dataframe
    """
    return a_df[a_df.ref_char==GAP_SYM]['count'].sum()/a_df['count'].sum()
  
def del_error_rate(a_df): 
    """
    rate of unnecessary deletion
    a_df: align count dataframe
    """
    return a_df[a_df.out_char==GAP_SYM]['count'].sum()/a_df['count'].sum()
  
def sub_error_rate(a_df): 
    """
    rate of substitution errors 
    a_df: align count dataframe
    """
    return char_error_rate(a_df) - ( ins_error_rate(a_df) + del_error_rate(a_df) )
  
def vowel_error_rate(a_df,lang): 
    """
     a_df: align count dataframe

     return vowel error rate
    """
    #sel_rows=filter(lambda r: cci.is_vowel(r[1]['ref_char'],lang), a_df.iterrows())
    #a_df=x=pd.DataFrame([x[1] for x in sel_rows])
    #return a_df[a_df.ref_char!=a_df.out_char]['count'].sum()/a_df['count'].sum()

    rows=[x[1] for x in a_df.iterrows()]

    ## deletion and substition errors 
    vds_err_df=pd.DataFrame(filter(lambda r: cci.is_vowel(r['ref_char'],lang) and r['ref_char']!=r['out_char'], rows ) )
    ## insertion errors 
    vi_err_df=pd.DataFrame(filter(lambda r: r['ref_char']==GAP_SYM and cci.is_vowel(r['out_char'],lang) , rows ) )
    ## all vowel rows
    all_vowel_df=pd.DataFrame(filter(lambda r: cci.is_vowel(r['ref_char'],lang), rows))

    ## total vowel errors 
    n_vow_err =vds_err_df['count'].sum() + vi_err_df['count'].sum()
    ## total vowels occurences to consider for error calcuation (ins,del and subst)
    n_vow=all_vowel_df['count'].sum() + vi_err_df['count'].sum()

    return n_vow_err/n_vow

def consonant_error_rate(a_df,lang): 
    """
     a_df: align count dataframe

     return tuple consonant error rate
    """
    #sel_rows=filter(lambda r: cci.is_consonant(r[1]['ref_char'],lang), a_df.iterrows())
    #a_df=x=pd.DataFrame([x[1] for x in sel_rows])
    #return a_df[a_df.ref_char!=a_df.out_char]['count'].sum()/a_df['count'].sum()

    rows=[x[1] for x in a_df.iterrows()]

    ## deletion and substition errors 
    cds_err_df=pd.DataFrame(filter(lambda r: cci.is_consonant(r['ref_char'],lang) and r['ref_char']!=r['out_char'], rows ) )
    ## insertion errors 
    ci_err_df=pd.DataFrame(filter(lambda r: r['ref_char']==GAP_SYM and cci.is_consonant(r['out_char'],lang) , rows ) )
    ## all vowel rows
    all_cons_df=pd.DataFrame(filter(lambda r: cci.is_consonant(r['ref_char'],lang), rows))

    ## total consonant errors 
    n_cons_err =cds_err_df['count'].sum() + ci_err_df['count'].sum()
    ## total consonant occurences to consider for error calcuation (ins,del and subst)
    n_cons=all_cons_df['count'].sum() + ci_err_df['count'].sum()

    return n_cons_err/n_cons

def err_dist(a_df,lang): 
    """
     a_df: align count dataframe

     returns tuple (n_vow_err,n_cons_err,n_oth_err,n_tot_err)
    """
    rows=[x[1] for x in a_df.iterrows()]

    ## deletion and substition errors 
    vds_err_df=pd.DataFrame(filter(lambda r: cci.is_vowel(r['ref_char'],lang) and r['ref_char']!=r['out_char'], rows ) )
    cds_err_df=pd.DataFrame(filter(lambda r: cci.is_consonant(r['ref_char'],lang) and r['ref_char']!=r['out_char'], rows ) )

    ## insertion errors 
    vi_err_df=pd.DataFrame(filter(lambda r: r['ref_char']==GAP_SYM and cci.is_vowel(r['out_char'],lang) , rows ) )
    ci_err_df=pd.DataFrame(filter(lambda r: r['ref_char']==GAP_SYM and cci.is_consonant(r['out_char'],lang) , rows ) )

    ## vowel errors 
    n_vow_err =vds_err_df['count'].sum() + vi_err_df['count'].sum()
    n_cons_err=cds_err_df['count'].sum() + ci_err_df['count'].sum()

    ## total errors
    n_tot_err=a_df[a_df.ref_char!=a_df.out_char]['count'].sum()

    # other errors 
    n_oth_err=n_tot_err - (n_vow_err+n_cons_err)

    return (n_vow_err,n_cons_err,n_oth_err,n_tot_err)

if __name__ == '__main__': 

    from indicnlp import loader
    loader.load()

    #reffname=sys.argv[1]
    #outfname=sys.argv[2]
    #tgtlang=sys.argv[3]
    #outdir=sys.argv[4]

    #if not os.path.exists(outdir): 
    #    print outdir
    #    os.mkdir(outdir)

    #save_analysis_artifacts(reffname, outfname, tgtlang, outdir)

    #a_df=read_align_count_file('/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_official/2_multilingual/onehot_shared/multi-conf/outputs/022_analysis_en-bn/alignment_count.csv')
    #print char_error_rate(a_df)
    #print vowel_error_rate(a_df,'bn')
    #print consonant_error_rate(a_df,'bn')


    #a_df=read_align_count_file('/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_indic/2_multilingual/onehot_shared/multi-conf/outputs/010_analysis_bn-hi/alignment_count.csv')
    #char_error_rate(a_df)
    #print ins_error_rate(a_df)
    #print del_error_rate(a_df)
    #print sub_error_rate(a_df)
    #print vowel_error_rate(a_df,'hi')
    #print consonant_error_rate(a_df,'hi')

