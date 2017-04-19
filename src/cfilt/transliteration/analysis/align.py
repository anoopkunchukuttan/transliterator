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

import nwalign as nw

from indicnlp import langinfo 
from indicnlp.transliterate import itrans_transliterator
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from indicnlp.script import indic_scripts as isc

from cfilt.transliteration.utilities import *

def restore_from_ascii_char(c,lang_code):

    if langinfo.SCRIPT_RANGES.has_key(lang_code):
        cn=ord(c)
        oc=''
        if c=='-': # - character for alignment
            oc=unicode(c)
        elif c=='|': # placeholder character for invalid characters 
            oc=unicode(c)
        else:
            if cn==0xff: # '-' character 
                cn=0x2d
            elif cn==0xfe: ## space 
                cn=0x20
            elif cn==0xfd: ## line feed
                cn=0x0a
            elif cn==0xfc: ## carriage return 
                cn=0x0d
            elif cn==0xfb: ## tab
                cn=0x09
            oc=unichr(cn+langinfo.SCRIPT_RANGES[lang_code][0])

        return oc
   
    elif lang_code=='ar': 
        pass 
    else: 
        return unicode(c) 

def restore_from_ascii(aln,lang_code):

    text=[]
    
    for c in aln:
        text.append(restore_from_ascii_char(c,lang_code))

    return u''.join(text)

def make_ascii_char(c,lang_code): 

    PLACEHOLDER='|'

    if langinfo.SCRIPT_RANGES.has_key(lang_code):
        offset=ord(c)-langinfo.SCRIPT_RANGES[lang_code][0]
        result=None 
        if offset >=0 and offset <= 0x7F:
            if offset==0x2d:
                offset=0xff
            elif offset==0x20:
                offset=0xfe
            elif offset==0x0a:
                offset=0xfd
            elif offset==0x0d:
                offset=0xfc
            elif offset==0x09:
                offset=0xfb
            result = chr(offset)
        else: 
            result = PLACEHOLDER   # placeholder character for invalid characters in stream
    elif lang_code == 'ar': 
        pass 
    else: 
        return  str(c) if (ord(c) < 127) else PLACEHOLDER 

    return result 

def make_ascii(text,lang_code):
    """
    Convert to ASCII due to requirement of nwalign library
    """
    trans_lit_text=[]
    for c in text: 
        trans_lit_text.append(make_ascii_char(c,lang_code))
    return trans_lit_text

#def align_transliterations(src_wordlist,tgt_wordlist,lang):
#
#    for srcw,tgtw in itertools.izip(src_wordlist,tgt_wordlist): 
#        # convert to ascii required by align library 
#        nsrcw=''.join(make_ascii(srcw,lang) if lang in langinfo.SCRIPT_RANGES else [str(c) for c in srcw ])
#        ntgtw=''.join(make_ascii(tgtw,lang) if lang in langinfo.SCRIPT_RANGES else [str(c) for c in tgtw ])
#        
#        # use global alignment 
#        src_aln,tgt_aln=nw.global_align(nsrcw,ntgtw)
#        
#        # make it readable again 
#        
#        src_aln=restore_from_ascii(src_aln,lang) if lang in langinfo.SCRIPT_RANGES else [unicode(c) for c in src_aln ]
#        tgt_aln=restore_from_ascii(tgt_aln,lang) if lang in langinfo.SCRIPT_RANGES else [unicode(c) for c in tgt_aln ]
#
#        yield (src_aln,tgt_aln)

def align_transliterations(src_wordlist,tgt_wordlist,lang):

    for srcw,tgtw in itertools.izip(src_wordlist,tgt_wordlist): 
        # convert to ascii required by align library 
        nsrcw=''.join(make_ascii(srcw,lang))
        ntgtw=''.join(make_ascii(tgtw,lang))
        
        # use global alignment 
        src_aln,tgt_aln=nw.global_align(nsrcw,ntgtw)
        
        # make it readable again 
        
        src_aln=restore_from_ascii(src_aln,lang) 
        tgt_aln=restore_from_ascii(tgt_aln,lang) 

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

def score_phonetic_alignment(srcw,tgtw,slang,tlang,sim_matrix_path,gap_start_p=-1.0,gap_extend_p=-1.0):

    # convert to ascii required by align library 
    nsrcw=''.join(make_ascii(srcw,slang) if slang in langinfo.SCRIPT_RANGES else [str(c) for c in srcw ])
    ntgtw=''.join(make_ascii(tgtw,tlang) if tlang in langinfo.SCRIPT_RANGES else [str(c) for c in tgtw ])
    
    ## use global alignment 
    src_aln,tgt_aln=nw.global_align(nsrcw,ntgtw,matrix=sim_matrix_path, gap_open=gap_start_p, gap_extend=gap_extend_p)
    return nw.score_alignment(src_aln,tgt_aln,matrix=sim_matrix_path, gap_open=gap_start_p, gap_extend=gap_extend_p)

    #src_aln,tgt_aln=nw.global_align(nsrcw,ntgtw,gap_open=gap_start_p, gap_extend=gap_extend_p)
    #return nw.score_alignment(src_aln,tgt_aln,gap_open=gap_start_p, gap_extend=gap_extend_p)



################  ERROR ANALYSIS ######

class CharClassIdentifier(object): 

    def __init__(self): 
        
        self.vowel_set={}
        self.consonant_set={}

        ## vowel set 
        self.vowel_set['en']=set(['A','E','I','O','U'])
        #for lang in [ 'pl', 'cs', 'sl', 'sk' ]:
        #    self.vowel_set[lang]=set(['A','E','I','O','U'])  # add vowels 

        ##consonant set 
        for lang in [ 'en']:
            self.consonant_set['en']=set([ unichr(i) for i in range(ord('A'),ord('Z')) ]) \
                                        - self.vowel_set['en']

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

cci=CharClassIdentifier()

def read_align_count_file(align_count_fname):
    return pd.read_csv(align_count_fname,header=0,index_col=0,sep=',',encoding='utf-8')

def char_error_rate(a_df): 
    """
     a_df: align count dataframe
    """
    return a_df[a_df.ref_char!=a_df.out_char]['count'].sum()/a_df['count'].sum()

def ins_error_rate(a_df): 
    """
    rate of unnecessary insertion 
    a_df: align count dataframe
    """
    return a_df[a_df.ref_char=='-']['count'].sum()/a_df['count'].sum()
  
def del_error_rate(a_df): 
    """
    rate of unnecessary deletion
    a_df: align count dataframe
    """
    return a_df[a_df.out_char=='-']['count'].sum()/a_df['count'].sum()
  
def sub_error_rate(a_df): 
    """
    rate of substitution errors 
    a_df: align count dataframe
    """
    return char_error_rate(a_df) - ( ins_error_rate(a_df) + del_error_rate(a_df) )
  
def vowel_error_rate(a_df,lang): 
    """
     a_df: align count dataframe
    """
    sel_rows=filter(lambda r: cci.is_vowel(r[1]['ref_char'],lang), a_df.iterrows())
    a_df=x=pd.DataFrame([x[1] for x in sel_rows])
    return a_df[a_df.ref_char!=a_df.out_char]['count'].sum()/a_df['count'].sum()

def consonant_error_rate(a_df,lang): 
    """
     a_df: align count dataframe
    """
    sel_rows=filter(lambda r: cci.is_consonant(r[1]['ref_char'],lang), a_df.iterrows())
    a_df=x=pd.DataFrame([x[1] for x in sel_rows])
    return a_df[a_df.ref_char!=a_df.out_char]['count'].sum()/a_df['count'].sum()

#def err_dist(a_df,lang): 
#    """
#     a_df: align count dataframe
#    """
#    rows=[x[1] for x in a_df.iterrows()]
#    v_err_rows=filter(lambda r: cci.is_vowel(r['ref_char'],lang) and r['ref_char']!=r['out_char'], rows )
#    c_err_rows=filter(lambda r: cci.is_consonant(r['ref_char'],lang) and r['ref_char']!=r['out_char'], rows )
#    a_df=x=pd.DataFrame([x[1] for x in sel_rows])
#    return a_df[a_df.ref_char!=a_df.out_char]['count'].sum()/a_df['count'].sum()

if __name__ == '__main__': 

    #srcfname=sys.argv[1]
    #tgtfname=sys.argv[2]
    #lang=sys.argv[3]
    #outdir=sys.argv[4]
    #reranked=''
    #if len(sys.argv)>=6: 
    #    reranked='.reranked'

    #if not os.path.exists(outdir): 
    #    os.mkdir(outdir)

    ## align words
    #alignments=list(align_transliterations(read_monolingual_corpus(srcfname),read_monolingual_corpus(tgtfname),lang))
    #with open(outdir+'/alignments{}.pickle'.format(reranked),'w') as align_file: 
    #    pickle.dump(alignments,align_file)

    ## create confusion matrix 
    #aligncount_df,confusion_df=gather_alignment_info(alignments)
    #confusion_df.to_csv(outdir+'/confusion_mat{}.csv'.format(reranked),encoding='utf-8')
    #aligncount_df.to_csv(outdir+'/alignment_count{}.csv'.format(reranked),encoding='utf-8')
   
    from indicnlp import loader
    loader.load()

    #a_df=read_align_count_file('/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_official/2_multilingual/onehot_shared/multi-conf/outputs/022_analysis_en-bn/alignment_count.csv')
    #print char_error_rate(a_df)
    #print vowel_error_rate(a_df,'bn')
    #print consonant_error_rate(a_df,'bn')


    a_df=read_align_count_file('/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_indic/2_multilingual/onehot_shared/multi-conf/outputs/010_analysis_bn-hi/alignment_count.csv')
    char_error_rate(a_df)
    #print ins_error_rate(a_df)
    #print del_error_rate(a_df)
    #print sub_error_rate(a_df)
    #print vowel_error_rate(a_df,'hi')
    #print consonant_error_rate(a_df,'hi')

#src=hi
#tgt=kn
#
#
#### 20_4_4
##python align.py /home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/$src-$tgt/test.$tgt /home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb/20_4_4/$src-$tgt/evaluation/test.reranked.$tgt $tgt /home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb/20_4_4/$src-$tgt/evaluation/   reranked
#python debug.py /home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb/20_4_4/$src-$tgt/evaluation/confusion_mat.reranked.pickle /home/development/anoop/experiments/unsupervised_transliterator/experiments/nonparallel/pb/20_4_4/$src-$tgt/evaluation/confusion_mat.reranked.png $tgt
#
#### supervised
##python align.py /home/development/anoop/experiments/unsupervised_transliterator/data/nonparallel/pb/$src-$tgt/test.$tgt /home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/pb/$src-$tgt/evaluation/test.$tgt $tgt /home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/pb/$src-$tgt/evaluation/
##
##python debug.py /home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/pb/$src-$tgt/evaluation//confusion_mat.pickle /home/development/anoop/experiments/unsupervised_transliterator/experiments/parallel_news_2015_indic/pb/$src-$tgt/evaluation/confusion_mat.png $tgt
