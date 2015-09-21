import sys, codecs, string, itertools, re, operator, os

from collections import defaultdict
import pandas as pd
import numpy as np
import pickle

import nwalign as nw

from indicnlp import langinfo 
from indicnlp.transliterate import itrans_transliterator
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

from cfilt.transliteration.utilities import *

def restore_from_ascii(aln,lang_code):
    """
    For Indic scripts only 
    """
    assert(langinfo.SCRIPT_RANGES.has_key(lang_code))
    text=[]
    
    for c in aln:
        cn=ord(c)
        oc=''
        if c=='-': # - character for alignment
            oc=unicode(c)
        elif c=='|': # placeholder character for invalid characters 
            oc=unicode(c)
        else:
            if cn==0xff:
                cn=0x2d
            oc=unichr(cn+langinfo.SCRIPT_RANGES[lang_code][0])

        text.append(oc)

    return u''.join(text)
    

def make_ascii(text,lang_code):
    """
    Convert to ASCII due to requirement of nwalign library
    """
    assert(langinfo.SCRIPT_RANGES.has_key(lang_code))
    trans_lit_text=[]
    for c in text: 
        offset=ord(c)-langinfo.SCRIPT_RANGES[lang_code][0]
        if offset >=0 and offset <= 0x7F:
            if offset==0x2d:
                offset=0xff
            trans_lit_text.append(chr(offset))
        else: 
            trans_lit_text.append('|')   # placeholder character for invalid characters in stream
    return trans_lit_text

def align_transliterations(src_wordlist,tgt_wordlist,lang):

    for srcw,tgtw in itertools.izip(src_wordlist,tgt_wordlist): 
        # convert to ascii required by align library 
        nsrcw=''.join(make_ascii(srcw,lang) if lang in langinfo.SCRIPT_RANGES else [str(c) for c in srcw ])
        ntgtw=''.join(make_ascii(tgtw,lang) if lang in langinfo.SCRIPT_RANGES else [str(c) for c in tgtw ])
        
        # use global alignment 
        src_aln,tgt_aln=nw.global_align(nsrcw,ntgtw)
        
        # make it readable again 
        
        src_aln=restore_from_ascii(src_aln,lang) if lang in langinfo.SCRIPT_RANGES else [unicode(c) for c in src_aln ]
        tgt_aln=restore_from_ascii(tgt_aln,lang) if lang in langinfo.SCRIPT_RANGES else [unicode(c) for c in tgt_aln ]

        yield (src_aln,tgt_aln)


def create_confusion_matrix(alignments):
    conf_dict=defaultdict(dict)

    for src_aln, tgt_aln in alignments: 
        for s,t in zip(src_aln,tgt_aln): 
            conf_dict[s][t]=conf_dict[s].get(t,0)+1

    conf_df=pd.DataFrame(conf_dict,dtype=float).T.fillna(0.0)
    return  conf_df 

if __name__ == '__main__': 

    srcfname=sys.argv[1]
    tgtfname=sys.argv[2]
    lang=sys.argv[3]
    outdir=sys.argv[4]
    reranked=''
    if len(sys.argv)>=6: 
        reranked='.reranked'

    if not os.path.exists(outdir): 
        os.mkdir(outdir)

    # align words
    alignments=list(align_transliterations(read_monolingual_corpus(srcfname),read_monolingual_corpus(tgtfname),lang))
    with open(outdir+'/alignments{}.pickle'.format(reranked),'w') as align_file: 
        pickle.dump(alignments,align_file)

    # create confusion matrix 
    confusion_df=create_confusion_matrix(alignments)
    confusion_df.to_pickle(outdir+'/confusion_mat{}.pickle'.format(reranked))
