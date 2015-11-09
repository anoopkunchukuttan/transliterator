from indicnlp import loader
from indicnlp import langinfo
from indicnlp.script.indic_scripts import * 
import numpy as np
import pandas as pd
import codecs,sys 

from cfilt.transliteration.analysis import align

def equal(v1,v2): 
    return 0.0 if  np.sum( xor_vectors(v1, v2)) > 0  else 1.0

def dice(v1,v2):
    dotprod=2*float(np.dot( v1, v2.T ))
    return dotprod/float(len(v1)+len(v2))

def jaccard(v1,v2):
    dotprod=float(np.dot( v1, v2.T ))
    return dotprod/float(len(v1)+len(v2)-dotprod)

def cosine(v1,v2):
    dotprod=float(np.dot( v1, v2.T ))
    norm1=float(np.dot( v1, v1.T ))
    norm2=float(np.dot( v2, v2.T ))
    return ((dotprod)/(np.sqrt(norm1*norm2)+0.00001))

def dotprod(v1,v2): 
    return float(np.dot( v1, v2.T ))

def sim1(v1,v2,base=5.0): 
    return np.power(base,dotprod(v1,v2)) 

def sim2(v1,v2,base=5.0): 

    ## Weight vector
    # phonetic_weight_vector=np.array([
    #     60.0,60.0,60.0,60.0,60.0,60.0,
    #     1.0,1.0,
    #     30.0,30.0,30.0,
    #     40.0,40.0,
    #     50.0,50.0,50.0,50.0,50.0,
    #     40.0,40.0,40.0,40.0,40.0,
    #     5.0,5.0,
    #     10.0,10.0,
    #     10.0,10.0,
    # ])
    
    phonetic_weight_vector=np.array([
        #6.0,6.0,6.0,6.0,6.0,6.0,
        0.01,0.01,0.01,0.01,0.01,0.01,
        0.1,0.1,
        3.0,3.0,3.0,
        4.0,4.0,
        5.0,5.0,5.0,5.0,5.0,
        4.0,4.0,4.0,4.0,4.0,
        0.5,0.5,
        1.0,1.0,
        1.0,1.0,
    ])

    v1_weighted=np.multiply(v1,phonetic_weight_vector)
    dotprod=float(np.dot( v1_weighted, v2.T ))
    #return np.power(base,dotprod) 
    return dotprod

def accumulate_vectors(v1,v2): 
    """
    not a commutative operation
    """

    if is_consonant(v1) and is_halant(v2): 
        v1[PVIDX_BT_HALANT]=1
        return v1
    elif is_consonant(v1) and is_nukta(v2): 
        v1[PVIDX_BT_NUKTA]=1
        return v1
    elif is_consonant(v1) and is_dependent_vowel(v2): 
        return or_vectors(v1,v2)
    elif is_anusvaar(v1) and is_consonant(v2): 
        return or_vectors(v1,v2)
    else: 
        return invalid_vector()


def create_similarity_matrix(sim_func,slang,tlang,normalize=True):

    dim=langinfo.COORDINATED_RANGE_END_INCLUSIVE-langinfo.COORDINATED_RANGE_START_INCLUSIVE+1    
    sim_mat=np.zeros((dim,dim))    

    for offset1 in xrange(langinfo.COORDINATED_RANGE_START_INCLUSIVE, langinfo.COORDINATED_RANGE_END_INCLUSIVE+1): 
        v1=get_phonetic_feature_vector(offset_to_char(offset1,slang),slang)
        for offset2 in xrange(langinfo.COORDINATED_RANGE_START_INCLUSIVE, langinfo.COORDINATED_RANGE_END_INCLUSIVE+1): 
            v2=get_phonetic_feature_vector(offset_to_char(offset2,tlang),tlang)
            sim_mat[offset1,offset2]=sim_func(v1,v2)

    if normalize: 
        sums=np.sum(sim_mat, axis=1)
        sim_mat=(sim_mat.transpose()/sums).transpose()

    return sim_mat

def score_phonetic_alignment(srcw,tgtw,slang,tlang,sim_matrix,mismatch_p=-0.2,gap_start_p=-0.05,gap_extend_p=0.0):

    score_mat=np.zeros((len(srcw)+1,len(tgtw)+1))
    soff=[ get_offset(c,slang) for c in srcw ]
    toff=[ get_offset(c,tlang) for c in tgtw ]

    score_mat[:,0]=np.array([si*gap_start_p for si in xrange(score_mat.shape[0])])
    score_mat[0,:]=np.array([ti*gap_start_p for ti in xrange(score_mat.shape[1])])

    for si,sc in enumerate(soff,1): 
        for ti,tc in enumerate(toff,1): 
            score_mat[si,ti]= max(
                    score_mat[si-1,ti-1]+(sim_matrix[sc,tc] if ( sc>=0 and tc>=0 and sc<sim_matrix.shape[0] and tc<sim_matrix.shape[1]) else mismatch_p),
                    score_mat[si,ti-1]+gap_start_p,
                    score_mat[si-1,ti]+gap_start_p,
                )
    return score_mat[-1,-1]/float(max(len(srcw),len(tgtw)))

def iterate_phrase_table(phrase_table_fname): 
    with codecs.open(phrase_table_fname,'r','utf-8') as phrase_table: 
        for line in phrase_table: 
            yield line.split(u'|||')

def add_phonetic_alignment_phrase_table(phrase_table_fname,
                                        out_phrase_table_fname,
                                        src_lang,tgt_lang,
                                        similarity_metrics=[(sim1,True)],
                                        mismatch_p=0.0,gap_start_p=0.0): 

    sim_mats=[create_similarity_matrix(metric_func,src_lang,tgt_lang,normalize) for metric_func, normalize in similarity_metrics]

    with codecs.open(out_phrase_table_fname,'w','utf-8') as out_phrase_table: 
        for fields in iterate_phrase_table(phrase_table_fname): 
            #assert(len(fields)>=5))
            src_words= fields[0].strip().split(u' ')
            tgt_words= fields[1].strip().split(u' ')
            feat_values=[ float(x) for x in fields[2].strip().split(u' ') ]

            for (metric,_),sim_mat in zip(similarity_metrics,sim_mats): 
                score=score_phonetic_alignment(src_words,tgt_words,src_lang,tgt_lang,sim_mat,
                        0.0 if metric==equal else mismatch_p,0.0 if metric==equal else gap_start_p)
                feat_values.append(score)

            fields[2]=u' '+u' '.join([str(x) for x in feat_values])+u' '

            out_phrase_table.write(u'|||'.join(fields))

            ## lexical weighting - all wrong
            #alignments=[ [ int(x) for x in ap.strip().split(u'-') ]  for ap in fields[3].strip().split(u' ') ]
            #c_tpos={}
            #for spos, tpos in alignments: 
            #    c_tpos[tpos]=c_tpos.get(tpos,0.0)+1.0
            #score=0.0
            #for si, sw in enumerate(src_words): 
            #    term=0.0
            #    c=0.0
            #    for ti, tw in enumerate(tgt_words): 
            #        if [si,ti] in alignments: 
            #            c+=1.0
            #            term+=score_phonetic_alignment(src_words[si],tgt_words[ti],src_lang,tgt_lang,sim_mat)
            #    term=0.0 if c==0.0 else term/c
            #    score*=term

            # average 
            #score=0.0
            #fo


if __name__ == '__main__': 
    loader.load()
    #create_similarity_matrix(sim1,'hi','pa')
    add_phonetic_alignment_phrase_table(*sys.argv[1:])

