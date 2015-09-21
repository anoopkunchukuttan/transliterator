import string, codecs, re, os, sys, itertools, functools, operator
import numpy
from srilm import * 
from cfilt.transliteration.utilities import *

#def rerank_topn_moses_format(infname,outfname,n,lm_fname,lm_order): 
#    """
#    Reranks the n-best list from the base SMT system with language model and the baseline SMT system scores
#    """
#
#    # read the language model 
#    lm_model=load_lm_model(lm_fname,lm_order)
#    lm_model_bigram=load_lm_model(lm_fname,2)
#
#    def parse_line(line):
#        fields=[ x.strip() for x in  line.strip().split('|||') ]
#        fields[0]=int(fields[0])
#        fields[3]=float(fields[3])
#        return fields
#
#    def rescore_candidate(fields):
#        # desegment
#        sentence=fields[1]
#        #score with language model and the baseline SMT model 
#        score = fields[3] - sent_score(sentence,lm_model_bigram) +  sent_score(sentence,lm_model)              
#        fields[3]=score
#
#        return fields
#
#    # desegment and rank the candidates by the LM probability
#    with codecs.open(infname,'r','utf-8') as infile: 
#        with codecs.open(outfname,'w','utf-8') as outfile: 
#            for sent_no, lines in itertools.groupby(itertools.imap(parse_line,iter(infile)),key=operator.itemgetter(0)) :
#                ranked_candidates=list(itertools.imap(rescore_candidate, itertools.islice(lines,0,n)))
#                ranked_candidates.sort(key=operator.itemgetter(3),reverse=True)
#    
#                for cand in ranked_candidates: 
#                    outfile.write(u'{} ||| {} ||| {} ||| {}\n'.format(*cand))

def sent_score(sentence,lm_model):
    return getSentenceProb(lm_model,sentence.encode('utf-8'),len(sentence.split(' ')))/LOG_E_BASE10

### Methods for parsing n-best lists
def parse_line(line):
    """
        line in n-best file 
        return list of fields
    """
    fields=[ x.strip() for x in  line.strip().split('|||') ]
    fields[0]=int(fields[0])
    fields[3]=float(fields[3])
    return fields

def iterate_nbest_list(nbest_fname): 
    """
        nbest_fname: moses format nbest file name 
        return iterator over tuple of sent_no, list of n-best candidates

    """

    infile=codecs.open(nbest_fname,'r','utf-8')
    
    for sent_no, lines in itertools.groupby(iter(infile),key=lambda x:parse_line(x)[0]):
        parsed_lines = [ parse_line(line) for line in lines ]
        yield((sent_no,parsed_lines))

    infile.close()

class ReRanker(object): 

    def __init__(self,lm_model,lm_model_org):
        self._lm_model=lm_model
        self._lm_model_org=lm_model_org 
    
    def rerank_candidates(self,candidates): 
        new_candidates=[ (x, score - sent_score(''.join(x),self._lm_model_org) + sent_score(''.join(x),self._lm_model)) \
                        for x, score in candidates]
        new_candidates.sort(key=operator.itemgetter(1),reverse=True)
        return new_candidates

if __name__ == '__main__': 

    #infname=sys.argv[1]
    #outfname=sys.argv[2]
    #n=int(sys.argv[3])
    #lm_fname=sys.argv[4]
    #order=int(sys.argv[5])

    #lm_model=load_lm_model(lm_fname,order)
    #lm_model_org=load_lm_model(lm_fname,2)

    #r=ReRanker(lm_model,lm_model_org)

    #with codecs.open(outfname,'w','utf-8') as outfile: 
    #    for sent_no, candidates in iterate_nbest_list(infname): 
    #        ranked_candidates=r.rerank_candidates([ (x[1], x[3]) for x in candidates] )
    #        for cand in ranked_candidates: 
    #            outfile.write(u'{} ||| {} ||| {} ||| {}\n'.format(sent_no,cand[0],u'',cand[1]))

    import pickle 
    with open(sys.argv[1],'r') as infile:
        with codecs.open(sys.argv[2],'w','utf-8') as outfile: 
            triplets=pickle.load(infile)
            for sent_no, (_, outputs, _) in enumerate(triplets):
                #for cand in outputs: 
                #    outfile.write(u'{} ||| {} ||| {} ||| {}\n'.format(sent_no,u''.join(cand[0]),u'',cand[1]))
                #outfile.write(u'{} ||| {} ||| {} ||| {}\n'.format(sent_no,u''.join(outputs[0][0]),u'',outputs[0][1]))
                outfile.write(u'{}\n'.format(u''.join(outputs[0][0])))
