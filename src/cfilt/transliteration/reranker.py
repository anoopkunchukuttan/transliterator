import string, codecs, re, os, sys, itertools, functools, operator
import numpy
from srilm import * 
from cfilt.transliteration.utilities import *

def rerank_topn(infname,outfname,n,lm_fname,lm_order): 
    """
    Reranks the n-best list from the base SMT system with language model and the baseline SMT system scores
    """

    # read the language model 
    lm_model=load_lm_model(lm_fname,lm_order)

    def parse_line(line):
        fields=[ x.strip() for x in  line.strip().split('|||') ]
        fields[0]=int(fields[0])
        fields[3]=float(fields[3])
        return fields

    def rescore_candidate(fields):
        # desegment
        sentence=fields[1]
        #score with language model and the baseline SMT model 
        score=getSentenceProb(lm_model,sentence.encode('utf-8'),len(sentence.split(' ')))+fields[3]
        fields[3]=score

        return fields

    # desegment and rank the candidates by the LM probability
    with codecs.open(infname,'r','utf-8') as infile: 
        with codecs.open(outfname,'w','utf-8') as outfile: 
            for sent_no, lines in itertools.groupby(itertools.imap(parse_line,iter(infile)),key=operator.itemgetter(0)) :
                ranked_candidates=list(itertools.imap(rescore_candidate, itertools.islice(lines,0,n)))
                ranked_candidates.sort(key=operator.itemgetter(3),reverse=True)
    
                for cand in ranked_candidates: 
                    outfile.write(u'{} ||| {} ||| {} ||| {}\n'.format(*cand))

if __name__ == '__main__': 
    rerank_topn(sys.argv[1],
                sys.argv[2],
                int(sys.argv[3]),
                sys.argv[4],
                int(sys.argv[5]),
                )

