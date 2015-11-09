import string, codecs, re, os, sys, itertools, functools, operator
import numpy
from srilm import * 
from cfilt.transliteration.utilities import *
from collections import defaultdict
import aggregate_api

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


def rerank_file(infname, outfname,str_n,lm_fname,str_order): 

    n=int(str_n)
    order=int(str_order)

    lm_model=load_lm_model(lm_fname,order)
    lm_model_org=load_lm_model(lm_fname,2)

    r=ReRanker(lm_model,lm_model_org)

    with codecs.open(outfname,'w','utf-8') as outfile: 
        for sent_no, candidates in iterate_nbest_list(infname): 
            ranked_candidates=r.rerank_candidates([ (x[1], x[3]) for x in candidates] )
            for cand in ranked_candidates[:n]: 
                outfile.write(u'{} ||| {} ||| {} ||| {}\n'.format(sent_no,cand[0],u'',cand[1]))


def lin_combination(nbest_for_systems): 
    """
    Linear combination of multiple n-best lists for one instance

    nbest_for_systems: is a list with one element for each system being combined. Each element is an n-best list in the form of tuple (transliteration,score)

    return list of tuples (candidates, score) sorted in decreasing order. This is the best linear combination of the multiple n-best lists 
    """
    
    combination_scores={}

    for nbest_list in nbest_for_systems: 
        min_score=min(nbest_list[1],key=lambda x:x[3])[3]
        max_score=max(nbest_list[1],key=lambda x:x[3])[3]
        nbest_listx= [ (x1, x2, x3, (x4-min_score+0.0001)/(max_score-min_score+0.0001) ) for x1,x2,x3,x4 in nbest_list[1] ]
        for _, cand, _, score in nbest_listx: 
            combination_scores[cand]=combination_scores.get(cand,0.0)+score

    return sorted(combination_scores.items(), key=lambda x:x[1], reverse=True)

def rank_aggregation(nbest_for_systems, nrankers, agg, arguments): 
    """
    Rank aggregation of multiple n-best lists for one instance

    nbest_for_systems: is a list with one element for each system being combined. Each element is an n-best list in the form of tuple (transliteration,score)

    return list of tuples (candidates, score) sorted in decreasing order. This is the best linear combination of the multiple n-best lists 
    """

    rankings=defaultdict(lambda:[None]*nrankers)
    ranker_names=[]

    for list_id, nbest_list in enumerate(nbest_for_systems): 
        for r, (_, cand, _, score) in enumerate(nbest_list[1]): 
            rankings[cand][list_id]=r

    ## create input for API
    ranking_objects={}
    
    oid=0
    id_cand_mapping={}
    for cand, rankers in rankings.iteritems(): 
        oid+=1
        ranking_objects[oid]=rankers
        id_cand_mapping[oid]=cand

    ranker_names=[ str(x)  for x in xrange(nrankers) ]

    #print ranking_objects
    #print id_cand_mapping
    ranked_objects=aggregate_api.aggregate(ranking_objects, ranker_names, agg, arguments)

    return [ (id_cand_mapping[x],0.0) for x in ranking_objects ]

def combine_nbest_files(list_fname, out_fname, str_n, combination_method=lin_combination): 
    """
    list_fname: list of files to combine 
    out_fname: combined output file 
    """

    infname_list=[]
    with codecs.open(list_fname, 'r','utf-8') as infnamefile: 
        infname_list=[ x.strip() for x in infnamefile ]

    system_iter_list=[] 

    for infname in infname_list: 
        system_iter_list.append(iter(iterate_nbest_list(infname)))

    #with codecs.open(out_fname, 'w','utf-8') as outfile: 
    #    for sno, nbest_for_systems in enumerate(itertools.izip(*system_iter_list)): 
    #        combined_output=combination_method(nbest_for_systems) 

    #        for c in combined_output[:min(int(str_n) ,len(combined_output))]: 
    #            outfile.write(u'{} ||| {} ||| ||| {}\n'.format(sno,c[0],c[1]))

    agg='pg'
    arguments=['0.85','ibf']

    with codecs.open(out_fname, 'w','utf-8') as outfile: 
        for sno, nbest_for_systems in enumerate(itertools.izip(*system_iter_list)): 
            print sno
            combined_output=rank_aggregation(nbest_for_systems, len(nbest_for_systems), agg, arguments) 

            for c in combined_output[:min(int(str_n) ,len(combined_output))]: 
                outfile.write(u'{} ||| {} ||| ||| {}\n'.format(sno,c[0],c[1]))

if __name__ == '__main__': 
   
    commands={
                'rerank_file': rerank_file,
                'combine_nbest_files': combine_nbest_files, 
            }

    commands[sys.argv[1]](*sys.argv[2:])

