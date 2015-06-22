import itertools as it
import operator, functools
from indicnlp import langinfo
import scipy
import numpy as np

def count_characters(charlist,is_valid_char,lang): 
    char_map={}
    
    for c in it.ifilter( is_valid_char, charlist ): 
        char_map[c]=char_map.get(c,0)+1

    return char_map

def gen_coord_map(char_map,lang): 
    offset_char_map={}
    for c, f in char_map.iteritems():
        offset=langinfo.get_offset(c,lang)
        if langinfo.in_coordinated_range(offset,lang): 
            offset_char_map[offset]=f

    return offset_char_map    

def calculate_char_proportions(char_map): 

    total_char=sum(it.imap(operator.itemgetter(1),char_map.itervalues()))
   
    char_prop_map={}
    for c,f in char_map.iteritems(): 
        char_prop_map[c]=float(f)/float(total_char)

    return char_prop_map                        

def most_frequent_char(char_map,k): 
    assert len(char_map)>=k
    return sorted(char_map.iteritems(),key=operator.itemgetter(1),reverse=True)[:k]

def least_frequent_char(char_map,k): 
    assert len(char_map)>=k
    return sorted(char_map.iteritems(),key=operator.itemgetter(1),reverse=False)[:k]

def coord_char_fraction(char_map, coord_map): 

    total_char=sum(char_map.itervalues())
    total_coord=sum(coord_map.itervalues())

    return float(total_coord)/float(total_char)

def phonetic_stats(coord_map): 
    pass 

def max_abs_diff_char(cmap1,cmap2,k): 
    assert len(cmap1)>=k and len(cmap2)>=k
    diff_list=[]

    for coffset in xrange(langinfo.COORDINATED_RANGE_START_INCLUSIVE,langinfo.COORDINATED_RANGE_END_INCLUSIVE+1): 
        abs_diff=abs(cmap1[coffset]-cmap2[coffset])
        diff_list.append((coffset,abs_diff))

    return sorted(diff_list,key=operator.itemgetter(1),reverse=True)[:k]
    
def max_rel_diff_char(cmap1,cmap2,k): 
    assert len(cmap1)>=k and len(cmap2)>=k
    diff_list=[]

    for coffset in xrange(langinfo.COORDINATED_RANGE_START_INCLUSIVE,langinfo.COORDINATED_RANGE_END_INCLUSIVE+1): 
        abs_diff=abs(cmap1[coffset]-cmap2[coffset])
        min_val=min(cmap1[coffset],cmap2[coffset])
        frac_diff=abs_diff/(min_val if min_val>0.0 else float('inf'))
        diff_list.append((coffset,frac_diff))

    return sorted(diff_list,key=operator.itemgetter(1),reverse=True)[:k]

def kl_divergence(cmap_p,cmap_q): 
    pk=[]
    qk=[]
    for coffset in xrange(langinfo.COORDINATED_RANGE_START_INCLUSIVE,langinfo.COORDINATED_RANGE_END_INCLUSIVE+1):
        pk.append(cmap_p[coffset])
        qk.append(cmap_q[coffset])
    return scipy.stats.entropy(pk,qk)

#def extract_lang_stats(infname,ofname,lang): 
#    charlist=[]
#    with codecs.open(infname,ofname,lang): 

if __name__=='__main__': 

    commands={
            count_chars: 
            }
            
