import itertools as it
import operator, functools, sys
from indicnlp import langinfo
import scipy
import scipy.stats
import numpy as np
import yaml
import codecs


def count_characters(charlist,is_valid_char): 
    char_map={}
    
    for c in it.ifilter( is_valid_char, charlist ): 
        char_map[c]=char_map.get(c,0)+1

    return char_map

def gen_coord_map(char_map,lang): 
    offset_char_map={}
    for c, f in char_map.iteritems():
        offset=langinfo.get_offset(c,lang)
        if langinfo.in_coordinated_range(offset): 
            offset_char_map[offset]=f

    return offset_char_map    

def calculate_char_proportions(char_map): 

    total_char=sum(list(char_map.itervalues()))
   
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


def monolingual_analysis(fname,outfname,lang): 

    charlist=[]
    m={}

    with codecs.open(fname,'r','utf-8') as infile: 
        charlist=infile.read()

    """
    Language 
    """
    m['lang']=lang

    """
    Count of all characters in the range 
    """
    m['charlist_count']=count_characters(charlist,functools.partial(langinfo.is_indiclang_char,lang=lang))

    """
    Count of all characters in coordinated range
    """
    m['offset_charlist_count']=gen_coord_map(m['charlist_count'],lang)


    """
    Proportion of each character in the data 
    """
    m['char_proportions']=calculate_char_proportions(m['offset_charlist_count'])

    #    """
    #    Ratio of total number of coordinated chars to all language characters
    #    """
    #    self.coord_char_fraction=None

    #    """
    #    Most frequent characters
    #    """
    #    self.most_freq_char=None

    #    """
    #    Least frequent characters
    #    """
    #    self.least_freq_char=None

    with codecs.open(outfname,'w','utf-8') as outfile:
        yaml.dump(m,outfile)         


def kl_divergence(cmap_p,cmap_q): 
    pk=[]
    qk=[]
    for coffset in xrange(langinfo.COORDINATED_RANGE_START_INCLUSIVE,langinfo.COORDINATED_RANGE_END_INCLUSIVE+1):
        pk.append(cmap_p.get(coffset,0.0001))
        qk.append(cmap_q.get(coffset,0.0001))
    return scipy.stats.entropy(pk,qk)

#def extract_lang_stats(infname,ofname,lang): 
#    charlist=[]
#    with codecs.open(infname,ofname,lang): 

if __name__=='__main__': 

    commands={
                'monolingual_analysis': monolingual_analysis, 
            }

    commands[sys.argv[1]](*sys.argv[2:])
