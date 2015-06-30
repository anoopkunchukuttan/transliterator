import os, sys, re, itertools, codecs 
import xml.etree.ElementTree as ET
#from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import numpy as np, math, scipy
import random
import shutil
from collections import defaultdict 

from cfilt.transliteration.decoder import *
from cfilt.transliteration.utilities import *

#infname=sys.argv[1]
#outdir=sys.argv[2]
#prefix=sys.argv[3]
#src_lang=sys.argv[4]
#tgt_lang=sys.argv[5]

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


### OLD : obsolete
#def parse_news_2015(infname, 
#                    outdir, 
#                    prefix, 
#                    src_lang, 
#                    tgt_lang): 
#
#    if not os.path.exists(outdir):        
#        os.mkdir(outdir)
#
#    # create normalizer
#    factory=IndicNormalizerFactory()
#    normalizer=factory.get_normalizer(tgt_lang,False)
#
#    # parser
#    tree = ET.parse(infname)
#    root = tree.getroot()
#    
#    # open files
#    srcfile=codecs.open(outdir+'/{}.{}'.format(prefix,src_lang),'w','utf-8')
#    tgtfile=codecs.open(outdir+'/{}.{}'.format(prefix,tgt_lang),'w','utf-8')
#    idfile=codecs.open(outdir+'/{}.{}'.format(prefix,'id'),'w','utf-8')
#    
#    # stats
#    pairs=0
#    chars_src=0
#    chars_org=0
#    chars_norm=0
#
#    for name in root: 
#        srcnode=name.find('SourceName')
#        name_id=name.attrib['ID']
#    
#        src_text=srcnode.text
#
#        for tgtnode in name.findall('TargetName'):
#            tgt_id=tgtnode.attrib['ID']
#            tgt_text=tgtnode.text
#   
#            srcfile.write(u' '.join(src_text)+'\n')
#            tgtfile.write(u' '.join(tgt_text)+'\n')
#            idfile.write('{}_{}\n'.format(name_id,tgt_id))
#   
#            pairs+=1
#            chars_src+=len(src_text)
#            chars_org+=len(tgt_text)
#            chars_norm+=len(normalizer.normalize(tgt_text))
#
#    print '{}|{}|{}|{}|{}|{}'.format(src_lang,tgt_lang,
#            pairs,chars_src,chars_org,chars_norm)
#
#    srcfile.close()
#    tgtfile.close()
#    idfile.close()




def parse_news_2015(infname, 
                    outdir, 
                    prefix, 
                    src_lang, 
                    tgt_lang): 
    """
        infname: input XML file  
        outdir: output dir  
        prefix: 'test', or 'train' 
        src_lang 
        tgt_lang 
    """

    # NEWS code to ISO code for Indian language    
    lang_code_mapping={
                        'Ba':'bn',
                        'Ka':'kn',
                        'Hi':'hi',
                        'Ta':'ta',
                      }

    if not os.path.exists(outdir):        
        os.mkdir(outdir)

    # create normalizer
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer( lang_code_mapping[tgt_lang] if tgt_lang in lang_code_mapping else tgt_lang ,False)

    # parser
    tree = ET.parse(infname)
    root = tree.getroot()
    
    # open files
    srcfile=codecs.open(outdir+'/{}.{}'.format(prefix,src_lang),'w','utf-8')
    tgtfile=codecs.open(outdir+'/{}.{}'.format(prefix,tgt_lang),'w','utf-8')
    idfile=codecs.open(outdir+'/{}.{}'.format(prefix,'id'),'w','utf-8')
    
    # stats
    pairs=0
    chars_src=0
    chars_org=0
    chars_norm=0

    for name in root: 
        srcnode=name.find('SourceName')
        name_id=name.attrib['ID']
    
        src_text=srcnode.text
        src_words=src_text.split(' ')

        children=None
        if prefix=='train':
            ## use for training corpus
            children=name.findall('TargetName')
        else:                        
            # use for test corpus
            children=[name.find('TargetName')]

        for tgtnode in children: 
            tgt_id=tgtnode.attrib['ID']
            tgt_text=tgtnode.text
            tgt_words=tgt_text.split(' ')
 
            # if an input entry contains multiple words 

            # Case 1: generate one line per word   
            #if len(src_words)==len(tgt_words):
            #    for offsetno, (src_word,tgt_word) in enumerate(zip(src_words,tgt_words)): 
            #        srcfile.write(u' '.join(src_word)+'\n')
            #        tgtfile.write(u' '.join(tgt_word)+'\n')
            #        idfile.write('{}_{}_{}\n'.format(name_id,tgt_id,offsetno))
   
            #        pairs+=1
            #        chars_src+=len(src_word)
            #        chars_org+=len(tgt_word)
            #        if tgt_lang in lang_code_mapping:
            #            tgt_word=normalizer.normalize(tgt_word)
            #        chars_norm+=len(tgt_word)

            # Case 2: generate just a single word 

            srcfile.write( u' '.join([  u' '.join(src_word) for src_word in src_words ])   +'\n')
            tgtfile.write( u' '.join([  u' '.join(tgt_word) for tgt_word in tgt_words ])   +'\n')
            idfile.write('{}_{}_{}\n'.format(name_id,tgt_id,0))

    print '{}|{}|{}|{}|{}|{}|{}'.format(prefix,src_lang,tgt_lang,
            pairs,chars_src,chars_org,chars_norm)

    srcfile.close()
    tgtfile.close()
    idfile.close()



#### For parsing the official test set 
def parse_news_2015_official(infname, 
                    outdir, 
                    prefix, 
                    src_lang, 
                    tgt_lang): 
    """
        infname: input XML file  
        outdir: output dir  
        prefix: 'test', or 'train' 
        src_lang 
        tgt_lang 
    """

    # hack for the test corpus 
    src_lang=src_lang.split('_')[2][:2]
    tgt_lang=tgt_lang.split('_')[2][2:]

    outdir='{}/{}-{}'.format(outdir,src_lang,tgt_lang)

    # hack ends
    
    # NEWS code to ISO code for Indian language    
    lang_code_mapping={
                        'Ba':'bn',
                        'Ka':'kn',
                        'Hi':'hi',
                        'Ta':'ta',
                      }

    if not os.path.exists(outdir):        
        os.mkdir(outdir)

    # create normalizer
    #factory=IndicNormalizerFactory()
    #normalizer=factory.get_normalizer( lang_code_mapping[tgt_lang] if tgt_lang in lang_code_mapping else tgt_lang ,False)

    # parser
    tree = ET.parse(infname)
    root = tree.getroot()
    
    # open files
    srcfile=codecs.open(outdir+'/{}.{}'.format(prefix,src_lang),'w','utf-8')
    idfile=codecs.open(outdir+'/{}.{}'.format(prefix,'id'),'w','utf-8')
    

    for name in root: 
        srcnode=name.find('SourceName')
        name_id=name.attrib['ID']
   
        src_text=''
        if src_lang=='En' and tgt_lang=='Pe': 
            src_text=srcnode.text.lower()
        elif src_lang=='En': 
            src_text=srcnode.text.upper()
        else:
            src_text=srcnode.text
        src_words=src_text.split(' ')

        srcfile.write( u' '.join([  u' '.join(src_word) for src_word in src_words ])   +'\n')
        idfile.write('{}_{}_{}\n'.format(name_id,0,0))

    srcfile.close()
    idfile.close()

def iterate_line_format_op(src_fname,tgt_fname,id_fname): 
    """
        iterate line format output file along with other files for generation of NEWS output XML 
    """

    # open files
    srcfile=codecs.open(src_fname,'r','utf-8')
    tgtfile=codecs.open(tgt_fname,'r','utf-8')
    idfile=codecs.open(id_fname,'r','utf-8')
 
    prev_name_id='-1'
    current_record=[]

    for srcline,tgtline,idline in itertools.izip(iter(srcfile),iter(tgtfile),iter(idfile)):
        
        id_fields=idline.strip().split('_')
        src_text=srcline.strip().replace(u' ',u'')    
        tgt_text=tgtline.strip().replace(u' ',u'')    

        if id_fields[0]!=prev_name_id: 
            if prev_name_id!='-1':
                ## consolidate gather record
                #payload=[ [ ii[1]+'_'+ii[2] , t ]  for n,(ii,s,t) in enumerate(current_record) ] 
                payload=[ [ n+1 , r[2] ]  for n,r in enumerate(current_record)] 
                yield [ current_record[0][0][0], current_record[0][1], payload ]

            ## create new record
            current_record=[]
            prev_name_id=id_fields[0]

        ## gather records 
        current_record.append( [id_fields,src_text,tgt_text] ) 

    if len(current_record)>0:
        payload=[ [ n+1 , r[2] ]  for n,r in enumerate(current_record)] 
        yield [ current_record[0][0][0], current_record[0][1], payload ]

    # close files 
    srcfile.close()
    tgtfile.close()
    idfile.close()

def iterate_moses_nbest_op(src_fname,tgt_fname,id_fname,topk=10): 
    """
        iterate moses format nbest output file along with other files for generation of NEWS output XML 

        src_fname: xml file for the reference data  
    """

    def iter_src_names(infname):
    
        # parser
        tree = ET.parse(infname)
        root = tree.getroot()
        
        for name in root: 
            srcnode=name.find('SourceName')
            src_text=srcnode.text
            yield src_text

    # open files
    idfile=codecs.open(id_fname,'r','utf-8')
 
    prev_name_id='-1'
    current_record=[]

    for srcline,nbest_record,idline in itertools.izip(iter_src_names(src_fname),iterate_nbest_list(tgt_fname),iter(idfile)):
        
        id_fields=idline.strip().split('_')
        src_text=srcline
        sent_no, nbest_list=nbest_record
        
        tgt_text_list = [ candidate[1].strip().replace(u' ',u'') for candidate in nbest_list ]

        if id_fields[0]!=prev_name_id: 
            if prev_name_id!='-1':

                ### consolidate gather record
                xlists=[]
                for c_id,csrc_text,xlist in current_record: 
                    xlist.extend( [xlist[-1]]*(topk-len(xlist)) )
                    xlists.append(xlist)

                yield [ 
                        c_id[0], 
                        csrc_text, 
                        enumerate ( [ u' '.join(l) for l in zip(*xlists) ], 1 ) 
                      ]  

            ## create new record
            current_record=[]
            prev_name_id=id_fields[0]

        ## gather records 
        current_record.append( [id_fields,src_text,tgt_text_list] ) 

    if len(current_record)>0:
        xlists=[]
        for c_id,csrc_text,x_list in current_record: 
            xlist.extend( xlist[-1]*(topk-len(xlist)) )
            xlists.append(xlist)
        
        yield [ 
                c_id[0], 
                csrc_text, 
                enumerate ( [ u' '.join(l) for l in zip(*xlists) ], 1 ) 
              ]  


    # close files 
    idfile.close()

def generate_news_2015_output(id_fname,src_fname,tgt_fname,out_fname,systemtype,corpus,srclang,tgtlang,topk=10):

    outfile_header=u"""<?xml version="1.0" encoding="UTF-8"?>
    
    <TransliterationTaskResults
    	SourceLang = "{}"
    	TargetLang = "{}"
    	GroupID = "IIT Bombay - 1"
    	RunID = "1"
    	RunType = "Standard"
        Comments = "System: {} Corpus: {}">\n
""".format(srclang,tgtlang,systemtype,corpus)

    # (src_id,src_str,( (tgt_id,tgt_str), ....  ) )

    with codecs.open(out_fname,'w','utf-8') as outfile:  

        outfile.write(outfile_header)

        #for record in iterate_line_format_op(src_fname,tgt_fname,id_fname): 
        for record in iterate_moses_nbest_op(src_fname,tgt_fname,id_fname, topk): 
            src_id=record[0]
            src_str=record[1]

            outfile.write(u'<Name ID="{}">\n'.format(src_id))
            outfile.write(u'<SourceName>{}</SourceName>\n'.format(src_str))

            for tgt_id,tgt_str in record[2]:
	    	     outfile.write(u'<TargetName ID="{}">{}</TargetName>\n'.format(
                     tgt_id,tgt_str))
                
            outfile.write(u'</Name>\n\n')

        outfile.write('\n</TransliterationTaskResults>\n')

def generate_news_2015_gold_standard(src_fname,tgt_fname,out_fname,corpus,corpus_size,srclang,tgtlang):

    outfile_header=u"""<?xml version="1.0" encoding="UTF-8"?>
    
    <TransliterationCorpus 
        CorpusFormat="UTF-8" 
        CorpusID="{}" 
        CorpusSize="{}" 
        CorpusType="Dev" 
        NameSource="Mixed" 
    	SourceLang = "{}"
    	TargetLang = "{}">\n
""".format(corpus,corpus_size,srclang,tgtlang)


#<TransliterationCorpus CorpusFormat="UTF-8" CorpusID="NEWS2012-EnHi" CorpusSize="997" CorpusType="Dev" NameSource="Mixed" SourceLang="En" TargetLang="Hi">

    # (src_id,src_str,( (tgt_id,tgt_str), ....  ) )

    srcfile=codecs.open(src_fname,'r','utf-8')
    tgtfile=codecs.open(tgt_fname,'r','utf-8')

    with codecs.open(out_fname,'w','utf-8') as outfile:  

        outfile.write(outfile_header)

        for src_id, (src_str,tgt_str) in enumerate(itertools.izip(iter(srcfile),iter(tgtfile)),1):
            src_str=''.join(src_str.strip().split(' '))
            tgt_str=''.join(tgt_str.strip().split(' '))

            outfile.write(u'<Name ID="{}">\n'.format(src_id))
            outfile.write(u'<SourceName>{}</SourceName>\n'.format(src_str))

	    outfile.write(u'<TargetName ID="{}">{}</TargetName>\n'.format(1,tgt_str))
                
            outfile.write(u'</Name>\n\n')

        outfile.write('\n</TransliterationCorpus>\n')

    srcfile.close()
    tgtfile.close()

#### ngram related methods 

def create_ngram_corpus(datadir,src,tgt,order=2,overlap=True):
   
    #for ftype in ['test','train','tun']: 
        #for lang in [src,tgt]:
    for ftype in ['test']: 
        for lang in [src]:
            infile=codecs.open(datadir+'/{}.{}'.format(ftype,lang),'r','utf-8')
            dataset=list( iter(infile) )
            infile.close()

            outfile=codecs.open(datadir+'/{}.{}'.format(ftype,lang),'w','utf-8')
            for line in dataset: 
                char_list=line.strip().split(' ')
                ngrams=[]
                if not overlap: 
                    ngrams = [ char_list[i:i+order]   for i in range(0,len(char_list),order) ]
                else: 
                    ngrams = [ char_list[i:i+order]   for i in range(0,len(char_list),order-1) ]
                outfile.write( ' '.join(  [u''.join(x) for x in ngrams] ) + u'\n')    
            outfile.close()

def postedit_ngram_line(line,order=2,overlap=True): 

    char_list=line.strip().split(' ')
    outchars=None
    if not overlap: 
        outchars = u' '.join(u''.join(char_list))
    else: 
        outchars = u' '.join(u''.join([ x[:-1] if len(x)==order else x for x in char_list ]))
    return outchars

def postedit_ngram_output(fname, order=2, overlap=True): 

    infile=codecs.open(fname,'r','utf-8')
    dataset=list( iter(infile) )
    infile.close()
    
    outfile=codecs.open(fname,'w','utf-8')
    for line in dataset: 
        outfile.write( postedit_ngram_output(line,order,overlap)  + u'\n')

### Marker related methods 

def add_markers_corpus(datadir,src,tgt):
   
    #for ftype in ['test','train','tun']: 
        #for lang in [src,tgt]:
    for ftype in ['test']: 
        for lang in [src]:
            infile=codecs.open(datadir+'/{}.{}'.format(ftype,lang),'r','utf-8')
            dataset=list( iter(infile) )
            infile.close()

            outfile=codecs.open(datadir+'/{}.{}'.format(ftype,lang),'w','utf-8')
            for line in dataset: 
                outfile.write( u'^ ' + line.strip() + u' $' + u'\n' )
            outfile.close()


def remove_markers_line(line):
    chars=filter( lambda x:x not in [u'^',u'$'],
                    line.strip().split(' ') 
                )
    return u' '.join(chars)

def remove_markers(fname):
   
    infile=codecs.open(fname,'r','utf-8')
    dataset=list( iter(infile) )
    infile.close()
    
    outfile=codecs.open(fname,'w','utf-8')
    for line in dataset: 
        outfile.write( remove_markers_line(line)  + u'\n' )
    outfile.close()

### Brahminet related methods 

def prepare_brahminet_training_corpus(infname, outsrcfname, outtgtfname): 
    with codecs.open(infname,'r','utf-8') as infile:
        with codecs.open(outsrcfname,'w','utf-8') as outsrcfile:
            with codecs.open(outtgtfname,'w','utf-8') as outtgtfile:
                for line in infile: 
                    fields=line.strip().split('|')
                    outsrcfile.write( ' '.join(fields[0].upper()) + '\n' )
                    outtgtfile.write( ' '.join(fields[1]) + '\n' )

def compute_parallel_corpus_statistics(srcfname,tgtfname):

    ratio_list=[]

    with codecs.open(srcfname,'r','utf-8') as srcfile:
        with codecs.open(tgtfname,'r','utf-8') as  tgtfile:
            for srcline,tgtline in itertools.izip(iter(srcfile),iter(tgtfile)): 
                srcchars=srcline.strip().split(' ')        
                tgtchars=tgtline.strip().split(' ')        

                ratio=float(len(tgtchars))/float(len(srcchars))
                ratio_list.append(ratio)

    ratio_array=np.array( ratio_list  )    
    mean=np.mean(ratio_array)
    std=np.std(ratio_array)

    print '{} {}'.format(mean,std)

def filter_brahminet_length_ratio(datadir,src,tgt,lratio,std):
   
    srcfile=codecs.open(datadir+'/train.{}'.format(src),'r','utf-8')
    tgtfile=codecs.open(datadir+'/train.{}'.format(tgt),'r','utf-8')
    
    dataset=list( itertools.izip( iter(srcfile) , iter(tgtfile) ) )

    srcfile.close()
    tgtfile.close()

    trainsrcfile=codecs.open(datadir+'/train.{}'.format(src),'w','utf-8')
    traintgtfile=codecs.open(datadir+'/train.{}'.format(tgt),'w','utf-8')

    for srcline,tgtline in dataset: 
        srcchars=srcline.strip().split(' ')        
        tgtchars=tgtline.strip().split(' ')        
    
        ratio=float(len(tgtchars))/float(len(srcchars))
        if ratio>=(lratio-std) and ratio<=(lratio+std):
            trainsrcfile.write(srcline)
            traintgtfile.write(tgtline)

    trainsrcfile.close()
    traintgtfile.close()


def extract_common_msr_corpus(c0_dir, c1_dir, c0_lang, c1_lang, outdir ): 

    data_cache=defaultdict(lambda : [set(),set()])

    # read corpus 0
    en0_f=codecs.open(c0_dir+'/train.En','r','utf-8')
    l0_f=codecs.open(c0_dir+'/train.'+c0_lang,'r','utf-8')

    for en_l,c_l in itertools.izip(iter(en0_f),iter(l0_f)): 
        data_cache[en_l.strip()][0].add(c_l.strip())

    en0_f.close()
    l0_f.close()

    # read corpus 1                
    en1_f=codecs.open(c1_dir+'/train.En','r','utf-8')
    l1_f=codecs.open(c1_dir+'/train.'+c1_lang,'r','utf-8')

    for en_l,c_l in itertools.izip(iter(en1_f),iter(l1_f)): 
        data_cache[en_l.strip()][1].add(c_l.strip())

    en1_f.close()
    l1_f.close()

    # write the common data
    cc0_f=codecs.open(outdir+'/train.'+c0_lang,'w','utf-8')
    cc1_f=codecs.open(outdir+'/train.'+c1_lang,'w','utf-8')

    for en_l, other_l_lists in data_cache.iteritems(): 
        if len(other_l_lists[0]) >0 and len(other_l_lists[1]) >0 : 
            #print 'inside: {}'.format(len(other_l_lists[0])*len(other_l_lists[1]))
            for c0_str, c1_str in itertools.product( other_l_lists[0] , other_l_lists[1] ):  
                cc0_f.write(c0_str +u'\n') 
                cc1_f.write(c1_str +u'\n') 

    cc0_f.close()
    cc1_f.close()

def count_common_msr_corpus(en1_fname,en2_fname): 

    with codecs.open(en1_fname,'r','utf-8') as en1_file:
        with codecs.open(en2_fname,'r','utf-8') as en2_file:

            en1_words=set([ w.strip() for w in en1_file.readlines()])
            en2_words=set([ w.strip() for w in en2_file.readlines()])

            print len(en1_words.intersection(en2_words))


### Corpus management methods 

def randomize_and_select(insrcfname,intgtfname,outsrcfname,outtgtfname, n):

    insrcfile=codecs.open(insrcfname,'r','utf-8')
    intgtfile=codecs.open(intgtfname,'r','utf-8')

    outsrcfile=codecs.open(outsrcfname,'w','utf-8')
    outtgtfile=codecs.open(outtgtfname,'w','utf-8')


    dataset=list( itertools.izip( iter(insrcfile) , iter(intgtfile) ) )
    random.shuffle(dataset)

    for src,tgt in dataset[:int(n)]: 
        outsrcfile.write(src)
        outtgtfile.write(tgt)

    insrcfile.close()
    intgtfile.close()
    outsrcfile.close()
    outtgtfile.close()

def create_train_tun(datadir,src,tgt,tun_size='500'):

    tun_size=int(tun_size)

    # rename files 
    os.rename(datadir+'/train.{}'.format(src),datadir+'/train+tun.{}'.format(src))
    os.rename(datadir+'/train.{}'.format(tgt),datadir+'/train+tun.{}'.format(tgt))
    os.rename(datadir+'/train.{}'.format('id'),datadir+'/train+tun.{}'.format('id'))

    # read original training file     
    srcfile=codecs.open(datadir+'/train+tun.{}'.format(src),'r','utf-8')
    tgtfile=codecs.open(datadir+'/train+tun.{}'.format(tgt),'r','utf-8')
    idfile=codecs.open(datadir+'/train+tun.{}'.format('id'),'r','utf-8')

    dataset=list( itertools.izip( iter(srcfile) , iter(tgtfile) , iter(idfile) ) )
    random.shuffle(dataset)
    print len(dataset)

    srcfile.close()
    tgtfile.close()
    idfile.close()

    # create new training file 
    trainsrcfile=codecs.open(datadir+'/train.{}'.format(src),'w','utf-8')
    traintgtfile=codecs.open(datadir+'/train.{}'.format(tgt),'w','utf-8')
    trainidfile=codecs.open(datadir+'/train.{}'.format('id'),'w','utf-8')

    for srcline,tgtline,idn in dataset[tun_size:]: 
        trainsrcfile.write(srcline)
        traintgtfile.write(tgtline)
        trainidfile.write(idn)

    trainsrcfile.close()
    traintgtfile.close()
    trainidfile.close()

    # create new tuning file 
    tunsrcfile=codecs.open(datadir+'/tun.{}'.format(src),'w','utf-8')
    tuntgtfile=codecs.open(datadir+'/tun.{}'.format(tgt),'w','utf-8')
    tunidfile=codecs.open(datadir +'/tun.{}'.format('id'),'w','utf-8')

    for srcline,tgtline,idn in dataset[:tun_size]: 
        tunsrcfile.write(srcline)
        tuntgtfile.write(tgtline)
        tunidfile.write(idn)

    tunsrcfile.close()
    tuntgtfile.close()
    tunidfile.close()

    # unlink file 
    os.unlink(datadir+'/train+tun.{}'.format(src))
    os.unlink(datadir+'/train+tun.{}'.format(tgt))
    os.unlink(datadir+'/train+tun.{}'.format('id'))

def postprocess_nbest_list(nbest_fname, systemtype): 
    #parameters to pipeline elements 
    order=2 
    overlap=True
   
    print 'System type: ' + systemtype

    # readfile
    dataset=list(iterate_nbest_list(nbest_fname))

    outfile=codecs.open(nbest_fname,'w','utf-8')
    for sent_no, parsed_lines in dataset: 
        for parsed_line in parsed_lines: 

            ## marker
            #parsed_line[1] = remove_markers_line(  parsed_line[1] ) 
            ## 2gram
            #parsed_line[1] = postedit_ngram_line( parsed_line[1], order, overlap ) 
            ## marker+2gram
            parsed_line[1] = remove_markers_line( postedit_ngram_line( parsed_line[1], order, overlap ) )

            outfile.write( u'{} ||| {} ||| {} ||| {}\n'.format( *parsed_line  ) )

    outfile.close()

def convert_to_nbest_format(infname,outfname):
    with codecs.open(infname,'r','utf-8') as infile: 
        with codecs.open(outfname,'w','utf-8') as outfile: 
            for n,line in enumerate(iter(infile)):
                outfile.write( u'{} ||| {} ||| {} ||| {}\n'.format( n, line.strip(), 
                    u'Distortion0= 0 LM0= -25.8624 WordPenalty0= -6 PhrasePenalty0= 3 TranslationModel0= -1.12625 -4.78717 -0.83186 -6.77739', u'-5.38081' ) )

def convert_to_1best_format(infname,outfname):
    with codecs.open(outfname,'w','utf-8') as outfile:
        for sent_no, parsed_lines in iterate_nbest_list(infname): 
            outfile.write(parsed_lines[0][1].strip()+u'\n')

def correct_vowels(nbest_fname,lang): 
    """
    Correcting initial vowel mark
    """
    # readfile
    dataset=list(iterate_nbest_list(nbest_fname))

    basepoint=0x900

    outfile=codecs.open(nbest_fname,'w','utf-8')
    for sent_no, parsed_lines in dataset: 
        for parsed_line in parsed_lines: 

            # correcting initial vowel mark    
            offset=ord(parsed_line[1][0])-basepoint
            if offset>=0x3e and  offset<=0x4c: 
                newc=unichr(basepoint+offset-0x3e+0x6)
                parsed_line[1]=newc+parsed_line[1][1:]

            ## correcting internal vowel 
            #teststr=parsed_line[1].split(' ')[1:]

            #outc=[]
            #for c in teststr:
            #    offset=ord(c)-basepoint
            #    if offset>=0x6 and  offset<=0x14: 
            #        outc.append(unichr(basepoint+offset-0x6+0x3e))
            #    else:
            #        outc.append(c)
            #parsed_line[1]=parsed_line[1][0]  + u' ' +u' '.join(outc)    

            outfile.write( u'{} ||| {} ||| {} ||| {}\n'.format( *parsed_line  ) )

    outfile.close()


### Analysis methods 
#def compute_entropy(giza_lex_fname):
#    prob_map=defaultdict(list)
#    entropy_map={}
#
#    with codecs.open(giza_lex_fname,'r','utf-8') as giza_lex_file: 
#        for line in iter(giza_lex_file):
#            c1,c2,prob=line.strip().split(' ')
#            prob=float(prob)
#            prob_map[c2].append(prob)
#
#    for c2,prob2 in prob_map.iteritems():
#        entropy_map[c2]=-sum([ p*math.log(p)  for p in prob2 ])
#
#    # compute average entropy
#    ave_entropy=float(sum([y for x,y in entropy_map.iteritems()]))/float(len(entropy_map))
#    print ave_entropy
            
def compute_entropy(model_fname):

    translit_model=TransliterationModel.load_translit_model(model_fname)

    logz_func=np.vectorize(log_z,otypes=[np.float])

    print np.average( np.sum( -1.0*translit_model.param_values*logz_func(translit_model.param_values),
                axis=1)
        )

### Wrapper methods 

def filter_brahminet_length_ratio_wrapper(datadir,src,tgt,lratio,std):
    filter_brahminet_length_ratio(datadir,src,tgt,float(lratio),float(std))

### scripting methods 
def extract_corpus_from_xml(xmldir,outdir,corpusinfo_fname):
    with codecs.open(corpusinfo_fname,'r','utf-8') as corpusinfo_file: 
        for line in corpusinfo_file: 
            
            # read fields
            fields=line.strip().split('|')
            src_lang=fields[0]
            tgt_lang=fields[1]
            train_fname=fields[2]
            test_fname=fields[3]

            parse_news_2015(xmldir+'/'+train_fname, 
                    '{}/{}-{}'.format(outdir,src_lang,tgt_lang), 
                    'train', 
                    src_lang, 
                    tgt_lang) 

            parse_news_2015(xmldir+'/'+test_fname, 
                    '{}/{}-{}'.format(outdir,src_lang,tgt_lang), 
                    'test', 
                    src_lang, 
                    tgt_lang) 

def gather_evaluation_results(results_dir,lang_list_fname,eval_fname): 

    with codecs.open(eval_fname,'w','utf-8') as eval_file: 
        with codecs.open(lang_list_fname,'r','utf-8') as lang_list_file: 
            for line in lang_list_file: 
                # read fields
                fields=line.strip().split('-')
                src_lang=fields[0]
                tgt_lang=fields[1]

                acc_list=[]
                fsc_list=[]
                mrr_list=[]
                map_list=[]
                acc_10_list=[]

                for systype in ['pb','marker','2gram','marker+2gram']:
                    try: 
                        eval_fname='{0}/{1}/{2}-{3}/evaluation/test.nbest.{3}.eval'.format(results_dir,systype,src_lang,tgt_lang)
                        with codecs.open(eval_fname,'r','utf-8') as evalfile: 
                            scores=[ l.strip().split(':')[1].strip()  for l in evalfile.readlines()] 
                            acc_list.append(scores[0]) 
                            fsc_list.append(scores[1]) 
                            mrr_list.append(scores[2]) 
                            map_list.append(scores[3]) 
                            acc_10_list.append(scores[4])
                    except Exception as e:
                        print 'Error gathering scores for {}-{}: {}'.format(src_lang,tgt_lang,systype)


                eval_file.write('|'.join( [src_lang,tgt_lang] + acc_list + 
                                                               fsc_list + 
                                                               mrr_list + 
                                                               map_list  +
                                                               acc_10_list
                                ) + '\n' )                                                           


def copy_output(lang_info_fname, results_dir, out_dir,  run_number):

    with codecs.open(lang_info_fname,'r','utf-8') as lang_info_file: 
        for line in lang_info_file: 
            fields=line.strip().split('|')
           
            fname_prefix=fields[3].split('.')[0]
            shutil.copyfile( 
                            '{0}/{1}-{2}/evaluation/test.nbest.{2}.xml'.format(
                            results_dir,fields[0],fields[1]),

                             '{0}/{1}_run{2}_{3}.xml'.format(
                            out_dir,fname_prefix,run_number,'anoop.kunchukuttan')
                           )
           
            #print '{0}/{1}-{2}/evaluation/test.nbest.{2}.xml'.format(
            #                results_dir,fields[0],fields[1])

            #print '{0}/{1}_run{2}_{3}.xml'.format(
            #                out_dir,fname_prefix,run_number,'anoop.kunchukuttan')
            #NEWS12_test_ChEn_1019_run1_myname.xml

def preprocess(datadir,src,tgt):
   
    srcfile=codecs.open(datadir+'/test.{}'.format(src),'r','utf-8')
    tgtfile=codecs.open(datadir+'/test.{}'.format(tgt),'r','utf-8')
    
    dataset=list( itertools.izip( iter(srcfile) , iter(tgtfile) ) )

    srcfile.close()
    tgtfile.close()

    trainsrcfile=codecs.open(datadir+'/test.{}'.format(src),'w','utf-8')
    traintgtfile=codecs.open(datadir+'/test.{}'.format(tgt),'w','utf-8')

    for srcline,tgtline in dataset: 
        srcchars=[ c for c in srcline.strip()]
        tgtchars=[ c for c in tgtline.strip()]
    
        trainsrcfile.write(u' '.join(srcchars) + u'\n')
        traintgtfile.write(u' '.join(tgtchars) + u'\n')

    trainsrcfile.close()
    traintgtfile.close()

if __name__=='__main__': 

    commands={
        'parse':parse_news_2015,        
        'gen_news_output':generate_news_2015_output,
        'generate_news_2015_gold_standard':generate_news_2015_gold_standard,

        'create_ngram_corpus': create_ngram_corpus,
        'postedit_ngram_output':postedit_ngram_output,

        'prepare_brahminet_training_corpus':prepare_brahminet_training_corpus,
        'filter_brahminet_length_ratio':filter_brahminet_length_ratio_wrapper,
        'compute_stats': compute_parallel_corpus_statistics,
        'randomize_and_select':randomize_and_select,
        'count_common_msr_corpus':count_common_msr_corpus,
        'extract_common_msr_corpus':extract_common_msr_corpus,

        'create_train_tun':create_train_tun,

        'add_markers_corpus':add_markers_corpus,
        'remove_markers':remove_markers,

        'postprocess_nbest_list':postprocess_nbest_list,
        'convert_to_nbest_format':convert_to_nbest_format,
        'convert_to_1best_format':convert_to_1best_format,

        'correct_vowels':correct_vowels,

        'compute_entropy':compute_entropy,

        'extract_corpus_from_xml':extract_corpus_from_xml,
        'gather_evaluation_results':gather_evaluation_results,
        'copy_output':copy_output,
        'preprocess':preprocess,

    }

    commands[sys.argv[1]](*sys.argv[2:])

