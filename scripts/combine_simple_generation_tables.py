import codecs 
import gzip 
import sys

def combine_generation_tables(indir, outfname, topn=10): 

    entries=set()
    
    for n in xrange(0,topn): 
        gtable_fname='{}/{}/moses_data/model/generation.0-1.gz'.format(indir,n)
        zf = gzip.open(gtable_fname)
        reader = codecs.getreader("utf-8")
        contents = reader( zf )
        for line in contents:
            entries.add(line)
    
    with codecs.open(outfname,'w','utf-8') as outfile: 
        for line in entries: 
            outfile.write(line)


if __name__=='__main__': 
    combine_generation_tables(*sys.argv[1:])
