import codecs, sys

with codecs.open(sys.argv[1],'r','utf-8') as infile, \
     codecs.open(sys.argv[2],'w','utf-8') as s_outfile, \
     codecs.open(sys.argv[3],'w','utf-8') as t_outfile:
        for line in infile: 
            srcl, tgts=line.strip().split(u'|') 
            tgts=tgts.split(u'^')
            for tgtl in tgts:
                s_outfile.write(u' '.join(srcl)+u'\n')
                t_outfile.write(u' '.join(tgtl)+u'\n')
