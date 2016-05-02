from cfilt.transliteration.utilities import *
from indicnlp.transliterate.unicode_transliterate import * 


slist=read_monolingual_corpus(sys.argv[1])
tlist=read_monolingual_corpus(sys.argv[2])

slist_a=set()
tlist_a=set()

slist_a.update([ UnicodeIndicTransliterator.transliterate(x,sys.argv[3],'hi') for x in slist ] )
tlist_a.update([ UnicodeIndicTransliterator.transliterate(x,sys.argv[4],'hi') for x in tlist ] )

print len(tlist_a.intersection(slist_a))
