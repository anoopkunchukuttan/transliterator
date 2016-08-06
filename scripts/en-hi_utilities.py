import codecs, sys,re, itertools 

from indicnlp import loader
from indicnlp.script import english_script

def filter_for_g2p(infname,outfname): 
    pattern=re.compile(r'[^A-Z ]')
    with codecs.open(infname,'r','utf-8') as infile, \
         codecs.open(outfname,'w','utf-8') as outfile: 
            for line in itertools.ifilter(lambda x: pattern.search(x.strip()) is None,iter(infile)): 
                outfile.write(line)

def convert_to_phonetisaurus_input_format(infname,outfname): 
    with codecs.open(infname,'r','utf-8') as infile, \
         codecs.open(outfname,'w','utf-8') as outfile: 
        for line in infile: 
            outfile.write(u''.join(line.strip().split(u' '))+'\n')

def convert_phonetisaurus_to_xlit_format(infname,ph_outfname,enc_outfname): 

    num_pat=re.compile(r'\d+')

    with codecs.open(infname,'r','utf-8') as infile, \
         codecs.open(ph_outfname,'w','utf-8') as ph_outfile, \
         codecs.open(enc_outfname,'w','utf-8') as enc_outfile: 
        for line in infile: 

            ## extract fields 
            start=line.find('<s>')+3
            end=line.find('</s>')
            ph_str=re.sub( r'\s+'  ,' ' ,line[start:end].strip())

            ### remove multiple options 
            phonemes = [ x.split('|')[0] if x.split('|')[0]!='' else x.split('|')[1] for x in ph_str.split(' ') ]
            ph_str=' '.join(phonemes)

            ## replace stress markers
            ph_str=num_pat.sub('',ph_str)

            ## substitution rules 
            phonemes=ph_str.split(' ')
            phonemes=[  'AH R'  if ph=='ER' else ph for ph in phonemes ]
            phonemes=[  'AH R'  if ph=='AXR' else ph for ph in phonemes ]
            ph_str=' '.join(phonemes)
            ph_outfile.write(ph_str + '\n')

            ## encode the phoneme representation in a transliteration app friendly format
            phonemes= [ english_script.phoneme_to_enc(ph) for ph in ph_str.split(' ') ]
            ph_str=' '.join(phonemes)
            enc_outfile.write(ph_str + '\n')

if __name__ == '__main__': 

    ### Indic NLP lib load 
    loader.load()

    commands = {
            'filter_for_g2p':filter_for_g2p,
            'convert_to_phonetisaurus_input_format': convert_to_phonetisaurus_input_format,
            'convert_phonetisaurus_to_xlit_format': convert_phonetisaurus_to_xlit_format,
            }

    commands[sys.argv[1]](*sys.argv[2:])
