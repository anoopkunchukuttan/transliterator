from indicnlp.script import indic_scripts
from indicnlp import loader
from cfilt.transliteration import utilities 
import codecs, sys 

def create_factor_file(infname,outfname,lang): 

    properties=['basic_type',
        'vowel_length',
        'vowel_strength',
        'vowel_status',
        'consonant_type',
        'articulation_place',
        'aspiration',
        'voicing',
        'nasalization',]

    with codecs.open(outfname,'w','utf-8') as outfile: 
        for w in utilities.read_monolingual_corpus(infname): 
            factor_v_list=[]
            for c in w: 
                f_vec=indic_scripts.get_phonetic_feature_vector(c,lang)

                factor_v=[]
                factor_v.append(c)

                for p in properties: 
                    factor_v.append(str(indic_scripts.get_property_value(f_vec,p)))

                factor_v_list.append(factor_v)

            line=u' '.join( [ '|'.join(x) for x in factor_v_list ]  )                
            outfile.write(line + u'\n')

def create_factormono_file(infname,outdir,lang): 

    properties=['basic_type',
        'vowel_length',
        'vowel_strength',
        'vowel_status',
        'consonant_type',
        'articulation_place',
        'aspiration',
        'voicing',
        'nasalization',]

    for p in properties: 
        with codecs.open(outdir+'/'+p+'.'+lang,'w','utf-8') as outfile: 
            for w in utilities.read_monolingual_corpus(infname): 
                factor_v_list=[]
                for c in w: 
                    f_vec=indic_scripts.get_phonetic_feature_vector(c,lang)
                    factor_v_list.append(str(indic_scripts.get_property_value(f_vec,p)))

                line=u' '.join( factor_v_list )                
                outfile.write(line + u'\n')

if __name__ == '__main__': 
    
    loader.load()
    create_factor_file(sys.argv[1],sys.argv[2],sys.argv[3])
    #create_factormono_file(sys.argv[1],sys.argv[2],sys.argv[3])

