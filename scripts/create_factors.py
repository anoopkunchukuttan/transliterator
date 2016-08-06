from indicnlp.script import indic_scripts
from indicnlp import loader
from cfilt.transliteration import utilities 
import codecs, sys 

def create_factor_file(infname,outfname,lang): 

    with codecs.open(outfname,'w','utf-8') as outfile: 
        for w in utilities.read_monolingual_corpus(infname): 
            factor_v_list=[]
            for c in w: 
                f_vec=indic_scripts.get_phonetic_feature_vector(c,lang)

                factor_v=[]
                factor_v.append(c)

                for p in indic_scripts.PV_PROP: 
                    factor_v.append(str(indic_scripts.get_property_value(f_vec,p)))

                factor_v_list.append(factor_v)

            line=u' '.join( [ '|'.join(x) for x in factor_v_list ]  )                
            outfile.write(line + u'\n')

def create_factormono_file(infname,outdir,lang): 

    for p in indic_scripts.PV_PROP: 
        with codecs.open(outdir+'/'+p+'.'+lang,'w','utf-8') as outfile: 
            for w in utilities.read_monolingual_corpus(infname): 
                factor_v_list=[]
                for c in w: 
                    f_vec=indic_scripts.get_phonetic_feature_vector(c,lang)
                    factor_v_list.append(str(indic_scripts.get_property_value(f_vec,p)))

                line=u' '.join( factor_v_list )                
                outfile.write(line + u'\n')

def create_single_factor_moses_conf(template_fname,outfname,slang,tlang,factorname,factnum): 
    with codecs.open(template_fname,'r','utf-8') as tfile: 
        with codecs.open(outfname,'w','utf-8') as ofile: 
            template=tfile.read()
            contents=template.format(slang=slang,tlang=tlang,prop=factorname,prop_num=factnum)
            ofile.write(contents)

def create_test_moses_ems_conf(template_fname,outfname,slang,tlang,blang,factorname,factnum): 
    with codecs.open(template_fname,'r','utf-8') as tfile: 
        with codecs.open(outfname,'w','utf-8') as ofile: 
            template=tfile.read()
            contents=template.format(slang=slang,tlang=tlang,blang=blang,prop=factorname,prop_num=factnum)
            ofile.write(contents)

if __name__ == '__main__': 
    
    loader.load()
    commands={
                'create_factor_file':create_factor_file,
                'create_factormono_file':create_factormono_file,
                'create_single_factor_moses_conf': create_single_factor_moses_conf,
                'create_test_moses_ems_conf': create_test_moses_ems_conf,
            }

    commands[sys.argv[1]](*sys.argv[2:])

    #create_factor_file(sys.argv[1],sys.argv[2],sys.argv[3])
    #create_factormono_file(sys.argv[1],sys.argv[2],sys.argv[3])
    #create_single_factor_moses_conf(*sys.argv[1:])
    #create_test_moses_ems_conf(*sys.argv[1:])

