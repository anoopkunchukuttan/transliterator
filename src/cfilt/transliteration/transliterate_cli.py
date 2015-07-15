from cfilt.transliteration.decoder import *
from cfilt.transliteration.utilities import *
from cfilt.transliteration.parallel_decoder import *
from cfilt.transliteration.unsupervised import *

def unsupervised_training(fcorpus_fname, ecorpus_fname, config_param_fname, lm_fname, model_fname):

    lm_model=load_lm_model(lm_fname)
    config_params=read_yaml_file(config_param_fname)
    em=UnsupervisedTransliteratorTrainer(config_params,lm_model)
    em.em_unsupervised_train(read_monolingual_corpus(fcorpus_fname),generate_char_set(ecorpus_fname))
    TransliterationModel.save_translit_model(em._translit_model, model_fname) 

def supervised_training(fcorpus_fname, ecorpus_fname, config_param_fname, lm_fname, model_fname):

    lm_model=load_lm_model(lm_fname)
    config_params=read_yaml_file(config_param_fname)
    em=UnsupervisedTransliteratorTrainer(config_params,lm_model)
    em.em_supervised_train(read_parallel_corpus(fcorpus_fname,ecorpus_fname))
    TransliterationModel.save_translit_model(em._translit_model, model_fname) 

def transliterate(translit_model_fname, lm_fname, fcorpus_fname, ecorpus_fname, n_processes=None):

    if n_processes is not None: 
        n_processes=int(n_processes)

    output=parallel_decode(TransliterationModel.load_translit_model(translit_model_fname), 
                                load_lm_model(lm_fname),
                                read_monolingual_corpus(fcorpus_fname),
                                n_processes
                              )

    write_monolingual_corpus(ecorpus_fname,output)

def transliterate_topn(translit_model_fname, lm_fname, fcorpus_fname, ecorpus_fname, topn,  n_processes=None):

    if n_processes is not None: 
        n_processes=int(n_processes)

    topn=int(topn)

    output=parallel_decode_topn(TransliterationModel.load_translit_model(translit_model_fname), 
                                load_lm_model(lm_fname),
                                read_monolingual_corpus(fcorpus_fname),
                                topn,
                                n_processes
                              )

    with codecs.open(ecorpus_fname,'w','utf-8') as ofile: 
        for i, ocand_list in enumerate(output): 
            for candidate, score in ocand_list: 
                ofile.write( u'{} ||| {} ||| {} ||| {}\n'.format( i, u' '.join(candidate), u' ', score  ) )


def log_likelihood(translit_model_fname, lm_fname, fcorpus_fname, ecorpus_fname, n_processes=None):

    if n_processes is not None: 
        n_processes=int(n_processes)

    likelihood=parallel_likelihood(TransliterationModel.load_translit_model(translit_model_fname), 
                                load_lm_model(lm_fname),
                                read_parallel_corpus(fcorpus_fname,ecorpus_fname),
                                n_processes
                              )

    print likelihood

if __name__=='__main__': 

    commands={
                'unsup_train':unsupervised_training,
                'sup_train':supervised_training,
                'transliterate':transliterate,
                'transliterate_topn':transliterate_topn,
                'log_likelihood':log_likelihood,
            }

    commands[sys.argv[1]](*sys.argv[2:])
