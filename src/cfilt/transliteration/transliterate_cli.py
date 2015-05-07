from cfilt.transliteration.decoder import *
from cfilt.transliteration.utilities import *
from cfilt.transliteration.parallel_decoder import *
from cfilt.transliteration.unsupervised import *

def unsupervised_training(fcorpus_fname, ecorpus_fname, lm_fname, model_fname):

    lm_model=load_lm_model(lm_fname)
    em=UnsupervisedTransliteratorTrainer(lm_model)
    em.em_unsupervised_train(read_monolingual_string_corpus(fcorpus_fname),generate_char_set(ecorpus_fname))
    TransliterationModel.save_translit_model(em._translit_model, model_fname) 

def supervised_training(fcorpus_fname, ecorpus_fname, lm_fname, model_fname):

    lm_model=load_lm_model(lm_fname)
    em=UnsupervisedTransliteratorTrainer(lm_model)
    em.em_supervised_train(read_parallel_string_corpus(fcorpus_fname,ecorpus_fname))
    TransliterationModel.save_translit_model(em._translit_model, model_fname) 

def transliterate(translit_model_fname, lm_fname, fcorpus_fname, ecorpus_fname, n_processes=None):

    if n_processes is not None: 
        n_processes=int(n_processes)

    output=parallel_decode_char_string(TransliterationModel.load_translit_model(translit_model_fname), 
                                load_lm_model(lm_fname),
                                read_monolingual_string_corpus(fcorpus_fname),
                                n_processes
                              )

    write_monolingual_string_corpus(ecorpus_fname,output)

if __name__=='__main__': 

    commands={
                'unsup_train':unsupervised_training,
                'sup_train':supervised_training,
                'transliterate':transliterate,
            }

    commands[sys.argv[1]](*sys.argv[2:])
