from cfilt.transliteration.decoder import *
from cfilt.transliteration.utilities import *
from cfilt.transliteration.parallel_decoder import *
from cfilt.transliteration.unsupervised import *

from indicnlp import loader 

def unsupervised_training(fcorpus_fname, ecorpus_fname, config_param_fname, lm_fname, model_fname):

    config_params=read_yaml_file(config_param_fname)
    em=UnsupervisedTransliteratorTrainer(config_params)

    decoder_params=config_params.get('decoder_params',{})
    lm_model=load_lm_model(lm_fname,decoder_params.get('lm_order',TransliterationDecoder.DEFAULT_LM_ORDER))
    lm_model_rerank=load_lm_model(lm_fname,5)
    em.em_unsupervised_train(read_monolingual_corpus(fcorpus_fname),generate_char_set(ecorpus_fname),lm_model,lm_model_rerank)
    TransliterationModel.save_translit_model(em._translit_model, model_fname) 

def supervised_training(fcorpus_fname, ecorpus_fname, config_param_fname, model_fname):

    config_params=read_yaml_file(config_param_fname)
    em=UnsupervisedTransliteratorTrainer(config_params)
    em.em_supervised_train(read_parallel_corpus(fcorpus_fname,ecorpus_fname))
    TransliterationModel.save_translit_model(em._translit_model, model_fname) 

def transliterate(translit_model_fname, lm_fname, fcorpus_fname, ecorpus_fname, decoder_config_fname=None, n_processes=None):
    
    decoder_params={}
    if decoder_config_fname is not None: 
        decoder_params=read_yaml_file(decoder_config_fname)
    lm_order=decoder_params.get('lm_order',TransliterationDecoder.DEFAULT_LM_ORDER)

    if n_processes is not None: 
        n_processes=int(n_processes)

    output=parallel_decode(TransliterationModel.load_translit_model(translit_model_fname), 
                                load_lm_model(lm_fname,lm_order),
                                read_monolingual_corpus(fcorpus_fname),
                                decoder_params,
                                n_processes
                              )

    write_monolingual_corpus(ecorpus_fname,output)

def transliterate_topn(translit_model_fname, lm_fname, fcorpus_fname, ecorpus_fname, topn,  decoder_config_fname=None, n_processes=None):

    decoder_params={}
    if decoder_config_fname is not None: 
        decoder_params=read_yaml_file(decoder_config_fname)
    lm_order=decoder_params.get('lm_order',TransliterationDecoder.DEFAULT_LM_ORDER)

    if n_processes is not None: 
        n_processes=int(n_processes)

    topn=int(topn)

    output=parallel_decode_topn(TransliterationModel.load_translit_model(translit_model_fname), 
                                load_lm_model(lm_fname,lm_order),
                                read_monolingual_corpus(fcorpus_fname),
                                topn,
                                decoder_params,
                                n_processes
                              )

    #decoder=TransliterationDecoder(TransliterationModel.load_translit_model(translit_model_fname), 
    #                            load_lm_model(lm_fname,lm_order),
    #                            decoder_params
    #                          )
    #output= [ decoder.decode_topn_ngram(x,topn) for x in  read_monolingual_corpus(fcorpus_fname)  ]

    with codecs.open(ecorpus_fname,'w','utf-8') as ofile: 
        for i, ocand_list in enumerate(output): 
            for candidate, score in ocand_list: 
                ofile.write( u'{} ||| {} ||| {} ||| {}\n'.format( i, u' '.join(candidate), u' ', score  ) )


def log_likelihood_unsupervised(translit_model_fname, lm_fname, fcorpus_fname, ecorpus_fname, decoder_config_fname=None, n_processes=None):

    decoder_params={}
    if decoder_config_fname is not None: 
        decoder_params=read_yaml_file(decoder_config_fname)
    lm_order=decoder_params.get('lm_order',TransliterationDecoder.DEFAULT_LM_ORDER)

    if n_processes is not None: 
        n_processes=int(n_processes)

    likelihood=parallel_likelihood_unsupervised(TransliterationModel.load_translit_model(translit_model_fname), 
                                load_lm_model(lm_fname,lm_order),
                                read_parallel_corpus(fcorpus_fname,ecorpus_fname),
                                decoder_params,
                                n_processes
                              )

    print likelihood

if __name__=='__main__': 

    loader.load()

    commands={
                'unsup_train':unsupervised_training,
                'sup_train':supervised_training,
                'transliterate':transliterate,
                'transliterate_topn':transliterate_topn,
                'log_likelihood_unsup':log_likelihood_unsupervised,
            }

    commands[sys.argv[1]](*sys.argv[2:])
