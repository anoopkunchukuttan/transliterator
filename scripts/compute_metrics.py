import sys, codecs
from cfilt.transliteration.news2015_utilities import *
from cfilt.transliteration.analysis import align   

#fname=sys.argv[1]
#
#with codecs.open(fname,'r','utf-8') as infile: 
#
#    print '|'.join(['exp_no','src','tgt','ACC','Mean F-scor','MRR','MAP_ref','ACC@10'])
#
#    for line in infile: 
#        line=line.strip()
#        fields=line.split('/')
#
#        exp_no=fields[1]
#        src=fields[2].split('-')[0]
#        tgt=fields[2].split('-')[1]
#
#        try: 
#            with codecs.open(line,'r','utf-8') as resfile:
#                scores=[ l.strip().split(':')[1].strip()  for l in resfile.readlines()] 
#                #acc=scores[0] 
#                #fsc=scores[1] 
#                #mrr=scores[2] 
#                #maps=scores[3] 
#                #acc_10=scores[4]
#
#                print '|'.join([exp_no,src,tgt]+scores)
#        except Exception as e:
#            print 'Error gathering scores for {}-{}: {}'.format(src,tgt,exp_no)
#            print e
        

def g_evaluation_results(results_dir, exp_list, ref_dir,lang_list,res_fname): 

    with codecs.open(res_fname,'w','utf-8') as res_file: 
        #res_file.write('|'.join(['exp_no','src','tgt','ACC','Mean F-score','MRR','MAP_ref','ACC@10','Av-Entropy','Fuzzy-ACC','Av-Err-Rate'])+'\n')
        res_file.write('|'.join(['exp_no','src','tgt','ACC','Mean F-score','MRR','MAP_ref','ACC@10'])+'\n')

       # with codecs.open(lang_list_fname,'r','utf-8') as lang_list_file: 
       #     for line in lang_list_file: 
       #         # read fields
       #         fields=line.strip().split('-')
       #         src=fields[0]
       #         tgt=fields[1]

        for src, tgt in lang_list: 

            for exp_no in exp_list: 
                scores=[]

                ####  character level 
                #try: 
                #    eval_fname='{0}/{1}/{2}-{3}/evaluation/test.nbest.reranked.{3}.eval'.format(results_dir,exp_no,src,tgt)
                #    with codecs.open(eval_fname,'r','utf-8') as evalfile:
                #        scores.extend([ l.strip().split(':')[1].strip()  for l in evalfile.readlines()])
                #except Exception as e:
                #    scores.extend(['-1.0']*5)
                #    print 'Error gathering scores for {}-{}: {}'.format(src,tgt,exp_no)
                #    print e

                ###  phrase level 
                try: 
                    eval_fname='{0}/{1}/{2}-{3}/evaluation/test.nbest.reranked.{3}.eval'.format(results_dir,exp_no,src,tgt)
                    with codecs.open(eval_fname,'r','utf-8') as evalfile:
                        scores.extend([ l.strip().split(':')[1].strip()  for l in evalfile.readlines()])
                except Exception as e:
                    scores.extend(['-1.0']*5)
                    print 'Error gathering scores for {}-{}: {}'.format(src,tgt,exp_no)
                    print e

                #### average entropy 
                #try: 
                #    model_fname='{0}/{1}/{2}-{3}/model/translit.model'.format(results_dir,exp_no,src,tgt)
                #    scores.append(str(compute_entropy(model_fname)))
                #except Exception as e:
                #    scores.append('-1.0')
                #    print 'Error gathering scores for {}-{}: {}'.format(src,tgt,exp_no)
                #    print e

                ### fuzzy match score & ave. no of errors per word
                #try: 
                #    hyp_list=read_monolingual_corpus('{0}/{1}/{2}-{3}/evaluation/test.{3}'.format(results_dir,exp_no,src,tgt))
                #    ref_list=read_monolingual_corpus('{0}/{1}-{2}/test.{2}'.format(ref_dir,src,tgt))
                #    alignments=align.align_transliterations(ref_list,hyp_list,tgt)

                #    correct_flag_list=[]
                #    nmismatches_list=[]

                #    for ref_w, (ref_a, hyp_a) in itertools.izip(ref_list,alignments): 

                #        nmismatches=len(filter(lambda x: x[0]!=x[1], zip(ref_a,hyp_a))) 
                #        nmismatches_list.append(float(nmismatches))

                #        if len(ref_w)<=3:
                #            if ref_a==hyp_a:
                #                correct_flag_list.append(1.0)
                #            else: 
                #                correct_flag_list.append(0.0)
                #        else: 
                #            if nmismatches/float(len(ref_w))<=0.2: 
                #                correct_flag_list.append(1.0)
                #            else: 
                #                correct_flag_list.append(0.0)

                #    scores.append(str( sum(correct_flag_list)/float(len(ref_list)) ))
                #    scores.append(str( sum(nmismatches_list)/float(len(ref_list)) ))

                except Exception as e:
                        scores.extend(['-1.0','-1.0'])
                        print 'Error gathering Fuzzy match for {}-{}: {}'.format(src,tgt,exp_no)
                        print e 

                res_file.write('|'.join([exp_no,src,tgt]+scores)+'\n')

def evaluation_results_from_file_list(eval_list_fname, res_fname): 

    with codecs.open(res_fname,'w','utf-8') as res_file: 
        res_file.write('|'.join(['Exp_Info','ACC','Mean F-score','MRR','MAP_ref','ACC@10'])+'\n')

        with codecs.open(eval_list_fname,'r','utf-8') as eval_list_file: 

            for line in eval_list_file:                 
                scores=[]
                eval_fname=line.strip()
                try: 
                    with codecs.open(eval_fname,'r','utf-8') as evalfile:
                        scores.extend([ l.strip().split(':')[1].strip()  for l in evalfile.readlines()])
                except Exception as e:
                    scores.extend(['-1.0']*5)
                    print 'Error gathering scores for {}-{}: {}'.format(eval_fname)
                    print e

                res_file.write('|'.join([eval_fname]+scores)+'\n')

if __name__ == '__main__': 
#    exp_list=['pb',  'f_lm1', 'f_lm2',  'f_lm3',  'f_lm4', 'f_lm5', 
#                       'f_lm6', 'f_lm7', 'f_lm8', 'f_lm9', 
#                       'f_lm_1+4+5_c1', 'f_lm_1+2+5+7+8_c1',  'f_lm_1+2_c1',
#                       'f_lm_1+4+5_c2', 'f_lm_1+2+5+7+8_c2',  'f_lm_1+2_c2','f_lm_all_c2',
#                       'f_lm_all_c3',
#                       'f_lm_all_c4',
#                       ]

    #exp_list=['20_0', '20_3', '20_3_2', '20_3_3', '20_4_3', '20_4_4',  '20_7', '20_7_2', '20_9', '20_9_2']
    #lang_list=[('bn','hi'),('kn','hi'), ('hi','kn'), ('kn','ta'), ('ta','kn'), ('ta','hi'), ('hi','ta')]
    #g_evaluation_results(sys.argv[1],exp_list,sys.argv[2],lang_list,sys.argv[3])

    evaluation_results_from_file_list(sys.argv[1],sys.argv[2])
