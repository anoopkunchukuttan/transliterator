#Copyright Anoop Kunchukuttan 2015 - present
# 
#This file is part of the IITB Unsupervised Transliterator 
#
#IITB Unsupervised Transliterator is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#IITB Unsupervised Transliterator  is distributed in the hope that it will be useful, 
#but WITHOUT ANY WARRANTY; without even the implied warranty of 
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
#GNU General Public License for more details. 
#
#You should have received a copy of the GNU General Public License 
#along with IITB Unsupervised Transliterator.   If not, see <http://www.gnu.org/licenses/>.

from cfilt.transliteration.analysis import transliteration_analysis as ta
import sys, codecs, yaml
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from indicnlp import langinfo

langs=['bn','gu','hi','pa','mr','kK','ml','ta','te']

def monolingual_analysis_all(indir,outdir): 
    """
    indir: contains monolingual files of the name: 'lang_code.txt'
    outdir: contains output files for each language: 'lang_code.pickle'
    """

    for lang in langs: 
        ta.monolingual_analysis('{}/{}.txt'.format(indir,lang),'{}/{}.yaml'.format(outdir,lang),lang)


def compare_character_ratios(mono_datadir): 

    ## read language data 
    lang_data=[]    
    for lang in langs: 
        with codecs.open('{}/{}.yaml'.format(mono_datadir,lang),'r','utf-8') as datafile:
            lang_data.append(yaml.load(datafile))

    ### Plot character ratios
    charratio_mat=np.zeros((len(langs),langinfo.COORDINATED_RANGE_END_INCLUSIVE-langinfo.COORDINATED_RANGE_START_INCLUSIVE+1))
    for i,lang in enumerate(langs): 
        for c,v in lang_data[i]['char_proportions'].iteritems(): 
            charratio_mat[i,c]=v

    ## plot 
    matplotlib.rc('font', family='Lohit Hindi')
    fig, ax = plt.subplots()

    plt.pcolor(charratio_mat,cmap=plt.cm.hot_r,edgecolors='k')
    plt.colorbar()

    plt.xticks(np.arange(0,charratio_mat.shape[1])+0.5,[ langinfo.offset_to_char(o,'hi') for o in xrange(langinfo.COORDINATED_RANGE_START_INCLUSIVE,langinfo.COORDINATED_RANGE_END_INCLUSIVE+1)])
    plt.yticks(np.arange(0,charratio_mat.shape[0])+0.5,xrange(len(langs)))
    
    plt.show()
    plt.close()


def compare_kl_divergence(mono_datadir): 

    ## read language data 
    lang_data=[]    
    for lang in langs: 
        with codecs.open('{}/{}.yaml'.format(mono_datadir,lang),'r','utf-8') as datafile:
            lang_data.append(yaml.load(datafile))

    # compute kl divergence
    kl_div_mat=np.zeros((len(langs),len(langs)))
    for i,langi in enumerate(langs):
        for j,langj in enumerate(langs): 
            kl_div_mat[i,j]=ta.kl_divergence(lang_data[i]['char_proportions'],lang_data[j]['char_proportions'])


    ## plot 
    fig, ax = plt.subplots()

    plt.pcolor(kl_div_mat,cmap=plt.cm.hot_r,edgecolors='k')
    plt.colorbar()

    plt.xticks(np.arange(0,kl_div_mat.shape[1])+0.5,langs)
    plt.yticks(np.arange(0,kl_div_mat.shape[0])+0.5,langs)
    
    plt.show()
    plt.close()

if __name__=='__main__': 

    commands={
                'monolingual_analysis_all': monolingual_analysis_all, 
                'compare_character_ratios': compare_character_ratios, 
                'compare_kl_divergence': compare_kl_divergence, 
            }
            
    commands[sys.argv[1]](*sys.argv[2:])
