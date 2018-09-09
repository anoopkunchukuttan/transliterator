# IITB Unsupervised Transliterator

**Documentation is not yet complete, please mail me if you need information about using the system. The documentation should be complete in a week** 

Unsupervised transliteration system which uses phoentic features to define transliteration priors.  This is an EM based method which builds on Ravi and Knight's 2009 work. In addition, self training is used to iteratively build a substring based transliteration system in order to incorporate contextual information. 

## Pre-requisites

- [Indic NLP Library] (http://anoopkunchukuttan.github.io/indic_nlp_library)
- [nwalign 0.3.1](https://pypi.python.org/pypi/nwalign/): for Needleman-Wunsch global sequence alignment

## Usage

To train the complete model, follow the steps mentioned below:  

0. Prepare environment, training and test data

  - Set the environment variable `XLIT_HOME` to the directory where this repository has been cloned
        export XLIT_HOME=<repo_directory>
  - Create an directory to house training and test files (henceforth referred to as `$corpusdir`)
  - The training files contain monolingual lists of names for source and target languages in one line per name format (delimited by space). The files should be named `train.${srclang}` and `train.${tgtlang}`
  - There will be two test files named (`test.${srclang}` and `test.${tgtlang`}. These will contain the parallel names in source and target languages respectively in tne one line per name format (delimited by space).

2. Train a character-based model 
   
   - Create a character-level bigram language model using the target language corpus in ARPA format with your favourite language modelling toolkit
   - Create experiment configuration file  (See a sample file here:  `test/sample_corpus/supervised.yaml`)
   - Create experiment details file  (See a sample file here:  `test/sample_corpus/sample_ems.config`)
   - Run the following command: 
          $XLIT_HOME/scripts/ems.sh <path to experiment details file>

3. Re-rank the training set transliteration (based on the character-based model) with a higher-order language model 
   - Create a character-level 5-gram language model using the target language corpus in ARPA format with your favourite language modelling toolkit
   - Decode the training set source language corpus 
       ./decode_training.sh <path to experiment details file>  <synthesized\_corpus\_directory>
where, `synthesized_corpus_directory` will contain the pseudo-parallel corpus required for the next stage       


4. Train a substring-based model: 
    Use the pseudo parallel corpus created in the previous stage to train a supervised model. You can use Moses or any other translation/transliteration system to train the transliteration model. We used Moses in our experiments with monotonic reordering. Please see the paper for details of Moses settings. In order to apply self-learning, you can decode the training set using the newly trained substring level model and generate a new pseudo-parallel corpus. You can apply self-learning for k iterations till convergence or a fixed upper limit. In practice, we found 5 iterations to be sufficient for convergence.

## Papers

The details of the unsupervised transliterator are described in: 

Anoop Kunchukuttan, Pushpak Bhattacharyya, Mitesh Khapra. _Substring-based unsupervised transliteration with phonetic and
contextual knowledge_. Conference on Natural Language Learning (CoNLL). 2016.

## Contact

Anoop Kunchukuttan: <anoop.kunchukuttan@gmail.com>

## LICENSE

Copyright Anoop Kunchukuttan 2015 - present
 
IITB Unsupervised Transliterator is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

IITB Unsupervised Transliterator  is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
GNU General Public License for more details. 

You should have received a copy of the GNU General Public License 
along with IITB Unsupervised Transliterator.   If not, see <http://www.gnu.org/licenses/>.

