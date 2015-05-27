Evaluation script news_evaluation.py reads transliteration results file and test file in XML format according to the specification in the NEWS 2010 whitepaper. The following metrics are calculated:
1) ACC: Accuracy in top-1, or word error rate.
2) Fuzziness in top-1, or mean F-score.
3) Mean Reciprocal Rank (MRR)
4) Mean Average Precision MAP_ref

For detailed description of each metric please refer to the NEWS 2010 whitepaper.

Running the script requires Python interpreter. The script was tested on Python 2.5; it likely works in the earlier versions as well. Python is usually supplied with the operating system in UNIX-like systems, including Mac OS X and various distributions of Linux. Windows users can <a href="http://www.python.org/ftp/python/2.6.1/python-2.6.1.msi">download the Python installer</a> from the official Python website.

Two included files are news_results.xml and news_test.xml. 

The former is a sample result file with transliterations for 4 words. Note that even though there are more than 10 transliteration candidates per name in the file, the evaluation script will only consider the first 10. This can be changed by passing --max-candidates argument to the script; however, maximum 10 candidates will be considered in NEWS 2010 evaluation.

The latter is a sample test file. The format of the test data file is the same as that of the training and development data files.

Evaluation of news_results.xml is done by running:
python news_evaluation.py -i news_results.xml -t news_test.xml

All 6 metrics will be printed to the standard output.

Running the evaluation script with -h or --help argument reveals more options.

Optionally, the detailed evaluation results for each source name can be saved into a comma-separated values file by passing -o filename option to the script:
python news_evaluation.py -i news_results.xml -t news_test.xml -o evaluation_details.csv

Note that ACC, Fuzziness in top-1, MRR, and MAP values for each source word are not divided by the number of source words. For example, the value of ACC will be either 0 or 1.

If you have any questions regarding the evaluation script or would like to report a bug, please contact Dr. Vladimir Pervouchine at vpervouchine@i2r.a-star.edu.sg
