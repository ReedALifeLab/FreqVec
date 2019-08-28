This directory contains code for making word vector models.
In order to run the code, you need to put all of the .py files into the directory from which you plan to run the code. The code takes as input a CSV document with the format id, text. For instance:

id1, hello world
id2, I am doing data science
 .           .
 .           .
 .           .

This document is the input file. Each line of the input file represents a single document, for which the model will construct an associated vector. The input file should be put into the same directory as the model code. You can move it to the folder on unix
and linux systems via:

                   mv [path/to/file] . 

If you plan to make a model with the source "sic", you need to put a folder named sic_descriptions inside this directory with all of the SIC descriptions you want the model to use. Previously the code needed other folders (test_data and hnp_proc) in order to run, but these can now be constructed by the code if they are needed.
To control the code, you edit the global variables in Autotest.py. Open it with your favorite text editor. There are five variables you need to set (not listed in the order that they appear):

1. INPUTPATH - set this to the name of the input file.

2., 3. & 4. PLOT_HEAT, PLOT_SHUFFLE, and HEATMAP_LABELSIZE - if you want to make a heatmap of your model, set PLOT_HEAT to True. If you want to make a randomly shuffled version of the heatmap, set PLOT_SHUFFLE to True. If you make a heatmap, you can control the text size by changing HEATMAP_LABELSIZE. Making heatmaps takes longer than building the model, and isn't suggested for input data approaching or exceeding 10,000 firms.

4. MODELS_TO_TEST - this is a list of sets of parameters with one entry for every model you want to make. Each set of parameters is a 4-tuple with the following entries:
    0. tags - "words" or "nouns". If set to "nouns", all other words will be discarded from the model's vocabulary. "Words" will include all words in the source.
    1. source - "10k" or "sic". If set to "10k", the model will construct its vocabulary from the input documents. "sic" tells it to use only words appearing in documents inside the sic_descriptions folder.
    2. kind - "boolean", "tfidf", or "freq". This sets the rule according to which a document's vector is normalized.
        boolean - each word in the vocabulary is marked with a 0 or a 1, with 1 indicating that it appears in the document. The resulting vector is then normalized.
        freq - each word in the vocabulary is marked with the number of times it appears in the document. The resulting vector is then normalized.
        tfidf - each word in the vocabulary is marked with the number of times it appears in the document. Common words then have their values reduced, and the resulting vector is normalized.
    3. threshold - a floating point number between 0 and 1. Words that appear in MORE than this frequency of documents will be discarded from the model's vocabulary. If the source is "sic", words in the vocabulary are drawn from files in the sic_descriptions folder, but the frequency of words is still determined from the input documents.

So for instance if you wanted to build a tfidf model with all of 
the words in the documents you would type:

MODELS_TO_TEST = [('all',tfidf,1.0)]

If you wanted to also make a model boolean model with only nouns 
then you would type:

MODELS_TO_TEST = [('all',tfidf,1.0),('nouns','tfidf',1.0)]

etc.

Once you have all the .py files together, the input CSV with them, the parameters in Autotest.py set, and the sic_descriptions folder made (if needed), you can then build your model by running:

            python3 Autotest.py

The models you built will each be placed into a separate directory titled with the model's paramters as written in MODELS_TO_TEST. Each of those directories will contain the dictionary, vectors, and similarity matrix for the model, as well as a folder named sims_by_year which contains a series of smaller similarity matrices representing the similarities only in a given year.

-------------------------------------------------------------------------

Errors:

If you get an error which looks like this:

FileNotFoundError: [Errno 2] No such file or directory: 'sic_descriptions'

Then you're trying to build a model with "sic" as the source, but don't have an sic_descriptions folder. Make this folder and put the files containing the alternate vocabulary in there.

--------------------------------------------------------------------------

For other functionality email:

James Mccaull    at jamccaull@reed.edu
Tobias Rubel     at rubelato@reed.edu
Ananthan Nambiar at annambiar@reed.edu

and we will sort out your problem. 

--------------------------------------------------------------------------







