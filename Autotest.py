#Automatically makes similarity matrices and heat maps for given models.
#To work, all the word vector files (WV models, Autotest, plots, process, sims, and Splitter) need to be in the same folder, along with the input csv test_data and hnp_proc folders.

# import sys
import os
import Splitter
import sims
import plots
import WV_model

# INPUTPATH should be the name of the csv containing raw extractor output.
# INPUTPATH = "SP_v1_raw.csv"
INPUTPATH = "dow_test_v2_raw.csv"
# For each model that you want to make, put a tuple in MODELS_TO_TEST (tags, kind, threshold)
# MODELS_TO_TEST = [('all', 'boolean', 0.2),('all', 'boolean', 1.0),('all', 'freq', 0.2),('all', 'freq', 1.0),('all', 'tfidf', 0.2),('all', 'tfidf', 1.0),('nouns', 'boolean', 0.2),('nouns', 'boolean', 1.0),('nouns', 'freq', 0.2),('nouns', 'freq', 1.0),('nouns', 'tfidf', 0.2),('nouns', 'tfidf', 1.0)]
MODELS_TO_TEST = [('nouns', 'boolean', 0.2),('nouns', 'boolean', 1.0),('nouns', 'freq', 0.2),('nouns', 'freq', 1.0),('nouns', 'tfidf', 0.2),('nouns', 'tfidf', 1.0)]

for filename in os.listdir('test_data'):
    os.unlink('test_data/' + filename)
Splitter.splitfile(INPUTPATH)
for modelspecs in MODELS_TO_TEST:
    modelname = modelspecs[0] + "_" + modelspecs[1] + "_" + str(modelspecs[2])
    os.mkdir(modelname)
    # PLOTPATHS = [modelname + "/" + modelname + "_similarities.csv", modelname + "/" + modelname + "_similarities_shuffled.csv"]
    PLOTPATHS = [modelname + "/" + modelname + "_similarities_shuffled.csv"]
    ERRORFILE = modelname + "/" + modelname + "_errors.txt"
    EF = open(ERRORFILE, 'w')
    numfiles = len(os.listdir('test_data'))
    EF.write("Building " + modelname + " on " + INPUTPATH + " with " + str(numfiles) + " documents.\n")
    sims.run_sims(WV_model.make_model(modelspecs[0], modelspecs[1], modelspecs[2], EF, modelname), modelname, modelname, True)
    plots.run_plots(PLOTPATHS)
    EF.close()
    print('finished building ' + modelname)