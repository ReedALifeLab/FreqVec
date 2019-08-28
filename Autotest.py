#Automatically makes similarity matrices and heat maps for given models.
#To work, all the word vector files (WV models, Autotest, plots, process, sims, and Splitter) need to be in the same folder, along with the input csv test_data and hnp_proc folders.

# import sys
import os
import Splitter
import sims
import plots
import WV_model
import yearly_pad
# import SIC_Compiler

# INPUTPATH should be the name of the csv containing raw extractor output.
# INPUTPATH = "SP_v6_raw.csv"
INPUTPATH = "SP_v8_raw.csv"
# For each model that you want to make, put a tuple in MODELS_TO_TEST (tags, source, kind, threshold)
# MODELS_TO_TEST = [('all', 'boolean', 0.2),('all', 'boolean', 1.0),('all', 'freq', 0.2),('all', 'freq', 1.0),('all', 'tfidf', 0.2),('all', 'tfidf', 1.0),('nouns', 'boolean', 0.2),('nouns', 'boolean', 1.0),('nouns', 'freq', 0.2),('nouns', 'freq', 1.0),('nouns', 'tfidf', 0.2),('nouns', 'tfidf', 1.0)]
# MODELS_TO_TEST = [('sic', 'tfidf', 0.2),('sic', 'tfidf', 1.0),('sic', 'boolean', 0.2),('sic', 'boolean', 1.0),('sic', 'freq', 0.2),('sic', 'freq', 1.0)]
# MODELS_TO_TEST = [('nouns', 'tfidf', 1.0)]
MODELS_TO_TEST = [('nouns', 'sic', 'tfidf', 1.0), ('nouns', '10k', 'boolean', 1.0), ('nouns', '10k', 'tfidf', 1.0)]
# MODELS_TO_TEST = [('nouns', '10k', 'boolean', 0.2)]

PLOT_HEAT = False
PLOT_SHUFFLE = False
HEATMAP_LABELSIZE = 0.4

if not os.path.exists('test_data'):
    os.mkdir('test_data')
if not os.path.exists('hnp_proc'):
    os.mkdir('hnp_proc')
for filename in os.listdir('test_data'):
    os.unlink('test_data/' + filename)
Splitter.splitfile(INPUTPATH)

# for i in range(0, 10):
    # SIC_Compiler.compile_sic_groups("sic_descriptions", runID = str(i))

for modelspecs in MODELS_TO_TEST:
    modelname = modelspecs[0]+"_"+modelspecs[1] + "_" + modelspecs[2] + "_" + str(modelspecs[3])
    # modelname = modelspecs[0] + "_" + modelspecs[1] + "_" + str(modelspecs[2]) + "_" + INPUTPATH[:-8]
    outputpath = modelname
    # outputpath = "dow_v5_12_models"
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
    PLOTPATHS = []
    if PLOT_HEAT:
        PLOTPATHS.append(outputpath + "/" + modelname + "_similarities.csv")
    if PLOT_SHUFFLE:
        PLOTPATHS.append(modelname + "/" + modelname + "_similarities_shuffled.csv")
    ERRORFILE = outputpath + "/" + modelname + "_errors.txt"
    EF = open(ERRORFILE, 'w')
    numfiles = len(os.listdir('test_data'))
    EF.write("Building " + modelname + " on " + INPUTPATH + " with " + str(numfiles) + " documents.\n")
    sims.run_sims(WV_model.make_model(modelspecs[0], modelspecs[1], modelspecs[2], modelspecs[3], EF, outputpath), modelname, outputpath, PLOT_SHUFFLE, diag_value = 1.0)
    plots.run_plots(PLOTPATHS)
    if not os.path.exists(outputpath + "/" + "sims_by_year"):
        os.mkdir(outputpath + '/' + 'sims_by_year')
    yearly_pad.yearly_with_padding(outputpath + "/" + modelname + "_similarities.csv", outputpath + '/' + 'sims_by_year/' + modelname + "_similarities", 1994, 2019)
    EF.close()
    print('finished building ' + modelname)