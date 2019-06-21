#Automatically makes similarity matrices and heat maps for freq_vec. Needs to be run with the name of the raw output from Tobias' cleaner as a command line argument.
#As an optional second argument, you can make output files with a specific suffix - useful if you want to test on several datasets in a row.
import sys
import os
import Splitter
import freq_vec
import sims
import plots
import hnp_vec

prefix = sys.argv[2]

os.mkdir(prefix)

# PLOTPATHS = [prefix + "/freqvec_similarities.csv", prefix + "/HandP_similarities.csv", prefix + "/HandP_similarities_null.csv", prefix + "/freqvec_similarities_null.csv"]
PLOTPATHS = [prefix + "/HandP_similarities.csv", prefix + "/freqvec_similarities.csv"]
ERRORFILES = [prefix + "/freqvec_errors.txt", prefix + "/HandP_errors.txt"]
# ERRORFILES = [prefix + "/freqvec_errors.txt"]
# ERRORFILES = [None, prefix + "/HandP_errors.txt"]
# MAKENULL = True
MAKENULL = False

for filename in os.listdir('test_data'):
    os.unlink('test_data/' + filename)

Splitter.splitfile(sys.argv[1])
EFs = []
for ERRORFILE in ERRORFILES:
    if ERRORFILE is not None:
        EF = open(ERRORFILE, 'w')
        numfiles = len(os.listdir('test_data'))
        EF.write("Building test on " + sys.argv[1] + " with " + str(numfiles) + " documents.\n")
    else:
        EF = None
    EFs += [EF]

sims.run_sims(freq_vec.run_freq_vec(prefix, EFs[0]), "freqvec", prefix, MAKENULL)
sims.run_sims(hnp_vec.run_hnp_vec(prefix, EFs[1]), "HandP", prefix, MAKENULL)
plots.run_plots(PLOTPATHS)
for EF in EFs:
    if EF is not None:
        EF.close()