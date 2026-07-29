import os
# Paths to input and output parent directory during a GA run
# Do not change
inputdir = 'input/'
outputdir = 'output/'

# Paths to input (datapath) and output (outpath) neccesary for
# running the GA_analysis.py script.
# - Your run output should be in a directory inside datapath_analysis
# - Output of the script will be put in a direcotry inside
#   outpath_analysis, that will be made by the script
MSc_project_path = '/Users/hulya/Desktop/Alles/UvA/MSc_project/Anja_NN/Statistical-Inference-for-Massive-Stars-MSc-Project'
KIWI_GA_path = os.path.join(MSc_project_path, 'KIWI-GA')
inpath = os.path.join(KIWI_GA_path, 'input')
outpath_analysis = os.path.join(KIWI_GA_path, 'output')
datapath_analysis = os.path.join(KIWI_GA_path, 'output')
fastwind_local = 'Dir/from/which/to/run/fastwind/'


# needed for running Anja's emulator
norm_path = os.path.join(MSc_project_path, 'normalisation.json')
master_wl_array_path = os.path.join(MSc_project_path, 'wavelengths.csv')
keras_model_path = os.path.join(MSc_project_path, 'NN_model.keras')

# saving generated spectra and MCMC results
MCMC_results_path = os.path.join(MSc_project_path, 'MCMC_results/')

# saving Nested Sampling results
nested_sampling_results_path = os.path.join(MSc_project_path, 'nested_sampling_results/')

# BLOeM
BLOeM_fits_path = os.path.join(MSc_project_path, 'BLOeM_4-074_Coadded.fits')

# filters
filter_path = os.path.join(MSc_project_path, 'filter_transmissions/')

# example input
example_input_path = os.path.join(KIWI_GA_path, 'example_input')

if not inputdir.endswith('/'):
    inputdir = inputdir + '/'
if not outputdir.endswith('/'):
     outputdir = outputdir + '/'
if not outpath_analysis.endswith('/'):
     outpath_analysis = outpath_analysis + '/'
if not datapath_analysis.endswith('/'):
     datapath_analysis = datapath_analysis + '/'
if not fastwind_local.endswith('/'):
     fastwind_local = fastwind_local + '/'
