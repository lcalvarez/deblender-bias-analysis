BiasAnalysis
============

Repository contains set of functions to run the deblender, simultaneous fitter, and fits to true
galsim objects using the python library, lmfit, in order to assess and compare bias of
different methods.

# InitializeAnalysis.py
This program runs through different separations
for two overlapping objects, runs a number of trials
in which two overlapping objects are simultaneously
fitted, deblended and then fitted, and then fitted 
to the true individual objects. The information is
then plotted and triangle plot and correlation plot 
files are saved to appropriate directories.

# BiasOverSeparationLibrary.py
Set of functions for running the deblender, simultaneous fitter
and fits to true objects to compare results and perform
bias analysis over a specific set of separations.

# deblender.py
Module containing the approximate LSST symmetrization deblender.
