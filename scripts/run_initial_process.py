"""
CL Spectroscopy Pipeline
Processes .sur files from a directory tree, separating spectra, CL images,
and SEM images, then applies standard preprocessing and analysis. 
Saves under mirrored folder structure under plots/.
"""

from sem_cl_analyzer import initial_process

initial_process.run()

