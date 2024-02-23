This repository contains the code used to generate the data shown in the manuscript "Spatial analysis identifies PD-L1 expressing DC niches are predictors of immunotherapy outcomes in head and neck squamous cell cancer".

The pipeline is split in four parts:
1. Illumination correction: Correcting raw images for uneven illumination
2. Alignment, quench subtraction, segmentation and data extraction: Extracting cell level quantitative information for further analysis
3. Cell phenotyping: Assigning cellular phenotypes and perform subtyping
4. Spatial processing: Extract spatial metrics

For details on the approach for each of these points, see extended methods section.

For each part, the "_functions.py" file contains the lower level functions and data structures necessary for data processing,
the "_processing.py" file contains higher level functions directly involved in processing, required import commands and variables, as well as the data processing code itself.