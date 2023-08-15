# Dislocation models for Hawke's Bay, Aotearoa NZ
This respository contains scripts used in "Upper plate faults may contribute to the paleoseismic subsidence record along
the central Hikurangi subduction zone, Aotearoa New Zealand" by Delano et al., (2023) in the journal Geochemistry, 
Geophysics, Geosystems. 

The "HB_modelling" directory includes the workflow scripts for running forward and inversion models and generating 
figures. Note that a vtk mesh file with rake data is required for the subduction zone forward model (e.g., see 
Williams et al., 2012).


The "dislocation_models" directory includes additional dislocation functions used in HB_modelling. 

More workflow-specific info is included in a second README in the "HB_modelling" directory.

## Installation notes

The required packages can be downloaded by installing the conda environment file "environment.yml" using the following
command:

```
conda env create -f environment.yml
```
For more information on conda environments, see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html .

Note that for the Okada code to work on Windows, the VS2014 (or later) C++ build tools must be installed. See https://visualstudio.microsoft.com/visual-cpp-build-tools/ for more information.
