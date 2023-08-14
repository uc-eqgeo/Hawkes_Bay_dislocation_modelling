# 
Scripts for inverting for maximum subsidence at Ahuriri Lagoon.

## Scripts
Scripts are all in the `HB_modelling` directory. Thre are three types of models: upper plate inversion, upper plate 
forward, and subduction zone forward.

# Upper plate inversion (/hb_upf_inversion)
There are four main scripts that should be run in the following order:
1. `upf_inversion_define_fault_geometries.py` This script defines the fault shapes and traces, and saves a pickle 
   file to an out_data folder.
2. `upf_inversion_create_fault_model.py` This script makes fault patches for each fault geometry and calculates 
   green's functions. It outputs fault patches with greens function results as a pickle file. User can define fault 
   patch size and multiple rake values to loop over.
3. `upf_run_inversion.py` Calculates preferred slip distrbution for maximum Ahuriri Lagoon subsidence. Outputs a 
   preffered slip array (.npy), summary of results (.txt), and patch outlines with slip (.geojson).
4. `upf_inversion_make_figures.py` Calculates surface displacements and writes to png or pdf file.

`upf_inversion_tools.py` Additional functions used in the above scripts.

# Upper plate forward modeling (/hb_upf_forward_models)
There is one main script to run:
1. `upf_run_forward_model.py` Calculates surface displacements and writes to a figure file. Pickle file with slip 
   data on each tiled fault is also saved. Can define slip taper along strike, dip, or both. 

`upf_forward_tools.py` Additional functions used in the main script.
`upf_forward_make_figures.py` Additional functions used to make figures in the main script.

# Subduction zone forward modeling (/hb_sz_forward_models)
There is one main script to run:
1. `hsz_run_forward_model.py` Calculates surface displacements and writes to a figure file. Pickle file with slip 
   data on each tiled fault, slip triangle geojson, and results text file is also saved to a results folder. Slip 
   patches are defined by a geojson polygon file. Can define slip taper amount from the edge of the slip patch. 

need mesh and rake data from vtk file (e.g., Williams et al., 2013) to run this script.
`hsz_forward_tools.py` Additional functions used in the main script.
`hsz_make_figures.py` Additional functions used to make figures in the main script.