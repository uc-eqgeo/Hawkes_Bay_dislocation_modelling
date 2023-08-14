import numpy as np
import matplotlib.pyplot as plt
from dislocations.faults import ListricFault
from dislocations.displacements import DisplacementTable, DisplacementGrid
import pickle as pkl
import os

# makes fault patches for each fault geometry and calculates green's functions
# outputs fault patches with greens function data as pickle file for each geometry

###### user inputs
fault_patch_size = 3.e3         # in meters
rake_list = [90., 135.]         # reverse = 90; RL = 180; oblique = 135
disp_grid_size = 2000.           # (meters) make larger if you want to process faster at low resolution.
#########

#load fault geometries
with open("out_data/fault_geometries.pkl", "rb") as fid:
    fault_geometries_list = pkl.load(fid)

#make output directory for tiled fault dictionaries
if not os.path.exists(f"tiled_fault_dictionaries"):
    os.mkdir(f"tiled_fault_dictionaries")

all_tiled_faults = []
for rake_val in rake_list:
    rake = rake_val
    for fault_geom_dict in fault_geometries_list:
        patch_heights, patch_dips = fault_geom_dict['patch_heights'], fault_geom_dict['patch_dips']
        cen_x, cen_y = fault_geom_dict['trace_center_x'], fault_geom_dict['trace_center_y']
        top_cen_z = 0.
        strike = fault_geom_dict['strike']
        fault_length = 80.e3  # takes into account taper on ends, actual rupture is length - 2(taper)
        extension = fault_geom_dict['geom_extension']

        # Create center fault from top to bottom
        centre_fault = ListricFault.from_depth_profile_square_tile(tile_size=fault_patch_size, patch_heights=patch_heights,
                                                                   patch_dips=patch_dips, top_centre_x=cen_x,
                                                                   top_centre_y=cen_y, top_centre_z=top_cen_z,
                                                                   strike=strike)

        # get cross-section vertices (shapely linestring)
        along_dip_line = centre_fault.cross_section_linestring()

        # Replicate fault patches along strike (e.g., 5 new patches in each direction)
        added_patches = round(fault_length/(2 * fault_patch_size) - 1)
        tiled_fault = centre_fault.tile_along_strike(added_patches)
        # Create laplacian smoothing matrix for fault model
        tiled_fault.compute_laplacian(double=False)

        # Displacements at Ahuriri and Cape Kidnappers
        disps_AL = DisplacementTable(tiled_fault.patches, np.array([1932228.]), np.array([5628830.]))
        disps_CK = DisplacementTable(tiled_fault.patches, np.array([1945817.]), np.array([5601485.]))

        # Area of interest
        x1, y1, x2, y2 = (1855000.0, 5382500.0, 2130000.0, 5827500.0)
        disp_grid = DisplacementGrid.from_bounds(tiled_fault.patches, x1, x2, disp_grid_size, y1, y2)

        # Calculate greens functions to give subsidence at Ahuriri for each patch
        gf_grid = disp_grid.greens_functions_array(rake=rake, vertical_only=True)
        gf_array_AL = disps_AL.greens_functions_array(rake=rake, vertical_only=True).flatten()
        gf_array_CK = disps_CK.greens_functions_array(rake=rake, vertical_only=True).flatten()

        # Array of ones and zeros to identify edge patches
        edge_patches = tiled_fault.edge_patch_bool(top_edge=True)

        tiled_fault_dict = {'model_extension': extension, 'tiled_fault': tiled_fault, 'gf_array_AL': gf_array_AL,
                            'gf_array_CK': gf_array_CK, 'gf_grid': gf_grid, 'gf_grid_x': disp_grid.x_values,
                            'gf_grid_y': disp_grid.y_values, 'laplacian': tiled_fault.laplacian,
                            'edge_patches': edge_patches, 'rake': rake, 'fault_patch_size': fault_patch_size,
                            'along_dip_line': along_dip_line}


        with open(f"tiled_fault_dictionaries/tiled_fault_dict_{extension}_r{int(rake)}.pkl", "wb") as fid:
            pkl.dump(tiled_fault_dict, fid)
        print(f"finished {extension}_r{int(rake)} pickle")
