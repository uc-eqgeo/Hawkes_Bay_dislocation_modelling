from dislocations.inversions import HawkeBayInversion
import numpy as np
import pygmo as pg
import pickle as pkl
import geopandas as gpd
import os
import shutil
import statistics
from upf_inversion_tools import get_rect_Mw

### USER: define peak slip, inversion variables, and which model names you want to run inversion for
peak_slip = 8      # in meters
displacement_weight = 1.e5
laplacian_weight = 2.e3
edge_weight = 1.e3
stop = 1

extension_list = ["lis_A1_r90"]

# extension_list = ["lis_A1_r90", "lis_A1_r135", "lis_A2_r90", "lis_A2_r135",
#                   "lis_B1_r90", "lis_B1_r135", "lis_B2_r90", "lis_B2_r135",
#                   "lis_C1_r90", "lis_C1_r135", "lis_C2_r90", "lis_C2_r135",
#                   "lis_F1_r90", "lis_F1_r135",
#                   "plnr_P1_r90", "plnr_P1_r135", "plnr_P2_r90", "plnr_P2_r135",
#                   "plnr_P3_r90", "plnr_P3_r135"]
###

def run_inversion(extension_list, peak_slip=8, displacement_weight=1.e5, laplacian_weight=2.e3,
                  edge_weight=1.e3):
    """runs the inversion for one fault geometry and outputs a preferred slip array
    Fiddle with displacement weight, laplacian weight, and edge weight to affect slip distributuon"""

    ### ## load variables from pickle file
    for extension_name in extension_list:
        with open(f"tiled_fault_dictionaries/tiled_fault_dict_{extension_name}.pkl", "rb") as fid:
            tiled_fault_dict = pkl.load(fid)

        fault_patch_size = tiled_fault_dict['fault_patch_size']
        tiled_fault = tiled_fault_dict['tiled_fault']

        # Green's functions: uplift/subsidence per 1m slip
        gf_array_AL = tiled_fault_dict['gf_array_AL']
        gf_array_CK = tiled_fault_dict['gf_array_CK']
        gf_grid = tiled_fault_dict['gf_grid']
        # Smoothing
        laplacian = tiled_fault_dict['laplacian']
        # Zero slip at fault edges
        edge_patches = tiled_fault_dict['edge_patches']

        # Set up inversion problem
        inversion = HawkeBayInversion(gf_array=gf_array_AL, laplacian=laplacian, edge_array=edge_patches,
                                      disp_weight=displacement_weight, laplacian_weight=laplacian_weight,
                                      edge_weight=edge_weight, max_slip=peak_slip)

        # Choose optimisation algorithm
        nl = pg.nlopt('slsqp')
        # Tolerance to make algorithm stop trying to improve result
        nl.ftol_abs = 1.

        # Set up basin-hopping metaalgorithm
        algo = pg.algorithm(uda=pg.mbh(nl, stop=stop, perturb=.4))
        # Lots of output to check on progress
        algo.set_verbosity(1)

        # set up inversion class to run algorithm on
        pop = pg.population(prob=inversion)

        # Initially set slip to peak slip on each patch
        initial_array = np.ones(gf_array_AL.shape) * peak_slip
        # Tell population object what starting values will be
        pop.push_back(initial_array)

        # Run algorithm
        pop = algo.evolve(pop)

        # Best slip distribution
        preferred_slip = pop.champion_x

        # Best subsidence value at Ahuriri Lagoon
        ahuriri_subsidence = np.dot(preferred_slip, gf_array_AL)
        print(f"Ahuriri Subsidence: {ahuriri_subsidence:.3f} m")
        # Corresponding uplift at Cape Kidnappers
        ck_uplift = np.dot(preferred_slip, gf_array_CK)
        print(f"Cape Kidnappers uplift: {ck_uplift:.3f} m")

        if not os.path.exists(f"geojson"):
            os.mkdir(f"geojson")

        # export slip patches to shapefile
        patch_variable_dict = {"dip": [np.round(patch.dip, 2) for patch in tiled_fault.patches],
                          #"tot_slp": ["{:.2f}".format(patch.total_slip) for patch in tiled_fault.patches]
                          "tot_slp": [np.round(value, 2) for value in preferred_slip]
                          #"ds": [f"{patch.dip_slip:.3f}" for patch in tiled_fault.patches],
                          #"ss": [f"{patch.strike_slip:.3f}" for patch in tiled_fault.patches]
                               }
        outlines = gpd.GeoDataFrame(patch_variable_dict, geometry=[patch.corner_polygon for patch in tiled_fault.patches],
                                    crs=2193)

        # Save patches as shapefiles
        outlines.to_file(f"geojson/patch_outlines_{extension_name}.geojson", index="FID",
                         driver="GeoJSON")

        ### Calculate magnitude
        # summed_slip, patches_with_slip = 0, 0
        # # solve for total slip
        # for slip_val in preferred_slip:
        #     if slip_val != 0:
        #         summed_slip += slip_val
        #         patches_with_slip += 1
        # # calculate average slip on patches that moved
        # avg_slip = summed_slip / patches_with_slip
        # m0 = 3.e10 * avg_slip * (patches_with_slip * fault_patch_size * fault_patch_size)
        # mw = (2. / 3.) * (np.log10(m0) - 9.05)
        # print("Mw = " + "{:.2f}".format(mw))

        Mw, avg_slip = get_rect_Mw(preferred_slip=preferred_slip, patch_edge_size=fault_patch_size)

        #get max and min deformation. disp grid is only vertical component
        disp_grid = np.sum(np.array([preferred_slip[i] * gf_grid[i] for i in range(len(preferred_slip))]), axis=0)
        min_vert, max_vert = disp_grid.min(), disp_grid.max()

        #get average dip of patches that moved
        moved_patches_dips = []
        for i in range(len(tiled_fault.patches)):
            if preferred_slip[i] != 0:
                dip = tiled_fault.patches[i].dip
                moved_patches_dips.append(dip)
        average_moved_dip = statistics.mean(moved_patches_dips)

        #save results to text file
        with open(f"out_data/results_{extension_name}.txt", "w") as f:
            f.write(f"Ahuriri Subsidence: {ahuriri_subsidence:.3f} m\n")
            f.write(f"Cape Kidnappers uplift: {ck_uplift:.3f} m\n")
            f.write(f"Mw = {Mw:.2f}\n")
            #f.write("average dip = " + "{:.2f}".format(statistics.mean(patch_dips)) + "\n")
            f.write(f"Minimum vertical def = {min_vert:.3f} m\n")
            f.write(f"Maximum vertical def = {max_vert:.3f} m\n")
            f.write(f"Average slip (on patches that slipped): {avg_slip:.3f} m\n")
            f.write(f"Average dip (on patches that slipped): {average_moved_dip:.3f} degrees")

        # save preferred slip array
        np.save(f"out_data/preferred_slip_array_{extension_name}", preferred_slip)

#### run scripts
run_inversion(extension_list=extension_list, peak_slip=peak_slip,
              displacement_weight=displacement_weight,
              laplacian_weight=laplacian_weight,
              edge_weight=edge_weight)


# write tiff of displacements
# gf_grid = np.load("gf_grid_" + extension + ".npy")
# xgrid = np.load("gf_grid_x_" + extension + ".npy")
# ygrid = np.load("gf_grid_y_" + extension + ".npy")
# disp_grid = np.sum(np.array([preferred_slip[i] * gf_grid[i] for i in range(len(preferred_slip))]), axis=0)
# option to write tiff of displacements
# write_tiff("displacements_" + extension + ".tif", xgrid, ygrid, disp_grid)
