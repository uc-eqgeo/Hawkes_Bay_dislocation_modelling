from dislocations.faults import ListricFault
from dislocations.displacements import DisplacementGrid, DisplacementTable
import numpy as np
from shapely.geometry import Point, LineString
import os
import pickle as pkl
from upf_forward_make_figures import make_figure

# INPUTS
#######
extension1 = "forward"              # used for naming files
taper_along_dip = True              # True: taper slip along dip, False: uniform slip value along dip
taper_along_strike = False          # False: uniform slip value along strike
move_updip_extent = False           # True: shifts updip slip patch extent, False: ruptures to surface
rake = 90                           # degrees: 0 LL, 90 reverse, 180 RL, -90 normal
disp_grid_size = 2000.              # resolution of grid for calculating surface displacements (meters)

# define fault geometry
cen_x, cen_y, cen_z = 1965700, 5598000, 0.      # center point coordinates (NZTM 2000)
strike = 227                                    # azimuth and follows right hand rule (degrees)
fault_length = 80.e3                            # (meters)
slip = 8                                        # slip: (meters), peak slip if tapering.
fault_patch_size = 3.e3                         # defines square edge length for tiles that make up the fault

# Set boundary of area of interest. Square, ideally. In meters.
x1, y1, x2, y2 = (cen_x - 11.e4, cen_y - 11.e4, cen_x + 11.e4, cen_y + 11.e4)

# define listric geometry
patch_heights = np.array([1.e3, 1.e3, 4.e3, 15.e3, 2.e3])       # in meters
patch_dips = np.array([80., 60., 38., 22., 14.])                # in degrees

### define slip taper and offset parameters you want to investigate
# taper_dist_km: distance (kilometers) over which slip tapers from peak slip to zero. Can be applied to up-dip or
# down-dip edge.
# taper_offset_km: distance (kilometers) along fault from down-dip edge. Defines where slip patch begins.
# updip_offset_km = distance in kilometers from the updip fault edge that slip begins.

# uniform slip
uniform_dict = {'name': 'uniform_0a', 'taper_dist_km': 0., 'taper_offset_km': 0., 'along_dip_slip_vals': []}
uniform_dict_list = [uniform_dict]

# These dictionaries taper slip along dip at donw-dip slip patch edge, with varying offsets
taper_downdip_9a_dict = {'name': 'downdip_9a', 'taper_dist_km': 9., 'taper_offset_km': 0., 'along_dip_slip_vals': []}
taper_downdip_9b_dict = {'name': 'downdip_9b', 'taper_dist_km': 9., 'taper_offset_km': 6., 'along_dip_slip_vals': []}
taper_downdip_9c_dict = {'name': 'downdip_9c', 'taper_dist_km': 9., 'taper_offset_km': 12., 'along_dip_slip_vals': []}
taper_downdip_9d_dict = {'name': 'downdip_9d', 'taper_dist_km': 9., 'taper_offset_km': 18., 'along_dip_slip_vals': []}
taper_downdip_12a_dict = {'name': 'downdip_12a', 'taper_dist_km': 12., 'taper_offset_km': 0., 'along_dip_slip_vals': []}
taper_downdip_12b_dict = {'name': 'downdip_12b', 'taper_dist_km': 12., 'taper_offset_km': 6., 'along_dip_slip_vals': []}
taper_downdip_12c_dict = {'name': 'downdip_12c', 'taper_dist_km': 12., 'taper_offset_km': 12., 'along_dip_slip_vals': []}
taper_downdip_12d_dict = {'name': 'downdip_12d', 'taper_dist_km': 12., 'taper_offset_km': 18., 'along_dip_slip_vals': []}
taper_downdip_15a_dict = {'name': 'downdip_15a', 'taper_dist_km': 15., 'taper_offset_km': 0., 'along_dip_slip_vals': []}
taper_downdip_15b_dict = {'name': 'downdip_15b', 'taper_dist_km': 15., 'taper_offset_km': 6., 'along_dip_slip_vals': []}
taper_downdip_15c_dict = {'name': 'downdip_15c', 'taper_dist_km': 15., 'taper_offset_km': 12., 'along_dip_slip_vals': []}
taper_downdip_15d_dict = {'name': 'downdip_15d', 'taper_dist_km': 15., 'taper_offset_km': 18., 'along_dip_slip_vals': []}
taper_downdip_18a_dict = {'name': 'downdip_18a', 'taper_dist_km': 18., 'taper_offset_km': 0., 'along_dip_slip_vals': []}
taper_downdip_18b_dict = {'name': 'downdip_18b', 'taper_dist_km': 18., 'taper_offset_km': 6., 'along_dip_slip_vals': []}
taper_downdip_18c_dict = {'name': 'downdip_18c', 'taper_dist_km': 18., 'taper_offset_km': 12., 'along_dip_slip_vals': []}
taper_downdip_18d_dict = {'name': 'downdip_18d', 'taper_dist_km': 18., 'taper_offset_km': 18., 'along_dip_slip_vals': []}
taper_downdip_21a_dict = {'name': 'downdip_21a', 'taper_dist_km': 21., 'taper_offset_km': 0., 'along_dip_slip_vals': []}
taper_downdip_21b_dict = {'name': 'downdip_21b', 'taper_dist_km': 21., 'taper_offset_km': 6., 'along_dip_slip_vals': []}
taper_downdip_21c_dict = {'name': 'downdip_21c', 'taper_dist_km': 21., 'taper_offset_km': 12., 'along_dip_slip_vals': []}
taper_downdip_21d_dict = {'name': 'downdip_21d', 'taper_dist_km': 21., 'taper_offset_km': 18., 'along_dip_slip_vals': []}

taper_downdip_dict_list = [taper_downdip_9a_dict, taper_downdip_9b_dict, taper_downdip_9c_dict, taper_downdip_9d_dict,
                            taper_downdip_12a_dict, taper_downdip_12b_dict, taper_downdip_12c_dict, taper_downdip_12d_dict,
                            taper_downdip_15a_dict, taper_downdip_15b_dict, taper_downdip_15c_dict, taper_downdip_15d_dict,
                            taper_downdip_18a_dict, taper_downdip_18b_dict, taper_downdip_18c_dict, taper_downdip_18d_dict,
                            taper_downdip_21a_dict, taper_downdip_21b_dict, taper_downdip_21c_dict, taper_downdip_21d_dict]

# These dictionaries taper along dip and along strike by the same amount.
taper_dip_strike_12j_dict = {'name': '12j', 'taper_dist_km': 12., 'taper_offset_km': 12.,
                  'tiled_fault_slip_vals': []}
taper_dip_strike_15j_dict = {'name': '15j', 'taper_dist_km': 15., 'taper_offset_km': 12.,
                  'tiled_fault_slip_vals': []}

taper_dip_strike_dict_list = [taper_dip_strike_12j_dict, taper_dip_strike_15j_dict]

# keeps bottom taper the same, moves upper extent of slip
taper_updip_12a_dict = {'name': 'updip_12a', 'taper_dist_km': 12., 'updip_offset_km': 3., 'along_dip_slip_vals': []}
taper_updip_12b_dict = {'name': 'updip_12b', 'taper_dist_km': 12., 'updip_offset_km': 6., 'along_dip_slip_vals': []}
taper_updip_12c_dict = {'name': 'updip_12c', 'taper_dist_km': 12., 'updip_offset_km': 9., 'along_dip_slip_vals': []}
taper_updip_12d_dict = {'name': 'updip_12d', 'taper_dist_km': 12., 'updip_offset_km': 12., 'along_dip_slip_vals': []}

taper_updip_dict_list = [taper_updip_12a_dict, taper_updip_12b_dict, taper_updip_12c_dict, taper_updip_12d_dict]

# Run the script
#################################
centre_fault = ListricFault.from_depth_profile_square_tile(
    tile_size=fault_patch_size, patch_heights=patch_heights, patch_dips=patch_dips,
    top_centre_x=cen_x, top_centre_y=cen_y, top_centre_z=cen_z, strike=strike)

# Replicate fault patches along strike (e.g., 5 new patches in each direction)
added_patches = round(fault_length/(2 * fault_patch_size) - 1)
tiled_fault = centre_fault.tile_along_strike(added_patches)
# Create laplacian smoothing matrix for fault model
tiled_fault.compute_laplacian(double=False)

# make a grid for calculating displacements
disp_grid = DisplacementGrid.from_bounds(tiled_fault.patches, x1, x2, disp_grid_size, y1, y2)

### Calculate slip values
# Find cross section dist/depths for patch centroids. Same for all fault segments.
# Make a shapely line that includes the centroids as a vertex.
# line coords: starts at 0,0, line vertex distance and depth. verts: centroid distance and depth.
centroid_cross_sec_verts, centroid_cross_sec_line_coords = centre_fault.calc_centroid_cross_section()

along_dip_line = LineString(centroid_cross_sec_line_coords)

center_fault_xy_coords = [patch.top_centre[:2] for patch in centre_fault.patches]
center_fault_xy_coords.append(centre_fault.patches[-1].bottom_centre[:2])
center_fault_xy_line = LineString(Point(vertex) for vertex in center_fault_xy_coords)

# make a shapely line of the top row of patches (for a strike line)
# goes through top centers, not edge to edge
upper_edge_centers = []
for patch in tiled_fault.patches:
    if patch.top_z ==0.:
        upper_edge_centers.append(patch.top_centre)

upper_edge_points = []
for center in upper_edge_centers:
    center_point = Point(center)
    upper_edge_points.append(center_point)

upper_edge_line = LineString(upper_edge_points)


if move_updip_extent is False:
    if taper_along_strike is False and taper_along_dip is True:
        model_dict_list = taper_downdip_dict_list
        for taper_name in model_dict_list:
            along_dip_slip_vals = []
            taper_dist_km = taper_name['taper_dist_km']
            for centroid in centroid_cross_sec_verts:
                taper_power = 2.
                # find distance along fault to centroid (along the dip)
                dist_from_top_edge = along_dip_line.project(Point(centroid))
                dist_to_bottom_edge = along_dip_line.length - dist_from_top_edge - taper_name['taper_offset_km'] * 1000.

                # find distance along dip to set slip value
                if dist_from_top_edge <= 3000.:      # set top edge slip to zero (blind rupture)
                    patch_slip = 0.
                elif dist_to_bottom_edge <= 0.:    # sets offset for bottom edge of slip
                    patch_slip = 0.
                elif 0 < dist_to_bottom_edge < (taper_dist_km * 1000.):   # applies taper to patches within taper distance.
                    # linear taper.
                    patch_slip = slip * (dist_to_bottom_edge / (taper_dist_km * 1000.))
                else:
                    patch_slip = slip
                along_dip_slip_vals.append(patch_slip)
            taper_name['along_dip_slip_vals'] = along_dip_slip_vals

    elif taper_along_strike is True and taper_along_dip is True:
        model_dict_list = taper_dip_strike_dict_list
        for model_name in model_dict_list:
            tiled_fault_slip_vals = []
            taper_dist_km = model_name['taper_dist_km']
            for patch in tiled_fault.patches:

                centroid_xy = patch.centroid[:2]

                #find cross sec distance along xy cross sec line
                xy_cross_sec_dist = center_fault_xy_line.project(Point(centroid_xy))
                point_dist_z = Point(xy_cross_sec_dist, patch.centroid[2])

                # find distance along fault dip to centroid (curved distance if listric)
                dist_from_top_edge = along_dip_line.project(point_dist_z)
                dist_to_bottom_edge = along_dip_line.length - dist_from_top_edge - model_name['taper_offset_km'] * 1000.

                # find shortest distance along strike to the lateral edges to set strike taper
                strike_dist1 = upper_edge_line.project(Point(centroid_xy)) # distance along strike from one edge
                strike_dist2 = upper_edge_line.length - strike_dist1             # distance along strike from the
                dist_to_lateral_edge = min(strike_dist1, strike_dist2)  # distance along strike over which to taper slip

                if dist_to_bottom_edge <= dist_to_lateral_edge:
                    # set slip value
                    if dist_from_top_edge <= 3000.:  # set top edge slip to zero (blind rupture)
                        patch_slip = 0.
                    elif dist_to_bottom_edge <= 0.:  # sets offset for bottom edge of slip
                        patch_slip = 0.
                    elif dist_from_top_edge > 3000. and 0 < dist_to_bottom_edge < (taper_dist_km * 1000.):  # applies
                        # taper to patches within taper distance.
                        # linear taper.
                        patch_slip = slip * (dist_to_bottom_edge / (taper_dist_km * 1000.))
                    else:
                        patch_slip = slip
                    tiled_fault_slip_vals.append(patch_slip)

                elif dist_to_lateral_edge < dist_to_bottom_edge:
                    if dist_from_top_edge <= 3000.:  # set top edge slip to zero (blind rupture)
                        patch_slip = 0.
                    elif dist_from_top_edge > 3000. and 0 <= dist_to_lateral_edge <= (taper_dist_km * 1000.):  # applies
                        # taper to patches within taper distance.
                        # linear taper.
                        patch_slip = slip * (dist_to_lateral_edge / (taper_dist_km * 1000.))
                    else:
                        patch_slip = slip
                    tiled_fault_slip_vals.append(patch_slip)

            model_name['tiled_fault_slip_vals'] = tiled_fault_slip_vals
    # uniform slip
    elif taper_along_strike is False and taper_along_dip is False:
        model_dict_list = uniform_dict_list
        for name in model_dict_list:
            along_dip_slip_vals = []
            for centroid in centroid_cross_sec_verts:
                patch_slip = slip
                along_dip_slip_vals.append(patch_slip)
            name['along_dip_slip_vals'] = along_dip_slip_vals

if move_updip_extent is True:
    model_dict_list = taper_updip_dict_list
    for updip_name in model_dict_list:
        along_dip_slip_vals = []
        for centroid in centroid_cross_sec_verts:
            taper_dist_km = updip_name['taper_dist_km']
            #taper_power = 2.
            # find distance along fault to centroid (along the dip)
            dist_from_top_edge = along_dip_line.project(Point(centroid))
            dist_to_bottom_edge = along_dip_line.length - dist_from_top_edge - 12. * 1000.

            # find distance along dip to set slip value
            # set top edge slip to zero based on upper constraint (blind rupture)
            if dist_from_top_edge < updip_name['updip_offset_km'] * 1000.:
                patch_slip = 0.
            elif dist_to_bottom_edge <= 0.:  # sets offset for bottom edge of slip
                patch_slip = 0.
            elif 0 < dist_to_bottom_edge < (taper_dist_km * 1000.):  # applies taper to patches within taper distance.
                # linear taper.
                patch_slip = slip * (dist_to_bottom_edge / (taper_dist_km * 1000.))
            else:
                patch_slip = slip
            along_dip_slip_vals.append(patch_slip)
        updip_name['along_dip_slip_vals'] = along_dip_slip_vals


# loop through each scenario to calculate displacements and generate a figure
for model_name in model_dict_list:
    extension2 = model_name['name']
    ### Set slip on patches. assigns for each listric fault segment, from top to bottom.
    if taper_along_strike is False:
        along_dip_slip_vals = model_name['along_dip_slip_vals']
        for j, fault in enumerate(tiled_fault.faults):
            # sets zero slip on end listric fault objects
            if j == 0 or j == len(tiled_fault.faults)-1:
                for i, patch in enumerate(fault.patches):
                    patch.set_slip_rake(0, rake)
            # sets tapered slip values to rest of patches
            else:
                for i, patch in enumerate(fault.patches):
                    patch_slip = along_dip_slip_vals[i]
                    patch.set_slip_rake(patch_slip, rake)

    elif taper_along_strike is True:
        for k, patch in enumerate(tiled_fault.patches):
            slip = model_name['tiled_fault_slip_vals'][k]
            patch.set_slip_rake(slip, rake)

    # Array of ones and zeros to identify edge patches
    edge_patches = tiled_fault.edge_patch_bool(top_edge=True)

    # Calculate GREENS FUNCTIONS
    # this method applies the slip to the patches, then calculates the displacements.
    # AL = ahuriri lagoon, CK = Cape Kidnappers
    disp_table_AL = DisplacementTable(tiled_fault.patches, np.array([1932228.]), np.array([5628830.]))
    disp_table_CK = DisplacementTable(tiled_fault.patches, np.array([1945817.]), np.array([5601485.]))
    disp_table_AL.collect_greens_functions()
    disp_table_CK.collect_greens_functions()
    print(f"collecting greens functions for {extension2}")
    disp_grid.collect_greens_functions()
    # Extract vertical disps to a 2D array
    vertical_disp_grid = disp_grid.total_displacements[2]

    # make dictionary of variables for figure making and geojson export
    tiled_fault_dict = {'extension1': extension1, 'extension2': extension2, 'tiled_fault': tiled_fault,
                        'disp_table_AL': disp_table_AL, 'disp_table_CK': disp_table_CK, 'disp_grid': disp_grid,
                        'vertical_disp_grid': vertical_disp_grid, 'laplacian': tiled_fault.laplacian,
                        "total_slip_values": [patch.total_slip for patch in tiled_fault.patches],
                        'edge_patches': edge_patches, 'rake': rake, 'fault_patch_size': fault_patch_size,
                        'along_dip_line': along_dip_line}

    patch_variable_dict = {"dip": [np.round(patch.dip, 2) for patch in tiled_fault.patches],
                              "tot_slp": [np.round(patch.total_slip, 2) for patch in tiled_fault.patches]
                              #"ds": [f"{patch.dip_slip:.3f}" for patch in tiled_fault.patches],
                              #"ss": [f"{patch.strike_slip:.3f}" for patch in tiled_fault.patches]
                                   }

    #make output directory for tiled fault dictionaries
    if not os.path.exists(f"tiled_fault_dictionaries"):
        os.mkdir(f"tiled_fault_dictionaries")

    with open(f"tiled_fault_dictionaries/tiled_fault_dict_{extension1}_{extension2}_r{int(rake)}.pkl", "wb") as fid:
        pkl.dump(tiled_fault_dict, fid)

    # Write out corners of patches for QGIS
    # outlines = gpd.GeoDataFrame(patch_variable_dict, geometry=[patch.corner_polygon for patch in tiled_fault.patches],
    #                                    crs=2193)
    # outlines.to_file(f"{extension1}_figures/patch_outlines_{extension1}_{extension2}.geojson", index="FID",
    #                         driver="GeoJSON")

    # make figure
    print(f"making figure for {extension2}")
    make_figure(extension1=extension1, extension2=extension2, tiled_fault_dict=tiled_fault_dict)

    #disp_grid.write_displacement_tiff(f"forward_figures/{extension}_vert.tif")


