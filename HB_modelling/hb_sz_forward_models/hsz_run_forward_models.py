import numpy as np
import geopandas as gpd
from shapely.geometry import  box
import meshio
from hsz_forward_tools import map_slip_dist
import cutde.halfspace as HS
import math
import pickle as pkl
import os
from hsz_forward_tools import get_Mw
from hsz_make_figures import make_figure

def get_slip_dist_dict(extension, taper_dist_km):
    """Get slip distribution dictionary for Hikurangi subduction zone.
    inputs:
        extension = the extension of the rupture patch group you want to model (e.g., "sml"). The extension
    points to a geojson file that contains multiple polygons. Each polygon defines a rupture patch.

        taper_dist_km = distance (in km) from the edge of the patch that slip tapers from peak to zero.

    outputs:
        slip_dist_dict= dictionary of geodataframes for each rupture scenario. Has slip distribution, mesh,
                max slip, taper distance.
        hsz_triangles_{extension}_p{number}.geojson = traingles with slip distribution for each patch"""

    # Read in subduction mesh
    mesh = meshio.read("../input_data/hik_kerm3k_rake_slip.vtk")

    # Read in polygons
    slip_dist_dict = {}
    polygons = gpd.read_file(f"../input_data/hsz_patches_{extension}.geojson")

    for i, row in polygons.iterrows():
        print(f"getting slip distribution for: {extension} polygon {i+1}")
        # Get slip distribution polygon
        slip_dist_poly = row.geometry.geoms[0]
        slip_poly_area = slip_dist_poly.area

        # buffer on polygon to make up for added buffer on slip distribution later.
        # negative = internal buffer. Convert from Km to meters
        slip_dist_poly_buff = slip_dist_poly.buffer(taper_dist_km * (-1000))
        slip_dist_poly_buff_area = slip_dist_poly_buff.area

        ### calculate average slip using scaling relationships from Stirling
        # Stirling uses, km^2, convert to m^2.
        scaling_rel_Mw = math.log10(slip_poly_area / 1.e6) + 4.0
        #print("scaling relation (Mw) = " + "{:.2f}".format(scaling_rel_Mw))

        scaled_avg_slip = (10 ** (1.5 * scaling_rel_Mw + 9.05)) / (3.e10 * slip_poly_area)
        #print("calculated average slip = " + "{:.2f}".format(scaled_avg_slip))

        # calculate ratio of total area to inside of taper for altering max slip
        F = 0.64 # average slip in tapered area compared to peak slip
        multiplier =  slip_poly_area / (F * (slip_poly_area - slip_dist_poly_buff_area) + slip_dist_poly_buff_area)
        peak_slip = scaled_avg_slip * multiplier

        # Map slip distribution to mesh
        slip_dist_gdf = map_slip_dist(slip_dist_poly=slip_dist_poly_buff, fault_mesh=mesh, max_slip=peak_slip,
                           taper_dist_km=taper_dist_km, buffer_dist_km=150., taper_power=1.)

        # make output directory if it doesn't exist
        if not os.path.exists(f"hsz_result_data"):
            os.mkdir(f"hsz_result_data")
        # save geojson file with slip dist
        slip_dist_gdf.to_file(f"hsz_result_data/hsz_triangles_{extension}_P{i+1}.geojson", driver="GeoJSON")
        slip_dist_dict[i] = slip_dist_gdf

    return slip_dist_dict

def get_displacements(slip_dist_dict, extension1, grid_res=2000.):
    # Area of interest defined in lonlat
    latlong_xmin, latlong_ymin, latlong_xmax, latlong_ymax = 176.0, -40.5, 179.0, -38.6
    latlong_box = box(latlong_xmin, latlong_ymin, latlong_xmax, latlong_ymax)
    latlong_gs = gpd.GeoSeries(latlong_box, crs=4326)
    # Convert to NZTM
    nztm_gs = latlong_gs.to_crs(epsg=2193)

    # Define displacement grid in NZTM
    grid_res = grid_res   # in meters
    xmin, ymin, xmax, ymax = grid_res * np.around(nztm_gs.bounds.values[0] / grid_res, 0)
    # buffer_n is scalar; multiplied by grid cell size to add buffer around area of interest
    buffer_number = 5

    # X and Y coordinates of grid, with buffer added
    x_data = np.arange(xmin - buffer_number * grid_res, xmax + (buffer_number + 1) * grid_res, grid_res)
    y_data = np.arange(ymin - (buffer_number + 1) * grid_res, ymax + (buffer_number + 1) * grid_res, grid_res)

    # Mesh grid and create (N,3) array of points to calculate displacements
    xmesh, ymesh = np.meshgrid(x_data, y_data)
    xpoints = xmesh.flatten()
    ypoints = ymesh.flatten()
    pts = np.column_stack((xpoints, ypoints, xpoints * 0.))


    AL_x, AL_y, AL_z = np.array([1932228.]), np.array([5628830.]), np.array([0.])
    AL_pts = np.column_stack((AL_x, AL_y, AL_z))
    ### Loop through slip distributions in patch group and calculate displacements

    # list of slip dictionaries for each patch in the group
    hsz_def_dict_all = []
    for i in slip_dist_dict.keys():
        extension2 = f"P{i + 1}"
        print(f"getting Green's function and deformation for: {extension1} {extension2}")
        # Get slip distribution
        slip_dist = slip_dist_dict[i]
        slip_vals = slip_dist["slip"].values
        rake_vals = slip_dist["rake"].values
        # Turn rake into strike-slip, dip-slip
        sd_dip_slip = slip_vals * np.sin(np.radians(rake_vals))
        sd_strike_slip = slip_vals * np.cos(np.radians(rake_vals))
        # Slip array for cutde
        sd_slip_array = np.vstack([sd_strike_slip, sd_dip_slip, 0. * sd_dip_slip]).T

        # Triangles from geodataframe (cutde needs triangles in (N,3,3) format)
        sd_triangles = np.array([np.array(triangle.exterior.coords)[:-1] for triangle in slip_dist["geometry"].values])


        ############# is this the greens function step?  or is there no greens
        # Calculate displacements
        disps = HS.disp_free(obs_pts=pts, tris=sd_triangles, slips=sd_slip_array, nu=0.25)
        disps_AL = HS.disp_free(obs_pts=AL_pts, tris=sd_triangles, slips=sd_slip_array, nu=0.25)
        vert_disp_AL = disps_AL[:, -1]

        # Reshape to grid
        out_data_vert_grid = disps[:, -1].reshape(xmesh.shape)

        # Write to tiff file
        #out_name = f"_hsz_patches_vertical_disps_{extension1}_{extension2}.tif"
        #write_tiff(out_name, x_data, y_data, out_data)


        # get rupture magnitude
        Mw, avg_slip = get_Mw(sd_triangles, slip_vals)

        hsz_def_dict_i = {"disps": disps, "x_data": x_data, "y_data": y_data, "grid_res": grid_res,
                        "vert_disp_grid": out_data_vert_grid, "vert_disp_AL": vert_disp_AL,
                        "sd_triangles": sd_triangles, "triangle_slip_vals": slip_vals,
                        "triangle_rake_vals": rake_vals, "Mw": Mw, "avg_slip": avg_slip,
                        "extension1": extension1, "extension2": extension2}

        # make output directory for scenario dictionaries
        #pkl.dump(hsz_def_dict_i, open(f"hsz_result_data/hsz_def_dict_{extension1}_{extension2}.pkl", "wb"))

        # make master list for plotting later
        hsz_def_dict_all.append(hsz_def_dict_i)
        pkl.dump(hsz_def_dict_all, open(f"hsz_result_data/hsz_def_dict_p_all_{extension1}.pkl", "wb"))
    return hsz_def_dict_all

def write_results(hsz_def_dict_all):
    """Write results to file for each patch group"""

    extension1 = hsz_def_dict_all[0]["extension1"]

    with open(f'hsz_result_data/hsz_results_{extension1}.txt', 'w') as f:
        for scenario in hsz_def_dict_all:
            extension2 = scenario["extension2"]
            Mw, avg_slip = scenario["Mw"], scenario["avg_slip"]
            min_vert = np.min(scenario["vert_disp_grid"])
            max_vert = np.max(scenario["vert_disp_grid"])
            vert_disp_AL = scenario["vert_disp_AL"]

            f.write(f"Scenario name: {extension1} {extension2} \n")
            f.write(f"Mw = {Mw:.2f} \n")
            f.write(f"Vertical deformation at Ahuriri Lagoon = {vert_disp_AL[0]:.3f} m\n")
            f.write(f"Minimum vertical def = {min_vert:.3f} m\n")
            f.write(f"Maximum vertical def = {max_vert:.3f} m\n")
            f.write(f"Average slip (on patches that slipped): {avg_slip:.3f} m\n\n")


def run_forward_model(extension1_list, grid_res, slip_taper_dist_km):
    """Run forward model. List is a group of slip pathches"""

    for extension1 in extension1_list:
        # get slip distribution dictionary fand greens functions for each patch in group
        slip_dict_group = get_slip_dist_dict(extension1, taper_dist_km=slip_taper_dist_km)

        # calculate displacements on each patch in group. Make pickle of disps and triangles for the whole group.
        hsz_def_dict_all = get_displacements(slip_dict_group, extension1, grid_res=grid_res)

        # write scenario results to a text file
        write_results(hsz_def_dict_all)


### INPUTS
# extension for geojson file that contains multiple polygons.
# requires there to be a geojson file in the "input data" folder with "hsz_patches_{extension1}.geojson" name
extension1_list = ["sml", "coupling"]

# slip taper distance
slip_taper_dist_km = 12.

# grid resolution for surface displacements/green's functions (in meters
grid_res = 2000.

####

run_forward_model(extension1_list, grid_res=grid_res)
make_figure(extension1_list)

