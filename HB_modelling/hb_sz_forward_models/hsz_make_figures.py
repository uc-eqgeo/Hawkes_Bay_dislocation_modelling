import numpy as np
import matplotlib
from matplotlib import pyplot as plt, cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle as pkl
import geopandas as gpd
import pandas as pd
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from scipy.interpolate import griddata
import os
import rasterio

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def make_slip_colormap():
    """makes magma-ish colormap for plotting slip values
    modified to remove black max value end color and replace 0 end color with grey"""

    # use existing color ramp as a template. 256 is the number of colors sampled from the ramp
    starter_colormap = cm.get_cmap('magma_r', 256)  # makes a copy of the color map

    # makes an array of values of color map; 0-0.85 chops off black color (max value for middle is 1)
    newcolors = starter_colormap(np.linspace(0, 0.87, 256))

    # set replacement color, set how many entries you want new color to replace
    # note that this chops off the first color in the original array a little bit
    # replacement color here is grey
    ### FOR SZ MODELS: set zero slip to transparent (alpha = 0
    zero_color = np.array([224 / 256, 224 / 256, 224 / 256, 1])
    newcolors[:1, :] = zero_color
    # turn array into colormap
    custom_colormap = ListedColormap(newcolors)
    return custom_colormap

def make_vertdef_colormap():
    """makes blue white red colormap for plotting"""

    # choose color RGBs. If you want white at zero it has to be in the middle
    color0 = np.array([0 / 255, 46 / 255, 92 / 255, 1])     # blue
    color1 = np.array([4 / 255, 79 / 255, 140 / 255, 1])
    color2 = np.array([5 / 255, 113 / 255, 176 / 255, 1])
    color3 = np.array([146 / 255, 197 / 255, 222 / 255, 1])
    color4 = np.array([1, 1, 1, 1])                         # white
    color5 = np.array([227 / 255, 133 / 255, 133 / 255, 1])
    color6 = np.array([202 / 255, 0, 0, 1])
    color7 = np.array([179 / 255, 0, 0, 1])
    color8 = np.array([150 / 255, 0, 0, 1])                 # red
    newcolors = [color0, color1, color2, color3, color4, color5, color6, color7, color8]

    colors = []
    for color in newcolors:
        colors.append(tuple(color))
    custom_colormap = LinearSegmentedColormap.from_list('vertdef_colors', colors, N=200)

    return custom_colormap


def make_vert_def_cross_section(rupture_dict, ax):
    """ makes a cross-section of deformation through the gridded vertical def data"""

    #load data
    AL_disp = round(rupture_dict['vert_disp_AL'][0], 1)
    disp_grid = rupture_dict['vert_disp_grid']
    x_array = rupture_dict['x_data']
    y_array = rupture_dict['y_data']

    # make grid out of x and y coordinates
    x_grid, y_grid = np.meshgrid(x_array, y_array)
    # flatten into 1D arrays
    x_coords_flat, y_coords_flat = x_grid.flatten(), y_grid.flatten()
    z_vals_flat = disp_grid.flatten()
    # stack x and y into one column of xy pairs
    xy_coords = np.column_stack((x_coords_flat, y_coords_flat))

    ## get grid values along cross-section line
    line = gpd.read_file("../input_data/hsz_cross_section_line.geojson")
    # returns bounds of line as [minx, miny, maxx, maxy]
    line_bounds = line.geometry[0].bounds
    # makes 100 points along the line between endpoints. Start from max y for Hawke's Bay!
    line_x = np.linspace(line_bounds[0], line_bounds[2], 100)
    line_y = np.linspace(line_bounds[3], line_bounds[1], 100)
    # stack into one column of xy pairs
    line_xy = np.column_stack((line_x, line_y))

    # extract grid value along cross-section line using "linear" interpolation
    cross_sect_values = griddata(xy_coords, z_vals_flat, line_xy, method='linear')

    # calculate distances along cross-section line, starting from upper left corner
    # make shapely point object from upper left corner
    start_point = Point([line_bounds[0], line_bounds[3]])
    distances_km = []
    # make shapely point for each point along the cross-section line and calculate distance to start point
    for coord in line_xy:
        next_point = Point(coord)
        distance_km = start_point.distance(next_point) / 1000
        distances_km.append(distance_km)
    # make distances into np array
    distances_km = np.array(distances_km)

    # make boolean array to split cross-section into positive and negative values
    below_zero = cross_sect_values <= 0

    #subset cross section x and y by + and - y values
    x_below_zero, y_below_zero = distances_km[below_zero], cross_sect_values[below_zero]

    # plot cross-section and make it pretty
    plt.close("all")
    ax.plot(distances_km, np.zeros(len(distances_km)), 'k', linewidth=0.5)
    ax.plot(distances_km, cross_sect_values, color=(179 / 255, 0, 0, 1), linewidth=2)
    ax.plot(x_below_zero, y_below_zero, color=(5 / 255, 113 / 255, 176 / 255, 1), linewidth=2)

    # make it pretty
    ax.set(xlim=(0., distances_km.max()), ylim=(-3, 3.5))
    ax.tick_params(color='black', axis="both", which='major', direction='in', labelsize=6)
    ax.xaxis.set_ticklabels([])
    ax.set_yticks(np.arange(-3, 4, 1))
    ax.set_ylabel("Vertical disp. (m)", fontsize=6)
    ax.set_aspect(15)
    ax.set_anchor('S')
    # set ahuriri disp label. text location (first 2 arguments) and x and y in plot coords
    ax.text(5, 0.1, f'A.L. disp.= {AL_disp:.2f} m', fontsize=6,
            color=(5 / 255, 113 / 255, 176 / 255, 1))

    #plt.savefig(f"figures/{extension_name}/vert_def_cross_section_{extension_name}.png")

def make_slip_cross_section(rupture_dict, ax):
    #for extension_name in extension_list:
    #load slip data
    sd_triangles = rupture_dict['sd_triangles']
    triangle_slip = rupture_dict['triangle_slip_vals']
    upf_plotting_data = pkl.load(open("../input_data/upf_plotting_data.pkl", "rb"))

    # load elevation profile csv file
    elev_profile = pd.read_csv("../input_data/hsz_cross_section_topo_profile.csv")

    # extract mesh vertices
    triangle_verts_xy_coords = np.concatenate([triangle[:, 0:2] for triangle in sd_triangles])
    triangle_coords_z = np.concatenate([triangle[:, 2] for triangle in sd_triangles])

    # convert to km
    triangle_coords_z_km = triangle_coords_z / 1000

    # make slip value array that is the same length as the triangle vertices array
    triangle_coords_slip = np.repeat(triangle_slip, 3)

    ## get xy points along cross-section line
    line1 = gpd.read_file("../input_data/hsz_cross_section_line.geojson")
    line2 = gpd.read_file("../input_data/hsz_mesh_cross_section_line.geojson")
    line1_bounds = line1.geometry[0].bounds
    line2_bounds = line2.geometry[0].bounds
    # makes 1000 points along the line between endpoints. Start from max y for Hawke's Bay!
    # line2 used for fault geometry and slip, shorter than line 1 used for displacement and plot width.
    line2_x_dense = np.linspace(line2_bounds[0], line2_bounds[2], 1000)
    line2_y_dense = np.linspace(line2_bounds[3], line2_bounds[1], 1000)
    line2_xy_dense = np.column_stack((line2_x_dense, line2_y_dense))

    #### extract depths and slip value along cross-section points
    # set points outside the cross section value to nan
    cross_sect_z = griddata(triangle_verts_xy_coords, triangle_coords_z_km, line2_xy_dense, method='linear',
                                      fill_value=np.nan)
    # extract slip values using nearest neighbor interpolation (otherwise it's a smooth slip dist)
    cross_sect_slip_values = griddata(triangle_verts_xy_coords, triangle_coords_slip, line2_xy_dense,
                                      method='nearest', fill_value=np.nan)

    # figure out line distances along cross-section line
    # line used for figure bounds to match displacement plot. Just calculates total line distance.
    start_point1 = Point([line1_bounds[0], line1_bounds[3]])
    end_point1 = Point([line1_bounds[2], line1_bounds[1]])
    xs1_distances_km = start_point1.distance(end_point1) / 1000
    # line and distances used for fault geometry (grey line) and slip (colored dots)
    start_point2 = Point([line2_bounds[0], line2_bounds[3]])
    xs2_distances_km = []
    for coord in line2_xy_dense:
        next_point = Point(coord)
        distance_km = start_point2.distance(next_point) / 1000
        xs2_distances_km.append(distance_km)

    # convert to np array
    xs1_distances_km = np.array(xs1_distances_km)
    xs2_distances_km = np.array(xs2_distances_km)

    # get custom slip colormap
    custom_slip_colormap = make_slip_colormap()

    # subset slip cross-section x and y by values of ~0 and >0
    above_break = cross_sect_slip_values >= 0.01  # takes the opposite of the boolean array
    x_above_break, y_above_break = xs2_distances_km[above_break], cross_sect_z[above_break]
    cross_sect_slip_values_above_break = cross_sect_slip_values[above_break]

    #### make cross section
    # plot fault geometry
    ax.plot(xs2_distances_km, cross_sect_z, color="0.7", linewidth=5, zorder=1)
    hsz_cross_section_pts = [xs2_distances_km, cross_sect_z]
    #plot UPF geometry
    ax.plot(upf_plotting_data['xs_distances_km'], upf_plotting_data['cross_sect_z'], color="0.7",
                                                                         linewidth=2, linestyle='dashed', zorder=2)
    #save pickle file of cross section points for UPF fault
    if rupture_dict['extension1'] == "coupling" and rupture_dict['extension2'] == "P1":
        pkl.dump(hsz_cross_section_pts, open(f"hsz_figures/hsz_geom_cross_section_pts.pkl", "wb"))
    # plot slip values as colored points
    vmin, vmax = 0, 9
    ax.scatter(x_above_break, y_above_break, c=cross_sect_slip_values_above_break, cmap=custom_slip_colormap,
                s=8, vmin=vmin, vmax=vmax, zorder=3)
    # plot elevation profile
    ax.plot(elev_profile['dist_m']/1000, elev_profile['el_m']/1000, color="k", linewidth=1, zorder=4)

    # make it pretty
    ax.set(xlim=(0., xs1_distances_km.max()), ylim=(-30, 2))
    ax.set_ylabel("Depth (km)", fontsize=6)
    ax.set_xlabel("Distance (km)", fontsize=6, labelpad=4)
    ax.tick_params(color='black', axis="both", which='major', direction='in', labelsize=6)
    ax.set_yticks(np.arange(-30., 0.5, 10.))
    aspect = 2.     # vertical exaggeration
    ax.set_aspect(aspect)   # vertical exaggeration

    # cross section ends, ahuriri and cape kidnappers, V.E. ticks and labels
    ax.vlines(x=26.730, ymin=0., ymax=4, color='k', linewidth=1.0)
    ax.vlines(x=54.800, ymin=0., ymax=4, color='k', linewidth=1.0)
    ax.text(26.730, 4, f'A.L.', ha='center', fontsize=6, color='k')
    ax.text(54.800, 4, f'C.K.', ha='center', fontsize=6, color='k')
    ax.text(0, 4, f'Y', ha='center', fontsize=6, color='g')
    ax.text(xs1_distances_km.max(), 4, f"Y'", ha='center', fontsize=6, color='g')
    ax.text(205, -25, f'{int(aspect)}x V.E.', ha='right', fontsize=6, color='k')

    # set figure location within subplot and remove top border
    ax.set_anchor('N')
    ax.spines['top'].set_visible(False)

def write_tiff(filename: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, epsg: int = 2193, reverse_y: bool = False,
               compress_lzw: bool = True):
    """
    Write x, y, z into geotiff format.
    :param filename:
    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates: must have ny rows and nx columns
    :param epsg: Usually NZTM (2193)
    :param reverse_y: y starts at y_max and decreases
    :param compress_lzw: lzw compression
    :return:
    """
    # Check data have correct dimensions
    assert all([a.ndim == 1 for a in (x, y)])
    assert z.shape == (len(y), len(x))

    # Change into y-ascending format (reverse_y option changes it back later)
    if y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]

    # To allow writing in correct format
    z = np.array(z, dtype=np.float64)

    # Calculate x and y spacing
    x_spacing = (max(x) - min(x)) / (len(x) - 1)
    y_spacing = (max(y) - min(y)) / (len(y) - 1)

    # Set affine transform from x and y
    if reverse_y:
        # Sometimes GIS prefer y values to descend.
        transform = rasterio.transform.Affine(x_spacing, 0., min(x), 0., -1 * y_spacing, max(y))
    else:
        transform = rasterio.transform.Affine(x_spacing, 0., min(x), 0., y_spacing, min(y))

    # create tiff profile (no. bands, data type etc.)
    profile = rasterio.profiles.DefaultGTiffProfile(count=1, dtype=np.float64, transform=transform, width=len(x),
                                                    height=len(y))

    # Set coordinate system if specified
    if epsg is not None:
        if epsg not in [2193, 4326, 32759, 32760, 27200]:
            print("EPSG:{:d} Not a recognised NZ coordinate system...".format(epsg))
            print("Writing anyway...")
        crs = rasterio.crs.CRS.from_epsg(epsg)
        profile["crs"] = crs

    # Add compression if desired
    if compress_lzw:
        profile["compress"] = "lzw"

    # Open raster file for writing
    fid = rasterio.open(filename, "w", **profile)
    # Write z to band one (depending whether y ascending/descending required).
    if reverse_y:
        fid.write(z[-1::-1], 1)
    else:
        fid.write(z, 1)
    # Close file
    fid.close()

def make_figure(extension_list):
    """makes a 3 part plot with the fault patches and slip, vertical deformation, and cross-section

        must run the inversion first to get the slip and deformation dictionaries (get_slip_dist_dict and
        get_displacements; or run_forward_model)

        extension list: list of extensions that point to different geojson and deformation dictionary files,
        same as extensions used to calculate the slip and vertical deformation from the inversion.
            """

    # load mapping data
    coastline = gpd.read_file("../input_data/coastline.geojson")
    plate_boundary = gpd.read_file("../input_data/hsz_interface_model_trace.geojson")
    cross_section_line = gpd.read_file("../input_data/hsz_cross_section_line.geojson")
    interest_faults = gpd.read_file("../input_data/interest_faults.geojson")
    ahuriri_outline = gpd.read_file("../input_data/ahuriri_simple_outline.geojson")

    # get cross section line endpoints for plotting labels
    cross_section_line_bounds = cross_section_line.geometry[0].bounds
    xs_label1_x, xs_label2_x = cross_section_line_bounds[0], cross_section_line_bounds[2]
    xs_label1_y, xs_label2_y = cross_section_line_bounds[3], cross_section_line_bounds[1]

    for extension1_name in extension_list:
        # load deformation dictionary from the get_displacements script.
        rupture_group_dict = pkl.load(open(f"hsz_result_data/hsz_def_dict_p_all_{extension1_name}.pkl", 'rb'))
        for rupture_dict in rupture_group_dict:
            triangle_slip_vals = rupture_dict['triangle_slip_vals']
            Mw = rupture_dict['Mw']

            # set tiled fault variable and load deformation
            sd_triangles = rupture_dict['sd_triangles']
            extension2 = rupture_dict['extension2']
            disp_grid = rupture_dict['vert_disp_grid']
            x_data = rupture_dict['x_data']
            y_data = rupture_dict['y_data']

            rupture_triangles_gdf = gpd.read_file(f"hsz_result_data/hsz_triangles_{extension1_name}"
                                                  f"_{extension2}.geojson")

            # use custom colors
            custom_slip_colormap = make_slip_colormap()
            custom_vertdef_colormap = make_vertdef_colormap()

            # Set up plot area
            plt.close("all")
            fig = plt.figure()
            fig.set_figheight(3)
            fig.set_figwidth(7)
            ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=2)
            ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=1, rowspan=2)
            ax3 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
            ax4 = plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)
            axs = [ax1, ax2, ax3, ax4]

            plot_xmin, plot_ymin, plot_xmax, plot_ymax = min(x_data) - 20000, min(y_data), max(x_data), max(y_data)

            # Plot slip distribution on triangles. Needs to make Scalar Mappable object for the colorbar.
            vmin, vmax = 0, 9
            rupture_triangles_gdf.plot(ax=ax1, column="slip", cmap=custom_slip_colormap, vmin=vmin,
                                                        vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=custom_slip_colormap, norm = plt.Normalize(vmin=vmin, vmax=vmax))

            # plot vertical deformation
            disps = axs[1].imshow(disp_grid[-1::-1], extent=[plot_xmin, plot_xmax, plot_ymin, plot_ymax],
                                  cmap=custom_vertdef_colormap, vmin=-3, vmax=3)

            #make plot and axes pretty
            for ax in axs[0:2]:
                coastline.plot(ax=ax, color="k", linewidth=1.0)
                plate_boundary.plot(ax=ax, color="0.5", linewidth=1.0)
                cross_section_line.plot(ax=ax, color="g", linewidth=1.0)
                ax.text(xs_label1_x - 5000, xs_label1_y + 5000, "Y", ha='right', fontsize=6, color='g')
                ax.text(xs_label2_x + 5000, xs_label2_y - 3000, "Y'", ha='left', fontsize=6, color='g')
                ahuriri_outline.plot(ax=ax, color="k")
                ax.text(1925000., 5620000., f'A.L.', ha='right', fontsize=6, color='k')
                ax.text(1959000., 5605000., f'C.K.', ha='left', fontsize=6, color='k')

                ax.set_xticks(np.arange(1900000, plot_xmax, 100000.))
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mE'))
                ax.set_yticks(np.arange(5550000., plot_ymax, 100000.))
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mN'))
                plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
                ax.tick_params(color='black', axis="both", which='major', direction='in', labelsize=6)
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                ax.set_xlim(plot_xmin, plot_xmax)
                ax.set_ylim(plot_ymin, plot_ymax)
                ax.set_anchor('S')
                ax.set_aspect("equal")

            #plot fault mapping and on left plot
            interest_faults.plot(ax=axs[0], color="0.5", linewidth=0.5)
            ax.text(1830000., 5705000., f'Mw = {Mw:.1f}', ha='left', fontsize=6, color='k')

            #hide yaxis label on right plots
            axs[1].yaxis.set_ticklabels([])
            #add contour lines
            # axs[1].contour(x_data, y_data, disp_grid, levels=[-1, -0.5], colors="b",
            #             linewidths=0.5, linestyles="dashed")
            #axs[1].contour(x_data, y_data, disp_grid, levels=[-0, 1, 2, 3], colors="0.5",
            #               linewidths=0.5)

            # make the colorbars pretty
            divider = make_axes_locatable(axs[0])
            cax1 = divider.append_axes('top', size='6%', pad=0.05)
            cbar1 = fig.colorbar(sm, cax=cax1, ticks=[*range(vmin, vmax + 2, 2)],
                                 orientation='horizontal')
            cbar1.ax.set_aspect(0.5)
            cbar1.ax.set_title("total slip (m)", fontsize=6)
            cax1.xaxis.set_ticks_position("top")
            cbar1.ax.tick_params(labelsize=6, direction='in')

            divider2 = make_axes_locatable(axs[1])
            cax2 = divider2.append_axes('top', size='6%', pad=0.05)
            cbar2 = fig.colorbar(disps, cax=cax2, ticks=[-3, -2, -1, 0, 1, 2, 3], orientation='horizontal')
            cbar2.ax.set_aspect(0.4)
            cbar2.ax.set_title("Vertical deformation (m)", fontsize=6)
            cax2.xaxis.set_ticks_position("top")
            cbar2.ax.tick_params(labelsize=6, direction='in')

            # plot deformation cross section
            make_vert_def_cross_section(rupture_dict, ax=axs[2])
            make_slip_cross_section(rupture_dict, ax=axs[3])

            fig.suptitle(f"Model {extension1_name} {extension2}", fontsize=8)
            fig.tight_layout()
            #fig.subplots_adjust(right=0.95, left=0.05, top=0.95, bottom=0.05, wspace=0.2)

            # make output directory if it doesn't exist
            if not os.path.exists(f"hsz_figures"):
                os.mkdir(f"hsz_figures")
            if not os.path.exists(f"hsz_figures/{extension1_name}"):
                os.mkdir(f"hsz_figures/{extension1_name}")

            #save figure
            # fig.savefig(f"hsz_figures/{extension1_name}/fig_{extension1_name}_{extension2}.pdf", transparent=True,
            # dpi=300)
            fig.savefig(f"hsz_figures/{extension1_name}/fig_{extension1_name}_{extension2}.png", transparent=True,
                        dpi=300)



