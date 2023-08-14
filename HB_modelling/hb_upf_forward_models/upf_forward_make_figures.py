import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import pickle as pkl
import os
from dislocations.utilities import write_tiff
import geopandas as gpd
import pandas as pd
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import LineString, Point
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
from upf_forward_tools_jde import get_rect_Mw

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rcParams({'font.sans-serif':'Helvetica'})


def make_slip_colormap():
    """makes colormap for plotting"""

    # use existing color ramp as a template. 256 is the number of colors sampled from the ramp
    starter_colormap = cm.get_cmap('magma_r', 256)  # makes a copy of the color map

    # makes an array of values of color map; 0-0.85 chops off black color (max value for middle is 1)
    newcolors = starter_colormap(np.linspace(0, 0.87, 256))


    # set replacement color, set how many entries you want new color to replace
    # note that this chops off the first color in the original array a little bit
    # replacement color here is grey
    zero_color = np.array([224 / 256, 224 / 256, 224 / 256, 1])
    newcolors[:1, :] = zero_color
    # turn array into colormap
    custom_colormap = ListedColormap(newcolors)
    return custom_colormap

def make_vertdef_colormap():
    """makes blue white red colormap for plotting"""

    # choose color RGBs. if you want white at zero it has to be in the middle
    color0 = np.array([0 / 255, 46 / 255, 92 / 255, 1]) # blue
    color1 = np.array([4 / 255, 79 / 255, 140 / 255, 1])
    color2 = np.array([5 / 255, 113 / 255, 176 / 255, 1])
    color3 = np.array([146 / 255, 197 / 255, 222 / 255, 1])
    color4 = np.array([1, 1, 1, 1]) # white
    color5 = np.array([227 / 255, 133 / 255, 133 / 255, 1])
    color6 = np.array([202 / 255, 0, 0, 1])
    color7 = np.array([179 / 255, 0, 0, 1])
    color8 = np.array([150 / 255, 0, 0, 1]) # red
    newcolors = [color0, color1, color2, color3, color4, color5, color6, color7, color8]

    colors = []
    for color in newcolors:
        colors.append(tuple(color))
    custom_colormap = LinearSegmentedColormap.from_list('vertdef_colors', colors, N=200)

    return custom_colormap


def make_def_cross_section(tiled_fault_dict, ax):
    """ makes a cross section of deformation through the gridded vertical def data"""

    #for extension_name in extension_list:
    #load data
    disp_grid = tiled_fault_dict['disp_grid']
    vertical_disp_grid = tiled_fault_dict['vertical_disp_grid']
    disp_table_AL = tiled_fault_dict['disp_table_AL']
    disp_AL = float(disp_table_AL.total_displacements[2])

    # xy coords of grid in array form
    x_grid, y_grid = disp_grid.x, disp_grid.y

    # flatten into 1D arrays
    x_coords_flat, y_coords_flat = x_grid.flatten(), y_grid.flatten()
    z_vals_flat = vertical_disp_grid.flatten()
    # stack x and y into one column of xy pairs
    xy_coords = np.column_stack((x_coords_flat, y_coords_flat))

    ## get grid values along cross-section line
    line = gpd.read_file("../input_data/upf_cross_section_line.geojson")
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
    #subset cross section x and y by boolean array
    x_below_zero, y_below_zero = distances_km[below_zero], cross_sect_values[below_zero]

    first_half = x_below_zero < 60  # remove values at the end that might also be below zero
    x_plot, y_plot = x_below_zero[first_half], y_below_zero[first_half]


    # plot cross-section and make it pretty
    plt.close("all")
    ax.plot(distances_km, np.zeros(len(distances_km)), 'k', linewidth=0.5)
    ax.plot(distances_km, cross_sect_values, color=(179 / 255, 0, 0, 1), linewidth=2)
    ax.plot(x_plot, y_plot, color=(5 / 255, 113 / 255, 176 / 255, 1), linewidth=2)

    # make it pretty
    ax.set(xlim=(0., distances_km.max()), ylim=(-1.5, 5))
    ax.tick_params(color='black', axis="both", which='major', direction='in', labelsize=6)
    ax.xaxis.set_ticklabels([])
    ax.set_yticks(np.arange(-1, 6, 1))
    ax.set_ylabel("Vertical disp. (m)", fontsize=6)
    ax.set_aspect(5)
    ax.set_anchor('S')
    # set ahuriri disp label. text location (first 2 arguments) and x and y in plot coords
    ax.text(1, 0.2, f'A.L. disp. = {disp_AL:.2f} m', fontsize=6,
            color=(5 / 255, 113 / 255, 176 / 255, 1))

    #plt.savefig(f"figures/{extension_name}/vert_def_cross_section_{extension_name}.png")

def make_slip_cross_section(tiled_fault_dict, ax):
    #for extension_name in extension_list:
    #load data
    # set tiled fault variable and load deformation
    tiled_fault = tiled_fault_dict['tiled_fault']
    total_slip_values = tiled_fault_dict['total_slip_values']

    # load elevation profile csv file and inteface geometry cross section pts
    elev_profile = pd.read_csv("../input_data/hsz_cross_section_topo_profile.csv")
    hsz_xs_pts = pkl.load(open(f"../input_data/hsz_geom_cross_section_pts.pkl", 'rb'))

    # extract patch corners
    patch_corner_xy_coords = np.concatenate([array[:, 0:2] for array in tiled_fault.patch_corners])
    corner_coords_z = np.concatenate([array[:, 2] for array in tiled_fault.patch_corners])
    # convert to km
    corner_coords_z_km = corner_coords_z / 1000

    #extract centroids for slip values
    patch_centroids = []
    for patch in tiled_fault.patches:
        patch_centroids.append(patch.centroid)
    patch_centroids_x = np.concatenate([array[0:1] for array in patch_centroids])
    patch_centroids_y = np.concatenate([array[1:2] for array in patch_centroids])
    patch_centroids_xy = np.column_stack((patch_centroids_x, patch_centroids_y))


    ## get xy points along cross-section line
    line = gpd.read_file("../input_data/upf_cross_section_line.geojson")
    line_bounds = line.geometry[0].bounds
    # makes 100 points along the line between endpoints. Start from max y!
    line_x_dense = np.linspace(line_bounds[0], line_bounds[2], 1000)
    line_y_dense = np.linspace(line_bounds[3], line_bounds[1], 1000)
    line_xy_dense = np.column_stack((line_x_dense, line_y_dense))

    # extract depths and slip value along cross-section points using "linear" interpolation
    # set points outside the cross section value to nan
    cross_sect_z = griddata(patch_corner_xy_coords, corner_coords_z_km, line_xy_dense, method='linear',
                                      fill_value=np.nan)
    # extract slip values using nearest neighbor interpolation (otherwise it's a smooth slip dist)
    cross_sect_slip_values = griddata(patch_centroids_xy, total_slip_values, line_xy_dense,
                                      method='nearest', fill_value=np.nan)

    # figure out line distances along cross-section line
    start_point = Point([line_bounds[0], line_bounds[3]])
    xs_distances_km = []
    for coord in line_xy_dense:
        next_point = Point(coord)
        distance_km = start_point.distance(next_point) / 1000
        xs_distances_km.append(distance_km)


    # convert to np array
    xs_distances_km = np.array(xs_distances_km)

    # get custom slip colormap
    custom_slip_colormap = make_slip_colormap()

    below_break = cross_sect_slip_values <= 0.01
    above_break = ~below_break  # takes the opposite of the boolean array
#   # subset cross section x and y by + and - y value
    #x_below_berak, y_below_zero = xs_distances_km[below_break], cross_sect_slip_values[below_break]
    x_above_break, y_above_break = xs_distances_km[above_break], cross_sect_z[above_break]
    cross_sect_slip_values_above_break = cross_sect_slip_values[above_break]
    # # plot slip as a colored points

    #### plot coss section
    # plot SZ geometry
    ax.plot(hsz_xs_pts[0], hsz_xs_pts[1], color="0.7", linewidth=2, linestyle='dashed', zorder=1)

    # plot fault geometry
    ax.plot(xs_distances_km, cross_sect_z, color="0.7", linewidth=5, zorder=1)
    # plot slip values
    ax.scatter(x_above_break, y_above_break, c=cross_sect_slip_values_above_break, cmap=custom_slip_colormap,
                s=10, vmin=0, vmax=8, zorder=2)

    #### make it pretty
    ax.tick_params(color='black', axis="both", which='major', direction='in', labelsize=6)
    ax.set_yticks(np.arange(-30., 10., 10.))
    ax.set(xlim=(0., xs_distances_km.max()), ylim=(-27, 3))
    ax.set_ylabel("Depth (km)", fontsize=6)
    ax.set_xlabel("Distance (km)", fontsize=6, labelpad=4)
    ax.spines['top'].set_visible(False)
    ax.set_aspect('equal')

    #cover up fault above surface
    ax.fill_between(elev_profile['dist_m'] / 1000, ax.get_ylim()[1], (elev_profile['el_m'] / 1000).squeeze(),
                    color='w')
    # plot elevation profile
    ax.plot(elev_profile['dist_m'] / 1000, elev_profile['el_m'] / 1000, color="k", linewidth=1, zorder=3)

    # cross section ends, ahuriri and cape kidnappers, V.E. ticks and labels
    ax.vlines(x=26.730, ymin=0., ymax=2, color='k', linewidth=1.0)
    ax.vlines(x=54.800, ymin=0., ymax=2, color='k', linewidth=1.0)
    ax.text(26.730, 4, f'A.L.', ha='center', fontsize=6, color='k')
    ax.text(54.800, 4, f'C.K.', ha='center', fontsize=6, color='k')
    ax.text(0, 4, f'X', ha='center', fontsize=6, color='g')
    ax.text(xs_distances_km.max(), 4, f"X'", ha='center', fontsize=6, color='g')
    ax.text(80, -25, f'no V.E.', ha='right', fontsize=6, color='k')

    # set plot location within subplot box
    ax.set_anchor('N')

def make_figure(extension1, extension2, tiled_fault_dict):
    """makes a 4 part plot with the fault patches and slip and vertical deformation"""
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42

    coastline = gpd.read_file("../input_data/coastline.geojson")
    cross_section_line = gpd.read_file("../input_data/upf_cross_section_line.geojson")
    interest_faults = gpd.read_file("../input_data/interest_faults.geojson")
    ahuriri_outline = gpd.read_file("../input_data/ahuriri_simple_outline.geojson")

    # set tiled fault variable and load deformation
    tiled_fault = tiled_fault_dict['tiled_fault']
    #extension = tiled_fault_dict['model_extension']
    disp_grid = tiled_fault_dict['disp_grid']
    xgrid = disp_grid.x_values
    ygrid = disp_grid.y_values
    vertical_disp_grid = tiled_fault_dict['vertical_disp_grid']
    patch_size = tiled_fault_dict['fault_patch_size']
    rake = tiled_fault_dict['rake']

    total_slip_values = tiled_fault_dict['total_slip_values']

    # get cross section line endpoints for plotting labels
    cross_section_line_bounds = cross_section_line.geometry[0].bounds
    xs_label1_x, xs_label2_x = cross_section_line_bounds[0], cross_section_line_bounds[2]
    xs_label1_y, xs_label2_y = cross_section_line_bounds[3], cross_section_line_bounds[1]

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

    plot_xmin, plot_ymin, plot_xmax, plot_ymax = 1899000., 5570000., 2015000., 5670000.


    # Plot slip distribution on patches
    patches = tiled_fault.patch_collection2d(cmap=custom_slip_colormap, patch_alpha=1, line_colour="none",
                                             line_width=0.5)
    patches.set_array(total_slip_values)
    # set colorbar limits to 0 and 8
    patches.set_clim([0, 8])
    axs[0].add_collection(patches)

    #make outline of whole fault
    all_patch_coords = np.concatenate([array[:, 0:2] for array in tiled_fault.patch_corners])
    x_coords = np.concatenate([array[:, 0] for array in tiled_fault.patch_corners])
    y_coords = np.concatenate([array[:, 1] for array in tiled_fault.patch_corners])
    minx_i, miny_i = np.where(x_coords == x_coords.min()), np.where(y_coords == y_coords.min())
    maxx_i, maxy_i = np.where(x_coords == x_coords.max()), np.where(y_coords == y_coords.max())
    outline_coords = np.concatenate([all_patch_coords[miny_i], all_patch_coords[minx_i], all_patch_coords[maxy_i],
                                       all_patch_coords[maxx_i], all_patch_coords[miny_i]])
    outline_gs = gpd.GeoSeries(LineString([Point(corner) for corner in outline_coords]), crs=2193)
    outline_gs.plot(ax=axs[0], color="k", linewidth=0.5)

    # plot vertical deformation
    disps = axs[1].imshow(vertical_disp_grid[-1::-1], extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
                          cmap=custom_vertdef_colormap, vmin=-3, vmax=3)

    # get moment magnitude
    Mw, avg_slip = get_rect_Mw(preferred_slip=total_slip_values, patch_edge_size=patch_size)

    #make plot and axes pretty
    for ax in axs[0:2]:
        coastline.plot(ax=ax, color="k", linewidth=1.0)
        ahuriri_outline.plot(ax=ax, color="k")
        cross_section_line.plot(ax=ax, color="g", linewidth=1.0)
        ax.text(xs_label1_x - patch_size, xs_label1_y + patch_size, "X", ha='right', fontsize=6, color='g')
        ax.text(xs_label2_x + patch_size, xs_label2_y - patch_size, "X'", ha='left', fontsize=6, color='g')
        ax.set_xticks(np.arange(1880000., 2400000., 40000.))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mE'))
        ax.set_yticks(np.arange(5200000., 5720000., 40000.))
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
    interest_faults.plot(ax=axs[0], color="0.5", linewidth=1.0)
    ax.text(plot_xmin + patch_size, plot_ymax - 2*patch_size, f'Mw = {Mw:.1f}', ha='left', fontsize=6, color='k')

    #hide yaxis label on right plots
    axs[1].yaxis.set_ticklabels([])
    #add contour lines
    neg_contour = axs[1].contour(xgrid, ygrid, vertical_disp_grid, levels=[-1, -0.5], colors="b",
                linewidths=0.5, linestyles="dashed")
    pos_contour = axs[1].contour(xgrid, ygrid, vertical_disp_grid, levels=[-0, 1, 2, 3], colors="0.5",
                   linewidths=0.5)

    # make the colorbars pretty
    divider = make_axes_locatable(axs[0])
    cax1 = divider.append_axes('top', size='6%', pad=0.05)
    cbar1 = fig.colorbar(patches, cax=cax1, ticks=[0, 2, 4, 6, 8], orientation='horizontal')
    #cax1.set_aspect(0.5)
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
    make_def_cross_section(tiled_fault_dict, ax=axs[2])
    make_slip_cross_section(tiled_fault_dict, ax=axs[3])

    fig.suptitle(f"Model {extension1} {extension2} r{rake}", fontsize=8)
    fig.tight_layout()

    if not os.path.exists(f"{extension1}_figures"):
        os.mkdir(f"{extension1}_figures")

    #save figure
    #fig.savefig(f"figures/fig_{extension1}_{extension2}_r{rake}.pdf", transparent=True, dpi=300)
    fig.savefig(f"{extension1}_figures/fig_{extension1}_{extension2}_r{rake}.png", transparent=True, dpi=300)


#make_def_cross_section(extension_list)
#make_slip_cross_section(extension_list)

