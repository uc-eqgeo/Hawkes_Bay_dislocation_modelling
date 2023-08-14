import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import meshio
from scipy.interpolate import griddata
import math


def map_slip_dist(slip_dist_poly: Polygon, fault_mesh: meshio.Mesh, max_slip: float, taper_dist_km: float = 10.,
                  buffer_dist_km: float = 100., taper_power: float = 2.):
    """
    Map slip distribution to fault mesh

    :param slip_dist_poly:
    :param fault_mesh:
    :param max_slip:
    :param taper_dist_km: number of kilometres to taper slip distribution over (excess to polygon)
    :param buffer_dist_km: number of kilometres to buffer slip distribution by
    :param slip_poly_interp_dist_km: number of kilometres to interpolate slip distribution by
    :return:
    """
    assert isinstance(slip_dist_poly, Polygon)
    assert isinstance(fault_mesh, meshio.Mesh)

    buffered_poly = slip_dist_poly.buffer(buffer_dist_km * 1000.)
    slip_dist_poly_outline = LineString(slip_dist_poly.exterior.coords)

    # Get triangle centres
    mesh_tris = fault_mesh.cells_dict["triangle"].byteswap().newbyteorder()
    mesh_points = fault_mesh.points.byteswap().newbyteorder()
    tri_centres = np.mean(mesh_points[mesh_tris], axis=1).byteswap().newbyteorder()
    tri_rakes = fault_mesh.cell_data["rake"][0].byteswap().newbyteorder()
    centres_with_rake = np.column_stack((tri_centres, tri_rakes, np.zeros(tri_rakes.shape)))
    centres_gdf = gpd.GeoDataFrame(centres_with_rake, columns=["x", "y", "z", "rake", "slip"],
                                   geometry=[Point(x, y, z) for (x, y, z) in tri_centres], crs=2193)
    triangles_gdf = gpd.GeoDataFrame({"rake": tri_rakes, "slip": np.zeros(tri_rakes.shape)},
                                     geometry=[Polygon(tri) for tri in mesh_points[mesh_tris]], crs=2193)

    # Get points within buffer
    points_within_buffer = centres_gdf[centres_gdf.within(buffered_poly)]
    triangles_within_buffer = triangles_gdf.iloc[points_within_buffer.index]
    points_within_slip_dist = centres_gdf[centres_gdf.within(slip_dist_poly)]
    points_for_taper = points_within_buffer[~points_within_buffer.within(slip_dist_poly)]
    nearest_points = gpd.GeoSeries([slip_dist_poly_outline.interpolate(slip_dist_poly_outline.project(point)) for point in points_for_taper["geometry"]])

    nearest_points_array = np.array([(point.x, point.y) for point in nearest_points])
    centres_array = np.array([(point.x, point.y, point.z) for point in points_within_buffer["geometry"]])
    points_for_taper_array = np.array([(point.x, point.y, point.z) for point in points_for_taper["geometry"]])

    nearest_z = griddata(centres_array[:, :2], centres_array[:, -1], nearest_points_array, method="linear")
    nearest_xyz = np.column_stack((nearest_points_array, nearest_z))
    shortest_dists = np.linalg.norm(nearest_xyz - points_for_taper_array, axis=1)

    shortest_dists_slip = max_slip * (1 - (shortest_dists / (taper_dist_km * 1000.))**taper_power)
    shortest_dists_slip[shortest_dists_slip < 0.] = 0.
    triangles_within_buffer.loc[points_for_taper.index, "slip"] = shortest_dists_slip
    triangles_within_buffer.loc[points_within_slip_dist.index, "slip"] = max_slip

    return triangles_within_buffer

def calc_edges(triangle_coords):
    """
    calculate triangle edge lengths given xyz coordinates in tuples/arrays
    """

    x = [triangle_coords[0, 0], triangle_coords[1, 0], triangle_coords[2, 0]]
    y = [triangle_coords[0, 1], triangle_coords[1, 1], triangle_coords[2, 1]]
    z = [triangle_coords[0, 2], triangle_coords[1, 2], triangle_coords[2, 2]]
    edge1 = math.sqrt(((x[0] - x[1]) ** 2) + ((y[0] - y[1]) ** 2) + ((z[0] - z[1]) ** 2))
    edge2 = math.sqrt(((x[1] - x[2]) ** 2) + ((y[1] - y[2]) ** 2) + ((z[1] - z[2]) ** 2))
    edge3 = math.sqrt(((x[2] - x[0]) ** 2) + ((y[2] - y[0]) ** 2) + ((z[2] - z[0]) ** 2))

    return [edge1, edge2, edge3]

def get_Mw(triangle_coords, slip_dist_array):
    """calculate moment magnitude of rupture patch from triangular mesh
    mesh should be in nx3x3 format (three sets of xyz coords per triangle)
    """

    # calculate area of all ruptured triangles within mesh
    triangle_areas = []
    for triangle in triangle_coords:
        #calcualte traingle edge lengths
        edge_ls = calc_edges(triangle)
        # calculate semi perimeter
        sp = sum(edge_ls)/2
        # calculate triangle area with heron's formula
        triangle_area = math.sqrt(sp * (sp - edge_ls[0]) * (sp - edge_ls[1]) * (sp - edge_ls[2]))
        triangle_areas.append(triangle_area)

    # calculate average slip for triangles with slip values > 0
    total_slip_area = 0
    total_slip_times_area = 0

    for i in range(len(slip_dist_array)):
        if slip_dist_array[i] > 0:
            #print("slip = " + "{:.2f}".format(slip_dist_array[i]))
            total_slip_times_area += slip_dist_array[i] * triangle_areas[i]
            total_slip_area += triangle_areas[i]

    avg_slip = total_slip_times_area / total_slip_area
    #print("avg slip = " + "{:.2f}".format(avg_slip))

   # calculate magnitude
    # m0 = 3.e10 * avg slip * fault area
    m0 = 3.e10 * avg_slip * total_slip_area
    Mw = (2./3.) * (np.log10(m0) - 9.05)
    return Mw, avg_slip




