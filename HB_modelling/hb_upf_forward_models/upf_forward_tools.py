import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, MultiPoint
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

def get_rect_Mw(preferred_slip, patch_edge_size):
    ### Calculate magnitude
    summed_slip, patches_with_slip = 0, 0
    # solve for total slip
    for slip_val in preferred_slip:
        if slip_val != 0:
            summed_slip += slip_val
            patches_with_slip += 1
    # calculate average slip on patches that moved
    avg_slip = summed_slip / patches_with_slip
    m0 = 3.e10 * avg_slip * (patches_with_slip * patch_edge_size * patch_edge_size)
    Mw = (2. / 3.) * (np.log10(m0) - 9.05)

    return Mw, avg_slip




