import shapely.geometry as geom
from shapely.strtree import STRtree
from shapely.prepared import PreparedGeometry
from search_city import cities_polygons_geometry

# =========================
# GRID INITIALIZATION (done once at startup)
# =========================

# Fixed grid parameters (same for all species)
min_lon, min_lat, max_lon, max_lat = -167.99, 14.01, -13.01, 72.99
bbox = geom.box(min_lon, min_lat, max_lon, max_lat)

# Step sizes for each species
step_no2_lat = 0.02 * 25  # 0.5 degrees
step_no2_lon = 0.02 * 25  # 0.5 degrees
step_hcho_lat = 0.02 * 25  # 0.5 degrees
step_hcho_lon = 0.02 * 25  # 0.5 degrees
step_o3_lat = 0.04 * 12  # 0.48 degrees
step_o3_lon = 0.02 * 25  # 0.5 degrees


# Function to initialize a grid for a given species
def initialize_grid(step_lon, step_lat):
    nx = int((max_lon - min_lon) / step_lon)
    ny = int((max_lat - min_lat) / step_lat)

    pixels = []
    pixel_indices = []  # parallel list: (ix, iy)
    for iy in range(ny):
        lat0 = min_lat + iy * step_lat
        lat1 = lat0 + step_lat
        for ix in range(nx):
            lon0 = min_lon + ix * step_lon
            lon1 = lon0 + step_lon
            pixels.append(geom.box(lon0, lat0, lon1, lat1))
            pixel_indices.append((ix, iy))

    # Build spatial index (STRtree for fast spatial queries)
    tree = STRtree(pixels)

    # Build id→index mapping (for fast lookup of pixel indices)
    pixel_map = {id(p): idx for idx, p in enumerate(pixels)}

    return pixels, pixel_indices, tree, pixel_map


no2_grid = initialize_grid(step_no2_lon, step_no2_lat)
hcho_grid = initialize_grid(step_hcho_lon, step_hcho_lat)
o3_grid = initialize_grid(step_o3_lon, step_o3_lat)

grids = {
    'no2': no2_grid,
    'hcho': hcho_grid,
    'o3': o3_grid
}

# =========================
# QUERY FUNCTION
# =========================

def query_pixels(geometry, species):
    # Get the grid for the specified species
    pixels, pixel_indices, tree, pixel_map = grids[species]

    prep_poly = PreparedGeometry(geometry)

    candidate_idxs = tree.query(geometry)

    hits = []
    for idx in candidate_idxs:
        cand = pixels[idx]  # get actual geometry
        if prep_poly.intersects(cand):
            hits.append(pixel_indices[idx])

    return [int(pixel[1]) * 310 + int(pixel[0]) for pixel in hits]

def get_polygon_pixels(polygon_points, species):
    poly = geom.Polygon(polygon_points)
    if bbox.contains(poly):
        return query_pixels(geom.Polygon(polygon_points), species)
    else:
        return []

def get_city_pixels(city_id, species):
    return query_pixels(cities_polygons_geometry[city_id], species)
