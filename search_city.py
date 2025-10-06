from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.strtree import STRtree
from utilities import cities_polygons


cities_polygons_size = len(cities_polygons)

cities_polygons_geometry = []

for idx in range(cities_polygons_size):
    coords = cities_polygons[idx]["coordinates"]
    geometry_type = cities_polygons[idx]["type"]

    # Check if this is a multipolygon (list of polygons)
    if geometry_type == "MultiPolygon":
        poly_list = [Polygon(ring) for ring in coords]   # Build list of Polygon objects
        cities_polygons_geometry.append(MultiPolygon(poly_list))
    else:
        cities_polygons_geometry.append(Polygon(coords))

# Build spatial index
tree = STRtree(cities_polygons_geometry)

# Example point



def get_city_id(lon: float, lat: float):
    point = Point([lon, lat])
    candidates = tree.query(point)
    print(candidates)
    if candidates.size > 0:
        return candidates[0]
    else:
        return None
