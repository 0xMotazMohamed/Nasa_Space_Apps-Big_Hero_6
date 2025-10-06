from utilities import (earthaccess, session_data, doi_data,
                       get_no2_pixel_idx, get_hcho_pixel_idx, get_o3_pixel_idx,
                       cities_ids, cities_polygons,
                       get_forecasted_pixel, get_forecasted_polygon_pixels, read_forecasted_data)


from get_layers_photos import get_urls



from search_polygon import get_city_pixels, get_polygon_pixels
from search_city import get_city_id


from fastapi import FastAPI, Query, HTTPException
from typing import Literal

import json

app = FastAPI()

# Endpoint for getting points with lon and lat
@app.get("/points")
def get_points(
        lon: float = Query(..., description="Longitude", ge=-167.99, le=-13.01),
        lat: float = Query(..., description="Latitude", ge=14.01, le=72.99)
):

    no2_hcho_pixel_idx = get_no2_pixel_idx(lat, lon)
    o3_pixel_idx = get_o3_pixel_idx(lat, lon)

    no2_values, hcho_values, o3_values, AQI_General_values, no2_AQI_values, hcho_AQI_values, o3_AQI_values = (
        get_forecasted_pixel(no2_hcho_pixel_idx, o3_pixel_idx))

    forecasted_dates = session_data["days_dates_forecasted"]

    city_result = {}

    city = get_city_id(lon, lat)
    if city:
        city_id = int(city)
        name = cities_ids[city_id]["name"]
        no2_hcho_pixels = get_city_pixels(city_id, "no2")
        o3_pixels = get_city_pixels(city_id, "o3")

        no2_values, hcho_values, o3_values, AQI_General_values, no2_AQI_values, hcho_AQI_values, o3_AQI_values =\
            get_forecasted_polygon_pixels(no2_hcho_pixels, o3_pixels)

        city_result = {
            "name": name,
            "polygon_arr": cities_polygons[city_id],
            "values": [
                {
                    "date": forecasted_dates[day],
                    "no2": {"value": no2_values[day], "AQI": {"value": no2_AQI_values[day]["AQI"], "category": no2_AQI_values[day]["category"]}},
                    "hcho": {"value": hcho_values[day], "AQI": {"value": hcho_AQI_values[day]["AQI"], "category": hcho_AQI_values[day]["category"]}},
                    "o3": {"value": o3_values[day],"AQI": {"value": o3_AQI_values[day]["AQI"], "category": o3_AQI_values[day]["category"]}},
                    "AQI_General": {"value": AQI_General_values[day]["AQI"], "category": AQI_General_values[day]["category"]}
                }
                for day in range(7)
            ]
        },
    response = {
        "dates": forecasted_dates,
        "point": {
            "values": [
                {
                    "date": forecasted_dates[day],
                    "no2": {"value": no2_values[day], "AQI": {"value": no2_AQI_values[day]["AQI"], "category": no2_AQI_values[day]["category"]}},
                    "hcho": {"value": hcho_values[day], "AQI": {"value": hcho_AQI_values[day]["AQI"], "category": hcho_AQI_values[day]["category"]}},
                    "o3": {"value": o3_values[day],"AQI": {"value": o3_AQI_values[day]["AQI"], "category": o3_AQI_values[day]["category"]}},
                    "AQI_General": {"value": AQI_General_values[day]["AQI"], "category": AQI_General_values[day]["category"]}
                }
                for day in range(7)
            ]
        },
        "city":city_result
    }

    return response


# Endpoint for getting city with city_name as query parameter
@app.get("/get_city")
def get_city(city_id: str = Query(..., description="city_id")):

    try:
        city_id_int = int(city_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="city_id must be an integer")

    if city_id_int > 5485 or city_id_int < 0:
        raise HTTPException(status_code=400, detail="city_id out of valid range (0–5485)")

    name = cities_ids[int(city_id)]["name"]
    no2_hcho_pixels = get_city_pixels(city_id_int, "no2")
    o3_pixels = get_city_pixels(city_id_int, "o3")
    no2_values, hcho_values, o3_values, AQI_General_values, no2_AQI_values, hcho_AQI_values, o3_AQI_values = \
        get_forecasted_polygon_pixels(no2_hcho_pixels, o3_pixels)

    forecasted_dates = session_data["days_dates_forecasted"]

    city_result = {
        "city": {
            "name": name,
            "polygon_arr": cities_polygons[city_id_int],
            "values": [
                {
                    "date": forecasted_dates[day],
                    "no2": {"value": no2_values[day],
                            "AQI": {"value": no2_AQI_values[day]["AQI"], "category": no2_AQI_values[day]["category"]}},
                    "hcho": {"value": hcho_values[day],
                             "AQI": {"value": hcho_AQI_values[day]["AQI"], "category": hcho_AQI_values[day]["category"]}},
                    "o3": {"value": o3_values[day],
                           "AQI": {"value": o3_AQI_values[day]["AQI"], "category": o3_AQI_values[day]["category"]}},
                    "AQI_General": {"value": AQI_General_values[day]["AQI"],
                                    "category": AQI_General_values[day]["category"]}
                }
                for day in range(7)
            ]
        }
    }

    return city_result


# Endpoint for getting layer with layer_type as path parameter (options: No2, o3, hcho)
@app.get("/layer/{layer_type}")
def get_layer(layer_type: Literal["no2", "o3", "hcho"]):
    # Define available photo links
    layers_ = get_urls(layer_type)

    # Return the photo link for the requested layer_type
    if layer_type in doi_data:
        if layers_:
            return {"layers": layers_}
        else:
            return {"layers": []}
    else:
        raise HTTPException(status_code=422, detail=f"Invalid layer_type: {layer_type}")


# Endpoint for getting polygon with coordinates as a JSON-string query parameter
@app.get("/get_polygon")
async def get_polygon(coordinates: str = Query(..., description="JSON array of coordinates")):
    try:
        # Parse the JSON string into Python list
        polygon = json.loads(coordinates)

        # Validate
        if not isinstance(polygon, list) or not all(isinstance(p, list) and len(p) == 2 for p in polygon):
            raise ValueError("Invalid polygon format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    no2_hcho_pixels = get_polygon_pixels(polygon, "no2")
    o3_pixels = get_polygon_pixels(polygon, "o3")

    if not no2_hcho_pixels or not o3_pixels:
        raise HTTPException(status_code=400, detail=f"Invalid polygon shape out of bbox")

    no2_values, hcho_values, o3_values, AQI_General_values, no2_AQI_values, hcho_AQI_values, o3_AQI_values = \
        get_forecasted_polygon_pixels(no2_hcho_pixels, o3_pixels)

    forecasted_dates = session_data["days_dates_forecasted"]

    polygon_result = {
        "polygon": {
            "values": [
                {
                    "date": forecasted_dates[day],
                    "no2": {"value": no2_values[day],
                            "AQI": {"value": no2_AQI_values[day]["AQI"], "category": no2_AQI_values[day]["category"]}},
                    "hcho": {"value": hcho_values[day],
                             "AQI": {"value": hcho_AQI_values[day]["AQI"],
                                     "category": hcho_AQI_values[day]["category"]}},
                    "o3": {"value": o3_values[day],
                           "AQI": {"value": o3_AQI_values[day]["AQI"], "category": o3_AQI_values[day]["category"]}},
                    "AQI_General": {"value": AQI_General_values[day]["AQI"],
                                    "category": AQI_General_values[day]["category"]}
                }
                for day in range(7)
            ]
        }
    }

    return polygon_result

@app.on_event("startup")
async def startup_event():
    read_forecasted_data()



# Stop the scheduler when the FastAPI app shuts down
# @app.on_event("shutdown")
# async def shutdown_event():
#     scheduler.shutdown()
