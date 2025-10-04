from utilities import (earthaccess, session_data, doi_data,
                       get_no2_pixel_idx, get_hcho_pixel_idx, get_o3_pixel_idx,
                       cities_ids, cities_polygons,
                       get_forecasted_pixel, get_forecasted_polygon_pixels, read_forecasted_data)
import pandas as pd
from fastapi import Body


from search_polygon import query_pixels
from search_city import get_city_id

from AQI import get_pollutant_AQI, get_general_aqi

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Literal, List, Union, Optional
from enum import Enum
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from apscheduler.triggers.cron import CronTrigger
# from apscheduler.triggers.interval import IntervalTrigger
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
        coordinates = cities_polygons[int(city)]["coordinates"]
        name = cities_ids[int(city)]["name"]
        no2_hcho_pixels = query_pixels(coordinates, "no2")
        o3_pixels = query_pixels(coordinates, "o3")
        no2_values, hcho_values, o3_values, AQI_General_values, no2_AQI_values, hcho_AQI_values, o3_AQI_values =\
            get_forecasted_polygon_pixels(no2_hcho_pixels, o3_pixels)

        city_result = {
            "name": name,
            "polygon_arr": cities_polygons[int(city)],
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
@app.get("/get_polygon")
def get_city(city_id: str = Query(..., description="city_id")):
    coordinates = cities_polygons[int(city_id)]["coordinates"]
    name = cities_ids[int(city_id)]["name"]
    no2_hcho_pixels = query_pixels(coordinates, "no2")
    o3_pixels = query_pixels(coordinates, "o3")
    no2_values, hcho_values, o3_values, AQI_General_values, no2_AQI_values, hcho_AQI_values, o3_AQI_values = \
        get_forecasted_polygon_pixels(no2_hcho_pixels, o3_pixels)

    forecasted_dates = session_data["days_dates_forecasted"]

    city_result = {
        "city": {
            "name": name,
            "polygon_arr": cities_polygons[int(city_id)],
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
    layers_ = session_data[f"{layer_type}_layers"]

    # Return the photo link for the requested layer_type
    if layer_type in doi_data:
        if layers_:
            return {"status": True, "photos": layers_}
        else:
            return {"status": False}
    else:
        raise HTTPException(status_code=422, detail=f"Invalid layer_type: {layer_type}")


# Endpoint for getting polygon with coordinates as a JSON-string query parameter
@app.post("/get_polygon")
async def get_polygon(
    coordinates: List[List[float | int]] = Body(..., description="List of coordinates")
):
    no2_hcho_pixels = query_pixels(coordinates, "no2")
    o3_pixels = query_pixels(coordinates, "o3")
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

# Start the scheduler when the FastAPI app starts
@app.on_event("startup")
async def startup_event():
    read_forecasted_data()
    print(len(session_data["pixels_predicted"]["no2"][0]))
    print(len(session_data["pixels_predicted"]["o3"][0]))

    print(len(session_data["pixels_predicted"]["hcho"][0]))

    # print(session_data["pixels_predicted"])

# Stop the scheduler when the FastAPI app shuts down
# @app.on_event("shutdown")
# async def shutdown_event():
#     scheduler.shutdown()
