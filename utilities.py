import earthaccess
import json
import pandas as pd
from AQI import get_general_aqi, get_pollutant_AQI

cities_ids = json.loads(open("filtered_cities_names.json", "r").read())["cities"]
cities_polygons = json.loads(open("filtered_cities_polygons.json", "r").read())

cities_ids = [cities_ids[idx] for idx in range(len(cities_ids))]

#auth = earthaccess.login("netrc")
#print(auth.authenticated)

no2_pixels = hcho_pixels = 118*310
o3_pixels = 122*310

session_data = {
    "forecasted_day": "2025-10-03",

    "pixels": {
        "no2":[0 for _ in range(no2_pixels)],
        "hcho":[0 for _ in range(hcho_pixels)],
        "o3":[0 for _ in range(o3_pixels)],
    },
    "pixels_predicted": {
        "no2": [[0 for _ in range(no2_pixels)] for day in range(7)],
        "hcho": [[0 for _ in range(no2_pixels)] for day in range(7)],
        "o3": [[0 for _ in range(no2_pixels)] for day in range(7)]
    },

    "days_dates_forecasted": ["2025-10-04", "2025-10-05", "2025-10-06", "2025-10-07" "2025-10-08", "2025-10-09", "2025-10-10"]
}

# fixed data
layers = ["no2", "hcho", "o3"]

doi_data = {
    "no2": "10.5067/IS-40e/TEMPO/NO2_L3.004",
    "hcho": "10.5067/IS-40e/TEMPO/HCHO_L3.004",
    "o3": "10.5067/IS-40e/TEMPO/O3TOT_L3.004"
}

def get_no2_pixel_idx(lat: float, lon: float) -> int:
    lat_idx = 2 * (lat - 14.01)
    lon_idx = 2 * (lon + 167.99)
    return int(lat_idx) * 310 + int(lon_idx)

def get_hcho_pixel_idx(lat: float, lon: float) -> int:
    lat_idx = 2 * (lat - 14.01)
    lon_idx = 2 * (lon + 167.99)
    return int(lat_idx) * 310 + int(lon_idx)

def get_o3_pixel_idx(lat: float, lon: float) -> int:
    lat_idx = (lat - 14.01)/(12*.04)
    lon_idx = 2 * (lon + 167.99)
    return int(lat_idx) * 310 + int(lon_idx)


def get_forecasted_pixel(no2_hcho_pixel_idx, o3_pixel_idx):
    no2_values = []
    hcho_values = []
    o3_values = []

    no2_AQI_values = []
    hcho_AQI_values = []
    o3_AQI_values = []

    AQI_General_values = []
    for day in range(7):
        no2_value = session_data["pixels_predicted"]["no2"][day][no2_hcho_pixel_idx]
        hcho_value = session_data["pixels_predicted"]["hcho"][day][no2_hcho_pixel_idx]
        o3_value = session_data["pixels_predicted"]["o3"][day][o3_pixel_idx]

        AQI_General = get_general_aqi(no2_value, hcho_value, o3_value)

        no2_AQI_value = get_pollutant_AQI("no2", no2_value)
        hcho_AQI_value = get_pollutant_AQI("hcho", hcho_value)
        o3_AQI_value = get_pollutant_AQI("o3", o3_value)

        no2_values.append(no2_value)
        hcho_values.append(hcho_value)
        o3_values.append(o3_value)

        no2_AQI_values.append(no2_AQI_value)
        hcho_AQI_values.append(hcho_AQI_value)
        o3_AQI_values.append(o3_AQI_value)

        AQI_General_values.append(AQI_General)

    return no2_values, hcho_values, o3_values, AQI_General_values, no2_AQI_values, hcho_AQI_values, o3_AQI_values


def get_no2_hcho_pixels_means(no2_hcho_pixels):
    results_no2 = [[] for _ in range(7)]
    results_hcho = [[] for _ in range(7)]

    for day_idx in range(7):
        for pixel_idx in no2_hcho_pixels:

            no2_value = session_data["pixels_predicted"]["no2"][day_idx][pixel_idx]
            hcho_value = session_data["pixels_predicted"]["hcho"][day_idx][pixel_idx]

            results_no2[day_idx].append(no2_value)
            results_hcho[day_idx].append(hcho_value)
    return ([sum(result_no2) / 7 for result_no2 in results_no2],
            [sum(result_hcho) / 7 for result_hcho in results_hcho])


def get_o3_pixels_means(o3_pixels):
    results_o3 = [[] for _ in range(7)]

    for day_idx in range(7):
        for pixel_idx in o3_pixels:
            o3_value = session_data["pixels_predicted"]["o3"][day_idx][pixel_idx]
            results_o3[day_idx].append(o3_value)
    return [sum(result_o3) / 7 for result_o3 in results_o3]


def get_forecasted_polygon_pixels(no2_hcho_pixels, o3_pixels):
    no2_pixels_means, hcho_pixels_means = get_no2_hcho_pixels_means(no2_hcho_pixels)
    o3_pixels_means = get_o3_pixels_means(o3_pixels)

    no2_values = []
    hcho_values = []
    o3_values = []

    no2_AQI_values = []
    hcho_AQI_values = []
    o3_AQI_values = []

    AQI_General_values = []

    for day in range(7):
        no2_value = no2_pixels_means[day]
        hcho_value = hcho_pixels_means[day]
        o3_value = o3_pixels_means[day]

        AQI_General = get_general_aqi(no2_value, hcho_value, o3_value)

        no2_AQI_value = get_pollutant_AQI("no2", no2_value)
        hcho_AQI_value = get_pollutant_AQI("hcho", hcho_value)
        o3_AQI_value = get_pollutant_AQI("o3", o3_value)

        no2_values.append(no2_value)
        hcho_values.append(hcho_value)
        o3_values.append(o3_value)

        no2_AQI_values.append(no2_AQI_value)
        hcho_AQI_values.append(hcho_AQI_value)
        o3_AQI_values.append(o3_AQI_value)

        AQI_General_values.append(AQI_General)
    return no2_values, hcho_values, o3_values, AQI_General_values, no2_AQI_values, hcho_AQI_values, o3_AQI_values


def read_forecasted_data(no2='/home/ahmed/Desktop/nasa_space_apps_2025_backend/daily_data/no2_forecast_7days.csv',
                         o3='/home/ahmed/Desktop/nasa_space_apps_2025_backend/daily_data/o3_forecast_7days.csv',
                         hcho='/home/ahmed/Desktop/nasa_space_apps_2025_backend/daily_data/hcho_forecast_7days.csv'):
    no2 = pd.read_csv(no2)
    o3 = pd.read_csv(o3)
    hcho = pd.read_csv(hcho)

    no2_dates = no2['datetime'].values.tolist()
    o3_dates = o3['datetime'].values.tolist()
    hcho_dates = hcho['datetime'].values.tolist()

    no2 = no2.drop(columns=['datetime']).values.tolist()
    o3 = o3.drop(columns=['datetime']).values.tolist()
    hcho = hcho.drop(columns=['datetime']).values.tolist()

    session_data["days_dates_forecasted"] = list(no2_dates)
    session_data["pixels_predicted"]["no2"] = no2
    session_data["pixels_predicted"]["hcho"] = hcho
    session_data["pixels_predicted"]["o3"] = o3

