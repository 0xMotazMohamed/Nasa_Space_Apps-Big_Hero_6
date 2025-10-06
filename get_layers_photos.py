from arcgis.gis import GIS
from arcgis.raster import ImageryLayer
from datetime import datetime, timezone
import requests
from urllib.parse import urlencode

gis = GIS()

bbox = "-168.00000549350375,14.00000081336853,-13.00001324350373,72.99999786336853"  # North America
size = "1920,1080"
output_format = "png"  # TIFF not supported
image_sr = "4326"
bbox_sr = "4326"
params = {
    "f": "image",
    "format": output_format,
    "bbox": bbox,
    "size": size,
    "imageSR": image_sr,
    "bboxSR": bbox_sr,
    "time": "",
    "renderingRule": ""
}

raster_urls = {
    "no2": "https://gis.earthdata.nasa.gov/image/rest/services/C3685896708-LARC_CLOUD/TEMPO_NO2_L3_V04_HOURLY_TROPOSPHERIC_VERTICAL_COLUMN/ImageServer",
    "hcho": "https://gis.earthdata.nasa.gov/image/rest/services/C3685897141-LARC_CLOUD/TEMPO_HCHO_L3_V04_HOURLY_VERTICAL_COLUMN/ImageServer",
    "o3": "https://gis.earthdata.nasa.gov/image/rest/services/C3685896625-LARC_CLOUD/TEMPO_O3TOT_L3_V04_HOURLY_OZONE_COLUMN_AMOUNT/ImageServer"
}

layers_templates = {
    "no2": "vertical_column_troposphere",
    "hcho": "vertical_column",  # vertical_column HCHO
    "o3": "column_amount_o3"
}


def get_dates(layer: str):
    try:
        raster_ImageryLayer = ImageryLayer(raster_urls[layer], gis=gis)
        md_info = raster_ImageryLayer.multidimensional_info
        return md_info["multidimensionalInfo"]["variables"][0]["dimensions"][0]["values"]
    except:
        print(" --- no urs for this layer --- ")
        return []


def get_latest_day_photos(timestamps):
    try:
        if not timestamps:
            return []

        # Convert all timestamps to datetime objects (UTC)
        datetimes = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in timestamps]

        # Find the most recent date
        latest_date = max(dt.date() for dt in datetimes)

        # Return all timestamps that match that latest date
        latest_photos = [ts for ts, dt in zip(timestamps, datetimes) if dt.date() == latest_date]

        return latest_photos, [str(datetime.fromtimestamp(t / 1000).strftime('%Y-%m-%d %H:%M:%S')) for t in
                               latest_photos]
    except:
        print(" --- no layers photos --- ")
        return [], []


def get_urls(layer: str):
    try:
        urls = get_dates(layer)

        if not urls:
            return []
        layers_photos = get_latest_day_photos(urls)

        if not layers_photos[0]:
            return []
        result = []
        dates, dates_str = layers_photos
        for i in range(len(dates)):
            date_str = dates_str[i]
            t = dates[i]
            params["time"] = f"{t},{t}"
            params["renderingRule"] = f'{{"rasterFunction":"{layers_templates[layer]}"}}'
            raster_url = raster_urls[layer]
            query_str = urlencode(params)
            full_url = f"{raster_url}/exportImage?{query_str}"

            result.append({
                "date": date_str,
                "url": full_url
            })
        return result
    except:
        return []
