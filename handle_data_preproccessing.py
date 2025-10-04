import json
import numpy as np
import xarray as xr

from utilities import (earthaccess, 
                       session_data, 
                       doi_data, hcho_pixels,  no2_pixels, o3_pixels)


open_options = {
    "access": "indirect",  # access to cloud data (faster in AWS with "direct")
    "load": True,  # Load metadata immediately (required for indexing)
    "concat_dim": "time",  # Concatenate files along the time dimension
    "data_vars": "minimal",  # Only load data variables that include the concat_dim
    "coords": "minimal",  # Only load coordinate variables that include the concat_dim
    "compat": "override",  # Avoid coordinate conflicts by picking the first
    "combine_attrs": "override",  # Avoid attribute conflicts by picking the first
}

lat_idx, lon_idx = np.indices((2950, 7750), dtype=np.int32)
pixel_indices_no2_hcho = (lat_idx // 25) * 310 + (lon_idx // 25)

lat_idx_2, lon_idx_2 = np.indices((1475, 7750), dtype=np.int32)
pixel_indices_o3 = (lat_idx_2 // 12) * 310 + (lon_idx_2 // 25)


def preprocess_no2():
    today = session_data["today"]
    results = earthaccess.search_data(
        doi=doi_data["no2"],
        temporal=(f"{today} 00:00", f"{today} 23:59"),
    )
    if not results:
        for pixel_idx in range(no2_pixels):
            session_data["pixels"]["no2"][pixel_idx] = 0
        return

    result_product = earthaccess.open_virtual_mfdataset(granules=results, group="product", **open_options)
    result_support = earthaccess.open_virtual_mfdataset(granules=results, group="support_data", **open_options)

    # merge
    result_merged = xr.merge([result_product, result_support])
    clean_merged = result_merged.where(
        (result_merged["main_data_quality_flag"] == 0) &
        (result_merged["eff_cloud_fraction"] <= 0.2)
    ).mean(dim="time")

    values = clean_merged["vertical_column_troposphere"].values

    # Filter non-NaN values and their pixel indices
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    valid_pixel_indices = pixel_indices_no2_hcho[valid_mask]

    # Compute sums and counts per pixel
    sums = np.bincount(valid_pixel_indices, weights=valid_values, minlength=no2_pixels)
    counts = np.bincount(valid_pixel_indices, minlength=no2_pixels)

    # Compute averages (0 for empty bins), including negative values
    session_data["pixels"]["no2"] = np.divide(
        sums, counts, out=np.zeros(no2_pixels, dtype=float), where=counts != 0
    ).tolist()

    # Clean up
    del values, valid_mask, valid_values, valid_pixel_indices, sums, counts


def preprocess_hcho():
    today = session_data["today"]
    results = earthaccess.search_data(
        doi=doi_data["hcho"],
        temporal=(f"{today} 00:00", f"{today} 23:59"),
    )
    if not results:
        for pixel_idx in range(hcho_pixels):
            session_data["pixels"]["hcho"][pixel_idx] = 0
        return
    result_product = earthaccess.open_virtual_mfdataset(granules=results, group="product", **open_options)
    result_support = earthaccess.open_virtual_mfdataset(granules=results, group="support_data", **open_options)

    # merge
    result_merged = xr.merge([result_product, result_support])

    clean_merged = result_merged.where(
        (result_merged["main_data_quality_flag"] == 0) &
        (result_merged["eff_cloud_fraction"] <= 0.5)
    ).mean(dim="time")

    values = clean_merged["vertical_column"].values

    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    valid_pixel_indices = pixel_indices_no2_hcho[valid_mask]

    # Compute sums and counts per pixel
    sums = np.bincount(valid_pixel_indices, weights=valid_values, minlength=hcho_pixels)
    counts = np.bincount(valid_pixel_indices, minlength=hcho_pixels)

    # Compute averages (0 for empty bins), including negative values
    session_data["pixels"]["hcho"] = np.divide(
        sums, counts, out=np.zeros(hcho_pixels, dtype=float), where=counts != 0
    ).tolist()

    # Clean up
    del values, valid_mask, valid_values, valid_pixel_indices, sums, counts


def preprocess_o3():
    today = session_data["today"]
    results = earthaccess.search_data(
        doi=doi_data["o3"],
        temporal=(f"{today} 00:00", f"{today} 23:59"),
    )
    if not results:
        for pixel_idx in range(o3_pixels):
            session_data["pixels"]["o3"][pixel_idx] = 0
        return
    result_product = earthaccess.open_virtual_mfdataset(granules=results, group="product", **open_options)
    result_geolocation = earthaccess.open_virtual_mfdataset(granules=results, group="geolocation", **open_options)

    # merge
    result_merged = xr.merge([result_product, result_geolocation])

    clean_merged = result_merged.where(
        (result_merged["fc"] <= 0.2) &
        (result_merged['solar_zenith_angle'] <= 80) &
        (result_merged['viewing_zenith_angle'] <= 80)
    ).mean(dim="time")

    values = clean_merged["column_amount_o3"].values

    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    valid_pixel_indices = pixel_indices_o3[valid_mask]

    # Compute sums and counts per pixel
    sums = np.bincount(valid_pixel_indices, weights=valid_values, minlength=o3_pixels)
    counts = np.bincount(valid_pixel_indices, minlength=o3_pixels)

    # Compute averages (0 for empty bins), including negative values
    session_data["pixels"]["o3"] = np.divide(
        sums, counts, out=np.zeros(o3_pixels, dtype=float), where=counts != 0
    ).tolist()

    # Clean up
    del values, valid_mask, valid_values, valid_pixel_indices, sums, counts

