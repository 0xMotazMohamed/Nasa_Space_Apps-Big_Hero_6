import math

AVOGADRO = 6.02214076e23  # molecules/mol
DEFAULT_AIR_COLUMN = 5e22  # molecules/cm2 (typical air column at 1 atm)

BREAKPOINTS_PPB = {
    "no2": [
        (0.0, 53.0, 0, 50),
        (54.0, 100.0, 51, 100),
        (101.0, 360.0, 101, 150),
        (361.0, 649.0, 151, 200),
        (650.0, 1249.0, 201, 300),
        (1250.0, 1649.0, 301, 400),
        (1650.0, 2049.0, 401, 500),
    ],
    "hcho": [
        (0.0, 8.0, 0, 50),
        (9.0, 20.0, 51, 100),
        (21.0, 50.0, 101, 150),
        (51.0, 100.0, 151, 200),
        (101.0, 200.0, 201, 300),
        (201.0, 300.0, 301, 400),
        (301.0, 400.0, 401, 500),
    ],
    "o3": [
        (0.0, 54.0, 0, 50),
        (55.0, 70.0, 51, 100),
        (71.0, 85.0, 101, 150),
        (86.0, 105.0, 151, 200),
        (106.0, 200.0, 201, 300),
        (201.0, 300.0, 301, 400),
        (301.0, 400.0, 401, 500),
    ]
}

AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def _interpolate_aqi(C, bp):
    for (C_low, C_high, I_low, I_high) in bp:
        if C_low <= C <= C_high:
            if C_high == C_low:
                return float(I_high)
            return ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
    if C < bp[0][0]:
        return float(bp[0][2])
    return 500.0


def get_aqi_category(aqi_value):
    for (low, high, label) in AQI_CATEGORIES:
        if low <= aqi_value <= high:
            return label
    return "Beyond Index"


def column_to_ppb(column, *, column_units="molec/cm2", amf=1.0, air_column=DEFAULT_AIR_COLUMN):
    """Convert satellite column density (molecules/cm²) to ppb."""
    if column_units == "mol/m2":
        column_molec_cm2 = column * AVOGADRO / 1e4
    elif column_units == "molec/cm2":
        column_molec_cm2 = column
    else:
        raise ValueError("column_units must be 'molec/cm2' or 'mol/m2'")
    column_vert = column_molec_cm2 / amf
    return (column_vert / air_column) * 1e9


def ppb_to_aqi(ppb, pollutant):
    key = pollutant.strip()

    bp = BREAKPOINTS_PPB[key]
    return int(round(_interpolate_aqi(ppb, bp)))


def column_to_aqi(pollutant, column, *, column_units="molec/cm2", amf=1.0, air_column=DEFAULT_AIR_COLUMN):
    """Return only pollutant, AQI, and category."""
    mixing_ppb = column_to_ppb(column, column_units=column_units, amf=amf, air_column=air_column)
    aqi_value = ppb_to_aqi(mixing_ppb, pollutant)
    return {"AQI": aqi_value, "category": get_aqi_category(aqi_value)}

def get_pollutant_AQI(pollutant: str, column):
    return column_to_aqi(pollutant, column)


def get_general_aqi(no2_column, o3_column, hcho_column, *, column_units="molec/cm2", amf=1.0,
                             air_column=DEFAULT_AIR_COLUMN):
    no2_aqi = column_to_aqi("no2", no2_column, column_units=column_units, amf=amf, air_column=air_column)
    o3_aqi = column_to_aqi("o3", o3_column, column_units=column_units, amf=amf, air_column=air_column)
    hcho_aqi = column_to_aqi("hcho", hcho_column, column_units=column_units, amf=amf, air_column=air_column)

    aqi_list = [no2_aqi, o3_aqi, hcho_aqi]

    max_aqi = max(d["AQI"] for d in aqi_list)
    category = get_aqi_category(max_aqi)

    return {
        "AQI": max_aqi,
        "category": category,
    }