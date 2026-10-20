from typing import Iterable
import numpy as np
import geopandas as gpd
from haversine import haversine_vector

def flag_border_crossing(icc_O: str, icc_D: str)-> bool:
    """Compare if a link connects 2 different countries."""
    return icc_O != icc_D


def great_circle_distance_km(arr1: gpd.GeoSeries,
                             arr2: gpd.GeoSeries,
                             unit="km",
                             comb=False) -> np.array:
    """Calculate GCD between 2 GeoSeries arrays, element-wise"""

    def check_external_dtype(*arrays: gpd.GeoSeries):
        output = []
        for idx, arr in enumerate(arrays):
            if not isinstance(arr, gpd.GeoSeries):
                # attempt to cast to geoseries
                try:
                    output.append(gpd.GeoSeries(arr))
                except Exception as e:
                    print(
                        f"Array {idx} cannot be casted to geopandas GeoSeries."
                    )
                    raise e
            else:
                output.append(arr)
        return output

    # check_internal_dtype(arr1, arr2)
    arr1, arr2 = check_external_dtype(arr1, arr2)
    points1 = list(zip(arr1.y, arr1.x)) # haversine uses l
    points2 = list(zip(arr2.y, arr2.x))
    return haversine_vector(points1, points2, unit=unit, comb=comb)
