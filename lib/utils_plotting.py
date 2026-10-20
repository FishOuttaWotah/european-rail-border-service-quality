import typing
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TODO: check refactor breakage, renew docstring
def make_basemap(basemap: Path | gpd.GeoDataFrame | gpd.GeoSeries,
                 bounds_source, crs=None,figsize=(12, 10), pad_frac=0.05, basecolor="gainsboro", edgecolor="darkgray", linewidth=0.5):
    """Create a geopandas map figure with an optional basemap and auto-fitted bounds.

        Parameters
        ----------
        basemap_path : pathlib.Path or Geodataframe
            Path to the basemap shapefile/GeoPackage. Loaded only if it exists.
        crs : any
            CRS to reproject the basemap into (typically the CRS of your data layer).
        bounds_source : GeoDataFrame
            GeoDataFrame whose total_bounds determine the map extent.
        figsize : tuple, optional
            Figure size in inches.
        pad_frac : float, optional
            Fractional padding added around the bounding box on each side.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(basemap, Path):
        if basemap.exists():
            _basemap = gpd.read_file(basemap).to_crs(crs)
    elif isinstance(basemap, (gpd.GeoDataFrame, gpd.GeoSeries)):
        _basemap = basemap
    else: raise TypeError(f"arg basemap_path expects Path or geopandas GeoDF/GeoSeries, got instead {type(basemap)}.")
    
    _basemap.plot(
        ax=ax, color=basecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=0, aspect=None,
    )
    minx, miny, maxx, maxy = bounds_source.total_bounds
    pad_x = (maxx - minx) * pad_frac if maxx > minx else 1.0
    pad_y = (maxy - miny) * pad_frac if maxy > miny else 1.0
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_axis_off()
    ax.set_aspect("auto")
    return fig, ax



# def build_station_geo_from_vertices(
#     vertices_df: pd.DataFrame,
#     stops_gdf: gpd.GeoDataFrame,
# ) -> gpd.GeoDataFrame:
#     """Build Station-only GeoDataFrame from igraph vertex dataframe + stops."""
#     station_nodes = vertices_df.loc[vertices_df["node_type"].eq("Station")].copy()
#     station_nodes["stop_id"] = pd.to_numeric(station_nodes["ori_id"], errors="coerce").astype("Int64")

#     stops_plot = stops_gdf.copy()
#     stops_plot["stop_id"] = pd.to_numeric(stops_plot.index, errors="coerce").astype("Int64")

#     stations_geo = station_nodes.merge(
#         stops_plot[["stop_id", "geometry"]],
#         on="stop_id",
#         how="inner",
#     )
#     stations_geo = gpd.GeoDataFrame(stations_geo, geometry="geometry", crs=stops_plot.crs)
#     return stations_geo


# def plot_station_importance_map(
#     metric_col: str,
#     stations_geo: gpd.GeoDataFrame | None = None,
#     vertices_df: pd.DataFrame | None = None,
#     stops_gdf: gpd.GeoDataFrame | None = None,
#     ne_path: str = "../maps/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp",
#     figsize: tuple = (10, 8),
#     point_color: str = "#1f78b4",
#     zero_point_color: str = "#b0a2a2",
#     alpha: float = 0.8,
#     scale_mode: str = "log",
# ):
#     """Plot station importance over Natural Earth basemap.

#     Use either:
#     - stations_geo directly, OR
#     - vertices_df (e.g., gl_results / gvs_results) + stops_gdf.
#     """
#     if stations_geo is None:
#         stations_geo = build_station_geo_from_vertices(vertices_df, stops_gdf)

#     plot_df = stations_geo.copy()
#     if scale_mode == "log":
#         _sizes, _mask_positive = set_nodesize_logscale(stations_geo[metric_col])
#     elif scale_mode == "linear":
#         _sizes, _mask_positive = set_nodesize_linearscale(stations_geo[metric_col])
#     else:
#         raise ValueError("scale_mode must be one of {'log', 'linear'}")
#     plot_df["marker_size"] = _sizes
#     plot_df["_metric_positive"] = _mask_positive
#     zero_mask = plot_df[metric_col].eq(0)
#     min_marker_size = float(np.nanmin(plot_df["marker_size"])) if len(plot_df) else 0.0

#     basemap = gpd.read_file(ne_path)
#     if plot_df.crs is None: raise AttributeError("CRS is not set in the reference")
#     target_crs = plot_df.crs
#     basemap = basemap.to_crs(target_crs)

#     fig, ax = plt.subplots(figsize=figsize)
#     basemap.plot(
#         ax=ax,
#         color="lightgrey",
#         edgecolor="dimgray",        
#         linewidth=0.5,
#         zorder=0,
#     )

#     plot_df.loc[zero_mask].plot(
#         ax=ax,
#         color=zero_point_color,
#         markersize=min_marker_size,
#         alpha=alpha,
#         linewidth=0.15,
#         edgecolor="white",
#         zorder=8,
#     )

#     plot_df.loc[~zero_mask].plot(
#         ax=ax,
#         color=point_color,
#         markersize=plot_df.loc[~zero_mask, "marker_size"],
#         alpha=alpha,
#         linewidth=0.15,
#         edgecolor="white",
#         zorder=10,
#     )

#     minx, miny, maxx, maxy = plot_df.total_bounds
#     pad_x = (maxx - minx) * 0.05 if maxx > minx else 1.0
#     pad_y = (maxy - miny) * 0.05 if maxy > miny else 1.0
#     ax.set_xlim(minx - pad_x, maxx + pad_x)
#     ax.set_ylim(miny - pad_y, maxy + pad_y)

#     n_total = len(plot_df)
#     n_pos = int(plot_df["_metric_positive"].sum())
#     ax.set_title(
#         f"Station importance ({metric_col}, {scale_mode}-scaled sizes)\n"
#         f"Stations: {n_total:,} | Positive metric: {n_pos:,}",
#     )
#     ax.set_axis_off()
#     return fig, ax, plot_df

def set_nodesize_logscale(
    metric_col: pd.Series,
    min_size: float = 8.0,
    max_size: float = 220.0,
    log_base: float = 10.0,
    zero_na_size: float = 8.0,
) -> typing.Tuple[np.ndarray, pd.Series]:
    """Prepare station points with marker sizes from a metric using log scaling.

    Positive values are scaled logarithmically to [min_size, max_size].
    Zero/negative/NaN values are kept with fixed size `zero_na_size`.
    Returns a tuple of the scaled sizes (numpy array) and mask of positive values.
    """

    # check for numeric dtype
    if not pd.api.types.is_numeric_dtype(metric_col.dtype):
        raise TypeError(f"Given field '{metric_col}' expected to be numeric, instead got dtype {metric_col.dtype}.")

    # calculate in numpy scaling
    sizes = np.full(len(metric_col), float(zero_na_size), dtype=float)
    positive_mask = metric_col > 0
    if positive_mask.any():
        logs = np.log(metric_col.loc[positive_mask]) / np.log(log_base)
        lo, hi = logs.min(), logs.max()

        if np.isclose(lo, hi):
            scaled = np.full(logs.shape[0], (min_size + max_size) / 2.0, dtype=float)
        else:
            scaled = min_size + (logs - lo) * (max_size - min_size) / (hi - lo)

        sizes[np.flatnonzero(positive_mask.to_numpy())] = scaled

    return sizes, positive_mask


# def set_nodesize_linearscale(
#     metric_col: pd.Series,
#     min_size: float = 8.0,
#     max_size: float = 220.0,
#     zero_na_size: float = 8.0,
# ) -> typing.Tuple[np.ndarray, pd.Series]:
#     """Prepare station points with marker sizes from a metric using linear scaling.

#     Positive values are scaled linearly to [min_size, max_size].
#     Zero/negative/NaN values are kept with fixed size `zero_na_size`.
#     Returns a tuple of the scaled sizes (numpy array) and mask of positive values.
#     """

#     if not pd.api.types.is_numeric_dtype(metric_col.dtype):
#         raise TypeError(f"Given field '{metric_col}' expected to be numeric, instead got dtype {metric_col.dtype}.")

#     sizes = np.full(len(metric_col), float(zero_na_size), dtype=float)
#     positive_mask = metric_col > 0
#     if positive_mask.any():
#         vals = metric_col.loc[positive_mask]
#         lo, hi = vals.min(), vals.max()

#         if np.isclose(lo, hi):
#             scaled = np.full(vals.shape[0], (min_size + max_size) / 2.0, dtype=float)
#         else:
#             scaled = min_size + (vals - lo) * (max_size - min_size) / (hi - lo)

#         sizes[np.flatnonzero(positive_mask.to_numpy())] = scaled

#     return sizes, positive_mask