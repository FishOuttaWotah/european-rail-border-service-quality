"""
Plotting utilities for network visualizations.
"""
import typing

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Tuple, Union, Optional, List, Any

CBAR_WIDTH = 0.4
CBAR_THICK = 10

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def _make_colorscale(norm:mcolors.Normalize, cmap:mcolors.Colormap
    )-> Tuple[List[Tuple[float, str]], cm.ScalarMappable]:
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Sample and convert to Plotly colorscale format
    sample = np.linspace(norm.vmin, norm.vmax, 256)
    colorscale = []
    for val in sample:
        rgba = sm.to_rgba(val)
        rgb_str = f"rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})"
        pos = (val - norm.vmin) / (norm.vmax - norm.vmin)
        colorscale.append([pos, rgb_str])
    
    return colorscale, sm
    

def _build_colorbar_trace(
    colorscale: List[Tuple[float, str]],
    norm: mcolors.Normalize,
    title: str,
    x: float,
    y: float,
    orientation: str = "h",
    thickness: int = CBAR_THICK,
    length: float = CBAR_WIDTH,
    tickvals: Optional[List[float]] = None,
    ticktext: Optional[List[str]] = None,
) -> go.Scatter:
    """Build a horizontal colorbar trace positioned in plot coordinates."""
    colorbar_kwargs = dict(
        title=dict(text=title, side="top"),
        thickness=thickness, len=length,
        x=x, y=y, orientation=orientation,
        xanchor="left", yanchor="top",
    )
    if tickvals is not None and ticktext is not None:
        colorbar_kwargs["tickvals"] = tickvals
        colorbar_kwargs["ticktext"] = ticktext
    return go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale=colorscale, showscale=True,
            colorbar=colorbar_kwargs,
            cmin=norm.vmin, cmax=norm.vmax,
        ),
        showlegend=False, hoverinfo="none", name=""
    )


def _prepare_edges(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str,
    ) -> Tuple[pd.DataFrame, mcolors.Normalize]:
    """
        Validate, extract, and normalize edge data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing edge data.
        source_col, target_col, weight_col : str
            Column names for source, target, weight.
        norm : matplotlib.colors.Normalize, optional
            Custom normalizer; if None, auto compute min-max.
        
        Returns
        -------
        tuple
            (processed edges DataFrame, normalizer)
    """
    # Validate columns
    required_cols = {source_col, target_col, weight_col}
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Missing columns: {missing}")
    
    # Extract and standardize column names
    edges = (data[[source_col, target_col, weight_col]].copy()
             .rename(columns={
                source_col:"source",
                target_col: "target",
                weight_col: "weight"}))
    
    return edges


def _build_edges(
    edges: pd.DataFrame, 
    pos_dict: Dict[str, Tuple[float, float]],
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    colorbar_title:str,
    node_standoff:float,
    offset_deg:float,
    nan_color: str = "rgb(160, 160, 160)",
    ): 
    # Generate arrow edge coordinates, with transverse bias
    basetraces = _get_edge_coordinates(edges, pos_dict)
    edges_coords = _build_edges_directed(*basetraces, offset_deg)
    
    colorscale, sm = _make_colorscale(norm, cmap)
    
    edge_traces = []
    edge_sep_traces = []
    for coords, weight in zip(edges_coords, edges["weight"].values):
        # print(coords_O, coords_D, weight)
        # print(mpos, mdeg, coords_O, coords_D)
        
        if pd.isna(weight):
            arrow_color = nan_color
        else:
            rgba = [str(int(elem * 255)) for elem in sm.to_rgba(weight)]
            arrow_color = f"rgb({", ".join(rgba)})"

        edge_traces.append(dict(

            x=coords[0], y=coords[1],
            fill="toself",
            fillcolor=arrow_color,
            mode="lines",
            line=dict(color=arrow_color, width=1),
            hoverinfo="skip",
            showlegend=False, 
            name=""
        )) 
        # add separator trace (to distinguish to-from lines)
        edge_sep_traces.append(dict(
            x=coords[0,2:3], y=coords[1,2:3],  # use the hypotenuse of triangle
            mode="lines", line=dict(color="gainsboro", width=1), 
            showlegend=False, name="",
        ))

    return edge_traces, colorscale, edge_sep_traces

def _build_edges_directed(
    O_coords:pd.Series, D_coords:pd.Series,
    offset_deg:float,
    )-> np.ndarray:
    """Generates edge vertices as a 3-vertex triangle polygon. 
       Parameters:
            offset_deg: distance (in map terms) to offset from the main line between nodes. The polygon vertices will be offset by 50% of this value left and right.
        Returns: numpy 3d array, with axes (start/end, x/y coords , edge vertices)
            
    """
    # print(O_coords.values)
    O = np.array(O_coords.to_list(), dtype=float)
    D = np.array(D_coords.to_list(), dtype=float)

    # 1. Displacement vectors (N, 2) and Euclidean lengths (N, 1)
    disp = D - O
    lengths = np.linalg.norm(disp, axis=1, keepdims=True)
    # Avoid division by zero for self-loops/coincident nodes
    lengths[lengths == 0] = 1.0
    
    # 2. Normalized unit direction vectors u = [u_x, u_y]
    u = disp / lengths
    
    # 3. Rightward perpendicular normal vectors n = [u_y, -u_x]
    # Replaces [x, y] with [y, -x] across all rows
    n = np.column_stack((u[:, 1], -u[:, 0]))
    
    # 4. Apply transverse shift simultaneously across all N pairs
    # define triangle vertices
    midpoint = (O + 0.5 * disp)
    shift = 0.5 * offset_deg * n
    # return O_prime, D_prime    
    base_right = O + shift
    base_left = O - shift
    end_right = midpoint + shift
    end_left = midpoint - shift
    # vertex = D + shift # vertex is right-biased
    
    # stack in 3dim, extract by 3rd dim TODO document!
    # polygon_vertices = np.dstack([base_left, base_right, vertex, base_left])
    polygon_vertices = np.dstack([base_left, base_right, end_right, end_left, base_left])
     
    # print(polygon_vertices[0, :, :])
    # raise NotImplementedError
    return polygon_vertices
    
    
def _get_edge_coordinates(
    edges: pd.DataFrame,
    pos_dict: Dict[str, Tuple[float, float]],
    )-> Tuple[pd.Series, pd.Series]:
    return edges["source"].map(pos_dict), edges["target"].map(pos_dict)

def _build_node_trace(
    all_nodes: set,
    pos_dict: Dict[str, Tuple[float, float]],
    node_size: float = 40,
    node_text_scale: float = 0.3,
    # node_size_dict: Optional[Dict[str, float]] = None,
    node_colors: Optional[Dict[str, str]] = None,
    # node_cmap: Optional[mcolors.Colormap] = None,
    nan_color=None,
    ) -> Tuple[go.Scatter, Tuple]:
    """
        Build Plotly scatter trace for nodes.
        
        Parameters
        ----------
        all_nodes : set
            Set of all node codes.
        pos_dict : dict
            Node position mapping {code: (x, y)}.
        node_size : float, optional
            Node marker size in pixels (default: 40).
        node_text_scale : float, optional
            Node text font size as a proportion of node_size (default: 0.3).
            Computed as: int(node_size * node_text_scale).
        node_size_dict : dict, optional
            Per-node size overrides as {node_code: size_value}. If provided, node sizes
            are taken from this dict; missing nodes use node_size default (default: None).
        node_colors : dict, optional
            Per-node color overrides as {node_code: rgb_color_string}. If provided, node
            marker colors are taken from this dict; missing nodes use a default gray
            (default: None).
        node_cmap : matplotlib.colors.Colormap, optional
            Colormap used for node coloring (default: None).
        
        Returns
        -------
        go.Scatter
            Node trace object.
    """
    node_codes = sorted(all_nodes)
    node_x = [pos_dict[code][0] for code in node_codes]
    node_y = [pos_dict[code][1] for code in node_codes]
    bounds: Tuple[float, float, float, float] = [min(node_x), max(node_x), min(node_y), max(node_y)]
    
    sizes = node_size
    font_sizes = int(node_size * node_text_scale)
    
    # Optional per-node colouring
    if node_colors is not None:
        marker_colors = [node_colors.get(code, nan_color) for code in node_codes]
        text_colors = []
        for code in node_codes:
            rgb_str = node_colors.get(code, nan_color)
            rgb_vals = rgb_str.replace("rgb(", "").replace(")", "").split(",")
            r, g, b = (int(v.strip()) for v in rgb_vals)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_colors.append("white" if luminance < 140 else "black")
        textfont = dict(size=font_sizes, color=text_colors, family="Arial")
    else:
        marker_colors = nan_color
        textfont = dict(size=font_sizes, color="white", family="Arial")
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=sizes,
            opacity=1,
            symbol="hexagon",
            color=marker_colors,
            line=dict(width=2, color="rgb(30, 30, 30)")
        ),
        text=node_codes,
        textposition="middle center",
        textfont=textfont,
        hoverinfo="none",
        showlegend=False,
        name=""
    )
    
    return node_trace, bounds


def _configure_layout(
    fig: go.Figure,
    title: str,
    plot_bounds:Tuple[float,float,float,float],
    width: int = 1000,
    height: int = 800,
    square_plot: bool = False,
    bound_pad:float= 0.07,
    ) -> None:
    """
        Configure Plotly figure layout (axes, background, margins, etc.).
        
        Parameters
        ----------
        fig : go.Figure
            Figure to configure (modified in-place).
        title : str
            Plot title.
        width, height : int, optional
            Figure dimensions in pixels (default: 1000x800).
        square_plot : bool, optional
            Lock plot area to square aspect ratio (1:1 axes) (default: False).
    """
    # Configure axes with optional square aspect ratio
    minx, maxx, miny, maxy = plot_bounds
    padx = (maxx - minx) * bound_pad
    pady = (maxy - miny) * bound_pad
    xaxis_config = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[minx-padx, maxx+padx]
    )
    yaxis_config = dict(
        showgrid=False,
        zeroline=False,
        range=[miny-pady, maxy+pady],
        showticklabels=False
    )
    
    # Apply square plot locking if requested
    if square_plot:
        xaxis_config["scaleanchor"] = "y"
        xaxis_config["scaleratio"] = 1
        yaxis_config["scaleanchor"] = "x"
        yaxis_config["scaleratio"] = 1
    
    fig.update_layout(
        # title=dict(text=title, font=dict(size=18), 
        #            xanchor="center", x=0.5, y=0.97),
        showlegend=False,
        hovermode="closest",
        # margin=dict(b=20, l=20, r=20, t=20),  
        margin=dict(b=5, l=5, r=5, t=5),  
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        plot_bgcolor="rgb(240, 240, 240)",
        width=width,
        height=height,
    )


def _build_basemap_trace(
    basemap: gpd.GeoSeries,
    basecolor: str="gainsboro",
    edgecolor: str = "darkgray",
    linewidth: str= 0.5
    ) -> go.Scatter:
    """
        Build Plotly scatter trace for vector basemap layer.
        
        Parameters
        ----------
        basemap : gpd.GeoSeries
            GeoSeries of Polygon/MultiPolygon geometries. Must be in target CRS.
        basemap_style : dict
            Style dict with keys: "basecolor", "edgecolor", "linewidth".
        
        Returns
        -------
        go.Scatter
            Trace with fill="toself", no legend, no hover.
    """
    x_coords, y_coords = [], []
    for geom in basemap:
        # geom = basemap[col]
        geoms = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for poly in geoms:
            x, y = poly.exterior.xy
            x_coords.extend(list(x) + [None])
            y_coords.extend(list(y) + [None])
    
    return go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="lines",
        fill="toself",
        fillcolor=basecolor,
        line=dict(color=edgecolor, width=linewidth),
        hoverinfo="none",
        showlegend=False,
        name=""
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Function
# ─────────────────────────────────────────────────────────────────────────────


def plot_country_network(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str,
    pos_dict: Dict[str, Tuple[float, float]],
    title: str = "European Country Border Network",
    basemap: Optional[gpd.GeoSeries] = None,
    basemap_style: Dict[str, Any] = {},
    edge_thickness: float = 4,
    node_size: float = 40,
    node_text_scale: float = 0.3,
    # node_size_dict: Optional[Dict[str, float]] = None,
    width: int = 1000,
    height: int = 800,
    square_plot: bool = False,
    # arrowhead_radius: float = 0.08,
    edge_nan_color: str = "rgb(180, 180, 180)",
    node_nan_color: str = "rgb(255, 255, 255)",
    cmap: Optional[mcolors.Colormap] = None,
    norm: Optional[mcolors.Normalize] = None,
    colorbar_title: str = "Weight",
    colorbar_ypos: float = 0.98,
    colorbar_xpos: float= 0.02,
    # arrowsize: float = 30,,
    node_data: Optional[pd.DataFrame] = None,
    # node_plot_col: Optional[str] = None,
    node_cmap: Optional[mcolors.Colormap] = None,
    node_norm: Optional[mcolors.Normalize] = None,
    node_colorbar_title: str = "Node Value",
    node_colorbar_ypos: Optional[float] = 0.83,
    colorbar_tickvals: Optional[List[float]] = None,
    colorbar_ticktext: Optional[List[str]] = None,
    node_colorbar_tickvals: Optional[List[float]] = None,
    node_colorbar_ticktext: Optional[List[str]] = None,
    show_nan_legend: bool = False
) -> go.Figure:
    """
        Plot a directed graph of country border networks using Plotly.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing edge data.
        source_col : str
            Column name for edge source (country code).
        target_col : str
            Column name for edge target (country code).
        weight_col : str
            Column name for edge weight (border metric).
        node_positions : dict, pd.DataFrame, or gpd.GeoSeries
            Node coordinates. Formats:
            - dict: {country_code: (x, y)}
            - pd.DataFrame: must have columns 'code', 'x', 'y'
            - gpd.GeoSeries: indexed by country code, with Point geometries in EPSG:4326.
                Coordinates extracted as (lon, lat) for plotting.
        basemap : gpd.GeoSeries, optional
            Vector geometries (Polygons/MultiPolygons) for basemap layer. Must share
            same CRS as specified in `crs` parameter. If provided, `crs` must be non-None.
            Default: None (no basemap).
        crs : str, optional
            Coordinate Reference System as string (e.g., "EPSG:4326"). Required if `basemap`
            is not None. Used to validate CRS consistency. Default: None.
        basemap_style : dict, optional
            Style dict with keys "basecolor", "edgecolor", "linewidth" (all strings).
            If None and basemap is not None, defaults to {"basecolor": "gainsboro", 
            "edgecolor": "darkgray", "linewidth": "0.5"}. Default: None.
        title : str, optional
            Plot title (default: "European Country Border Network").
        edge_thickness : float, optional
            Edge line width (default: 4).
        node_size : float, optional
            Node marker size in pixels (default: 40).
        node_text_scale : float, optional
            Node text font size as proportion of node_size (default: 0.3).
            E.g., node_size=40 with node_text_scale=0.3 → font size ~12px.
        apply_kamada_kawai : bool, optional
            Apply Kamada-Kawai layout refinement using existing coordinates as warm-start.
            Refines node positions iteratively while respecting initial geography (default: False).
        node_size_dict : dict, optional
            Per-node size overrides as {node_code: size_value}. If provided, node sizes
            are taken from this dict; missing nodes use node_size default (default: None).
        width : int, optional
            Figure width in pixels (default: 1000).
        height : int, optional
            Figure height in pixels (default: 800).
        square_plot : bool, optional
            Lock plot area to square aspect ratio (1:1 axes) (default: False).
        arrowhead_radius : float, optional
            Fixed distance from target node to place arrowhead in plot coordinates (default: 0.08).
            Ensures consistent arrowhead visibility regardless of edge length.
        nan_color : str, optional
            RGB color string for edges with NaN weights (default: "rgb(255, 255, 255)" white).
        cmap : matplotlib.colors.Colormap, optional
            Matplotlib colormap for arrowhead coloring by weight. If None, arrowheads are gray.
        norm : matplotlib.colors.Normalize, optional
            Custom normalizer for weight → colormap mapping. If None, auto min-max normalization.
        colorbar_title : str, optional
            Label for colorbar legend (default: "Weight").
        node_data : pd.DataFrame, optional
            DataFrame indexed by the same keys as `pos_dict`, with a single metric
            column to plot (default: None).
        node_plot_col : str, optional
            Column name of the metric to plot for nodes (default: None).
        node_cmap : matplotlib.colors.Colormap, optional
            Colormap for node values; if None, uses viridis (default: None).
        node_norm : matplotlib.colors.Normalize, optional
            Normalizer for node values; if None, auto min-max normalization (default: None).
        node_colorbar_title : str, optional
            Label for the node colorbar (default: "Node Value").
        colorbar_tickvals : list of float, optional
            Explicit tick positions for the edge colorbar (default: None).
        colorbar_ticktext : list of str, optional
            Explicit tick labels for the edge colorbar (default: None).
        node_colorbar_tickvals : list of float, optional
            Explicit tick positions for the node colorbar (default: None).
        node_colorbar_ticktext : list of str, optional
            Explicit tick labels for the node colorbar (default: None).
        show_nan_legend : bool, optional
            If True, add a legend entry for NaN (grey) nodes (default: False).
        save : str, optional
            Path to save figure as HTML. If empty (default), returns fig without saving.

        Returns
        -------
        go.Figure
            Plotly figure object ready for display.

        Raises
        ------
        ValueError
            If required columns are missing, node positions don't cover all nodes,
            or GeoSeries is not in EPSG:4326.
    """
    # Phase 1: Data Preparation
    edges = _prepare_edges(data, source_col, target_col, weight_col)
    
    all_nodes = set(edges["source"]) | set(edges["target"])
    
    # Validate all nodes have positions
    missing_nodes = all_nodes - set(pos_dict.keys())
    if missing_nodes:
        raise ValueError(f"Missing positions for nodes: {missing_nodes}")
    
    # Phase 2: Build Figure
    fig = go.Figure()
    
    # Add basemap trace (renders first, at bottom z-index)
    if basemap is not None:
        basemap_trace = _build_basemap_trace(basemap, **basemap_style)
        fig.add_trace(basemap_trace)
    
    # Add edge traces
    edge_traces, colorscale, edge_separator_traces = _build_edges(edges, 
        pos_dict, cmap, norm, node_standoff=node_size/1.5,
        offset_deg=edge_thickness, colorbar_title=colorbar_title,
        nan_color=edge_nan_color)
    for etrace, septrace in zip(edge_traces, edge_separator_traces):
        # fig.add_annotation(trace) # TODO: misnomer
        fig.add_trace(etrace)
        fig.add_trace(septrace)
    colorbar_trace = _build_colorbar_trace(colorscale, norm, colorbar_title,
                                           y=colorbar_ypos, x=colorbar_xpos,
                                           tickvals=colorbar_tickvals,
                                           ticktext=colorbar_ticktext)
    fig.add_trace(colorbar_trace)
    
    # Optional node colouring
    node_colors = None
    node_colorbar_trace = None
    if node_data is not None:
        missing_idx = set(node_data.index) - set(pos_dict.keys())
        if missing_idx:
            raise ValueError(f"node_data index contains unknown nodes: {missing_idx}")
        if node_norm is None:
            node_norm = mcolors.Normalize(
                vmin=node_data[weight_col].min(),
                vmax=node_data[weight_col].max(),
            )
        if node_cmap is None:
            node_cmap = plt.get_cmap("viridis")
        node_colors = {
            code: node_nan_color if pd.isna(val) else 
                f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            for code, val in node_data[weight_col].items()
            for r, g, b, _ in [node_cmap(node_norm(val))]
        }
        node_colorscale, _ = _make_colorscale(node_norm, node_cmap)
        node_colorbar_trace = _build_colorbar_trace(
            node_colorscale, node_norm, node_colorbar_title,
            x=colorbar_xpos, y=node_colorbar_ypos,
            tickvals=node_colorbar_tickvals,
            ticktext=node_colorbar_ticktext
        )
    # Add node trace
    node_trace, plot_bounds = _build_node_trace(all_nodes, pos_dict, node_size, node_text_scale, node_colors=node_colors, nan_color=node_nan_color)
    fig.add_trace(node_trace)
    # if node_colorbar_trace is not None:
        # fig.add_trace(node_colorbar_trace)
    if show_nan_legend:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=node_nan_color, 
                        line=dict(width=2, color="DarkSlateGrey")),
            name="no domestic data",
            showlegend=True, hoverinfo="none",
        ))
    
    # Phase 3: Configure 
    # print(plot_bounds)
    _configure_layout(fig, title, plot_bounds, width, height, square_plot)
    if show_nan_legend:
        fig.update_layout(showlegend=True, legend=dict(x=1.0, y=0.0, xanchor="right", yanchor="bottom", bgcolor="rgba(255,255,255,0.0)"))

    return fig


# 1. Define your 4 anchor colors: [min_val, zero_anchor, positive_mid, max_val]
colors = [
    "#a7a7a7",  # Neutral (maps to 0.0)
    "#db6b0f",  # orange
    "#7a0808",  # dark red
]
symmetric_fill_colour = [
    "#1055CD",  # dark blue
    "#3caec0",  # light blue
]

# node colormap: white -> dark blue
node_colors = ["#ffffff", "#00499c"]
node_greens = ["#fcfcfc", "#4D966E", "#076635"]


def add_cmap_and_normalise(dict_in,
                           metric_df: pd.DataFrame,
                           prefix: str,
                           base_cmap=colors,
                           symmetric_fill_color=symmetric_fill_colour,
                           ):
    """Populate dict_in subdicts with prefixed cmap/norm (+ tick) entries.

    Reads `{prefix}_clip` (optional sequence of tick steps; first/last are
    the clip bounds) and `asc` from each subdict; writes
    `{prefix}_color`, `{prefix}_norm`, and — when `{prefix}_clip` is
    present — `{prefix}_tickvals` / `{prefix}_ticktext`.
    """
    for col, subdict in dict_in.items():
        values = metric_df[col]
        vmin, vmax = values.min(), values.max()
        vcenter = subdict.get(f"{prefix}_center", None)
        is_asc = subdict.get(f"{prefix}_asc", True)
        ticksteps = subdict.get(f"{prefix}_clip", None)
        # vcenter = subdict.get(f"{prefix}_vcenter", None)

        # build color list (padding for fake symmetry, optional reversal)
        if symmetric_fill_color is not None:
            if isinstance(symmetric_fill_color, str):
                cmap = (len(base_cmap)-1) * [symmetric_fill_color] \
                    + base_cmap
            elif isinstance(symmetric_fill_color, typing.Sequence):
                cmap = symmetric_fill_color + base_cmap
            else:
                cmap = base_cmap
        else:
            cmap = base_cmap
        subdict[f"{prefix}_color"] = cmap if is_asc else list(reversed(cmap))

        # norm: clipped (ticksteps) or plain min-max
        if ticksteps is not None:
            lo, hi = ticksteps[0], ticksteps[-1]
            if lo < 0 and hi > 0:
                subdict[f"{prefix}_norm"] = mcolors.TwoSlopeNorm(
                    vcenter=0, vmin=lo, vmax=hi, )
            elif vcenter is not None:
                subdict[f"{prefix}_norm"] = mcolors.TwoSlopeNorm(
                    vcenter=vcenter, vmin=lo, vmax=hi, )
            else:
                subdict[f"{prefix}_norm"] = mcolors.Normalize(
                    vmin=lo, vmax=hi)
            subdict[f"{prefix}_tickvals"] = list(ticksteps)
            txt_vmin = f"<{lo:g}" if lo > vmin else f"{lo:g}"
            txt_vmax = f">{hi:g}" if hi < vmax else f"{hi:g}"
            subdict[f"{prefix}_ticktext"] = [
                txt_vmin, *[str(t) for t in ticksteps[1:-1]], txt_vmax]
        elif col in {"n_od", } or vmax < 0:
            subdict[f"{prefix}_norm"] = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            subdict[f"{prefix}_norm"] = mcolors.TwoSlopeNorm(
                vcenter=0, vmax=vmax, vmin=vmin)


def make_list_cmap_mpl(cmap_name: str, steps: int, lbound=0, ubound=1):
    return [colormaps[cmap_name](i) for i in np.linspace(lbound, ubound, steps)]