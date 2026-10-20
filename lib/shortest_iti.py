import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from lib.geoutils import great_circle_distance_km

logger = logging.getLogger(__name__)


GTT_TRANS_WEIGHT = 1.7
GTT_PER_TRANSFER = 10

EXPORT_FILENAMES = {
    "features": "shortest-itineraries.parquet",
    "enriched": "itineraries-enriched.parquet",
}


@dataclass(frozen=True)
class Config:
    data_dir: Path
    out_dir: Path
    adjacency_path: Path


@dataclass(frozen=True)
class ShortestItiResult:
    enriched: pd.DataFrame
    shortest: pd.DataFrame
    features: pd.DataFrame


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.Series]:
    """Read itineraries, stations, FUAs, and adjacency, indexed by their keys."""
    itins = pd.read_parquet(cfg.data_dir / "sherman-rail-path-times.parquet")
    assert itins.index.is_unique, "itinerary index (iti_id) must be unique"
    stations = gpd.read_parquet(cfg.data_dir / "ic-stations2.parquet").set_index("stn_id", drop=False)
    assert stations.index.is_unique, "stn_id must be unique"
    fuas = gpd.read_parquet(cfg.data_dir / "fuas.parquet").set_index("fua_id", drop=False)
    assert fuas.index.is_unique, "fua_id must be unique"
    adjacency = pd.read_csv(cfg.adjacency_path, index_col=0).stack().rename("adjacency")
    return itins, stations, fuas, adjacency


def _add_station_fua(df: pd.DataFrame, stations: gpd.GeoDataFrame) -> pd.DataFrame:
    """Map source and target station IDs to their FUA IDs."""
    return df.assign(
        fua_src=df["src"].map(stations["fua_id"]),
        fua_trg=df["trg"].map(stations["fua_id"]),
    )

def _add_od_distance(df: pd.DataFrame, stations: gpd.GeoDataFrame) -> pd.DataFrame:
    """Add straight-line station-to-station distance (km) and speed (km/h)."""
    dist = np.asarray(
        great_circle_distance_km(
            stations.loc[df["src"], "geometry"],
            stations.loc[df["trg"], "geometry"],
        )
    )
    return df.assign(
        dist_od_stns=dist,
        speed_od_stns=dist / (df["time"] / 60),
    )

def _add_fua_icc(df: pd.DataFrame, fuas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Map source and target FUA IDs to their country codes."""
    return df.assign(
        icc_src=df["fua_src"].map(fuas["icc"]),
        icc_trg=df["fua_trg"].map(fuas["icc"]),
    )

def _add_adjacency(df: pd.DataFrame, adjacency: pd.Series) -> pd.DataFrame:
    """Join country-pair adjacency onto each itinerary."""
    return df.join(adjacency, on=["icc_src", "icc_trg"])

def enrich_itineraries(
    itins: pd.DataFrame,
    stations: gpd.GeoDataFrame,
    fuas: gpd.GeoDataFrame,
    adjacency: pd.Series,
) -> pd.DataFrame:
    """Join station FUAs, OD distance, country codes, and adjacency onto itineraries. GTT is computed separately by compute_gtt."""
    return (
        itins.pipe(_add_station_fua, stations)
        .pipe(_add_od_distance, stations)
        .pipe(_add_fua_icc, fuas)
        .pipe(_add_adjacency, adjacency)
    )


def compute_gtt(df: pd.DataFrame, *, out_col: str) -> pd.DataFrame:
    """Write recomputed GTT to out_col, overwriting any existing column of that name. The R5 station-processing term is not included."""
    return df.assign(
        **{
            out_col: df["inveh_time"]
            + GTT_TRANS_WEIGHT * df["trans_time"]
            + GTT_PER_TRANSFER * df["n_trans"]
        }
    )


def select_shortest(df: pd.DataFrame, *, gtt_col: str) -> pd.DataFrame:
    """Select one itinerary per FUA OD with minimum gtt_col; ties go to lowest n_trans, then time, then iti_id."""
    valid = df.dropna(subset=[gtt_col])
    logger.info("select_shortest: dropped %d rows with NaN %s", len(df) - len(valid), gtt_col)
    ordered = valid.assign(_iti_id=valid.index).sort_values(["n_trans", "time", "_iti_id"])
    # idxmin keeps the first occurrence, so the sort above fixes tie-breaks
    best = ordered.groupby(["fua_src", "fua_trg"])[gtt_col].idxmin()
    return df.loc[best.to_numpy()]


def add_features(shortest: pd.DataFrame, fuas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Add population, circuity, speed, transfer-time, and OD population features to the selected itineraries."""
    popu_src = shortest["fua_src"].map(fuas["popu"]) / 1e6
    popu_trg = shortest["fua_trg"].map(fuas["popu"]) / 1e6
    features = shortest.assign(
        circuity_osm_od=shortest["dist"] / shortest["dist_od_stns"],
        circuity_osm_leg=shortest["dist"] / shortest["gcd"],
        speed_inveh=shortest["dist_od_stns"] / shortest["inveh_time"] * 60,
        speed_net=shortest["dist"] / shortest["time"] * 60,
        trans_time_per=shortest["trans_time"].where(
            shortest["trans_time"] == 0,
            shortest["trans_time"] / shortest["n_trans"],
        ),
        popu_src=popu_src,
        popu_trg=popu_trg,
        popu_od_sum=popu_src + popu_trg,
        # popu_od_sqsum=popu_src.pow(0.5) + popu_trg.pow(0.5),
        # popu_od_logsum=np.log10(popu_src) + np.log10(popu_trg),
        popu_od_prod=popu_src * popu_trg,
        # popu_od_eucmean=(popu_src.pow(2) + popu_trg.pow(2)).pow(0.5),
    )
    return features.assign(popu_od_geomean=features["popu_od_prod"].pow(0.5))


def build_shortest_itineraries(cfg: Config, *, gtt_out_col: str) -> ShortestItiResult:
    """Run load, enrich, GTT, shortest selection, and feature stages in order. Setting gtt_out_col to "gtt" overwrites the R5 gtt column."""
    itins, stations, fuas, adjacency = load_inputs(cfg)
    enriched = enrich_itineraries(itins, stations, fuas, adjacency)
    enriched = compute_gtt(enriched, out_col=gtt_out_col)
    shortest = select_shortest(enriched, gtt_col=gtt_out_col)
    features = add_features(shortest, fuas)
    return ShortestItiResult(enriched=enriched, shortest=shortest, features=features)


def export_results(result: ShortestItiResult, cfg: Config) -> dict[str, Path]:
    """Write the features and enriched frames as parquet to cfg.out_dir and return their paths by frame name."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for field, filename in EXPORT_FILENAMES.items():
        out_path = cfg.out_dir / filename
        getattr(result, field).to_parquet(out_path)
        logger.info("export_results: wrote %s", out_path)
        paths[field] = out_path
    return paths
