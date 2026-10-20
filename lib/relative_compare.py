import numpy as np
import pandas as pd

def _weighted_mean_nanzero(values, weights):
    """
    Per-column weighted mean with NaN rows excluded (masked re-normalization),
    NOT treated as zero. For each metric column, rows where the column value is
    NaN are masked out and the weighted mean is computed over the remaining rows
    using the weights restricted to those rows. If all values in a column are
    NaN (or the total weight over non-NaN rows is 0), the result is NaN for
    that column.

    Args:
        values: 2D np.ndarray [n_rows, n_metrics] (or DataFrame — converted
            with .values).
        weights: 1D array-like [n_rows].

    Returns:
        1D np.ndarray of length n_metrics.
    """
    if hasattr(values, "values"):
        values = values.values
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n_metrics = values.shape[1]
    out = np.full(n_metrics, np.nan)
    for j in range(n_metrics):
        col = values[:, j]
        mask = ~np.isnan(col)
        if not mask.any():
            continue
        w = weights[mask]
        total_w = w.sum()
        if total_w == 0:
            continue
        out[j] = np.average(col[mask], weights=w)
    return out

def build_domestic_metric_arrays(domestic_by_country_sorted, metric_cols):
    """
    Extract metric arrays from domestic ODs per country.

    Returns:
        dict[country: dict[metric: np.ndarray]]
        Also computes dom_pop_weights per country as side effect.
    """
    dom_metric_arrays = {}
    dom_pop_weights_by_country = {}

    for country, dom_df in domestic_by_country_sorted.items():
        dom_metric_arrays[country] = {}
        for metric in metric_cols:
            dom_metric_arrays[country][metric] = dom_df[metric].values
        dom_pop_weights_by_country[country] = dom_df["popu_od_prod"].values

    return dom_metric_arrays, dom_pop_weights_by_country


def compute_iod_deltas_dual_reference(
    neighbour_it2,
    domestic_by_country_sorted,
    dom_metric_arrays,
    dom_pop_weights_by_country,
    metric_cols
):
    """
    For each international OD pair, compute deltas vs all domestic ODs.
    Returns iod-level deltas for both endpoint countries as reference.
    """
    iod_deltas_dict = {}

    # Group by country pair
    for pair, pair_df in neighbour_it2.groupby(["icc1", "icc2"], observed=True):
        iod_deltas_dict[pair] = {}

        # Extract iod metrics and metadata
        iod_metrics = pair_df[metric_cols].values  # [n_iods, n_metrics]
        iod_indices = pair_df.index  # Preserve multiindex (fua_src, fua_trg)
        iod_pop_weights = pair_df["popu_od_prod"].values  # [n_iods]

        # For each reference country in the pair
        for ref_country in pair:
            other_country = pair[1] if ref_country == pair[0] else pair[0]

            # If reference country not in domestic set, emit NaN deltas (kept for one-directional plotting)
            if ref_country not in domestic_by_country_sorted:
                iod_deltas_dict[pair][ref_country] = {
                    "deltas": np.full((len(iod_metrics), len(metric_cols)), np.nan),
                    "indices": iod_indices,
                    "iod_pop_weights": iod_pop_weights,
                    "other_country": other_country,
                    "missing_reference": True,
                }
                continue

            # Retrieve domestic reference arrays
            dom_metrics = np.array([
                dom_metric_arrays[ref_country][m] for m in metric_cols
            ]).T  # [n_doms, n_metrics]
            # [n_doms]
            dom_pop_weights = dom_pop_weights_by_country[ref_country]

            # Numpy broadcast: compute element-wise deltas
            # iod_metrics: [n_iods, n_metrics]
            # dom_metrics: [n_doms, n_metrics]
            # result: [n_iods, n_doms, n_metrics]
            deltas = iod_metrics[:, None, :] - dom_metrics[None, :, :]

            # Pop-weight aggregate across domestic dimension (axis=1)
            # Result: [n_iods, n_metrics]
            weighted_delta = np.average(
                deltas, weights=dom_pop_weights, axis=1
            )

            # Store result with metadata
            iod_deltas_dict[pair][ref_country] = {
                "deltas": weighted_delta,
                "indices": iod_indices,
                "iod_pop_weights": iod_pop_weights,
                "other_country": other_country,
                "missing_reference": False,
            }

    return iod_deltas_dict



def aggregate_iod_deltas_to_pairs(iod_deltas_dict, metric_cols):
    """
    Aggregate iod-level deltas to country-pair level via pop-weighting.
    Returns long-format DataFrame: (pair, reference_country, other_country, + metrics)
    """
    df_iod_level_list = []

    # For each pair and reference country
    for pair, ref_dict in iod_deltas_dict.items():
        for ref_country, delta_data in ref_dict.items():
            weighted_delta = delta_data["deltas"]
            iod_indices = delta_data["indices"]
            iod_pop_weights = delta_data["iod_pop_weights"]
            other_country = delta_data["other_country"]
            missing_reference = delta_data["missing_reference"]

            # Build temporary DataFrame with iod-level data
            df_temp = pd.DataFrame(
                weighted_delta,
                index=iod_indices,
                columns=metric_cols
            ).assign(
                pair="-".join(pair),
                reference_country=ref_country,
                other_country=other_country,
                pop_weight=iod_pop_weights,
                missing_reference=missing_reference
            )

            df_iod_level_list.append(df_temp)

    # Concatenate all iod-level rows
    df_iod_level = pd.concat(df_iod_level_list, ignore_index=False)

    # Aggregate to pair level via pop-weighted mean per reference country
    df_pair_agg = _aggregate_pair_groups(df_iod_level, metric_cols)

    return df_pair_agg, df_iod_level


def _aggregate_pair_groups(df_long, metric_cols, weight_col="pop_weight"):
    """
    Aggregate a long-format DataFrame to (pair, reference_country, other_country)
    groups via weighted mean per metric. NaN rows are excluded per column
    (masked re-normalization), not treated as zero.

    Returns DataFrame with columns
    [pair, reference_country, other_country, *metric_cols].
    """
    pair_agg_list = []
    for (pair, ref_country, other_country), group in df_long.groupby(
        ["pair", "reference_country", "other_country"], observed=True
    ):
        row = {
            "pair": pair,
            "reference_country": ref_country,
            "other_country": other_country,
        }
        weighted_means = _weighted_mean_nanzero(
            group[metric_cols], group[weight_col]
        )
        for metric, val in zip(metric_cols, weighted_means):
            row[metric] = val
        pair_agg_list.append(row)

    return pd.DataFrame(pair_agg_list)


def compute_intl_pair_means(neighbour_it2, metric_cols):
    """
    Single-stage aggregation of raw international metrics (no domestic
    reference), pop-weighted, with mirrored 2-rows-per-pair output for
    join/subtract compatibility with the dual-stage df_pair_agg.

    Groups neighbour_it2 by ["icc1", "icc2"]; for each pair computes the
    pop-weighted mean of raw metric values (weights = popu_od_prod; NaNs
    masked, not zero-filled) and emits two mirrored rows:
    (reference_country=icc1, other_country=icc2) and
    (reference_country=icc2, other_country=icc1), both carrying identical
    metric values (symmetry is always expected for this case).

    Returns DataFrame with columns
    [pair, reference_country, other_country, *metric_cols, n_iods,
    pop_coverage, pop_var] — same semantics/schema as df_pair_agg after
    add_iod_diagnostics.
    """
    rows = []
    for (icc1, icc2), group in neighbour_it2.groupby(
        ["icc1", "icc2"], observed=True
    ):
        weighted_means = _weighted_mean_nanzero(
            group[metric_cols], group["popu_od_prod"]
        )
        base = {
            "pair": f"{icc1}-{icc2}",
            "n_iods": len(group),
            "pop_coverage": group["popu_od_prod"].sum(),
            "pop_var": group["popu_od_prod"].var(),
        }
        for metric, val in zip(metric_cols, weighted_means):
            base[metric] = val
        rows.append({**base, "reference_country": icc1, "other_country": icc2})
        rows.append({**base, "reference_country": icc2, "other_country": icc1})

    return pd.DataFrame(
        rows,
        columns=["pair", "reference_country", "other_country", *metric_cols,
                 "n_iods", "pop_coverage", "pop_var"]
    )


def add_iod_diagnostics(df_pair_agg, df_iod_level):
    """
    Annotate pair aggregates with iod-level diagnostic columns.
    """
    diagnostics_list = []
    for (pair, ref_country, other_country), group in df_iod_level.groupby(
        ["pair", "reference_country", "other_country"], observed=True
    ):
        if group["missing_reference"].all():
            diagnostics_list.append({
                "pair": pair,
                "reference_country": ref_country,
                "other_country": other_country,
                "n_iods": np.nan,
                "pop_coverage": np.nan,
                "pop_var": np.nan,
            })
        else:
            diagnostics_list.append({
                "pair": pair,
                "reference_country": ref_country,
                "other_country": other_country,
                "n_iods": len(group),
                "pop_coverage": group["pop_weight"].sum(),
                "pop_var": group["pop_weight"].var(),
            })

    df_diag = pd.DataFrame(diagnostics_list)

    # Merge diagnostics into pair aggregates
    df_pair_agg = df_pair_agg.merge(
        df_diag,
        on=["pair", "reference_country", "other_country"],
        how="left"
    )

    return df_pair_agg


def compute_domestic_popweighted_means(domestic_by_country_sorted, neighbour_it2, metric_cols):
    """
    Per-country population-weighted mean of each metric over the domestic set,
    weighted by popu_od_prod. Covers all countries present in the international
    dataset (icc_src ∪ icc_trg); countries without a domestic set get NaN rows.
    Returns DataFrame indexed by country code, columns = metric_cols (flat).
    """
    intl_countries = set(neighbour_it2["icc_src"]).union(
        neighbour_it2["icc_trg"])
    rows = {}
    for country in sorted(intl_countries):
        dom_df = domestic_by_country_sorted.get(country)
        if dom_df is None or dom_df.empty:
            rows[country] = {m: np.nan for m in metric_cols}
        else:
            rows[country] = dict(zip(metric_cols, np.average(
                dom_df[metric_cols], weights=dom_df["popu_od_prod"], axis=0)))
    return pd.DataFrame.from_dict(rows, orient="index")


def compute_domestic_reference_diagnostics(domestic_by_country_sorted):
    """
    Compute domestic reference bias metrics per country (inspection only).
    Flags high variance in domestic population distribution.
    """
    diag_list = []
    for country, dom_df in domestic_by_country_sorted.items():
        pop_weights = dom_df["popu_od_prod"].values
        diag_list.append({
            "country": country,
            "pop_coverage": pop_weights.sum(),
            "pop_var": pop_weights.var(),
        })

    df_domestic_diag = pd.DataFrame(diag_list).set_index("country")

    return df_domestic_diag
