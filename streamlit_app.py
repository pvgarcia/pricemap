from io import BytesIO
from pathlib import Path
from itertools import count

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib import font_manager as fm
from shapely import make_valid
from pyproj import Transformer
from cartopy.io import shapereader as shp  # cartopy for borders


# ---------------- Page setup ----------------
st.set_page_config(page_title="ISO Price Map", layout="wide")
st.title("ISO Price Map")


# ---------------- Constants ----------------
TARGET_CRS = "EPSG:3857"

ISO_COL   = "ISO/RTO"
HUB_COL   = "Settlement Hub"
PRICE_COL = "Price ($/MWh)"
ATC_COL   = "ATC Price ($/MWh)"
DEV_COL   = "Developer"
TECH_COL  = "Technology"

MASTER_GEO = Path("us_canada_rto_na.geojson")

REGION_COLORS = {
    "CAISO":  "#F68220",
    "ERCOT":  "#853F05",
    "MISO":   "#0EC477",
    "PJM":    "#5B6670",
    "ISO-NE": "#FF0404",
    "NYISO":  "#578EB1",
    "SPP":    "#00B0F0",
    "AESO":   "#C00000",
    "IESO":   "#00519B",
}

BORDER_COLOR = "#75838FFF"
COUNTRY_LW   = 0.10
STATE_LW     = 0.10

OFFSET_WHEEL = [
    (0, 12), (12, 0), (-12, 0), (0, -12),
    (10, 10), (-10, 10), (-10, -10), (10, -10),
    (16, 6), (-16, 6),
]

ISO_CONFIGS = [
    {"iso": "CAISO", "hub_proxy": {
        "NP15": (-121.4944, 38.5816),
        "SP15": (-117.8880, 34.0522),
    }},
    {"iso": "ERCOT", "hub_proxy": {
        "HB_NORTH": (-97.3308, 32.7555),
        "HB_HOUSTON": (-95.36327, 29.76328),
        "HB_SOUTH": (-96.9978, 28.8053),
        "HB_WEST": (-102.3677, 31.8457),
    }},
    {"iso": "MISO", "hub_proxy": {
        "INDIANA.HUB": (-86.1581, 39.7684),
        "ILLINOIS.HUB": (-89.6500, 39.7817),
        "MICHIGAN.HUB": (-84.5555, 42.7325),
        "MINN.HUB": (-93.2650, 44.9778),
        "TEXAS.HUB": (-94.1266, 30.0802),
        "ARKANSAS.HUB": (-92.2896, 34.7465),
        "LOUISIANA.HUB": (-91.1871, 30.4515),
    }},
    {"iso": "PJM", "hub_proxy": {
        "AD Hub": (-84.1917, 39.7589),
        "NI Hub": (-87.6298, 41.8781),
        "PEPCO": (-77.0369, 38.9072),
        "BGE": (-76.6122, 39.2904),
        "PSEG": (-74.1724, 40.7357),
        "JCPL": (-74.4830, 40.7968),
        "METED": (-75.9269, 40.3356),
        "PPL": (-75.4902, 40.6023),
        "COMED": (-87.6298, 41.8781),
        "DEOK": (-84.5120, 39.1031),
        "ATSI": (-81.6944, 41.4993),
        "DOMINION HUB": (-77.4360, 37.5407),
        "PECO": (-75.1652, 39.9526),
        "PENELEC": (-78.3947, 40.5187),
        "RECO": (-74.0324, 40.9263),
        "AEP-DAYTON HUB": (-84.1917, 39.7589),
        "APS": (-79.9559, 39.6295),
        "DPL": (-75.5467, 39.7447),
        "DUQ": (-80.0000, 40.4406),
        "EASTERN HUB": (-75.1652, 39.9526),
        "WESTERN HUB": (-80.0000, 40.4406),
        "AEP": (-82.9988, 39.9612),
    }},
    {"iso": "ISO-NE", "hub_proxy": {
        "Mass Hub": (-71.0589, 42.3601),
        ".Z.CONNECTICUT": (-72.6851, 41.7637),
        ".Z.MAINE": (-69.7795, 44.3106),
        ".Z.NEMASSBOST": (-71.0120, 42.3954),
        ".Z.NEWHAMPSHIRE": (-71.4548, 43.2070),
        ".Z.RHODEISLAND": (-71.4128, 41.8236),
        ".Z.SEMASS": (-70.9167, 41.6362),
        ".Z.VERMONT": (-72.5754, 44.2601),
        ".Z.WCMASS": (-72.5898, 42.1015),
    }},
    {"iso": "NYISO", "hub_proxy": {
        "West - A": (-78.8784, 42.8864),
        "Genesee - B": (-77.6156, 43.1566),
        "Central - C": (-76.1474, 43.0481),
        "North - D": (-75.9108, 44.6995),
        "Mohawk Valley - E": (-75.2327, 43.1009),
        "Capital - F": (-73.7562, 42.6526),
        "Hudson Valley - G": (-73.9235, 41.7004),
        "N.Y.C. - J": (-73.9857, 40.7484),
        "Long Island - K": (-73.1960, 40.7891),
    }},
    {"iso": "AESO", "hub_proxy": {
        "Alberta": (-113.4938, 53.5461),
    }},
    {"iso": "SPP", "hub_proxy": {
        "SPPNORTH_HUB": (-103.9208, 46.2950),
        "SPPSOUTH_HUB": (-94.7040, 37.4109),
    }},
]

DEFAULT_OFFSETS = {
    "HB_HOUSTON": (-5, -20),
    "HB_SOUTH": (-70, -10),
    "HB_WEST": (-80, -10),
    "MICHIGAN.HUB": (10, 60),
    "INDIANA.HUB": (-25, -40),
    "SPPSOUTH_HUB": (-130, 9),
    "SPPNORTH_HUB": (-90, 9),
    "AEP": (0, 50),
    "ILLINOIS.HUB": (-30, 0),
    "WESTERN HUB": (50, -20),
    "NP15": (-60, 5),
    "SP15": (-60, 6),
    ".Z.MAINE": (20, 0),
    "LOUISIANA.HUB": (24, -20),
    "ARKANSAS.HUB": (35, -20),
    "AEP-DAYTON HUB": (30, -71),
    "Alberta": (50, 7),
}


# ---------------- Styling for chips ----------------
st.markdown(
    """
    <style>
    label[data-testid="stWidgetLabel"] > div {
        font-weight: 600;
        color: #00519B;
    }
    div[data-baseweb="tag"] {
        background-color: #E6F0FA !important;
        color: #00519B !important;
        border-radius: 12px !important;
        padding: 2px 8px !important;
        border: 1px solid #B7D3F6 !important;
    }
    div[data-baseweb="tag"] svg { fill: #00519B !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Utilities ----------------
@st.cache_data(show_spinner=False)
def conus_limits():
    t = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    x0, y0 = t.transform(-130, 24)
    x1, y1 = t.transform(-60, 55)
    pad = 100_000
    return min(x0, x1) - pad, max(x0, x1) + pad, min(y0, y1) - pad, max(y0, y1) + pad


@st.cache_data(show_spinner=False)
def load_geo() -> gpd.GeoDataFrame | None:
    if not MASTER_GEO.exists():
        st.warning(f"GeoJSON not found: {MASTER_GEO}.")
        return None
    g = gpd.read_file(MASTER_GEO)
    if "region" not in g.columns:
        st.warning("GeoJSON missing 'region' column. Polygons still draw but filtering uses that field.")
        g["region"] = None
    g["geometry"] = g["geometry"].apply(make_valid)
    return g.to_crs(TARGET_CRS)


@st.cache_data(show_spinner=False)
def load_us_borders():
    """
    Load USA outline and lower 48 states from Natural Earth via cartopy.
    Cached so we do not keep hitting disk.
    """
    try:
        countries = gpd.read_file(
            shp.natural_earth("110m", "cultural", "admin_0_countries")
        ).to_crs(TARGET_CRS)
        iso_col = next(
            c for c in ["ADM0_A3", "ISO_A3", "adm0_a3", "iso_a3"] if c in countries.columns
        )
        usa = countries[countries[iso_col] == "USA"].copy()

        states = gpd.read_file(
            shp.natural_earth("10m", "cultural", "admin_1_states_provinces")
        ).to_crs(TARGET_CRS)
        a3 = "adm0_a3" if "adm0_a3" in states.columns else "ADM0_A3"
        states = states[
            (states[a3] == "USA")
            & (~states["name"].isin({"Alaska", "Hawaii", "Puerto Rico"}))
        ].copy()

        return usa, states
    except Exception as e:
        st.error(f"Could not load Natural Earth borders: {e}")
        return None, None


@st.cache_data(show_spinner=False)
def get_system_fonts():
    font_files = fm.findSystemFonts(fontext="ttf") + fm.findSystemFonts(fontext="otf")
    names = set()
    for fpath in font_files:
        try:
            prop = fm.FontProperties(fname=fpath)
            name = prop.get_name()
            if name:
                names.add(name)
        except Exception:
            pass
    fonts = sorted(names, key=str.lower)
    if not fonts:
        fonts = ["Calibri", "Cambria", "Arial", "Helvetica", "Times New Roman"]
    return fonts


def _normalize_tech_value(x: str) -> str | None:
    s = str(x).lower()
    if "wind" in s:
        return "Wind"
    if "solar" in s or "pv" in s or "photovoltaic" in s or ("bess" in s and ("solar" in s or "pv" in s)):
        return "Solar"
    return None


def _normalize_hub_value(iso: str, hub: str) -> str:
    iso_clean = str(iso).strip()
    hub_clean = str(hub).strip()

    if iso_clean == "AESO" and hub_clean.lower().startswith("alberta"):
        return "Alberta"

    return hub_clean


@st.cache_data(show_spinner=False)
def load_financials(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    if file.name.lower().endswith(".csv"):
        fin = pd.read_csv(file)
    else:
        fin = pd.read_excel(file)

    fin = fin.copy()

    rename_map = {}
    for c in fin.columns:
        cl = c.strip().lower()
        if cl in {"iso", "rto", "iso/rto", "region"}:
            rename_map[c] = ISO_COL
        elif cl in {
            "hub",
            "settlement hub",
            "settlement_hub",
            "node",
            "settlement point",
            "settlement_point",
        }:
            rename_map[c] = HUB_COL
        elif cl in {"price ($/mwh)", "price", "lmp", "value", "price_mwh"}:
            rename_map[c] = PRICE_COL
        elif cl in {"atc price ($/mwh)", "atc", "atc price", "atc_mwh"}:
            rename_map[c] = ATC_COL
        elif cl in {"developer", "dev", "sponsor"}:
            rename_map[c] = DEV_COL
        elif cl in {"technology", "tech", "resource", "fuel", "asset type", "asset_type"}:
            rename_map[c] = TECH_COL

    fin = fin.rename(columns=rename_map)

    missing = [c for c in [ISO_COL, HUB_COL, PRICE_COL] if c not in fin.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return pd.DataFrame()

    fin[HUB_COL] = (
        fin[HUB_COL]
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.strip()
    )

    fin[ISO_COL] = fin[ISO_COL].astype(str).str.strip()

    fin[HUB_COL] = fin.apply(lambda row: _normalize_hub_value(row[ISO_COL], row[HUB_COL]), axis=1)

    fin[PRICE_COL] = (
        fin[PRICE_COL].astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )

    if ATC_COL in fin.columns:
        fin[ATC_COL] = (
            fin[ATC_COL].astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", pd.NA)
            .pipe(pd.to_numeric, errors="coerce")
        )

    if DEV_COL in fin.columns:
        fin[DEV_COL] = fin[DEV_COL].astype(str).str.strip()
    if TECH_COL in fin.columns:
        fin[TECH_COL] = fin[TECH_COL].apply(_normalize_tech_value)

    fin = fin.dropna(subset=[PRICE_COL])
    return fin


def _cluster_by_distance_fast(pts_gdf, threshold_m=150000):
    xy = np.column_stack([pts_gdf.geometry.x.values, pts_gdf.geometry.y.values])
    idxs = pts_gdf.index.to_list()
    n = len(xy)
    if n <= 1:
        return []

    diff = xy[:, None, :] - xy[None, :, :]
    dist2 = (diff ** 2).sum(axis=2)
    close = dist2 <= threshold_m**2

    parent = {i: i for i in idxs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if close[i, j]:
                union(idxs[i], idxs[j])

    groups = {}
    for i in idxs:
        r = find(i)
        groups.setdefault(r, set()).add(i)
    return list(groups.values())


def compute_auto_offsets(pts_gdf, threshold_m=150000):
    offsets = {i: (0, 0) for i in pts_gdf.index}
    for group in _cluster_by_distance_fast(pts_gdf, threshold_m=threshold_m):
        if len(group) <= 1:
            continue
        cyc = count(0)
        for idx in group:
            k = next(cyc) % len(OFFSET_WHEEL)
            offsets[idx] = OFFSET_WHEEL[k]
    return offsets


def prepare_labels(fin: pd.DataFrame, selected_isos: list[str], developer: str | None) -> pd.DataFrame:
    if fin.empty or not selected_isos:
        return pd.DataFrame()

    scoped = fin[fin[ISO_COL].isin(selected_isos)].copy()
    if developer and DEV_COL in scoped.columns:
        scoped = scoped[scoped[DEV_COL] == developer]

    if scoped.empty:
        return pd.DataFrame()

    hubs = (
        scoped.groupby([ISO_COL, HUB_COL], dropna=True)
        .size()
        .reset_index()[[ISO_COL, HUB_COL]]
    )

    if ATC_COL in scoped.columns:
        atc_tbl = (
            scoped.groupby([ISO_COL, HUB_COL], dropna=True)[ATC_COL]
            .mean()
            .reset_index()
            .rename(columns={ATC_COL: "ATC"})
        )
    else:
        atc_tbl = pd.DataFrame(columns=[ISO_COL, HUB_COL, "ATC"])

    solar_tbl = (
        scoped[scoped.get(TECH_COL) == "Solar"]
        .groupby([ISO_COL, HUB_COL], dropna=True)[PRICE_COL]
        .mean()
        .reset_index()
        .rename(columns={PRICE_COL: "Solar"})
    )

    wind_tbl = (
        scoped[scoped.get(TECH_COL) == "Wind"]
        .groupby([ISO_COL, HUB_COL], dropna=True)[PRICE_COL]
        .mean()
        .reset_index()
        .rename(columns={PRICE_COL: "Wind"})
    )

    vals = hubs.merge(atc_tbl, on=[ISO_COL, HUB_COL], how="left")
    vals = vals.merge(solar_tbl, on=[ISO_COL, HUB_COL], how="left")
    vals = vals.merge(wind_tbl, on=[ISO_COL, HUB_COL], how="left")

    cfg_map = {c["iso"]: c for c in ISO_CONFIGS}
    proxies = []
    for iso in selected_isos:
        cfg = cfg_map.get(iso)
        if not cfg:
            continue
        for hub, (lon, lat) in cfg["hub_proxy"].items():
            proxies.append({ISO_COL: iso, HUB_COL: hub, "lon": lon, "lat": lat})

    proxy_df = pd.DataFrame(proxies)
    out = proxy_df.merge(vals, on=[ISO_COL, HUB_COL], how="inner")
    return out


# ---------------- Plotting ----------------
def draw_map(
    labels: pd.DataFrame,
    g: gpd.GeoDataFrame | None,
    selected_isos: list[str],
    manual_offsets: dict | None,
    replace_auto: bool,
    show_state_borders: bool,
    label_fontsize: float,
    use_halo: bool,
    halo_color: str,
    label_font_family: str,
    font_style: str,
    font_weight: str,
    label_color: str,
    show_atc: bool,
    show_solar: bool,
    show_wind: bool,
    transparent_bg: bool = True,
):
    if labels.empty:
        st.info("No hubs match the current filters.")
        return None, None

    plt.rcParams["font.family"] = label_font_family

    dpi = 300
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=dpi)

    # Precompute union of ISO polygons for masking state borders
    iso_union = None
    if g is not None and not g.empty:
        iso_geom = g[g["region"].isin(selected_isos)]
        if not iso_geom.empty:
            iso_union = iso_geom.geometry.unary_union

    # 1) Draw borders first, but clip state borders where ISO regions exist
    if show_state_borders:
        usa, states = load_us_borders()
        if usa is not None:
            usa.boundary.plot(
                ax=ax,
                color=BORDER_COLOR,
                linewidth=COUNTRY_LW,
                zorder=2,
            )
        if states is not None and not states.empty:
            if iso_union is not None and not iso_union.is_empty:
                states_for_plot = states.copy()
                states_for_plot["geometry"] = states_for_plot.geometry.map(
                    lambda geom: geom.difference(iso_union)
                )
                states_for_plot = states_for_plot[~states_for_plot.geometry.is_empty]
            else:
                states_for_plot = states

            if not states_for_plot.empty:
                states_for_plot.boundary.plot(
                    ax=ax,
                    color=BORDER_COLOR,
                    linewidth=STATE_LW,
                    zorder=2.5,
                )

    # 2) Draw ISO regions on top
    if g is not None and not g.empty:
        for iso in selected_isos:
            sub = g[g["region"] == iso]
            if sub.empty:
                continue
            sub.plot(
                ax=ax,
                color=REGION_COLORS.get(iso, "#999999"),
                alpha=1.0,
                edgecolor=REGION_COLORS.get(iso, "#666666"),
                linewidth=1.15,
                zorder=3,
            )

    # 3) Label points
    pts = gpd.GeoDataFrame(
        labels,
        geometry=gpd.points_from_xy(labels["lon"], labels["lat"], crs="EPSG:4326"),
    ).to_crs(TARGET_CRS)

    auto = compute_auto_offsets(pts, threshold_m=150000)
    manual_offsets = manual_offsets or {}

    def fmt(v):
        return None if pd.isna(v) else f"${v:.2f}"

    for ridx, r in pts.iterrows():
        iso = r[ISO_COL]
        hub = r[HUB_COL]

        atc_txt = fmt(r.get("ATC")) if show_atc else None
        sol_txt = fmt(r.get("Solar")) if show_solar else None
        win_txt = fmt(r.get("Wind")) if show_wind else None

        x, y = r.geometry.x, r.geometry.y

        dx_auto, dy_auto = auto.get(ridx, (0, 0))
        dx_man, dy_man = manual_offsets.get(hub, DEFAULT_OFFSETS.get(hub, (0, 0)))

        if replace_auto:
            dx, dy = dx_man, dy_man
        else:
            dx = dx_auto + dx_man
            dy = dy_auto + dy_man

        region_color = REGION_COLORS.get(iso, "#666666")

        lines = [hub]
        if atc_txt is not None:
            lines.append(f"ATC: {atc_txt}")
        if sol_txt is not None:
            lines.append(f"Solar: {sol_txt}")
        if win_txt is not None:
            lines.append(f"Wind: {win_txt}")

        txt = "\n".join(lines)

        ax.annotate(
            txt,
            xy=(x, y),
            xycoords="data",
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=label_fontsize,
            fontfamily=label_font_family,
            fontstyle=font_style,
            weight=font_weight,
            color=label_color,
            arrowprops=dict(
                arrowstyle="-",
                color=region_color,
                lw=0.9,
                shrinkA=0,
                shrinkB=2,
            ),
            path_effects=(
                [withStroke(linewidth=1.6, foreground=halo_color)]
                if use_halo
                else None
            ),
            zorder=12,
        )

    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(
            facecolor=REGION_COLORS[i],
            edgecolor=REGION_COLORS[i],
            label=i,
            alpha=1.0,
        )
        for i in selected_isos
        if i in REGION_COLORS
    ]

    if handles:
        legend_fontprops = fm.FontProperties(
            family=label_font_family,
            style=font_style,
            weight=font_weight,
            size=label_fontsize,
        )

        ax.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, -0.04),
            borderaxespad=0.6,
            prop=legend_fontprops,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()

    xmin, xmax, ymin, ymax = conus_limits()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=transparent_bg,
    )
    buf.seek(0)
    return fig, buf


# ---------------- Streamlit UI ----------------
with st.sidebar:
    st.header("Inputs")

    fin_file = st.file_uploader(
        "Financials (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=False,
    )
    fin = load_financials(fin_file)
    g = load_geo()

    st.markdown("### Font")
    font_options = get_system_fonts()
    default_font = "Cambria" if "Cambria" in font_options else font_options[0]

    label_font_family = st.selectbox(
        "Label font family",
        options=font_options,
        index=font_options.index(default_font),
    )

    font_style_choice = st.radio(
        "Font style",
        options=["Normal", "Italic"],
        index=0,
        horizontal=True,
    )
    font_style = "italic" if font_style_choice == "Italic" else "normal"

    font_weight_choice = st.radio(
        "Font weight",
        options=["Normal", "Bold"],
        index=0,
        horizontal=True,
    )
    font_weight = "bold" if font_weight_choice == "Bold" else "normal"

    developer = None
    if DEV_COL in fin.columns and not fin.empty:
        devs = sorted(
            [d for d in fin[DEV_COL].dropna().unique().tolist() if str(d).strip()]
        )
        dev_choice = st.selectbox(
            "Developer filter (optional)", options=["All developers"] + devs, index=0
        )
        if dev_choice != "All developers":
            developer = dev_choice
    else:
        st.caption("Tip: Add a 'Developer' column to enable developer filtering.")

    iso_all = [c["iso"] for c in ISO_CONFIGS]
    if not fin.empty:
        fin_scoped = fin.copy()
        if developer and DEV_COL in fin_scoped.columns:
            fin_scoped = fin_scoped[fin_scoped[DEV_COL] == developer]
        iso_in_data = sorted(fin_scoped[ISO_COL].dropna().unique().tolist())
        iso_options = [i for i in iso_all if i in iso_in_data]
    else:
        iso_options = iso_all

    if not iso_options:
        st.warning("No ISOs available for current filters.")
        st.stop()

    st.markdown("### ISO/RTO Selection")

    prior = st.session_state.get("iso_selected", [])
    sanitized_prior = [i for i in prior if i in iso_options]
    if not sanitized_prior:
        sanitized_prior = iso_options[:]

    select_all = st.checkbox("Select all ISO/RTOs", value=True, key="select_all_isos")
    current_default = iso_options[:] if select_all else sanitized_prior

    if hasattr(st, "pills"):
        selected_isos = st.pills(
            "Choose ISO/RTOs",
            options=iso_options,
            selection_mode="multi",
            default=current_default,
        )
    else:
        selected_isos = st.multiselect(
            "Choose ISO/RTOs",
            options=iso_options,
            default=current_default,
        )

    st.session_state["iso_selected"] = selected_isos

    transparent = st.checkbox("Transparent background (PNG)", value=True)
    show_states = st.checkbox("Show US state borders", value=True)

    st.markdown("### Price types")
    show_atc = st.checkbox("Show ATC Prices", value=True)
    show_solar = st.checkbox("Show Solar Prices", value=False)
    show_wind = st.checkbox("Show Wind Prices", value=False)

    st.markdown("### Label styling")
    label_fontsize = st.number_input(
        "Label font size",
        min_value=5.5,
        max_value=7.5,
        value=6.5,
        step=0.05,
        format="%.2f",
    )

    label_color = st.color_picker("Label font color", value="#000000")
    use_halo = st.checkbox("Add text halo (outline)", value=True)
    halo_color = st.color_picker("Halo color", value="#FFFFFF")

    labels_preview = pd.DataFrame()
    if fin_file is not None and selected_isos:
        labels_preview = prepare_labels(fin, selected_isos, developer)

    manual_offsets = {}
    replace_auto = False
    with st.expander("Label offsets (optional)"):
        st.caption("Values are pixels relative to each hub point. Positive dx right, positive dy up.")
        mode = st.radio(
            "Offset mode",
            ["Add to auto offsets", "Replace auto offsets"],
            index=0,
            horizontal=True,
        )
        replace_auto = mode == "Replace auto offsets"

        if not labels_preview.empty:
            for hub in sorted(labels_preview[HUB_COL].unique()):
                base_dx, base_dy = DEFAULT_OFFSETS.get(hub, (0, 0))
                c1, c2 = st.columns(2)
                dx = c1.number_input(f"{hub} dx", value=int(base_dx), step=1, format="%d")
                dy = c2.number_input(f"{hub} dy", value=int(base_dy), step=1, format="%d")
                if dx != base_dx or dy != base_dy:
                    manual_offsets[hub] = (dx, dy)
        else:
            st.write("Upload financials and choose ISOs to enable offset controls.")


if fin_file is None:
    st.info("Upload your Financials file to begin.")
    st.stop()

if not selected_isos:
    st.warning("Pick at least one ISO/RTO.")
    st.stop()


def _compose_offsets(user_offsets: dict) -> dict:
    out = DEFAULT_OFFSETS.copy()
    out.update(user_offsets or {})
    return out


labels = prepare_labels(fin, selected_isos, developer)
final_offsets = _compose_offsets(manual_offsets)

fig, png_buf = draw_map(
    labels=labels,
    g=g,
    selected_isos=selected_isos,
    manual_offsets=final_offsets,
    replace_auto=replace_auto,
    show_state_borders=show_states,
    label_fontsize=label_fontsize,
    use_halo=use_halo,
    halo_color=halo_color,
    label_font_family=label_font_family,
    font_style=font_style,
    font_weight=font_weight,
    label_color=label_color,
    show_atc=show_atc,
    show_solar=show_solar,
    show_wind=show_wind,
    transparent_bg=transparent,
)

if fig is not None:
    st.pyplot(fig, clear_figure=True)
    st.download_button(
        label="Download PNG (no background)",
        data=png_buf,
        file_name="price_map.png",
        mime="image/png",
    )
from io import BytesIO
from pathlib import Path
from itertools import count

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib import font_manager as fm
from shapely import make_valid
from pyproj import Transformer
from cartopy.io import shapereader as shp  # cartopy for borders


# ---------------- Page setup ----------------
st.set_page_config(page_title="ISO Price Map", layout="wide")
st.title("ISO Price Map")


# ---------------- Constants ----------------
TARGET_CRS = "EPSG:3857"

ISO_COL   = "ISO/RTO"
HUB_COL   = "Settlement Hub"
PRICE_COL = "Price ($/MWh)"
ATC_COL   = "ATC Price ($/MWh)"
DEV_COL   = "Developer"
TECH_COL  = "Technology"

MASTER_GEO = Path("us_canada_rto_na.geojson")

REGION_COLORS = {
    "CAISO":  "#F68220",
    "ERCOT":  "#853F05",
    "MISO":   "#0EC477",
    "PJM":    "#5B6670",
    "ISO-NE": "#FF0404",
    "NYISO":  "#578EB1",
    "SPP":    "#00B0F0",
    "AESO":   "#C00000",
    "IESO":   "#00519B",
}

BORDER_COLOR = "#75838FFF"
COUNTRY_LW   = 0.10
STATE_LW     = 0.10

OFFSET_WHEEL = [
    (0, 12), (12, 0), (-12, 0), (0, -12),
    (10, 10), (-10, 10), (-10, -10), (10, -10),
    (16, 6), (-16, 6),
]

ISO_CONFIGS = [
    {"iso": "CAISO", "hub_proxy": {
        "NP15": (-121.4944, 38.5816),
        "SP15": (-117.8880, 34.0522),
    }},
    {"iso": "ERCOT", "hub_proxy": {
        "HB_NORTH": (-97.3308, 32.7555),
        "HB_HOUSTON": (-95.36327, 29.76328),
        "HB_SOUTH": (-96.9978, 28.8053),
        "HB_WEST": (-102.3677, 31.8457),
    }},
    {"iso": "MISO", "hub_proxy": {
        "INDIANA.HUB": (-86.1581, 39.7684),
        "ILLINOIS.HUB": (-89.6500, 39.7817),
        "MICHIGAN.HUB": (-84.5555, 42.7325),
        "MINN.HUB": (-93.2650, 44.9778),
        "TEXAS.HUB": (-94.1266, 30.0802),
        "ARKANSAS.HUB": (-92.2896, 34.7465),
        "LOUISIANA.HUB": (-91.1871, 30.4515),
    }},
    {"iso": "PJM", "hub_proxy": {
        "AD Hub": (-84.1917, 39.7589),
        "NI Hub": (-87.6298, 41.8781),
        "PEPCO": (-77.0369, 38.9072),
        "BGE": (-76.6122, 39.2904),
        "PSEG": (-74.1724, 40.7357),
        "JCPL": (-74.4830, 40.7968),
        "METED": (-75.9269, 40.3356),
        "PPL": (-75.4902, 40.6023),
        "COMED": (-87.6298, 41.8781),
        "DEOK": (-84.5120, 39.1031),
        "ATSI": (-81.6944, 41.4993),
        "DOMINION HUB": (-77.4360, 37.5407),
        "PECO": (-75.1652, 39.9526),
        "PENELEC": (-78.3947, 40.5187),
        "RECO": (-74.0324, 40.9263),
        "AEP-DAYTON HUB": (-84.1917, 39.7589),
        "APS": (-79.9559, 39.6295),
        "DPL": (-75.5467, 39.7447),
        "DUQ": (-80.0000, 40.4406),
        "EASTERN HUB": (-75.1652, 39.9526),
        "WESTERN HUB": (-80.0000, 40.4406),
        "AEP": (-82.9988, 39.9612),
    }},
    {"iso": "ISO-NE", "hub_proxy": {
        "Mass Hub": (-71.0589, 42.3601),
        ".Z.CONNECTICUT": (-72.6851, 41.7637),
        ".Z.MAINE": (-69.7795, 44.3106),
        ".Z.NEMASSBOST": (-71.0120, 42.3954),
        ".Z.NEWHAMPSHIRE": (-71.4548, 43.2070),
        ".Z.RHODEISLAND": (-71.4128, 41.8236),
        ".Z.SEMASS": (-70.9167, 41.6362),
        ".Z.VERMONT": (-72.5754, 44.2601),
        ".Z.WCMASS": (-72.5898, 42.1015),
    }},
    {"iso": "NYISO", "hub_proxy": {
        "West - A": (-78.8784, 42.8864),
        "Genesee - B": (-77.6156, 43.1566),
        "Central - C": (-76.1474, 43.0481),
        "North - D": (-75.9108, 44.6995),
        "Mohawk Valley - E": (-75.2327, 43.1009),
        "Capital - F": (-73.7562, 42.6526),
        "Hudson Valley - G": (-73.9235, 41.7004),
        "N.Y.C. - J": (-73.9857, 40.7484),
        "Long Island - K": (-73.1960, 40.7891),
    }},
    {"iso": "AESO", "hub_proxy": {
        "Alberta": (-113.4938, 53.5461),
    }},
    {"iso": "SPP", "hub_proxy": {
        "SPPNORTH_HUB": (-103.9208, 46.2950),
        "SPPSOUTH_HUB": (-94.7040, 37.4109),
    }},
]

DEFAULT_OFFSETS = {
    "HB_HOUSTON": (-5, -20),
    "HB_SOUTH": (-70, -10),
    "HB_WEST": (-80, -10),
    "MICHIGAN.HUB": (10, 60),
    "INDIANA.HUB": (-25, -40),
    "SPPSOUTH_HUB": (-130, 13),
    "SPPNORTH_HUB": (-90, 12),
    "AEP": (0, 50),
    "ILLINOIS.HUB": (-30, 0),
    "WESTERN HUB": (50, -20),
    "NP15": (-60, 5),
    "SP15": (-60, 6),
    ".Z.MAINE": (20, 0),
    "LOUISIANA.HUB": (24, -20),
    "ARKANSAS.HUB": (35, -20),
    "AEP-DAYTON HUB": (30, -71),
    "Alberta": (50, 7),
}


# ---------------- Styling for chips ----------------
st.markdown(
    """
    <style>
    label[data-testid="stWidgetLabel"] > div {
        font-weight: 600;
        color: #00519B;
    }
    div[data-baseweb="tag"] {
        background-color: #E6F0FA !important;
        color: #00519B !important;
        border-radius: 12px !important;
        padding: 2px 8px !important;
        border: 1px solid #B7D3F6 !important;
    }
    div[data-baseweb="tag"] svg { fill: #00519B !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Utilities ----------------
@st.cache_data(show_spinner=False)
def conus_limits():
    t = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    x0, y0 = t.transform(-130, 24)
    x1, y1 = t.transform(-60, 55)
    pad = 100_000
    return min(x0, x1) - pad, max(x0, x1) + pad, min(y0, y1) - pad, max(y0, y1) + pad


@st.cache_data(show_spinner=False)
def load_geo() -> gpd.GeoDataFrame | None:
    if not MASTER_GEO.exists():
        st.warning(f"GeoJSON not found: {MASTER_GEO}.")
        return None
    g = gpd.read_file(MASTER_GEO)
    if "region" not in g.columns:
        st.warning("GeoJSON missing 'region' column. Polygons still draw but filtering uses that field.")
        g["region"] = None
    g["geometry"] = g["geometry"].apply(make_valid)
    return g.to_crs(TARGET_CRS)


@st.cache_data(show_spinner=False)
def load_us_borders():
    """
    Load USA outline and lower 48 states from Natural Earth via cartopy.
    Cached so we do not keep hitting disk.
    """
    try:
        countries = gpd.read_file(
            shp.natural_earth("110m", "cultural", "admin_0_countries")
        ).to_crs(TARGET_CRS)
        iso_col = next(
            c for c in ["ADM0_A3", "ISO_A3", "adm0_a3", "iso_a3"] if c in countries.columns
        )
        usa = countries[countries[iso_col] == "USA"].copy()

        states = gpd.read_file(
            shp.natural_earth("10m", "cultural", "admin_1_states_provinces")
        ).to_crs(TARGET_CRS)
        a3 = "adm0_a3" if "adm0_a3" in states.columns else "ADM0_A3"
        states = states[
            (states[a3] == "USA")
            & (~states["name"].isin({"Alaska", "Hawaii", "Puerto Rico"}))
        ].copy()

        return usa, states
    except Exception as e:
        st.error(f"Could not load Natural Earth borders: {e}")
        return None, None


@st.cache_data(show_spinner=False)
def get_system_fonts():
    font_files = fm.findSystemFonts(fontext="ttf") + fm.findSystemFonts(fontext="otf")
    names = set()
    for fpath in font_files:
        try:
            prop = fm.FontProperties(fname=fpath)
            name = prop.get_name()
            if name:
                names.add(name)
        except Exception:
            pass
    fonts = sorted(names, key=str.lower)
    if not fonts:
        fonts = ["Calibri", "Cambria", "Arial", "Helvetica", "Times New Roman"]
    return fonts


def _normalize_tech_value(x: str) -> str | None:
    s = str(x).lower()
    if "wind" in s:
        return "Wind"
    if "solar" in s or "pv" in s or "photovoltaic" in s or ("bess" in s and ("solar" in s or "pv" in s)):
        return "Solar"
    return None


def _normalize_hub_value(iso: str, hub: str) -> str:
    iso_clean = str(iso).strip()
    hub_clean = str(hub).strip()

    if iso_clean == "AESO" and hub_clean.lower().startswith("alberta"):
        return "Alberta"

    return hub_clean


@st.cache_data(show_spinner=False)
def load_financials(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    if file.name.lower().endswith(".csv"):
        fin = pd.read_csv(file)
    else:
        fin = pd.read_excel(file)

    fin = fin.copy()

    rename_map = {}
    for c in fin.columns:
        cl = c.strip().lower()
        if cl in {"iso", "rto", "iso/rto", "region"}:
            rename_map[c] = ISO_COL
        elif cl in {
            "hub",
            "settlement hub",
            "settlement_hub",
            "node",
            "settlement point",
            "settlement_point",
        }:
            rename_map[c] = HUB_COL
        elif cl in {"price ($/mwh)", "price", "lmp", "value", "price_mwh"}:
            rename_map[c] = PRICE_COL
        elif cl in {"atc price ($/mwh)", "atc", "atc price", "atc_mwh"}:
            rename_map[c] = ATC_COL
        elif cl in {"developer", "dev", "sponsor"}:
            rename_map[c] = DEV_COL
        elif cl in {"technology", "tech", "resource", "fuel", "asset type", "asset_type"}:
            rename_map[c] = TECH_COL

    fin = fin.rename(columns=rename_map)

    missing = [c for c in [ISO_COL, HUB_COL, PRICE_COL] if c not in fin.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return pd.DataFrame()

    fin[HUB_COL] = (
        fin[HUB_COL]
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.strip()
    )

    fin[ISO_COL] = fin[ISO_COL].astype(str).str.strip()

    fin[HUB_COL] = fin.apply(lambda row: _normalize_hub_value(row[ISO_COL], row[HUB_COL]), axis=1)

    fin[PRICE_COL] = (
        fin[PRICE_COL].astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )

    if ATC_COL in fin.columns:
        fin[ATC_COL] = (
            fin[ATC_COL].astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", pd.NA)
            .pipe(pd.to_numeric, errors="coerce")
        )

    if DEV_COL in fin.columns:
        fin[DEV_COL] = fin[DEV_COL].astype(str).str.strip()
    if TECH_COL in fin.columns:
        fin[TECH_COL] = fin[TECH_COL].apply(_normalize_tech_value)

    fin = fin.dropna(subset=[PRICE_COL])
    return fin


def _cluster_by_distance_fast(pts_gdf, threshold_m=150000):
    xy = np.column_stack([pts_gdf.geometry.x.values, pts_gdf.geometry.y.values])
    idxs = pts_gdf.index.to_list()
    n = len(xy)
    if n <= 1:
        return []

    diff = xy[:, None, :] - xy[None, :, :]
    dist2 = (diff ** 2).sum(axis=2)
    close = dist2 <= threshold_m**2

    parent = {i: i for i in idxs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if close[i, j]:
                union(idxs[i], idxs[j])

    groups = {}
    for i in idxs:
        r = find(i)
        groups.setdefault(r, set()).add(i)
    return list(groups.values())


def compute_auto_offsets(pts_gdf, threshold_m=150000):
    offsets = {i: (0, 0) for i in pts_gdf.index}
    for group in _cluster_by_distance_fast(pts_gdf, threshold_m=threshold_m):
        if len(group) <= 1:
            continue
        cyc = count(0)
        for idx in group:
            k = next(cyc) % len(OFFSET_WHEEL)
            offsets[idx] = OFFSET_WHEEL[k]
    return offsets


def prepare_labels(fin: pd.DataFrame, selected_isos: list[str], developer: str | None) -> pd.DataFrame:
    if fin.empty or not selected_isos:
        return pd.DataFrame()

    scoped = fin[fin[ISO_COL].isin(selected_isos)].copy()
    if developer and DEV_COL in scoped.columns:
        scoped = scoped[scoped[DEV_COL] == developer]

    if scoped.empty:
        return pd.DataFrame()

    hubs = (
        scoped.groupby([ISO_COL, HUB_COL], dropna=True)
        .size()
        .reset_index()[[ISO_COL, HUB_COL]]
    )

    if ATC_COL in scoped.columns:
        atc_tbl = (
            scoped.groupby([ISO_COL, HUB_COL], dropna=True)[ATC_COL]
            .mean()
            .reset_index()
            .rename(columns={ATC_COL: "ATC"})
        )
    else:
        atc_tbl = pd.DataFrame(columns=[ISO_COL, HUB_COL, "ATC"])

    solar_tbl = (
        scoped[scoped.get(TECH_COL) == "Solar"]
        .groupby([ISO_COL, HUB_COL], dropna=True)[PRICE_COL]
        .mean()
        .reset_index()
        .rename(columns={PRICE_COL: "Solar"})
    )

    wind_tbl = (
        scoped[scoped.get(TECH_COL) == "Wind"]
        .groupby([ISO_COL, HUB_COL], dropna=True)[PRICE_COL]
        .mean()
        .reset_index()
        .rename(columns={PRICE_COL: "Wind"})
    )

    vals = hubs.merge(atc_tbl, on=[ISO_COL, HUB_COL], how="left")
    vals = vals.merge(solar_tbl, on=[ISO_COL, HUB_COL], how="left")
    vals = vals.merge(wind_tbl, on=[ISO_COL, HUB_COL], how="left")

    cfg_map = {c["iso"]: c for c in ISO_CONFIGS}
    proxies = []
    for iso in selected_isos:
        cfg = cfg_map.get(iso)
        if not cfg:
            continue
        for hub, (lon, lat) in cfg["hub_proxy"].items():
            proxies.append({ISO_COL: iso, HUB_COL: hub, "lon": lon, "lat": lat})

    proxy_df = pd.DataFrame(proxies)
    out = proxy_df.merge(vals, on=[ISO_COL, HUB_COL], how="inner")
    return out


# ---------------- Plotting ----------------
def draw_map(
    labels: pd.DataFrame,
    g: gpd.GeoDataFrame | None,
    selected_isos: list[str],
    manual_offsets: dict | None,
    replace_auto: bool,
    show_state_borders: bool,
    label_fontsize: float,
    use_halo: bool,
    halo_color: str,
    label_font_family: str,
    font_style: str,
    font_weight: str,
    label_color: str,
    show_atc: bool,
    show_solar: bool,
    show_wind: bool,
    transparent_bg: bool = True,
):
    if labels.empty:
        st.info("No hubs match the current filters.")
        return None, None

    plt.rcParams["font.family"] = label_font_family

    dpi = 300
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=dpi)

    # 1) Draw borders first, without clipping to ISO shapes
    if show_state_borders:
        usa, states = load_us_borders()
        if usa is not None:
            usa.boundary.plot(
                ax=ax,
                color=BORDER_COLOR,
                linewidth=COUNTRY_LW,
                zorder=2,
            )
        if states is not None and not states.empty:
            states.boundary.plot(
                ax=ax,
                color=BORDER_COLOR,
                linewidth=STATE_LW,
                zorder=2.5,
            )

    # 2) Draw ISO regions on top
    if g is not None and not g.empty:
        for iso in selected_isos:
            sub = g[g["region"] == iso]
            if sub.empty:
                continue
            sub.plot(
                ax=ax,
                color=REGION_COLORS.get(iso, "#999999"),
                alpha=1.0,
                edgecolor=REGION_COLORS.get(iso, "#666666"),
                linewidth=1.15,
                zorder=3,
            )

    # 3) Label points
    pts = gpd.GeoDataFrame(
        labels,
        geometry=gpd.points_from_xy(labels["lon"], labels["lat"], crs="EPSG:4326"),
    ).to_crs(TARGET_CRS)

    auto = compute_auto_offsets(pts, threshold_m=150000)
    manual_offsets = manual_offsets or {}

    def fmt(v):
        return None if pd.isna(v) else f"${v:.2f}"

    for ridx, r in pts.iterrows():
        iso = r[ISO_COL]
        hub = r[HUB_COL]

        atc_txt = fmt(r.get("ATC")) if show_atc else None
        sol_txt = fmt(r.get("Solar")) if show_solar else None
        win_txt = fmt(r.get("Wind")) if show_wind else None

        x, y = r.geometry.x, r.geometry.y

        dx_auto, dy_auto = auto.get(ridx, (0, 0))
        dx_man, dy_man = manual_offsets.get(hub, DEFAULT_OFFSETS.get(hub, (0, 0)))

        if replace_auto:
            dx, dy = dx_man, dy_man
        else:
            dx = dx_auto + dx_man
            dy = dy_auto + dy_man

        region_color = REGION_COLORS.get(iso, "#666666")

        lines = [hub]
        if atc_txt is not None:
            lines.append(f"ATC: {atc_txt}")
        if sol_txt is not None:
            lines.append(f"Solar: {sol_txt}")
        if win_txt is not None:
            lines.append(f"Wind: {win_txt}")

        txt = "\n".join(lines)

        ax.annotate(
            txt,
            xy=(x, y),
            xycoords="data",
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=label_fontsize,
            fontfamily=label_font_family,
            fontstyle=font_style,
            weight=font_weight,
            color=label_color,
            arrowprops=dict(
                arrowstyle="-",
                color=region_color,
                lw=0.9,
                shrinkA=0,
                shrinkB=2,
            ),
            path_effects=(
                [withStroke(linewidth=1.6, foreground=halo_color)]
                if use_halo
                else None
            ),
            zorder=12,
        )

    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(
            facecolor=REGION_COLORS[i],
            edgecolor=REGION_COLORS[i],
            label=i,
            alpha=1.0,
        )
        for i in selected_isos
        if i in REGION_COLORS
    ]

    if handles:
        legend_fontprops = fm.FontProperties(
            family=label_font_family,
            style=font_style,
            weight=font_weight,
            size=label_fontsize,
        )

        ax.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, -0.04),
            borderaxespad=0.6,
            prop=legend_fontprops,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()

    xmin, xmax, ymin, ymax = conus_limits()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=transparent_bg,
    )
    buf.seek(0)
    return fig, buf


# ---------------- Streamlit UI ----------------
with st.sidebar:
    st.header("Inputs")

    fin_file = st.file_uploader(
        "Financials (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=False,
    )
    fin = load_financials(fin_file)
    g = load_geo()

    st.markdown("### Font")
    font_options = get_system_fonts()
    default_font = "Cambria" if "Cambria" in font_options else font_options[0]

    label_font_family = st.selectbox(
        "Label font family",
        options=font_options,
        index=font_options.index(default_font),
    )

    font_style_choice = st.radio(
        "Font style",
        options=["Normal", "Italic"],
        index=0,
        horizontal=True,
    )
    font_style = "italic" if font_style_choice == "Italic" else "normal"

    font_weight_choice = st.radio(
        "Font weight",
        options=["Normal", "Bold"],
        index=0,
        horizontal=True,
    )
    font_weight = "bold" if font_weight_choice == "Bold" else "normal"

    developer = None
    if DEV_COL in fin.columns and not fin.empty:
        devs = sorted(
            [d for d in fin[DEV_COL].dropna().unique().tolist() if str(d).strip()]
        )
        dev_choice = st.selectbox(
            "Developer filter (optional)", options=["All developers"] + devs, index=0
        )
        if dev_choice != "All developers":
            developer = dev_choice
    else:
        st.caption("Tip: Add a 'Developer' column to enable developer filtering.")

    iso_all = [c["iso"] for c in ISO_CONFIGS]
    if not fin.empty:
        fin_scoped = fin.copy()
        if developer and DEV_COL in fin_scoped.columns:
            fin_scoped = fin_scoped[fin_scoped[DEV_COL] == developer]
        iso_in_data = sorted(fin_scoped[ISO_COL].dropna().unique().tolist())
        iso_options = [i for i in iso_all if i in iso_in_data]
    else:
        iso_options = iso_all

    if not iso_options:
        st.warning("No ISOs available for current filters.")
        st.stop()

    st.markdown("### ISO/RTO Selection")

    prior = st.session_state.get("iso_selected", [])
    sanitized_prior = [i for i in prior if i in iso_options]
    if not sanitized_prior:
        sanitized_prior = iso_options[:]

    select_all = st.checkbox("Select all ISO/RTOs", value=True, key="select_all_isos")
    current_default = iso_options[:] if select_all else sanitized_prior

    if hasattr(st, "pills"):
        selected_isos = st.pills(
            "Choose ISO/RTOs",
            options=iso_options,
            selection_mode="multi",
            default=current_default,
        )
    else:
        selected_isos = st.multiselect(
            "Choose ISO/RTOs",
            options=iso_options,
            default=current_default,
        )

    st.session_state["iso_selected"] = selected_isos

    transparent = st.checkbox("Transparent background (PNG)", value=True)
    show_states = st.checkbox("Show US state borders", value=True)

    st.markdown("### Price types")
    show_atc = st.checkbox("Show ATC Prices", value=True)
    show_solar = st.checkbox("Show Solar Prices", value=False)
    show_wind = st.checkbox("Show Wind Prices", value=False)

    st.markdown("### Label styling")
    label_fontsize = st.number_input(
        "Label font size",
        min_value=5.5,
        max_value=7.5,
        value=6.5,
        step=0.05,
        format="%.2f",
    )

    label_color = st.color_picker("Label font color", value="#000000")
    use_halo = st.checkbox("Add text halo (outline)", value=True)
    halo_color = st.color_picker("Halo color", value="#FFFFFF")

    labels_preview = pd.DataFrame()
    if fin_file is not None and selected_isos:
        labels_preview = prepare_labels(fin, selected_isos, developer)

    manual_offsets = {}
    replace_auto = False
    with st.expander("Label offsets (optional)"):
        st.caption("Values are pixels relative to each hub point. Positive dx right, positive dy up.")
        mode = st.radio(
            "Offset mode",
            ["Add to auto offsets", "Replace auto offsets"],
            index=0,
            horizontal=True,
        )
        replace_auto = mode == "Replace auto offsets"

        if not labels_preview.empty:
            for hub in sorted(labels_preview[HUB_COL].unique()):
                base_dx, base_dy = DEFAULT_OFFSETS.get(hub, (0, 0))
                c1, c2 = st.columns(2)
                dx = c1.number_input(f"{hub} dx", value=int(base_dx), step=1, format="%d")
                dy = c2.number_input(f"{hub} dy", value=int(base_dy), step=1, format="%d")
                if dx != base_dx or dy != base_dy:
                    manual_offsets[hub] = (dx, dy)
        else:
            st.write("Upload financials and choose ISOs to enable offset controls.")


if fin_file is None:
    st.info("Upload your Financials file to begin.")
    st.stop()

if not selected_isos:
    st.warning("Pick at least one ISO/RTO.")
    st.stop()


def _compose_offsets(user_offsets: dict) -> dict:
    out = DEFAULT_OFFSETS.copy()
    out.update(user_offsets or {})
    return out


labels = prepare_labels(fin, selected_isos, developer)
final_offsets = _compose_offsets(manual_offsets)

fig, png_buf = draw_map(
    labels=labels,
    g=g,
    selected_isos=selected_isos,
    manual_offsets=final_offsets,
    replace_auto=replace_auto,
    show_state_borders=show_states,
    label_fontsize=label_fontsize,
    use_halo=use_halo,
    halo_color=halo_color,
    label_font_family=label_font_family,
    font_style=font_style,
    font_weight=font_weight,
    label_color=label_color,
    show_atc=show_atc,
    show_solar=show_solar,
    show_wind=show_wind,
    transparent_bg=transparent,
)

if fig is not None:
    st.pyplot(fig, clear_figure=True)
    st.download_button(
        label="Download PNG (no background)",
        data=png_buf,
        file_name="price_map.png",
        mime="image/png",
    )
