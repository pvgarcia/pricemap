from io import BytesIO
from pathlib import Path
from itertools import count
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from shapely import make_valid
from pyproj import Transformer

# ---------------- Page setup ----------------
st.set_page_config(page_title="ISO Price Map", layout="wide")
st.title("ISO Price Map")

# ---------------- Constants ----------------
TARGET_CRS = "EPSG:3857"
ISO_COL   = "ISO/RTO"
HUB_COL   = "Settlement Hub"
PRICE_COL = "Price ($/MWh)"
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

FONT_FAMILY    = "Cambria"
FONT_STYLE     = "normal"
FONT_WEIGHT    = "normal"
HALO_EFFECT    = [withStroke(linewidth=1.6, foreground="white")]
BORDER_COLOR   = "#75838F87"
COUNTRY_LW     = 0.10
STATE_LW       = 0.10

OFFSET_WHEEL = [(0,12), (12,0), (-12,0), (0,-12), (10,10), (-10,10), (-10,-10), (10,-10), (16,6), (-16,6)]

# Hub proxy coordinates (lon, lat in EPSG:4326)
ISO_CONFIGS = [
    {"iso": "CAISO", "hub_proxy": {"NP15": (-121.4944, 38.5816), "SP15": (-118.2437, 34.0522)}},
    {"iso": "ERCOT", "hub_proxy": {
        "HB_NORTH": (-97.3308, 32.7555), "HB_HOUSTON": (-95.36327, 29.76328),
        "HB_SOUTH": (-98.4936, 29.4241), "HB_WEST": (-101.8552, 31.9973)
    }},
    {"iso": "MISO", "hub_proxy": {
        "INDIANA.HUB": (-86.1581, 39.7684), "ILLINOIS.HUB": (-89.6501, 39.7817),
        "MICHIGAN.HUB": (-84.5555, 42.7325), "MINN.HUB": (-93.2650, 44.9778),
        "TEXAS.HUB": (-94.1266, 30.0802), "ARKANSAS.HUB": (-92.2896, 34.7465),
        "LOUISIANA.HUB": (-91.1871, 30.4515)
    }},
    {"iso": "PJM", "hub_proxy": {
        "AD Hub": (-84.1917, 39.7589), "NI Hub": (-87.6298, 41.8781),
        "PEPCO": (-77.0369, 38.9072), "BGE": (-76.6122, 39.2904),
        "PSEG": (-74.1724, 40.7357), "JCPL": (-74.4830, 40.7968),
        "METED": (-75.9269, 40.3356), "PPL": (-75.4902, 40.6023),
        "COMED": (-87.6298, 41.8781), "DEOK": (-84.5120, 39.1031),
        "ATSI": (-81.6944, 41.4993), "DOMINION HUB": (-77.4360, 37.5407),
        "PECO": (-75.1652, 39.9526), "PENELEC": (-78.3947, 40.5187),
        "RECO": (-74.0324, 40.9263), "AEP-DAYTON HUB": (-84.1917, 39.7589),
        "APS": (-79.9559, 39.6295), "DPL": (-75.5467, 39.7447),
        "DUQ": (-80.0000, 40.4406), "EASTERN HUB": (-75.1652, 39.9526),
        "WESTERN HUB": (-80.0000, 40.4406), "AEP": (-82.9988, 39.9612)
    }},
    {"iso": "ISO-NE", "hub_proxy": {
        "Mass Hub": (-71.0589, 42.3601), ".Z.CONNECTICUT": (-72.6851, 41.7637),
        ".Z.MAINE": (-69.7795, 44.3106), ".Z.NEMASSBOST": (-71.0120, 42.3954),
        ".Z.NEWHAMPSHIRE": (-71.4548, 43.2070), ".Z.RHODEISLAND": (-71.4128, 41.8236),
        ".Z.SEMASS": (-70.9167, 41.6362), ".Z.VERMONT": (-72.5754, 44.2601),
        ".Z.WCMASS": (-72.5898, 42.1015)
    }},
    {"iso": "NYISO", "hub_proxy": {
        "West - A": (-78.8784, 42.8864), "Genesee - B": (-77.6156, 43.1566),
        "Central - C": (-76.1474, 43.0481), "North - D": (-75.9108, 44.6995),
        "Mohawk Valley - E": (-75.2327, 43.1009), "Capital - F": (-73.7562, 42.6526),
        "Hudson Valley - G": (-73.9235, 41.7004), "N.Y.C. - J": (-73.9857, 40.7484),
        "Long Island - K": (-73.1960, 40.7891)
    }},
    {"iso": "SPP", "hub_proxy": {
        "SPPNORTH_HUB": (-95.9980, 41.2524), "SPPSOUTH_HUB": (-97.5164, 35.4676)
    }},
]

# ---------------- Default manual offsets ----------------
DEFAULT_OFFSETS = {
    "HB_HOUSTON": (0, -20),      # dx, dy
    "MICHIGAN.HUB": (0, 20),
    "INDIANA.HUB": (0, -20),
    "AEP": (0, -5),
    "ILLINOIS.HUB": (0, -3),
    "WESTERN HUB": (5, 0),
    "AEP-DAYTON HUB": (-2, 0),
}

# ---------------- Nice chip styling for multiselect ----------------
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
def conus_limits():
    t = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    x0, y0 = t.transform(-125, 24)
    x1, y1 = t.transform(-66, 50)
    pad = 100_000
    return min(x0, x1) - pad, max(x0, x1) + pad, min(y0, y1) - pad, max(y0, y1) + pad

def load_geo() -> gpd.GeoDataFrame | None:
    if not MASTER_GEO.exists():
        st.warning(f"GeoJSON not found: {MASTER_GEO}.")
        return None
    g = gpd.read_file(MASTER_GEO)
    if "region" not in g.columns:
        st.warning("GeoJSON missing 'region' column; polygons will still draw but filtering by ISO uses that field.")
        g["region"] = None
    g["geometry"] = g["geometry"].apply(make_valid)
    return g.to_crs(TARGET_CRS)

def _normalize_tech_value(x: str) -> str | None:
    s = str(x).lower()
    if "wind" in s:
        return "Wind"
    if "solar" in s or "pv" in s or "photovoltaic" in s or ("bess" in s and ("solar" in s or "pv" in s)):
        return "Solar"
    return None

@st.cache_data(show_spinner=False)
def load_financials(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    if file.name.lower().endswith(".csv"):
        fin = pd.read_csv(file)
    else:
        fin = pd.read_excel(file)
    fin = fin.copy()
    # Normalize column names
    rename_map = {}
    for c in fin.columns:
        cl = c.strip().lower()
        if cl in {"iso", "rto", "iso/rto", "region"}: rename_map[c] = ISO_COL
        elif cl in {"hub", "settlement hub", "settlement_hub", "node", "settlement point", "settlement_point"}: rename_map[c] = HUB_COL
        elif cl in {"price ($/mwh)", "price", "lmp", "value", "price_mwh"}: rename_map[c] = PRICE_COL
        elif cl in {"developer", "dev", "sponsor"}: rename_map[c] = DEV_COL
        elif cl in {"technology", "tech", "resource", "fuel", "asset type", "asset_type"}: rename_map[c] = TECH_COL
    fin = fin.rename(columns=rename_map)
    missing = [c for c in [ISO_COL, HUB_COL, PRICE_COL] if c not in fin.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return pd.DataFrame()
    # Clean
    fin[HUB_COL] = fin[HUB_COL].astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.strip()
    fin[ISO_COL] = fin[ISO_COL].astype(str).str.strip()
    fin[PRICE_COL] = (
        fin[PRICE_COL].astype(str)
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

def _cluster_by_distance(pts_gdf, threshold_m=150000):
    coords = list(zip(pts_gdf.index, pts_gdf.geometry.x, pts_gdf.geometry.y))
    parent = {i:i for i,_,_ in coords}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i, xi, yi in coords:
        for j, xj, yj in coords:
            if j <= i:
                continue
            if ((xi-xj)**2 + (yi-yj)**2) ** 0.5 <= threshold_m:
                union(i, j)
    groups = {}
    for i,_,_ in coords:
        r = find(i)
        groups.setdefault(r, set()).add(i)
    return list(groups.values())

def compute_auto_offsets(pts_gdf, threshold_m=150000):
    offsets = {i:(0,0) for i in pts_gdf.index}
    for group in _cluster_by_distance(pts_gdf, threshold_m=threshold_m):
        if len(group) <= 1:
            continue
        cyc = count(0)
        for idx in group:
            k = next(cyc) % len(OFFSET_WHEEL)
            offsets[idx] = OFFSET_WHEEL[k]
    return offsets

def prepare_labels(fin: pd.DataFrame, selected_isos: list[str], developer: str | None, tech: str | None) -> pd.DataFrame:
    """Return merged labels table with lon/lat and P10 for selected ISOs and optional developer/tech filters."""
    fin = fin[fin[ISO_COL].isin(selected_isos)].copy()
    if developer and DEV_COL in fin.columns:
        fin = fin[fin[DEV_COL] == developer]
    if tech and TECH_COL in fin.columns:
        fin = fin[fin[TECH_COL] == tech]
    if fin.empty:
        return pd.DataFrame()
    hub_p10 = (
        fin.groupby([ISO_COL, HUB_COL], dropna=True)[PRICE_COL]
           .quantile(0.10, interpolation="linear")
           .reset_index()
           .rename(columns={PRICE_COL: "P10"})
    )
    cfg_map = {c["iso"]: c for c in ISO_CONFIGS}
    proxies = []
    for iso in selected_isos:
        if iso not in cfg_map:
            continue
        for hub, (lon, lat) in cfg_map[iso]["hub_proxy"].items():
            proxies.append({ISO_COL: iso, HUB_COL: hub, "lon": lon, "lat": lat})
    proxy_df = pd.DataFrame(proxies)
    return proxy_df.merge(hub_p10, on=[ISO_COL, HUB_COL], how="inner")

# ---------------- Plotting ----------------
def draw_map(labels: pd.DataFrame, g: gpd.GeoDataFrame | None, selected_isos: list[str],
             manual_offsets: dict | None,
             replace_auto: bool,
             show_state_borders: bool,
             label_fontsize: int,
             use_halo: bool,
             transparent_bg: bool = True):
    if labels.empty:
        st.info("No hubs match the current filters.")
        return None, None

    dpi = 300
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=dpi)

    # Optional borders
    if show_state_borders:
        try:
            from cartopy.io import shapereader as shp
            countries = gpd.read_file(shp.natural_earth("110m", "cultural", "admin_0_countries")).to_crs(TARGET_CRS)
            iso_col = next(c for c in ["ADM0_A3","ISO_A3","adm0_a3","iso_a3"] if c in countries.columns)
            usa = countries[countries[iso_col] == "USA"].copy()
            states = gpd.read_file(shp.natural_earth("10m", "cultural", "admin_1_states_provinces")).to_crs(TARGET_CRS)
            a3 = "adm0_a3" if "adm0_a3" in states.columns else "ADM0_A3"
            states = states[(states[a3] == "USA") & (~states["name"].isin({"Alaska","Hawaii","Puerto Rico"}))].copy()
            usa.boundary.plot(ax=ax, color=BORDER_COLOR, linewidth=COUNTRY_LW, zorder=1)
            states.boundary.plot(ax=ax, color=BORDER_COLOR, linewidth=STATE_LW, zorder=2)
        except Exception:
            pass

    # Territories
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
                zorder=5,
            )

    # Points & labels
    pts = gpd.GeoDataFrame(
        labels,
        geometry=gpd.points_from_xy(labels["lon"], labels["lat"], crs="EPSG:4326")
    ).to_crs(TARGET_CRS)

    auto = compute_auto_offsets(pts, threshold_m=150000)
    manual_offsets = manual_offsets or {}

    for ridx, r in pts.iterrows():
        iso = r[ISO_COL]
        hub = r[HUB_COL]
        val = r["P10"]
        x, y = r.geometry.x, r.geometry.y
        dx_auto, dy_auto = auto.get(ridx, (0,0))
        dx_man, dy_man   = manual_offsets.get(hub, DEFAULT_OFFSETS.get(hub, (0,0)))[0:2] if hub in manual_offsets else DEFAULT_OFFSETS.get(hub, (0,0))
        if replace_auto:
            dx, dy = dx_man, dy_man
        else:
            dx = dx_auto + dx_man
            dy = dy_auto + dy_man
        region_color = REGION_COLORS.get(iso, "#666666")
        txt = f"{hub}\n${val:.2f}"
        ax.annotate(
            txt, xy=(x,y), xycoords="data",
            xytext=(dx,dy), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=label_fontsize,
            fontfamily=FONT_FAMILY,
            fontstyle=FONT_STYLE,
            weight=FONT_WEIGHT,
            color="#000000",
            arrowprops=dict(arrowstyle="-", color=region_color, lw=0.9, shrinkA=0, shrinkB=2),
            path_effects=(HALO_EFFECT if use_halo else None),
            zorder=12,
        )

    # Legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=REGION_COLORS[i], edgecolor=REGION_COLORS[i], label=i, alpha=1.0)
               for i in selected_isos if i in REGION_COLORS]
    if handles:
        ax.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                  bbox_to_anchor=(0.5, -0.02), borderaxespad=0.6, fontsize=8)

    ax.set_aspect("equal")
    ax.set_axis_off()
    xmin, xmax, ymin, ymax = conus_limits()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.15, transparent=transparent_bg)
    buf.seek(0)
    return fig, buf

# ---------------- Streamlit UI ----------------
with st.sidebar:
    st.header("Inputs")
    fin_file = st.file_uploader("Financials (Excel/CSV)", type=["xlsx","xls","csv"], accept_multiple_files=False)
    fin = load_financials(fin_file)
    g = load_geo()

    # Developer FIRST (optional)
    developer = None
    if DEV_COL in fin.columns and not fin.empty:
        devs = sorted([d for d in fin[DEV_COL].dropna().unique().tolist() if str(d).strip()])
        dev_choice = st.selectbox("Developer filter (optional)", options=["All developers"] + devs, index=0)
        if dev_choice != "All developers":
            developer = dev_choice
    else:
        st.caption("Tip: Add a 'Developer' column to enable developer filtering.")

    # Technology SECOND (optional)
    tech = None
    tech_options_all = ["Solar", "Wind"]
    if TECH_COL in fin.columns and not fin.empty:
        fin_scoped = fin.copy()
        if developer:
            fin_scoped = fin_scoped[fin_scoped[DEV_COL] == developer]
        available_techs = sorted([t for t in fin_scoped[TECH_COL].dropna().unique().tolist() if t in tech_options_all])
        if available_techs:
            tech_choice = st.selectbox("Technology filter", options=["All technologies"] + available_techs, index=0)
            if tech_choice != "All technologies":
                tech = tech_choice
        else:
            st.selectbox("Technology filter", options=["No tech available for this developer"], index=0, disabled=True)
            tech = None
    else:
        st.caption("Tip: Add a 'Technology' column (or alias) to enable tech filtering.")

    # ---------------- ISO options reflect Developer & Technology ----------------
    iso_all = [c["iso"] for c in ISO_CONFIGS]
    if not fin.empty:
        fin_scoped = fin.copy()
        if developer:
            fin_scoped = fin_scoped[fin_scoped[DEV_COL] == developer]
        if tech and TECH_COL in fin_scoped.columns:
            fin_scoped = fin_scoped[fin_scoped[TECH_COL] == tech]
        iso_in_data = sorted(fin_scoped[ISO_COL].dropna().unique().tolist())
        iso_options = [i for i in iso_all if i in iso_in_data]
    else:
        iso_options = iso_all

    if not iso_options:
        st.warning("No ISOs available for the current Developer/Technology filters. Adjust filters to continue.")
        st.stop()

    st.markdown("### ISO/RTO Selection")

    # Restore a sane starting value for the multiselect
    prior = st.session_state.get("iso_multiselect", st.session_state.get("iso_selected", []))
    sanitized_prior = [i for i in prior if i in iso_options]
    if not sanitized_prior:
        sanitized_prior = iso_options[: min(3, len(iso_options))]

    # Select all checkbox
    select_all = st.checkbox("Select all ISO/RTOs", value=False, key="select_all_isos")

    # Set session_state BEFORE rendering the multiselect and DO NOT pass default
    if select_all:
        st.session_state["iso_multiselect"] = iso_options[:]         # all
    elif "iso_multiselect" not in st.session_state:
        st.session_state["iso_multiselect"] = sanitized_prior        # first render only

    # Multiselect reads session_state
    selected_isos = st.multiselect(
        "Choose ISO/RTOs",
        options=iso_options,
        key="iso_multiselect"
    )

    # Persist and legacy alias
    st.session_state["iso_selected"] = selected_isos
    selected = selected_isos

    # Options
    transparent = st.checkbox("Transparent background (PNG)", value=True)
    show_states = st.checkbox("Show US state borders", value=True)

    # Label styling in sidebar
    st.markdown("### Label styling")
    label_fontsize = st.slider("Label font size", min_value=6, max_value=24, value=10, step=1)
    use_halo = st.checkbox("Add text halo (white outline)", value=True)

    # Prepare labels early so we can show offset controls
    labels_preview = pd.DataFrame()
    if fin_file is not None and selected_isos:
        labels_preview = prepare_labels(fin, selected_isos, developer, tech)

    # Manual offsets start with your defaults; you can override per hub
    manual_offsets = {}
    replace_auto = False
    with st.expander("Label offsets (optional)"):
        st.caption("Tip: Values are pixels relative to each hub point. Positive dx → right, positive dy → up.")
        mode = st.radio("Offset mode", ["Add to auto offsets", "Replace auto offsets"], index=0, horizontal=True)
        replace_auto = (mode == "Replace auto offsets")
        if not labels_preview.empty:
            for hub in sorted(labels_preview[HUB_COL].unique()):
                base_dx, base_dy = DEFAULT_OFFSETS.get(hub, (0,0))
                c1, c2 = st.columns(2)
                dx = c1.number_input(f"{hub} dx", value=int(base_dx), step=1, format="%d")
                dy = c2.number_input(f"{hub} dy", value=int(base_dy), step=1, format="%d")
                if dx != base_dx or dy != base_dy:
                    manual_offsets[hub] = (dx, dy)
        else:
            st.write("Upload financials, choose ISOs, and (optionally) developer/technology to enable offset controls.")

# Guardrails
if fin_file is None:
    st.info("Upload your Financials file to begin.")
    st.stop()

if not selected_isos:
    st.warning("Pick at least one ISO/RTO.")
    st.stop()

# Merge defaults + user overrides for the draw step
def _compose_offsets(user_offsets: dict) -> dict:
    out = DEFAULT_OFFSETS.copy()
    out.update(user_offsets or {})
    return out

# Final labels for drawing
labels = prepare_labels(fin, selected_isos, developer, tech)
final_offsets = _compose_offsets(manual_offsets)
fig, png_buf = draw_map(labels, g, selected_isos,
                        manual_offsets=final_offsets,
                        replace_auto=replace_auto,
                        show_state_borders=show_states,
                        label_fontsize=label_fontsize,
                        use_halo=use_halo,
                        transparent_bg=transparent)

if fig is not None:
    st.pyplot(fig, clear_figure=True)
    st.download_button(
        label="Download PNG (no background)",
        data=png_buf,
        file_name="price_map_P10.png",
        mime="image/png",
    )