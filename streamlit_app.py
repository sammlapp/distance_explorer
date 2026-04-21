import os
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def color_to_rgba_alpha(color, alpha=0.18):
    """Convert hex/rgb/rgba color strings to rgba with the requested alpha."""
    if isinstance(color, str):
        c = color.strip()
        if c.startswith("#") and len(c) == 7:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"
        if c.startswith("rgb(") and c.endswith(")"):
            nums = [v.strip() for v in c[4:-1].split(",")]
            if len(nums) == 3:
                return f"rgba({nums[0]},{nums[1]},{nums[2]},{alpha})"
        if c.startswith("rgba(") and c.endswith(")"):
            nums = [v.strip() for v in c[5:-1].split(",")]
            if len(nums) >= 3:
                return f"rgba({nums[0]},{nums[1]},{nums[2]},{alpha})"
    return f"rgba(0,0,0,{alpha})"


def inside_legend(x=0.8, y=0.98):
    return dict(
        x=x,
        y=y,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="rgba(0,0,0,0.12)",
        borderwidth=1,
        orientation="v",
    )


# ============================================================================
# Configuration & Constants
# ============================================================================

# Load species data from CSV
SPECIES_CSV_PATH = os.path.join(os.path.dirname(__file__), "species_summary.csv")
SPECIES_DF = pd.read_csv(SPECIES_CSV_PATH)
SPECIES_INFO = {
    row["Species"]: {
        "source_level": row["mean_source_level"],
        "singing_height": row["singing_height"],
        "frequency": row["narrowband_frequency"],
    }
    for _, row in SPECIES_DF.iterrows()
}
SPECIES_LIST = sorted(list(SPECIES_INFO.keys()))


# ============================================================================
# Core Functions (from notebook)
# ============================================================================


def atmospheric_att_coef_dB(f, t=20, rh=60, pa=101325):
    """Calculate atmospheric attenuation coefficient in dB/m"""
    f = np.asarray(f)
    if hasattr(t, "__len__"):
        raise TypeError("t must be a scalar")
    if hasattr(rh, "__len__"):
        raise TypeError("rh must be a scalar")
    if hasattr(pa, "__len__"):
        raise TypeError("pa must be a scalar")

    pr = 101.325e3
    To1 = 273.16
    To = 293.15
    t = t + 273.15

    psat = pr * 10 ** (-6.8346 * (To1 / t) ** 1.261 + 4.6151)
    h = rh * (psat / pa)
    frO = (pa / pr) * (24 + 4.04e4 * h * ((0.02 + h) / (0.391 + h)))
    frN = (
        (pa / pr)
        * (t / To) ** (-0.5)
        * (9 + 280 * h * np.exp(-4.170 * ((t / To) ** (-1 / 3) - 1)))
    )

    z = 0.1068 * np.exp(-3352 / t) / (frN + f**2 / frN)
    y = (t / To) ** (-5 / 2) * (
        0.01275 * np.exp(-2239.1 / t) * 1 / (frO + f**2 / frO) + z
    )
    Aatm_coef_dB = 8.686 * f**2 * ((1.84e-11 * 1 / (pa / pr) * np.sqrt(t / To)) + y)

    return Aatm_coef_dB


def simulate_level_mechanistic(
    horizontal_distance,
    source_level_mean,
    song_rl_sd,
    offset,
    frequency,
    hab_atten_per_kHz_100m,
    atm_atten_per_m,
    height,
    residual_sd,
    N=1000,
):
    """Simulate received levels across distances using mechanistic model"""
    heights = height * np.ones_like(horizontal_distance)
    ds = np.sqrt(horizontal_distance**2 + heights**2)
    ds = np.clip(ds, 1, None)

    expectation = (
        source_level_mean
        + offset
        - 20 * np.log10(ds)
        - atm_atten_per_m * ds
        - hab_atten_per_kHz_100m * (ds / 100) * frequency / 1000
    )
    expectation += np.random.normal(0, song_rl_sd, size=N) + np.random.normal(
        0, residual_sd, size=N
    )
    return expectation


@lru_cache(maxsize=32)
def simulate_levels_across_distances(
    source_level_mean,
    song_rl_sd,
    offset,
    frequency,
    hab_atten_per_kHz_100m,
    atm_atten_per_m,
    height,
    residual_sd,
    max_distance=600,
    N_per_bin=1000,
    bin_width=2,
):
    """Simulate levels across a range of distances"""
    distance_bins = np.arange(0, max_distance, bin_width)
    distance_bin_levels = []

    for bin_edge in distance_bins:
        distance_bin_levels.append(
            simulate_level_mechanistic(
                horizontal_distance=bin_edge,
                source_level_mean=source_level_mean,
                song_rl_sd=song_rl_sd,
                offset=offset,
                frequency=frequency,
                hab_atten_per_kHz_100m=hab_atten_per_kHz_100m,
                atm_atten_per_m=atm_atten_per_m,
                height=height,
                residual_sd=residual_sd,
                N=N_per_bin,
            )
        )

    all_levels = np.concatenate(distance_bin_levels)
    all_distances = np.repeat(distance_bins, [len(x) for x in distance_bin_levels])

    return distance_bins, distance_bin_levels, all_distances, all_levels


def calculate_expected_received_level(
    distance,
    source_level_mean,
    offset,
    frequency,
    hab_atten_per_kHz_100m,
    atm_atten_per_m,
    singing_height,
):
    """Return the deterministic expected received level at a given horizontal distance."""
    distance_3d = np.sqrt(np.asarray(distance) ** 2 + singing_height**2)
    distance_3d = np.clip(distance_3d, 1, None)
    return (
        source_level_mean
        + offset
        - 20 * np.log10(distance_3d)
        - atm_atten_per_m * distance_3d
        - hab_atten_per_kHz_100m * (distance_3d / 100) * frequency / 1000
    )


def measure_mean_and_pct(
    simulated_levels, simulated_distances, levels, tolerance=5, pcts=None
):
    """Measure mean distance and percentiles for given level thresholds"""
    if pcts is None:
        pcts = [2.5, 5, 10, 90, 95, 97.5]

    expectation_distances = []
    percentiles_list = []

    for target_level in levels:
        mask = np.abs(simulated_levels - target_level) < tolerance
        distances_at_level = simulated_distances[mask]

        if len(distances_at_level) > 0:
            expectation_distances.append(np.mean(distances_at_level))
            percentiles_list.append(np.percentile(distances_at_level, pcts))
        else:
            expectation_distances.append(np.nan)
            percentiles_list.append(np.full(len(pcts), np.nan))

    return np.array(expectation_distances), np.array(percentiles_list)


def simulate_and_evaluate_truncation(
    source_level_mean,
    song_rl_sd,
    offset,
    frequency,
    hab_atten_per_kHz_100m,
    atm_atten_per_m,
    height,
    residual_sd,
    truncation_distances,
    max_distance=600,
    N_per_bin=1000,
):
    """Simulate and evaluate truncation classifier performance"""
    _, _, all_distances, all_levels = simulate_levels_across_distances(
        source_level_mean=source_level_mean,
        song_rl_sd=song_rl_sd,
        offset=offset,
        frequency=frequency,
        hab_atten_per_kHz_100m=hab_atten_per_kHz_100m,
        atm_atten_per_m=atm_atten_per_m,
        height=height,
        residual_sd=residual_sd,
        max_distance=max_distance,
        N_per_bin=N_per_bin,
        bin_width=4,
    )

    # Estimate distance|level curve
    levels = np.linspace(all_levels.min(), all_levels.max(), 40)
    expectation_distances, _ = measure_mean_and_pct(
        simulated_levels=all_levels,
        simulated_distances=all_distances,
        levels=levels,
        tolerance=5,
    )

    # Filter to finite values, sort, and deduplicate
    valid_mask = np.isfinite(expectation_distances)
    valid_distances = expectation_distances[valid_mask]
    valid_levels = levels[valid_mask]

    sort_idx = np.argsort(valid_distances)
    valid_distances = valid_distances[sort_idx]
    valid_levels = valid_levels[sort_idx]

    # Remove duplicates
    unique_distances, unique_indices = np.unique(valid_distances, return_index=True)
    unique_levels = valid_levels[unique_indices]

    def find_truncation_level(target_distance):
        """Find the nearest level for a given target distance"""
        idx = np.argmin(np.abs(unique_distances - target_distance))
        return unique_levels[idx]

    metrics = []
    for target_dist in truncation_distances:
        truncation_level = find_truncation_level(target_dist)

        # Compute confusion matrix
        tp = np.sum((all_distances <= target_dist) & (all_levels >= truncation_level))
        fn = np.sum((all_distances <= target_dist) & (all_levels < truncation_level))
        fp = np.sum((all_distances > target_dist) & (all_levels >= truncation_level))
        tn = np.sum((all_distances > target_dist) & (all_levels < truncation_level))

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        type_i_error = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        metrics.append(
            {
                "truncation_distance": target_dist,
                "truncation_level": truncation_level,
                "precision": precision,
                "recall": recall,
                "type_i_error": type_i_error,
            }
        )

    return pd.DataFrame(metrics)


def evaluate_truncation_from_simulation(
    all_distances,
    all_levels,
    truncation_distances,
):
    """Evaluate truncation metrics from pre-simulated distance/level data."""
    valid_level_mask = all_levels >= 30
    all_distances = all_distances[valid_level_mask]
    all_levels = all_levels[valid_level_mask]

    levels = np.linspace(all_levels.min(), all_levels.max(), 100)
    expectation_distances, _ = measure_mean_and_pct(
        simulated_levels=all_levels,
        simulated_distances=all_distances,
        levels=levels,
        tolerance=2,
    )

    valid_mask = np.isfinite(expectation_distances)
    valid_distances = expectation_distances[valid_mask]
    valid_levels = levels[valid_mask]

    sort_idx = np.argsort(valid_distances)
    valid_distances = valid_distances[sort_idx]
    valid_levels = valid_levels[sort_idx]

    unique_distances, unique_indices = np.unique(valid_distances, return_index=True)
    unique_levels = valid_levels[unique_indices]

    def find_truncation_level(target_distance):
        idx = np.argmin(np.abs(unique_distances - target_distance))
        return unique_levels[idx]

    metrics = []
    for target_dist in truncation_distances:
        truncation_level = find_truncation_level(target_dist)
        tp = np.sum((all_distances <= target_dist) & (all_levels >= truncation_level))
        fn = np.sum((all_distances <= target_dist) & (all_levels < truncation_level))
        fp = np.sum((all_distances > target_dist) & (all_levels >= truncation_level))
        tn = np.sum((all_distances > target_dist) & (all_levels < truncation_level))

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        type_i_error = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        metrics.append(
            {
                "truncation_distance": target_dist,
                "truncation_level": truncation_level,
                "precision": precision,
                "recall": recall,
                "type_i_error": type_i_error,
            }
        )

    return pd.DataFrame(metrics)


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def build_simulation_bundle(
    frequency,
    hab_attens,
    source_level,
    song_rl_sd,
    offset,
    temp_C,
    rel_humidity,
    pressure_atm,
    singing_height,
    residual_sd,
    max_distance,
    N_per_bin,
):
    """Run simulations once per habitat attenuation and reuse across charts."""
    bundle = {}
    pressure_pa = 101325 * pressure_atm
    atm_atten = atmospheric_att_coef_dB(
        frequency,
        temp_C,
        rel_humidity,
        pressure_pa,
    )
    for hab_atten in hab_attens:
        _, _, all_distances, all_levels = simulate_levels_across_distances(
            source_level_mean=source_level,
            song_rl_sd=song_rl_sd,
            offset=offset,
            frequency=frequency,
            hab_atten_per_kHz_100m=hab_atten,
            atm_atten_per_m=atm_atten,
            height=singing_height,
            residual_sd=residual_sd,
            max_distance=max_distance,
            N_per_bin=N_per_bin,
            bin_width=4,
        )

        levels = np.linspace(all_levels.min(), all_levels.max(), 40)
        exp_dists, percentiles = measure_mean_and_pct(
            all_levels,
            all_distances,
            levels,
            tolerance=5,
            pcts=[2.5, 5, 10, 90, 95, 97.5],
        )
        bundle[hab_atten] = {
            "all_distances": all_distances,
            "all_levels": all_levels,
            "levels": levels,
            "exp_dists": exp_dists,
            "percentiles": percentiles,
        }

    return bundle


# ============================================================================
# Streamlit App
# ============================================================================


st.set_page_config(layout="wide")  # Optional: uses more screen width

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            margin-top: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)

hide_main_content_style = """
                <style>
                .st-emotion-cache-1jicfl2{
                     padding: 0rem 1rem 10rem;
                }
                </style>
                """
st.markdown(hide_main_content_style, unsafe_allow_html=True)

st.set_page_config(page_title="Bird Detection Parameter Explorer", layout="wide")

st.markdown(
    """
    <style>
    @media (max-width: 1100px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        div[data-testid="stHorizontalBlock"] > div,
        div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
            flex-basis: 100% !important;
            max-width: 100% !important;
            display: block !important;
        }
        div[data-testid="column"] > div {
            width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "use_species" not in st.session_state:
    st.session_state.use_species = True

# ============================================================================
# SETTINGS PANEL (Left Sidebar)
# ============================================================================

with st.sidebar:
    with st.expander("Info/Instructions", expanded=False):
        st.markdown(
            """
    This app explores how received level, uncertainty, attenuation, and truncation behavior change with species/acoustic settings and background noise.

    Throughout, source level, received level, and background level measurments are _narrow-band_ values (1kHz bandwidth) in dB SPL re 20 µPa. Distance values are for horizontal distance. Shaded areas show +/-1 SD from the mean across simulated data.

    **Plots**
    - **Detection Distance**: Expected detection distance versus selected background noise levels, with +/-1 SD ribbons from simulation.
    - **Distance Estimate CI Width**: CI width of distance estimates versus estimated distance, using your selected CI level (80/90/95%).
    - **Received Level vs Distance**: Mean simulated received level versus distance with +/-1 SD band, censored where expected level falls below the selected noise floor.
    - **Atmospheric + Habitat Attenuation**: Additive attenuation components per 100 m across frequency. Atmospheric attenuation is the base layer; habitat attenuation is shown as stacked increments for each selected habitat value.
    - **Truncation Performance**: Precision and recall versus truncation distance for the selected truncation habitat attenuation.

    **How Controls Map to Panels**
    - **Use species defaults** locks frequency, source level, and singing height to the chosen species.
    - Turning species defaults off enables manual frequency/source level (and singing height) control.
    - **Habitat Attenuation for Detection/CI** drives Detection, CI Width, Received Level, and Attenuation panels.
    - **Habitat Attenuation for Truncation** drives only the Truncation panel.
    - **Narrow-band Background Noise Levels** controls noise values used in Detection and the censoring threshold for Received Level.
    - **Atmospheric Conditions** (temperature, RH, pressure in atm) are used to calculate atmospheric attenuation (in combination with frequency).
    - **Advanced Settings** control song source level variance, residual received level variance, and region offset.

    The mechanistic model follows Lapp, Kitzes et al. (in prep). Expand each table under a plot to inspect values and download the displayed data.
    """
        )

    download_full_sim_placeholder = st.empty()

    st.header("Settings")

    # === Main Settings ===
    st.subheader("Main Settings")

    use_species = st.toggle("Use species defaults", value=st.session_state.use_species)
    st.session_state.use_species = use_species

    if use_species:
        species = st.selectbox("Species", options=SPECIES_LIST)
        source_level = SPECIES_INFO[species]["source_level"]
        base_frequency = int(round(SPECIES_INFO[species]["frequency"]))
        singing_height = SPECIES_INFO[species]["singing_height"]
        st.number_input(
            "Frequency (Hz)",
            min_value=100,
            max_value=16000,
            value=base_frequency,
            step=100,
            disabled=True,
            help="Locked to selected species narrow-band frequency.",
        )
    else:
        source_level = st.slider(
            "Mean Narrow-band Source Level (dB SPL)",
            min_value=50,
            max_value=110,
            value=90,
            step=1,
        )
        base_frequency = st.slider(
            "Frequency (Hz)",
            min_value=100,
            max_value=16000,
            value=5000,
            step=100,
        )
        singing_height = st.slider(
            "Singing Height (m)",
            min_value=0.5,
            max_value=40.0,
            value=5.0,
            step=0.5,
        )

    st.subheader("Atmospheric Conditions")
    temp_C = st.slider("Temperature (°C)", min_value=-20, max_value=40, value=20)
    rel_humidity = st.slider(
        "Relative Humidity (%)", min_value=10, max_value=100, value=60
    )
    pressure_atm = st.number_input(
        "Pressure (atm)", value=1.0, step=0.01, format="%.2f"
    )

    # === Advanced Settings ===
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            song_rl_sd = st.slider(
                "Song RL SD (dB)", min_value=1.0, max_value=10.0, value=3.0, step=0.1
            )
            residual_sd = st.slider(
                "Residual SD (dB)", min_value=1.0, max_value=10.0, value=4.2, step=0.1
            )

        with col2:
            offset = st.slider(
                "Region Offset (dB)",
                min_value=-10.0,
                max_value=5.0,
                value=-7.0,
                step=0.5,
            )

        max_distance = st.slider(
            "Max Distance for Plots (m)",
            min_value=50,
            max_value=1000,
            value=200,
            step=25,
            help="Simulations run to 3x this distance; detection chart is limited to this value.",
        )
        N_per_bin = st.selectbox(
            "Simulations per bin", options=[500, 1000, 2000], index=1
        )

    st.markdown("---")
    st.subheader("Chart Controls")
    hab_atten_options = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    hab_attens_for_plots = st.multiselect(
        "Habitat Attenuation for Detection/CI (dB/kHz/100m)",
        options=hab_atten_options,
        default=[0, 2],
        key="hab_multi",
    )
    truncation_hab_atten = st.selectbox(
        "Habitat Attenuation for Truncation",
        options=hab_atten_options,
        index=1,
        key="hab_trunc",
    )
    noise_range = st.slider(
        "Narrow-band Background Noise Levels (dB SPL)",
        min_value=10,
        max_value=60,
        value=(25, 45),
        step=5,
        key="dd_noise",
    )
    noise_levels = list(range(noise_range[0], noise_range[1] + 1, 5))
    if use_species:
        st.caption(f"Frequency locked to species: {int(base_frequency)} Hz")

    ci_pct = st.selectbox("CI Level", options=[80, 90, 95], index=2, key="ci_pct")
    truncation_distances = st.multiselect(
        "Truncation Distances (m)",
        options=list(range(25, 351, 5)),
        default=[25, 50, 100, 150],
        key="trunc_dist",
    )


# ============================================================================
# MAIN CONTENT - SINGLE-SCREEN DASHBOARD
# ============================================================================

selected_frequency = int(base_frequency)
requested_hab_attens = sorted(set(hab_attens_for_plots + [truncation_hab_atten]))

if not hab_attens_for_plots:
    download_full_sim_placeholder.info(
        "Select at least one habitat attenuation value to enable full simulated data download."
    )
    st.warning("Select at least one habitat attenuation value for Detection/CI.")
else:
    simulation_max_distance = 3 * max_distance

    with st.spinner("Running shared simulation bundle..."):
        simulation_bundle = build_simulation_bundle(
            frequency=selected_frequency,
            hab_attens=tuple(requested_hab_attens),
            source_level=source_level,
            song_rl_sd=song_rl_sd,
            offset=offset,
            temp_C=temp_C,
            rel_humidity=rel_humidity,
            pressure_atm=pressure_atm,
            singing_height=singing_height,
            residual_sd=residual_sd,
            max_distance=simulation_max_distance,
            N_per_bin=N_per_bin,
        )

    full_simulation_data = []
    for hab_atten in sorted(simulation_bundle.keys()):
        sim = simulation_bundle[hab_atten]
        full_simulation_data.append(
            pd.DataFrame(
                {
                    "Habitat Attenuation (dB/kHz/100m)": hab_atten,
                    "Frequency (Hz)": selected_frequency,
                    "Distance (m)": sim["all_distances"],
                    "Received Level (dB SPL)": sim["all_levels"],
                }
            )
        )

    if full_simulation_data:
        full_sim_df = pd.concat(full_simulation_data, ignore_index=True)
        download_full_sim_placeholder.download_button(
            label="Download Full Simulated Data (CSV)",
            data=dataframe_to_csv_bytes(full_sim_df),
            file_name="full_simulated_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    col_detect, col_level, col_ci = st.columns(3)

    with col_detect:
        if hab_attens_for_plots and noise_levels:
            detection_tol = 1
            detection_stats = []
            for hab_atten in sorted(hab_attens_for_plots):
                sim = simulation_bundle[hab_atten]
                all_levels = sim["all_levels"]
                all_distances = sim["all_distances"]

                for noise_level in noise_levels:
                    distances_at_level = all_distances[
                        np.abs(all_levels - noise_level) < detection_tol
                    ]
                    if len(distances_at_level) == 0:
                        continue

                    mean_distance = float(np.mean(distances_at_level))
                    sd_distance = float(np.std(distances_at_level, ddof=0))
                    detection_stats.append(
                        {
                            "Habitat Attenuation": hab_atten,
                            "Habitat Label": f"A_hab={hab_atten}",
                            "Noise Level (dB SPL)": noise_level,
                            "Mean Detection Distance (m)": mean_distance,
                            "Lower SD": max(0.0, mean_distance - sd_distance),
                            "Upper SD": mean_distance + sd_distance,
                        }
                    )

            if detection_stats:
                df_detect = pd.DataFrame(detection_stats)
                fig_detect = go.Figure()
                hab_labels = sorted(df_detect["Habitat Label"].unique())
                palette = px.colors.qualitative.Set2

                for i, hab_label in enumerate(hab_labels):
                    color = palette[i % len(palette)]
                    shade_color = color_to_rgba_alpha(color, alpha=0.18)
                    subset = df_detect[
                        df_detect["Habitat Label"] == hab_label
                    ].sort_values("Noise Level (dB SPL)")

                    # Lower bound (invisible line) for shaded SD band.
                    fig_detect.add_trace(
                        go.Scatter(
                            x=subset["Noise Level (dB SPL)"],
                            y=subset["Lower SD"],
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                    # Upper bound filled to previous trace to create +/-1 SD ribbon.
                    fig_detect.add_trace(
                        go.Scatter(
                            x=subset["Noise Level (dB SPL)"],
                            y=subset["Upper SD"],
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor=shade_color,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                    # Mean line on top of shaded band.
                    fig_detect.add_trace(
                        go.Scatter(
                            x=subset["Noise Level (dB SPL)"],
                            y=subset["Mean Detection Distance (m)"],
                            mode="lines+markers",
                            name=hab_label,
                            line=dict(color=color, width=2),
                            marker=dict(size=7),
                            customdata=np.stack(
                                [subset["Lower SD"], subset["Upper SD"]], axis=-1
                            ),
                            hovertemplate=(
                                "Noise level: %{x:.0f} dB SPL<br>"
                                "Mean distance: %{y:.1f} m<br>"
                                "-1 SD: %{customdata[0]:.1f} m<br>"
                                "+1 SD: %{customdata[1]:.1f} m<extra>"
                                + hab_label
                                + "</extra>"
                            ),
                        )
                    )

                fig_detect.update_layout(
                    hovermode="x unified",
                    height=340,
                    margin=dict(l=10, r=10, t=10, b=10),
                    template="plotly_white",
                    xaxis_title="Background Noise Level (dB SPL)",
                    yaxis_title="Detection Distance (m)",
                    legend=inside_legend(),
                    showlegend=True,
                )
                st.plotly_chart(fig_detect, use_container_width=True)

                with st.expander("Detection table"):
                    st.dataframe(
                        df_detect.sort_values(
                            ["Habitat Attenuation", "Noise Level (dB SPL)"]
                        ).round(
                            {
                                "Habitat Attenuation": 1,
                                "Noise Level (dB SPL)": 0,
                                "Mean Detection Distance (m)": 2,
                                "Lower SD": 2,
                                "Upper SD": 2,
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    with col_level:
        noise_floor = min(noise_levels) if noise_levels else 0
        level_bin_width = 10
        fig_level = go.Figure()
        level_tables = []
        pressure_pa = 101325 * pressure_atm

        for i, level_hab_atten in enumerate(sorted(hab_attens_for_plots)):
            sim = simulation_bundle[level_hab_atten]
            atm_atten = atmospheric_att_coef_dB(
                selected_frequency,
                temp_C,
                rel_humidity,
                pressure_pa,
            )
            distance_grid = np.arange(0, simulation_max_distance + 1, 1)
            expected_curve = calculate_expected_received_level(
                distance=distance_grid,
                source_level_mean=source_level,
                offset=offset,
                frequency=selected_frequency,
                hab_atten_per_kHz_100m=level_hab_atten,
                atm_atten_per_m=atm_atten,
                singing_height=singing_height,
            )
            above_noise = distance_grid[expected_curve >= noise_floor]
            noise_cutoff_distance = int(above_noise.max()) if len(above_noise) else 0

            level_df = pd.DataFrame(
                {
                    "Distance (m)": sim["all_distances"],
                    "Received Level (dB SPL)": sim["all_levels"],
                }
            )
            level_df = level_df[level_df["Distance (m)"] <= noise_cutoff_distance]
            level_df["Distance Bin (m)"] = (
                np.floor(level_df["Distance (m)"] / level_bin_width) * level_bin_width
            ).astype(int)
            level_summary = (
                level_df.groupby("Distance Bin (m)", as_index=False)
                .agg(
                    Mean_Level=("Received Level (dB SPL)", "mean"),
                    Lower_SD=(
                        "Received Level (dB SPL)",
                        lambda s: s.mean() - s.std(ddof=0),
                    ),
                    Upper_SD=(
                        "Received Level (dB SPL)",
                        lambda s: s.mean() + s.std(ddof=0),
                    ),
                )
                .sort_values("Distance Bin (m)")
            )
            level_summary["Habitat"] = f"A_hab={level_hab_atten}"
            level_tables.append(level_summary)

            color = px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
            fig_level.add_trace(
                go.Scatter(
                    x=level_summary["Distance Bin (m)"],
                    y=level_summary["Lower_SD"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig_level.add_trace(
                go.Scatter(
                    x=level_summary["Distance Bin (m)"],
                    y=level_summary["Upper_SD"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=color_to_rgba_alpha(color, alpha=0.18),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig_level.add_trace(
                go.Scatter(
                    x=level_summary["Distance Bin (m)"],
                    y=level_summary["Mean_Level"],
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    name=f"A_hab={level_hab_atten}",
                    hovertemplate="Distance: %{x:.0f} m<br>Mean level: %{y:.1f} dB SPL<extra>A_hab="
                    + str(level_hab_atten)
                    + "</extra>",
                )
            )

        level_df_all = (
            pd.concat(level_tables, ignore_index=True)
            if level_tables
            else pd.DataFrame()
        )
        fig_level.update_layout(
            hovermode="x unified",
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white",
            xaxis_title="Distance (m)",
            yaxis_title="Received Level (dB SPL)",
            legend=inside_legend(),
            showlegend=True,
        )
        st.plotly_chart(fig_level, use_container_width=True)

        with st.expander("Received level table"):
            st.dataframe(
                level_df_all.style.format(
                    {
                        "Distance Bin (m)": "{:.0f}",
                        "Mean_Level": "{:.2f}",
                        "Lower_SD": "{:.2f}",
                        "Upper_SD": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with col_ci:
        if hab_attens_for_plots:
            ci_rows = []
            for hab_atten in sorted(hab_attens_for_plots):
                sim = simulation_bundle[hab_atten]
                levels = sim["levels"]
                exp_dists = sim["exp_dists"]
                percentiles = sim["percentiles"]

                if ci_pct == 95:
                    ci_width = percentiles[:, -1] - percentiles[:, 0]
                elif ci_pct == 90:
                    ci_width = percentiles[:, -2] - percentiles[:, 1]
                else:
                    ci_width = percentiles[:, -3] - percentiles[:, 2]

                for received_level, dist, width in zip(levels, exp_dists, ci_width):
                    if (
                        received_level >= 30
                        and np.isfinite(received_level)
                        and np.isfinite(dist)
                        and np.isfinite(width)
                        and dist <= max_distance
                    ):
                        ci_rows.append(
                            {
                                "Distance Estimate (m)": dist,
                                f"Distance Estimate {ci_pct}% CI Width (m)": width,
                                "Habitat": f"A_hab={hab_atten}",
                            }
                        )

            if ci_rows:
                df_ci = pd.DataFrame(ci_rows)
                fig_ci = px.line(
                    df_ci,
                    x="Distance Estimate (m)",
                    y=f"Distance Estimate {ci_pct}% CI Width (m)",
                    color="Habitat",
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_ci.update_traces(marker=dict(size=7))
                fig_ci.update_layout(
                    hovermode="x unified",
                    height=340,
                    margin=dict(l=10, r=10, t=10, b=10),
                    template="plotly_white",
                    legend=inside_legend(x=0.3),
                    showlegend=True,
                )
                st.plotly_chart(fig_ci, use_container_width=True)

                with st.expander("CI table"):
                    st.dataframe(
                        df_ci.sort_values(["Habitat", "Distance Estimate (m)"]).round(
                            {
                                "Distance Estimate (m)": 2,
                                "CI Width (m)": 2,
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    col_atten, col_trunc, _ = st.columns(3)

    with col_atten:
        attenuation_freqs = np.arange(50, 12001, 50)
        atm_atten_per_m_curve = atmospheric_att_coef_dB(
            attenuation_freqs,
            temp_C,
            rel_humidity,
            pressure_pa,
        )
        atm_atten_per_100m = atm_atten_per_m_curve * 100
        selected_hab_attens = sorted(hab_attens_for_plots)
        fig_atten = go.Figure()
        fig_atten.add_trace(
            go.Scatter(
                x=attenuation_freqs,
                y=atm_atten_per_100m,
                mode="lines",
                name="Atmospheric attenuation",
                line=dict(color=px.colors.qualitative.Set2[0], width=2),
                stackgroup="atten",
                hovertemplate="Frequency: %{x:.0f} Hz<br>Atmospheric: %{y:.2f} dB/100m<extra>Atmospheric attenuation</extra>",
            )
        )

        previous_hab = 0.0
        base_hab_color = px.colors.qualitative.Set2[1]
        for i, hab_atten in enumerate(selected_hab_attens):
            increment_hab = hab_atten - previous_hab
            if increment_hab <= 0:
                previous_hab = hab_atten
                continue

            increment_curve = increment_hab * attenuation_freqs / 1000
            fig_atten.add_trace(
                go.Scatter(
                    x=attenuation_freqs,
                    y=increment_curve,
                    mode="lines",
                    name=f"A_hab={hab_atten} dB/kHz/100m",
                    line=dict(color=base_hab_color, width=1),
                    stackgroup="atten",
                    fillcolor=color_to_rgba_alpha(
                        base_hab_color, alpha=0.18 + 0.12 * i
                    ),
                    hovertemplate=(
                        "Frequency: %{x:.0f} Hz<br>Increment: %{y:.2f} dB/100m<extra>A_hab="
                        + str(hab_atten)
                        + " dB/kHz/100m</extra>"
                    ),
                )
            )
            previous_hab = hab_atten

        fig_atten.update_layout(
            hovermode="x unified",
            height=340,
            margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_white",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Attenuation (dB per 100 m)",
            legend=inside_legend(),
            showlegend=True,
        )
        fig_atten.update_layout(
            legend_title_text="",
        )
        st.plotly_chart(fig_atten, use_container_width=True)

        if selected_hab_attens:
            atten_table = pd.DataFrame(
                {
                    "Frequency (Hz)": attenuation_freqs,
                    "Atmospheric attenuation (dB/100m)": atm_atten_per_100m,
                }
            )
            for hab_atten in selected_hab_attens:
                atten_table[f"A_hab={hab_atten} (dB/100m)"] = (
                    hab_atten * attenuation_freqs / 1000
                )

            with st.expander("Attenuation table"):
                st.dataframe(
                    atten_table.round(2),
                    use_container_width=True,
                    hide_index=True,
                )

    with col_trunc:
        if truncation_distances:
            trunc_sim = simulation_bundle[truncation_hab_atten]
            sorted_truncation_distances = sorted(truncation_distances)
            trunc_df = evaluate_truncation_from_simulation(
                all_distances=trunc_sim["all_distances"],
                all_levels=trunc_sim["all_levels"],
                truncation_distances=sorted_truncation_distances,
            )
            trunc_df = trunc_df.sort_values("truncation_distance").reset_index(
                drop=True
            )

            trunc_long = trunc_df.melt(
                id_vars=["truncation_distance"],
                value_vars=["precision", "recall"],
                var_name="Metric",
                value_name="Value",
            )
            trunc_long = trunc_long.sort_values(["Metric", "truncation_distance"])
            fig_trunc = px.line(
                trunc_long,
                x="truncation_distance",
                y="Value",
                color="Metric",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_trunc.update_traces(marker=dict(size=7))
            fig_trunc.update_layout(
                hovermode="x unified",
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_white",
                yaxis=dict(range=[0, 1]),
                legend=inside_legend(x=0.9, y=0.3),
                showlegend=True,
            )
            st.plotly_chart(fig_trunc, use_container_width=True)

            with st.expander("Truncation table"):
                st.dataframe(
                    trunc_df.round(
                        {
                            "truncation_distance": 0,
                            "truncation_level": 2,
                            "precision": 3,
                            "recall": 3,
                            "type_i_error": 3,
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
