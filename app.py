
import math
import streamlit as st
import pandas as pd

# ============================================================
# AMD AI ENERGY, CARBON & WATER CALCULATOR
# TOKEN-BASED ONLY
# MXFP4: MI350P, MI350X and MI355X only
# ============================================================

st.set_page_config(
    page_title="AI Energy & Carbon Calculator",
    page_icon="⚡",
    layout="wide"
)

# ============================================================
# MODEL CATALOGUE
# ============================================================

POPULAR_MODELS = {
    "Custom model": {"architecture": "Dense", "total_params": 70, "active_params": 70, "precision": "BF16"},
    "Llama 3.1 8B": {"architecture": "Dense", "total_params": 8, "active_params": 8, "precision": "BF16"},
    "Llama 3.1 70B": {"architecture": "Dense", "total_params": 70, "active_params": 70, "precision": "BF16"},
    "Llama 3.1 405B": {"architecture": "Dense", "total_params": 405, "active_params": 405, "precision": "FP8"},
    "Llama 3.3 70B": {"architecture": "Dense", "total_params": 70, "active_params": 70, "precision": "BF16"},
    "Qwen2.5 7B": {"architecture": "Dense", "total_params": 7, "active_params": 7, "precision": "BF16"},
    "Qwen2.5 14B": {"architecture": "Dense", "total_params": 14, "active_params": 14, "precision": "BF16"},
    "Qwen2.5 32B": {"architecture": "Dense", "total_params": 32, "active_params": 32, "precision": "BF16"},
    "Qwen2.5 72B": {"architecture": "Dense", "total_params": 72, "active_params": 72, "precision": "BF16"},
    "Qwen3 8B": {"architecture": "Dense", "total_params": 8, "active_params": 8, "precision": "BF16"},
    "Qwen3 32B": {"architecture": "Dense", "total_params": 32, "active_params": 32, "precision": "BF16"},
    "Qwen3 235B-A22B": {"architecture": "MoE", "total_params": 235, "active_params": 22, "precision": "FP8"},
    "Qwen3-Next 80B-A3B": {"architecture": "MoE", "total_params": 80, "active_params": 3, "precision": "FP8"},
    "DeepSeek-V3 671B": {"architecture": "MoE", "total_params": 671, "active_params": 37, "precision": "FP8"},
    "DeepSeek-R1 671B": {"architecture": "MoE", "total_params": 671, "active_params": 37, "precision": "FP8"},
    "Mixtral 8x7B": {"architecture": "MoE", "total_params": 46.7, "active_params": 12.9, "precision": "BF16"},
    "Mixtral 8x22B": {"architecture": "MoE", "total_params": 141, "active_params": 39, "precision": "BF16"},
    "Gemma 3 27B": {"architecture": "Dense", "total_params": 27, "active_params": 27, "precision": "BF16"},
    "Gemma 3 12B": {"architecture": "Dense", "total_params": 12, "active_params": 12, "precision": "BF16"},
    "Phi-4 14B": {"architecture": "Dense", "total_params": 14, "active_params": 14, "precision": "BF16"},
    "GPT-OSS 20B": {"architecture": "MoE", "total_params": 20, "active_params": 3.6, "precision": "MXFP4"},
    "GPT-OSS 120B": {"architecture": "MoE", "total_params": 120, "active_params": 5.1, "precision": "MXFP4"},
}

# ============================================================
# AMD GPU CATALOGUE
# ============================================================

GPU_CATALOG = {
    "MI300X": {
        "architecture": "CDNA 3",
        "memory_gb": 192,
        "reference_power_w": 750,
        "mxfp4": False
    },
    "MI325X": {
        "architecture": "CDNA 3",
        "memory_gb": 256,
        "reference_power_w": 1000,
        "mxfp4": False
    },
    "MI350P": {
        "architecture": "CDNA 4",
        "memory_gb": 144,
        "reference_power_w": 600,
        "mxfp4": True
    },
    "MI350X": {
        "architecture": "CDNA 4",
        "memory_gb": 288,
        "reference_power_w": 1000,
        "mxfp4": True
    },
    "MI355X": {
        "architecture": "CDNA 4",
        "memory_gb": 288,
        "reference_power_w": 1400,
        "mxfp4": True
    }
}

# ============================================================
# PRECISION
# ============================================================

PRECISION_BITS = {
    "FP32": 32,
    "FP16": 16,
    "BF16": 16,
    "FP8": 8,
    "INT8": 8,
    "INT4": 4,
    "MXFP4": 4
}

PRECISIONS = list(PRECISION_BITS.keys())
MXFP4_GPUS = ["MI350P", "MI350X", "MI355X"]

# ============================================================
# CALCULATION FUNCTIONS
# ============================================================

def estimate_gpu_count(
    total_params,
    precision,
    gpu_memory,
    memory_overhead,
    hbm_usable
):
    bits = PRECISION_BITS[precision]

    # Parameters are in billions; result is decimal GB.
    weight_memory = total_params * bits / 8

    required_memory = weight_memory * (
        1 + memory_overhead / 100
    )

    usable_memory = gpu_memory * hbm_usable / 100

    gpu_count = max(
        1,
        math.ceil(required_memory / usable_memory)
    )

    return {
        "Weight Memory (GB)": weight_memory,
        "Required Memory (GB)": required_memory,
        "Usable HBM per GPU (GB)": usable_memory,
        "Estimated GPU Count": gpu_count
    }


def estimate_model_throughput(
    reference_tps,
    reference_active_params,
    reference_total_params,
    reference_precision,
    active_params,
    total_params,
    precision,
    architecture,
    compute_exponent,
    moe_residency_exponent,
    precision_exponent
):
    active_factor = (
        reference_active_params / active_params
    ) ** compute_exponent

    if architecture == "MoE":
        residency_factor = (
            reference_total_params / total_params
        ) ** moe_residency_exponent
    else:
        residency_factor = 1.0

    precision_factor = (
        PRECISION_BITS[reference_precision] /
        PRECISION_BITS[precision]
    ) ** precision_exponent

    effective_tps = (
        reference_tps
        * active_factor
        * residency_factor
        * precision_factor
    )

    return {
        "Effective TPS/GPU": effective_tps,
        "Compute Scaling Factor": active_factor,
        "MoE Residency Factor": residency_factor,
        "Precision Factor": precision_factor
    }


def calculate_token_workload(
    input_tokens,
    output_tokens,
    input_tps,
    output_tps,
    gpu_count,
    average_power,
    parallel_efficiency_exponent
):
    parallel_efficiency = (
        gpu_count ** (-parallel_efficiency_exponent)
    )

    aggregate_input_tps = (
        input_tps * gpu_count * parallel_efficiency
    )

    aggregate_output_tps = (
        output_tps * gpu_count * parallel_efficiency
    )

    input_seconds = (
        input_tokens / aggregate_input_tps
        if input_tokens > 0 and aggregate_input_tps > 0
        else 0
    )

    output_seconds = (
        output_tokens / aggregate_output_tps
        if output_tokens > 0 and aggregate_output_tps > 0
        else 0
    )

    input_hours = input_seconds / 3600
    output_hours = output_seconds / 3600

    # Required phase energy formula:
    # E_phase = GPU count x average GPU power x phase runtime / 1000

    input_energy = (
        gpu_count * average_power * input_hours / 1000
    )

    output_energy = (
        gpu_count * average_power * output_hours / 1000
    )

    return {
        "Input Runtime (Hours)": input_hours,
        "Output Runtime (Hours)": output_hours,
        "Total Runtime (Hours)": input_hours + output_hours,
        "Input Energy (kWh)": input_energy,
        "Output Energy (kWh)": output_energy,
        "IT Energy (kWh)": input_energy + output_energy,
        "Aggregate Input TPS": aggregate_input_tps,
        "Aggregate Output TPS": aggregate_output_tps,
        "Parallel Efficiency": parallel_efficiency
    }


def calculate_footprint(
    it_energy,
    pue,
    grid_ci,
    cooling_type,
    dlc_wue,
    grid_water
):
    facility_energy = it_energy * pue

    # Grid carbon intensity is entered in gCO2e/kWh.
    carbon = facility_energy * grid_ci / 1000

    # On-site cooling water is counted only for DLC.
    if cooling_type == "Direct Liquid Cooling (DLC)":
        onsite_water = facility_energy * dlc_wue
    else:
        onsite_water = 0.0

    # Optional indirect water associated with grid electricity.
    indirect_grid_water = facility_energy * grid_water

    total_water = onsite_water + indirect_grid_water

    return {
        "IT Energy (kWh)": it_energy,
        "Facility Energy (kWh)": facility_energy,
        "Carbon (kg CO2e)": carbon,
        "Cooling Water (L)": onsite_water,
        "Grid Water (L)": indirect_grid_water,
        "Total Water (L)": total_water
    }


# ============================================================
# SESSION STATE: MODEL DEFAULTS
# ============================================================

if "previous_model" not in st.session_state:
    st.session_state.previous_model = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "GPT-OSS 20B"


def update_model_defaults():
    model = st.session_state.selected_model

    if model == st.session_state.previous_model:
        return

    defaults = POPULAR_MODELS[model]

    st.session_state.architecture = defaults["architecture"]
    st.session_state.total_params = float(defaults["total_params"])
    st.session_state.active_params = float(defaults["active_params"])
    st.session_state.precision = defaults["precision"]
    st.session_state.model_name = model

    if defaults["precision"] == "MXFP4":
        st.session_state.gpu_name = "MI350X"

    st.session_state.previous_model = model


# ============================================================
# HEADER
# ============================================================

#st.title("⚡ AI Energy, Carbon & Water Footprint Calculator")


# ============================================================
# AMD BLACK HEADER BAR
# ============================================================

#AMD_LOGO = "https://tse1.mm.bing.net/th/id/OIP.-RXYqe3IgpK5d1m-UdL8XAHaC2?r=0&rs=1&pid=ImgDetMain&o=7&rm=3"
#AMD_ICON="https://th.bing.com/th/id/OIP.7_59SilB77D_CHSuzIJRiwHaFj?w=250&h=187&c=7&r=0&o=7&dpr=1.5&pid=1.7&rm=3"
st.set_page_config(
    page_title="AI Energy & Carbon Calculator",
    layout="wide"
)

st.caption(
    "AMD Instinct GPU | Token-based workload estimation"
)

st.markdown("""
<style>
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    padding: 12px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ENVIRONMENTAL ASSUMPTIONS
# ============================================================

st.sidebar.header("Environmental Assumptions")

pue = st.sidebar.number_input(
    "Data Centre PUE",
    min_value=1.0,
    value=1.15,
    step=0.01
)

grid_ci = st.sidebar.number_input(
    "Grid Carbon Intensity (gCO2e/kWh)",
    min_value=0.0,
    value=300.0,
    step=10.0
)

cooling_type = st.sidebar.selectbox(
    "Cooling Type",
    ["Air Cooled", "Direct Liquid Cooling (DLC)"]
)

if cooling_type == "Direct Liquid Cooling (DLC)":
    dlc_wue = st.sidebar.number_input(
        "DLC On-site WUE (L/kWh)",
        min_value=0.0,
        value=0.4,
        step=0.1
    )
else:
    dlc_wue = 0.0
    st.sidebar.info(
        "Air cooling selected: cooling-related on-site "
        "water footprint is set to zero."
    )

grid_water = st.sidebar.number_input(
    "Indirect Grid Water Intensity (L/kWh)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Optional indirect water footprint from electricity generation. Set to zero if excluded."
)

# ============================================================
# SECTION 1: MODEL CONFIGURATION
# ============================================================

st.header("1. AI Model Configuration")

st.selectbox(
    "Select AI Model",
    list(POPULAR_MODELS.keys()),
    key="selected_model",
    on_change=update_model_defaults
)

if st.session_state.previous_model is None:
    update_model_defaults()

c1, c2, c3 = st.columns(3)

with c1:
    model_name = st.text_input(
        "Model Name",
        key="model_name"
    )

    architecture = st.selectbox(
        "Architecture",
        ["Dense", "MoE"],
        key="architecture"
    )

with c2:
    total_params = st.number_input(
        "Total Parameters (Billions)",
        min_value=0.1,
        key="total_params",
        step=1.0
    )

    active_params = st.number_input(
        "Active Parameters per Token (Billions)",
        min_value=0.1,
        key="active_params",
        step=1.0
    )

with c3:
    precision = st.selectbox(
        "Weight Precision",
        PRECISIONS,
        key="precision"
    )

if active_params > total_params:
    st.error("Active parameters cannot exceed total parameters.")
    st.stop()

if architecture == "MoE":
    st.info(
        "MoE: Total parameters determine weight residency "
        "and memory sizing. Active parameters determine "
        "approximate compute requirements per token."
    )

# ============================================================
# SECTION 2: GPU CONFIGURATION
# ============================================================

st.divider()
st.header("2. AMD Instinct GPU Configuration")

# Restrict GPU options for MXFP4.
if precision == "MXFP4":
    gpu_options = MXFP4_GPUS
else:
    gpu_options = list(GPU_CATALOG.keys())

if st.session_state.get("gpu_name") not in gpu_options:
    st.session_state.gpu_name = (
        "MI350X" if precision == "MXFP4" else gpu_options[0]
    )

g1, g2, g3 = st.columns(3)

with g1:
    gpu_name = st.selectbox(
        "AMD Instinct GPU",
        gpu_options,
        key="gpu_name"
    )

    gpu = GPU_CATALOG[gpu_name]

    st.info(
        f"Architecture: {gpu['architecture']}  \n"
        f"HBM: {gpu['memory_gb']} GB  \n"
        f"Reference Power: {gpu['reference_power_w']} W"
    )

with g2:
    memory_overhead = st.number_input(
        "Memory Overhead (%)",
        min_value=0.0,
        max_value=200.0,
        value=30.0,
        step=5.0
    )

    hbm_usable = st.slider(
        "Usable HBM (%)",
        50,
        100,
        90
    )

with g3:
    average_power = st.number_input(
        "Average Power per GPU (W)",
        min_value=0.0,
        value=float(gpu["reference_power_w"]),
        step=10.0,
        key=f"average_power_{gpu_name}"
    )

# ============================================================
# MXFP4 COMPATIBILITY
# ============================================================

st.divider()
st.subheader("MXFP4 Platform Compatibility")

if precision == "MXFP4":
    st.success(f"MXFP4 is supported on {gpu_name}.")
else:
    st.info(
        f"{precision} selected. MXFP4 platform restriction does not apply."
    )

# ============================================================
# SECTION 3: AUTOMATIC GPU SIZING
# ============================================================

st.divider()
st.header("3. Automatic GPU Sizing")

gpu_sizing = estimate_gpu_count(
    total_params,
    precision,
    gpu["memory_gb"],
    memory_overhead,
    hbm_usable
)

automatic_gpu_count = gpu_sizing["Estimated GPU Count"]

gpu_count_mode = st.radio(
    "GPU Count Configuration",
    ["Automatic sizing", "Manual GPU count"],
    horizontal=True
)

if gpu_count_mode == "Automatic sizing":
    gpu_count = automatic_gpu_count
else:
    gpu_count = st.number_input(
        "Number of GPUs",
        min_value=1,
        max_value=1024,
        value=automatic_gpu_count,
        step=1
    )

s1, s2, s3, s4 = st.columns(4)

s1.metric("Estimated GPUs", f"{automatic_gpu_count} x {gpu_name}")
s2.metric("Weight Memory", f"{gpu_sizing['Weight Memory (GB)']:,.1f} GB")
s3.metric("Required Memory", f"{gpu_sizing['Required Memory (GB)']:,.1f} GB")
s4.metric("Usable HBM / GPU", f"{gpu_sizing['Usable HBM per GPU (GB)']:,.1f} GB")

if gpu_count < automatic_gpu_count:
    st.warning(
        "Configured GPU count is below the estimated "
        "weight-memory requirement."
    )

# ============================================================
# SECTION 4: TOKEN WORKLOAD CONFIGURATION
# ============================================================

st.divider()
st.header("4. Token Workload Configuration")

st.subheader("Token Workload")

t1, t2 = st.columns(2)

with t1:
    input_tokens = st.number_input(
        "Input Tokens / Prefill",
        min_value=0,
        value=15000000,
        step=100000
    )

    output_tokens = st.number_input(
        "Output Tokens / Decode",
        min_value=0,
        value=4500000,
        step=100000
    )

with t2:
    throughput_mode = st.selectbox(
        "Throughput Estimation",
        [
            "Model-aware estimated throughput",
            "Measured throughput for selected model"
        ]
    )

model_scaling_enabled = (
    throughput_mode == "Model-aware estimated throughput"
)

effective_input_tps = 0.0
effective_output_tps = 0.0
throughput_result = {}

# ============================================================
# MODEL-AWARE THROUGHPUT
# ============================================================

if model_scaling_enabled:

    st.subheader("Model-Aware Throughput Scaling")

    r1, r2, r3 = st.columns(3)

    with r1:
        reference_active = st.number_input(
            "Reference Active Parameters (B)",
            min_value=0.1,
            value=70.0,
            step=1.0
        )

    with r2:
        reference_total = st.number_input(
            "Reference Total Parameters (B)",
            min_value=0.1,
            value=70.0,
            step=1.0
        )

    with r3:
        reference_precision = st.selectbox(
            "Reference Precision",
            PRECISIONS,
            index=PRECISIONS.index("BF16")
        )

    p1, p2, p3 = st.columns(3)

    with p1:
        reference_input_tps = st.number_input(
            "Reference Input TPS / GPU",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

    with p2:
        reference_output_tps = st.number_input(
            "Reference Output TPS / GPU",
            min_value=0.0,
            value=250.0,
            step=10.0
        )

    with p3:
        compute_exponent = st.slider(
            "Active Parameter Scaling Exponent",
            0.0, 1.5, 0.85, 0.05
        )

    q1, q2 = st.columns(2)

    with q1:
        moe_residency_exponent = st.slider(
            "MoE Weight Residency Exponent",
            0.0, 1.0, 0.15, 0.05
        )

    with q2:
        precision_exponent = st.slider(
            "Precision Scaling Exponent",
            0.0, 1.0, 0.5, 0.05
        )

    if (
        reference_active <= 0
        or reference_total <= 0
        or reference_active > reference_total
    ):
        st.error("Invalid reference model parameters.")
        st.stop()

    input_scaling = estimate_model_throughput(
        reference_input_tps,
        reference_active,
        reference_total,
        reference_precision,
        active_params,
        total_params,
        precision,
        architecture,
        compute_exponent,
        moe_residency_exponent,
        precision_exponent
    )

    output_scaling = estimate_model_throughput(
        reference_output_tps,
        reference_active,
        reference_total,
        reference_precision,
        active_params,
        total_params,
        precision,
        architecture,
        compute_exponent,
        moe_residency_exponent,
        precision_exponent
    )

    effective_input_tps = input_scaling["Effective TPS/GPU"]
    effective_output_tps = output_scaling["Effective TPS/GPU"]

    throughput_result = {
        "Input": input_scaling,
        "Output": output_scaling
    }

    a, b, c = st.columns(3)

    a.metric("Estimated Input TPS / GPU", f"{effective_input_tps:,.2f}")
    b.metric("Estimated Output TPS / GPU", f"{effective_output_tps:,.2f}")
    c.metric("Active Parameters", f"{active_params:g} B")

# ============================================================
# MEASURED THROUGHPUT
# ============================================================

else:

    p1, p2 = st.columns(2)

    with p1:
        effective_input_tps = st.number_input(
            "Measured Input TPS / GPU",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

    with p2:
        effective_output_tps = st.number_input(
            "Measured Output TPS / GPU",
            min_value=0.0,
            value=250.0,
            step=10.0
        )

parallel_efficiency_exponent = st.slider(
    "Multi-GPU Scaling Loss Exponent",
    0.0,
    0.30,
    0.08,
    0.01
)

# ============================================================
# SECTION 5: TOKEN-BASED ENERGY CALCULATION
# ============================================================

st.divider()
st.header("5. Token-Based Energy Calculation")

token_result = {}
total_tokens = input_tokens + output_tokens

if total_tokens > 0 and (
    (input_tokens == 0 or effective_input_tps > 0)
    and (output_tokens == 0 or effective_output_tps > 0)
):
    token_result = calculate_token_workload(
        input_tokens,
        output_tokens,
        effective_input_tps,
        effective_output_tps,
        gpu_count,
        average_power,
        parallel_efficiency_exponent
    )

    input_energy = token_result["Input Energy (kWh)"]
    output_energy = token_result["Output Energy (kWh)"]
    total_it_energy = token_result["IT Energy (kWh)"]
    runtime_hours = token_result["Total Runtime (Hours)"]

else:
    input_energy = 0.0
    output_energy = 0.0
    total_it_energy = 0.0
    runtime_hours = 0.0

    if total_tokens == 0:
        st.warning("Enter input or output tokens to calculate energy.")
    else:
        st.warning("Enter valid throughput for each active token phase.")

if token_result:

    st.markdown("**Energy formula**")

    st.latex(
        r"E_{phase}="
        r"\frac{N_{GPU}\times P_{GPU}\times T_{phase}(hours)}{1000}"
    )

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Input / Prefill Energy",
        f"{input_energy:,.4f} kWh"
    )

    e2.metric(
        "Output / Decode Energy",
        f"{output_energy:,.4f} kWh"
    )

    e3.metric(
        "Total IT Energy",
        f"{total_it_energy:,.4f} kWh"
    )

    breakdown = pd.DataFrame([
        {
            "Phase": "Input / Prefill",
            "Tokens": input_tokens,
            "TPS / GPU": effective_input_tps,
            "Aggregate TPS": token_result["Aggregate Input TPS"],
            "Runtime (Hours)": token_result["Input Runtime (Hours)"],
            "Energy (kWh)": input_energy
        },
        {
            "Phase": "Output / Decode",
            "Tokens": output_tokens,
            "TPS / GPU": effective_output_tps,
            "Aggregate TPS": token_result["Aggregate Output TPS"],
            "Runtime (Hours)": token_result["Output Runtime (Hours)"],
            "Energy (kWh)": output_energy
        }
    ])

    st.subheader("Input and Output Energy Breakdown")

    st.dataframe(
        breakdown,
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Tokens", f"{total_tokens:,}")

    c2.metric(
        "Energy / 1M Tokens",
        f"{total_it_energy / total_tokens * 1e6:,.4f} kWh"
    )

    c3.metric("Total Runtime", f"{runtime_hours:,.3f} hours")

    c4.metric(
        "Parallel Efficiency",
        f"{token_result['Parallel Efficiency'] * 100:.2f}%"
    )

# ============================================================
# SECTION 6: ENVIRONMENTAL FOOTPRINT
# ============================================================

st.divider()
st.header("6. Energy, Carbon & Water Results")

result = calculate_footprint(
    total_it_energy,
    pue,
    grid_ci,
    cooling_type,
    dlc_wue,
    grid_water
)

if total_it_energy > 0:

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "IT Energy",
        f"{result['IT Energy (kWh)']:,.4f} kWh"
    )

    e2.metric(
        "Facility Energy",
        f"{result['Facility Energy (kWh)']:,.4f} kWh"
    )

    e3.metric(
        "Carbon Emissions",
        f"{result['Carbon (kg CO2e)']:,.4f} kg"
    )

    e4.metric(
        "Total Water Footprint",
        f"{result['Total Water (L)']:,.4f} L"
    )

    st.subheader("Environmental Breakdown")

    env_df = pd.DataFrame([
        {"Metric": "IT Energy", "Value": result["IT Energy (kWh)"], "Unit": "kWh"},
        {"Metric": "Facility Energy", "Value": result["Facility Energy (kWh)"], "Unit": "kWh"},
        {"Metric": "Carbon Emissions", "Value": result["Carbon (kg CO2e)"], "Unit": "kg CO2e"},
        {"Metric": "Cooling Water", "Value": result["Cooling Water (L)"], "Unit": "L"},
        {"Metric": "Indirect Grid Water", "Value": result["Grid Water (L)"], "Unit": "L"},
        {"Metric": "Total Water", "Value": result["Total Water (L)"], "Unit": "L"}
    ])

    st.dataframe(
        env_df,
        use_container_width=True,
        hide_index=True
    )

    if cooling_type == "Air Cooled":
        st.info(
            "Air cooling selected: on-site cooling water is zero. "
            "Any displayed water footprint is only the optional "
            "indirect grid water component."
        )
    else:
        st.info(
            "DLC selected: cooling water is calculated using "
            "facility energy multiplied by the configured DLC WUE."
        )

# ============================================================
# SECTION 7: MODEL AND GPU SUMMARY
# ============================================================

st.divider()
st.header("7. Model and GPU Summary")

summary1, summary2, summary3, summary4 = st.columns(4)

summary1.metric("Model", model_name)
summary2.metric("Architecture", architecture)
summary3.metric("Total Parameters", f"{total_params:g} B")
summary4.metric("Active Parameters", f"{active_params:g} B")

summary1, summary2, summary3, summary4 = st.columns(4)

summary1.metric("GPU", gpu_name)
summary2.metric("GPU Count", f"{gpu_count}")
summary3.metric("Precision", precision)
summary4.metric(
    "Total GPU Memory",
    f"{gpu_count * gpu['memory_gb']:,} GB"
)

# ============================================================
# SECTION 8: CSV EXPORT
# ============================================================

st.divider()
st.header("8. Export Results")

export_data = {
    "Model": model_name,
    "Architecture": architecture,
    "Total Parameters (B)": total_params,
    "Active Parameters (B)": active_params,
    "Precision": precision,
    "MXFP4 Compatibility": (
        "Supported" if precision == "MXFP4" else "Not applicable"
    ),
    "GPU": gpu_name,
    "GPU Architecture": gpu["architecture"],
    "GPU HBM (GB)": gpu["memory_gb"],
    "GPU Count": gpu_count,
    "Automatic GPU Count": automatic_gpu_count,
    "Total GPU HBM (GB)": gpu_count * gpu["memory_gb"],
    "Weight Memory (GB)": gpu_sizing["Weight Memory (GB)"],
    "Required Memory (GB)": gpu_sizing["Required Memory (GB)"],
    "Average Power per GPU (W)": average_power,
    "Workload Method": "Token-based",
    "Input Tokens": input_tokens,
    "Output Tokens": output_tokens,
    "Total Tokens": total_tokens,
    "Input TPS per GPU": effective_input_tps,
    "Output TPS per GPU": effective_output_tps,
    "Input Runtime (Hours)": token_result.get("Input Runtime (Hours)", 0),
    "Output Runtime (Hours)": token_result.get("Output Runtime (Hours)", 0),
    "Input Energy (kWh)": input_energy,
    "Output Energy (kWh)": output_energy,
    "Total Runtime (Hours)": runtime_hours,
    **result,
    "PUE": pue,
    "Grid Carbon Intensity (gCO2e/kWh)": grid_ci,
    "Cooling Type": cooling_type,
    "DLC WUE (L/kWh)": dlc_wue,
    "Indirect Grid Water Intensity (L/kWh)": grid_water
}

export_df = pd.DataFrame([export_data])

st.download_button(
    "Download Energy, Carbon & Water CSV",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="AMD_AI_Energy_Carbon_Water_Results.csv",
    mime="text/csv",
    type="primary"
)

# ============================================================
# SECTION 9: METHODOLOGY
# ============================================================

st.divider()

with st.expander("Calculation Methodology and Assumptions"):

    st.subheader("1. Model Weight Memory")

    st.latex(
        r"M_{weights}=\frac{Parameters(B)\times Precision(bits)}{8}"
    )

    st.subheader("2. GPU Count")

    st.latex(
        r"N_{GPU}=\left\lceil\frac{M_{required}}{M_{usable}}\right\rceil"
    )

    st.subheader("3. Model-Aware Throughput")

    st.latex(
        r"TPS_{effective}=TPS_{reference}"
        r"\times\left(\frac{A_{reference}}{A_{selected}}\right)^\alpha"
        r"\times\left(\frac{P_{reference}}{P_{selected}}\right)^\beta"
    )

    st.markdown(
        "MoE models additionally use a total-weight residency factor."
    )

    st.latex(
        r"F_{MoE}=\left(\frac{T_{reference}}{T_{selected}}\right)^\gamma"
    )

    st.subheader("4. Multi-GPU Throughput")

    st.latex(
        r"TPS_{aggregate}=TPS_{GPU}\times N_{GPU}"
        r"\times N_{GPU}^{-\delta}"
    )

    st.subheader("5. Token-Based Energy")

    st.latex(
        r"E_{phase}=\frac{N_{GPU}\times P_{GPU}"
        r"\times T_{phase}(hours)}{1000}"
    )

    st.subheader("6. Environmental Footprint")

    st.latex(r"E_{facility}=E_{IT}\times PUE")

    st.latex(
        r"Carbon=\frac{E_{facility}\times GridCI}{1000}"
    )

    st.markdown("**Cooling water:**")

    st.latex(
        r"Water_{cooling}="
        r"\begin{cases}"
        r"0 & \text{Air cooled}\\"
        r"E_{facility}\times WUE & \text{DLC}"
        r"\end{cases}"
    )

    st.latex(
        r"Water_{total}=Water_{cooling}"
        r"+E_{facility}\times GridWater"
    )

    st.subheader("Important Assumptions")

    st.markdown("""
    - Total parameters determine model weight memory.
    - Active parameters determine approximate per-token compute.
    - MoE total parameters influence estimated weight residency.
    - MXFP4 is modelled as 4 bits per weight.
    - Memory overhead approximates MXFP4 scales, metadata, KV cache,
      activations and memory fragmentation.
    - MXFP4 is enabled only for MI350P, MI350X and MI355X.
    - Model-aware throughput is a configurable heuristic, not a benchmark.
    - Measured throughput mode uses entered benchmark values directly.
    - Token phases are assumed to execute sequentially.
    - Parallel efficiency approximates multi-GPU scaling losses.
    - PUE accounts for facility overhead.
    - On-site cooling water is counted only for DLC.
    - Indirect grid water is optional and can be set to zero.
    - Actual results depend on context length, batch size, concurrency,
      KV cache, software stack and measured GPU power.
    """)

    st.subheader("AMD Instinct GPU Catalogue")

    gpu_df = pd.DataFrame([
        {
            "GPU": name,
            "Architecture": details["architecture"],
            "HBM (GB)": details["memory_gb"],
            "Reference Power (W)": details["reference_power_w"],
            "MXFP4 Support": "Supported" if details["mxfp4"] else "Not supported"
        }
        for name, details in GPU_CATALOG.items()
    ])

    st.dataframe(
        gpu_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AMD Instinct AI Energy, Carbon & Water Calculator | "
    "Token-based planning estimates. Validate using measured "
    "AMD SMI power and actual model benchmark throughput."
)
