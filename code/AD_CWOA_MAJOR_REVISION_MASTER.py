# Recovered AD-CWOA major-revision master experiment script
# Extracted verbatim from code cells of notebook02febea9bb.ipynb.

# ===== Notebook code cell 1 =====
# ======================================================================
# AD-CWOA FINAL K=15 — STANDALONE FRESH-RUNTIME RECOVERY + VALIDATION
# Kaggle: enable GPU T4 x2, then run this entire file/cell once.
# This script does NOT run the final 100-run experiment.
# ======================================================================


# ============================================================
# 1. ENVIRONMENT + CONFIGURATION
# ============================================================
import os, sys, math, json, time, random, shutil, platform, subprocess, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn.functional as F

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)

GLOBAL_SEED = 2026
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

@dataclass(frozen=True)
class WSNConfig:
    area_x: float = 200.0
    area_y: float = 200.0
    num_nodes: int = 200
    initial_energy: float = 1.0
    packet_size: int = 4000
    e_elec: float = 50e-9
    e_fs: float = 10e-12
    e_mp: float = 0.0013e-12
    e_da: float = 5e-9
    sink_x: float = 100.0
    sink_y: float = 100.0
    w_energy: float = 0.35
    w_intra: float = 0.30
    w_sink: float = 0.20
    w_load: float = 0.15

@dataclass(frozen=True)
class OptimizerConfig:
    num_ch: int = 10
    population_size: int = 30
    evaluation_budget: int = 1500
    local_search_trials: int = 5
    stagnation_limit: int = 4
    restart_fraction: float = 0.30

@dataclass(frozen=True)
class ExperimentConfig:
    base_topology_seed: int = 5000
    single_optimizer_seed: int = 2026
    independent_runs: int = 20
    run_seed_start: int = 2000
    topology_runs: int = 20
    topology_seed_start: int = 5000
    topology_optimizer_seed_start: int = 8000
    lifetime_max_rounds: int = 5000
    lifetime_reselection_interval: int = 10
    lifetime_evaluation_budget: int = 300
    lifetime_seed: int = 9500

CFG = WSNConfig()
OPT = OptimizerConfig()
EXP = ExperimentConfig()

QUICK_MODE = False
N_INDEPENDENT_RUNS = 3 if QUICK_MODE else EXP.independent_runs
N_TOPOLOGIES = 3 if QUICK_MODE else EXP.topology_runs
LIFETIME_MAX_ROUNDS = 1000 if QUICK_MODE else EXP.lifetime_max_rounds

D0 = math.sqrt(CFG.e_fs / CFG.e_mp)
NETWORK_DIAGONAL = math.hypot(CFG.area_x, CFG.area_y)
MAX_SINK_DISTANCE = math.hypot(
    max(CFG.sink_x, CFG.area_x - CFG.sink_x),
    max(CFG.sink_y, CFG.area_y - CFG.sink_y),
)

assert abs(CFG.w_energy + CFG.w_intra + CFG.w_sink + CFG.w_load - 1.0) < 1e-12
assert EXP.lifetime_evaluation_budget % OPT.population_size == 0

if Path("/kaggle/working").exists():
    OUTPUT_ROOT = Path("/kaggle/working/CWOA_WSN_Research")
else:
    OUTPUT_ROOT = Path.cwd() / "CWOA_WSN_Research"

FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
HTML_DIR = OUTPUT_ROOT / "interactive_html"
for p in [OUTPUT_ROOT, FIG_DIR, TABLE_DIR, HTML_DIR]:
    p.mkdir(parents=True, exist_ok=True)

GPU_DEVICES = (
    [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)
WORK_DEVICES = GPU_DEVICES[:2]

METHOD_ORDER = ["original_cwoa", "improved_cwoa", "random_search"]
METHOD_LABELS = {
    "original_cwoa": "Original CWOA",
    "improved_cwoa": "Improved CWOA",
    "random_search": "Random Search",
}
METHOD_COLORS = {
    "original_cwoa": "#0072B2",
    "improved_cwoa": "#009E73",
    "random_search": "#D55E00",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def sync_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def start_measurement(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    return time.perf_counter()

def end_measurement(device, t0):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak = 0.0
    return time.perf_counter() - t0, peak

def save_paper_figure(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIG_DIR / f"{stem}.{ext}", **kwargs)

def save_interactive(fig, stem):
    fig.write_html(HTML_DIR / f"{stem}.html", include_plotlyjs=True)

with open(OUTPUT_ROOT / "experiment_config.json", "w") as f:
    json.dump(
        {
            "wsn": asdict(CFG),
            "optimizer": asdict(OPT),
            "experiment": asdict(EXP),
            "quick_mode": QUICK_MODE,
            "d0_m": D0,
        },
        f,
        indent=2,
    )

print("=" * 70)
print("GPU ENVIRONMENT")
print("=" * 70)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
for d in GPU_DEVICES:
    if d.type == "cuda":
        p = torch.cuda.get_device_properties(d)
        print(f"{d}: {p.name} | {p.total_memory / 1024**3:.2f} GB")
    else:
        print("cpu")
print("Work devices:", WORK_DEVICES)
print("Output:", OUTPUT_ROOT)

# ============================================================
# 2. WSN MODEL + VECTORIZED COMMUNICATION ENGINE
# ============================================================
def create_sensor_network(topology_seed, num_nodes=None):
    n = CFG.num_nodes if num_nodes is None else int(num_nodes)
    rng = np.random.default_rng(int(topology_seed))
    network = pd.DataFrame({
        "node_id": np.arange(1, n + 1, dtype=int),
        "x": rng.uniform(0, CFG.area_x, n),
        "y": rng.uniform(0, CFG.area_y, n),
        "energy": np.full(n, CFG.initial_energy, dtype=np.float64),
        "alive": np.ones(n, dtype=bool),
        "is_CH": np.zeros(n, dtype=bool),
    })
    network["distance_to_sink"] = np.hypot(
        network["x"].to_numpy() - CFG.sink_x,
        network["y"].to_numpy() - CFG.sink_y,
    )
    return network

def transmission_energy(bits, distance):
    d = np.asarray(distance, dtype=np.float64)
    e = np.where(
        d < D0,
        bits * CFG.e_elec + bits * CFG.e_fs * d**2,
        bits * CFG.e_elec + bits * CFG.e_mp * d**4,
    )
    return float(e) if e.ndim == 0 else e

def reception_energy(bits):
    return bits * CFG.e_elec

def aggregation_energy(bits):
    return bits * CFG.e_da

def assign_nodes_to_cluster_heads(nodes_df, ch_indices):
    result = nodes_df.copy()
    result["is_CH"] = False
    result["cluster_id"] = -1
    result["ch_index"] = -1
    result["ch_node_id"] = -1
    result["distance_to_ch"] = np.nan

    ch_indices = np.asarray(ch_indices, dtype=int)
    alive_idx = result.index[result["alive"]].to_numpy(dtype=int)

    if len(ch_indices) == 0 or len(alive_idx) == 0:
        return result
    if len(np.unique(ch_indices)) != len(ch_indices):
        raise ValueError("Cluster-head indices must be unique.")
    if not result.loc[ch_indices, "alive"].all():
        raise ValueError("All selected cluster heads must be alive.")

    result.loc[ch_indices, "is_CH"] = True

    alive_xy = result.loc[alive_idx, ["x", "y"]].to_numpy(np.float64)
    ch_xy = result.loc[ch_indices, ["x", "y"]].to_numpy(np.float64)
    dist = np.sqrt(np.sum((alive_xy[:, None, :] - ch_xy[None, :, :]) ** 2, axis=2))
    nearest = np.argmin(dist, axis=1)
    chosen = ch_indices[nearest]

    result.loc[alive_idx, "cluster_id"] = nearest
    result.loc[alive_idx, "ch_index"] = chosen
    result.loc[alive_idx, "ch_node_id"] = result.loc[chosen, "node_id"].to_numpy(dtype=int)
    result.loc[alive_idx, "distance_to_ch"] = dist[np.arange(len(alive_idx)), nearest]
    return result

def simulate_communication_round(nodes_df, ch_indices):
    network = nodes_df.copy()
    network["alive"] = network["energy"].to_numpy() > 0.0
    alive_before = int(network["alive"].sum())

    ch_indices = np.asarray(
        [int(i) for i in ch_indices if network.at[int(i), "alive"]],
        dtype=int,
    )

    if len(ch_indices) == 0:
        return network, {
            "packets_to_ch": 0,
            "ch_transmissions_to_bs": 0,
            "source_packets_delivered_to_bs": 0,
            "energy_consumed": 0.0,
            "residual_energy": float(network["energy"].sum()),
            "alive_nodes": alive_before,
            "dead_nodes": len(network) - alive_before,
        }

    clustered = assign_nodes_to_cluster_heads(network, ch_indices)
    energy_before = float(clustered["energy"].sum())

    alive_mask = clustered["alive"].to_numpy()
    ch_mask = clustered["is_CH"].to_numpy()
    member_idx = np.flatnonzero(alive_mask & ~ch_mask)

    packets_to_ch = 0
    received = np.zeros(len(ch_indices), dtype=int)

    if len(member_idx):
        d = clustered.loc[member_idx, "distance_to_ch"].to_numpy(np.float64)
        costs = transmission_energy(CFG.packet_size, d)
        current = clustered.loc[member_idx, "energy"].to_numpy(np.float64)
        success = current >= costs
        clustered.loc[member_idx, "energy"] = np.where(success, current - costs, 0.0)
        packets_to_ch = int(success.sum())
        cluster_ids = clustered.loc[member_idx, "cluster_id"].to_numpy(dtype=int)
        received = np.bincount(cluster_ids[success], minlength=len(ch_indices))

    ch_energy = clustered.loc[ch_indices, "energy"].to_numpy(np.float64)
    source_packets = received + 1
    rx_cost = received * reception_energy(CFG.packet_size)
    da_cost = source_packets * aggregation_energy(CFG.packet_size)
    sink_d = clustered.loc[ch_indices, "distance_to_sink"].to_numpy(np.float64)
    tx_sink = transmission_energy(CFG.packet_size, sink_d)
    total_ch_cost = rx_cost + da_cost + tx_sink
    ch_success = (ch_energy > 0.0) & (ch_energy >= total_ch_cost)

    clustered.loc[ch_indices, "energy"] = np.where(
        ch_success, ch_energy - total_ch_cost, 0.0
    )

    clustered["energy"] = clustered["energy"].clip(lower=0.0)
    clustered["alive"] = clustered["energy"] > 0.0
    residual = float(clustered["energy"].sum())

    return clustered, {
        "packets_to_ch": packets_to_ch,
        "ch_transmissions_to_bs": int(ch_success.sum()),
        "source_packets_delivered_to_bs": int(source_packets[ch_success].sum()),
        "energy_consumed": energy_before - residual,
        "residual_energy": residual,
        "alive_nodes": int(clustered["alive"].sum()),
        "dead_nodes": int((~clustered["alive"]).sum()),
    }

base_nodes = create_sensor_network(EXP.base_topology_seed)
display(base_nodes.head())
print("Base topology mean sink distance:", base_nodes["distance_to_sink"].mean())

# ============================================================
# 3. BATCHED GPU FITNESS EVALUATOR + NUMERICAL VALIDATION
# ============================================================
class GPUFitnessEvaluator:
    def __init__(self, nodes_df, device):
        self.nodes_df = nodes_df
        self.device = torch.device(device)
        self.positions = torch.as_tensor(
            nodes_df[["x", "y"]].to_numpy(np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        self.energy = torch.as_tensor(
            nodes_df["energy"].to_numpy(np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        self.alive_mask = torch.as_tensor(
            nodes_df["alive"].to_numpy(dtype=bool),
            dtype=torch.bool,
            device=self.device,
        )
        self.alive_idx = torch.nonzero(self.alive_mask, as_tuple=False).squeeze(1)
        self.sink_distance = torch.sqrt(
            (self.positions[:, 0] - CFG.sink_x) ** 2
            + (self.positions[:, 1] - CFG.sink_y) ** 2
        )

    @property
    def alive_count(self):
        return int(self.alive_idx.numel())

    def evaluate_batch(self, ch_batch, return_details=False):
        ch = torch.as_tensor(ch_batch, dtype=torch.long, device=self.device)
        if ch.ndim == 1:
            ch = ch.unsqueeze(0)

        B, K = ch.shape
        M = self.alive_count
        if M < K:
            raise ValueError("Not enough alive nodes for requested CH count.")

        valid = self.alive_mask[ch].all(dim=1)
        if K > 1:
            s = torch.sort(ch, dim=1).values
            valid &= (s[:, 1:] != s[:, :-1]).all(dim=1)

        ch_xy = self.positions[ch]
        alive_xy = self.positions[self.alive_idx]
        diff = alive_xy.unsqueeze(0).unsqueeze(2) - ch_xy.unsqueeze(1)
        distances = torch.sqrt(torch.sum(diff * diff, dim=-1).clamp_min(0.0))
        min_distance, assignment = torch.min(distances, dim=2)

        mean_intra = (
            min_distance.sum(dim=1) / float(M - K)
            if M > K
            else torch.zeros(B, device=self.device)
        )
        intra_penalty = mean_intra / NETWORK_DIAGONAL

        normalized_energy = (
            self.energy[ch].clamp(0.0, CFG.initial_energy) / CFG.initial_energy
        )
        energy_penalty = 1.0 - normalized_energy.mean(dim=1)

        mean_sink = self.sink_distance[ch].mean(dim=1)
        sink_penalty = mean_sink / MAX_SINK_DISTANCE

        # Deterministic counts; avoids CUDA atomic scatter-add nondeterminism.
        cluster_sizes = F.one_hot(assignment, num_classes=K).sum(dim=1).to(torch.float32)
        load_penalty = cluster_sizes.std(dim=1, unbiased=False) / (float(M) / float(K))

        fitness = (
            CFG.w_energy * energy_penalty
            + CFG.w_intra * intra_penalty
            + CFG.w_sink * sink_penalty
            + CFG.w_load * load_penalty
        )
        fitness = torch.where(valid, fitness, torch.full_like(fitness, float("inf")))

        if not return_details:
            return fitness

        return fitness, {
            "energy_penalty": energy_penalty,
            "mean_intra_distance": mean_intra,
            "intra_penalty": intra_penalty,
            "mean_ch_sink_distance": mean_sink,
            "sink_penalty": sink_penalty,
            "cluster_sizes": cluster_sizes,
            "load_penalty": load_penalty,
        }

def details_to_python(details, row=0):
    out = {}
    for k, v in details.items():
        x = v[row].detach().cpu().numpy() if torch.is_tensor(v) else v
        out[k] = float(x) if np.ndim(x) == 0 else x
    return out

def decode_whales_gpu(whales, evaluator, num_ch):
    scores = whales.index_select(1, evaluator.alive_idx)
    top_pos = torch.topk(scores, k=num_ch, dim=1, largest=True, sorted=False).indices
    return evaluator.alive_idx[top_pos]

def evaluate_candidate_cpu_reference(nodes_df, ch_indices):
    ch = np.asarray(ch_indices, dtype=int)
    alive_idx = nodes_df.index[nodes_df["alive"]].to_numpy(dtype=int)
    alive_xy = nodes_df.loc[alive_idx, ["x", "y"]].to_numpy(np.float64)
    ch_xy = nodes_df.loc[ch, ["x", "y"]].to_numpy(np.float64)
    d = np.sqrt(np.sum((alive_xy[:, None, :] - ch_xy[None, :, :]) ** 2, axis=2))
    assign = np.argmin(d, axis=1)
    min_d = np.min(d, axis=1)
    mean_intra = min_d.sum() / (len(alive_idx) - len(ch)) if len(alive_idx) > len(ch) else 0.0
    energy_penalty = 1.0 - np.mean(
        np.clip(nodes_df.loc[ch, "energy"].to_numpy() / CFG.initial_energy, 0.0, 1.0)
    )
    mean_sink = nodes_df.loc[ch, "distance_to_sink"].mean()
    cluster_sizes = np.bincount(assign, minlength=len(ch))
    load_penalty = np.std(cluster_sizes) / np.mean(cluster_sizes)
    return (
        CFG.w_energy * energy_penalty
        + CFG.w_intra * (mean_intra / NETWORK_DIAGONAL)
        + CFG.w_sink * (mean_sink / MAX_SINK_DISTANCE)
        + CFG.w_load * load_penalty
    )

rng = np.random.default_rng(12345)

# Validate more than the all-1-J initial state:
# vary residual energy and mark a subset of nodes dead.
validation_nodes = base_nodes.copy()
validation_nodes["energy"] = rng.uniform(
    0.15, CFG.initial_energy, len(validation_nodes)
)
dead_idx = rng.choice(
    validation_nodes.index,
    size=max(1, len(validation_nodes) // 10),
    replace=False,
)
validation_nodes.loc[dead_idx, "energy"] = 0.0
validation_nodes["alive"] = validation_nodes["energy"] > 0.0

alive = validation_nodes.index[
    validation_nodes["alive"]
].to_numpy(dtype=int)

validation_candidates = np.vstack([
    rng.choice(alive, size=OPT.num_ch, replace=False)
    for _ in range(128)
])

cpu_values = np.array([
    evaluate_candidate_cpu_reference(validation_nodes, ch)
    for ch in validation_candidates
])

validator = GPUFitnessEvaluator(validation_nodes, WORK_DEVICES[0])
gpu_values = validator.evaluate_batch(validation_candidates).cpu().numpy()
abs_error = np.abs(cpu_values - gpu_values)

validation_df = pd.DataFrame([{
    "Max abs error": abs_error.max(),
    "Mean abs error": abs_error.mean(),
    "Max relative error": np.max(abs_error / np.maximum(np.abs(cpu_values), 1e-12)),
}])
display(validation_df)
assert abs_error.max() < 5e-5
print("PASS: GPU fitness matches CPU reference within FP32 tolerance.")

# ============================================================
# 4. GPU OPTIMIZERS
# ============================================================
def run_cwoa_gpu(nodes_df, num_ch=10, population_size=30, max_iterations=50,
                 seed=2026, device=None, verbose=False):
    device = WORK_DEVICES[0] if device is None else torch.device(device)
    evaluator = GPUFitnessEvaluator(nodes_df, device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n = len(nodes_df)
    t0 = start_measurement(device)

    whales = torch.rand((population_size, n), generator=gen, device=device)
    best_fit = float("inf")
    best_whale = None
    best_ch = None
    convergence = []

    for iteration in range(max_iterations):
        ch_batch = decode_whales_gpu(whales, evaluator, num_ch)
        fit = evaluator.evaluate_batch(ch_batch)
        value, idx = torch.min(fit, dim=0)

        if float(value) < best_fit:
            best_fit = float(value)
            best_whale = whales[int(idx)].clone()
            best_ch = ch_batch[int(idx)].clone()

        convergence.append(best_fit)
        a = 0.0 if max_iterations <= 1 else 2.0 - 2.0 * iteration / (max_iterations - 1)

        r1 = torch.rand((population_size, 1), generator=gen, device=device)
        r2 = torch.rand((population_size, 1), generator=gen, device=device)
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        p = torch.rand((population_size, 1), generator=gen, device=device)

        ridx = torch.randint(0, population_size, (population_size,), generator=gen, device=device)
        random_whales = whales[ridx]

        exploit = best_whale.unsqueeze(0) - A * torch.abs(C * best_whale.unsqueeze(0) - whales)
        explore = random_whales - A * torch.abs(C * random_whales - whales)
        encircle = torch.where(torch.abs(A) < 1.0, exploit, explore)

        l = -1.0 + 2.0 * torch.rand((population_size, 1), generator=gen, device=device)
        spiral = (
            torch.abs(best_whale.unsqueeze(0) - whales)
            * torch.exp(l)
            * torch.cos(2.0 * math.pi * l)
            + best_whale.unsqueeze(0)
        )
        whales = torch.where(p < 0.5, encircle, spiral).clamp_(0.0, 1.0)

        if verbose:
            print(iteration + 1, best_fit)

    runtime_s, peak_mb = end_measurement(device, t0)
    _, details_t = evaluator.evaluate_batch(best_ch, return_details=True)

    return {
        "method": "original_cwoa",
        "ch_indices": best_ch.cpu().numpy(),
        "fitness": best_fit,
        "convergence_x": np.arange(1, max_iterations + 1) * population_size,
        "convergence_y": np.asarray(convergence),
        "details": details_to_python(details_t),
        "evaluations": population_size * max_iterations,
        "runtime_s": runtime_s,
        "peak_gpu_memory_mb": peak_mb,
        "device": str(device),
    }

def run_random_search_gpu(nodes_df, num_ch=10, evaluations=1500, seed=2026,
                          device=None, batch_size=512, return_convergence=True):
    device = WORK_DEVICES[0] if device is None else torch.device(device)
    evaluator = GPUFitnessEvaluator(nodes_df, device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    alive_idx = evaluator.alive_idx
    t0 = start_measurement(device)

    best_fit = float("inf")
    best_ch = None
    convergence = []
    done = 0

    while done < evaluations:
        b = min(batch_size, evaluations - done)
        keys = torch.rand((b, len(alive_idx)), generator=gen, device=device)
        pos = torch.topk(keys, k=num_ch, dim=1, largest=True, sorted=False).indices
        ch_batch = alive_idx[pos]
        fit = evaluator.evaluate_batch(ch_batch)
        value, idx = torch.min(fit, dim=0)

        if float(value) < best_fit:
            best_fit = float(value)
            best_ch = ch_batch[int(idx)].clone()

        if return_convergence:
            running = torch.cummin(fit, dim=0).values
            if convergence:
                running = torch.minimum(running, torch.tensor(convergence[-1], device=device))
            convergence.extend(running.cpu().numpy().tolist())

        done += b

    runtime_s, peak_mb = end_measurement(device, t0)
    _, details_t = evaluator.evaluate_batch(best_ch, return_details=True)

    return {
        "method": "random_search",
        "ch_indices": best_ch.cpu().numpy(),
        "fitness": best_fit,
        "convergence_x": np.arange(1, evaluations + 1) if return_convergence else np.array([]),
        "convergence_y": np.asarray(convergence) if return_convergence else np.array([]),
        "details": details_to_python(details_t),
        "evaluations": evaluations,
        "runtime_s": runtime_s,
        "peak_gpu_memory_mb": peak_mb,
        "device": str(device),
    }

def generate_ch_swap_gpu(ch_indices, alive_idx, gen):
    candidate = ch_indices.clone()
    device = candidate.device
    non_ch = alive_idx[~torch.isin(alive_idx, candidate)]
    if len(non_ch) == 0:
        return candidate
    remove_pos = int(torch.randint(0, len(candidate), (1,), generator=gen, device=device).item())
    add_pos = int(torch.randint(0, len(non_ch), (1,), generator=gen, device=device).item())
    candidate[remove_pos] = non_ch[add_pos]
    return candidate

def encode_ch_to_random_keys_gpu(ch_indices, dimension, gen, device):
    whale = torch.empty(dimension, device=device)
    whale.uniform_(0.0, 0.45, generator=gen)
    high = torch.empty(len(ch_indices), device=device)
    high.uniform_(0.55, 1.0, generator=gen)
    whale[ch_indices] = high
    return whale

def run_improved_cwoa_gpu(nodes_df, num_ch=10, population_size=30, max_evaluations=1500,
                           local_search_trials=5, stagnation_limit=4, restart_fraction=0.30,
                           seed=2026, device=None, verbose=False):
    device = WORK_DEVICES[0] if device is None else torch.device(device)
    evaluator = GPUFitnessEvaluator(nodes_df, device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n = len(nodes_df)
    t0 = start_measurement(device)

    whales = torch.rand((population_size, n), generator=gen, device=device)
    fitness_values = torch.full((population_size,), float("inf"), device=device)

    initial_n = min(population_size, max_evaluations)
    ch_batch = decode_whales_gpu(whales[:initial_n], evaluator, num_ch)
    fit = evaluator.evaluate_batch(ch_batch)
    fitness_values[:initial_n] = fit
    value, idx = torch.min(fit, dim=0)

    best_fit = float(value)
    best_whale = whales[int(idx)].clone()
    best_ch = ch_batch[int(idx)].clone()
    evaluations = initial_n
    generation = 0
    stagnation = 0
    conv_x = [evaluations]
    conv_y = [best_fit]

    while evaluations < max_evaluations:
        generation += 1
        previous_best = best_fit
        progress = evaluations / max_evaluations
        a = 2.0 * (1.0 - progress)

        r1 = torch.rand((population_size, 1), generator=gen, device=device)
        r2 = torch.rand((population_size, 1), generator=gen, device=device)
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        p = torch.rand((population_size, 1), generator=gen, device=device)

        ridx = torch.randint(0, population_size, (population_size,), generator=gen, device=device)
        random_whales = whales[ridx]
        exploit = best_whale.unsqueeze(0) - A * torch.abs(C * best_whale.unsqueeze(0) - whales)
        explore = random_whales - A * torch.abs(C * random_whales - whales)
        encircle = torch.where(torch.abs(A) < 1.0, exploit, explore)

        l = -1.0 + 2.0 * torch.rand((population_size, 1), generator=gen, device=device)
        spiral = (
            torch.abs(best_whale.unsqueeze(0) - whales)
            * torch.exp(l)
            * torch.cos(2.0 * math.pi * l)
            + best_whale.unsqueeze(0)
        )

        new_whales = torch.where(p < 0.5, encircle, spiral)

        mutation_probability = 0.15 + 0.20 * progress
        mutation_count = max(1, int(0.05 * n))
        mutation_scale = 0.15 * (1.0 - progress) + 0.03

        for i in range(population_size):
            if float(torch.rand((), generator=gen, device=device)) < mutation_probability:
                midx = torch.randperm(n, generator=gen, device=device)[:mutation_count]
                new_whales[i, midx] += (
                    torch.randn((mutation_count,), generator=gen, device=device)
                    * mutation_scale
                )

        whales = new_whales.clamp_(0.0, 1.0)

        remaining = max_evaluations - evaluations
        eval_n = min(population_size, remaining)
        ch_batch = decode_whales_gpu(whales[:eval_n], evaluator, num_ch)
        fit = evaluator.evaluate_batch(ch_batch)
        fitness_values[:eval_n] = fit
        value, idx = torch.min(fit, dim=0)

        if float(value) < best_fit:
            best_fit = float(value)
            best_whale = whales[int(idx)].clone()
            best_ch = ch_batch[int(idx)].clone()

        evaluations += eval_n
        if evaluations >= max_evaluations:
            conv_x.append(evaluations)
            conv_y.append(best_fit)
            break

        for _ in range(local_search_trials):
            if evaluations >= max_evaluations:
                break
            candidate = generate_ch_swap_gpu(best_ch, evaluator.alive_idx, gen)
            candidate_fit = float(evaluator.evaluate_batch(candidate)[0])
            evaluations += 1

            if candidate_fit < best_fit:
                best_fit = candidate_fit
                best_ch = candidate.clone()
                best_whale = encode_ch_to_random_keys_gpu(best_ch, n, gen, device)
                worst = int(torch.argmax(fitness_values))
                whales[worst] = best_whale
                fitness_values[worst] = best_fit

        stagnation = 0 if best_fit < previous_best - 1e-12 else stagnation + 1

        if stagnation >= stagnation_limit:
            restart_count = max(1, int(population_size * restart_fraction))
            worst_idx = torch.topk(fitness_values, k=restart_count, largest=True).indices
            whales[worst_idx] = torch.rand((restart_count, n), generator=gen, device=device)
            fitness_values[worst_idx] = float("inf")
            stagnation = 0

        conv_x.append(evaluations)
        conv_y.append(best_fit)

        if verbose:
            print(generation, evaluations, best_fit)

    runtime_s, peak_mb = end_measurement(device, t0)
    _, details_t = evaluator.evaluate_batch(best_ch, return_details=True)

    return {
        "method": "improved_cwoa",
        "ch_indices": best_ch.cpu().numpy(),
        "fitness": best_fit,
        "convergence_x": np.asarray(conv_x),
        "convergence_y": np.asarray(conv_y),
        "details": details_to_python(details_t),
        "evaluations": evaluations,
        "runtime_s": runtime_s,
        "peak_gpu_memory_mb": peak_mb,
        "device": str(device),
    }

print("All GPU optimizers loaded.")

# ============================================================
# FINAL RECOVERY PATCH — validated K=15 engine interface
# Run after the original notebook cells above.
# ============================================================
from dataclasses import replace

# Frozen final constants used in the major-revision experiments
NUM_NODES = 200
NUM_CH = 15
POPULATION_SIZE = 30
CH_RESELECTION_INTERVAL = 10
LIFETIME_OPTIMIZER_BUDGET = 300
INITIAL_ENERGY = 1.0
DEVICES = list(WORK_DEVICES)


def set_fitness_weights(weights):
    global CFG
    wE, wI, wS, wL = [float(x) for x in weights]
    if abs((wE+wI+wS+wL)-1.0) > 1e-12:
        raise ValueError('Fitness weights must sum to 1.')
    CFG = replace(CFG, w_energy=wE, w_intra=wI, w_sink=wS, w_load=wL)
    print(f'Fitness weights -> Energy={wE:.2f}, Intra={wI:.2f}, Sink={wS:.2f}, Load={wL:.2f}')


def network_to_gpu(nodes_df, device):
    # Reuse the numerically validated GPU evaluator from the base notebook.
    return GPUFitnessEvaluator(nodes_df, device)


def evaluate_population_gpu(random_keys, state, num_ch):
    # Decode random-key vectors to unique alive CHs and evaluate the same
    # dimensionless four-component objective used by the base notebook.
    ch_indices = decode_whales_gpu(random_keys, state, int(num_ch))
    fitness, details = state.evaluate_batch(ch_indices, return_details=True)
    return fitness, ch_indices, details


def encode_ch_solution_gpu(ch_indices, dimension, generator, device):
    return encode_ch_to_random_keys_gpu(ch_indices, dimension, generator, device)


def simulate_round(nodes_df, ch_indices):
    network, s = simulate_communication_round(nodes_df, ch_indices)
    # Compatibility keys used by the final lifetime engine.
    return network, {
        'source_packets': int(s['source_packets_delivered_to_bs']),
        'ch_transmissions': int(s['ch_transmissions_to_bs']),
        'residual_energy': float(s['residual_energy']),
        'alive_nodes': int(s['alive_nodes']),
        'dead_nodes': int(s['dead_nodes']),
        'energy_consumed': float(s['energy_consumed']),
    }


@torch.no_grad()
def run_original_cwoa_gpu(nodes_df, num_ch=NUM_CH, population_size=30,
                          evaluation_budget=1500, seed=2026, device=None):
    if device is None:
        device = DEVICES[0]
    state = network_to_gpu(nodes_df, device)
    n = len(nodes_df)
    iterations = max(1, int(evaluation_budget) // int(population_size))
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    whales = torch.rand((population_size, n), generator=g, device=device)
    best_fitness = float('inf')
    best_whale = None
    best_ch = None
    best_details = None
    convergence = []

    for iteration in range(iterations):
        fitness, ch_indices, details = evaluate_population_gpu(whales, state, num_ch)
        current_best_idx = int(torch.argmin(fitness).item())
        current_best = float(fitness[current_best_idx].item())
        if current_best < best_fitness:
            best_fitness = current_best
            best_whale = whales[current_best_idx].clone()
            best_ch = ch_indices[current_best_idx].clone()
            best_details = {k: v[current_best_idx].detach().cpu() for k,v in details.items()}
        convergence.append(best_fitness)

        if iterations > 1:
            a = 2.0 - 2.0 * iteration / (iterations - 1)
        else:
            a = 0.0
        r1 = torch.rand((population_size,1), generator=g, device=device)
        r2 = torch.rand((population_size,1), generator=g, device=device)
        A = 2.0*a*r1 - a
        C = 2.0*r2
        p = torch.rand((population_size,1), generator=g, device=device)

        D_best = torch.abs(C*best_whale[None,:] - whales)
        exploitation = best_whale[None,:] - A*D_best
        random_indices = torch.randint(0,population_size,(population_size,),generator=g,device=device)
        random_whales = whales[random_indices]
        D_random = torch.abs(C*random_whales - whales)
        exploration = random_whales - A*D_random
        encircling = torch.where((torch.abs(A)<1).expand_as(whales), exploitation, exploration)
        l = torch.rand((population_size,1),generator=g,device=device)*2 - 1
        distance_best = torch.abs(best_whale[None,:] - whales)
        spiral = distance_best*torch.exp(l)*torch.cos(2*math.pi*l) + best_whale[None,:]
        whales = torch.where((p<0.5).expand_as(whales), encircling, spiral)
        whales.clamp_(0.0,1.0)

    return {
        'ch_indices': best_ch.detach().cpu().numpy(),
        'fitness': best_fitness,
        'convergence': np.asarray(convergence),
        'details': best_details,
        'evaluations': iterations*population_size,
    }


@torch.no_grad()
def run_improved_cwoa_gpu(nodes_df, num_ch=NUM_CH, population_size=30,
                          evaluation_budget=1500, local_trials=5,
                          stagnation_limit=4, restart_fraction=0.30,
                          seed=2026, device=None):
    if device is None:
        device = DEVICES[0]
    state = network_to_gpu(nodes_df, device)
    n = len(nodes_df)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    whales = torch.rand((population_size,n), generator=g, device=device)
    best_fitness = float('inf')
    best_whale = None
    best_ch = None
    best_details = None
    evaluations = 0
    stagnation_counter = 0
    evaluation_history = []
    fitness_history = []

    while evaluations < evaluation_budget:
        remaining = evaluation_budget - evaluations
        current_pop_size = min(population_size, remaining)
        population_slice = whales[:current_pop_size]
        fitness, ch_indices, details = evaluate_population_gpu(population_slice, state, num_ch)
        evaluations += current_pop_size
        current_best_idx = int(torch.argmin(fitness).item())
        current_best = float(fitness[current_best_idx].item())
        previous_best = best_fitness

        if current_best < best_fitness:
            best_fitness = current_best
            best_whale = population_slice[current_best_idx].clone()
            best_ch = ch_indices[current_best_idx].clone()
            best_details = {k:v[current_best_idx].detach().cpu() for k,v in details.items()}

        available_local = min(local_trials, evaluation_budget-evaluations)
        if available_local > 0 and best_ch is not None:
            alive_idx = torch.where(state.alive_mask)[0]
            local_keys = []
            for _ in range(available_local):
                candidate_ch = best_ch.clone()
                remove_position = int(torch.randint(0,num_ch,(1,),generator=g,device=device).item())
                ch_mask = torch.zeros(n,dtype=torch.bool,device=device)
                ch_mask[candidate_ch] = True
                possible_nodes = alive_idx[~ch_mask[alive_idx]]
                if len(possible_nodes) > 0:
                    selected = int(torch.randint(0,len(possible_nodes),(1,),generator=g,device=device).item())
                    candidate_ch[remove_position] = possible_nodes[selected]
                local_keys.append(encode_ch_solution_gpu(candidate_ch,n,g,device))
            local_population = torch.stack(local_keys)
            local_fitness, local_ch, local_details = evaluate_population_gpu(local_population,state,num_ch)
            evaluations += available_local
            local_best_idx = int(torch.argmin(local_fitness).item())
            local_best = float(local_fitness[local_best_idx].item())
            if local_best < best_fitness:
                best_fitness = local_best
                best_whale = local_population[local_best_idx].clone()
                best_ch = local_ch[local_best_idx].clone()
                best_details = {k:v[local_best_idx].detach().cpu() for k,v in local_details.items()}

        evaluation_history.append(evaluations)
        fitness_history.append(best_fitness)
        if best_fitness < previous_best - 1e-12:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        if evaluations >= evaluation_budget:
            break

        progress = evaluations/evaluation_budget
        a = 2.0*(1.0-progress)
        r1 = torch.rand((population_size,1),generator=g,device=device)
        r2 = torch.rand((population_size,1),generator=g,device=device)
        A = 2*a*r1-a
        C = 2*r2
        p = torch.rand((population_size,1),generator=g,device=device)
        D_best = torch.abs(C*best_whale[None,:]-whales)
        exploitation = best_whale[None,:]-A*D_best
        random_indices = torch.randint(0,population_size,(population_size,),generator=g,device=device)
        random_whales = whales[random_indices]
        D_random = torch.abs(C*random_whales-whales)
        exploration = random_whales-A*D_random
        encircling = torch.where((torch.abs(A)<1).expand_as(whales), exploitation, exploration)
        l = torch.rand((population_size,1),generator=g,device=device)*2-1
        distance_best = torch.abs(best_whale[None,:]-whales)
        spiral = distance_best*torch.exp(l)*torch.cos(2*math.pi*l)+best_whale[None,:]
        whales = torch.where((p<0.5).expand_as(whales),encircling,spiral)

        mutation_probability = 0.15 + 0.20*progress
        mutation_scale = 0.15*(1-progress)+0.03
        mutation_count = max(1,int(0.05*n))
        mutation_whales = torch.where(torch.rand(population_size,generator=g,device=device) < mutation_probability)[0]
        for whale_idx in mutation_whales.tolist():
            mutation_indices = torch.randperm(n,generator=g,device=device)[:mutation_count]
            whales[whale_idx,mutation_indices] += torch.randn(mutation_count,generator=g,device=device)*mutation_scale
        whales.clamp_(0.0,1.0)

        if stagnation_counter >= stagnation_limit:
            restart_count = max(1,int(population_size*restart_fraction))
            restart_indices = torch.randperm(population_size,generator=g,device=device)[:restart_count]
            whales[restart_indices] = torch.rand((restart_count,n),generator=g,device=device)
            stagnation_counter = 0

    return {
        'ch_indices': best_ch.detach().cpu().numpy(),
        'fitness': best_fitness,
        'evaluation_history': np.asarray(evaluation_history),
        'convergence': np.asarray(fitness_history),
        'details': best_details,
        'evaluations': evaluations,
    }


@torch.no_grad()
def run_random_search_gpu(nodes_df, num_ch=NUM_CH, evaluation_budget=1500,
                          batch_size=256, seed=2026, device=None):
    if device is None:
        device = DEVICES[0]
    state = network_to_gpu(nodes_df, device)
    n = len(nodes_df)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    best_fitness = float('inf')
    best_ch = None
    evaluations = 0
    while evaluations < evaluation_budget:
        current_batch = min(batch_size, evaluation_budget-evaluations)
        keys = torch.rand((current_batch,n),generator=g,device=device)
        fitness,ch_indices,_ = evaluate_population_gpu(keys,state,num_ch)
        best_idx = int(torch.argmin(fitness).item())
        current_best = float(fitness[best_idx].item())
        if current_best < best_fitness:
            best_fitness = current_best
            best_ch = ch_indices[best_idx].clone()
        evaluations += current_batch
    return {'ch_indices':best_ch.detach().cpu().numpy(),'fitness':best_fitness,'evaluations':evaluations}


def select_cluster_heads_gpu(nodes_df, method, num_ch, evaluation_budget, seed, device):
    if method == 'original_cwoa':
        return run_original_cwoa_gpu(nodes_df,num_ch=num_ch,population_size=POPULATION_SIZE,
                                     evaluation_budget=evaluation_budget,seed=seed,device=device)
    if method == 'improved_cwoa':
        return run_improved_cwoa_gpu(nodes_df,num_ch=num_ch,population_size=POPULATION_SIZE,
                                     evaluation_budget=evaluation_budget,local_trials=3,
                                     stagnation_limit=3,restart_fraction=0.30,seed=seed,device=device)
    if method == 'random_search':
        return run_random_search_gpu(nodes_df,num_ch=num_ch,evaluation_budget=evaluation_budget,
                                     seed=seed,device=device)
    if method == 'pso':
        return run_pso_gpu(nodes_df,num_ch=num_ch,population_size=POPULATION_SIZE,
                           evaluation_budget=evaluation_budget,seed=seed,device=device)
    if method == 'leach':
        raise ValueError("LEACH uses round-wise CH rotation; call simulate_lifetime_method_gpu.")
    raise ValueError(f'Unknown method: {method}')


def simulate_lifetime_gpu(topology_seed, method, optimizer_seed, device,
                          max_rounds=5000, num_ch=NUM_CH,
                          reselection_interval=CH_RESELECTION_INTERVAL,
                          evaluation_budget=LIFETIME_OPTIMIZER_BUDGET, verbose=False):
    network = create_sensor_network(topology_seed)
    rng = np.random.default_rng(int(optimizer_seed))
    current_ch = None
    history = []
    cumulative_source_packets = 0
    cumulative_ch_transmissions = 0
    total_optimizer_evaluations = 0
    FND = HND = LND = None
    half_nodes = int(np.ceil(len(network)/2))

    for round_number in range(1,max_rounds+1):
        alive_count = int(network['alive'].sum())
        if alive_count == 0:
            LND = round_number-1
            break
        active_num_ch = min(num_ch,alive_count)
        need_reselection = (current_ch is None or (round_number-1)%reselection_interval==0 or
                            any(not bool(network.loc[idx,'alive']) for idx in current_ch))
        if need_reselection:
            seed_event = int(rng.integers(0,2**31-1))
            result = select_cluster_heads_gpu(network,method,active_num_ch,evaluation_budget,seed_event,device)
            current_ch = result['ch_indices']
            total_optimizer_evaluations += result['evaluations']
        network,stats_round = simulate_round(network,current_ch)
        cumulative_source_packets += stats_round['source_packets']
        cumulative_ch_transmissions += stats_round['ch_transmissions']
        alive_now = stats_round['alive_nodes']
        dead_now = stats_round['dead_nodes']
        if FND is None and dead_now >= 1: FND = round_number
        if HND is None and dead_now >= half_nodes: HND = round_number
        if alive_now == 0: LND = round_number
        history.append({
            'round':round_number,'alive_nodes':alive_now,'dead_nodes':dead_now,
            'residual_energy':stats_round['residual_energy'],'source_packets':stats_round['source_packets'],
            'cumulative_source_packets':cumulative_source_packets,
            'cumulative_ch_transmissions':cumulative_ch_transmissions,
        })
        if verbose and round_number%500==0:
            print(method,'|',round_number,'| Alive:',alive_now,'| Energy:',round(stats_round['residual_energy'],3))
        if alive_now == 0: break

    history_df = pd.DataFrame(history)
    summary = {
        'method':method,'topology_seed':topology_seed,'optimizer_seed':optimizer_seed,
        'device':str(device),'FND':FND,'HND':HND,'LND':LND,
        'throughput':cumulative_source_packets,'ch_transmissions':cumulative_ch_transmissions,
        'optimizer_evaluations':total_optimizer_evaluations,
        'final_energy':float(network['energy'].sum()),
    }
    return summary,history_df


@torch.no_grad()
def run_pso_gpu(nodes_df, num_ch=15, population_size=30, evaluation_budget=300,
                seed=2026, device=None, inertia=0.70, c1=1.50, c2=1.50,
                velocity_limit=0.20):
    if device is None: device = DEVICES[0]
    state = network_to_gpu(nodes_df,device)
    n = len(nodes_df)
    g = torch.Generator(device=device); g.manual_seed(int(seed))
    particles = torch.rand((population_size,n),generator=g,device=device)
    velocity = torch.rand((population_size,n),generator=g,device=device)*0.10-0.05
    pbest_position = particles.clone()
    pbest_fitness = torch.full((population_size,),float('inf'),device=device)
    gbest_position=None; gbest_fitness=float('inf'); gbest_ch=None; gbest_details=None
    evaluations=0; evaluation_history=[]; convergence=[]
    while evaluations < evaluation_budget:
        remaining=evaluation_budget-evaluations
        current_population=min(population_size,remaining)
        current_particles=particles[:current_population]
        fitness,ch_indices,details=evaluate_population_gpu(current_particles,state,num_ch)
        evaluations += current_population
        improved_mask=fitness < pbest_fitness[:current_population]
        improved_indices=torch.where(improved_mask)[0]
        if improved_indices.numel()>0:
            pbest_fitness[improved_indices]=fitness[improved_indices]
            pbest_position[improved_indices]=current_particles[improved_indices]
        current_best_idx=int(torch.argmin(fitness).item())
        current_best_fitness=float(fitness[current_best_idx].item())
        if current_best_fitness < gbest_fitness:
            gbest_fitness=current_best_fitness
            gbest_position=current_particles[current_best_idx].clone()
            gbest_ch=ch_indices[current_best_idx].clone()
            gbest_details={k:v[current_best_idx].detach().cpu() for k,v in details.items()}
        evaluation_history.append(evaluations); convergence.append(gbest_fitness)
        if evaluations >= evaluation_budget: break
        r1=torch.rand((population_size,n),generator=g,device=device)
        r2=torch.rand((population_size,n),generator=g,device=device)
        velocity = inertia*velocity + c1*r1*(pbest_position-particles) + c2*r2*(gbest_position[None,:]-particles)
        velocity.clamp_(-velocity_limit,velocity_limit)
        particles=(particles+velocity).clamp_(0.0,1.0)
    return {
        'ch_indices':gbest_ch.detach().cpu().numpy(),'fitness':gbest_fitness,
        'evaluation_history':np.asarray(evaluation_history),'convergence':np.asarray(convergence),
        'details':gbest_details,'evaluations':evaluations,
    }


def simulate_lifetime_leach_gpu(topology_seed, optimizer_seed, device,
                                max_rounds=5000, num_ch=15, verbose=False):
    network=create_sensor_network(topology_seed)
    rng=np.random.default_rng(int(optimizer_seed))
    initial_nodes=len(network)
    p=float(num_ch)/float(initial_nodes)
    epoch_length=max(1,int(round(1.0/p)))
    history=[]; cumulative_source_packets=0; cumulative_ch_transmissions=0
    FND=HND=LND=None; half_nodes=int(np.ceil(initial_nodes/2))
    selected_in_epoch=set(); previous_epoch=-1
    for round_number in range(1,max_rounds+1):
        alive_indices=np.asarray(network.index[network['alive']],dtype=int)
        alive_count=len(alive_indices)
        if alive_count==0:
            LND=round_number-1; break
        zero_based_round=round_number-1
        current_epoch=zero_based_round//epoch_length
        epoch_phase=zero_based_round%epoch_length
        if current_epoch != previous_epoch:
            selected_in_epoch=set(); previous_epoch=current_epoch
        eligible_nodes=np.asarray([int(i) for i in alive_indices if int(i) not in selected_in_epoch],dtype=int)
        if len(eligible_nodes)==0:
            selected_in_epoch=set(); eligible_nodes=alive_indices.copy()
        denominator=1.0-p*epoch_phase
        threshold=1.0 if denominator<=0 else min(1.0,p/denominator)
        random_values=rng.random(len(eligible_nodes))
        current_ch=eligible_nodes[random_values < threshold]
        if len(current_ch)==0:
            current_ch=np.asarray([int(rng.choice(eligible_nodes))],dtype=int)
        reasonable_upper_bound=min(alive_count,max(1,int(np.ceil(2.0*p*alive_count))))
        if len(current_ch)>reasonable_upper_bound:
            current_ch=np.asarray(rng.choice(current_ch,size=reasonable_upper_bound,replace=False),dtype=int)
        selected_in_epoch.update(int(i) for i in current_ch)
        network,stats_round=simulate_round(network,current_ch)
        cumulative_source_packets += int(stats_round['source_packets'])
        cumulative_ch_transmissions += int(stats_round['ch_transmissions'])
        alive_now=int(stats_round['alive_nodes']); dead_now=int(stats_round['dead_nodes'])
        if FND is None and dead_now>=1: FND=round_number
        if HND is None and dead_now>=half_nodes: HND=round_number
        if alive_now==0: LND=round_number
        history.append({
            'round':round_number,'alive_nodes':alive_now,'dead_nodes':dead_now,
            'residual_energy':stats_round['residual_energy'],'source_packets':stats_round['source_packets'],
            'cumulative_source_packets':cumulative_source_packets,
            'cumulative_ch_transmissions':cumulative_ch_transmissions,
            'active_ch_count':len(current_ch),'leach_threshold':threshold,
            'epoch':current_epoch,'epoch_phase':epoch_phase,
        })
        if verbose and round_number%500==0:
            print('LEACH |',round_number,'| Alive:',alive_now,'| CH:',len(current_ch))
        if alive_now==0: break
    history_df=pd.DataFrame(history)
    summary={
        'method':'leach','topology_seed':topology_seed,'optimizer_seed':optimizer_seed,
        'device':str(device),'FND':FND,'HND':HND,'LND':LND,
        'throughput':cumulative_source_packets,'ch_transmissions':cumulative_ch_transmissions,
        'optimizer_evaluations':0,'final_energy':float(network['energy'].sum()),
    }
    return summary,history_df


def simulate_lifetime_method_gpu(topology_seed,method,optimizer_seed,device,
                                 max_rounds=5000,num_ch=15,
                                 reselection_interval=CH_RESELECTION_INTERVAL,
                                 evaluation_budget=300,verbose=False):
    if method=='leach':
        return simulate_lifetime_leach_gpu(topology_seed,optimizer_seed,device,max_rounds,num_ch,verbose)
    return simulate_lifetime_gpu(topology_seed,method,optimizer_seed,device,max_rounds,num_ch,
                                 reselection_interval,evaluation_budget,verbose)

# Freeze the final selected configuration.
set_fitness_weights((0.35,0.30,0.20,0.15))
print('='*75)
print('FINAL RECOVERY ENGINE INSTALLED')
print('Devices:', DEVICES)
print('K =', NUM_CH, '| population =', POPULATION_SIZE, '| lifetime budget =', LIFETIME_OPTIMIZER_BUDGET)
print('='*75)

# ============================================================
# RECOVERY VALIDATION — no full experiment yet
# ============================================================
required = [
    'create_sensor_network','simulate_round','network_to_gpu','evaluate_population_gpu',
    'run_original_cwoa_gpu','run_improved_cwoa_gpu','run_random_search_gpu',
    'run_pso_gpu','select_cluster_heads_gpu','simulate_lifetime_gpu',
    'simulate_lifetime_leach_gpu','simulate_lifetime_method_gpu','set_fitness_weights'
]
missing=[x for x in required if x not in globals()]
print('Missing functions:',missing)
if missing:
    raise RuntimeError('Recovery incomplete.')
print('✅ CORE + FINAL BASELINE ENGINE RESTORED')

# One paired topology sanity check for PSO and corrected LEACH.
for method in ['pso','leach']:
    summary,hist=simulate_lifetime_method_gpu(
        topology_seed=5000,method=method,optimizer_seed=9000,
        device=DEVICES[0],max_rounds=5000,num_ch=15,evaluation_budget=300,verbose=False)
    print('\n',method.upper(), summary)
    assert summary['FND'] <= summary['HND'] <= summary['LND']
    assert not (hist['alive_nodes'].diff().dropna()>0).any()
    assert not (hist['residual_energy'].diff().dropna()>1e-9).any()
    if method=='leach':
        print(hist['active_ch_count'].describe())
        assert hist['active_ch_count'].max() <= 30
print('\n✅ RECOVERY VALIDATION PASSED')

print("\n" + "="*78)
print("STANDALONE RECOVERY SCRIPT FINISHED")
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("DEVICES:", DEVICES)
print("FINAL K:", NUM_CH)
print("FINAL WEIGHTS:", (CFG.w_energy, CFG.w_intra, CFG.w_sink, CFG.w_load))
print("FINAL RESELECTION INTERVAL:", CH_RESELECTION_INTERVAL)
print("FINAL EVAL BUDGET:", LIFETIME_OPTIMIZER_BUDGET)
print("="*78)

# ===== Notebook code cell 2 =====

# ======================================================================
# AD-CWOA MAJOR-REVISION MASTER ANALYSIS
# Checkpoint/resume enabled | Kaggle T4 x2
#
# IMPORTANT:
# - Run ONLY after FINAL_K15_STANDALONE_RECOVERY_VALIDATION has passed.
# - This script does NOT rebuild the engine.
# - It resumes completed runs automatically from /kaggle/working.
# - It runs:
#     A) 7 x 20 = 140 K=15 weight-sensitivity lifetime runs
#     B) 3 x 20 = 60 K={12,15,18} confirmatory lifetime runs
#     C) 5 x 20 = 100 final method-comparison lifetime runs
#   Total = 300 lifetime simulations, minus any completed checkpoints.
# ======================================================================

import os, gc, json, math, time, shutil, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from scipy import stats

# ----------------------------------------------------------------------
# 0. VERIFY RECOVERED ENGINE
# ----------------------------------------------------------------------
_REQUIRED = [
    "DEVICES", "NUM_NODES", "NUM_CH", "POPULATION_SIZE",
    "CH_RESELECTION_INTERVAL", "LIFETIME_OPTIMIZER_BUDGET",
    "create_sensor_network", "simulate_round",
    "network_to_gpu", "evaluate_population_gpu",
    "run_original_cwoa_gpu", "run_improved_cwoa_gpu",
    "run_random_search_gpu", "run_pso_gpu",
    "select_cluster_heads_gpu", "simulate_lifetime_gpu",
    "simulate_lifetime_leach_gpu", "simulate_lifetime_method_gpu",
    "set_fitness_weights",
]
_missing = [x for x in _REQUIRED if x not in globals()]
if _missing:
    raise RuntimeError(
        "Recovered engine is not loaded. Missing: " + ", ".join(_missing)
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")
if len(DEVICES) < 1:
    raise RuntimeError("DEVICES is empty.")

WORK_DEVICES = list(DEVICES[:2])
print("Devices:", WORK_DEVICES)

# ----------------------------------------------------------------------
# 1. FROZEN EXPERIMENT SETTINGS
# ----------------------------------------------------------------------
FINAL_K = 15
FINAL_WEIGHTS = (0.35, 0.30, 0.20, 0.15)
EVAL_BUDGET = 300
MAX_ROUNDS = 5000
TOPOLOGY_SEEDS = list(range(5000, 5020))
OPTIMIZER_SEEDS = list(range(9000, 9020))
CHECKPOINT_ROUNDS = [500, 1000, 1500, 1800, 2000]

WEIGHT_CONFIGS = {
    "Baseline_35_30_20_15": (0.35, 0.30, 0.20, 0.15),
    "EnergyHeavy_50_20_15_15": (0.50, 0.20, 0.15, 0.15),
    "Equal_25_25_25_25": (0.25, 0.25, 0.25, 0.25),
    "IntraHeavy_25_45_15_15": (0.25, 0.45, 0.15, 0.15),
    "LegacyMapped_50_25_25_00": (0.50, 0.25, 0.25, 0.00),
    "LoadHeavy_25_20_15_40": (0.25, 0.20, 0.15, 0.40),
    "SinkHeavy_25_20_40_15": (0.25, 0.20, 0.40, 0.15),
}

K_CONFIRM = [12, 15, 18]

FINAL_METHODS = [
    "leach",
    "random_search",
    "original_cwoa",
    "pso",
    "improved_cwoa",
]
METHOD_LABELS = {
    "leach": "LEACH-style",
    "random_search": "Random Search",
    "original_cwoa": "Original CWOA",
    "pso": "PSO",
    "improved_cwoa": "AD-CWOA",
}

ROOT = Path("/kaggle/working/AD_CWOA_MAJOR_REVISION_RERUN")
WEIGHT_DIR = ROOT / "01_weight_sensitivity_K15"
K_DIR = ROOT / "02_K12_K15_K18_confirmatory"
FINAL_DIR = ROOT / "03_final_5_method"
HIST_DIR = FINAL_DIR / "histories"
for p in [ROOT, WEIGHT_DIR, K_DIR, FINAL_DIR, HIST_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# freeze master settings
set_fitness_weights(FINAL_WEIGHTS)

config = {
    "nodes": int(NUM_NODES),
    "final_k": FINAL_K,
    "population": int(POPULATION_SIZE),
    "final_weights": FINAL_WEIGHTS,
    "eval_budget_per_reselection": EVAL_BUDGET,
    "reselection_interval": int(CH_RESELECTION_INTERVAL),
    "max_rounds": MAX_ROUNDS,
    "topology_seeds": TOPOLOGY_SEEDS,
    "optimizer_seeds": OPTIMIZER_SEEDS,
    "devices": [str(d) for d in WORK_DEVICES],
    "torch_version": torch.__version__,
}
(ROOT / "frozen_config.json").write_text(json.dumps(config, indent=2))

# ----------------------------------------------------------------------
# 2. HELPERS
# ----------------------------------------------------------------------
def atomic_csv(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)

def load_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def completed_keys(df, cols):
    if df.empty:
        return set()
    return set(tuple(row) for row in df[cols].itertuples(index=False, name=None))

def safe_value_at_round(history_df, col, round_number):
    if history_df.empty:
        return np.nan
    hit = history_df.loc[history_df["round"] == round_number, col]
    if len(hit):
        return float(hit.iloc[0])
    # if lifetime ended before requested round:
    if int(history_df["round"].max()) < round_number:
        if col == "alive_nodes":
            return 0.0
        if col == "residual_energy":
            return 0.0
        if col == "cumulative_source_packets":
            return float(history_df["cumulative_source_packets"].iloc[-1])
    return np.nan

def add_metrics(summary, history_df):
    out = dict(summary)
    FND = float(out["FND"])
    HND = float(out["HND"])
    LND = float(out["LND"])
    throughput = float(out["throughput"])
    ch_tx = float(out["ch_transmissions"])

    out["Stability_Ratio"] = FND / LND
    out["Death_Transition_Period"] = LND - FND
    out["Throughput_per_Round"] = throughput / LND
    out["CH_Transmissions_per_Round"] = ch_tx / LND

    rounds = history_df["round"].to_numpy(dtype=float)
    alive = history_df["alive_nodes"].to_numpy(dtype=float)
    x = np.insert(rounds, 0, 0.0)
    y = np.insert(alive, 0, float(NUM_NODES))
    auc = float(np.trapezoid(y, x))
    out["Alive_Node_AUC"] = auc
    out["Normalized_AUC"] = auc / (float(NUM_NODES) * LND)

    for rr in CHECKPOINT_ROUNDS:
        out[f"ResidualEnergy_R{rr}"] = safe_value_at_round(history_df, "residual_energy", rr)
        out[f"AliveNodes_R{rr}"] = safe_value_at_round(history_df, "alive_nodes", rr)

    return out

def run_lifetime_one(topology_seed, optimizer_seed, method, device, k=FINAL_K):
    t0 = time.perf_counter()
    summary, hist = simulate_lifetime_method_gpu(
        topology_seed=int(topology_seed),
        method=method,
        optimizer_seed=int(optimizer_seed),
        device=device,
        max_rounds=MAX_ROUNDS,
        num_ch=int(k),
        reselection_interval=CH_RESELECTION_INTERVAL,
        evaluation_budget=EVAL_BUDGET,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    # structural sanity checks
    if not (summary["FND"] <= summary["HND"] <= summary["LND"]):
        raise RuntimeError(f"Milestone order failed: {summary}")
    if (hist["alive_nodes"].diff().dropna() > 0).any():
        raise RuntimeError("Alive nodes increased.")
    if (hist["residual_energy"].diff().dropna() > 1e-9).any():
        raise RuntimeError("Residual energy increased.")
    if (hist["residual_energy"] < -1e-9).any():
        raise RuntimeError("Negative residual energy.")

    row = add_metrics(summary, hist)
    row["runtime_s"] = elapsed
    row["k"] = int(k)
    return row, hist

def queue_jobs_two_gpus(jobs, worker):
    """
    One sequential queue per GPU. This avoids multiple jobs fighting on one T4.
    Each bucket runs serially; the two buckets run concurrently.
    """
    devices = WORK_DEVICES
    buckets = [[] for _ in devices]
    for i, job in enumerate(jobs):
        buckets[i % len(devices)].append(job)

    def run_bucket(device, bucket):
        out = []
        for job in bucket:
            out.append(worker(job, device))
        return out

    results = []
    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futs = [ex.submit(run_bucket, dev, bucket)
                for dev, bucket in zip(devices, buckets) if bucket]
        for fut in as_completed(futs):
            results.extend(fut.result())
    return results

def summarize_mean_sd(df, group_col, metrics):
    rows = []
    for group, g in df.groupby(group_col):
        row = {group_col: group, "n": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        corrected = min(1.0, (m-rank) * p[idx])
        running = max(running, corrected)
        adjusted[idx] = running
    return adjusted

def paired_dz(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    sd = d.std(ddof=1)
    return np.nan if sd == 0 else float(d.mean()/sd)

def safe_wilcoxon(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.allclose(x, y):
        return 0.0, 1.0
    w, p = stats.wilcoxon(x, y, alternative="two-sided")
    return float(w), float(p)

PRIMARY_METRICS = [
    "FND", "HND", "LND", "throughput",
    "Stability_Ratio", "Death_Transition_Period",
    "Throughput_per_Round", "Alive_Node_AUC",
    "Normalized_AUC", "CH_Transmissions_per_Round",
    "ResidualEnergy_R2000", "AliveNodes_R2000",
]

# ----------------------------------------------------------------------
# 3. WEIGHT SENSITIVITY: 7 x 20 = 140
# ----------------------------------------------------------------------
print("\n" + "="*90)
print("A) WEIGHT SENSITIVITY — K=15")
print("="*90)

weight_raw_path = WEIGHT_DIR / "weight_sensitivity_raw.csv"
weight_df = load_csv(weight_raw_path)
done = completed_keys(weight_df, ["weight_config", "topology_seed"])

for cfg_name, weights in WEIGHT_CONFIGS.items():
    set_fitness_weights(weights)
    jobs = []
    for topo, opt_seed in zip(TOPOLOGY_SEEDS, OPTIMIZER_SEEDS):
        if (cfg_name, topo) not in done:
            jobs.append((cfg_name, topo, opt_seed))

    if not jobs:
        print(f"[resume] {cfg_name}: already complete.")
        continue

    print(f"{cfg_name}: {len(jobs)} run(s) remaining")

    def weight_worker(job, device):
        cfg_name_, topo_, opt_seed_ = job
        row, _ = run_lifetime_one(
            topo_, opt_seed_, "improved_cwoa", device, k=FINAL_K
        )
        row["weight_config"] = cfg_name_
        row["w_energy"], row["w_intra"], row["w_sink"], row["w_load"] = weights
        return row

    # Since CFG is common global state, all jobs in this block use same weight.
    new_rows = queue_jobs_two_gpus(jobs, weight_worker)
    weight_df = pd.concat([weight_df, pd.DataFrame(new_rows)], ignore_index=True)
    weight_df = weight_df.drop_duplicates(
        subset=["weight_config", "topology_seed"], keep="last"
    ).sort_values(["weight_config", "topology_seed"])
    atomic_csv(weight_df, weight_raw_path)
    done = completed_keys(weight_df, ["weight_config", "topology_seed"])
    print(f"  checkpoint saved: {len(weight_df)}/140")

# restore final weights
set_fitness_weights(FINAL_WEIGHTS)

weight_summary = summarize_mean_sd(
    weight_df, "weight_config", PRIMARY_METRICS
)
atomic_csv(weight_summary, WEIGHT_DIR / "weight_sensitivity_mean_sd.csv")

# Friedman across 7 weight configurations, paired by topology
weight_friedman_rows = []
for metric in PRIMARY_METRICS:
    piv = weight_df.pivot(
        index="topology_seed", columns="weight_config", values=metric
    ).reindex(columns=list(WEIGHT_CONFIGS)).dropna()
    if len(piv) == 20:
        statv, pv = stats.friedmanchisquare(
            *[piv[c].to_numpy() for c in piv.columns]
        )
        weight_friedman_rows.append({
            "Metric": metric,
            "Friedman_statistic": float(statv),
            "Friedman_p": float(pv),
        })
atomic_csv(pd.DataFrame(weight_friedman_rows),
           WEIGHT_DIR / "weight_sensitivity_friedman.csv")

# Baseline vs each alternative, Holm per metric
weight_pair_rows = []
base_name = "Baseline_35_30_20_15"
alts = [x for x in WEIGHT_CONFIGS if x != base_name]
for metric in PRIMARY_METRICS:
    piv = weight_df.pivot(
        index="topology_seed", columns="weight_config", values=metric
    ).dropna()
    block = []
    for alt in alts:
        x = piv[base_name].to_numpy()
        y = piv[alt].to_numpy()
        w, p = safe_wilcoxon(x, y)
        block.append({
            "Metric": metric,
            "Comparison": f"{base_name} vs {alt}",
            "W": w, "Raw_p": p,
            "Effect_dz_baseline_minus_alt": paired_dz(x, y),
            "Mean_Difference_baseline_minus_alt": float((x-y).mean()),
        })
    block_df = pd.DataFrame(block)
    block_df["Holm_Adjusted_p"] = holm_adjust(block_df["Raw_p"])
    block_df["Significant_0.05"] = block_df["Holm_Adjusted_p"] < 0.05
    weight_pair_rows.append(block_df)
weight_pair_df = pd.concat(weight_pair_rows, ignore_index=True)
atomic_csv(weight_pair_df, WEIGHT_DIR / "weight_sensitivity_pairwise_Holm.csv")

print("Weight sensitivity complete:", len(weight_df), "/ 140")

# ----------------------------------------------------------------------
# 4. K=12/15/18 CONFIRMATORY: 3 x 20 = 60
# ----------------------------------------------------------------------
print("\n" + "="*90)
print("B) K=12/15/18 CONFIRMATORY LIFETIME STUDY")
print("="*90)

set_fitness_weights(FINAL_WEIGHTS)
k_raw_path = K_DIR / "K12_K15_K18_confirmatory_raw.csv"
k_df = load_csv(k_raw_path)
done_k = completed_keys(k_df, ["k", "topology_seed"])

for k in K_CONFIRM:
    jobs = []
    for topo, opt_seed in zip(TOPOLOGY_SEEDS, OPTIMIZER_SEEDS):
        if (k, topo) not in done_k:
            jobs.append((k, topo, opt_seed))

    if not jobs:
        print(f"[resume] K={k}: already complete.")
        continue

    print(f"K={k}: {len(jobs)} run(s) remaining")

    def k_worker(job, device):
        k_, topo_, opt_seed_ = job
        row, _ = run_lifetime_one(
            topo_, opt_seed_, "improved_cwoa", device, k=k_
        )
        row["k"] = int(k_)
        return row

    new_rows = queue_jobs_two_gpus(jobs, k_worker)
    k_df = pd.concat([k_df, pd.DataFrame(new_rows)], ignore_index=True)
    k_df = k_df.drop_duplicates(
        subset=["k", "topology_seed"], keep="last"
    ).sort_values(["k", "topology_seed"])
    atomic_csv(k_df, k_raw_path)
    done_k = completed_keys(k_df, ["k", "topology_seed"])
    print(f"  checkpoint saved: {len(k_df)}/60")

k_summary = summarize_mean_sd(k_df, "k", PRIMARY_METRICS)
atomic_csv(k_summary, K_DIR / "K12_K15_K18_confirmatory_mean_sd.csv")

# paired Friedman + pairwise Wilcoxon across K
k_friedman_rows = []
k_pair_blocks = []
for metric in PRIMARY_METRICS:
    piv = k_df.pivot(index="topology_seed", columns="k", values=metric).dropna()
    if all(k in piv.columns for k in K_CONFIRM):
        statv, pv = stats.friedmanchisquare(
            *[piv[k].to_numpy() for k in K_CONFIRM]
        )
        k_friedman_rows.append({
            "Metric": metric,
            "Friedman_statistic": float(statv),
            "Friedman_p": float(pv),
        })
        pairs = [(12,15),(12,18),(15,18)]
        block = []
        for a,b in pairs:
            x,y = piv[a].to_numpy(), piv[b].to_numpy()
            w,p = safe_wilcoxon(x,y)
            block.append({
                "Metric": metric, "Comparison": f"K{a} vs K{b}",
                "W": w, "Raw_p": p,
                "Effect_dz_Ka_minus_Kb": paired_dz(x,y),
                "Mean_Difference_Ka_minus_Kb": float((x-y).mean()),
            })
        bdf = pd.DataFrame(block)
        bdf["Holm_Adjusted_p"] = holm_adjust(bdf["Raw_p"])
        bdf["Significant_0.05"] = bdf["Holm_Adjusted_p"] < 0.05
        k_pair_blocks.append(bdf)

atomic_csv(pd.DataFrame(k_friedman_rows), K_DIR / "K12_K15_K18_friedman.csv")
atomic_csv(pd.concat(k_pair_blocks, ignore_index=True),
           K_DIR / "K12_K15_K18_pairwise_Holm.csv")

print("K-confirmatory complete:", len(k_df), "/ 60")

# ----------------------------------------------------------------------
# 5. FINAL FIVE-METHOD COMPARISON: 5 x 20 = 100
# ----------------------------------------------------------------------
print("\n" + "="*90)
print("C) FINAL FIVE-METHOD COMPARISON")
print("="*90)

set_fitness_weights(FINAL_WEIGHTS)

final_raw_path = FINAL_DIR / "FINAL_100_RUNS.csv"
final_df = load_csv(final_raw_path)
done_final = completed_keys(final_df, ["method", "topology_seed"])

jobs = []
for method in FINAL_METHODS:
    for topo, opt_seed in zip(TOPOLOGY_SEEDS, OPTIMIZER_SEEDS):
        if (method, topo) not in done_final:
            jobs.append((method, topo, opt_seed))

print("Remaining final-comparison runs:", len(jobs))

def final_worker(job, device):
    method, topo, opt_seed = job
    row, hist = run_lifetime_one(topo, opt_seed, method, device, k=FINAL_K)

    hpath = HIST_DIR / f"{method}_topo{topo}_seed{opt_seed}.csv.gz"
    hist.to_csv(hpath, index=False, compression="gzip")
    row["history_file"] = str(hpath)
    return row

# Save after EACH result here for stronger crash resistance.
if jobs:
    devices = WORK_DEVICES
    buckets = [[] for _ in devices]
    for i, job in enumerate(jobs):
        buckets[i % len(devices)].append(job)

    def final_bucket(device, bucket):
        local = []
        for job in bucket:
            try:
                row = final_worker(job, device)
                local.append(row)

                # per-run emergency checkpoint
                method, topo, opt_seed = job
                emergency = FINAL_DIR / "per_run_checkpoints"
                emergency.mkdir(exist_ok=True)
                atomic_json(row, emergency / f"{method}_topo{topo}.json")

                print(
                    f"[{device}] {method:16s} topo={topo} "
                    f"FND={row['FND']} HND={row['HND']} LND={row['LND']}"
                )
            except Exception as e:
                print(f"FAILED {job} on {device}: {e}")
                traceback.print_exc()
                raise
        return local

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futs = [ex.submit(final_bucket, d, b)
                for d,b in zip(devices,buckets) if b]
        for fut in as_completed(futs):
            rows = fut.result()
            final_df = pd.concat([final_df, pd.DataFrame(rows)], ignore_index=True)
            final_df = final_df.drop_duplicates(
                subset=["method","topology_seed"], keep="last"
            ).sort_values(["method","topology_seed"])
            atomic_csv(final_df, final_raw_path)
            print("Main final checkpoint:", len(final_df), "/100")

# Recover any emergency checkpoints that were written after last main checkpoint
emergency = FINAL_DIR / "per_run_checkpoints"
if emergency.exists():
    existing = completed_keys(final_df, ["method","topology_seed"])
    recovered = []
    for jp in emergency.glob("*.json"):
        try:
            row = json.loads(jp.read_text())
            key = (row["method"], int(row["topology_seed"]))
            if key not in existing:
                recovered.append(row)
                existing.add(key)
        except Exception:
            pass
    if recovered:
        final_df = pd.concat([final_df, pd.DataFrame(recovered)], ignore_index=True)
        final_df = final_df.drop_duplicates(
            subset=["method","topology_seed"], keep="last"
        ).sort_values(["method","topology_seed"])
        atomic_csv(final_df, final_raw_path)

if len(final_df) != 100:
    print(f"WARNING: final comparison currently has {len(final_df)}/100 runs.")
else:
    print("Final 5-method comparison complete: 100/100")

# ----------------------------------------------------------------------
# 6. FINAL STATISTICS
# ----------------------------------------------------------------------
if len(final_df) == 100:
    summary_rows = []
    for method in FINAL_METHODS:
        g = final_df[final_df["method"] == method]
        row = {"method": method, "Method": METHOD_LABELS[method], "n": len(g)}
        for metric in PRIMARY_METRICS:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1))
        row["optimizer_evaluations_mean"] = float(g["optimizer_evaluations"].mean())
        row["runtime_s_mean"] = float(g["runtime_s"].mean())
        summary_rows.append(row)
    final_summary = pd.DataFrame(summary_rows)
    atomic_csv(final_summary, FINAL_DIR / "FINAL_MEAN_SD.csv")

    friedman_rows = []
    pairwise_blocks = []

    for metric in PRIMARY_METRICS:
        piv = final_df.pivot(
            index="topology_seed", columns="method", values=metric
        ).reindex(columns=FINAL_METHODS).dropna()

        statv, pv = stats.friedmanchisquare(
            *[piv[m].to_numpy() for m in FINAL_METHODS]
        )
        friedman_rows.append({
            "Metric": metric,
            "Friedman_statistic": float(statv),
            "Friedman_p": float(pv),
        })

        x = piv["improved_cwoa"].to_numpy()
        block = []
        for baseline in ["leach","random_search","original_cwoa","pso"]:
            y = piv[baseline].to_numpy()
            w,p = safe_wilcoxon(x,y)
            block.append({
                "Metric": metric,
                "Comparison": f"AD-CWOA vs {METHOD_LABELS[baseline]}",
                "W": w,
                "Raw_p": p,
                "Effect_dz_AD_minus_baseline": paired_dz(x,y),
                "Mean_Difference_AD_minus_baseline": float((x-y).mean()),
                "Percent_Difference_AD_vs_baseline": (
                    float((x.mean()-y.mean())/y.mean()*100.0)
                    if y.mean() != 0 else np.nan
                ),
            })
        bdf = pd.DataFrame(block)
        bdf["Holm_Adjusted_p"] = holm_adjust(bdf["Raw_p"])
        bdf["Significant_0.05"] = bdf["Holm_Adjusted_p"] < 0.05
        pairwise_blocks.append(bdf)

    friedman_df = pd.DataFrame(friedman_rows)
    pairwise_df = pd.concat(pairwise_blocks, ignore_index=True)
    atomic_csv(friedman_df, FINAL_DIR / "FINAL_Friedman_5_methods.csv")
    atomic_csv(pairwise_df, FINAL_DIR / "FINAL_AD_CWOA_pairwise_Holm.csv")

    # Excel reviewer workbook
    xlsx_path = FINAL_DIR / "FINAL_5_METHOD_REVIEWER_RESULTS.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="Raw_100_Runs", index=False)
        final_summary.to_excel(writer, sheet_name="Mean_SD", index=False)
        friedman_df.to_excel(writer, sheet_name="Friedman", index=False)
        pairwise_df.to_excel(writer, sheet_name="AD_vs_Baselines", index=False)
        weight_summary.to_excel(writer, sheet_name="Weight_Mean_SD", index=False)
        k_summary.to_excel(writer, sheet_name="K12_15_18_Mean_SD", index=False)

    print("\nFINAL MEAN ± SD")
    display(final_summary)
    print("\nFINAL FRIEDMAN")
    display(friedman_df)
    print("\nAD-CWOA vs BASELINES")
    display(pairwise_df)

# ----------------------------------------------------------------------
# 7. COMPLETION MANIFEST + ZIP
# ----------------------------------------------------------------------
manifest = {
    "weight_runs": int(len(weight_df)),
    "k_confirmatory_runs": int(len(k_df)),
    "final_method_runs": int(len(final_df)),
    "expected_total": 300,
    "completed_total": int(len(weight_df) + len(k_df) + len(final_df)),
    "final_config": config,
}
atomic_json(manifest, ROOT / "RUN_MANIFEST.json")

zip_base = "/kaggle/working/AD_CWOA_MAJOR_REVISION_RERUN"
archive = shutil.make_archive(
    "/kaggle/working/AD_CWOA_MAJOR_REVISION_RERUN_PACKAGE",
    "zip",
    ROOT
)

print("\n" + "="*90)
print("MASTER ANALYSIS FINISHED / CHECKPOINTED")
print("="*90)
print("Weight sensitivity:", len(weight_df), "/ 140")
print("K confirmatory:", len(k_df), "/ 60")
print("Final five-method:", len(final_df), "/ 100")
print("Total:", len(weight_df)+len(k_df)+len(final_df), "/ 300")
print("Root:", ROOT)
print("ZIP:", archive)
print("="*90)

# restore final weights one last time
set_fitness_weights(FINAL_WEIGHTS)

# ===== Notebook code cell 3 =====
from pathlib import Path
import os

paths = [
    "/kaggle/working/AD_CWOA_MAJOR_REVISION_RERUN",
    "/kaggle/working/AD_CWOA_MAJOR_REVISION_RERUN_PACKAGE.zip",
]

print("=" * 80)
print("CHECKING SAVED OUTPUTS")
print("=" * 80)

for p in paths:
    x = Path(p)
    print(f"\n{p}")
    print("Exists:", x.exists())

    if x.exists():
        if x.is_file():
            print("Size:", round(x.stat().st_size / 1024**2, 2), "MB")
        else:
            files = list(x.rglob("*"))
            print("Files:", sum(f.is_file() for f in files))

print("\n/kaggle/working contents:")
for x in Path("/kaggle/working").iterdir():
    print(" -", x.name)

# ===== Notebook code cell 4 =====
from pathlib import Path
import pandas as pd

p = Path("/kaggle/working/AD_CWOA_SAFE_RERUN/MASTER_ALL_RUNS.csv")

if p.exists():
    df = pd.read_csv(p)
    print("TOTAL SAVED:", len(df))
    print("\nBATCH COUNTS:")
    print(df["batch_id"].value_counts())
else:
    print("MASTER FILE NOT FOUND")

# ===== Notebook code cell 5 =====
from pathlib import Path

for p in Path("/kaggle/input").rglob(
    "BATCH_01_A_Baseline_35_30_20_15_COMPLETE.zip"
):
    print("FOUND:", p)

# ===== Notebook code cell 6 =====
from pathlib import Path

root = Path("/kaggle/input")

print("KAGGLE INPUT EXISTS:", root.exists())
print("\nAVAILABLE INPUT FOLDERS / FILES:\n")

if root.exists():
    items = list(root.rglob("*"))

    if not items:
        print("❌ /kaggle/input is empty")
    else:
        for p in items[:200]:
            print(p)
else:
    print("❌ /kaggle/input does not exist")

# ===== Notebook code cell 7 =====
from pathlib import Path
import shutil
import pandas as pd

SRC = Path(
    "/kaggle/input/datasets/shivambhardwajshivi/"
    "ad-cwoa-batch1-backup/AD_CWOA_SAFE_RERUN"
)

DST = Path("/kaggle/working/AD_CWOA_SAFE_RERUN")

# Clean only old incomplete working copy
if DST.exists():
    shutil.rmtree(DST)

# Restore complete Batch-1 dataset
shutil.copytree(SRC, DST)

# ---------------------------------------------------------
# Kaggle dataset upload may wrap history CSVs inside folders
# like:
# histories/file.csv/file.csv
# Flatten them back to normal CSV files.
# ---------------------------------------------------------
hist_dir = DST / "histories"

if hist_dir.exists():
    nested_dirs = [p for p in hist_dir.iterdir() if p.is_dir() and p.name.endswith(".csv")]

    for folder in nested_dirs:
        inner = folder / folder.name

        if inner.exists() and inner.is_file():
            temp = hist_dir / (folder.name + ".restore_tmp")
            shutil.copy2(inner, temp)
            shutil.rmtree(folder)
            temp.rename(hist_dir / folder.name)

# Backup folder MUST remain outside experiment root
BACKUP_DIR = Path("/kaggle/working/AD_CWOA_SAFE_BACKUPS")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# VERIFY RESTORE
# ---------------------------------------------------------
master = DST / "MASTER_ALL_RUNS.csv"

if not master.exists():
    raise RuntimeError("MASTER_ALL_RUNS.csv missing after restore.")

df = pd.read_csv(master)

print("=" * 80)
print("BATCH-1 RESTORE SUCCESSFUL")
print("=" * 80)

print("TOTAL RECOVERED RUNS:", len(df))

print("\nBATCH COUNTS:")
print(df["batch_id"].value_counts())

print("\nHistory CSV files:", len(list(hist_dir.glob("*.csv"))))
print("Checkpoint files:", len(list((DST / "checkpoints").glob("*.json"))))

batch_csv = DST / "tables" / "01_A_Baseline_35_30_20_15_20_RUNS.csv"
mean_csv = DST / "tables" / "01_A_Baseline_35_30_20_15_MEAN_SD.csv"

print("\n20-RUN TABLE EXISTS :", batch_csv.exists())
print("MEAN-SD TABLE EXISTS:", mean_csv.exists())

print("\nRESTORED PATH:")
print(DST)

print("\nEXPECTED:")
print("Total recovered runs = 20")
print("Batch-1 count        = 20")
print("History CSV files    = 20")
print("Checkpoint files     = 20")

# ===== Notebook code cell 8 =====
# ======================================================================
# AD-CWOA FINAL K=15 — STANDALONE FRESH-RUNTIME RECOVERY + VALIDATION
# Kaggle: enable GPU T4 x2, then run this entire file/cell once.
# This script does NOT run the final 100-run experiment.
# ======================================================================


# ============================================================
# 1. ENVIRONMENT + CONFIGURATION
# ============================================================
import os, sys, math, json, time, random, shutil, platform, subprocess, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn.functional as F

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)

GLOBAL_SEED = 2026
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

@dataclass(frozen=True)
class WSNConfig:
    area_x: float = 200.0
    area_y: float = 200.0
    num_nodes: int = 200
    initial_energy: float = 1.0
    packet_size: int = 4000
    e_elec: float = 50e-9
    e_fs: float = 10e-12
    e_mp: float = 0.0013e-12
    e_da: float = 5e-9
    sink_x: float = 100.0
    sink_y: float = 100.0
    w_energy: float = 0.35
    w_intra: float = 0.30
    w_sink: float = 0.20
    w_load: float = 0.15

@dataclass(frozen=True)
class OptimizerConfig:
    num_ch: int = 10
    population_size: int = 30
    evaluation_budget: int = 1500
    local_search_trials: int = 5
    stagnation_limit: int = 4
    restart_fraction: float = 0.30

@dataclass(frozen=True)
class ExperimentConfig:
    base_topology_seed: int = 5000
    single_optimizer_seed: int = 2026
    independent_runs: int = 20
    run_seed_start: int = 2000
    topology_runs: int = 20
    topology_seed_start: int = 5000
    topology_optimizer_seed_start: int = 8000
    lifetime_max_rounds: int = 5000
    lifetime_reselection_interval: int = 10
    lifetime_evaluation_budget: int = 300
    lifetime_seed: int = 9500

CFG = WSNConfig()
OPT = OptimizerConfig()
EXP = ExperimentConfig()

QUICK_MODE = False
N_INDEPENDENT_RUNS = 3 if QUICK_MODE else EXP.independent_runs
N_TOPOLOGIES = 3 if QUICK_MODE else EXP.topology_runs
LIFETIME_MAX_ROUNDS = 1000 if QUICK_MODE else EXP.lifetime_max_rounds

D0 = math.sqrt(CFG.e_fs / CFG.e_mp)
NETWORK_DIAGONAL = math.hypot(CFG.area_x, CFG.area_y)
MAX_SINK_DISTANCE = math.hypot(
    max(CFG.sink_x, CFG.area_x - CFG.sink_x),
    max(CFG.sink_y, CFG.area_y - CFG.sink_y),
)

assert abs(CFG.w_energy + CFG.w_intra + CFG.w_sink + CFG.w_load - 1.0) < 1e-12
assert EXP.lifetime_evaluation_budget % OPT.population_size == 0

if Path("/kaggle/working").exists():
    OUTPUT_ROOT = Path("/kaggle/working/CWOA_WSN_Research")
else:
    OUTPUT_ROOT = Path.cwd() / "CWOA_WSN_Research"

FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
HTML_DIR = OUTPUT_ROOT / "interactive_html"
for p in [OUTPUT_ROOT, FIG_DIR, TABLE_DIR, HTML_DIR]:
    p.mkdir(parents=True, exist_ok=True)

GPU_DEVICES = (
    [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)
WORK_DEVICES = GPU_DEVICES[:2]

METHOD_ORDER = ["original_cwoa", "improved_cwoa", "random_search"]
METHOD_LABELS = {
    "original_cwoa": "Original CWOA",
    "improved_cwoa": "Improved CWOA",
    "random_search": "Random Search",
}
METHOD_COLORS = {
    "original_cwoa": "#0072B2",
    "improved_cwoa": "#009E73",
    "random_search": "#D55E00",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def sync_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def start_measurement(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    return time.perf_counter()

def end_measurement(device, t0):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak = 0.0
    return time.perf_counter() - t0, peak

def save_paper_figure(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIG_DIR / f"{stem}.{ext}", **kwargs)

def save_interactive(fig, stem):
    fig.write_html(HTML_DIR / f"{stem}.html", include_plotlyjs=True)

with open(OUTPUT_ROOT / "experiment_config.json", "w") as f:
    json.dump(
        {
            "wsn": asdict(CFG),
            "optimizer": asdict(OPT),
            "experiment": asdict(EXP),
            "quick_mode": QUICK_MODE,
            "d0_m": D0,
        },
        f,
        indent=2,
    )

print("=" * 70)
print("GPU ENVIRONMENT")
print("=" * 70)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
for d in GPU_DEVICES:
    if d.type == "cuda":
        p = torch.cuda.get_device_properties(d)
        print(f"{d}: {p.name} | {p.total_memory / 1024**3:.2f} GB")
    else:
        print("cpu")
print("Work devices:", WORK_DEVICES)
print("Output:", OUTPUT_ROOT)

# ============================================================
# 2. WSN MODEL + VECTORIZED COMMUNICATION ENGINE
# ============================================================
def create_sensor_network(topology_seed, num_nodes=None):
    n = CFG.num_nodes if num_nodes is None else int(num_nodes)
    rng = np.random.default_rng(int(topology_seed))
    network = pd.DataFrame({
        "node_id": np.arange(1, n + 1, dtype=int),
        "x": rng.uniform(0, CFG.area_x, n),
        "y": rng.uniform(0, CFG.area_y, n),
        "energy": np.full(n, CFG.initial_energy, dtype=np.float64),
        "alive": np.ones(n, dtype=bool),
        "is_CH": np.zeros(n, dtype=bool),
    })
    network["distance_to_sink"] = np.hypot(
        network["x"].to_numpy() - CFG.sink_x,
        network["y"].to_numpy() - CFG.sink_y,
    )
    return network

def transmission_energy(bits, distance):
    d = np.asarray(distance, dtype=np.float64)
    e = np.where(
        d < D0,
        bits * CFG.e_elec + bits * CFG.e_fs * d**2,
        bits * CFG.e_elec + bits * CFG.e_mp * d**4,
    )
    return float(e) if e.ndim == 0 else e

def reception_energy(bits):
    return bits * CFG.e_elec

def aggregation_energy(bits):
    return bits * CFG.e_da

def assign_nodes_to_cluster_heads(nodes_df, ch_indices):
    result = nodes_df.copy()
    result["is_CH"] = False
    result["cluster_id"] = -1
    result["ch_index"] = -1
    result["ch_node_id"] = -1
    result["distance_to_ch"] = np.nan

    ch_indices = np.asarray(ch_indices, dtype=int)
    alive_idx = result.index[result["alive"]].to_numpy(dtype=int)

    if len(ch_indices) == 0 or len(alive_idx) == 0:
        return result
    if len(np.unique(ch_indices)) != len(ch_indices):
        raise ValueError("Cluster-head indices must be unique.")
    if not result.loc[ch_indices, "alive"].all():
        raise ValueError("All selected cluster heads must be alive.")

    result.loc[ch_indices, "is_CH"] = True

    alive_xy = result.loc[alive_idx, ["x", "y"]].to_numpy(np.float64)
    ch_xy = result.loc[ch_indices, ["x", "y"]].to_numpy(np.float64)
    dist = np.sqrt(np.sum((alive_xy[:, None, :] - ch_xy[None, :, :]) ** 2, axis=2))
    nearest = np.argmin(dist, axis=1)
    chosen = ch_indices[nearest]

    result.loc[alive_idx, "cluster_id"] = nearest
    result.loc[alive_idx, "ch_index"] = chosen
    result.loc[alive_idx, "ch_node_id"] = result.loc[chosen, "node_id"].to_numpy(dtype=int)
    result.loc[alive_idx, "distance_to_ch"] = dist[np.arange(len(alive_idx)), nearest]
    return result

def simulate_communication_round(nodes_df, ch_indices):
    network = nodes_df.copy()
    network["alive"] = network["energy"].to_numpy() > 0.0
    alive_before = int(network["alive"].sum())

    ch_indices = np.asarray(
        [int(i) for i in ch_indices if network.at[int(i), "alive"]],
        dtype=int,
    )

    if len(ch_indices) == 0:
        return network, {
            "packets_to_ch": 0,
            "ch_transmissions_to_bs": 0,
            "source_packets_delivered_to_bs": 0,
            "energy_consumed": 0.0,
            "residual_energy": float(network["energy"].sum()),
            "alive_nodes": alive_before,
            "dead_nodes": len(network) - alive_before,
        }

    clustered = assign_nodes_to_cluster_heads(network, ch_indices)
    energy_before = float(clustered["energy"].sum())

    alive_mask = clustered["alive"].to_numpy()
    ch_mask = clustered["is_CH"].to_numpy()
    member_idx = np.flatnonzero(alive_mask & ~ch_mask)

    packets_to_ch = 0
    received = np.zeros(len(ch_indices), dtype=int)

    if len(member_idx):
        d = clustered.loc[member_idx, "distance_to_ch"].to_numpy(np.float64)
        costs = transmission_energy(CFG.packet_size, d)
        current = clustered.loc[member_idx, "energy"].to_numpy(np.float64)
        success = current >= costs
        clustered.loc[member_idx, "energy"] = np.where(success, current - costs, 0.0)
        packets_to_ch = int(success.sum())
        cluster_ids = clustered.loc[member_idx, "cluster_id"].to_numpy(dtype=int)
        received = np.bincount(cluster_ids[success], minlength=len(ch_indices))

    ch_energy = clustered.loc[ch_indices, "energy"].to_numpy(np.float64)
    source_packets = received + 1
    rx_cost = received * reception_energy(CFG.packet_size)
    da_cost = source_packets * aggregation_energy(CFG.packet_size)
    sink_d = clustered.loc[ch_indices, "distance_to_sink"].to_numpy(np.float64)
    tx_sink = transmission_energy(CFG.packet_size, sink_d)
    total_ch_cost = rx_cost + da_cost + tx_sink
    ch_success = (ch_energy > 0.0) & (ch_energy >= total_ch_cost)

    clustered.loc[ch_indices, "energy"] = np.where(
        ch_success, ch_energy - total_ch_cost, 0.0
    )

    clustered["energy"] = clustered["energy"].clip(lower=0.0)
    clustered["alive"] = clustered["energy"] > 0.0
    residual = float(clustered["energy"].sum())

    return clustered, {
        "packets_to_ch": packets_to_ch,
        "ch_transmissions_to_bs": int(ch_success.sum()),
        "source_packets_delivered_to_bs": int(source_packets[ch_success].sum()),
        "energy_consumed": energy_before - residual,
        "residual_energy": residual,
        "alive_nodes": int(clustered["alive"].sum()),
        "dead_nodes": int((~clustered["alive"]).sum()),
    }

base_nodes = create_sensor_network(EXP.base_topology_seed)
display(base_nodes.head())
print("Base topology mean sink distance:", base_nodes["distance_to_sink"].mean())

# ============================================================
# 3. BATCHED GPU FITNESS EVALUATOR + NUMERICAL VALIDATION
# ============================================================
class GPUFitnessEvaluator:
    def __init__(self, nodes_df, device):
        self.nodes_df = nodes_df
        self.device = torch.device(device)
        self.positions = torch.as_tensor(
            nodes_df[["x", "y"]].to_numpy(np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        self.energy = torch.as_tensor(
            nodes_df["energy"].to_numpy(np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        self.alive_mask = torch.as_tensor(
            nodes_df["alive"].to_numpy(dtype=bool),
            dtype=torch.bool,
            device=self.device,
        )
        self.alive_idx = torch.nonzero(self.alive_mask, as_tuple=False).squeeze(1)
        self.sink_distance = torch.sqrt(
            (self.positions[:, 0] - CFG.sink_x) ** 2
            + (self.positions[:, 1] - CFG.sink_y) ** 2
        )

    @property
    def alive_count(self):
        return int(self.alive_idx.numel())

    def evaluate_batch(self, ch_batch, return_details=False):
        ch = torch.as_tensor(ch_batch, dtype=torch.long, device=self.device)
        if ch.ndim == 1:
            ch = ch.unsqueeze(0)

        B, K = ch.shape
        M = self.alive_count
        if M < K:
            raise ValueError("Not enough alive nodes for requested CH count.")

        valid = self.alive_mask[ch].all(dim=1)
        if K > 1:
            s = torch.sort(ch, dim=1).values
            valid &= (s[:, 1:] != s[:, :-1]).all(dim=1)

        ch_xy = self.positions[ch]
        alive_xy = self.positions[self.alive_idx]
        diff = alive_xy.unsqueeze(0).unsqueeze(2) - ch_xy.unsqueeze(1)
        distances = torch.sqrt(torch.sum(diff * diff, dim=-1).clamp_min(0.0))
        min_distance, assignment = torch.min(distances, dim=2)

        mean_intra = (
            min_distance.sum(dim=1) / float(M - K)
            if M > K
            else torch.zeros(B, device=self.device)
        )
        intra_penalty = mean_intra / NETWORK_DIAGONAL

        normalized_energy = (
            self.energy[ch].clamp(0.0, CFG.initial_energy) / CFG.initial_energy
        )
        energy_penalty = 1.0 - normalized_energy.mean(dim=1)

        mean_sink = self.sink_distance[ch].mean(dim=1)
        sink_penalty = mean_sink / MAX_SINK_DISTANCE

        # Deterministic counts; avoids CUDA atomic scatter-add nondeterminism.
        cluster_sizes = F.one_hot(assignment, num_classes=K).sum(dim=1).to(torch.float32)
        load_penalty = cluster_sizes.std(dim=1, unbiased=False) / (float(M) / float(K))

        fitness = (
            CFG.w_energy * energy_penalty
            + CFG.w_intra * intra_penalty
            + CFG.w_sink * sink_penalty
            + CFG.w_load * load_penalty
        )
        fitness = torch.where(valid, fitness, torch.full_like(fitness, float("inf")))

        if not return_details:
            return fitness

        return fitness, {
            "energy_penalty": energy_penalty,
            "mean_intra_distance": mean_intra,
            "intra_penalty": intra_penalty,
            "mean_ch_sink_distance": mean_sink,
            "sink_penalty": sink_penalty,
            "cluster_sizes": cluster_sizes,
            "load_penalty": load_penalty,
        }

def details_to_python(details, row=0):
    out = {}
    for k, v in details.items():
        x = v[row].detach().cpu().numpy() if torch.is_tensor(v) else v
        out[k] = float(x) if np.ndim(x) == 0 else x
    return out

def decode_whales_gpu(whales, evaluator, num_ch):
    scores = whales.index_select(1, evaluator.alive_idx)
    top_pos = torch.topk(scores, k=num_ch, dim=1, largest=True, sorted=False).indices
    return evaluator.alive_idx[top_pos]

def evaluate_candidate_cpu_reference(nodes_df, ch_indices):
    ch = np.asarray(ch_indices, dtype=int)
    alive_idx = nodes_df.index[nodes_df["alive"]].to_numpy(dtype=int)
    alive_xy = nodes_df.loc[alive_idx, ["x", "y"]].to_numpy(np.float64)
    ch_xy = nodes_df.loc[ch, ["x", "y"]].to_numpy(np.float64)
    d = np.sqrt(np.sum((alive_xy[:, None, :] - ch_xy[None, :, :]) ** 2, axis=2))
    assign = np.argmin(d, axis=1)
    min_d = np.min(d, axis=1)
    mean_intra = min_d.sum() / (len(alive_idx) - len(ch)) if len(alive_idx) > len(ch) else 0.0
    energy_penalty = 1.0 - np.mean(
        np.clip(nodes_df.loc[ch, "energy"].to_numpy() / CFG.initial_energy, 0.0, 1.0)
    )
    mean_sink = nodes_df.loc[ch, "distance_to_sink"].mean()
    cluster_sizes = np.bincount(assign, minlength=len(ch))
    load_penalty = np.std(cluster_sizes) / np.mean(cluster_sizes)
    return (
        CFG.w_energy * energy_penalty
        + CFG.w_intra * (mean_intra / NETWORK_DIAGONAL)
        + CFG.w_sink * (mean_sink / MAX_SINK_DISTANCE)
        + CFG.w_load * load_penalty
    )

rng = np.random.default_rng(12345)

# Validate more than the all-1-J initial state:
# vary residual energy and mark a subset of nodes dead.
validation_nodes = base_nodes.copy()
validation_nodes["energy"] = rng.uniform(
    0.15, CFG.initial_energy, len(validation_nodes)
)
dead_idx = rng.choice(
    validation_nodes.index,
    size=max(1, len(validation_nodes) // 10),
    replace=False,
)
validation_nodes.loc[dead_idx, "energy"] = 0.0
validation_nodes["alive"] = validation_nodes["energy"] > 0.0

alive = validation_nodes.index[
    validation_nodes["alive"]
].to_numpy(dtype=int)

validation_candidates = np.vstack([
    rng.choice(alive, size=OPT.num_ch, replace=False)
    for _ in range(128)
])

cpu_values = np.array([
    evaluate_candidate_cpu_reference(validation_nodes, ch)
    for ch in validation_candidates
])

validator = GPUFitnessEvaluator(validation_nodes, WORK_DEVICES[0])
gpu_values = validator.evaluate_batch(validation_candidates).cpu().numpy()
abs_error = np.abs(cpu_values - gpu_values)

validation_df = pd.DataFrame([{
    "Max abs error": abs_error.max(),
    "Mean abs error": abs_error.mean(),
    "Max relative error": np.max(abs_error / np.maximum(np.abs(cpu_values), 1e-12)),
}])
display(validation_df)
assert abs_error.max() < 5e-5
print("PASS: GPU fitness matches CPU reference within FP32 tolerance.")

# ============================================================
# 4. GPU OPTIMIZERS
# ============================================================
def run_cwoa_gpu(nodes_df, num_ch=10, population_size=30, max_iterations=50,
                 seed=2026, device=None, verbose=False):
    device = WORK_DEVICES[0] if device is None else torch.device(device)
    evaluator = GPUFitnessEvaluator(nodes_df, device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n = len(nodes_df)
    t0 = start_measurement(device)

    whales = torch.rand((population_size, n), generator=gen, device=device)
    best_fit = float("inf")
    best_whale = None
    best_ch = None
    convergence = []

    for iteration in range(max_iterations):
        ch_batch = decode_whales_gpu(whales, evaluator, num_ch)
        fit = evaluator.evaluate_batch(ch_batch)
        value, idx = torch.min(fit, dim=0)

        if float(value) < best_fit:
            best_fit = float(value)
            best_whale = whales[int(idx)].clone()
            best_ch = ch_batch[int(idx)].clone()

        convergence.append(best_fit)
        a = 0.0 if max_iterations <= 1 else 2.0 - 2.0 * iteration / (max_iterations - 1)

        r1 = torch.rand((population_size, 1), generator=gen, device=device)
        r2 = torch.rand((population_size, 1), generator=gen, device=device)
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        p = torch.rand((population_size, 1), generator=gen, device=device)

        ridx = torch.randint(0, population_size, (population_size,), generator=gen, device=device)
        random_whales = whales[ridx]

        exploit = best_whale.unsqueeze(0) - A * torch.abs(C * best_whale.unsqueeze(0) - whales)
        explore = random_whales - A * torch.abs(C * random_whales - whales)
        encircle = torch.where(torch.abs(A) < 1.0, exploit, explore)

        l = -1.0 + 2.0 * torch.rand((population_size, 1), generator=gen, device=device)
        spiral = (
            torch.abs(best_whale.unsqueeze(0) - whales)
            * torch.exp(l)
            * torch.cos(2.0 * math.pi * l)
            + best_whale.unsqueeze(0)
        )
        whales = torch.where(p < 0.5, encircle, spiral).clamp_(0.0, 1.0)

        if verbose:
            print(iteration + 1, best_fit)

    runtime_s, peak_mb = end_measurement(device, t0)
    _, details_t = evaluator.evaluate_batch(best_ch, return_details=True)

    return {
        "method": "original_cwoa",
        "ch_indices": best_ch.cpu().numpy(),
        "fitness": best_fit,
        "convergence_x": np.arange(1, max_iterations + 1) * population_size,
        "convergence_y": np.asarray(convergence),
        "details": details_to_python(details_t),
        "evaluations": population_size * max_iterations,
        "runtime_s": runtime_s,
        "peak_gpu_memory_mb": peak_mb,
        "device": str(device),
    }

def run_random_search_gpu(nodes_df, num_ch=10, evaluations=1500, seed=2026,
                          device=None, batch_size=512, return_convergence=True):
    device = WORK_DEVICES[0] if device is None else torch.device(device)
    evaluator = GPUFitnessEvaluator(nodes_df, device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    alive_idx = evaluator.alive_idx
    t0 = start_measurement(device)

    best_fit = float("inf")
    best_ch = None
    convergence = []
    done = 0

    while done < evaluations:
        b = min(batch_size, evaluations - done)
        keys = torch.rand((b, len(alive_idx)), generator=gen, device=device)
        pos = torch.topk(keys, k=num_ch, dim=1, largest=True, sorted=False).indices
        ch_batch = alive_idx[pos]
        fit = evaluator.evaluate_batch(ch_batch)
        value, idx = torch.min(fit, dim=0)

        if float(value) < best_fit:
            best_fit = float(value)
            best_ch = ch_batch[int(idx)].clone()

        if return_convergence:
            running = torch.cummin(fit, dim=0).values
            if convergence:
                running = torch.minimum(running, torch.tensor(convergence[-1], device=device))
            convergence.extend(running.cpu().numpy().tolist())

        done += b

    runtime_s, peak_mb = end_measurement(device, t0)
    _, details_t = evaluator.evaluate_batch(best_ch, return_details=True)

    return {
        "method": "random_search",
        "ch_indices": best_ch.cpu().numpy(),
        "fitness": best_fit,
        "convergence_x": np.arange(1, evaluations + 1) if return_convergence else np.array([]),
        "convergence_y": np.asarray(convergence) if return_convergence else np.array([]),
        "details": details_to_python(details_t),
        "evaluations": evaluations,
        "runtime_s": runtime_s,
        "peak_gpu_memory_mb": peak_mb,
        "device": str(device),
    }

def generate_ch_swap_gpu(ch_indices, alive_idx, gen):
    candidate = ch_indices.clone()
    device = candidate.device
    non_ch = alive_idx[~torch.isin(alive_idx, candidate)]
    if len(non_ch) == 0:
        return candidate
    remove_pos = int(torch.randint(0, len(candidate), (1,), generator=gen, device=device).item())
    add_pos = int(torch.randint(0, len(non_ch), (1,), generator=gen, device=device).item())
    candidate[remove_pos] = non_ch[add_pos]
    return candidate

def encode_ch_to_random_keys_gpu(ch_indices, dimension, gen, device):
    whale = torch.empty(dimension, device=device)
    whale.uniform_(0.0, 0.45, generator=gen)
    high = torch.empty(len(ch_indices), device=device)
    high.uniform_(0.55, 1.0, generator=gen)
    whale[ch_indices] = high
    return whale

def run_improved_cwoa_gpu(nodes_df, num_ch=10, population_size=30, max_evaluations=1500,
                           local_search_trials=5, stagnation_limit=4, restart_fraction=0.30,
                           seed=2026, device=None, verbose=False):
    device = WORK_DEVICES[0] if device is None else torch.device(device)
    evaluator = GPUFitnessEvaluator(nodes_df, device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n = len(nodes_df)
    t0 = start_measurement(device)

    whales = torch.rand((population_size, n), generator=gen, device=device)
    fitness_values = torch.full((population_size,), float("inf"), device=device)

    initial_n = min(population_size, max_evaluations)
    ch_batch = decode_whales_gpu(whales[:initial_n], evaluator, num_ch)
    fit = evaluator.evaluate_batch(ch_batch)
    fitness_values[:initial_n] = fit
    value, idx = torch.min(fit, dim=0)

    best_fit = float(value)
    best_whale = whales[int(idx)].clone()
    best_ch = ch_batch[int(idx)].clone()
    evaluations = initial_n
    generation = 0
    stagnation = 0
    conv_x = [evaluations]
    conv_y = [best_fit]

    while evaluations < max_evaluations:
        generation += 1
        previous_best = best_fit
        progress = evaluations / max_evaluations
        a = 2.0 * (1.0 - progress)

        r1 = torch.rand((population_size, 1), generator=gen, device=device)
        r2 = torch.rand((population_size, 1), generator=gen, device=device)
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        p = torch.rand((population_size, 1), generator=gen, device=device)

        ridx = torch.randint(0, population_size, (population_size,), generator=gen, device=device)
        random_whales = whales[ridx]
        exploit = best_whale.unsqueeze(0) - A * torch.abs(C * best_whale.unsqueeze(0) - whales)
        explore = random_whales - A * torch.abs(C * random_whales - whales)
        encircle = torch.where(torch.abs(A) < 1.0, exploit, explore)

        l = -1.0 + 2.0 * torch.rand((population_size, 1), generator=gen, device=device)
        spiral = (
            torch.abs(best_whale.unsqueeze(0) - whales)
            * torch.exp(l)
            * torch.cos(2.0 * math.pi * l)
            + best_whale.unsqueeze(0)
        )

        new_whales = torch.where(p < 0.5, encircle, spiral)

        mutation_probability = 0.15 + 0.20 * progress
        mutation_count = max(1, int(0.05 * n))
        mutation_scale = 0.15 * (1.0 - progress) + 0.03

        for i in range(population_size):
            if float(torch.rand((), generator=gen, device=device)) < mutation_probability:
                midx = torch.randperm(n, generator=gen, device=device)[:mutation_count]
                new_whales[i, midx] += (
                    torch.randn((mutation_count,), generator=gen, device=device)
                    * mutation_scale
                )

        whales = new_whales.clamp_(0.0, 1.0)

        remaining = max_evaluations - evaluations
        eval_n = min(population_size, remaining)
        ch_batch = decode_whales_gpu(whales[:eval_n], evaluator, num_ch)
        fit = evaluator.evaluate_batch(ch_batch)
        fitness_values[:eval_n] = fit
        value, idx = torch.min(fit, dim=0)

        if float(value) < best_fit:
            best_fit = float(value)
            best_whale = whales[int(idx)].clone()
            best_ch = ch_batch[int(idx)].clone()

        evaluations += eval_n
        if evaluations >= max_evaluations:
            conv_x.append(evaluations)
            conv_y.append(best_fit)
            break

        for _ in range(local_search_trials):
            if evaluations >= max_evaluations:
                break
            candidate = generate_ch_swap_gpu(best_ch, evaluator.alive_idx, gen)
            candidate_fit = float(evaluator.evaluate_batch(candidate)[0])
            evaluations += 1

            if candidate_fit < best_fit:
                best_fit = candidate_fit
                best_ch = candidate.clone()
                best_whale = encode_ch_to_random_keys_gpu(best_ch, n, gen, device)
                worst = int(torch.argmax(fitness_values))
                whales[worst] = best_whale
                fitness_values[worst] = best_fit

        stagnation = 0 if best_fit < previous_best - 1e-12 else stagnation + 1

        if stagnation >= stagnation_limit:
            restart_count = max(1, int(population_size * restart_fraction))
            worst_idx = torch.topk(fitness_values, k=restart_count, largest=True).indices
            whales[worst_idx] = torch.rand((restart_count, n), generator=gen, device=device)
            fitness_values[worst_idx] = float("inf")
            stagnation = 0

        conv_x.append(evaluations)
        conv_y.append(best_fit)

        if verbose:
            print(generation, evaluations, best_fit)

    runtime_s, peak_mb = end_measurement(device, t0)
    _, details_t = evaluator.evaluate_batch(best_ch, return_details=True)

    return {
        "method": "improved_cwoa",
        "ch_indices": best_ch.cpu().numpy(),
        "fitness": best_fit,
        "convergence_x": np.asarray(conv_x),
        "convergence_y": np.asarray(conv_y),
        "details": details_to_python(details_t),
        "evaluations": evaluations,
        "runtime_s": runtime_s,
        "peak_gpu_memory_mb": peak_mb,
        "device": str(device),
    }

print("All GPU optimizers loaded.")

# ============================================================
# FINAL RECOVERY PATCH — validated K=15 engine interface
# Run after the original notebook cells above.
# ============================================================
from dataclasses import replace

# Frozen final constants used in the major-revision experiments
NUM_NODES = 200
NUM_CH = 15
POPULATION_SIZE = 30
CH_RESELECTION_INTERVAL = 10
LIFETIME_OPTIMIZER_BUDGET = 300
INITIAL_ENERGY = 1.0
DEVICES = list(WORK_DEVICES)


def set_fitness_weights(weights):
    global CFG
    wE, wI, wS, wL = [float(x) for x in weights]
    if abs((wE+wI+wS+wL)-1.0) > 1e-12:
        raise ValueError('Fitness weights must sum to 1.')
    CFG = replace(CFG, w_energy=wE, w_intra=wI, w_sink=wS, w_load=wL)
    print(f'Fitness weights -> Energy={wE:.2f}, Intra={wI:.2f}, Sink={wS:.2f}, Load={wL:.2f}')


def network_to_gpu(nodes_df, device):
    # Reuse the numerically validated GPU evaluator from the base notebook.
    return GPUFitnessEvaluator(nodes_df, device)


def evaluate_population_gpu(random_keys, state, num_ch):
    # Decode random-key vectors to unique alive CHs and evaluate the same
    # dimensionless four-component objective used by the base notebook.
    ch_indices = decode_whales_gpu(random_keys, state, int(num_ch))
    fitness, details = state.evaluate_batch(ch_indices, return_details=True)
    return fitness, ch_indices, details


def encode_ch_solution_gpu(ch_indices, dimension, generator, device):
    return encode_ch_to_random_keys_gpu(ch_indices, dimension, generator, device)


def simulate_round(nodes_df, ch_indices):
    network, s = simulate_communication_round(nodes_df, ch_indices)
    # Compatibility keys used by the final lifetime engine.
    return network, {
        'source_packets': int(s['source_packets_delivered_to_bs']),
        'ch_transmissions': int(s['ch_transmissions_to_bs']),
        'residual_energy': float(s['residual_energy']),
        'alive_nodes': int(s['alive_nodes']),
        'dead_nodes': int(s['dead_nodes']),
        'energy_consumed': float(s['energy_consumed']),
    }


@torch.no_grad()
def run_original_cwoa_gpu(nodes_df, num_ch=NUM_CH, population_size=30,
                          evaluation_budget=1500, seed=2026, device=None):
    if device is None:
        device = DEVICES[0]
    state = network_to_gpu(nodes_df, device)
    n = len(nodes_df)
    iterations = max(1, int(evaluation_budget) // int(population_size))
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    whales = torch.rand((population_size, n), generator=g, device=device)
    best_fitness = float('inf')
    best_whale = None
    best_ch = None
    best_details = None
    convergence = []

    for iteration in range(iterations):
        fitness, ch_indices, details = evaluate_population_gpu(whales, state, num_ch)
        current_best_idx = int(torch.argmin(fitness).item())
        current_best = float(fitness[current_best_idx].item())
        if current_best < best_fitness:
            best_fitness = current_best
            best_whale = whales[current_best_idx].clone()
            best_ch = ch_indices[current_best_idx].clone()
            best_details = {k: v[current_best_idx].detach().cpu() for k,v in details.items()}
        convergence.append(best_fitness)

        if iterations > 1:
            a = 2.0 - 2.0 * iteration / (iterations - 1)
        else:
            a = 0.0
        r1 = torch.rand((population_size,1), generator=g, device=device)
        r2 = torch.rand((population_size,1), generator=g, device=device)
        A = 2.0*a*r1 - a
        C = 2.0*r2
        p = torch.rand((population_size,1), generator=g, device=device)

        D_best = torch.abs(C*best_whale[None,:] - whales)
        exploitation = best_whale[None,:] - A*D_best
        random_indices = torch.randint(0,population_size,(population_size,),generator=g,device=device)
        random_whales = whales[random_indices]
        D_random = torch.abs(C*random_whales - whales)
        exploration = random_whales - A*D_random
        encircling = torch.where((torch.abs(A)<1).expand_as(whales), exploitation, exploration)
        l = torch.rand((population_size,1),generator=g,device=device)*2 - 1
        distance_best = torch.abs(best_whale[None,:] - whales)
        spiral = distance_best*torch.exp(l)*torch.cos(2*math.pi*l) + best_whale[None,:]
        whales = torch.where((p<0.5).expand_as(whales), encircling, spiral)
        whales.clamp_(0.0,1.0)

    return {
        'ch_indices': best_ch.detach().cpu().numpy(),
        'fitness': best_fitness,
        'convergence': np.asarray(convergence),
        'details': best_details,
        'evaluations': iterations*population_size,
    }


@torch.no_grad()
def run_improved_cwoa_gpu(nodes_df, num_ch=NUM_CH, population_size=30,
                          evaluation_budget=1500, local_trials=5,
                          stagnation_limit=4, restart_fraction=0.30,
                          seed=2026, device=None):
    if device is None:
        device = DEVICES[0]
    state = network_to_gpu(nodes_df, device)
    n = len(nodes_df)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    whales = torch.rand((population_size,n), generator=g, device=device)
    best_fitness = float('inf')
    best_whale = None
    best_ch = None
    best_details = None
    evaluations = 0
    stagnation_counter = 0
    evaluation_history = []
    fitness_history = []

    while evaluations < evaluation_budget:
        remaining = evaluation_budget - evaluations
        current_pop_size = min(population_size, remaining)
        population_slice = whales[:current_pop_size]
        fitness, ch_indices, details = evaluate_population_gpu(population_slice, state, num_ch)
        evaluations += current_pop_size
        current_best_idx = int(torch.argmin(fitness).item())
        current_best = float(fitness[current_best_idx].item())
        previous_best = best_fitness

        if current_best < best_fitness:
            best_fitness = current_best
            best_whale = population_slice[current_best_idx].clone()
            best_ch = ch_indices[current_best_idx].clone()
            best_details = {k:v[current_best_idx].detach().cpu() for k,v in details.items()}

        available_local = min(local_trials, evaluation_budget-evaluations)
        if available_local > 0 and best_ch is not None:
            alive_idx = torch.where(state.alive_mask)[0]
            local_keys = []
            for _ in range(available_local):
                candidate_ch = best_ch.clone()
                remove_position = int(torch.randint(0,num_ch,(1,),generator=g,device=device).item())
                ch_mask = torch.zeros(n,dtype=torch.bool,device=device)
                ch_mask[candidate_ch] = True
                possible_nodes = alive_idx[~ch_mask[alive_idx]]
                if len(possible_nodes) > 0:
                    selected = int(torch.randint(0,len(possible_nodes),(1,),generator=g,device=device).item())
                    candidate_ch[remove_position] = possible_nodes[selected]
                local_keys.append(encode_ch_solution_gpu(candidate_ch,n,g,device))
            local_population = torch.stack(local_keys)
            local_fitness, local_ch, local_details = evaluate_population_gpu(local_population,state,num_ch)
            evaluations += available_local
            local_best_idx = int(torch.argmin(local_fitness).item())
            local_best = float(local_fitness[local_best_idx].item())
            if local_best < best_fitness:
                best_fitness = local_best
                best_whale = local_population[local_best_idx].clone()
                best_ch = local_ch[local_best_idx].clone()
                best_details = {k:v[local_best_idx].detach().cpu() for k,v in local_details.items()}

        evaluation_history.append(evaluations)
        fitness_history.append(best_fitness)
        if best_fitness < previous_best - 1e-12:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        if evaluations >= evaluation_budget:
            break

        progress = evaluations/evaluation_budget
        a = 2.0*(1.0-progress)
        r1 = torch.rand((population_size,1),generator=g,device=device)
        r2 = torch.rand((population_size,1),generator=g,device=device)
        A = 2*a*r1-a
        C = 2*r2
        p = torch.rand((population_size,1),generator=g,device=device)
        D_best = torch.abs(C*best_whale[None,:]-whales)
        exploitation = best_whale[None,:]-A*D_best
        random_indices = torch.randint(0,population_size,(population_size,),generator=g,device=device)
        random_whales = whales[random_indices]
        D_random = torch.abs(C*random_whales-whales)
        exploration = random_whales-A*D_random
        encircling = torch.where((torch.abs(A)<1).expand_as(whales), exploitation, exploration)
        l = torch.rand((population_size,1),generator=g,device=device)*2-1
        distance_best = torch.abs(best_whale[None,:]-whales)
        spiral = distance_best*torch.exp(l)*torch.cos(2*math.pi*l)+best_whale[None,:]
        whales = torch.where((p<0.5).expand_as(whales),encircling,spiral)

        mutation_probability = 0.15 + 0.20*progress
        mutation_scale = 0.15*(1-progress)+0.03
        mutation_count = max(1,int(0.05*n))
        mutation_whales = torch.where(torch.rand(population_size,generator=g,device=device) < mutation_probability)[0]
        for whale_idx in mutation_whales.tolist():
            mutation_indices = torch.randperm(n,generator=g,device=device)[:mutation_count]
            whales[whale_idx,mutation_indices] += torch.randn(mutation_count,generator=g,device=device)*mutation_scale
        whales.clamp_(0.0,1.0)

        if stagnation_counter >= stagnation_limit:
            restart_count = max(1,int(population_size*restart_fraction))
            restart_indices = torch.randperm(population_size,generator=g,device=device)[:restart_count]
            whales[restart_indices] = torch.rand((restart_count,n),generator=g,device=device)
            stagnation_counter = 0

    return {
        'ch_indices': best_ch.detach().cpu().numpy(),
        'fitness': best_fitness,
        'evaluation_history': np.asarray(evaluation_history),
        'convergence': np.asarray(fitness_history),
        'details': best_details,
        'evaluations': evaluations,
    }


@torch.no_grad()
def run_random_search_gpu(nodes_df, num_ch=NUM_CH, evaluation_budget=1500,
                          batch_size=256, seed=2026, device=None):
    if device is None:
        device = DEVICES[0]
    state = network_to_gpu(nodes_df, device)
    n = len(nodes_df)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    best_fitness = float('inf')
    best_ch = None
    evaluations = 0
    while evaluations < evaluation_budget:
        current_batch = min(batch_size, evaluation_budget-evaluations)
        keys = torch.rand((current_batch,n),generator=g,device=device)
        fitness,ch_indices,_ = evaluate_population_gpu(keys,state,num_ch)
        best_idx = int(torch.argmin(fitness).item())
        current_best = float(fitness[best_idx].item())
        if current_best < best_fitness:
            best_fitness = current_best
            best_ch = ch_indices[best_idx].clone()
        evaluations += current_batch
    return {'ch_indices':best_ch.detach().cpu().numpy(),'fitness':best_fitness,'evaluations':evaluations}


def select_cluster_heads_gpu(nodes_df, method, num_ch, evaluation_budget, seed, device):
    if method == 'original_cwoa':
        return run_original_cwoa_gpu(nodes_df,num_ch=num_ch,population_size=POPULATION_SIZE,
                                     evaluation_budget=evaluation_budget,seed=seed,device=device)
    if method == 'improved_cwoa':
        return run_improved_cwoa_gpu(nodes_df,num_ch=num_ch,population_size=POPULATION_SIZE,
                                     evaluation_budget=evaluation_budget,local_trials=3,
                                     stagnation_limit=3,restart_fraction=0.30,seed=seed,device=device)
    if method == 'random_search':
        return run_random_search_gpu(nodes_df,num_ch=num_ch,evaluation_budget=evaluation_budget,
                                     seed=seed,device=device)
    if method == 'pso':
        return run_pso_gpu(nodes_df,num_ch=num_ch,population_size=POPULATION_SIZE,
                           evaluation_budget=evaluation_budget,seed=seed,device=device)
    if method == 'leach':
        raise ValueError("LEACH uses round-wise CH rotation; call simulate_lifetime_method_gpu.")
    raise ValueError(f'Unknown method: {method}')


def simulate_lifetime_gpu(topology_seed, method, optimizer_seed, device,
                          max_rounds=5000, num_ch=NUM_CH,
                          reselection_interval=CH_RESELECTION_INTERVAL,
                          evaluation_budget=LIFETIME_OPTIMIZER_BUDGET, verbose=False):
    network = create_sensor_network(topology_seed)
    rng = np.random.default_rng(int(optimizer_seed))
    current_ch = None
    history = []
    cumulative_source_packets = 0
    cumulative_ch_transmissions = 0
    total_optimizer_evaluations = 0
    FND = HND = LND = None
    half_nodes = int(np.ceil(len(network)/2))

    for round_number in range(1,max_rounds+1):
        alive_count = int(network['alive'].sum())
        if alive_count == 0:
            LND = round_number-1
            break
        active_num_ch = min(num_ch,alive_count)
        need_reselection = (current_ch is None or (round_number-1)%reselection_interval==0 or
                            any(not bool(network.loc[idx,'alive']) for idx in current_ch))
        if need_reselection:
            seed_event = int(rng.integers(0,2**31-1))
            result = select_cluster_heads_gpu(network,method,active_num_ch,evaluation_budget,seed_event,device)
            current_ch = result['ch_indices']
            total_optimizer_evaluations += result['evaluations']
        network,stats_round = simulate_round(network,current_ch)
        cumulative_source_packets += stats_round['source_packets']
        cumulative_ch_transmissions += stats_round['ch_transmissions']
        alive_now = stats_round['alive_nodes']
        dead_now = stats_round['dead_nodes']
        if FND is None and dead_now >= 1: FND = round_number
        if HND is None and dead_now >= half_nodes: HND = round_number
        if alive_now == 0: LND = round_number
        history.append({
            'round':round_number,'alive_nodes':alive_now,'dead_nodes':dead_now,
            'residual_energy':stats_round['residual_energy'],'source_packets':stats_round['source_packets'],
            'cumulative_source_packets':cumulative_source_packets,
            'cumulative_ch_transmissions':cumulative_ch_transmissions,
        })
        if verbose and round_number%500==0:
            print(method,'|',round_number,'| Alive:',alive_now,'| Energy:',round(stats_round['residual_energy'],3))
        if alive_now == 0: break

    history_df = pd.DataFrame(history)
    summary = {
        'method':method,'topology_seed':topology_seed,'optimizer_seed':optimizer_seed,
        'device':str(device),'FND':FND,'HND':HND,'LND':LND,
        'throughput':cumulative_source_packets,'ch_transmissions':cumulative_ch_transmissions,
        'optimizer_evaluations':total_optimizer_evaluations,
        'final_energy':float(network['energy'].sum()),
    }
    return summary,history_df


@torch.no_grad()
def run_pso_gpu(nodes_df, num_ch=15, population_size=30, evaluation_budget=300,
                seed=2026, device=None, inertia=0.70, c1=1.50, c2=1.50,
                velocity_limit=0.20):
    if device is None: device = DEVICES[0]
    state = network_to_gpu(nodes_df,device)
    n = len(nodes_df)
    g = torch.Generator(device=device); g.manual_seed(int(seed))
    particles = torch.rand((population_size,n),generator=g,device=device)
    velocity = torch.rand((population_size,n),generator=g,device=device)*0.10-0.05
    pbest_position = particles.clone()
    pbest_fitness = torch.full((population_size,),float('inf'),device=device)
    gbest_position=None; gbest_fitness=float('inf'); gbest_ch=None; gbest_details=None
    evaluations=0; evaluation_history=[]; convergence=[]
    while evaluations < evaluation_budget:
        remaining=evaluation_budget-evaluations
        current_population=min(population_size,remaining)
        current_particles=particles[:current_population]
        fitness,ch_indices,details=evaluate_population_gpu(current_particles,state,num_ch)
        evaluations += current_population
        improved_mask=fitness < pbest_fitness[:current_population]
        improved_indices=torch.where(improved_mask)[0]
        if improved_indices.numel()>0:
            pbest_fitness[improved_indices]=fitness[improved_indices]
            pbest_position[improved_indices]=current_particles[improved_indices]
        current_best_idx=int(torch.argmin(fitness).item())
        current_best_fitness=float(fitness[current_best_idx].item())
        if current_best_fitness < gbest_fitness:
            gbest_fitness=current_best_fitness
            gbest_position=current_particles[current_best_idx].clone()
            gbest_ch=ch_indices[current_best_idx].clone()
            gbest_details={k:v[current_best_idx].detach().cpu() for k,v in details.items()}
        evaluation_history.append(evaluations); convergence.append(gbest_fitness)
        if evaluations >= evaluation_budget: break
        r1=torch.rand((population_size,n),generator=g,device=device)
        r2=torch.rand((population_size,n),generator=g,device=device)
        velocity = inertia*velocity + c1*r1*(pbest_position-particles) + c2*r2*(gbest_position[None,:]-particles)
        velocity.clamp_(-velocity_limit,velocity_limit)
        particles=(particles+velocity).clamp_(0.0,1.0)
    return {
        'ch_indices':gbest_ch.detach().cpu().numpy(),'fitness':gbest_fitness,
        'evaluation_history':np.asarray(evaluation_history),'convergence':np.asarray(convergence),
        'details':gbest_details,'evaluations':evaluations,
    }


def simulate_lifetime_leach_gpu(topology_seed, optimizer_seed, device,
                                max_rounds=5000, num_ch=15, verbose=False):
    network=create_sensor_network(topology_seed)
    rng=np.random.default_rng(int(optimizer_seed))
    initial_nodes=len(network)
    p=float(num_ch)/float(initial_nodes)
    epoch_length=max(1,int(round(1.0/p)))
    history=[]; cumulative_source_packets=0; cumulative_ch_transmissions=0
    FND=HND=LND=None; half_nodes=int(np.ceil(initial_nodes/2))
    selected_in_epoch=set(); previous_epoch=-1
    for round_number in range(1,max_rounds+1):
        alive_indices=np.asarray(network.index[network['alive']],dtype=int)
        alive_count=len(alive_indices)
        if alive_count==0:
            LND=round_number-1; break
        zero_based_round=round_number-1
        current_epoch=zero_based_round//epoch_length
        epoch_phase=zero_based_round%epoch_length
        if current_epoch != previous_epoch:
            selected_in_epoch=set(); previous_epoch=current_epoch
        eligible_nodes=np.asarray([int(i) for i in alive_indices if int(i) not in selected_in_epoch],dtype=int)
        if len(eligible_nodes)==0:
            selected_in_epoch=set(); eligible_nodes=alive_indices.copy()
        denominator=1.0-p*epoch_phase
        threshold=1.0 if denominator<=0 else min(1.0,p/denominator)
        random_values=rng.random(len(eligible_nodes))
        current_ch=eligible_nodes[random_values < threshold]
        if len(current_ch)==0:
            current_ch=np.asarray([int(rng.choice(eligible_nodes))],dtype=int)
        reasonable_upper_bound=min(alive_count,max(1,int(np.ceil(2.0*p*alive_count))))
        if len(current_ch)>reasonable_upper_bound:
            current_ch=np.asarray(rng.choice(current_ch,size=reasonable_upper_bound,replace=False),dtype=int)
        selected_in_epoch.update(int(i) for i in current_ch)
        network,stats_round=simulate_round(network,current_ch)
        cumulative_source_packets += int(stats_round['source_packets'])
        cumulative_ch_transmissions += int(stats_round['ch_transmissions'])
        alive_now=int(stats_round['alive_nodes']); dead_now=int(stats_round['dead_nodes'])
        if FND is None and dead_now>=1: FND=round_number
        if HND is None and dead_now>=half_nodes: HND=round_number
        if alive_now==0: LND=round_number
        history.append({
            'round':round_number,'alive_nodes':alive_now,'dead_nodes':dead_now,
            'residual_energy':stats_round['residual_energy'],'source_packets':stats_round['source_packets'],
            'cumulative_source_packets':cumulative_source_packets,
            'cumulative_ch_transmissions':cumulative_ch_transmissions,
            'active_ch_count':len(current_ch),'leach_threshold':threshold,
            'epoch':current_epoch,'epoch_phase':epoch_phase,
        })
        if verbose and round_number%500==0:
            print('LEACH |',round_number,'| Alive:',alive_now,'| CH:',len(current_ch))
        if alive_now==0: break
    history_df=pd.DataFrame(history)
    summary={
        'method':'leach','topology_seed':topology_seed,'optimizer_seed':optimizer_seed,
        'device':str(device),'FND':FND,'HND':HND,'LND':LND,
        'throughput':cumulative_source_packets,'ch_transmissions':cumulative_ch_transmissions,
        'optimizer_evaluations':0,'final_energy':float(network['energy'].sum()),
    }
    return summary,history_df


def simulate_lifetime_method_gpu(topology_seed,method,optimizer_seed,device,
                                 max_rounds=5000,num_ch=15,
                                 reselection_interval=CH_RESELECTION_INTERVAL,
                                 evaluation_budget=300,verbose=False):
    if method=='leach':
        return simulate_lifetime_leach_gpu(topology_seed,optimizer_seed,device,max_rounds,num_ch,verbose)
    return simulate_lifetime_gpu(topology_seed,method,optimizer_seed,device,max_rounds,num_ch,
                                 reselection_interval,evaluation_budget,verbose)

# Freeze the final selected configuration.
set_fitness_weights((0.35,0.30,0.20,0.15))
print('='*75)
print('FINAL RECOVERY ENGINE INSTALLED')
print('Devices:', DEVICES)
print('K =', NUM_CH, '| population =', POPULATION_SIZE, '| lifetime budget =', LIFETIME_OPTIMIZER_BUDGET)
print('='*75)

# ============================================================
# RECOVERY VALIDATION — no full experiment yet
# ============================================================
required = [
    'create_sensor_network','simulate_round','network_to_gpu','evaluate_population_gpu',
    'run_original_cwoa_gpu','run_improved_cwoa_gpu','run_random_search_gpu',
    'run_pso_gpu','select_cluster_heads_gpu','simulate_lifetime_gpu',
    'simulate_lifetime_leach_gpu','simulate_lifetime_method_gpu','set_fitness_weights'
]
missing=[x for x in required if x not in globals()]
print('Missing functions:',missing)
if missing:
    raise RuntimeError('Recovery incomplete.')
print('✅ CORE + FINAL BASELINE ENGINE RESTORED')

# One paired topology sanity check for PSO and corrected LEACH.
for method in ['pso','leach']:
    summary,hist=simulate_lifetime_method_gpu(
        topology_seed=5000,method=method,optimizer_seed=9000,
        device=DEVICES[0],max_rounds=5000,num_ch=15,evaluation_budget=300,verbose=False)
    print('\n',method.upper(), summary)
    assert summary['FND'] <= summary['HND'] <= summary['LND']
    assert not (hist['alive_nodes'].diff().dropna()>0).any()
    assert not (hist['residual_energy'].diff().dropna()>1e-9).any()
    if method=='leach':
        print(hist['active_ch_count'].describe())
        assert hist['active_ch_count'].max() <= 30
print('\n✅ RECOVERY VALIDATION PASSED')

print("\n" + "="*78)
print("STANDALONE RECOVERY SCRIPT FINISHED")
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("DEVICES:", DEVICES)
print("FINAL K:", NUM_CH)
print("FINAL WEIGHTS:", (CFG.w_energy, CFG.w_intra, CFG.w_sink, CFG.w_load))
print("FINAL RESELECTION INTERVAL:", CH_RESELECTION_INTERVAL)
print("FINAL EVAL BUDGET:", LIFETIME_OPTIMIZER_BUDGET)
print("="*78)

# ===== Notebook code cell 9 =====
# FIXED VERSION
# Backup ZIPs are stored OUTSIDE /kaggle/working/AD_CWOA_SAFE_RERUN.
# This prevents the ZIP from recursively archiving itself and filling the disk.

# ======================================================================
# AD-CWOA SAFE STAGED RERUN
# One 20-run batch per execution | per-run checkpoints | ZIP every 5 runs
# Run AFTER the standalone recovery/validation cell has passed.
# ======================================================================

import os, json, time, shutil, traceback, gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

# ---------- verify engine ----------
REQUIRED = [
    "DEVICES", "NUM_NODES", "POPULATION_SIZE",
    "CH_RESELECTION_INTERVAL", "LIFETIME_OPTIMIZER_BUDGET",
    "simulate_lifetime_method_gpu", "set_fitness_weights",
]
missing = [x for x in REQUIRED if x not in globals()]
if missing:
    raise RuntimeError(
        "Recovered engine is not loaded. Run the standalone recovery/validation "
        "cell first. Missing: " + ", ".join(missing)
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

WORK_DEVICES = list(DEVICES[:2])
if not WORK_DEVICES:
    raise RuntimeError("No working CUDA devices found.")

# ---------- frozen settings ----------
FINAL_WEIGHTS = (0.35, 0.30, 0.20, 0.15)
FINAL_K = 15
EVAL_BUDGET = 300
MAX_ROUNDS = 5000
TOPOLOGY_SEEDS = list(range(5000, 5020))
OPTIMIZER_SEEDS = list(range(9000, 9020))
CHECKPOINT_ROUNDS = [500, 1000, 1500, 1800, 2000]

WEIGHT_CONFIGS = [
    ("Baseline_35_30_20_15", (0.35, 0.30, 0.20, 0.15)),
    ("EnergyHeavy_50_20_15_15", (0.50, 0.20, 0.15, 0.15)),
    ("Equal_25_25_25_25", (0.25, 0.25, 0.25, 0.25)),
    ("IntraHeavy_25_45_15_15", (0.25, 0.45, 0.15, 0.15)),
    ("LegacyMapped_50_25_25_00", (0.50, 0.25, 0.25, 0.00)),
    ("LoadHeavy_25_20_15_40", (0.25, 0.20, 0.15, 0.40)),
    ("SinkHeavy_25_20_40_15", (0.25, 0.20, 0.40, 0.15)),
]
K_CONFIRM = [12, 15, 18]
FINAL_METHODS = [
    ("leach", "LEACH-style"),
    ("random_search", "Random Search"),
    ("original_cwoa", "Original CWOA"),
    ("pso", "PSO"),
    ("improved_cwoa", "AD-CWOA"),
]

ROOT = Path("/kaggle/working/AD_CWOA_SAFE_RERUN")
CHECKPOINT_DIR = ROOT / "checkpoints"
HISTORY_DIR = ROOT / "histories"
BACKUP_DIR = Path("/kaggle/working/AD_CWOA_SAFE_BACKUPS")  # OUTSIDE ROOT to prevent recursive ZIP growth
TABLE_DIR = ROOT / "tables"
for d in [ROOT, CHECKPOINT_DIR, HISTORY_DIR, BACKUP_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MASTER_CSV = ROOT / "MASTER_ALL_RUNS.csv"
MANIFEST_JSON = ROOT / "RUN_MANIFEST.json"

# ---------- batch plan ----------
BATCH_PLAN = []
for name, weights in WEIGHT_CONFIGS:
    BATCH_PLAN.append({
        "batch_id": f"A_{name}",
        "stage": "A_weight_sensitivity",
        "label": name,
        "weights": weights,
        "k": FINAL_K,
        "method": "improved_cwoa",
    })
for k in K_CONFIRM:
    BATCH_PLAN.append({
        "batch_id": f"B_K{k}",
        "stage": "B_K_confirmatory",
        "label": f"K{k}",
        "weights": FINAL_WEIGHTS,
        "k": k,
        "method": "improved_cwoa",
    })
for method, label in FINAL_METHODS:
    BATCH_PLAN.append({
        "batch_id": f"C_{method}",
        "stage": "C_final_5_method",
        "label": label,
        "weights": FINAL_WEIGHTS,
        "k": FINAL_K,
        "method": method,
    })
assert len(BATCH_PLAN) == 15

# ---------- IO helpers ----------
def atomic_write_text(text, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def atomic_csv(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def atomic_json(obj, path):
    atomic_write_text(json.dumps(obj, indent=2, default=str), path)

def load_master():
    return pd.read_csv(MASTER_CSV) if MASTER_CSV.exists() else pd.DataFrame()

def completed_keys(df):
    if df.empty:
        return set()
    return set(
        (str(r.batch_id), int(r.topology_seed))
        for r in df[["batch_id", "topology_seed"]].itertuples(index=False)
    )

def rebuild_master_from_json():
    rows = []
    for p in sorted(CHECKPOINT_DIR.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["batch_id", "topology_seed"], keep="last"
    ).sort_values(["batch_order", "topology_seed"])
    atomic_csv(df, MASTER_CSV)
    return df

def safe_value_at_round(hist, col, rr):
    hit = hist.loc[hist["round"] == rr, col]
    if len(hit):
        return float(hit.iloc[0])
    if len(hist) and int(hist["round"].max()) < rr:
        if col in ("alive_nodes", "residual_energy"):
            return 0.0
        if col == "cumulative_source_packets":
            return float(hist[col].iloc[-1])
    return np.nan

def compute_metrics(summary, hist):
    row = dict(summary)
    FND, HND, LND = map(float, (row["FND"], row["HND"], row["LND"]))
    throughput = float(row["throughput"])
    ch_tx = float(row["ch_transmissions"])

    row["Stability_Ratio"] = FND / LND
    row["Death_Transition_Period"] = LND - FND
    row["Throughput_per_Round"] = throughput / LND
    row["CH_Transmissions_per_Round"] = ch_tx / LND

    rounds = hist["round"].to_numpy(dtype=float)
    alive = hist["alive_nodes"].to_numpy(dtype=float)
    x = np.insert(rounds, 0, 0.0)
    y = np.insert(alive, 0, float(NUM_NODES))
    auc = float(np.trapezoid(y, x))
    row["Alive_Node_AUC"] = auc
    row["Normalized_AUC"] = auc / (float(NUM_NODES) * LND)

    for rr in CHECKPOINT_ROUNDS:
        row[f"ResidualEnergy_R{rr}"] = safe_value_at_round(hist, "residual_energy", rr)
        row[f"AliveNodes_R{rr}"] = safe_value_at_round(hist, "alive_nodes", rr)
    return row

def zip_root(tag):
    # IMPORTANT: ZIP destination is OUTSIDE ROOT, so the archive never includes itself.
    base_name = BACKUP_DIR / tag
    zip_file = base_name.with_suffix(".zip")
    if zip_file.exists():
        zip_file.unlink()
    zip_path = shutil.make_archive(
        str(base_name),
        "zip",
        root_dir=str(ROOT.parent),
        base_dir=ROOT.name,
    )
    size_mb = Path(zip_path).stat().st_size / 1024**2
    print(f"\nBACKUP ZIP -> {zip_path} ({size_mb:.2f} MB)")
    return zip_path

def write_manifest(master_df, current_batch=None):
    counts = {}
    if not master_df.empty:
        counts = master_df.groupby("batch_id").size().to_dict()
    complete_batches = [
        b["batch_id"] for b in BATCH_PLAN
        if int(counts.get(b["batch_id"], 0)) >= 20
    ]
    manifest = {
        "torch_version": torch.__version__,
        "cuda_devices": [str(x) for x in WORK_DEVICES],
        "final_weights": FINAL_WEIGHTS,
        "final_k": FINAL_K,
        "population": int(POPULATION_SIZE),
        "evaluation_budget_per_reselection": EVAL_BUDGET,
        "reselection_interval": int(CH_RESELECTION_INTERVAL),
        "max_rounds": MAX_ROUNDS,
        "topology_seeds": TOPOLOGY_SEEDS,
        "optimizer_seeds": OPTIMIZER_SEEDS,
        "completed_runs": int(len(master_df)),
        "expected_runs": 300,
        "completed_batches": complete_batches,
        "current_batch": current_batch,
    }
    atomic_json(manifest, MANIFEST_JSON)

# ---------- recover checkpoints ----------
master = load_master()
json_count = len(list(CHECKPOINT_DIR.glob("*.json")))
if json_count > len(master):
    print(f"Recovering master from {json_count} per-run JSON checkpoints...")
    master = rebuild_master_from_json()

done = completed_keys(master)

# ---------- choose next incomplete batch ----------
next_batch = None
next_order = None
for order, batch in enumerate(BATCH_PLAN, start=1):
    n_done = sum((batch["batch_id"], topo) in done for topo in TOPOLOGY_SEEDS)
    if n_done < 20:
        next_batch = batch
        next_order = order
        break

if next_batch is None:
    print("=" * 90)
    print("ALL 300 RUNS ARE ALREADY COMPLETE.")
    print("=" * 90)
    print("Master rows:", len(master))
    print("Final ZIP:", zip_root("FINAL_AD_CWOA_SAFE_RERUN_ALL_300"))
    raise SystemExit

batch = next_batch
batch_id = batch["batch_id"]
completed_in_batch = sum((batch_id, topo) in done for topo in TOPOLOGY_SEEDS)
jobs = [
    (topo, opt_seed)
    for topo, opt_seed in zip(TOPOLOGY_SEEDS, OPTIMIZER_SEEDS)
    if (batch_id, topo) not in done
]

print("\n" + "=" * 94)
print(f"SAFE RERUN — BATCH {next_order}/15")
print("=" * 94)
print("Batch ID       :", batch_id)
print("Stage          :", batch["stage"])
print("Label          :", batch["label"])
print("Method         :", batch["method"])
print("K              :", batch["k"])
print("Weights        :", batch["weights"])
print("Already saved  :", completed_in_batch, "/20")
print("Remaining      :", len(jobs), "/20")
print("Total recovered:", len(master), "/300")
print("Devices        :", WORK_DEVICES)
print("=" * 94)

set_fitness_weights(tuple(batch["weights"]))

# ---------- one lifetime run ----------
def run_one(job, device):
    topo, opt_seed = job
    t0 = time.perf_counter()

    summary, hist = simulate_lifetime_method_gpu(
        topology_seed=int(topo),
        method=batch["method"],
        optimizer_seed=int(opt_seed),
        device=device,
        max_rounds=MAX_ROUNDS,
        num_ch=int(batch["k"]),
        reselection_interval=CH_RESELECTION_INTERVAL,
        evaluation_budget=EVAL_BUDGET,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    if not (summary["FND"] <= summary["HND"] <= summary["LND"]):
        raise RuntimeError(f"FND/HND/LND ordering failed: {summary}")
    if (hist["alive_nodes"].diff().dropna() > 0).any():
        raise RuntimeError("Alive-node count increased.")
    if (hist["residual_energy"].diff().dropna() > 1e-9).any():
        raise RuntimeError("Residual energy increased.")
    if (hist["residual_energy"] < -1e-9).any():
        raise RuntimeError("Negative residual energy detected.")
    if "cumulative_source_packets" in hist.columns:
        if (hist["cumulative_source_packets"].diff().dropna() < 0).any():
            raise RuntimeError("Cumulative throughput decreased.")

    row = compute_metrics(summary, hist)
    row.update({
        "batch_order": next_order,
        "batch_id": batch_id,
        "stage": batch["stage"],
        "label": batch["label"],
        "method": batch["method"],
        "k": int(batch["k"]),
        "w_energy": float(batch["weights"][0]),
        "w_intra": float(batch["weights"][1]),
        "w_sink": float(batch["weights"][2]),
        "w_load": float(batch["weights"][3]),
        "topology_seed": int(topo),
        "optimizer_seed": int(opt_seed),
        "runtime_s": float(elapsed),
        "device": str(device),
    })
    return row, hist

# ---------- run with max 2 concurrent jobs, save each result immediately ----------
def device_for_index(i):
    return WORK_DEVICES[i % len(WORK_DEVICES)]

submitted = {}
with ThreadPoolExecutor(max_workers=len(WORK_DEVICES)) as executor:
    for i, job in enumerate(jobs):
        dev = device_for_index(i)
        submitted[executor.submit(run_one, job, dev)] = (job, dev)

    for fut in as_completed(submitted):
        (topo, opt_seed), dev = submitted[fut]
        try:
            row, hist = fut.result()
        except Exception:
            print(f"\nFAILED: {batch_id} topo={topo} seed={opt_seed} on {dev}")
            traceback.print_exc()
            raise

        history_name = f"{next_order:02d}_{batch_id}_topo{topo}_seed{opt_seed}.csv.gz"
        history_path = HISTORY_DIR / history_name
        hist.to_csv(history_path, index=False, compression="gzip")
        row["history_file"] = str(history_path)

        checkpoint_name = f"{next_order:02d}_{batch_id}_topo{topo}_seed{opt_seed}.json"
        atomic_json(row, CHECKPOINT_DIR / checkpoint_name)

        master = load_master()
        row_df = pd.DataFrame([row])
        master = row_df if master.empty else pd.concat([master, row_df], ignore_index=True)
        master = master.drop_duplicates(
            subset=["batch_id", "topology_seed"], keep="last"
        ).sort_values(["batch_order", "topology_seed"])
        atomic_csv(master, MASTER_CSV)
        write_manifest(master, current_batch=batch_id)

        now_done = int((master["batch_id"].astype(str) == batch_id).sum())
        print(
            f"[SAVED {now_done:02d}/20] {batch_id} | topo={topo} | "
            f"FND={int(row['FND'])} HND={int(row['HND'])} "
            f"LND={int(row['LND'])} | {dev}"
        )

        if now_done in (5, 10, 15, 20):
            zip_root(f"BATCH_{next_order:02d}_{batch_id}_{now_done:02d}of20")

        del hist
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------- verify batch + export ----------
master = load_master()
batch_df = master.loc[
    master["batch_id"].astype(str) == batch_id
].copy().sort_values("topology_seed")

if len(batch_df) != 20:
    print("\nWARNING: batch contains", len(batch_df), "/20 saved runs.")
    print("Re-run THIS SAME CELL. It will execute only the missing topologies.")
else:
    batch_csv = TABLE_DIR / f"{next_order:02d}_{batch_id}_20_RUNS.csv"
    atomic_csv(batch_df, batch_csv)

    numeric_cols = [
        "FND", "HND", "LND", "throughput",
        "Stability_Ratio", "Death_Transition_Period",
        "Throughput_per_Round", "Alive_Node_AUC",
        "Normalized_AUC", "CH_Transmissions_per_Round",
        "ResidualEnergy_R2000", "AliveNodes_R2000",
        "optimizer_evaluations", "runtime_s",
    ]
    summary_rows = []
    for c in numeric_cols:
        if c in batch_df.columns:
            summary_rows.append({
                "Metric": c,
                "Mean": float(batch_df[c].mean()),
                "Std": float(batch_df[c].std(ddof=1)),
            })
    summary_df = pd.DataFrame(summary_rows)
    atomic_csv(summary_df, TABLE_DIR / f"{next_order:02d}_{batch_id}_MEAN_SD.csv")

    write_manifest(master, current_batch=None)
    final_batch_zip = zip_root(f"BATCH_{next_order:02d}_{batch_id}_COMPLETE")

    print("\n" + "=" * 94)
    print("BATCH COMPLETE — STOP HERE AND SAVE YOUR KAGGLE VERSION")
    print("=" * 94)
    print("Batch:", next_order, "/15")
    print("Batch ID:", batch_id)
    print("Saved runs in batch: 20/20")
    print("Total saved runs:", len(master), "/300")
    print("Batch CSV:", batch_csv)
    print("Backup ZIP:", final_batch_zip)
    print()
    print("NEXT ACTION:")
    print("1. Kaggle -> Save Version.")
    print("2. Download the COMPLETE batch ZIP shown above.")
    print("3. Then run this SAME master cell again for the next 20-run batch.")
    print("=" * 94)

set_fitness_weights(FINAL_WEIGHTS)

