# ===== Notebook code cell 0 =====
# ============================================================
# CELL 1: KAGGLE T4x2 ENVIRONMENT + RESEARCH PARAMETERS
# ============================================================

import os
import gc
import time
import math
import random
import warnings

import numpy as np
import pandas as pd

import torch

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# GPU DETECTION
# ============================================================

print("=" * 70)
print("KAGGLE GPU ENVIRONMENT")
print("=" * 70)

print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
print("CUDA version   :", torch.version.cuda)

NUM_GPUS = torch.cuda.device_count()

print("GPU count      :", NUM_GPUS)

for i in range(NUM_GPUS):

    properties = torch.cuda.get_device_properties(i)

    print(
        f"\nGPU {i}: {properties.name}"
    )

    print(
        f"Memory: "
        f"{properties.total_memory / 1024**3:.2f} GB"
    )


if NUM_GPUS == 0:

    DEVICES = [
        torch.device("cpu")
    ]

    print(
        "\nWARNING: No CUDA GPU found."
    )

else:

    DEVICES = [
        torch.device(
            f"cuda:{i}"
        )
        for i in range(NUM_GPUS)
    ]


# ============================================================
# REPRODUCIBILITY
# ============================================================

GLOBAL_SEED = 42

random.seed(
    GLOBAL_SEED
)

np.random.seed(
    GLOBAL_SEED
)

torch.manual_seed(
    GLOBAL_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        GLOBAL_SEED
    )


# ============================================================
# NETWORK PARAMETERS
# ============================================================

AREA_X = 200.0
AREA_Y = 200.0

NUM_NODES = 200

INITIAL_ENERGY = 1.0

PACKET_SIZE = 4000


# ============================================================
# RADIO ENERGY MODEL
# ============================================================

E_ELEC = 50e-9

E_FS = 10e-12

E_MP = 0.0013e-12

E_DA = 5e-9


D0 = np.sqrt(
    E_FS / E_MP
)


# ============================================================
# BASE STATION / SINK
# ============================================================

SINK_X = AREA_X / 2
SINK_Y = AREA_Y / 2


# ============================================================
# FITNESS WEIGHTS
# ============================================================

W_ENERGY = 0.35

W_INTRA = 0.30

W_SINK = 0.20

W_LOAD = 0.15


NETWORK_DIAGONAL = np.sqrt(
    AREA_X**2
    +
    AREA_Y**2
)


MAX_SINK_DISTANCE = np.sqrt(

    max(
        SINK_X,
        AREA_X - SINK_X
    ) ** 2

    +

    max(
        SINK_Y,
        AREA_Y - SINK_Y
    ) ** 2
)


# ============================================================
# OPTIMIZER PARAMETERS
# ============================================================

NUM_CH = 10

POPULATION_SIZE = 30

FULL_OPTIMIZER_BUDGET = 1500

LIFETIME_OPTIMIZER_BUDGET = 300

CH_RESELECTION_INTERVAL = 10


print("\n" + "=" * 70)
print("WSN CONFIGURATION")
print("=" * 70)

print(
    f"Area                : "
    f"{AREA_X:.0f} m × {AREA_Y:.0f} m"
)

print(
    f"Sensor nodes        : {NUM_NODES}"
)

print(
    f"Initial energy      : {INITIAL_ENERGY} J/node"
)

print(
    f"Total energy        : "
    f"{NUM_NODES * INITIAL_ENERGY:.2f} J"
)

print(
    f"Packet size         : {PACKET_SIZE} bits"
)

print(
    f"Threshold distance  : {D0:.2f} m"
)

print(
    f"Cluster Heads       : {NUM_CH}"
)

print(
    f"Population size     : {POPULATION_SIZE}"
)

print(
    f"Available devices   : {DEVICES}"
)

# ===== Notebook code cell 1 =====
# ============================================================
# CELL 2: SENSOR NETWORK GENERATOR
# ============================================================

def create_sensor_network(
    topology_seed,
    num_nodes=NUM_NODES
):

    rng = np.random.default_rng(
        topology_seed
    )

    network = pd.DataFrame({

        "node_id":
            np.arange(
                1,
                num_nodes + 1
            ),

        "x":
            rng.uniform(
                0,
                AREA_X,
                num_nodes
            ),

        "y":
            rng.uniform(
                0,
                AREA_Y,
                num_nodes
            ),

        "energy":
            np.full(
                num_nodes,
                INITIAL_ENERGY,
                dtype=np.float64
            ),

        "alive":
            np.ones(
                num_nodes,
                dtype=bool
            )
    })


    network[
        "distance_to_sink"
    ] = np.sqrt(

        (
            network["x"]
            - SINK_X
        ) ** 2

        +

        (
            network["y"]
            - SINK_Y
        ) ** 2
    )


    return network


# ------------------------------------------------------------
# TEST NETWORK
# ------------------------------------------------------------

nodes = create_sensor_network(
    topology_seed=42
)

print(
    "Network created successfully."
)

print(
    "Nodes:",
    len(nodes)
)

print(
    "Total energy:",
    nodes["energy"].sum()
)

print(
    "Average sink distance:",
    round(
        nodes[
            "distance_to_sink"
        ].mean(),
        2
    ),
    "m"
)

display(
    nodes.head()
)

# ===== Notebook code cell 2 =====
# ============================================================
# CELL 3: GPU NETWORK REPRESENTATION
# ============================================================

def network_to_gpu(
    nodes_df,
    device
):

    position = torch.tensor(

        nodes_df[
            ["x", "y"]
        ].to_numpy(),

        dtype=torch.float32,

        device=device
    )


    energy = torch.tensor(

        nodes_df[
            "energy"
        ].to_numpy(),

        dtype=torch.float32,

        device=device
    )


    alive = torch.tensor(

        nodes_df[
            "alive"
        ].to_numpy(),

        dtype=torch.bool,

        device=device
    )


    sink = torch.tensor(
        [
            SINK_X,
            SINK_Y
        ],
        dtype=torch.float32,
        device=device
    )


    return {

        "position":
            position,

        "energy":
            energy,

        "alive":
            alive,

        "sink":
            sink,

        "num_nodes":
            len(nodes_df)
    }


device0 = DEVICES[0]

gpu_state = network_to_gpu(
    nodes,
    device0
)

print(
    "GPU network loaded on:",
    device0
)

print(
    "Position tensor:",
    gpu_state[
        "position"
    ].shape
)

# ===== Notebook code cell 3 =====
# ============================================================
# CELL 4: GPU-BATCHED FITNESS FUNCTION
# ============================================================

@torch.no_grad()
def evaluate_population_gpu(
    whale_keys,
    state,
    num_ch=NUM_CH
):

    """
    whale_keys shape:
        [population, num_nodes]

    Returns:
        fitness
        selected CH indices
        details
    """

    device = whale_keys.device

    population_size = (
        whale_keys.shape[0]
    )

    n = state[
        "num_nodes"
    ]

    alive = state[
        "alive"
    ]

    alive_count = int(
        alive.sum().item()
    )

    if alive_count < num_ch:

        raise ValueError(
            "Alive node count smaller than requested CH count."
        )


    # ========================================================
    # DECODE RANDOM KEYS
    # ========================================================

    scores = whale_keys.clone()

    scores[
        :,
        ~alive
    ] = -1e9


    ch_indices = torch.topk(

        scores,

        k=num_ch,

        dim=1,

        largest=True

    ).indices


    positions = state[
        "position"
    ]


    # [P, K, 2]
    ch_positions = positions[
        ch_indices
    ]


    # ========================================================
    # NODE -> CH DISTANCES
    # ========================================================

    diff = (

        positions[
            None,
            :,
            None,
            :
        ]

        -

        ch_positions[
            :,
            None,
            :,
            :
        ]
    )


    distances = torch.linalg.vector_norm(
        diff,
        dim=-1
    )


    nearest_distance, cluster_id = (
        distances.min(
            dim=2
        )
    )


    # ========================================================
    # CH MASK
    # ========================================================

    is_ch = torch.zeros(

        (
            population_size,
            n
        ),

        dtype=torch.bool,

        device=device
    )


    is_ch.scatter_(
        1,
        ch_indices,
        True
    )


    alive_matrix = alive[
        None,
        :
    ].expand(
        population_size,
        -1
    )


    member_mask = (

        alive_matrix

        &

        (~is_ch)
    )


    # ========================================================
    # ENERGY PENALTY
    # ========================================================

    ch_energy = state[
        "energy"
    ][
        ch_indices
    ]


    normalized_energy = torch.clamp(

        ch_energy
        /
        INITIAL_ENERGY,

        0.0,
        1.0
    )


    energy_penalty = (

        1.0

        -

        normalized_energy.mean(
            dim=1
        )
    )


    # ========================================================
    # INTRA-CLUSTER DISTANCE
    # ========================================================

    member_count = member_mask.sum(
        dim=1
    ).clamp_min(1)


    intra_distance = (

        (
            nearest_distance
            *
            member_mask.float()
        ).sum(
            dim=1
        )

        /

        member_count
    )


    intra_penalty = (

        intra_distance

        /

        NETWORK_DIAGONAL
    )


    # ========================================================
    # CH -> SINK DISTANCE
    # ========================================================

    sink_distance = torch.linalg.vector_norm(

        ch_positions

        -

        state[
            "sink"
        ][
            None,
            None,
            :
        ],

        dim=-1
    )


    mean_sink_distance = (
        sink_distance.mean(
            dim=1
        )
    )


    sink_penalty = (

        mean_sink_distance

        /

        MAX_SINK_DISTANCE
    )


    # ========================================================
    # LOAD-BALANCE PENALTY
    # ========================================================

    cluster_one_hot = (
        torch.nn.functional.one_hot(

            cluster_id,

            num_classes=num_ch

        ).float()
    )


    cluster_one_hot *= (
        alive_matrix[
            :,
            :,
            None
        ].float()
    )


    cluster_sizes = (
        cluster_one_hot.sum(
            dim=1
        )
    )


    mean_cluster_size = (
        cluster_sizes.mean(
            dim=1
        )
    )


    cluster_std = (
        cluster_sizes.std(
            dim=1,
            unbiased=False
        )
    )


    load_penalty = (

        cluster_std

        /

        mean_cluster_size.clamp_min(
            1e-8
        )
    )


    # ========================================================
    # FINAL FITNESS
    # ========================================================

    fitness = (

        W_ENERGY
        *
        energy_penalty

        +

        W_INTRA
        *
        intra_penalty

        +

        W_SINK
        *
        sink_penalty

        +

        W_LOAD
        *
        load_penalty
    )


    details = {

        "energy_penalty":
            energy_penalty,

        "mean_intra_distance":
            intra_distance,

        "mean_ch_sink_distance":
            mean_sink_distance,

        "load_penalty":
            load_penalty,

        "cluster_sizes":
            cluster_sizes
    }


    return (
        fitness,
        ch_indices,
        details
    )


print(
    "GPU fitness function loaded."
)

# ===== Notebook code cell 4 =====
# ============================================================
# CELL 5: GPU FITNESS SANITY CHECK
# ============================================================

generator = torch.Generator(
    device=device0
)

generator.manual_seed(
    2026
)


test_population = torch.rand(

    (
        100,
        NUM_NODES
    ),

    generator=generator,

    device=device0
)


(
    test_fitness,
    test_ch,
    test_details
) = evaluate_population_gpu(

    test_population,

    gpu_state,

    NUM_CH
)


print("=" * 60)

print(
    "GPU FITNESS SANITY TEST"
)

print("=" * 60)

print(
    "Best fitness:",
    float(
        test_fitness.min()
    )
)

print(
    "Worst fitness:",
    float(
        test_fitness.max()
    )
)

print(
    "Mean fitness:",
    float(
        test_fitness.mean()
    )
)

print(
    "Std fitness:",
    float(
        test_fitness.std()
    )
)

# ===== Notebook code cell 5 =====
# ============================================================
# CELL 6: ORIGINAL CWOA - GPU VERSION
# ============================================================

@torch.no_grad()
def run_original_cwoa_gpu(
    nodes_df,
    num_ch=NUM_CH,
    population_size=30,
    evaluation_budget=1500,
    seed=2026,
    device=None
):

    if device is None:
        device = DEVICES[0]


    state = network_to_gpu(
        nodes_df,
        device
    )


    n = len(
        nodes_df
    )


    iterations = max(

        1,

        evaluation_budget
        //
        population_size
    )


    g = torch.Generator(
        device=device
    )

    g.manual_seed(
        seed
    )


    whales = torch.rand(

        (
            population_size,
            n
        ),

        generator=g,

        device=device
    )


    best_fitness = float(
        "inf"
    )

    best_whale = None

    best_ch = None

    best_details = None

    convergence = []


    for iteration in range(
        iterations
    ):

        (
            fitness,
            ch_indices,
            details
        ) = evaluate_population_gpu(

            whales,
            state,
            num_ch
        )


        current_best_idx = int(
            torch.argmin(
                fitness
            ).item()
        )


        current_best = float(
            fitness[
                current_best_idx
            ].item()
        )


        if current_best < best_fitness:

            best_fitness = (
                current_best
            )

            best_whale = (
                whales[
                    current_best_idx
                ].clone()
            )

            best_ch = (
                ch_indices[
                    current_best_idx
                ].clone()
            )

            best_details = {

                key:
                    value[
                        current_best_idx
                    ].detach().cpu()

                for key, value
                in details.items()
            }


        convergence.append(
            best_fitness
        )


        # ================================================
        # WOA PARAMETER
        # ================================================

        if iterations > 1:

            a = (
                2.0
                -
                2.0
                *
                iteration
                /
                (
                    iterations - 1
                )
            )

        else:

            a = 0.0


        r1 = torch.rand(

            (
                population_size,
                1
            ),

            generator=g,

            device=device
        )


        r2 = torch.rand(

            (
                population_size,
                1
            ),

            generator=g,

            device=device
        )


        A = (
            2.0
            *
            a
            *
            r1

            -

            a
        )


        C = (
            2.0
            *
            r2
        )


        p = torch.rand(

            (
                population_size,
                1
            ),

            generator=g,

            device=device
        )


        # ================================================
        # EXPLOITATION
        # ================================================

        D_best = torch.abs(

            C
            *
            best_whale[
                None,
                :
            ]

            -

            whales
        )


        exploitation = (

            best_whale[
                None,
                :
            ]

            -

            A
            *
            D_best
        )


        # ================================================
        # EXPLORATION
        # ================================================

        random_indices = (
            torch.randint(

                0,

                population_size,

                (
                    population_size,
                ),

                generator=g,

                device=device
            )
        )


        random_whales = whales[
            random_indices
        ]


        D_random = torch.abs(

            C
            *
            random_whales

            -

            whales
        )


        exploration = (

            random_whales

            -

            A
            *
            D_random
        )


        encircling = torch.where(

            (
                torch.abs(
                    A
                )
                <
                1
            ).expand_as(
                whales
            ),

            exploitation,

            exploration
        )


        # ================================================
        # SPIRAL
        # ================================================

        l = (

            torch.rand(

                (
                    population_size,
                    1
                ),

                generator=g,

                device=device
            )

            *
            2

            -

            1
        )


        distance_best = torch.abs(

            best_whale[
                None,
                :
            ]

            -

            whales
        )


        spiral = (

            distance_best

            *
            torch.exp(
                l
            )

            *
            torch.cos(
                2
                *
                math.pi
                *
                l
            )

            +

            best_whale[
                None,
                :
            ]
        )


        whales = torch.where(

            (
                p
                <
                0.5
            ).expand_as(
                whales
            ),

            encircling,

            spiral
        )


        whales.clamp_(
            0.0,
            1.0
        )


    return {

        "ch_indices":
            best_ch.detach().cpu().numpy(),

        "fitness":
            best_fitness,

        "convergence":
            np.asarray(
                convergence
            ),

        "details":
            best_details,

        "evaluations":
            iterations
            *
            population_size
    }


print(
    "Original GPU CWOA loaded."
)

# ===== Notebook code cell 6 =====
# ============================================================
# CELL 7: IMPROVED ADAPTIVE DISCRETE CWOA - GPU
# ============================================================

def encode_ch_solution_gpu(
    ch_indices,
    num_nodes,
    generator,
    device
):

    keys = (

        torch.rand(

            num_nodes,

            generator=generator,

            device=device
        )

        *
        0.45
    )


    keys[
        ch_indices
    ] = (

        0.55

        +

        torch.rand(

            len(
                ch_indices
            ),

            generator=generator,

            device=device
        )

        *
        0.45
    )


    return keys


@torch.no_grad()
def run_improved_cwoa_gpu(
    nodes_df,
    num_ch=NUM_CH,
    population_size=30,
    evaluation_budget=1500,
    local_trials=5,
    stagnation_limit=4,
    restart_fraction=0.30,
    seed=2026,
    device=None
):

    if device is None:

        device = DEVICES[0]


    state = network_to_gpu(
        nodes_df,
        device
    )


    n = len(
        nodes_df
    )


    g = torch.Generator(
        device=device
    )


    g.manual_seed(
        seed
    )


    whales = torch.rand(

        (
            population_size,
            n
        ),

        generator=g,

        device=device
    )


    best_fitness = float(
        "inf"
    )

    best_whale = None

    best_ch = None

    best_details = None


    evaluations = 0

    stagnation_counter = 0


    evaluation_history = []

    fitness_history = []


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while evaluations < evaluation_budget:

        remaining = (

            evaluation_budget

            -

            evaluations
        )


        current_pop_size = min(

            population_size,

            remaining
        )


        population_slice = whales[
            :current_pop_size
        ]


        (
            fitness,
            ch_indices,
            details
        ) = evaluate_population_gpu(

            population_slice,

            state,

            num_ch
        )


        evaluations += (
            current_pop_size
        )


        current_best_idx = int(
            torch.argmin(
                fitness
            ).item()
        )


        current_best = float(

            fitness[
                current_best_idx
            ].item()
        )


        previous_best = (
            best_fitness
        )


        if current_best < best_fitness:

            best_fitness = (
                current_best
            )

            best_whale = (

                population_slice[
                    current_best_idx
                ].clone()
            )

            best_ch = (

                ch_indices[
                    current_best_idx
                ].clone()
            )

            best_details = {

                key:
                    value[
                        current_best_idx
                    ].detach().cpu()

                for key, value
                in details.items()
            }


        # ====================================================
        # DISCRETE LOCAL SEARCH
        # ====================================================

        available_local = min(

            local_trials,

            evaluation_budget
            -
            evaluations
        )


        if (
            available_local > 0

            and

            best_ch is not None
        ):

            alive_idx = torch.where(

                state[
                    "alive"
                ]

            )[0]


            local_keys = []


            for _ in range(
                available_local
            ):

                candidate_ch = (
                    best_ch.clone()
                )


                remove_position = int(

                    torch.randint(

                        0,

                        num_ch,

                        (
                            1,
                        ),

                        generator=g,

                        device=device

                    ).item()
                )


                ch_mask = torch.zeros(

                    n,

                    dtype=torch.bool,

                    device=device
                )


                ch_mask[
                    candidate_ch
                ] = True


                possible_nodes = (

                    alive_idx[
                        ~ch_mask[
                            alive_idx
                        ]
                    ]
                )


                if len(
                    possible_nodes
                ) > 0:

                    selected = int(

                        torch.randint(

                            0,

                            len(
                                possible_nodes
                            ),

                            (
                                1,
                            ),

                            generator=g,

                            device=device

                        ).item()
                    )


                    candidate_ch[
                        remove_position
                    ] = possible_nodes[
                        selected
                    ]


                candidate_keys = (
                    encode_ch_solution_gpu(

                        candidate_ch,

                        n,

                        g,

                        device
                    )
                )


                local_keys.append(
                    candidate_keys
                )


            local_population = (
                torch.stack(
                    local_keys
                )
            )


            (
                local_fitness,
                local_ch,
                local_details
            ) = evaluate_population_gpu(

                local_population,

                state,

                num_ch
            )


            evaluations += (
                available_local
            )


            local_best_idx = int(

                torch.argmin(
                    local_fitness
                ).item()
            )


            local_best = float(

                local_fitness[
                    local_best_idx
                ].item()
            )


            if local_best < best_fitness:

                best_fitness = (
                    local_best
                )

                best_whale = (

                    local_population[
                        local_best_idx
                    ].clone()
                )

                best_ch = (

                    local_ch[
                        local_best_idx
                    ].clone()
                )

                best_details = {

                    key:
                        value[
                            local_best_idx
                        ].detach().cpu()

                    for key, value
                    in local_details.items()
                }


        evaluation_history.append(
            evaluations
        )

        fitness_history.append(
            best_fitness
        )


        # ====================================================
        # STAGNATION CHECK
        # ====================================================

        if (
            best_fitness
            <
            previous_best
            -
            1e-12
        ):

            stagnation_counter = 0

        else:

            stagnation_counter += 1


        if evaluations >= evaluation_budget:

            break


        # ====================================================
        # WOA UPDATE
        # ====================================================

        progress = (

            evaluations

            /
            evaluation_budget
        )


        a = (

            2.0

            *
            (
                1.0
                -
                progress
            )
        )


        r1 = torch.rand(

            (
                population_size,
                1
            ),

            generator=g,

            device=device
        )


        r2 = torch.rand(

            (
                population_size,
                1
            ),

            generator=g,

            device=device
        )


        A = (

            2
            *
            a
            *
            r1

            -

            a
        )


        C = (
            2
            *
            r2
        )


        p = torch.rand(

            (
                population_size,
                1
            ),

            generator=g,

            device=device
        )


        D_best = torch.abs(

            C
            *
            best_whale[
                None,
                :
            ]

            -

            whales
        )


        exploitation = (

            best_whale[
                None,
                :
            ]

            -

            A
            *
            D_best
        )


        random_indices = torch.randint(

            0,

            population_size,

            (
                population_size,
            ),

            generator=g,

            device=device
        )


        random_whales = whales[
            random_indices
        ]


        D_random = torch.abs(

            C
            *
            random_whales

            -

            whales
        )


        exploration = (

            random_whales

            -

            A
            *
            D_random
        )


        encircling = torch.where(

            (
                torch.abs(
                    A
                )
                <
                1
            ).expand_as(
                whales
            ),

            exploitation,

            exploration
        )


        l = (

            torch.rand(

                (
                    population_size,
                    1
                ),

                generator=g,

                device=device
            )

            *
            2

            -

            1
        )


        distance_best = torch.abs(

            best_whale[
                None,
                :
            ]

            -

            whales
        )


        spiral = (

            distance_best

            *
            torch.exp(
                l
            )

            *
            torch.cos(

                2
                *
                math.pi
                *
                l
            )

            +

            best_whale[
                None,
                :
            ]
        )


        whales = torch.where(

            (
                p
                <
                0.5
            ).expand_as(
                whales
            ),

            encircling,

            spiral
        )


        # ====================================================
        # ADAPTIVE MUTATION
        # ====================================================

        mutation_probability = (

            0.15

            +

            0.20
            *
            progress
        )


        mutation_scale = (

            0.15
            *
            (
                1
                -
                progress
            )

            +

            0.03
        )


        mutation_count = max(

            1,

            int(
                0.05
                *
                n
            )
        )


        mutation_whales = torch.where(

            torch.rand(

                population_size,

                generator=g,

                device=device

            )
            <
            mutation_probability

        )[0]


        for whale_idx in (
            mutation_whales.tolist()
        ):

            mutation_indices = (

                torch.randperm(

                    n,

                    generator=g,

                    device=device

                )[
                    :mutation_count
                ]
            )


            whales[
                whale_idx,
                mutation_indices
            ] += (

                torch.randn(

                    mutation_count,

                    generator=g,

                    device=device
                )

                *
                mutation_scale
            )


        whales.clamp_(
            0.0,
            1.0
        )


        # ====================================================
        # DIVERSITY RESTART
        # ====================================================

        if (
            stagnation_counter
            >=
            stagnation_limit
        ):

            restart_count = max(

                1,

                int(
                    population_size
                    *
                    restart_fraction
                )
            )


            restart_indices = (

                torch.randperm(

                    population_size,

                    generator=g,

                    device=device

                )[
                    :restart_count
                ]
            )


            whales[
                restart_indices
            ] = torch.rand(

                (
                    restart_count,
                    n
                ),

                generator=g,

                device=device
            )


            stagnation_counter = 0


    return {

        "ch_indices":
            best_ch.detach().cpu().numpy(),

        "fitness":
            best_fitness,

        "evaluation_history":
            np.asarray(
                evaluation_history
            ),

        "convergence":
            np.asarray(
                fitness_history
            ),

        "details":
            best_details,

        "evaluations":
            evaluations
    }


print(
    "Improved GPU CWOA loaded."
)

# ===== Notebook code cell 7 =====
# ============================================================
# CELL 8: GPU RANDOM SEARCH
# ============================================================

@torch.no_grad()
def run_random_search_gpu(
    nodes_df,
    num_ch=NUM_CH,
    evaluation_budget=1500,
    batch_size=256,
    seed=2026,
    device=None
):

    if device is None:

        device = DEVICES[0]


    state = network_to_gpu(
        nodes_df,
        device
    )


    n = len(
        nodes_df
    )


    g = torch.Generator(
        device=device
    )


    g.manual_seed(
        seed
    )


    best_fitness = float(
        "inf"
    )

    best_ch = None

    evaluations = 0


    while evaluations < evaluation_budget:

        current_batch = min(

            batch_size,

            evaluation_budget
            -
            evaluations
        )


        keys = torch.rand(

            (
                current_batch,
                n
            ),

            generator=g,

            device=device
        )


        (
            fitness,
            ch_indices,
            _
        ) = evaluate_population_gpu(

            keys,

            state,

            num_ch
        )


        best_idx = int(

            torch.argmin(
                fitness
            ).item()
        )


        current_best = float(

            fitness[
                best_idx
            ].item()
        )


        if current_best < best_fitness:

            best_fitness = (
                current_best
            )

            best_ch = (

                ch_indices[
                    best_idx
                ].clone()
            )


        evaluations += (
            current_batch
        )


    return {

        "ch_indices":
            best_ch.detach().cpu().numpy(),

        "fitness":
            best_fitness,

        "evaluations":
            evaluations
    }


print(
    "GPU Random Search loaded."
)

# ===== Notebook code cell 8 =====
# ============================================================
# CELL 9: GPU OPTIMIZER BENCHMARK
# ============================================================

benchmark_nodes = (
    create_sensor_network(
        topology_seed=5000
    )
)


device = DEVICES[0]


if torch.cuda.is_available():

    torch.cuda.synchronize(
        device
    )


start = time.perf_counter()


original_result = (
    run_original_cwoa_gpu(

        benchmark_nodes,

        evaluation_budget=1500,

        seed=2026,

        device=device
    )
)


if torch.cuda.is_available():

    torch.cuda.synchronize(
        device
    )


original_time = (
    time.perf_counter()
    -
    start
)


# ------------------------------------------------------------

if torch.cuda.is_available():

    torch.cuda.synchronize(
        device
    )


start = time.perf_counter()


improved_result = (
    run_improved_cwoa_gpu(

        benchmark_nodes,

        evaluation_budget=1500,

        seed=2026,

        device=device
    )
)


if torch.cuda.is_available():

    torch.cuda.synchronize(
        device
    )


improved_time = (
    time.perf_counter()
    -
    start
)


# ------------------------------------------------------------

if torch.cuda.is_available():

    torch.cuda.synchronize(
        device
    )


start = time.perf_counter()


random_result = (
    run_random_search_gpu(

        benchmark_nodes,

        evaluation_budget=1500,

        seed=2026,

        device=device
    )
)


if torch.cuda.is_available():

    torch.cuda.synchronize(
        device
    )


random_time = (
    time.perf_counter()
    -
    start
)


benchmark_df = pd.DataFrame({

    "Method": [

        "Original CWOA",

        "Improved CWOA",

        "Random Search"
    ],

    "Best Fitness": [

        original_result[
            "fitness"
        ],

        improved_result[
            "fitness"
        ],

        random_result[
            "fitness"
        ]
    ],

    "Evaluations": [

        original_result[
            "evaluations"
        ],

        improved_result[
            "evaluations"
        ],

        random_result[
            "evaluations"
        ]
    ],

    "Runtime (sec)": [

        original_time,

        improved_time,

        random_time
    ]
})


display(
    benchmark_df
)

# ===== Notebook code cell 9 =====
# ============================================================
# CELL 10: RADIO ENERGY MODEL
# ============================================================

def transmission_energy(
    bits,
    distance
):

    distance = np.asarray(
        distance
    )


    free_space = (

        bits
        *
        E_ELEC

        +

        bits
        *
        E_FS
        *
        distance**2
    )


    multipath = (

        bits
        *
        E_ELEC

        +

        bits
        *
        E_MP
        *
        distance**4
    )


    return np.where(

        distance
        <
        D0,

        free_space,

        multipath
    )


def reception_energy(
    bits
):

    return (
        bits
        *
        E_ELEC
    )


def aggregation_energy(
    bits
):

    return (
        bits
        *
        E_DA
    )


print(
    "Radio energy model loaded."
)

# ===== Notebook code cell 10 =====
# ============================================================
# CELL 11: CPU CLUSTER ASSIGNMENT
# ============================================================

def assign_clusters_cpu(
    nodes_df,
    ch_indices
):

    positions = nodes_df[
        ["x", "y"]
    ].to_numpy()


    alive = nodes_df[
        "alive"
    ].to_numpy()


    ch_indices = np.asarray(
        ch_indices,
        dtype=int
    )


    ch_positions = positions[
        ch_indices
    ]


    diff = (

        positions[
            :,
            None,
            :
        ]

        -

        ch_positions[
            None,
            :,
            :
        ]
    )


    distances = np.linalg.norm(
        diff,
        axis=2
    )


    cluster_id = np.argmin(
        distances,
        axis=1
    )


    nearest_distance = np.min(
        distances,
        axis=1
    )


    cluster_id[
        ~alive
    ] = -1


    nearest_distance[
        ~alive
    ] = np.nan


    return (
        cluster_id,
        nearest_distance
    )

# ===== Notebook code cell 11 =====
# ============================================================
# CELL 12: FAST WSN COMMUNICATION ROUND
# ============================================================

def simulate_round(
    nodes_df,
    ch_indices
):

    network = nodes_df.copy()


    energy = network[
        "energy"
    ].to_numpy(
        copy=True
    )


    alive = (
        energy
        >
        0
    )


    network[
        "alive"
    ] = alive


    ch_indices = np.asarray(

        [
            idx
            for idx in ch_indices
            if alive[
                idx
            ]
        ],

        dtype=int
    )


    if len(
        ch_indices
    ) == 0:

        return network, {

            "packets_to_ch": 0,

            "ch_transmissions":
                0,

            "source_packets":
                0,

            "residual_energy":
                energy.sum(),

            "energy_consumed":
                0.0,

            "alive_nodes":
                int(
                    alive.sum()
                ),

            "dead_nodes":
                int(
                    (~alive).sum()
                )
        }


    energy_before = (
        energy.sum()
    )


    (
        cluster_id,
        distance_to_ch
    ) = assign_clusters_cpu(

        network,
        ch_indices
    )


    is_ch = np.zeros(
        len(
            network
        ),
        dtype=bool
    )


    is_ch[
        ch_indices
    ] = True


    member_mask = (

        alive

        &
        (~is_ch)
    )


    member_indices = np.where(
        member_mask
    )[0]


    member_distances = (
        distance_to_ch[
            member_indices
        ]
    )


    tx_costs = (
        transmission_energy(

            PACKET_SIZE,

            member_distances
        )
    )


    can_send = (

        energy[
            member_indices
        ]

        >=

        tx_costs
    )


    successful_members = (

        member_indices[
            can_send
        ]
    )


    failed_members = (

        member_indices[
            ~can_send
        ]
    )


    energy[
        successful_members
    ] -= tx_costs[
        can_send
    ]


    energy[
        failed_members
    ] = 0.0


    packets_to_ch = len(
        successful_members
    )


    received_per_ch = np.zeros(

        len(
            ch_indices
        ),

        dtype=int
    )


    successful_clusters = (

        cluster_id[
            successful_members
        ]
    )


    for cid in (
        successful_clusters
    ):

        received_per_ch[
            cid
        ] += 1


    source_packets = 0

    ch_transmissions = 0


    positions = network[
        ["x", "y"]
    ].to_numpy()


    for local_ch_id, ch_idx in enumerate(
        ch_indices
    ):

        if energy[
            ch_idx
        ] <= 0:

            continue


        received = int(
            received_per_ch[
                local_ch_id
            ]
        )


        cluster_sources = (
            received
            +
            1
        )


        rx_cost = (

            received

            *
            reception_energy(
                PACKET_SIZE
            )
        )


        da_cost = (

            cluster_sources

            *
            aggregation_energy(
                PACKET_SIZE
            )
        )


        distance_sink = np.linalg.norm(

            positions[
                ch_idx
            ]

            -

            np.asarray(
                [
                    SINK_X,
                    SINK_Y
                ]
            )
        )


        tx_sink_cost = float(

            transmission_energy(

                PACKET_SIZE,

                distance_sink
            )
        )


        total_cost = (

            rx_cost

            +

            da_cost

            +

            tx_sink_cost
        )


        if energy[
            ch_idx
        ] >= total_cost:

            energy[
                ch_idx
            ] -= total_cost

            ch_transmissions += 1

            source_packets += (
                cluster_sources
            )


        else:

            energy[
                ch_idx
            ] = 0.0


    energy = np.clip(
        energy,
        0.0,
        None
    )


    alive = (
        energy
        >
        0
    )


    network[
        "energy"
    ] = energy


    network[
        "alive"
    ] = alive


    residual_energy = (
        energy.sum()
    )


    return network, {

        "packets_to_ch":
            packets_to_ch,

        "ch_transmissions":
            ch_transmissions,

        "source_packets":
            source_packets,

        "energy_consumed":
            energy_before
            -
            residual_energy,

        "residual_energy":
            residual_energy,

        "alive_nodes":
            int(
                alive.sum()
            ),

        "dead_nodes":
            int(
                (~alive).sum()
            )
    }

# ===== Notebook code cell 12 =====
# ============================================================
# CELL 13: UNIFIED GPU CH SELECTOR
# ============================================================

def select_cluster_heads_gpu(
    nodes_df,
    method,
    num_ch,
    evaluation_budget,
    seed,
    device
):

    if method == "original_cwoa":

        result = (
            run_original_cwoa_gpu(

                nodes_df,

                num_ch=num_ch,

                population_size=
                    POPULATION_SIZE,

                evaluation_budget=
                    evaluation_budget,

                seed=seed,

                device=device
            )
        )


    elif method == "improved_cwoa":

        result = (
            run_improved_cwoa_gpu(

                nodes_df,

                num_ch=num_ch,

                population_size=
                    POPULATION_SIZE,

                evaluation_budget=
                    evaluation_budget,

                local_trials=3,

                stagnation_limit=3,

                restart_fraction=0.30,

                seed=seed,

                device=device
            )
        )


    elif method == "random_search":

        result = (
            run_random_search_gpu(

                nodes_df,

                num_ch=num_ch,

                evaluation_budget=
                    evaluation_budget,

                seed=seed,

                device=device
            )
        )


    else:

        raise ValueError(
            f"Unknown method: {method}"
        )


    return result

# ===== Notebook code cell 13 =====
# ============================================================
# CELL 14: GPU-ASSISTED NETWORK LIFETIME SIMULATOR
# ============================================================

def simulate_lifetime_gpu(
    topology_seed,
    method,
    optimizer_seed,
    device,

    max_rounds=5000,

    num_ch=NUM_CH,

    reselection_interval=
        CH_RESELECTION_INTERVAL,

    evaluation_budget=
        LIFETIME_OPTIMIZER_BUDGET,

    verbose=False
):

    network = create_sensor_network(
        topology_seed
    )


    rng = np.random.default_rng(
        optimizer_seed
    )


    current_ch = None


    history = []


    cumulative_source_packets = 0

    cumulative_ch_transmissions = 0

    total_optimizer_evaluations = 0


    FND = None
    HND = None
    LND = None


    half_nodes = int(
        np.ceil(
            len(
                network
            )
            /
            2
        )
    )


    for round_number in range(

        1,

        max_rounds + 1
    ):

        alive_count = int(

            network[
                "alive"
            ].sum()
        )


        if alive_count == 0:

            LND = (
                round_number - 1
            )

            break


        active_num_ch = min(

            num_ch,

            alive_count
        )


        need_reselection = (

            current_ch is None

            or

            (
                (
                    round_number
                    -
                    1
                )
                %
                reselection_interval
                ==
                0
            )

            or

            any(

                not bool(

                    network.loc[
                        idx,
                        "alive"
                    ]
                )

                for idx
                in current_ch
            )
        )


        if need_reselection:

            seed_event = int(

                rng.integers(

                    0,

                    2**31
                    -
                    1
                )
            )


            result = (
                select_cluster_heads_gpu(

                    network,

                    method,

                    active_num_ch,

                    evaluation_budget,

                    seed_event,

                    device
                )
            )


            current_ch = (
                result[
                    "ch_indices"
                ]
            )


            total_optimizer_evaluations += (

                result[
                    "evaluations"
                ]
            )


        network, stats_round = (
            simulate_round(

                network,

                current_ch
            )
        )


        cumulative_source_packets += (

            stats_round[
                "source_packets"
            ]
        )


        cumulative_ch_transmissions += (

            stats_round[
                "ch_transmissions"
            ]
        )


        alive_now = (

            stats_round[
                "alive_nodes"
            ]
        )


        dead_now = (

            stats_round[
                "dead_nodes"
            ]
        )


        if (

            FND is None

            and

            dead_now >= 1
        ):

            FND = (
                round_number
            )


        if (

            HND is None

            and

            dead_now
            >=
            half_nodes
        ):

            HND = (
                round_number
            )


        if alive_now == 0:

            LND = (
                round_number
            )


        history.append({

            "round":
                round_number,

            "alive_nodes":
                alive_now,

            "dead_nodes":
                dead_now,

            "residual_energy":
                stats_round[
                    "residual_energy"
                ],

            "source_packets":
                stats_round[
                    "source_packets"
                ],

            "cumulative_source_packets":
                cumulative_source_packets,

            "cumulative_ch_transmissions":
                cumulative_ch_transmissions
        })


        if (

            verbose

            and

            round_number
            %
            500
            ==
            0
        ):

            print(

                method,

                "|",

                round_number,

                "| Alive:",

                alive_now,

                "| Energy:",

                round(
                    stats_round[
                        "residual_energy"
                    ],
                    3
                )
            )


        if alive_now == 0:

            break


    history_df = pd.DataFrame(
        history
    )


    summary = {

        "method":
            method,

        "topology_seed":
            topology_seed,

        "optimizer_seed":
            optimizer_seed,

        "device":
            str(
                device
            ),

        "FND":
            FND,

        "HND":
            HND,

        "LND":
            LND,

        "throughput":
            cumulative_source_packets,

        "ch_transmissions":
            cumulative_ch_transmissions,

        "optimizer_evaluations":
            total_optimizer_evaluations,

        "final_energy":
            float(
                network[
                    "energy"
                ].sum()
            )
    }


    return (
        summary,
        history_df
    )


print(
    "GPU-assisted lifetime simulator loaded."
)

# ===== Notebook code cell 14 =====
# ============================================================
# CELL 15: DUAL-GPU PARALLEL EXPERIMENT RUNNER
# ============================================================

from concurrent.futures import (
    ThreadPoolExecutor
)


def dual_gpu_worker(
    device,
    jobs
):

    results = []

    histories = {}


    for job in jobs:

        method = job[
            "method"
        ]

        topology_seed = job[
            "topology_seed"
        ]

        optimizer_seed = job[
            "optimizer_seed"
        ]


        summary, history = (
            simulate_lifetime_gpu(

                topology_seed=
                    topology_seed,

                method=
                    method,

                optimizer_seed=
                    optimizer_seed,

                device=
                    device,

                max_rounds=
                    job.get(
                        "max_rounds",
                        5000
                    )
            )
        )


        key = (

            f"{method}_"
            f"{topology_seed}_"
            f"{optimizer_seed}"
        )


        results.append(
            summary
        )

        histories[
            key
        ] = history


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    return (
        results,
        histories
    )


def run_dual_gpu_experiments(
    jobs
):

    device_count = len(
        DEVICES
    )


    job_groups = [

        []

        for _
        in range(
            device_count
        )
    ]


    for index, job in enumerate(
        jobs
    ):

        job_groups[
            index
            %
            device_count
        ].append(
            job
        )


    all_results = []

    all_histories = {}


    with ThreadPoolExecutor(

        max_workers=
            device_count

    ) as executor:

        futures = []


        for device_index in range(
            device_count
        ):

            futures.append(

                executor.submit(

                    dual_gpu_worker,

                    DEVICES[
                        device_index
                    ],

                    job_groups[
                        device_index
                    ]
                )
            )


        for future in futures:

            results, histories = (
                future.result()
            )


            all_results.extend(
                results
            )

            all_histories.update(
                histories
            )


    return (

        pd.DataFrame(
            all_results
        ),

        all_histories
    )


print(
    "Dual-GPU experiment runner loaded."
)

# ===== Notebook code cell 15 =====
# ============================================================
# CELL 16: SAME-TOPOLOGY LIFETIME COMPARISON
# ============================================================

jobs = [

    {
        "method":
            "original_cwoa",

        "topology_seed":
            5000,

        "optimizer_seed":
            9500
    },

    {
        "method":
            "random_search",

        "topology_seed":
            5000,

        "optimizer_seed":
            9500
    },

    {
        "method":
            "improved_cwoa",

        "topology_seed":
            5000,

        "optimizer_seed":
            9500
    }
]


start = time.perf_counter()


(
    lifetime_results_gpu,
    lifetime_histories_gpu
) = run_dual_gpu_experiments(
    jobs
)


runtime = (
    time.perf_counter()
    -
    start
)


print(
    f"Experiment runtime: "
    f"{runtime:.2f} seconds"
)


display(
    lifetime_results_gpu[
        [
            "method",
            "device",
            "FND",
            "HND",
            "LND",
            "throughput",
            "optimizer_evaluations"
        ]
    ]
)

# ===== Notebook code cell 16 =====
# ============================================================
# CELL 17: INTERACTIVE NETWORK LIFETIME GRAPH
# ============================================================

fig = go.Figure()


for key, history in (
    lifetime_histories_gpu.items()
):

    method = (
        key.split(
            "_5000_"
        )[0]
    )


    label = {

        "original_cwoa":
            "Original CWOA",

        "random_search":
            "Random Search",

        "improved_cwoa":
            "Improved CWOA"

    }.get(
        method,
        method
    )


    fig.add_trace(

        go.Scatter(

            x=
                history[
                    "round"
                ],

            y=
                history[
                    "alive_nodes"
                ],

            mode=
                "lines",

            name=
                label,

            hovertemplate=

                "Round: %{x}"
                "<br>"
                "Alive nodes: %{y}"
                "<extra></extra>"
        )
    )


fig.update_layout(

    title=
        "Network Lifetime Comparison",

    xaxis_title=
        "Round",

    yaxis_title=
        "Number of Alive Nodes",

    template=
        "plotly_white",

    width=
        1000,

    height=
        600,

    hovermode=
        "x unified"
)


fig.show()


fig.write_html(

    "/kaggle/working/"
    "network_lifetime_interactive.html"
)

# ===== Notebook code cell 17 =====
# ============================================================
# CELL 18: PAPER-QUALITY ALIVE-NODE FIGURE
# ============================================================

plt.figure(
    figsize=(
        10,
        6
    )
)


for key, history in (
    lifetime_histories_gpu.items()
):

    method = (
        key.split(
            "_5000_"
        )[0]
    )


    label = {

        "original_cwoa":
            "Original CWOA",

        "random_search":
            "Random Search",

        "improved_cwoa":
            "Improved CWOA"

    }.get(
        method,
        method
    )


    plt.plot(

        history[
            "round"
        ],

        history[
            "alive_nodes"
        ],

        linewidth=2,

        label=label
    )


plt.xlabel(
    "Round",
    fontsize=12
)

plt.ylabel(
    "Number of Alive Nodes",
    fontsize=12
)

plt.title(
    "Network Lifetime Comparison",
    fontsize=14
)

plt.grid(
    True,
    alpha=0.25
)

plt.legend()

plt.tight_layout()


plt.savefig(

    "/kaggle/working/"
    "network_lifetime_600dpi.png",

    dpi=600,

    bbox_inches="tight"
)


plt.savefig(

    "/kaggle/working/"
    "network_lifetime.pdf",

    bbox_inches="tight"
)


plt.savefig(

    "/kaggle/working/"
    "network_lifetime.svg",

    bbox_inches="tight"
)


plt.show()

# ===== Notebook code cell 18 =====
# ============================================================
# CELL 19: RESIDUAL ENERGY COMPARISON
# ============================================================

plt.figure(
    figsize=(
        10,
        6
    )
)


for key, history in (
    lifetime_histories_gpu.items()
):

    method = (
        key.split(
            "_5000_"
        )[0]
    )


    label = {

        "original_cwoa":
            "Original CWOA",

        "random_search":
            "Random Search",

        "improved_cwoa":
            "Improved CWOA"

    }.get(
        method,
        method
    )


    plt.plot(

        history[
            "round"
        ],

        history[
            "residual_energy"
        ],

        linewidth=2,

        label=label
    )


plt.xlabel(
    "Round"
)

plt.ylabel(
    "Residual Network Energy (J)"
)

plt.title(
    "Residual Network Energy Comparison"
)

plt.grid(
    True,
    alpha=0.25
)

plt.legend()

plt.tight_layout()


plt.savefig(

    "/kaggle/working/"
    "residual_energy_600dpi.png",

    dpi=600,

    bbox_inches="tight"
)


plt.savefig(

    "/kaggle/working/"
    "residual_energy.pdf",

    bbox_inches="tight"
)


plt.show()

# ===== Notebook code cell 19 =====
# ============================================================
# CELL 20: MULTI-TOPOLOGY DUAL-GPU EXPERIMENT
# ============================================================

NUM_TOPOLOGIES = 20


methods = [

    "original_cwoa",

    "random_search",

    "improved_cwoa"
]


jobs = []


for topology_index in range(
    NUM_TOPOLOGIES
):

    topology_seed = (
        5000
        +
        topology_index
    )


    optimizer_seed = (
        9000
        +
        topology_index
    )


    for method in methods:

        jobs.append({

            "method":
                method,

            "topology_seed":
                topology_seed,

            "optimizer_seed":
                optimizer_seed,

            "max_rounds":
                5000
        })


print(
    "Total lifetime jobs:",
    len(
        jobs
    )
)


start = time.perf_counter()


(
    multi_results,
    multi_histories
) = run_dual_gpu_experiments(
    jobs
)


elapsed = (
    time.perf_counter()
    -
    start
)


print(
    f"Total dual-GPU runtime: "
    f"{elapsed / 60:.2f} minutes"
)


display(
    multi_results.head(
        10
    )
)

# ===== Notebook code cell 20 =====
# ============================================================
# CELL 21: MULTI-TOPOLOGY STATISTICAL SUMMARY
# ============================================================

summary = (

    multi_results

    .groupby(
        "method"
    )

    .agg({

        "FND":
            [
                "mean",
                "std"
            ],

        "HND":
            [
                "mean",
                "std"
            ],

        "LND":
            [
                "mean",
                "std"
            ],

        "throughput":
            [
                "mean",
                "std"
            ]
    })
)


display(
    summary
)

# ===== Notebook code cell 21 =====
# ============================================================
# CELL 22: FND / HND / LND PAPER FIGURE
# ============================================================

metric_means = (

    multi_results

    .groupby(
        "method"
    )[
        [
            "FND",
            "HND",
            "LND"
        ]
    ]

    .mean()
)


metric_means.index = [

    {
        "original_cwoa":
            "Original CWOA",

        "random_search":
            "Random Search",

        "improved_cwoa":
            "Improved CWOA"

    }.get(
        x,
        x
    )

    for x
    in metric_means.index
]


ax = metric_means.plot(

    kind="bar",

    figsize=(
        10,
        6
    )
)


ax.set_ylabel(
    "Round"
)

ax.set_xlabel(
    "Method"
)

ax.set_title(
    "Comparison of FND, HND and LND"
)

plt.xticks(
    rotation=0
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()


plt.savefig(

    "/kaggle/working/"
    "FND_HND_LND_comparison.png",

    dpi=600,

    bbox_inches="tight"
)


plt.savefig(

    "/kaggle/working/"
    "FND_HND_LND_comparison.pdf",

    bbox_inches="tight"
)


plt.show()

# ===== Notebook code cell 22 =====
# ============================================================
# CELL 23: EXPORT RESEARCH RESULTS
# ============================================================

output_excel = (

    "/kaggle/working/"
    "CWOA_GPU_Research_Results.xlsx"
)


with pd.ExcelWriter(

    output_excel,

    engine="openpyxl"

) as writer:

    benchmark_df.to_excel(

        writer,

        sheet_name=
            "GPU_Benchmark",

        index=False
    )


    lifetime_results_gpu.to_excel(

        writer,

        sheet_name=
            "Single_Topology",

        index=False
    )


    multi_results.to_excel(

        writer,

        sheet_name=
            "Multi_Topology",

        index=False
    )


print(
    "Saved:",
    output_excel
)

# ===== Notebook code cell 23 =====
# ============================================================
# NEXT STEP: MULTI-METRIC STATISTICAL SIGNIFICANCE
# ============================================================

from scipy import stats
import numpy as np
import pandas as pd

# Pivot so each topology has one row
pivot = multi_results.pivot(
    index="topology_seed",
    columns="method",
    values=["FND", "HND", "LND", "throughput"]
)

metrics = [
    "FND",
    "HND",
    "LND",
    "throughput"
]

all_results = []

for metric in metrics:

    improved = pivot[
        (metric, "improved_cwoa")
    ].to_numpy()

    original = pivot[
        (metric, "original_cwoa")
    ].to_numpy()

    random_s = pivot[
        (metric, "random_search")
    ].to_numpy()

    # Friedman test
    f_stat, f_p = stats.friedmanchisquare(
        improved,
        original,
        random_s
    )

    print("\n" + "=" * 70)
    print(metric)
    print("=" * 70)

    print(
        f"Friedman statistic: {f_stat:.6f}"
    )

    print(
        f"Friedman p-value  : {f_p:.10f}"
    )

    pairwise = [
        (
            "Improved vs Original",
            improved,
            original
        ),
        (
            "Improved vs Random",
            improved,
            random_s
        ),
        (
            "Original vs Random",
            original,
            random_s
        )
    ]

    temp = []

    for name, x, y in pairwise:

        w, p = stats.wilcoxon(
            x,
            y,
            alternative="two-sided"
        )

        diff = x - y

        dz = (
            diff.mean()
            /
            diff.std(ddof=1)
        )

        temp.append({
            "Metric": metric,
            "Comparison": name,
            "W": w,
            "Raw_p": p,
            "Effect_dz": dz
        })

    temp_df = pd.DataFrame(temp)

    # Holm correction
    raw_p = temp_df["Raw_p"].to_numpy()

    order = np.argsort(raw_p)

    adjusted = np.empty_like(raw_p)

    m = len(raw_p)

    running_max = 0

    for rank, idx in enumerate(order):

        corrected = (
            (m - rank)
            *
            raw_p[idx]
        )

        corrected = min(
            corrected,
            1.0
        )

        running_max = max(
            running_max,
            corrected
        )

        adjusted[idx] = running_max

    temp_df[
        "Holm_Adjusted_p"
    ] = adjusted

    temp_df[
        "Significant_0.05"
    ] = (
        temp_df[
            "Holm_Adjusted_p"
        ]
        <
        0.05
    )

    display(
        temp_df.round(8)
    )

    all_results.append(
        temp_df
    )

stats_all_metrics = pd.concat(
    all_results,
    ignore_index=True
)

# ===== Notebook code cell 24 =====
# ============================================================
# NEXT CELL: EXTENDED LIFETIME QUALITY METRICS
# ============================================================

extended_metrics = multi_results.copy()

# Stability ratio
extended_metrics["Stability_Ratio"] = (
    extended_metrics["FND"]
    /
    extended_metrics["LND"]
)

# Mortality / death transition period
extended_metrics["Death_Transition_Period"] = (
    extended_metrics["LND"]
    -
    extended_metrics["FND"]
)

# Throughput per lifetime round
extended_metrics["Throughput_per_Round"] = (
    extended_metrics["throughput"]
    /
    extended_metrics["LND"]
)


summary_extended = (
    extended_metrics
    .groupby("method")
    [
        [
            "Stability_Ratio",
            "Death_Transition_Period",
            "Throughput_per_Round"
        ]
    ]
    .agg(["mean", "std"])
)

display(summary_extended)

# ===== Notebook code cell 25 =====
# ============================================================
# EXTENDED METRIC STATISTICAL TESTS
# ============================================================

from scipy import stats
import numpy as np
import pandas as pd

metrics_to_test = [
    "Stability_Ratio",
    "Death_Transition_Period",
    "Throughput_per_Round"
]

pivot_ext = extended_metrics.pivot(
    index="topology_seed",
    columns="method",
    values=metrics_to_test
)

extended_stats_results = []

for metric in metrics_to_test:

    improved = pivot_ext[
        (metric, "improved_cwoa")
    ].to_numpy()

    original = pivot_ext[
        (metric, "original_cwoa")
    ].to_numpy()

    random_s = pivot_ext[
        (metric, "random_search")
    ].to_numpy()

    # Friedman test
    f_stat, f_p = stats.friedmanchisquare(
        improved,
        original,
        random_s
    )

    print("\n" + "=" * 70)
    print(metric)
    print("=" * 70)

    print(
        f"Friedman statistic: {f_stat:.6f}"
    )

    print(
        f"Friedman p-value  : {f_p:.10f}"
    )

    comparisons = [
        (
            "Improved vs Original",
            improved,
            original
        ),
        (
            "Improved vs Random",
            improved,
            random_s
        ),
        (
            "Original vs Random",
            original,
            random_s
        )
    ]

    temp = []

    for name, x, y in comparisons:

        w, p = stats.wilcoxon(
            x,
            y,
            alternative="two-sided"
        )

        diff = x - y

        dz = (
            diff.mean()
            /
            diff.std(ddof=1)
        )

        temp.append({
            "Metric": metric,
            "Comparison": name,
            "W": w,
            "Raw_p": p,
            "Effect_dz": dz
        })

    temp_df = pd.DataFrame(temp)

    # Holm correction
    raw_p = temp_df["Raw_p"].to_numpy()

    order = np.argsort(raw_p)

    adjusted = np.empty_like(raw_p)

    m = len(raw_p)

    running_max = 0.0

    for rank, idx in enumerate(order):

        corrected = (
            (m - rank)
            *
            raw_p[idx]
        )

        corrected = min(
            corrected,
            1.0
        )

        running_max = max(
            running_max,
            corrected
        )

        adjusted[idx] = running_max

    temp_df[
        "Holm_Adjusted_p"
    ] = adjusted

    temp_df[
        "Significant_0.05"
    ] = (
        temp_df[
            "Holm_Adjusted_p"
        ]
        <
        0.05
    )

    display(
        temp_df.round(8)
    )

    extended_stats_results.append(
        temp_df
    )

extended_stats_df = pd.concat(
    extended_stats_results,
    ignore_index=True
)

# ===== Notebook code cell 26 =====
# ============================================================
# ALIVE-NODE AUC FOR EACH COMPLETE LIFETIME RUN
# ============================================================

auc_rows = []

for key, history in multi_histories.items():

    # Key structure:
    # method_topologyseed_optimizerseed

    parts = key.rsplit("_", 2)

    method = parts[0]
    topology_seed = int(parts[1])
    optimizer_seed = int(parts[2])

    rounds = history[
        "round"
    ].to_numpy()

    alive = history[
        "alive_nodes"
    ].to_numpy()

    # Include round zero with all nodes alive
    rounds_auc = np.insert(
        rounds,
        0,
        0
    )

    alive_auc = np.insert(
        alive,
        0,
        NUM_NODES
    )

    auc = np.trapz(
        alive_auc,
        rounds_auc
    )

    auc_rows.append({
        "method": method,
        "topology_seed": topology_seed,
        "optimizer_seed": optimizer_seed,
        "Alive_Node_AUC": auc
    })

auc_df = pd.DataFrame(
    auc_rows
)

display(
    auc_df.head()
)

auc_summary = (
    auc_df
    .groupby("method")[
        "Alive_Node_AUC"
    ]
    .agg(["mean", "std"])
)

print("\nAlive-Node AUC Summary")
display(auc_summary)

# ===== Notebook code cell 27 =====
# ============================================================
# ALIVE-NODE AUC + NORMALIZED AUC
# ============================================================

auc_rows = []

for key, history in multi_histories.items():

    # key format:
    # method_topologyseed_optimizerseed
    parts = key.rsplit("_", 2)

    method = parts[0]
    topology_seed = int(parts[1])
    optimizer_seed = int(parts[2])

    rounds = history["round"].to_numpy()
    alive = history["alive_nodes"].to_numpy()

    # Add round 0 with all nodes alive
    rounds_auc = np.insert(
        rounds,
        0,
        0
    )

    alive_auc = np.insert(
        alive,
        0,
        NUM_NODES
    )

    # Total alive-node-rounds
    auc = np.trapezoid(
        alive_auc,
        rounds_auc
    )

    lnd = rounds_auc[-1]

    # Normalized availability score
    # 1.0 would mean all 200 nodes survived until LND
    normalized_auc = (
        auc
        /
        (NUM_NODES * lnd)
        if lnd > 0
        else np.nan
    )

    auc_rows.append({
        "method": method,
        "topology_seed": topology_seed,
        "optimizer_seed": optimizer_seed,
        "Alive_Node_AUC": auc,
        "Normalized_AUC": normalized_auc
    })


auc_df = pd.DataFrame(auc_rows)

print("=" * 70)
print("ALIVE-NODE AUC SUMMARY")
print("=" * 70)

auc_summary = (
    auc_df
    .groupby("method")
    [
        [
            "Alive_Node_AUC",
            "Normalized_AUC"
        ]
    ]
    .agg(["mean", "std"])
)

display(auc_summary)

# ===== Notebook code cell 28 =====
# ============================================================
# AUC STATISTICAL TESTS
# ============================================================

auc_metrics = [
    "Alive_Node_AUC",
    "Normalized_AUC"
]

auc_stats_all = []

pivot_auc = auc_df.pivot(
    index="topology_seed",
    columns="method",
    values=auc_metrics
)

for metric in auc_metrics:

    improved = pivot_auc[
        (metric, "improved_cwoa")
    ].to_numpy()

    original = pivot_auc[
        (metric, "original_cwoa")
    ].to_numpy()

    random_s = pivot_auc[
        (metric, "random_search")
    ].to_numpy()

    friedman_stat, friedman_p = stats.friedmanchisquare(
        improved,
        original,
        random_s
    )

    print("\n" + "=" * 70)
    print(metric)
    print("=" * 70)

    print(
        f"Friedman statistic: {friedman_stat:.6f}"
    )

    print(
        f"Friedman p-value  : {friedman_p:.10f}"
    )

    comparisons = [
        (
            "Improved vs Original",
            improved,
            original
        ),
        (
            "Improved vs Random",
            improved,
            random_s
        ),
        (
            "Original vs Random",
            original,
            random_s
        )
    ]

    temp = []

    for name, x, y in comparisons:

        w, p = stats.wilcoxon(
            x,
            y,
            alternative="two-sided"
        )

        diff = x - y

        dz = (
            diff.mean()
            /
            diff.std(ddof=1)
        )

        temp.append({
            "Metric": metric,
            "Comparison": name,
            "W": w,
            "Raw_p": p,
            "Effect_dz": dz
        })

    temp_df = pd.DataFrame(temp)

    # Holm correction
    raw_p = temp_df["Raw_p"].to_numpy()

    order = np.argsort(raw_p)

    adjusted = np.empty_like(raw_p)

    m = len(raw_p)
    running_max = 0.0

    for rank, idx in enumerate(order):

        corrected = min(
            (m - rank) * raw_p[idx],
            1.0
        )

        running_max = max(
            running_max,
            corrected
        )

        adjusted[idx] = running_max

    temp_df["Holm_Adjusted_p"] = adjusted

    temp_df["Significant_0.05"] = (
        temp_df["Holm_Adjusted_p"] < 0.05
    )

    display(
        temp_df.round(8)
    )

    auc_stats_all.append(
        temp_df
    )

# ===== Notebook code cell 29 =====
# ============================================================
# PAPER-READY PERFORMANCE SUMMARY
# ============================================================

summary_means = (
    multi_results
    .groupby("method")
    [
        [
            "FND",
            "HND",
            "LND",
            "throughput"
        ]
    ]
    .mean()
)

summary_std = (
    multi_results
    .groupby("method")
    [
        [
            "FND",
            "HND",
            "LND",
            "throughput"
        ]
    ]
    .std()
)


extended_means = (
    extended_metrics
    .groupby("method")
    [
        [
            "Stability_Ratio",
            "Death_Transition_Period",
            "Throughput_per_Round"
        ]
    ]
    .mean()
)


extended_std = (
    extended_metrics
    .groupby("method")
    [
        [
            "Stability_Ratio",
            "Death_Transition_Period",
            "Throughput_per_Round"
        ]
    ]
    .std()
)


auc_means = (
    auc_df
    .groupby("method")
    [
        [
            "Alive_Node_AUC",
            "Normalized_AUC"
        ]
    ]
    .mean()
)


auc_std = (
    auc_df
    .groupby("method")
    [
        [
            "Alive_Node_AUC",
            "Normalized_AUC"
        ]
    ]
    .std()
)


paper_summary = pd.DataFrame(
    index=[
        "Improved CWOA",
        "Original CWOA",
        "Random Search"
    ]
)


mapping = {
    "Improved CWOA":
        "improved_cwoa",

    "Original CWOA":
        "original_cwoa",

    "Random Search":
        "random_search"
}


for label, method in mapping.items():

    paper_summary.loc[
        label,
        "FND"
    ] = (
        f"{summary_means.loc[method, 'FND']:.2f} "
        f"± {summary_std.loc[method, 'FND']:.2f}"
    )

    paper_summary.loc[
        label,
        "HND"
    ] = (
        f"{summary_means.loc[method, 'HND']:.2f} "
        f"± {summary_std.loc[method, 'HND']:.2f}"
    )

    paper_summary.loc[
        label,
        "LND"
    ] = (
        f"{summary_means.loc[method, 'LND']:.2f} "
        f"± {summary_std.loc[method, 'LND']:.2f}"
    )

    paper_summary.loc[
        label,
        "Throughput"
    ] = (
        f"{summary_means.loc[method, 'throughput']:.2f} "
        f"± {summary_std.loc[method, 'throughput']:.2f}"
    )

    paper_summary.loc[
        label,
        "Stability Ratio"
    ] = (
        f"{extended_means.loc[method, 'Stability_Ratio']:.4f} "
        f"± {extended_std.loc[method, 'Stability_Ratio']:.4f}"
    )

    paper_summary.loc[
        label,
        "Death Transition"
    ] = (
        f"{extended_means.loc[method, 'Death_Transition_Period']:.2f} "
        f"± {extended_std.loc[method, 'Death_Transition_Period']:.2f}"
    )

    paper_summary.loc[
        label,
        "Alive-Node AUC"
    ] = (
        f"{auc_means.loc[method, 'Alive_Node_AUC']:.2f} "
        f"± {auc_std.loc[method, 'Alive_Node_AUC']:.2f}"
    )

    paper_summary.loc[
        label,
        "Normalized AUC"
    ] = (
        f"{auc_means.loc[method, 'Normalized_AUC']:.4f} "
        f"± {auc_std.loc[method, 'Normalized_AUC']:.4f}"
    )


display(paper_summary)

# ===== Notebook code cell 30 =====
# ============================================================
# PUBLICATION FIGURE: ALIVE-NODE AUC DISTRIBUTION
# ============================================================

plot_auc = auc_df.copy()

plot_auc["Method"] = (
    plot_auc["method"]
    .replace({
        "improved_cwoa":
            "Improved CWOA",

        "original_cwoa":
            "Original CWOA",

        "random_search":
            "Random Search"
    })
)


fig, ax = plt.subplots(
    figsize=(8, 5.5)
)


methods_order = [
    "Original CWOA",
    "Random Search",
    "Improved CWOA"
]


data = [

    plot_auc.loc[
        plot_auc["Method"] == method,
        "Alive_Node_AUC"
    ].to_numpy()

    for method
    in methods_order
]


ax.boxplot(
    data,
    labels=methods_order,
    showmeans=True
)


ax.set_xlabel(
    "Clustering Method"
)

ax.set_ylabel(
    "Alive-Node AUC (node-rounds)"
)

ax.set_title(
    "Cumulative Network Availability Across 20 Topologies"
)

ax.grid(
    axis="y",
    alpha=0.25
)


plt.tight_layout()


plt.savefig(
    "/kaggle/working/"
    "alive_node_auc_boxplot_600dpi.png",
    dpi=600,
    bbox_inches="tight"
)


plt.savefig(
    "/kaggle/working/"
    "alive_node_auc_boxplot.pdf",
    bbox_inches="tight"
)


plt.savefig(
    "/kaggle/working/"
    "alive_node_auc_boxplot.svg",
    bbox_inches="tight"
)


plt.show()

# ===== Notebook code cell 31 =====
# ============================================================
# REVIEWER EXPERIMENT 1A
# FITNESS-WEIGHT SENSITIVITY CONFIGURATION
# ============================================================

WEIGHT_CONFIGS = {

    "Baseline_35_30_20_15":
        (0.35, 0.30, 0.20, 0.15),

    "Equal_25_25_25_25":
        (0.25, 0.25, 0.25, 0.25),

    "EnergyHeavy_50_20_15_15":
        (0.50, 0.20, 0.15, 0.15),

    "IntraHeavy_25_45_15_15":
        (0.25, 0.45, 0.15, 0.15),

    "SinkHeavy_30_20_35_15":
        (0.30, 0.20, 0.35, 0.15),

    "LoadHeavy_30_20_15_35":
        (0.30, 0.20, 0.15, 0.35),

    # Old manuscript's three-weight concept,
    # mapped to the revised four-term formulation.
    "LegacyMapped_50_25_25_00":
        (0.50, 0.25, 0.25, 0.00)
}


def validate_weight_configuration(weights):

    weights = np.asarray(
        weights,
        dtype=float
    )

    if len(weights) != 4:

        raise ValueError(
            "Exactly four fitness weights are required."
        )

    if np.any(weights < 0):

        raise ValueError(
            "Fitness weights cannot be negative."
        )

    if not np.isclose(
        weights.sum(),
        1.0
    ):

        raise ValueError(
            f"Weights must sum to 1. "
            f"Current sum = {weights.sum():.6f}"
        )

    return True


for name, weights in WEIGHT_CONFIGS.items():

    validate_weight_configuration(
        weights
    )

    print(
        f"{name:30s} -> "
        f"{weights} | "
        f"sum={sum(weights):.2f}"
    )

# ===== Notebook code cell 32 =====
W_ENERGY
W_INTRA
W_SINK
W_LOAD

# ===== Notebook code cell 33 =====
# ============================================================
# REVIEWER EXPERIMENT 1B
# FITNESS-WEIGHT SETTER
# ============================================================

ORIGINAL_FITNESS_WEIGHTS = (
    W_ENERGY,
    W_INTRA,
    W_SINK,
    W_LOAD
)


def set_fitness_weights(
    weights,
    verbose=True
):

    global W_ENERGY
    global W_INTRA
    global W_SINK
    global W_LOAD

    validate_weight_configuration(
        weights
    )

    (
        W_ENERGY,
        W_INTRA,
        W_SINK,
        W_LOAD
    ) = weights

    if verbose:

        print(
            "Fitness weights -> "
            f"Energy={W_ENERGY:.2f}, "
            f"Intra={W_INTRA:.2f}, "
            f"Sink={W_SINK:.2f}, "
            f"Load={W_LOAD:.2f}"
        )


# Test
set_fitness_weights(
    WEIGHT_CONFIGS[
        "Baseline_35_30_20_15"
    ]
)

# ===== Notebook code cell 34 =====
# ============================================================
# REVIEWER EXPERIMENT 1C
# QUICK SENSITIVITY DEBUG TEST
# ============================================================

DEBUG_WEIGHT_CONFIGS = {
    "Baseline_35_30_20_15":
        WEIGHT_CONFIGS[
            "Baseline_35_30_20_15"
        ],

    "Equal_25_25_25_25":
        WEIGHT_CONFIGS[
            "Equal_25_25_25_25"
        ]
}


debug_results = []


for weight_name, weights in (
    DEBUG_WEIGHT_CONFIGS.items()
):

    print(
        "\n" + "=" * 70
    )

    print(
        "TESTING:",
        weight_name
    )

    print(
        "=" * 70
    )

    set_fitness_weights(
        weights
    )


    jobs = []

    for topology_index in range(
        2
    ):

        jobs.append({

            "method":
                "improved_cwoa",

            "topology_seed":
                5000
                +
                topology_index,

            "optimizer_seed":
                9000
                +
                topology_index,

            "max_rounds":
                5000
        })


    batch_results, _ = (
        run_dual_gpu_experiments(
            jobs
        )
    )


    batch_results[
        "weight_config"
    ] = weight_name


    debug_results.append(
        batch_results
    )


debug_weight_df = pd.concat(
    debug_results,
    ignore_index=True
)


display(
    debug_weight_df[
        [
            "weight_config",
            "topology_seed",
            "FND",
            "HND",
            "LND",
            "throughput"
        ]
    ]
)


# Restore baseline
set_fitness_weights(
    ORIGINAL_FITNESS_WEIGHTS
)

# ===== Notebook code cell 35 =====
# ============================================================
# REVIEWER EXPERIMENT 1D
# FULL 20-TOPOLOGY WEIGHT SENSITIVITY ANALYSIS
# ============================================================

weight_sensitivity_results = []

weight_sensitivity_auc = []


overall_start = time.perf_counter()


for config_number, (
    weight_name,
    weights
) in enumerate(
    WEIGHT_CONFIGS.items(),
    start=1
):

    print(
        "\n" + "=" * 80
    )

    print(
        f"WEIGHT CONFIGURATION "
        f"{config_number}/"
        f"{len(WEIGHT_CONFIGS)}"
    )

    print(
        weight_name
    )

    print(
        "=" * 80
    )

    set_fitness_weights(
        weights
    )


    # --------------------------------------------------------
    # SAME 20 TOPOLOGIES AND SAME SEEDS FOR EVERY CONFIGURATION
    # --------------------------------------------------------

    jobs = []


    for topology_index in range(
        20
    ):

        jobs.append({

            "method":
                "improved_cwoa",

            "topology_seed":
                5000
                +
                topology_index,

            "optimizer_seed":
                9000
                +
                topology_index,

            "max_rounds":
                5000
        })


    batch_start = time.perf_counter()


    (
        batch_results,
        batch_histories
    ) = run_dual_gpu_experiments(
        jobs
    )


    batch_time = (
        time.perf_counter()
        -
        batch_start
    )


    print(
        f"Completed in "
        f"{batch_time / 60:.2f} minutes"
    )


    # --------------------------------------------------------
    # ADD CONFIGURATION INFORMATION
    # --------------------------------------------------------

    batch_results[
        "weight_config"
    ] = weight_name

    batch_results[
        "W_energy"
    ] = weights[0]

    batch_results[
        "W_intra"
    ] = weights[1]

    batch_results[
        "W_sink"
    ] = weights[2]

    batch_results[
        "W_load"
    ] = weights[3]


    # --------------------------------------------------------
    # DERIVED LIFETIME METRICS
    # --------------------------------------------------------

    batch_results[
        "Stability_Ratio"
    ] = (

        batch_results[
            "FND"
        ]

        /

        batch_results[
            "LND"
        ]
    )


    batch_results[
        "Death_Transition_Period"
    ] = (

        batch_results[
            "LND"
        ]

        -

        batch_results[
            "FND"
        ]
    )


    batch_results[
        "Throughput_per_Round"
    ] = (

        batch_results[
            "throughput"
        ]

        /

        batch_results[
            "LND"
        ]
    )


    # --------------------------------------------------------
    # AUC FOR THIS CONFIGURATION
    # --------------------------------------------------------

    auc_rows = []


    for key, history in (
        batch_histories.items()
    ):

        parts = key.rsplit(
            "_",
            2
        )

        method = parts[0]

        topology_seed = int(
            parts[1]
        )

        optimizer_seed = int(
            parts[2]
        )


        rounds = history[
            "round"
        ].to_numpy()


        alive = history[
            "alive_nodes"
        ].to_numpy()


        rounds_auc = np.insert(
            rounds,
            0,
            0
        )


        alive_auc = np.insert(
            alive,
            0,
            NUM_NODES
        )


        alive_node_auc = (
            np.trapezoid(
                alive_auc,
                rounds_auc
            )
        )


        lnd = rounds_auc[
            -1
        ]


        normalized_auc = (

            alive_node_auc

            /

            (
                NUM_NODES
                *
                lnd
            )
        )


        auc_rows.append({

            "method":
                method,

            "topology_seed":
                topology_seed,

            "optimizer_seed":
                optimizer_seed,

            "weight_config":
                weight_name,

            "Alive_Node_AUC":
                alive_node_auc,

            "Normalized_AUC":
                normalized_auc
        })


    auc_batch_df = pd.DataFrame(
        auc_rows
    )


    batch_results = (
        batch_results.merge(

            auc_batch_df,

            on=[
                "method",
                "topology_seed",
                "optimizer_seed",
                "weight_config"
            ],

            how="left"
        )
    )


    weight_sensitivity_results.append(
        batch_results
    )


# ============================================================
# RESTORE ORIGINAL WEIGHTS
# ============================================================

set_fitness_weights(
    ORIGINAL_FITNESS_WEIGHTS
)


weight_sensitivity_df = pd.concat(

    weight_sensitivity_results,

    ignore_index=True
)


overall_time = (

    time.perf_counter()

    -

    overall_start
)


print(
    "\n" + "=" * 80
)

print(
    "FULL WEIGHT SENSITIVITY EXPERIMENT COMPLETE"
)

print(
    "=" * 80
)

print(
    "Total simulations:",
    len(
        weight_sensitivity_df
    )
)

print(
    f"Total runtime: "
    f"{overall_time / 60:.2f} minutes"
)

# ===== Notebook code cell 36 =====
# ============================================================
# REVIEWER EXPERIMENT 1E
# WEIGHT SENSITIVITY SUMMARY
# ============================================================

weight_metrics = [

    "FND",
    "HND",
    "LND",
    "throughput",
    "Stability_Ratio",
    "Death_Transition_Period",
    "Throughput_per_Round",
    "Alive_Node_AUC",
    "Normalized_AUC"
]


weight_summary = (

    weight_sensitivity_df

    .groupby(
        "weight_config"
    )[
        weight_metrics
    ]

    .agg(
        [
            "mean",
            "std"
        ]
    )
)


display(
    weight_summary
)

# ===== Notebook code cell 37 =====
# ============================================================
# REVIEWER EXPERIMENT 1F
# RELATIVE CHANGE FROM BASELINE WEIGHTS
# ============================================================

weight_mean_df = (

    weight_sensitivity_df

    .groupby(
        "weight_config"
    )[
        weight_metrics
    ]

    .mean()
)


baseline_name = (
    "Baseline_35_30_20_15"
)


baseline_values = (
    weight_mean_df.loc[
        baseline_name
    ]
)


percent_change_weights = (

    (
        weight_mean_df

        -

        baseline_values
    )

    /

    baseline_values

    *

    100
)


percent_change_weights = (
    percent_change_weights.round(
        3
    )
)


display(
    percent_change_weights
)

# ===== Notebook code cell 38 =====
# ============================================================
# REVIEWER EXPERIMENT 1G
# FRIEDMAN + PAIRED WILCOXON + HOLM
# BASELINE VERSUS ALTERNATIVE WEIGHTS
# ============================================================

from scipy import stats


sensitivity_stat_rows = []


for metric in weight_metrics:

    pivot = (
        weight_sensitivity_df
        .pivot(
            index="topology_seed",
            columns="weight_config",
            values=metric
        )
    )


    # --------------------------------------------------------
    # FRIEDMAN TEST ACROSS ALL WEIGHT CONFIGURATIONS
    # --------------------------------------------------------

    arrays = [

        pivot[
            config
        ].to_numpy()

        for config
        in WEIGHT_CONFIGS.keys()
    ]


    friedman_stat, friedman_p = (
        stats.friedmanchisquare(
            *arrays
        )
    )


    print(
        "\n" + "=" * 80
    )

    print(
        metric
    )

    print(
        "=" * 80
    )

    print(
        f"Friedman statistic: "
        f"{friedman_stat:.6f}"
    )

    print(
        f"Friedman p-value  : "
        f"{friedman_p:.10f}"
    )


    baseline = pivot[
        baseline_name
    ].to_numpy()


    temp_rows = []


    for alternative in (
        WEIGHT_CONFIGS.keys()
    ):

        if alternative == baseline_name:
            continue


        comparison = pivot[
            alternative
        ].to_numpy()


        w_stat, raw_p = (
            stats.wilcoxon(

                baseline,

                comparison,

                alternative="two-sided"
            )
        )


        # Difference is baseline - alternative
        paired_difference = (

            baseline

            -

            comparison
        )


        paired_sd = (
            paired_difference.std(
                ddof=1
            )
        )


        if paired_sd > 0:

            effect_dz = (

                paired_difference.mean()

                /

                paired_sd
            )

        else:

            effect_dz = np.nan


        temp_rows.append({

            "Metric":
                metric,

            "Comparison":
                (
                    "Baseline vs "
                    +
                    alternative
                ),

            "Friedman_p":
                friedman_p,

            "W":
                w_stat,

            "Raw_p":
                raw_p,

            "Effect_dz":
                effect_dz
        })


    temp_df = pd.DataFrame(
        temp_rows
    )


    # --------------------------------------------------------
    # HOLM CORRECTION
    # --------------------------------------------------------

    raw_p_values = (
        temp_df[
            "Raw_p"
        ].to_numpy()
    )


    order = np.argsort(
        raw_p_values
    )


    adjusted = np.empty_like(
        raw_p_values
    )


    m = len(
        raw_p_values
    )


    running_max = 0.0


    for rank, idx in enumerate(
        order
    ):

        corrected = min(

            (
                m
                -
                rank
            )

            *

            raw_p_values[
                idx
            ],

            1.0
        )


        running_max = max(

            running_max,

            corrected
        )


        adjusted[
            idx
        ] = running_max


    temp_df[
        "Holm_Adjusted_p"
    ] = adjusted


    temp_df[
        "Significant_0.05"
    ] = (

        temp_df[
            "Holm_Adjusted_p"
        ]

        <

        0.05
    )


    display(
        temp_df.round(
            8
        )
    )


    sensitivity_stat_rows.append(
        temp_df
    )


weight_sensitivity_stats = pd.concat(

    sensitivity_stat_rows,

    ignore_index=True
)

# ===== Notebook code cell 39 =====
# ============================================================
# EXACT FRIEDMAN P-VALUES FOR REPORTING
# ============================================================

exact_friedman_rows = []

for metric in weight_metrics:

    pivot = (
        weight_sensitivity_df
        .pivot(
            index="topology_seed",
            columns="weight_config",
            values=metric
        )
    )

    arrays = [
        pivot[c].to_numpy()
        for c in WEIGHT_CONFIGS.keys()
    ]

    stat, p = stats.friedmanchisquare(
        *arrays
    )

    exact_friedman_rows.append({
        "Metric": metric,
        "Friedman_statistic": stat,
        "Friedman_p": p,
        "Friedman_p_scientific": f"{p:.3e}"
    })

exact_friedman_df = pd.DataFrame(
    exact_friedman_rows
)

display(exact_friedman_df)

# ===== Notebook code cell 40 =====
# ============================================================
# SAVE WEIGHT-SENSITIVITY EVIDENCE
# ============================================================

weight_sensitivity_df.to_csv(
    "/kaggle/working/weight_sensitivity_raw_140_runs.csv",
    index=False
)

weight_sensitivity_stats.to_csv(
    "/kaggle/working/weight_sensitivity_statistics.csv",
    index=False
)

with pd.ExcelWriter(
    "/kaggle/working/weight_sensitivity_reviewers.xlsx"
) as writer:

    weight_sensitivity_df.to_excel(
        writer,
        sheet_name="Raw_140_Runs",
        index=False
    )

    weight_summary.to_excel(
        writer,
        sheet_name="Mean_SD"
    )

    percent_change_weights.to_excel(
        writer,
        sheet_name="Percent_Change"
    )

    weight_sensitivity_stats.to_excel(
        writer,
        sheet_name="Statistics",
        index=False
    )

print(
    "Weight-sensitivity evidence saved."
)

# ===== Notebook code cell 41 =====
# ============================================================
# REVIEWER EXPERIMENT 2A
# FLEXIBLE DUAL-GPU WORKER FOR CH-COUNT ANALYSIS
# ============================================================

from concurrent.futures import ThreadPoolExecutor


def ch_count_gpu_worker(
    device,
    jobs
):

    results = []
    histories = {}

    for job in jobs:

        method = job["method"]
        topology_seed = job["topology_seed"]
        optimizer_seed = job["optimizer_seed"]
        num_ch = job["num_ch"]

        summary, history = simulate_lifetime_gpu(

            topology_seed=topology_seed,

            method=method,

            optimizer_seed=optimizer_seed,

            device=device,

            max_rounds=job.get(
                "max_rounds",
                5000
            ),

            num_ch=num_ch,

            reselection_interval=
                CH_RESELECTION_INTERVAL,

            evaluation_budget=
                LIFETIME_OPTIMIZER_BUDGET
        )

        summary["num_ch"] = num_ch

        key = (
            f"{method}_"
            f"{num_ch}_"
            f"{topology_seed}_"
            f"{optimizer_seed}"
        )

        results.append(summary)

        histories[key] = history

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, histories


def run_ch_count_dual_gpu(
    jobs
):

    device_count = len(DEVICES)

    job_groups = [
        []
        for _ in range(device_count)
    ]

    for index, job in enumerate(jobs):

        job_groups[
            index % device_count
        ].append(job)

    all_results = []
    all_histories = {}

    with ThreadPoolExecutor(
        max_workers=device_count
    ) as executor:

        futures = []

        for device_index in range(
            device_count
        ):

            futures.append(
                executor.submit(
                    ch_count_gpu_worker,
                    DEVICES[device_index],
                    job_groups[device_index]
                )
            )

        for future in futures:

            results, histories = (
                future.result()
            )

            all_results.extend(
                results
            )

            all_histories.update(
                histories
            )

    return (
        pd.DataFrame(all_results),
        all_histories
    )


print(
    "Flexible CH-count runner loaded."
)

# ===== Notebook code cell 42 =====
# ============================================================
# REVIEWER EXPERIMENT 2B
# CH-COUNT DEBUG
# ============================================================

set_fitness_weights(
    WEIGHT_CONFIGS[
        "Baseline_35_30_20_15"
    ]
)

debug_ch_jobs = [

    {
        "method":
            "improved_cwoa",

        "topology_seed":
            5000,

        "optimizer_seed":
            9000,

        "num_ch":
            5,

        "max_rounds":
            5000
    },

    {
        "method":
            "improved_cwoa",

        "topology_seed":
            5000,

        "optimizer_seed":
            9000,

        "num_ch":
            15,

        "max_rounds":
            5000
    }
]


debug_ch_results, _ = (
    run_ch_count_dual_gpu(
        debug_ch_jobs
    )
)


display(
    debug_ch_results[
        [
            "num_ch",
            "FND",
            "HND",
            "LND",
            "throughput"
        ]
    ]
)

# ===== Notebook code cell 43 =====
display(
    debug_ch_results[
        [
            "num_ch",
            "FND",
            "HND",
            "LND",
            "throughput"
        ]
    ]
)

# ===== Notebook code cell 44 =====
# ============================================================
# REVIEWER EXPERIMENT 2C
# FULL CH-COUNT SENSITIVITY
# 5 CH SETTINGS × 20 TOPOLOGIES = 100 RUNS
# ============================================================

import time
import gc
import numpy as np
import pandas as pd
import torch


CH_COUNT_CONFIGS = [
    5,
    8,
    10,
    12,
    15
]


# Make absolutely sure manuscript weights are active
set_fitness_weights(
    WEIGHT_CONFIGS[
        "Baseline_35_30_20_15"
    ]
)


ch_count_all_results = []

checkpoint_path = (
    "/kaggle/working/"
    "AD_CWOA_CH_count_sensitivity_partial.csv"
)


overall_start = time.perf_counter()


for count_index, num_ch in enumerate(
    CH_COUNT_CONFIGS,
    start=1
):

    print(
        "\n" + "=" * 80
    )

    print(
        f"CH CONFIGURATION "
        f"{count_index}/"
        f"{len(CH_COUNT_CONFIGS)}"
    )

    print(
        f"Number of Cluster Heads = {num_ch}"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # CREATE SAME 20 PAIRED TOPOLOGIES
    # --------------------------------------------------------

    jobs = []


    for topology_index in range(
        20
    ):

        jobs.append({

            "method":
                "improved_cwoa",

            "topology_seed":
                5000
                +
                topology_index,

            "optimizer_seed":
                9000
                +
                topology_index,

            "num_ch":
                num_ch,

            "max_rounds":
                5000
        })


    batch_start = (
        time.perf_counter()
    )


    (
        batch_results,
        batch_histories
    ) = run_ch_count_dual_gpu(
        jobs
    )


    batch_runtime = (
        time.perf_counter()
        -
        batch_start
    )


    print(
        f"Completed in "
        f"{batch_runtime / 60:.2f} minutes"
    )


    # ========================================================
    # DERIVED METRICS
    # ========================================================

    batch_results[
        "Stability_Ratio"
    ] = (
        batch_results["FND"]
        /
        batch_results["LND"]
    )


    batch_results[
        "Death_Transition_Period"
    ] = (
        batch_results["LND"]
        -
        batch_results["FND"]
    )


    batch_results[
        "Throughput_per_Round"
    ] = (
        batch_results["throughput"]
        /
        batch_results["LND"]
    )


    # ========================================================
    # ALIVE-NODE AUC
    # ========================================================

    auc_rows = []


    for key, history in (
        batch_histories.items()
    ):

        # key format:
        # improved_cwoa_10_5000_9000

        parts = key.rsplit(
            "_",
            3
        )

        method = parts[0]

        key_num_ch = int(
            parts[1]
        )

        topology_seed = int(
            parts[2]
        )

        optimizer_seed = int(
            parts[3]
        )


        rounds = history[
            "round"
        ].to_numpy()


        alive = history[
            "alive_nodes"
        ].to_numpy()


        # Add round 0 state
        rounds_auc = np.insert(
            rounds,
            0,
            0
        )


        alive_auc = np.insert(
            alive,
            0,
            NUM_NODES
        )


        alive_node_auc = (
            np.trapezoid(
                alive_auc,
                rounds_auc
            )
        )


        lnd = rounds_auc[-1]


        normalized_auc = (
            alive_node_auc
            /
            (
                NUM_NODES
                *
                lnd
            )
            if lnd > 0
            else np.nan
        )


        auc_rows.append({

            "method":
                method,

            "num_ch":
                key_num_ch,

            "topology_seed":
                topology_seed,

            "optimizer_seed":
                optimizer_seed,

            "Alive_Node_AUC":
                alive_node_auc,

            "Normalized_AUC":
                normalized_auc
        })


    auc_df = pd.DataFrame(
        auc_rows
    )


    batch_results = (
        batch_results.merge(

            auc_df,

            on=[
                "method",
                "num_ch",
                "topology_seed",
                "optimizer_seed"
            ],

            how="left"
        )
    )


    ch_count_all_results.append(
        batch_results
    )


    current_df = pd.concat(
        ch_count_all_results,
        ignore_index=True
    )


    current_df.to_csv(
        checkpoint_path,
        index=False
    )


    print(
        "Completed simulations:",
        len(current_df),
        "/",
        len(CH_COUNT_CONFIGS) * 20
    )


    print(
        "Checkpoint saved:",
        checkpoint_path
    )


    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# FINAL DATAFRAME
# ============================================================

ch_count_sensitivity_df = pd.concat(
    ch_count_all_results,
    ignore_index=True
)


runtime = (
    time.perf_counter()
    -
    overall_start
)


print(
    "\n" + "=" * 80
)

print(
    "CH-COUNT SENSITIVITY EXPERIMENT COMPLETE"
)

print(
    "=" * 80
)

print(
    "Total simulations:",
    len(
        ch_count_sensitivity_df
    )
)

print(
    "Expected simulations:",
    len(CH_COUNT_CONFIGS) * 20
)

print(
    f"Total runtime: "
    f"{runtime / 60:.2f} minutes"
)

# ===== Notebook code cell 45 =====
# ============================================================
# REVIEWER EXPERIMENT 2D
# CH-COUNT SUMMARY: MEAN ± SD
# ============================================================

ch_metrics = [

    "FND",
    "HND",
    "LND",
    "throughput",

    "Stability_Ratio",

    "Death_Transition_Period",

    "Throughput_per_Round",

    "Alive_Node_AUC",

    "Normalized_AUC"
]


ch_count_summary = (

    ch_count_sensitivity_df

    .groupby(
        "num_ch"
    )[
        ch_metrics
    ]

    .agg(
        [
            "mean",
            "std"
        ]
    )
)


display(
    ch_count_summary
)

# ===== Notebook code cell 46 =====
# ============================================================
# REVIEWER EXPERIMENT 2E
# PERCENT CHANGE RELATIVE TO K = 10
# ============================================================

ch_mean_df = (

    ch_count_sensitivity_df

    .groupby(
        "num_ch"
    )[
        ch_metrics
    ]

    .mean()
)


baseline_ch = 10


baseline_ch_values = (
    ch_mean_df.loc[
        baseline_ch
    ]
)


ch_percent_change = (

    (
        ch_mean_df
        -
        baseline_ch_values
    )

    /
    baseline_ch_values

    *

    100
)


ch_percent_change = (
    ch_percent_change.round(
        3
    )
)


display(
    ch_percent_change
)

# ===== Notebook code cell 47 =====
# ============================================================
# REVIEWER EXPERIMENT 2F
# FRIEDMAN TEST ACROSS CH COUNTS
# ============================================================

from scipy import stats


ch_friedman_rows = []


for metric in ch_metrics:

    pivot = (
        ch_count_sensitivity_df
        .pivot(
            index="topology_seed",
            columns="num_ch",
            values=metric
        )
    )


    arrays = [

        pivot[
            k
        ].to_numpy()

        for k in CH_COUNT_CONFIGS
    ]


    stat, p = (
        stats.friedmanchisquare(
            *arrays
        )
    )


    ch_friedman_rows.append({

        "Metric":
            metric,

        "Friedman_statistic":
            stat,

        "Friedman_p":
            p,

        "Friedman_p_scientific":
            f"{p:.3e}"
    })


ch_friedman_df = pd.DataFrame(
    ch_friedman_rows
)


display(
    ch_friedman_df
)

# ===== Notebook code cell 48 =====
# ============================================================
# REVIEWER EXPERIMENT 2G
# K=10 VS ALTERNATIVE CH COUNTS
# WILCOXON + HOLM CORRECTION
# ============================================================

ch_pairwise_rows = []


for metric in ch_metrics:

    pivot = (
        ch_count_sensitivity_df
        .pivot(
            index="topology_seed",
            columns="num_ch",
            values=metric
        )
    )


    baseline = (
        pivot[
            10
        ].to_numpy()
    )


    metric_rows = []


    for alternative_k in [
        5,
        8,
        12,
        15
    ]:

        alternative = (
            pivot[
                alternative_k
            ].to_numpy()
        )


        w_stat, raw_p = (
            stats.wilcoxon(
                baseline,
                alternative,
                alternative="two-sided"
            )
        )


        difference = (
            baseline
            -
            alternative
        )


        sd_difference = (
            difference.std(
                ddof=1
            )
        )


        effect_dz = (

            difference.mean()
            /
            sd_difference

            if sd_difference > 0

            else np.nan
        )


        metric_rows.append({

            "Metric":
                metric,

            "Comparison":
                f"K=10 vs K={alternative_k}",

            "W":
                w_stat,

            "Raw_p":
                raw_p,

            "Effect_dz":
                effect_dz
        })


    metric_df = pd.DataFrame(
        metric_rows
    )


    # ========================================================
    # HOLM CORRECTION
    # ========================================================

    raw_p_values = (
        metric_df[
            "Raw_p"
        ].to_numpy()
    )


    order = np.argsort(
        raw_p_values
    )


    adjusted = np.empty_like(
        raw_p_values
    )


    m = len(
        raw_p_values
    )


    running_max = 0.0


    for rank, idx in enumerate(
        order
    ):

        corrected = min(

            (
                m
                -
                rank
            )
            *
            raw_p_values[
                idx
            ],

            1.0
        )


        running_max = max(
            running_max,
            corrected
        )


        adjusted[
            idx
        ] = running_max


    metric_df[
        "Holm_Adjusted_p"
    ] = adjusted


    metric_df[
        "Significant_0.05"
    ] = (
        metric_df[
            "Holm_Adjusted_p"
        ]
        <
        0.05
    )


    ch_pairwise_rows.append(
        metric_df
    )


ch_count_pairwise_stats = (
    pd.concat(
        ch_pairwise_rows,
        ignore_index=True
    )
)


display(
    ch_count_pairwise_stats
)

# ===== Notebook code cell 49 =====
# ============================================================
# SAVE CH-COUNT REVIEWER EVIDENCE
# ============================================================

ch_count_sensitivity_df.to_csv(

    "/kaggle/working/"
    "CH_count_sensitivity_raw_100_runs.csv",

    index=False
)


ch_count_pairwise_stats.to_csv(

    "/kaggle/working/"
    "CH_count_sensitivity_statistics.csv",

    index=False
)


with pd.ExcelWriter(

    "/kaggle/working/"
    "CH_count_sensitivity_reviewers.xlsx"

) as writer:

    ch_count_sensitivity_df.to_excel(
        writer,
        sheet_name="Raw_100_Runs",
        index=False
    )

    ch_count_summary.to_excel(
        writer,
        sheet_name="Mean_SD"
    )

    ch_percent_change.to_excel(
        writer,
        sheet_name="Percent_Change"
    )

    ch_friedman_df.to_excel(
        writer,
        sheet_name="Friedman",
        index=False
    )

    ch_count_pairwise_stats.to_excel(
        writer,
        sheet_name="Pairwise_Holm",
        index=False
    )


print(
    "CH-count sensitivity evidence saved."
)

# ===== Notebook code cell 50 =====
print(
    ch_count_sensitivity_df.columns.tolist()
)

# ===== Notebook code cell 51 =====


# ===== Notebook code cell 52 =====
import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. Create a dedicated folder for your assets
output_dir = "/kaggle/working/my_report_assets"
os.makedirs(output_dir, exist_ok=True)

# 2. Save your tables (CSVs) into that folder
df = pd.DataFrame({"Metric": ["Accuracy", "Loss"], "Value": [0.95, 0.05]})
df.to_csv(f"{output_dir}/results_table.csv", index=False)

# 3. Save your graphs (PNG/PDF) into that folder
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Performance Over Time")
plt.savefig(f"{output_dir}/performance_chart.png", dpi=300)
plt.close()

# ===== Notebook code cell 53 =====
# ============================================================
# CHECK FINAL ENERGY BY CH COUNT
# ============================================================

energy_check = (

    ch_count_sensitivity_df

    .groupby("num_ch")[
        [
            "final_energy",
            "ch_transmissions",
            "optimizer_evaluations"
        ]
    ]

    .agg(["mean", "std"])
)

display(energy_check)

# ===== Notebook code cell 54 =====
# ============================================================
# CH-COUNT ENERGY-EFFICIENCY METRICS
# ============================================================

INITIAL_TOTAL_ENERGY = (
    NUM_NODES
    *
    INITIAL_ENERGY
)


ch_count_sensitivity_df[
    "Energy_Consumed_J"
] = (

    INITIAL_TOTAL_ENERGY

    -

    ch_count_sensitivity_df[
        "final_energy"
    ]
)


# ------------------------------------------------------------
# Average energy expenditure per operational round
# ------------------------------------------------------------

ch_count_sensitivity_df[
    "Energy_per_Round_J"
] = (

    ch_count_sensitivity_df[
        "Energy_Consumed_J"
    ]

    /

    ch_count_sensitivity_df[
        "LND"
    ]
)


# ------------------------------------------------------------
# Successfully represented source packets per Joule
# ------------------------------------------------------------

ch_count_sensitivity_df[
    "Packets_per_Joule"
] = (

    ch_count_sensitivity_df[
        "throughput"
    ]

    /

    ch_count_sensitivity_df[
        "Energy_Consumed_J"
    ]
)


# ------------------------------------------------------------
# Energy required for 1000 successfully delivered
# source-packet equivalents
# ------------------------------------------------------------

ch_count_sensitivity_df[
    "Joules_per_1000_Packets"
] = (

    ch_count_sensitivity_df[
        "Energy_Consumed_J"
    ]

    /

    (
        ch_count_sensitivity_df[
            "throughput"
        ]

        /

        1000.0
    )
)


# ------------------------------------------------------------
# Physical CH-to-BS transmissions per round
# ------------------------------------------------------------

ch_count_sensitivity_df[
    "CH_Transmissions_per_Round"
] = (

    ch_count_sensitivity_df[
        "ch_transmissions"
    ]

    /

    ch_count_sensitivity_df[
        "LND"
    ]
)

# ===== Notebook code cell 55 =====
energy_metrics = [

    "Energy_Consumed_J",

    "Energy_per_Round_J",

    "Packets_per_Joule",

    "Joules_per_1000_Packets",

    "ch_transmissions",

    "CH_Transmissions_per_Round"
]


ch_energy_summary = (

    ch_count_sensitivity_df

    .groupby(
        "num_ch"
    )[
        energy_metrics
    ]

    .agg([
        "mean",
        "std"
    ])
)


display(
    ch_energy_summary
)

# ===== Notebook code cell 56 =====
display(
    ch_count_summary
)

# ===== Notebook code cell 57 =====
display(
    ch_percent_change
)

# ===== Notebook code cell 58 =====
display(ch_count_summary)

# ===== Notebook code cell 59 =====
# ============================================================
# REVIEWER EXPERIMENT 2I
# EXTEND CH-COUNT RANGE ABOVE K=15
# ============================================================

EXTENDED_CH_COUNTS = [
    18,
    20,
    25,
    30
]

set_fitness_weights(
    WEIGHT_CONFIGS[
        "Baseline_35_30_20_15"
    ]
)

extended_ch_results = []

extended_start = time.perf_counter()


for count_index, num_ch in enumerate(
    EXTENDED_CH_COUNTS,
    start=1
):

    print("\n" + "=" * 80)

    print(
        f"EXTENDED CH CONFIGURATION "
        f"{count_index}/{len(EXTENDED_CH_COUNTS)}"
    )

    print(
        f"Number of Cluster Heads = {num_ch}"
    )

    print("=" * 80)


    jobs = []

    for topology_index in range(20):

        jobs.append({

            "method":
                "improved_cwoa",

            "topology_seed":
                5000 + topology_index,

            "optimizer_seed":
                9000 + topology_index,

            "num_ch":
                num_ch,

            "max_rounds":
                5000
        })


    batch_results, batch_histories = (
        run_ch_count_dual_gpu(
            jobs
        )
    )


    # --------------------------------------------------------
    # DERIVED METRICS
    # --------------------------------------------------------

    batch_results[
        "Stability_Ratio"
    ] = (
        batch_results["FND"]
        /
        batch_results["LND"]
    )

    batch_results[
        "Death_Transition_Period"
    ] = (
        batch_results["LND"]
        -
        batch_results["FND"]
    )

    batch_results[
        "Throughput_per_Round"
    ] = (
        batch_results["throughput"]
        /
        batch_results["LND"]
    )


    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc_rows = []

    for key, history in batch_histories.items():

        parts = key.rsplit("_", 3)

        method = parts[0]
        key_num_ch = int(parts[1])
        topology_seed = int(parts[2])
        optimizer_seed = int(parts[3])

        rounds = history[
            "round"
        ].to_numpy()

        alive = history[
            "alive_nodes"
        ].to_numpy()

        rounds_auc = np.insert(
            rounds,
            0,
            0
        )

        alive_auc = np.insert(
            alive,
            0,
            NUM_NODES
        )

        auc = np.trapezoid(
            alive_auc,
            rounds_auc
        )

        lnd = rounds_auc[-1]

        normalized_auc = (
            auc /
            (
                NUM_NODES
                *
                lnd
            )
        )

        auc_rows.append({

            "method":
                method,

            "num_ch":
                key_num_ch,

            "topology_seed":
                topology_seed,

            "optimizer_seed":
                optimizer_seed,

            "Alive_Node_AUC":
                auc,

            "Normalized_AUC":
                normalized_auc
        })


    auc_df = pd.DataFrame(
        auc_rows
    )


    batch_results = (
        batch_results.merge(

            auc_df,

            on=[
                "method",
                "num_ch",
                "topology_seed",
                "optimizer_seed"
            ],

            how="left"
        )
    )


    extended_ch_results.append(
        batch_results
    )


extended_ch_df = pd.concat(
    extended_ch_results,
    ignore_index=True
)


print(
    "Extended simulations completed:",
    len(extended_ch_df)
)

# ===== Notebook code cell 60 =====
# ============================================================
# COMBINE ORIGINAL + EXTENDED CH SENSITIVITY
# ============================================================

full_ch_sensitivity_df = pd.concat(
    [
        ch_count_sensitivity_df,
        extended_ch_df
    ],
    ignore_index=True
)


print(
    "Total CH sensitivity simulations:",
    len(full_ch_sensitivity_df)
)

# ===== Notebook code cell 61 =====
full_ch_summary = (

    full_ch_sensitivity_df

    .groupby(
        "num_ch"
    )[
        [
            "FND",
            "HND",
            "LND",
            "throughput",
            "Stability_Ratio",
            "Death_Transition_Period",
            "Throughput_per_Round",
            "Alive_Node_AUC",
            "Normalized_AUC"
        ]
    ]

    .agg([
        "mean",
        "std"
    ])
)


display(
    full_ch_summary
)

# ===== Notebook code cell 62 =====
sample_history = next(
    iter(
        batch_histories.values()
    )
)

print(
    sample_history.columns.tolist()
)

# ===== Notebook code cell 63 =====
display(ch_percent_change)

# ===== Notebook code cell 64 =====
# ============================================================
# PAPER-READY PARETO FRONT
# CH COMMUNICATION OVERHEAD VS NETWORK AVAILABILITY
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. ENSURE REQUIRED DERIVED METRIC EXISTS
# ------------------------------------------------------------

if "CH_Transmissions_per_Round" not in full_ch_sensitivity_df.columns:

    full_ch_sensitivity_df[
        "CH_Transmissions_per_Round"
    ] = (
        full_ch_sensitivity_df[
            "ch_transmissions"
        ]
        /
        full_ch_sensitivity_df[
            "LND"
        ]
    )


# ------------------------------------------------------------
# 2. CALCULATE MEAN VALUES FOR EACH CH COUNT
# ------------------------------------------------------------

pareto_df = (
    full_ch_sensitivity_df
    .groupby("num_ch")
    .agg(
        CH_Transmissions_per_Round=(
            "CH_Transmissions_per_Round",
            "mean"
        ),
        Alive_Node_AUC=(
            "Alive_Node_AUC",
            "mean"
        ),
        FND=(
            "FND",
            "mean"
        ),
        Throughput=(
            "throughput",
            "mean"
        )
    )
    .reset_index()
)


display(pareto_df)

# ===== Notebook code cell 65 =====
# ============================================================
# 3. IDENTIFY PARETO-OPTIMAL CH COUNTS
#
# Objective 1: MINIMIZE CH transmissions per round
# Objective 2: MAXIMIZE Alive-Node AUC
# ============================================================

def find_pareto_front(
    dataframe,
    minimize_col,
    maximize_col
):

    df = dataframe.copy()

    pareto_flags = []

    for i, row_i in df.iterrows():

        dominated = False

        for j, row_j in df.iterrows():

            if i == j:
                continue

            # j dominates i if:
            # - communication cost is no greater
            # - availability is no lower
            # - and at least one objective is strictly better

            no_worse = (
                row_j[minimize_col]
                <=
                row_i[minimize_col]
                and
                row_j[maximize_col]
                >=
                row_i[maximize_col]
            )

            strictly_better = (
                row_j[minimize_col]
                <
                row_i[minimize_col]
                or
                row_j[maximize_col]
                >
                row_i[maximize_col]
            )

            if no_worse and strictly_better:

                dominated = True
                break

        pareto_flags.append(
            not dominated
        )

    df["Pareto_Optimal"] = pareto_flags

    return df


pareto_df = find_pareto_front(
    pareto_df,
    minimize_col="CH_Transmissions_per_Round",
    maximize_col="Alive_Node_AUC"
)


display(pareto_df)

# ===== Notebook code cell 66 =====
# ============================================================
# 4. PUBLICATION-READY PARETO GRAPH
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


# ------------------------------------------------------------
# ALL CH CONFIGURATIONS
# ------------------------------------------------------------

ax.scatter(
    pareto_df[
        "CH_Transmissions_per_Round"
    ],
    pareto_df[
        "Alive_Node_AUC"
    ],
    s=85
)


# ------------------------------------------------------------
# LABEL EACH POINT WITH K VALUE
# ------------------------------------------------------------

for _, row in pareto_df.iterrows():

    ax.annotate(
        f"K={int(row['num_ch'])}",
        (
            row[
                "CH_Transmissions_per_Round"
            ],
            row[
                "Alive_Node_AUC"
            ]
        ),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10
    )


# ------------------------------------------------------------
# PARETO FRONT
# ------------------------------------------------------------

pareto_points = (
    pareto_df[
        pareto_df[
            "Pareto_Optimal"
        ]
    ]
    .sort_values(
        "CH_Transmissions_per_Round"
    )
)


ax.plot(
    pareto_points[
        "CH_Transmissions_per_Round"
    ],
    pareto_points[
        "Alive_Node_AUC"
    ],
    linewidth=2,
    marker="o",
    label="Pareto frontier"
)


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

ax.set_xlabel(
    "CH-to-BS transmissions per round",
    fontsize=12
)

ax.set_ylabel(
    "Alive-Node AUC (node-rounds)",
    fontsize=12
)

ax.set_title(
    "Pareto Trade-off Between Communication Overhead "
    "and Network Availability",
    fontsize=13
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

plt.tight_layout()


# ------------------------------------------------------------
# SAVE FOR PAPER
# ------------------------------------------------------------

plt.savefig(
    "/kaggle/working/"
    "Pareto_CH_Overhead_vs_Alive_AUC.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "/kaggle/working/"
    "Pareto_CH_Overhead_vs_Alive_AUC.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# ===== Notebook code cell 67 =====
pareto_fnd_df = find_pareto_front(
    pareto_df,
    minimize_col="CH_Transmissions_per_Round",
    maximize_col="FND"
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.scatter(
    pareto_fnd_df[
        "CH_Transmissions_per_Round"
    ],
    pareto_fnd_df[
        "FND"
    ],
    s=85
)


for _, row in pareto_fnd_df.iterrows():

    ax.annotate(
        f"K={int(row['num_ch'])}",
        (
            row[
                "CH_Transmissions_per_Round"
            ],
            row[
                "FND"
            ]
        ),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10
    )


front = (
    pareto_fnd_df[
        pareto_fnd_df[
            "Pareto_Optimal"
        ]
    ]
    .sort_values(
        "CH_Transmissions_per_Round"
    )
)


ax.plot(
    front[
        "CH_Transmissions_per_Round"
    ],
    front[
        "FND"
    ],
    marker="o",
    linewidth=2,
    label="Pareto frontier"
)


ax.set_xlabel(
    "CH-to-BS transmissions per round",
    fontsize=12
)

ax.set_ylabel(
    "First Node Death (rounds)",
    fontsize=12
)

ax.set_title(
    "Pareto Trade-off Between Communication Overhead "
    "and Network Stability",
    fontsize=13
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "/kaggle/working/"
    "Pareto_CH_Overhead_vs_FND.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "/kaggle/working/"
    "Pareto_CH_Overhead_vs_FND.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# ===== Notebook code cell 68 =====
# ============================================================
# FINAL RESEARCH OUTPUT PACKAGE
# AD-CWOA WSN MAJOR REVISION
# ============================================================

import os
import shutil
from pathlib import Path
from datetime import datetime

WORKING_DIR = Path("/kaggle/working")

PACKAGE_DIR = WORKING_DIR / "FINAL_AD_CWOA_RESEARCH_PACKAGE"

PACKAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# ORGANIZED SUBFOLDERS
# ------------------------------------------------------------

subfolders = {
    "raw_results": PACKAGE_DIR / "01_Raw_Results",
    "summary_tables": PACKAGE_DIR / "02_Summary_Tables",
    "statistics": PACKAGE_DIR / "03_Statistical_Analysis",
    "figures": PACKAGE_DIR / "04_Figures",
    "reviewer_results": PACKAGE_DIR / "05_Reviewer_Experiments",
    "other": PACKAGE_DIR / "06_Other_Files",
}

for folder in subfolders.values():
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

# ===== Notebook code cell 69 =====
# ============================================================
# COPY ALL RESEARCH OUTPUTS
# ============================================================

for file_path in WORKING_DIR.iterdir():

    if not file_path.is_file():
        continue

    name = file_path.name.lower()

    # Avoid copying the ZIP itself later
    if "final_ad_cwoa_all_outputs" in name:
        continue

    if name.endswith(".csv"):

        if (
            "stat" in name
            or "friedman" in name
            or "wilcoxon" in name
            or "holm" in name
        ):
            destination = subfolders["statistics"]

        elif (
            "sensitivity" in name
            or "reviewer" in name
        ):
            destination = subfolders["reviewer_results"]

        else:
            destination = subfolders["raw_results"]

    elif (
        name.endswith(".xlsx")
        or name.endswith(".xls")
    ):

        destination = subfolders[
            "summary_tables"
        ]

    elif (
        name.endswith(".png")
        or name.endswith(".jpg")
        or name.endswith(".jpeg")
        or name.endswith(".svg")
        or name.endswith(".pdf")
    ):

        destination = subfolders[
            "figures"
        ]

    elif (
        name.endswith(".txt")
        or name.endswith(".md")
    ):

        destination = subfolders[
            "other"
        ]

    else:
        continue

    shutil.copy2(
        file_path,
        destination / file_path.name
    )


print(
    "✅ Research files organized."
)

# ===== Notebook code cell 70 =====
# ============================================================
# SAVE IMPORTANT DATAFRAMES
# ============================================================

dataframes_to_save = {

    "weight_sensitivity_df":
        "Weight_Sensitivity_140_Runs.csv",

    "weight_sensitivity_stats":
        "Weight_Sensitivity_Statistics.csv",

    "ch_count_sensitivity_df":
        "CH_Count_Initial_100_Runs.csv",

    "extended_ch_df":
        "CH_Count_Extended_Runs.csv",

    "full_ch_sensitivity_df":
        "CH_Count_All_Runs.csv",

    "ch_count_pairwise_stats":
        "CH_Count_Pairwise_Statistics.csv",

    "exact_friedman_df":
        "Weight_Friedman_Exact.csv"
}


for variable_name, filename in (
    dataframes_to_save.items()
):

    if variable_name in globals():

        dataframe = globals()[
            variable_name
        ]

        dataframe.to_csv(
            PACKAGE_DIR
            /
            "01_Raw_Results"
            /
            filename,

            index=False
        )

        print(
            "Saved:",
            filename
        )

# ===== Notebook code cell 71 =====
# ============================================================
# SAVE SUMMARY TABLES TO ONE EXCEL WORKBOOK
# ============================================================

import pandas as pd

excel_path = (
    PACKAGE_DIR
    /
    "02_Summary_Tables"
    /
    "FINAL_AD_CWOA_All_Summary_Tables.xlsx"
)


with pd.ExcelWriter(
    excel_path
) as writer:

    if "weight_summary" in globals():

        weight_summary.to_excel(
            writer,
            sheet_name="Weight_Sensitivity"
        )


    if "percent_change_weights" in globals():

        percent_change_weights.to_excel(
            writer,
            sheet_name="Weight_Percent_Change"
        )


    if "ch_count_summary" in globals():

        ch_count_summary.to_excel(
            writer,
            sheet_name="CH_Count_Initial"
        )


    if "full_ch_summary" in globals():

        full_ch_summary.to_excel(
            writer,
            sheet_name="CH_Count_Full"
        )


    if "ch_percent_change" in globals():

        ch_percent_change.to_excel(
            writer,
            sheet_name="CH_Percent_Change"
        )


print(
    "✅ Master Excel workbook saved."
)

# ===== Notebook code cell 72 =====
full_ch_sensitivity_df.to_csv(
    "/kaggle/working/full_ch_sensitivity_all_runs.csv",
    index=False
)

# ===== Notebook code cell 73 =====
# ============================================================
# CREATE README
# ============================================================

readme_text = """
# AD-CWOA WSN Research Reproducibility Package

## Study
Adaptive Discrete Whale Optimization for
IoT-Enabled Wireless Sensor Networks

## Core network configuration

- Sensor nodes: 200
- Network area: 200 x 200 m
- Base station: (100, 100)
- Initial node energy: 1 J
- Packet size: 4000 bits
- Population size: 30
- CH reselection interval: 10 rounds
- Lifetime optimization budget: 300 evaluations
- Independent network topologies: 20

## Radio parameters

- E_elec = 50 nJ/bit
- E_fs = 10 pJ/bit/m^2
- E_mp = 0.0013 pJ/bit/m^4
- E_DA = 5 nJ/bit
- d0 ≈ 87.71 m

## Fitness configuration

Baseline weights:

- Energy = 0.35
- Intra-cluster distance = 0.30
- CH-to-BS distance = 0.20
- Load imbalance = 0.15

## Reviewer experiments

1. Fitness-weight sensitivity
2. CH-count sensitivity
3. Normalized throughput analysis
4. Alive-Node AUC analysis
5. Friedman statistical testing
6. Wilcoxon signed-rank testing
7. Holm multiple-comparison correction
8. Pareto trade-off analysis
9. Algorithm baseline comparison

## Important assumptions

- Member -> CH -> BS communication
- Direct CH-to-BS forwarding
- No inter-CH relay routing
- Optimization/control energy not deducted
  from sensor-node energy
- Sensing energy not explicitly modeled

## Environment

- Python
- PyTorch
- Kaggle
- Dual NVIDIA T4 GPUs
"""

readme_path = (
    PACKAGE_DIR
    /
    "README.md"
)

readme_path.write_text(
    readme_text,
    encoding="utf-8"
)

print(
    "✅ README created."
)

# ===== Notebook code cell 74 =====
# ============================================================
# CREATE FINAL ZIP
# ============================================================

zip_base = (
    "/kaggle/working/"
    "FINAL_AD_CWOA_All_Outputs"
)

shutil.make_archive(
    zip_base,
    "zip",
    PACKAGE_DIR
)

print(
    "\n✅ FINAL PACKAGE CREATED:"
)

print(
    zip_base + ".zip"
)

