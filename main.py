import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network_generation import gen_single_network
from mfq_generation_scale_up import generate_mfq


def simulate_baseline_social_network(p_connection_1: float = 0.36, increase_support_level: float = 0.0) -> pd.DataFrame:
    # Baseline parameters
    num_networks = 1000
    family_min = 2
    family_max = 10
    friend_min = 1
    friend_max = 13
    closest_layer_nodes = 5
    max_nodes = 15
    p_connection_2 = 0.15

    mfq_min = 0
    mfq_max = 66
    num_episodes = 10

    generated_network = [gen_single_network(
        network_id,
        family_min,
        family_max,
        friend_min,
        friend_max,
        closest_layer_nodes,
        max_nodes,
        p_connection_1,
        p_connection_2
    ) for network_id in range(num_networks)]

    generated_mfq = [
        generate_mfq(
            network_id,
            mfq_min,
            mfq_max,
            network.family_support,
            network.friend_support,
            num_episodes,
            increase_support_level,
        ) for network_id, network in enumerate(generated_network)
    ]

    df_network = pd.DataFrame(generated_network)
    df_mfq = pd.DataFrame(generated_mfq)
    df_mfq_with_network = df_mfq[['initial_mfq', 'mfq_with_network']]
    df_mfq_without_network = df_mfq[['mfq_without_network']]

    df_mfq_with_network = pd.concat((df_network, df_mfq_with_network), axis=1)
    df_mfq_with_network = df_mfq_with_network.explode('mfq_with_network').reset_index(drop=True)
    df_mfq_without_network = df_mfq_without_network.explode('mfq_without_network').reset_index(drop=True)
    df_result = pd.concat((df_mfq_with_network, df_mfq_without_network['mfq_without_network']), axis=1)
    df_result["episode_number"] = np.tile(range(num_episodes+1), len(df_result) // (num_episodes+1))
    return df_result


def simulate_network_varying_p1_scaleup() -> pd.DataFrame:
    result = pd.DataFrame()
    for p_connection_1 in np.arange(0.15, 0.7, 0.1):
        temp_res = simulate_baseline_social_network(p_connection_1)
        temp_res["p_connection_1"] = p_connection_1
        result = pd.concat((result, temp_res), axis=0)
    return result

def simulate_network_varying_increase_support_level() -> pd.DataFrame:
    result = pd.DataFrame()
    for increase_support_level in [0.25, 0.5, 1.0, 3.0, 5.0, 7.0]:
        temp_res = simulate_baseline_social_network(increase_support_level=increase_support_level)
        temp_res["increase_support_level"] = increase_support_level
        result = pd.concat((result, temp_res), axis=0)
    return result


if __name__ == "__main__":
    #simulate_baseline_social_network().to_csv("simulated_data_baseline.csv", index=False)
   # simulate_network_varying_p1_scaleup().to_csv("network_varying_p1.csv", index=False)
   # simulate_network_varying_increase_support_level().to_csv("network_varying_increase_support_level.csv", index=False)
    simulate_baseline_social_network().to_csv("simulated_data_baseline.0.01.csv", index=False)
