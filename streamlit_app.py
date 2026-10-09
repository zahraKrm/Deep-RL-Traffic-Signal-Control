import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib import style

style.use("ggplot")

SHOW_EVERY = 100
FIGURES_DIR = "figures"

RESULTS = {
    "Q-Learning": {
        "rewards": "RL_Test_same_0.9_episode_rewards.npy",
        "waiting": "RL_Test_same_0.9_waiting_time.npy",
        "figures": {
            "reward": "qlearning_reward.png",
            "reward_ma": "qlearning_reward_ma.png",
            "waiting": "qlearning_waiting_time.png",
            "waiting_ma": "qlearning_waiting_time_ma.png",
        },
    },
    "Deep RL": {
        "rewards": "DeepRL_Test_same_0.9_episode_rewards.npy",
        "waiting": "DeepRL_Test_same_0.9_waiting_time.npy",
        "figures": {
            "reward": "deeprl_test_reward.png",
            "reward_ma": "deeprl_test_reward_ma.png",
            "waiting": "deeprl_test_waiting_time.png",
            "waiting_ma": "deeprl_test_waiting_time_ma.png",
        },
    },
    "DQN Training": {
        "rewards": "DeepRL_env_episode_rewards.npy",
        "waiting": None,
        "figures": None,
    },
}


def load_array(path):
    if not os.path.exists(path):
        return None
    return np.load(path)


def moving_average(values, window):
    if len(values) < window:
        return np.array([])
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_series(values, title, ylabel, window=SHOW_EVERY):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(values)
    axes[0].set_title(f"{title} (Raw)")
    axes[0].set_xlabel("Episode #")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True)

    ma = moving_average(values, window)
    if len(ma) > 0:
        axes[1].plot(range(len(ma)), ma)
        axes[1].set_title(f"{title} ({window}-Episode Moving Average)")
        axes[1].set_xlabel("Episode #")
        axes[1].set_ylabel(ylabel)
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, "Not enough episodes for moving average", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    return fig


def figure_path(filename):
    return os.path.join(FIGURES_DIR, filename)


def show_saved_figures(figure_files):
    reward_col, reward_ma_col = st.columns(2)
    reward_path = figure_path(figure_files["reward"])
    reward_ma_path = figure_path(figure_files["reward_ma"])
    if os.path.exists(reward_path):
        reward_col.image(reward_path, caption="Reward (raw)", use_container_width=True)
    if os.path.exists(reward_ma_path):
        reward_ma_col.image(reward_ma_path, caption="Reward (moving average)", use_container_width=True)

    waiting_col, waiting_ma_col = st.columns(2)
    waiting_path = figure_path(figure_files["waiting"])
    waiting_ma_path = figure_path(figure_files["waiting_ma"])
    if os.path.exists(waiting_path):
        waiting_col.image(waiting_path, caption="Waiting time (raw)", use_container_width=True)
    if os.path.exists(waiting_ma_path):
        waiting_ma_col.image(waiting_ma_path, caption="Waiting time (moving average)", use_container_width=True)


def show_method_results(method_name, rewards_path, waiting_path, figure_files=None):
    rewards = load_array(rewards_path)
    waiting = load_array(waiting_path) if waiting_path else None
    has_figures = figure_files and any(
        os.path.exists(figure_path(name)) for name in figure_files.values()
    )

    if rewards is None and waiting is None and not has_figures:
        st.warning(f"No result files found for **{method_name}**. Run the corresponding script first.")
        st.code(rewards_path, language="text")
        return

    st.subheader(method_name)

    if rewards is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Episodes", len(rewards))
        col2.metric("Mean reward", f"{np.mean(rewards):.2f}")
        col3.metric("Best reward", f"{np.max(rewards):.2f}")
        if has_figures:
            show_saved_figures(figure_files)
        else:
            st.pyplot(plot_series(rewards, "Reward", "Reward"))
    else:
        st.info(f"Reward file not found: `{rewards_path}`")

    if waiting_path:
        if waiting is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Episodes", len(waiting))
            col2.metric("Mean waiting time", f"{np.mean(waiting):.2f}")
            col3.metric("Lowest waiting time", f"{np.min(waiting):.2f}")
            if not has_figures:
                st.pyplot(plot_series(waiting, "Waiting Time", "Mean wait per car"))
        else:
            st.info(f"Waiting time file not found: `{waiting_path}`")


st.set_page_config(page_title="Traffic RL Results", layout="wide")
st.title("Traffic Light RL — Results Dashboard")
st.caption(
    "Metrics load from `.npy` files. Charts load from saved PNGs in `figures/` when available "
    "(replace those files to update the dashboard)."
)

selected = st.sidebar.multiselect(
    "Show results for",
    options=list(RESULTS.keys()),
    default=["Q-Learning", "Deep RL"],
)

st.sidebar.markdown("### Expected files")
for method, files in RESULTS.items():
    st.sidebar.markdown(f"**{method}**")
    st.sidebar.markdown(f"- `{files['rewards']}`")
    if files["waiting"]:
        st.sidebar.markdown(f"- `{files['waiting']}`")

for method in selected:
    files = RESULTS[method]
    show_method_results(method, files["rewards"], files["waiting"], files.get("figures"))
    st.divider()
