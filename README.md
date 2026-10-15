# RL from Scratch

A hands-on series of Jupyter notebooks that builds reinforcement learning up step by step, from estimating a value
function in a 5-state random walk to Deep Q-Networks with large-scale hyperparameter tuning.

Every notebook is self-contained, written in **PyTorch** and **Gymnasium**, and explains the *why* behind each line of
code, not just the *what*. Each algorithm is implemented from scratch. Libraries are used only for environments,
neural-network primitives and tuning infrastructure.

---

## Notebooks

| # | Notebook | Topics | Environment(s) |
|---|----------|--------|----------------|
| 1 | [Value Functions: Monte Carlo vs TD](01_value_functions_mc_vs_td.ipynb) | prediction, Monte Carlo, TD(0), n-step TD, bias vs variance, neural value function, semi-gradient | Random Walk (Sutton & Barto 6.2) |
| 2 | [Q-Learning with Gymnasium](02_q_learning_gymnasium.ipynb) | control, Q-learning vs SARSA, on-policy vs off-policy, ε-greedy, `terminated` vs `truncated` | CliffWalking, FrozenLake |
| 3 | [Deep Q-Learning (DQN)](03_dqn_deep_q_learning.ipynb) | Q-networks, experience replay, target networks, Atari preprocessing, overestimation, Optuna tuning | CartPole, Atari Breakout |
| 4 | [Tuning DQN with Ray Tune](04_dqn_ray_tune.ipynb) | parallel tuning, ASHA, Optuna inside Ray, Population Based Training, validation on unseen seeds | CartPole |
| 5 | [Policy Gradients: REINFORCE to A2C](policy_gradients.ipynb) | policy gradient theorem, REINFORCE, baselines and advantages, entropy bonus, A2C, GAE, on-policy vs off-policy | CartPole, LunarLander (optional) |
| 6 | [PPO, Curiosity (RND) and RLlib](ppo_curiosity_rllib.ipynb) | PPO from scratch, clipped surrogate, PPO diagnostics, RND curiosity, sparse rewards, Ray RLlib (new API stack), RLlib + Tune | CartPole, MountainCar, MiniGrid DoorKey, LunarLander (optional) |
<!-- | 5 | [Next notebook](05_....ipynb) | ... | ... | -->


---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rl-from-scratch.git
cd rl-from-scratch
```

### 2. Create an environment

With conda or mamba:

```bash
conda create -n rl python=3.11 -y
conda activate rl
```

or with venv:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch gymnasium "gymnasium[atari]" ale-py matplotlib numpy pandas jupyter
pip install optuna "ray[tune]"   # for the tuning parts (notebooks 3 and 4)
```

### 4. Run the notebooks

```bash
jupyter lab
```



---

## Acknowledgements

The notebooks are based on the course taught by Huggingface https://huggingface.co/learn/deep-rl-course/en/

## License

<!-- Choose a license, e.g. MIT, and add a LICENSE file. -->
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
