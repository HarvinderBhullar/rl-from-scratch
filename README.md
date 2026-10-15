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
<!-- | 5 | [Next notebook](05_....ipynb) | ... | ... | -->

### The learning path

```
 V(s) for a fixed policy  ──►  Q(s,a) and the best policy  ──►  Q-network for large state spaces  ──►  tuning it at scale
 (1) MC vs TD                  (2) Q-learning                   (3) DQN                                 (4) Ray Tune
```

Each notebook reuses ideas from the previous one. For example, the TD update from notebook 1 becomes the Q-learning
update in notebook 2. Its semi-gradient form (no gradient through the target) is the core of the DQN loss in notebook 3.

---

## Highlights

**1. Value Functions: Monte Carlo vs TD**
- Implements MC, TD(0) and n-step TD in one framework, and shows that n = 1 is TD(0) and n = ∞ is Monte Carlo.
- Runs 100 experiments in parallel as one batched tensor operation, so the comparisons are averaged rather than single noisy runs.
- Recreates Sutton & Barto Figure 6.2: TD(0) learns faster than MC on the random walk.

**2. Q-Learning with Gymnasium**
- Q-learning vs SARSA on CliffWalking. Q-learning finds the optimal cliff-edge path (return −13), while SARSA learns a
  safer route (−17) but collects *more* reward during training. This is a clear illustration of off-policy vs on-policy learning.
- Q-learning on slippery FrozenLake reaches the goal **~75%** of the time, which is essentially optimal for that map.

**3. Deep Q-Learning (DQN)**
- Follows the structure of CleanRL's [`dqn_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py),
  updated for the Gymnasium 1.x API.
- Solves CartPole: the greedy policy scores **500/500**.
- Shows DQN's **overestimation bias** live: the Q-values overshoot to about 215, though the maximum possible value is 100.
- Includes the full Atari pipeline (frame skipping, reward clipping, frame stacking, Nature CNN) and hyperparameter
  tuning with Optuna (TPE sampler + median pruning).

**4. Tuning DQN with Ray Tune**
- Three strategies compared: random search + ASHA, Optuna TPE + ASHA, and Population Based Training.
- PBT discovers hyperparameter *schedules* (for example, a learning rate that changes during training) by cloning and mutating agents.
- Every winner is re-validated on unseen seeds, because the best tuning score is always optimistically biased.

<!-- Add your own results here, e.g. a table of validation scores per tuning strategy, or plots in an `assets/` folder:
![Validation results](assets/ray_validation.png)
-->

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

Or, if you keep a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks

```bash
jupyter lab
```

Open the notebooks in order. Each one starts with a short recap of the previous one.

### Hardware and runtimes

| Notebook | Typical runtime (CPU) | Notes |
|----------|----------------------|-------|
| 1 | ~1 min | |
| 2 | ~2 min | |
| 3 | ~3 min (CartPole) + tuning | Full Atari training (10M steps) needs a GPU and many hours; it is off by default (`RUN_FULL_TRAINING = False`) |
| 4 | minutes to an hour | scales with your CPU cores: each trial uses 1 core |

---

## Repository structure

```
rl-from-scratch/
├── README.md
├── requirements.txt
├── 01_value_functions_mc_vs_td.ipynb
├── 02_q_learning_gymnasium.ipynb
├── 03_dqn_deep_q_learning.ipynb
├── 04_dqn_ray_tune.ipynb
└── assets/                  # (optional) figures for this README
```

Generated files are not committed (see `.gitignore`): `ray_results/` (Ray Tune experiments), `*.db` (Optuna
studies), `*.pt` (model checkpoints) and `videos/`.

---

## Roadmap

- [x] Value functions: Monte Carlo vs TD
- [x] Tabular Q-learning and SARSA
- [x] DQN (CartPole + Atari)
- [x] Hyperparameter tuning: Optuna and Ray Tune
- [ ] Double DQN and Dueling DQN
- [ ] Prioritized experience replay
- [ ] Policy gradients: REINFORCE
- [ ] Actor-critic: A2C and PPO
- [ ] Continuous control: DDPG, TD3 and SAC
<!-- - [ ] your next topic -->

---

## References

**Books and courses**
- Sutton & Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), 2nd edition, 2018
- [Spinning Up in Deep RL](https://spinningup.openai.com/) (OpenAI)

**Papers**
- Mnih et al., 2015, *Human-level control through deep reinforcement learning* (DQN), Nature
- van Hasselt, Guez & Silver, 2016, *Deep Reinforcement Learning with Double Q-learning*, AAAI
- Jaderberg et al., 2017, *Population Based Training of Neural Networks*
- Li et al., 2020, *A System for Massively Parallel Hyperparameter Tuning* (ASHA), MLSys

**Libraries**
- [CleanRL](https://github.com/vwxyzjn/cleanrl): single-file RL implementations; notebook 3 follows its DQN
- [Gymnasium](https://gymnasium.farama.org/) and the [Arcade Learning Environment](https://ale.farama.org/)
- [Optuna](https://optuna.org/) and [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)

---

## Acknowledgements

The DQN notebook is based on the structure of CleanRL's `dqn_atari.py` by Shengyi Huang and contributors. The Atari
preprocessing wrappers follow the implementations in CleanRL and Stable-Baselines3.

## License

<!-- Choose a license, e.g. MIT, and add a LICENSE file. -->
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
