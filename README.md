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
| 7 | [PPO from Pixels: ViZDoom](ppo_pixels_vizdoom.ipynb) | CNN policies, frame stacking, AsyncVectorEnv, GPU/MPS training, sync vs async PPO, V-trace, RLlib APPO and PPO, reward scaling | ViZDoom Basic, DefendCenter, Health Gathering Supreme |
<!-- | 8 | [Next notebook](08_....ipynb) | ... | ... | -->


---


**7. PPO from Pixels: ViZDoom**
- Our PPO from notebook 6 extended to raw pixels: Nature CNN, 84×84 grayscale frames with frame stacking,
  `AsyncVectorEnv` with correct `SAME_STEP` autoreset handling, and training on CUDA or Apple's MPS.
- Solves ViZDoom **Basic** (return ~70–75) in about **100k steps, around 1 minute** on a MacBook.
- Synchronous vs asynchronous PPO explained: policy lag and the **V-trace** correction, implemented and checked
  (it reduces exactly to the n-step return when there is no lag).
- RLlib on the same task, compared per environment step and per wall-clock minute:

  | | Throughput (MacBook, CPU learner) | Basic |
  |---|---|---|
  | our PPO | ~1,550 steps/s | solved, ~70–75 |
  | RLlib PPO | ~230 steps/s | solved, **~80** after ~60k steps |
  | RLlib APPO | ~1,150 steps/s | not solved within 300k steps |

- Health Gathering Supreme (the HF course's task) with RLlib: APPO on a CUDA GPU, PPO on CPU-only machines.
- Replaces the HF course's Sample Factory with Gymnasium-native ViZDoom and RLlib (Sample Factory flags mapped in the appendix).


## Lessons learned: debugging RL with metrics

Notebook 7 includes a real debugging case. RLlib APPO stayed at random-level performance on ViZDoom Basic while our
own PPO solved it. Three measurements found the cause, step by step:

1. **`trained/sampled` = 0.16.** With 8 env runners and a CPU learner, 84% of the experience was never trained on.
   Fix: fewer env runners, so the learner keeps up.
2. **Still no learning, and the action frequencies were uniform** (entropy ≈ log 4). The observations checked out fine,
   so the problem had to be inside the update.
3. **`vf_loss` ≈ 1500, `policy_loss` ≈ 0.2, gradient norm ≈ 580.** Basic's large returns made the value loss dominate
   the shared CNN, and global gradient clipping shrank the policy's share of the gradient about 1000×.
   Fix: **reward scaling** (×0.01).

A control experiment (RLlib's synchronous PPO with the identical setup) then solved the task. That isolated the
remaining gap to APPO's asynchronous design, which pays off with fast GPU learners, not on a laptop CPU.

**Takeaways**
- **Log the diagnostics, not just the return:** `vf_loss`, gradient norm, entropy, and how much of your data is
  actually trained on.
- **Scale large rewards** when the policy and value function share a network.
- **Use control experiments** to separate "the algorithm" from "my setup".

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
