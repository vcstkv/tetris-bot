import os

from pyvirtualdisplay import Display

import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gymnasium as gym

# Hugging Face Hub
import imageio

import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from dataclasses import dataclass
from typing import Tuple
import time

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

@torch.no_grad()
def obs_to_torch(obs: dict, device: str | torch.device) -> torch.Tensor:
    """
    Supports ONLY env-provided dict observations:
      - single env: board (24,18), mask (24,18)
      - vector env: board (N,24,18), mask (N,24,18)

    Output: torch.float32 (N,2,24,18) on `device`
    """
    board = obs["board"].astype(np.float32)
    mask  = obs["active_tetromino_mask"].astype(np.float32)

    static_occ = ((board > 0) * (1.0 - mask)).astype(np.float32)

    if board.ndim == 2:
        # (24,18) -> (1,2,24,18)
        x = np.stack([static_occ, mask], axis=0)[None, ...]
    elif board.ndim == 3:
        # (N,24,18) -> (N,2,24,18)
        x = np.stack([static_occ, mask], axis=1)
    else:
        raise ValueError(f"Unexpected board shape {board.shape}")

    return torch.from_numpy(x).to(device).contiguous()


def obs_to_torch_fast(obs, x_cpu, x_gpu):
    board = obs["board"]                      # numpy
    mask  = obs["active_tetromino_mask"]      # numpy

    # fill pinned CPU tensor without extra stack allocations
    # channel0: static_occ, channel1: mask
    # (cast once)
    b = (board > 0).astype(np.float32)
    m = mask.astype(np.float32)
    static = b * (1.0 - m)

    if board.ndim == 2:
        x_cpu[0, 0].copy_(torch.from_numpy(static))
        x_cpu[0, 1].copy_(torch.from_numpy(m))
    else:
        # (N,H,W)
        x_cpu[:, 0].copy_(torch.from_numpy(static))
        x_cpu[:, 1].copy_(torch.from_numpy(m))

    x_gpu.copy_(x_cpu, non_blocking=True)
    return x_gpu


class TetrisPolicy(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.GELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 24, 18)
            n_flat = self.cnn(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.GELU(),
        )
        self.pi = nn.Linear(512, n_actions)  # logits
        self.v = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,C,H,W)
        z = self.cnn(x).flatten(1)
        z = self.fc(z)
        logits = self.pi(z)
        value = self.v(z).squeeze(-1)
        return logits, value

    def act(self, obs):
        """
        Uses your obs_to_torch(obs, device) that returns (N,2,24,18) float32.
        Assumes self(x) returns (logits, value) or just logits.
        """
        device = next(self.parameters()).device
        x = obs_to_torch(obs, device)          # (N,2,24,18); for single env should be N=1
    
        out = self(x)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            logits, value = out
        else:
            logits, value = out, None
    
        dist = Categorical(logits=logits)      # logits: (N, A)
        action = dist.sample()                 # (N,)
        logp = dist.log_prob(action)           # (N,)
    
        # If N==1, return scalars like your example
        if action.numel() == 1:
            return int(action.item()), logp.squeeze(0), (None if value is None else value.squeeze(0))
    
        # Otherwise return batched tensors
        return action, logp, value

# -----------------------------
# PPO utilities
# -----------------------------
@dataclass
class PPOHyperparams:
    # rollout / total
    rollout_steps: int = 128            # T
    total_steps: int = 1_000_000        # env-steps across ALL envs

    # discount / GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO losses
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01

    # optimization
    update_epochs: int = 4
    minibatch_size: int = 1024
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    lr: float = 1e-4


def compute_gae(
    rewards: torch.Tensor,      # (T,N)
    dones: torch.Tensor,        # (T,N) float {0,1}
    values: torch.Tensor,       # (T,N)
    last_value: torch.Tensor,   # (N,)
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    adv = torch.zeros((T, N), device=rewards.device)
    last_gae = torch.zeros((N,), device=rewards.device)

    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae

    ret = adv + values
    return adv, ret

# --- metrics on the STATIC pile (board without active mask) ---
def _static_occ(obs: dict) -> np.ndarray:
    board = obs["board"].astype(np.uint8)
    mask  = obs["active_tetromino_mask"].astype(np.uint8)
    return ((board > 0) & (mask == 0)).astype(np.uint8)  # (N,H,W) or (H,W)

def metrics_from_static_torch(static_occ: torch.Tensor):
    """
    static_occ: (N,H,W) or (H,W), dtype bool/uint8/float
    Returns (holes, agg_height, bumpiness) float32 tensors of shape (N,)
    """
    if static_occ.ndim == 2:
        static_occ = static_occ.unsqueeze(0)     # (1,H,W)

    occ = static_occ.to(dtype=torch.bool)
    N, H, W = occ.shape
    device = occ.device

    filled = occ.any(dim=1)                      # (N,W)

    # first filled index along height (top->bottom)
    # argmax works on numeric, so cast to int
    first = occ.to(torch.int32).argmax(dim=1)    # (N,W)
    top = torch.where(filled, first, torch.full_like(first, H))  # (N,W)

    heights = torch.where(filled, (H - top), torch.zeros_like(top)).to(torch.int32)
    agg_height = heights.sum(dim=1).to(torch.float32)

    rows = torch.arange(H, device=device).view(1, H, 1)         # (1,H,1)
    below_top = rows >= top.unsqueeze(1)                        # (N,H,W)

    holes = ((~occ) & below_top).sum(dim=(1, 2)).to(torch.float32)

    bumpiness = (heights[:, 1:] - heights[:, :-1]).abs().sum(dim=1).to(torch.float32)

    return holes, agg_height, bumpiness


def metrics_from_static_vec(static_occ: np.ndarray):
    """
    static_occ: (N,H,W) or (H,W), {0,1} or bool
    Returns float32 arrays (holes, agg_height, bumpiness) each shape (N,)
    Matches the semantics of your original function.
    """
    if static_occ.ndim == 2:
        static_occ = static_occ[None, ...]   # (1,H,W)

    occ = static_occ.astype(bool)            # (N,H,W)
    N, H, W = occ.shape

    # filled per column
    filled = occ.any(axis=1)                 # (N,W) bool

    # first filled cell index from top.
    # argmax returns 0 if all False, so we must mask with `filled`.
    first = occ.argmax(axis=1)               # (N,W) int
    top = np.where(filled, first, H)         # (N,W) int, H means empty column

    # heights: number of cells from topmost filled to bottom
    heights = np.where(filled, H - top, 0).astype(np.int32)   # (N,W)
    agg_height = heights.sum(axis=1).astype(np.float32)       # (N,)

    # holes: zeros *below the topmost filled cell*
    # Create a mask selecting rows >= top for each (N,W)
    rows = np.arange(H, dtype=np.int32)[None, :, None]        # (1,H,1)
    below_top = rows >= top[:, None, :]                       # (N,H,W) bool

    holes = ((~occ) & below_top).sum(axis=(1, 2)).astype(np.float32)  # (N,)

    # bumpiness: sum abs diff between adjacent column heights
    bumpiness = np.abs(np.diff(heights, axis=1)).sum(axis=1).astype(np.float32)  # (N,)

    return holes, agg_height, bumpiness

def _detect_lock(prev_obs: dict, obs: dict) -> np.ndarray:
    """
    Returns locked: (N,) bool.
    Lock detected if some previously-active cells became non-active and are now filled.
    """
    pm = prev_obs["active_tetromino_mask"].astype(np.uint8)
    b  = obs["board"].astype(np.uint8)
    m  = obs["active_tetromino_mask"].astype(np.uint8)

    if pm.ndim == 2:
        pm = pm[None, ...]; b = b[None, ...]; m = m[None, ...]

    became_static = (pm == 1) & (m == 0) & (b > 0)  # (N,H,W)
    return became_static.reshape(became_static.shape[0], -1).any(axis=1)

def _line_reward(lines: np.ndarray) -> np.ndarray:
    table = np.array([0, 10, 30, 60, 120], dtype=np.float32)
    return table[np.clip(lines, 0, 4)]

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    global_step: int, hp: PPOHyperparams, extra: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hp": vars(hp),
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

def train_ppo(
    model: nn.Module,
    env,  # gymnasium.vector.VectorEnv (preferred) or any env w/ .num_envs + vector step/reset API
    optimizer: torch.optim.Optimizer,
    print_every: int = 10_000,
    hp: PPOHyperparams = PPOHyperparams(),
    ckpt_dir: str = "checkpoints",
    ckpt_every_steps: int = 100_000_000,   # save every 10M env-steps
):
    """
    PPO training loop for vectorized Gymnasium envs returning obs as (N,H,W,C) float32.
    Assumes Discrete action space and that `model(obs)` returns (logits, value).
    """
    device = next(model.parameters()).device

    # Vector env assumptions
    if not hasattr(env, "num_envs"):
        raise ValueError("env must be a vectorized environment with attribute `num_envs`.")
    N = env.num_envs
    T = hp.rollout_steps

    # Infer action count if possible
    n_actions = getattr(getattr(env, "single_action_space", None), "n", None)
    if n_actions is None:
        n_actions = getattr(getattr(env, "action_space", None), "n", None)
    if n_actions is None:
        raise ValueError("Discrete action space required (couldn't infer .n).")

    # Rollout buffers (allocated after first reset to infer H/W/C)
    obs, _ = env.reset()

    # ep_return = np.zeros((N,), dtype=np.float32)
    # ep_len = np.zeros((N,), dtype=np.int32)
    prev_obs = obs
    
    # shaping state
    static0 = _static_occ(obs)
    prev_holes, prev_h, prev_b = metrics_from_static_vec(static0)
    x0 = obs_to_torch(obs, device)
    C, H, W = x0.shape[1], x0.shape[2], x0.shape[3]

    obs_buf = torch.zeros((T, N, C, H, W), device=device)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)

    global_step = 0
    lines_total = 0
    next_print = print_every
    start_time = time.time()

    next_ckpt = ckpt_every_steps
    os.makedirs(ckpt_dir, exist_ok=True)
    def ckpt_name(tag: str):
        return os.path.join(ckpt_dir, f"ppo_tetris_{tag}_step{global_step}.pt")

    try:
        while global_step < hp.total_steps:
            # ---- Collect rollout ----
            for t in range(T):
                global_step += N

                x = obs_to_torch(obs, device)            # (N,C,H,W)
                with torch.no_grad():
                    logits, values = model(x)
                    logits[:, [6]] = -1e9 # forbid swap
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    logp = dist.log_prob(actions)

                obs_buf[t].copy_(x)
                val_buf[t] = values.detach()
                logp_buf[t] = logp.detach()
                act_buf[t] = actions

                next_obs, rewards, terminated, truncated, infos = env.step(actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated)
                term = terminated.astype(np.float32)
                r = np.asarray(rewards, dtype=np.float32)

                # ---- shaped reward (classic + safe shaping) ----
                # 1) lines cleared (if env doesn't provide it, we default to 0)
                lines = np.asarray(infos["lines_cleared"], dtype=np.int32)
                # r = _line_reward(lines)
                lines_total += int(lines.sum())

                # 3) apply shaping only when a lock happened
                locked = _detect_lock(prev_obs, next_obs).astype(np.float32)  # (N,)
                if locked.any():
                    static1 = _static_occ(next_obs)
                    holes, agg_h, bump = metrics_from_static_vec(static1)
                    k_holes, k_height, k_bump = 0.5, 0.1, 0.01
                    r += locked * (-(k_holes * (holes - prev_holes)) - (k_height * (agg_h - prev_h)) - (k_bump * (bump - prev_b)))
                    # update prev_* only on lock (so deltas stay meaningful)
                    prev_holes = np.where(locked > 0, holes, prev_holes)
                    prev_h     = np.where(locked > 0, agg_h, prev_h)
                    prev_b     = np.where(locked > 0, bump, prev_b)

                # 4) terminal penalty
                # r -= term * 2.0

                # r -= 0.0002  # per step

                # a = actions.cpu().numpy()  # (N,)
                # r -= (a == 7).astype(np.float32) * 0.001 # no_op penalty

                # 5) optional clip for stability
                # r = np.clip(r, -20.0, 20.0).astype(np.float32)

                # r should be the same array you store into rew_buf[t] (shape (N,))
                # ep_return += r
                # ep_len += 1

                # ---- IMPORTANT: reset shaping baselines on episode end ----
                if dones.any():
                    # baseline must correspond to the NEW episode state (which is next_obs after auto-reset)
                    static_reset = _static_occ(next_obs)
                    holes_r, agg_h_r, bump_r = metrics_from_static_vec(static_reset)

                    prev_holes = np.where(dones, holes_r, prev_holes)
                    prev_h     = np.where(dones, agg_h_r, prev_h)
                    prev_b     = np.where(dones, bump_r, prev_b)

                    # optional: reset episodic trackers
                    # ep_return[dones] = 0.0
                    # ep_len[dones] = 0

                # store shaped reward
                rew_buf[t] = torch.as_tensor(r, device=device, dtype=torch.float32)
                done_buf[t] = torch.as_tensor(term, device=device, dtype=torch.float32)

                # advance
                prev_obs = next_obs
                obs = next_obs

            # Bootstrap value
            with torch.no_grad():
                x_last = obs_to_torch(obs, device)
                _, last_value = model(x_last)  # (N,)
            last_value = last_value * (1.0 - done_buf[-1])

            # ---- GAE ----
            adv, ret = compute_gae(
                rewards=rew_buf,
                dones=done_buf,
                values=val_buf,
                last_value=last_value,
                gamma=hp.gamma,
                gae_lambda=hp.gae_lambda,
            )

            # Flatten
            B = T * N
            b_obs = obs_buf.reshape(B, C, H, W)
            b_act = act_buf.reshape(B)
            b_logp_old = logp_buf.reshape(B)
            b_adv = adv.reshape(B)
            b_ret = ret.reshape(B)
            b_val_old = val_buf.reshape(B)

            # Advantage norm
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            # ---- PPO updates ----
            idxs = np.arange(B)
            approx_kl = 0.0

            for epoch in range(hp.update_epochs):
                np.random.shuffle(idxs)
                kl_epoch = 0.0
                n_mb = 0

                for start in range(0, B, hp.minibatch_size):
                    mb_idx = idxs[start : start + hp.minibatch_size]
                    mb_obs = b_obs[mb_idx]
                    mb_act = b_act[mb_idx]
                    mb_logp_old = b_logp_old[mb_idx]
                    mb_adv = b_adv[mb_idx]
                    mb_ret = b_ret[mb_idx]
                    mb_val_old = b_val_old[mb_idx]

                    logits, value = model(mb_obs)
                    logits[:, [6]] = -1e9 #forbid swap
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()

                    # policy loss
                    ratio = torch.exp(logp - mb_logp_old)
                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1.0 - hp.clip_coef, 1.0 + hp.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    # value loss (clipped)
                    v_clipped = mb_val_old + torch.clamp(value - mb_val_old, -hp.clip_coef, hp.clip_coef)
                    vf1 = (value - mb_ret).pow(2)
                    vf2 = (v_clipped - mb_ret).pow(2)
                    vf_loss = 0.5 * torch.max(vf1, vf2).mean()

                    loss = pg_loss + hp.vf_coef * vf_loss - hp.ent_coef * entropy

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), hp.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        kl_mb = (mb_logp_old - logp).mean().item()
                    kl_epoch += kl_mb
                    n_mb += 1

                approx_kl = kl_epoch / max(1, n_mb)
                if approx_kl > hp.target_kl:
                    break

            # ---- Logging ----
            if global_step >= next_print:
                sps = int(global_step / (time.time() - start_time))
                # episodic returns can be inside infos depending on env; here we log rollout stats
                print(
                    f"step={global_step:,}  sps={sps}  kl={approx_kl:.4f}  "
                    f"rew_mean={rew_buf.mean().item():.3f}  v_mean={val_buf.mean().item():.3f}"
                )
                print("lines/1e6steps", lines_total / (global_step / 1_000_000))
                next_print += print_every

            # ---- Periodic checkpoint ----
            if ckpt_every_steps and global_step >= next_ckpt:
                save_checkpoint(
                    ckpt_name("ckpt"),
                    model=model,
                    optimizer=optimizer,
                    global_step=global_step,
                    hp=hp,
                    extra={"lines_total": lines_total},
                )
                next_ckpt += ckpt_every_steps
    finally:
        # save "final" snapshot regardless (normal end or exception)
        print("Saving checkpoint...")
        save_checkpoint(
            ckpt_name("final"),
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            hp=hp,
            extra={"lines_total": lines_total},
        )

    return model

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_id = "tetris_gymnasium/Tetris"
    num_envs = 256
    env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_id) for _ in range(num_envs)], shared_memory=True)

    tetris_policy = TetrisPolicy(2, env.single_action_space.n)
    tetris_policy = tetris_policy.to(device)
    hyperparams = PPOHyperparams(minibatch_size=2048, rollout_steps=256, total_steps=1_000_000_000, update_epochs=3, lr=1e-3, ent_coef=0.03)
    optimizer = optim.Adam(tetris_policy.parameters(), lr=hyperparams.lr)
    trained_policy = train_ppo(model=tetris_policy, env=env, optimizer=optimizer, hp=hyperparams, ckpt_every_steps=1_000_000)