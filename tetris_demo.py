import argparse
import os
import re
import numpy as np
import torch
import gymnasium as gym
import imageio
from .tetris import obs_to_torch, TetrisPolicy

@torch.no_grad()
def policy_greedy_action(policy, obs):
    device = next(policy.parameters()).device
    x = obs_to_torch(obs, device)          # (1,2,24,18) for single env
    logits, _ = policy(x)
    return int(torch.argmax(logits, dim=-1).item())

def record_video(env, policy, out_path, fps=30, greedy=True):
    images = []
    obs, _ = env.reset()
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None. Create env with render_mode='rgb_array'.")
    images.append(frame)

    done = False
    while not done:
        if greedy:
            action = policy_greedy_action(policy, obs)
        else:
            action, _, _ = policy.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        images.append(env.render())

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, [np.asarray(im) for im in images], fps=fps)

def make_out_path_from_ckpt(ckpt_path: str, out_dir: str, ext: str) -> str:
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base + ext)

def run_ckpt_and_record(
    ckpt_path: str,
    out_dir: str = "videos",
    ext: str = ".mp4",
    env_id: str = "tetris_gymnasium/Tetris",
    fps: int = 30,
    device: str = "cuda",
    greedy: bool = True,
):
    device_t = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device_t)

    env = gym.make(env_id, render_mode="rgb_array")
    n_actions = env.action_space.n

    policy = TetrisPolicy(in_ch=2, n_actions=n_actions).to(device_t)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    policy.load_state_dict(state_dict)
    policy.eval()

    out_path = make_out_path_from_ckpt(ckpt_path, out_dir=out_dir, ext=ext)
    record_video(env, policy, out_path, fps=fps, greedy=greedy)
    env.close()
    return out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Load a PPO Tetris checkpoint and record an episode video.")
    p.add_argument("ckpt", type=str, help="Path to .pt checkpoint (e.g. checkpoints/ppo_tetris_final_step123.pt)")
    p.add_argument("--out-dir", type=str, default="videos", help="Directory to save the video")
    p.add_argument("--ext", type=str, default=".mp4", help="Output extension: .mp4 or .gif")
    p.add_argument("--env-id", type=str, default="tetris_gymnasium/Tetris", help="Gymnasium env id")
    p.add_argument("--fps", type=int, default=30, help="Video FPS")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--greedy", action="store_true", help="Use argmax actions (recommended for eval)")
    args = p.parse_args()

    out = run_ckpt_and_record(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        ext=args.ext,
        env_id=args.env_id,
        fps=args.fps,
        device=args.device,
        greedy=args.greedy,
    )
    print("Saved video:", out)