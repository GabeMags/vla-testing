# Closed-loop SmolVLA -> IsaacSim SO-ARM100.
# Scene is distribution-matched to SmolVLA's SO-100 training data via scene_match.py:
# debug markers off, webcam-like 4:3 camera framing the whole arm, wrist cam, realistic
# arm color. See scene_match.py for the reasoning behind each.

# Standard library
import argparse
import datetime
import io
import os
import sys

# Third-party
import numpy as np
import requests
import torch
from PIL import Image

# Isaac Lab — AppLauncher must run before other isaaclab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="front_right", help="Camera preset (see scene_match.CAMERA_PRESETS).")
parser.add_argument("--arm-color", default="white", choices=["white", "black", "yellow"])
parser.add_argument("--no-wrist", action="store_true", help="Duplicate the third-person view into all 3 slots.")
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False   # False while iterating camera pose in GUI; True for real runs
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks           # type: ignore # registers built-in Isaac Lab tasks with gym
import isaac_so_arm101.tasks    # type: ignore # registers SO-ARM100/101 tasks (external project; import = registration)
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_match as sm        # type: ignore

# Adjustable vars
LENGTH_S = 30   # episode timeout (sim-seconds) before env auto-resets
TASK = "Isaac-SO-ARM100-Lift-Cube-Play-v0"

env_cfg = parse_env_cfg(TASK, num_envs=1)
env_cfg.episode_length_s = LENGTH_S

# --- distribution matching (all three must happen before gym.make) ---
sm.disable_debug_vis(env_cfg)   # frame triads / goal markers never appear in training data
sm.add_scene_cameras(env_cfg, preset=args_cli.preset, wrist=not args_cli.no_wrist)

env = gym.make(TASK, cfg=env_cfg)
obs, _ = env.reset()

sm.aim_external_cam(env, args_cli.preset)
sm.recolor_arm(body_color=args_cli.arm_color)   # yellow -> white PLA; servos stay black

# Zero action sized from the env itself — never hardcode action dims
zero_action = torch.zeros((1, env.action_space.shape[1]), device=env.unwrapped.device)

# Debug to see if the joint names match what SmolVLA expects
# print("action_space:", env.action_space)
# print("joint_names:", env.unwrapped.scene["robot"].data.joint_names)


def get_output_dir(tag):
    """Unique output dir: date_tag_counter."""
    base_dir = "/home/gabriel/vla-testing/frames"
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    run_counter = 0
    while True:
        out_dir = os.path.join(base_dir, f"{date_str}_{tag}_{run_counter}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        run_counter += 1


out_dir = get_output_dir(f"soarm-lift_{args_cli.preset}_{args_cli.arm_color}")

use_wrist = not args_cli.no_wrist


def grab(cam_name):
    rgb = env.unwrapped.scene[cam_name].data.output["rgb"][0]
    return rgb.cpu().numpy().astype(np.uint8)[..., :3]


def as_png(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="PNG")
    return buf.getvalue()


for _ in range(50):          # settle physics + let camera render
    env.step(zero_action)

for step in range(args_cli.steps):
    # --- state: Isaac radians -> degrees (SO-100/LeRobot convention) ---
    joint_pos_deg = torch.rad2deg(env.unwrapped.scene["robot"].data.joint_pos[0])
    state_str = ",".join(f"{v:.4f}" for v in joint_pos_deg.cpu().numpy())

    # --- frames: send them at native 4:3; SmolVLA pad-resizes to 512x512 itself ---
    ext = grab("external_cam")
    files = {"image": as_png(ext)}
    if use_wrist:
        files["wrist"] = as_png(grab("wrist_cam"))

    # --- inference over HTTP ---
    r = requests.post("http://127.0.0.1:8000/act",
                      files=files,
                      data={"state": state_str, "instruction": "pick up the cube"})

    # --- action: SmolVLA degrees -> Isaac radians; native 6-dim, no padding/scale ---
    a = torch.deg2rad(torch.tensor(r.json()["action"], dtype=torch.float32, device=env.unwrapped.device))
    env.step(a.unsqueeze(0))

    print(f"step {step}: action(rad) {a.cpu().numpy().round(4)}")
    if step % 20 == 0:
        Image.fromarray(ext).save(os.path.join(out_dir, f"loop_{step:03d}.png"))
        if use_wrist:
            Image.fromarray(grab("wrist_cam")).save(os.path.join(out_dir, f"wrist_{step:03d}.png"))

env.close()
simulation_app.close()
