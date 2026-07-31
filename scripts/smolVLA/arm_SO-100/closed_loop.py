# Phase 1: probe SO-ARM task's action space + joint names (loop disabled below)

# Standard library
import argparse
import datetime
import io
import os

# Third-party
import numpy as np
import requests
import torch
from PIL import Image

# Isaac Lab — AppLauncher must run before other isaaclab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False   # False while iterating camera pose in GUI; True for real runs
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import isaaclab_tasks           # type: ignore # registers built-in Isaac Lab tasks with gym
import isaac_so_arm101.tasks    # type: ignore # registers SO-ARM100/101 tasks (external project; import = registration)
from isaaclab_tasks.utils import parse_env_cfg

# Debug
# so_tasks = [k for k in gym.registry.keys() if "SO" in k.upper() or "ARM" in k.upper()]
# print("SO-ARM registered tasks:", so_tasks)

# Adjustable vars
LENGTH_S = 30   # episode timeout (sim-seconds) before env auto-resets
TASK = "Isaac-SO-ARM100-Lift-Cube-Play-v0"

# --- build env config, then INJECT the camera into its scene ---
env_cfg = parse_env_cfg(TASK, num_envs=1)
env_cfg.episode_length_s = LENGTH_S
# Debug-vis markers: config tree differs per task — inspect before disabling:
# print(env_cfg.commands)   # then e.g. env_cfg.commands.<term>.debug_vis = False

env_cfg.scene.external_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/external_cam",   # world-fixed (not parented to robot)
    update_period=0.1,
    height=256, width=256,                      # SmolVLA input res — render small, save VRAM
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
    ),
    # Placeholder pose — overridden below by set_world_poses_from_view.
    offset=CameraCfg.OffsetCfg(pos=(1.6, 0.0, 0.9), convention="ros"),
)

env = gym.make(TASK, cfg=env_cfg)
obs, _ = env.reset()

# Camera aim — NOTE: eye was tuned for the Franka scene; SO-ARM is much smaller,
# so expect to re-iterate this in the GUI (closer in, lower down).
cam = env.unwrapped.scene["external_cam"]
eye = torch.tensor([[-0.212, -1.409, 1.294]], device=env.unwrapped.device)
target = torch.tensor([[0.0, 0.0, 0.1]], device=env.unwrapped.device)   # workspace center-ish
cam.set_world_poses_from_view(eyes=eye, targets=target)

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

out_dir = get_output_dir("soarm-lift")

for _ in range(50):          # settle physics + let camera render
    env.step(zero_action)

for step in range(200):
    # --- state: Isaac radians -> degrees (SO-100/LeRobot convention) ---
    joint_pos_deg = torch.rad2deg(env.unwrapped.scene["robot"].data.joint_pos[0])
    state_str = ",".join(f"{v:.4f}" for v in joint_pos_deg.cpu().numpy())

    # --- frame ---
    rgb = env.unwrapped.scene["external_cam"].data.output["rgb"][0]
    rgb_np = rgb.cpu().numpy().astype(np.uint8)[..., :3]
    buf = io.BytesIO(); Image.fromarray(rgb_np).save(buf, format="PNG")

    # --- inference over HTTP ---
    r = requests.post("http://127.0.0.1:8000/act",
                      files={"image": buf.getvalue()},
                      data={"state": state_str, "instruction": "pick up the cube"})

    # --- action: SmolVLA degrees -> Isaac radians; native 6-dim, no padding/scale ---
    a = torch.deg2rad(torch.tensor(r.json()["action"], dtype=torch.float32, device=env.unwrapped.device))
    env.step(a.unsqueeze(0))

    print(f"step {step}: action(rad) {a.cpu().numpy().round(4)}")
    if step % 20 == 0:
        Image.fromarray(rgb_np).save(os.path.join(out_dir, f"loop_{step:03d}.png"))

env.close()
simulation_app.close()