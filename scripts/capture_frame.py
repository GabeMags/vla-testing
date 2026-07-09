# capture_frame.py — run via: ./isaaclab.sh -p capture_frame.py --enable_cameras
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from PIL import Image
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import isaaclab_tasks  # registers the tasks
from isaaclab_tasks.utils import parse_env_cfg

# --- build env config, then INJECT the camera into its scene ---
env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-v0", num_envs=1)
env_cfg.scene.external_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/external_cam",
    update_period=0.1,
    height=480, width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
    ),
    # ~1.2m out on +X, 0.7m up, looking back toward the workspace.
    # Quaternion (w,x,y,z), ROS convention. Iterate on this after seeing the first frame.
    offset=CameraCfg.OffsetCfg(pos=(1.6, 0.0, 0.9), convention="ros"),
)

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
obs, _ = env.reset()

import torch
cam = env.unwrapped.scene["external_cam"]
eye = torch.tensor([[-0.212, -1.409, 1.294]], device=env.unwrapped.device)   # your GUI position
target = torch.tensor([[0.0, 0.0, 0.1]], device=env.unwrapped.device)        # table center-ish
cam.set_world_poses_from_view(eyes=eye, targets=target)

# step with zero actions so physics settles and the camera renders
zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
for _ in range(100):
    env.step(zero_action)

# grab the frame
rgb = env.unwrapped.scene["external_cam"].data.output["rgb"][0]
rgb_np = rgb.cpu().numpy().astype(np.uint8)[..., :3]  # drop alpha if present

import os
os.makedirs("/home/gabriel/vla-testing/frames", exist_ok=True)
Image.fromarray(rgb_np).save("/home/gabriel/vla-testing/frames/frame_000.png")
print("Saved frame_000.png")

env.close()
simulation_app.close()