# closed_loop.py — run via: ./isaaclab.sh -p closed_loop.py --enable_cameras
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
env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Rel-v0", num_envs=1)
env_cfg.scene.external_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/external_cam",
    update_period=0.1,
    height=256, width=256,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
    ),
    # ~1.2m out on +X, 0.7m up, looking back toward the workspace.
    # Quaternion (w,x,y,z), ROS convention. Iterate on this after seeing the first frame.
    offset=CameraCfg.OffsetCfg(pos=(1.6, 0.0, 0.9), convention="ros"),
)
env_cfg.commands.object_pose.debug_vis = False

env = gym.make("Isaac-Lift-Cube-Franka-IK-Rel-v0", cfg=env_cfg)
obs, _ = env.reset()

cam = env.unwrapped.scene["external_cam"]
eye = torch.tensor([[-0.212, -1.409, 1.294]], device=env.unwrapped.device)   # your GUI position
target = torch.tensor([[0.0, 0.0, 0.1]], device=env.unwrapped.device)        # table center-ish
cam.set_world_poses_from_view(eyes=eye, targets=target)

zero_action = torch.zeros((1, 7), device=env.unwrapped.device)
for _ in range(50):
    env.step(zero_action)

import requests, io
for step in range(200):
    rgb = env.unwrapped.scene["external_cam"].data.output["rgb"][0]
    rgb_np = rgb.cpu().numpy().astype(np.uint8)[..., :3]
    buf = io.BytesIO(); Image.fromarray(rgb_np).save(buf, format="PNG")
    r = requests.post("http://127.0.0.1:8000/act",
                      files={"image": buf.getvalue()},
                      data={"instruction": "pick up the blue cube"})
    a = torch.tensor(r.json()["action"], dtype=torch.float32, device=env.unwrapped.device)
    # OpenVLA: [dx,dy,dz,droll,dpitch,dyaw,gripper(0..1)] -> IK-Rel expects same 7 dims,
    # but gripper convention may be binary open/close: map a[6] = 1.0 if a[6] > 0.5 else -1.0
    a[6] = 1.0 if a[6] > 0.5 else -1.0
    env.step(a.unsqueeze(0))

    print(f"step {step}: action {a.cpu().numpy().round(4)}")
    if step % 20 == 0:
        Image.fromarray(rgb_np).save(f"/home/gabriel/vla-testing/frames/loop_{step:03d}.png")

# grab the frame
# rgb = env.unwrapped.scene["external_cam"].data.output["rgb"][0]
# rgb_np = rgb.cpu().numpy().astype(np.uint8)[..., :3]  # drop alpha if present

# import os
# os.makedirs("/home/gabriel/vla-testing/frames", exist_ok=True)
# Image.fromarray(rgb_np).save("/home/gabriel/vla-testing/frames/frame_000.png")
# print("Saved frame_000.png")

env.close()
simulation_app.close()