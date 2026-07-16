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

# Isaac Lab
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import isaaclab_tasks  # registers the tasks
from isaaclab_tasks.utils import parse_env_cfg

# Adjustable vars
SCALE = 0.5 #scale by which smolvla operates on the arm
LENGTH_S = 30 #how long the sim has before it times out

# --- build env config, then INJECT the camera into its scene ---
env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Rel-v0", num_envs=1)
env_cfg.episode_length_s = LENGTH_S 
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


# Frames folder for today's images
def get_output_dir(scale):
    """Get unique output directory with date, scale, and run counter."""
    base_dir = "/home/gabriel/vla-testing/frames"
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    run_counter = 0
    
    while True:
        run_name = f"{date_str}_scale-{scale}_{run_counter}"
        out_dir = os.path.join(base_dir, run_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        run_counter += 1

out_dir = get_output_dir(SCALE)

# Request
for step in range(200):
    joint_pos = env.unwrapped.scene["robot"].data.joint_pos[0][:6]   # first 6 of Franka's 9
    state_str = ",".join(f"{v:.4f}" for v in joint_pos.cpu().numpy())

    rgb = env.unwrapped.scene["external_cam"].data.output["rgb"][0]
    rgb_np = rgb.cpu().numpy().astype(np.uint8)[..., :3]
    buf = io.BytesIO(); Image.fromarray(rgb_np).save(buf, format="PNG")
    r = requests.post("http://127.0.0.1:8000/act",
                      files={"image": buf.getvalue()},
                      data={"state": state_str, "instruction": "pick up the blue cube"})
    a = torch.tensor(r.json()["action"], dtype=torch.float32, device=env.unwrapped.device)

    # SmolVLA uses 6 dims but the franka wants 7
    a6 = torch.tensor(r.json()["action"], dtype=torch.float32, device=env.unwrapped.device)
    a = torch.zeros(7, device=env.unwrapped.device)
    a[:6] = a6 * SCALE
    a[6] = 1.0
    env.step(a.unsqueeze(0))

    # Save frame
    print(f"step {step}: action {a.cpu().numpy().round(4)}")
    if step % 20 == 0:
        Image.fromarray(rgb_np).save(os.path.join(out_dir, f"loop_{step:03d}.png"))

env.close()
simulation_app.close()