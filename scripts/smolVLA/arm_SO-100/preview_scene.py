"""Render one frame per camera preset so camera choice is a comparison, not a guess.

Writes to frames/<date>_preview_<n>/:
    external_<preset>.png     what SmolVLA's third-person slot would see
    external_<preset>_pad.png the same frame after SmolVLA's 512x512 pad-resize
    wrist.png                 the wrist slot (fixed to the gripper, so pose-independent)

The _pad images are the important ones: that is literally the tensor SigLIP receives, black
letterbox bar and all. Compare those against real SO-100 dataset frames.

Usage (isaaclab env, no server needed):
    python scripts/smolVLA/arm_SO-100/preview_scene.py
    python scripts/smolVLA/arm_SO-100/preview_scene.py --arm-color black --headless
"""

import argparse
import datetime
import os
import sys

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--arm-color", default="white", choices=["white", "black", "yellow"])
parser.add_argument("--servo-color", default=None, help="Leave unset to keep the black servos.")
parser.add_argument("--presets", default=None, help="Comma-separated subset; default is all.")
parser.add_argument("--list-materials", action="store_true", help="Dump shader names, then exit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401  # registers built-in tasks
import isaac_so_arm101.tasks  # noqa: F401  # registers SO-ARM100/101 tasks
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_match as sm

TASK = "Isaac-SO-ARM100-Lift-Cube-Play-v0"

env_cfg = parse_env_cfg(TASK, num_envs=1)
env_cfg.episode_length_s = 1000.0  # no auto-reset while we sweep poses
sm.disable_debug_vis(env_cfg)
sm.add_scene_cameras(env_cfg, preset=sm.DEFAULT_PRESET)

env = gym.make(TASK, cfg=env_cfg)
env.reset()

if args_cli.list_materials:
    sm.list_arm_materials()
    env.close()
    simulation_app.close()
    sys.exit(0)

sm.recolor_arm(body_color=args_cli.arm_color, servo_color=args_cli.servo_color)

import torch

zero_action = torch.zeros((1, env.action_space.shape[1]), device=env.unwrapped.device)
for _ in range(60):  # settle physics and let the renderer warm up
    env.step(zero_action)


def out_dir():
    base = "/home/gabriel/vla-testing/frames"
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    n = 0
    while True:
        d = os.path.join(base, f"{date_str}_preview_{n}")
        if not os.path.exists(d):
            os.makedirs(d)
            return d
        n += 1


def grab(cam_name):
    rgb = env.unwrapped.scene[cam_name].data.output["rgb"][0]
    return rgb.cpu().numpy().astype(np.uint8)[..., :3]


def pad_like_smolvla(img_np, size=512):
    """Mirror lerobot resize_with_pad: fit inside size x size, pad on left and top."""
    h, w = img_np.shape[:2]
    ratio = max(w / size, h / size)
    rw, rh = int(w / ratio), int(h / ratio)
    resized = Image.fromarray(img_np).resize((rw, rh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, (size - rw, size - rh))  # left/top padding
    return canvas


d = out_dir()
presets = args_cli.presets.split(",") if args_cli.presets else list(sm.CAMERA_PRESETS)

for preset in presets:
    sm.aim_external_cam(env, preset)
    for _ in range(6):  # let the new pose render
        env.step(zero_action)
    img = grab("external_cam")
    Image.fromarray(img).save(os.path.join(d, f"external_{preset}.png"))
    pad_like_smolvla(img).save(os.path.join(d, f"external_{preset}_pad.png"))

if "wrist_cam" in env.unwrapped.scene.keys():
    wrist = grab("wrist_cam")
    Image.fromarray(wrist).save(os.path.join(d, "wrist.png"))
    pad_like_smolvla(wrist).save(os.path.join(d, "wrist_pad.png"))

print(f"\n[preview] wrote {len(os.listdir(d))} images to {d}")
env.close()
simulation_app.close()
