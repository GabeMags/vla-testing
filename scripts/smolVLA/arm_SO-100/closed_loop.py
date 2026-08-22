# Closed-loop SmolVLA -> IsaacSim SO-ARM100.
# Scene is distribution-matched to SmolVLA's SO-100 training data via scene_match.py:
# debug markers off, webcam-like 4:3 camera framing the whole arm, wrist cam, realistic
# arm color. See scene_match.py for the reasoning behind each.

# Standard library
import argparse
import datetime
import io
import json
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
parser.add_argument("--preset2", default="front_left", help="Second third-person view; pass '' to disable.")
parser.add_argument("--arm-color", default="white", choices=["white", "black", "yellow"])
parser.add_argument("--no-wrist", action="store_true", help="Duplicate the third-person view into all 3 slots.")
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False   # False while iterating camera pose in GUI; True for real runs
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Sim's Kit app keeps non-daemon threads alive, so an uncaught Python exception does
# NOT end the process -- it stops stepping and sits there looking like a frozen sim (and
# ignores SIGTERM). Force a real exit so failures are visible as failures.
def _exit_on_uncaught(exc_type, exc, tb):
    import traceback
    traceback.print_exception(exc_type, exc, tb)
    try:
        env.close()          # noqa: F821  (may not exist yet)
    except Exception:
        pass
    try:
        simulation_app.close()
    except Exception:
        pass
    os._exit(1)


sys.excepthook = _exit_on_uncaught

import gymnasium as gym
import isaaclab_tasks           # type: ignore # registers built-in Isaac Lab tasks with gym
import isaac_so_arm101.tasks    # type: ignore # registers SO-ARM100/101 tasks (external project; import = registration)
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_match as sm        # type: ignore

# Adjustable vars
LENGTH_S = 30   # episode timeout (sim-seconds) before env auto-resets
INSTRUCTION = "pick up the cube"   # recorded into run_info.json; keep the two in sync
# This folder is SO-ARM100 only -- the arm is pinned, not a flag, so a run can never
# be mislabelled. The SO-ARM101 copy lives in ../arm_SO-101.
ARM = "so100"
TASK = sm.ARMS[ARM]["task"]

env_cfg = parse_env_cfg(TASK, num_envs=1)
env_cfg.episode_length_s = LENGTH_S

# --- action interface: absolute joint targets, not deltas --------------------------------
# SmolVLA emits ABSOLUTE joint positions in degrees (the checkpoint's own action mean is
# ~[1.6, 120, 110, 57, -27, 12], i.e. a pose, not a nudge). The lift task ships arm_action
# as scale=0.5 + use_default_offset=True, which computes
#     target = 0.5 * action + rest_pose
# and so reinterprets an absolute target as a half-strength delta from rest. Make the
# commanded value the target itself.
env_cfg.actions.arm_action.scale = 1.0
env_cfg.actions.arm_action.use_default_offset = False

# --- distribution matching (all three must happen before gym.make) ---
sm.disable_debug_vis(env_cfg)   # frame triads / goal markers never appear in training data
sm.add_scene_cameras(env_cfg, preset=args_cli.preset, preset2=args_cli.preset2 or None,
                     wrist=not args_cli.no_wrist, arm=ARM)

env = gym.make(TASK, cfg=env_cfg)
obs, _ = env.reset()

sm.aim_external_cam(env, args_cli.preset)
if args_cli.preset2:
    sm.aim_external_cam(env, args_cli.preset2, cam_name="external_cam2")
sm.recolor_arm(body_color=args_cli.arm_color)   # yellow -> white PLA; servos stay black

# --- sanity: does our 6-dim vector line up with what Isaac expects? ----------------------
# State order (what we send SmolVLA) and action order (what we command) are built by
# different machinery and can disagree independently. Warn rather than fail so a run is
# never blocked by this check.
_EXPECTED = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_state_order = list(env.unwrapped.scene["robot"].data.joint_names)
_am = env.unwrapped.action_manager
_action_order = [j for _t in _am.active_terms for j in getattr(_am.get_term(_t), "_joint_names", [])]
print(f"[check] state order : {_state_order}")
print(f"[check] action order: {_action_order}")
if _state_order != _EXPECTED:
    print(f"[check] WARNING state order != SmolVLA convention {_EXPECTED}")
if _action_order != _state_order:
    print("[check] WARNING action order != state order; the 6-dim vector is not consistent")
_arm_term = _am.get_term("arm_action")
# _offset is a plain float 0.0 when use_default_offset=False, and a tensor when it is True.
_off = _arm_term._offset
_off_max = float(_off.abs().max()) if hasattr(_off, "abs") else abs(float(_off))
print(f"[check] arm_action scale={_arm_term._scale} offset_max={_off_max:.3f}")

# Zero action sized from the env itself — never hardcode action dims
zero_action = torch.zeros((1, env.action_space.shape[1]), device=env.unwrapped.device)

# Debug to see if the joint names match what SmolVLA expects
# print("action_space:", env.action_space)
# print("joint_names:", env.unwrapped.scene["robot"].data.joint_names)


def get_output_dir(tag):
    """Unique output dir: <date>_<tag>-<run number>, e.g. 2026-08-22_so101-lift-0."""
    base_dir = "/home/gabriel/vla-testing/frames"
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    run_counter = 0
    while True:
        out_dir = os.path.join(base_dir, f"{date_str}_{tag}-{run_counter}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        run_counter += 1


out_dir = get_output_dir(f"{ARM}-lift")

# The folder name no longer carries preset/colour, so record the run's settings beside the
# frames -- otherwise there is no way to tell two runs apart after the fact.
with open(os.path.join(out_dir, "run_info.json"), "w") as _f:
    json.dump(
        {
            "arm": ARM,
            "task": TASK,
            "preset": args_cli.preset,
            "preset2": args_cli.preset2 or None,
            "arm_color": args_cli.arm_color,
            "wrist_cam": not args_cli.no_wrist,
            "steps": args_cli.steps,
            "instruction": INSTRUCTION,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        _f,
        indent=2,
    )

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
    if args_cli.preset2:
        files["image2"] = as_png(grab("external_cam2"))

    # --- inference over HTTP ---
    r = requests.post("http://127.0.0.1:8000/act",
                      files=files,
                      data={"state": state_str, "instruction": INSTRUCTION})

    # --- action: SmolVLA absolute degrees -> Isaac radians (native 6-dim, no padding) ---
    # deg2rad is correct only because the server now unnormalizes to real degrees; before
    # that fix these were z-scores and deg2rad silently shrank them ~100x.
    a = torch.deg2rad(torch.tensor(r.json()["action"], dtype=torch.float32, device=env.unwrapped.device))
    env.step(a.unsqueeze(0))

    print(f"step {step}: action(rad) {a.cpu().numpy().round(4)}")
    if step % 20 == 0:
        Image.fromarray(ext).save(os.path.join(out_dir, f"loop_{step:03d}.png"))
        if use_wrist:
            Image.fromarray(grab("wrist_cam")).save(os.path.join(out_dir, f"wrist_{step:03d}.png"))
        if args_cli.preset2:
            Image.fromarray(grab("external_cam2")).save(os.path.join(out_dir, f"cam2_{step:03d}.png"))

env.close()
simulation_app.close()
