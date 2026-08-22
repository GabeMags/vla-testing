import io, torch, numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from PIL import Image
from flask import Flask, request, jsonify

# Swap this import per-policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# load a policy
model_id = "lerobot/smolvla_base"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
# --- bind normalization stats explicitly -------------------------------------------------
# The checkpoint ships its stats under dataset-namespaced keys ("so100.buffer.action.mean"),
# but the processor configs ask for a bare "action". Nothing binds, both normalizers silently
# pass through, and the policy returns z-scores instead of degrees -- which is why every
# action logged before 2026-08-22 was ~100x too small (|max| ~3 instead of ~120).
STATS_DATASET = "so100"     # "so100" is the general mix; -blue/-red are tighter single-scene
NORMALIZE_STATE = True      # see note below

_STATS_FILE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
try:
    _stats_path = hf_hub_download(model_id, _STATS_FILE)
except Exception:
    _stats_path = hf_hub_download(model_id, _STATS_FILE, local_files_only=True)
_raw_stats = load_file(_stats_path)
_a_mean = _raw_stats[f"{STATS_DATASET}.buffer.action.mean"]
_a_std = _raw_stats[f"{STATS_DATASET}.buffer.action.std"]

# Why the shipped state file does not bind on its own: load_state_dict() splits each key on
# its LAST dot, so "so100.buffer.action.mean" becomes feature "so100.buffer.action" + stat
# "mean". The processor looks features up as "action" / "observation.state", so nothing
# matches and both normalizers pass through. Note also that make_pre_post_processors()
# IGNORES its dataset_stats kwarg whenever pretrained_path is given -- it goes straight to
# PolicyProcessorPipeline.from_pretrained(), which honours only `overrides`. Supplying
# `stats` through overrides sets _stats_explicitly_provided, and load_state_dict() then
# deliberately preserves ours instead of overwriting with the unusable file keys.
_action_stats = {"mean": _a_mean, "std": _a_std}
_pre_stats = {"action": _action_stats}
if NORMALIZE_STATE:
    # No observation.state stats ship with the checkpoint at all. In SO-100 teleop the action
    # is the leader arm's joint positions and the state is the follower's, so the two track
    # each other closely -- reusing the action stats is an approximation, not ground truth.
    # Set NORMALIZE_STATE = False to leave state unnormalized instead.
    _pre_stats["observation.state"] = _action_stats

print(f"[server] normalization stats bound from '{STATS_DATASET}'")
print(f"[server]   action mean {[round(v, 2) for v in _a_mean.tolist()]}")
print(f"[server]   action std  {[round(v, 2) for v in _a_std.tolist()]}")
print(f"[server]   state normalization: {'on (action stats as proxy)' if NORMALIZE_STATE else 'off'}")

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "normalizer_processor": {"stats": _pre_stats},
    },
    postprocessor_overrides={
        "unnormalizer_processor": {"stats": {"action": _action_stats}},
    },
)

# Fail loudly at boot rather than silently emitting z-scores again.
_bound = postprocess.steps[0].stats if hasattr(postprocess.steps[0], "stats") else {}
assert "action" in _bound, f"unnormalizer did not bind action stats; got keys {list(_bound)}"
print("[server] unnormalizer verified: action stats are live")

#server
app = Flask(__name__)

def to_tensor(file_storage):
    """Bytes -> (1,3,H,W) float tensor in [0,1], at the sender's native resolution.

    Deliberately no resize here. SmolVLA's prepare_images() calls resize_with_pad(512,512),
    an aspect-preserving resize that pads left/top. Squashing to a square first would throw
    away that letterbox: real 640x480 training frames arrive as 512x384 content with a black
    bar on top, and we want the sim frames to land the same way.
    """
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    # (H,W,3) -> permute to (3,H,W), the (channels, height, width) order torch vision models
    # expect, then normalize the 0-255 ints to 0.0-1.0 floats and add a batch dim.
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


@app.route("/act", methods=["POST"])
def act():
    img_t = to_tensor(request.files["image"])          # third-person view
    # Wrist view if the client sends one; otherwise fall back to duplicating the third-person
    # frame, which is what we did before the wrist cam existed.
    wrist_t = to_tensor(request.files["wrist"]) if "wrist" in request.files else img_t
    # Second third-person view for the third slot; falls back to duplicating the first so
    # an older client that only posts image+wrist still works.
    img2_t = to_tensor(request.files["image2"]) if "image2" in request.files else img_t

    instruction = request.form.get("instruction", "pick up the blue cube")
    state_str = request.form.get("state", "0,0,0,0,0,0")
    state = torch.tensor([[float(x) for x in state_str.split(",")]],
                         dtype=torch.float32, device=device)

    # smolvla_base's config declares exactly camera1/2/3 with empty_cameras=0 (checked in
    # the checkpoint's config.json), so all three slots want a real tensor. VISUAL
    # normalization is IDENTITY, so there are no per-camera stats to mismatch and the slots
    # are interchangeable — the assignment below is just our convention.
    batch = {
        "observation.images.camera1": img_t,     # third-person (default front_right)
        "observation.images.camera2": wrist_t,   # wrist
        "observation.images.camera3": img2_t,    # second third-person (default front_left)
        "observation.state": state,
        "task": instruction,
    }

    batch = preprocess(batch) # tokenizes task, normalizes state/images, moves to device
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = postprocess(action)       # unnormalizes back to real units
    print(action.shape, action)

    # action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    return jsonify({"action": action.cpu().numpy().tolist()[0]})

app.run(host="127.0.0.1", port=8000)