import io, torch, numpy as np
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
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

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

    instruction = request.form.get("instruction", "pick up the blue cube")
    state_str = request.form.get("state", "0,0,0,0,0,0")
    state = torch.tensor([[float(x) for x in state_str.split(",")]],
                         dtype=torch.float32, device=device)

    # smolvla_base's config declares exactly camera1/2/3 (checked in the checkpoint's
    # config.json), and VISUAL normalization is IDENTITY, so there are no per-camera stats
    # to mismatch — the slots are interchangeable. Training datasets were mostly 2-camera
    # (one third-person + one wrist), so we fill the third slot with the third-person view.
    batch = {
        "observation.images.camera1": img_t,     # third-person
        "observation.images.camera2": wrist_t,   # wrist
        "observation.images.camera3": img_t,     # third slot: duplicate third-person
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