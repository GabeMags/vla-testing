import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# Swap this import per-policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# load a policy
model_id = "lerobot/smolvla_base"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()

import numpy as np
from PIL import Image

img = Image.open("/home/gabriel/vla-testing/frames/frame_000.png").convert("RGB").resize((256, 256))
img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0   # (3,256,256)
img_t = img_t.unsqueeze(0).to(device)                                       # (1,3,256,256)

batch = {
    "observation.images.camera1": img_t,
    "observation.images.camera2": img_t,   # duplicate your one view for the other slots
    "observation.images.camera3": img_t,
    "observation.state": torch.zeros(1, 6, device=device),
    "task": "pick up the blue cube",
}

from lerobot.policies.factory import make_pre_post_processors

preprocess, postprocess = make_pre_post_processors(
    policy.config, "lerobot/smolvla_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

batch = preprocess(batch)          # tokenizes task, normalizes state/images, moves to device
with torch.inference_mode():
    action = policy.select_action(batch)
action = postprocess(action)       # unnormalizes back to real units
print(action.shape, action)