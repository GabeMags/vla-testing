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

@app.route("/act", methods=["POST"])
def act():
    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB").resize((256, 256)) #smolvla likes this resolution 256x256
    # convert image to tensor, start as (256,256,3), then change the order of dimensions with permute to (3,256,256) which is how PyTorch vision models expect things (channels, height, width). Then normalize; convert the 0-255 integer pixel values to 0.0-1.0 floats.
    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    # Add a batch dimension. We're only doing one image per batch. (1,3,256,256)
    img_t = img_t.unsqueeze(0).to(device)

    instruction = request.form.get("instruction", "pick up the blue cube")
    state_str = request.form.get("state", "0,0,0,0,0,0")
    state = torch.tensor([[float(x) for x in state_str.split(",")]],
                         dtype=torch.float32, device=device)

    batch = {
        "observation.images.camera1": img_t,
        "observation.images.camera2": img_t,   # duplicate one view for the other slots
        "observation.images.camera3": img_t,
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