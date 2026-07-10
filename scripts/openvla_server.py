import io, torch, numpy as np
from flask import Flask, request, jsonify
from PIL import Image

#model + processor
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16)

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code = True)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config = quant_config,
    trust_remote_code = True, 
    low_cpu_mem_usage = True,
    device_map = "auto"
)

#server
app = Flask(__name__)

@app.route("/act", methods=["POST"])
def act():
    image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    instruction = request.form["instruction"]
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image).to("cuda", dtype=torch.float16)
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    return jsonify({"action": np.asarray(action).tolist()})

app.run(host="127.0.0.1", port=8000)