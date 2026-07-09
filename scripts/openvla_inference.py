import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=quant_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto"
)

image = Image.open("frames/frame_000.png")  # any image of a tabletop scene
prompt = "In: Pick up the blue cube\nOut:"

inputs = processor(prompt, image).to("cuda", dtype=torch.float16)
action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print("Predicted 7-DOF action:", action)