# VLA Thesis Research

Independent research toward M.S. thesis in vision-language-action (VLA) robotic manipulation.

## Goals

- Build foundational understanding of VLA architectures for robotic manipulation
- Reproduce and evaluate pretrained OpenVLA in Isaac Lab simulation
- Fine-tune VLA models on custom manipulation tasks via behavior cloning
- Develop a research portfolio to demonstrate independent capability

## Hardware

- Desktop with NVIDIA RTX 3070 (8GB VRAM)
- Ubuntu 22.04 LTS (dual-boot with Windows 11)
- 2TB storage, ~1TB allocated to Ubuntu

## Software stack

- NVIDIA driver 580 (proprietary)
- conda env: `isaaclab` (Python 3.11)
- PyTorch 2.x with CUDA 12.1
- Isaac Sim 5.1.0 (pip install)
- Isaac Lab (cloned from main)

## Install log

### 05-24-26: Initial environment setup
- Ubuntu 22.04 dual-boot completed
- NVIDIA driver 580 verified via `nvidia-smi`
- conda + Python 3.11 env created (`isaaclab`)
- PyTorch CUDA confirmed: `torch.cuda.is_available() == True`, RTX 3070 detected
- Isaac Sim 5.1.0 installed via pip
- Isaac Lab cloned and installed
- Smoke test passed: `Isaac-Lift-Cube-Franka-v0` task launches and runs RL training
- Known noise: pip dependency conflicts on psutil, click, torchaudio — not blocking
- Observation: Isaac Sim startup is heavy on 3070; reducing monitor count helps

### 2026-07-03: PyTorch quickstart completed
Worked through the PyTorch quickstart tutorial (FashionMNIST classifier).
Built conceptual understanding of:
- Model definition with nn.Module, forward pass, and layer stacking
- Weights vs. biases and how each is learned via gradient descent
- Cross-entropy loss and softmax for classification
- Training vs. evaluation modes, batch size, epochs
- Why VRAM matters and how parameters map to memory


### 2026-07-07: Ran OpenVLA inference locally (quantized for my RTX3070); SmolVLA is next to compare
Loaded up and ran inference on OpenVLA (quantized so it fits on my 3070). Giving it a random image of a cup on my desk and then later comparing it to SmolVLA which fits better on my GPU.
- Had to fuss with a lot of mismatched libraries because I was trying to run a quantized model from 2024 on libraries from now.
- Worked with Claude to find that OpenVLA actually recommends to just have a dedicated conda env for era-correct libs
- Discovered that inputs need to match the weights (in my case my processor was creating an image tensor in 32 when the model weights are set to 16)
- Learned a bunch of things in the process so I'm blurry on some things like timm, how bitsandbytes works....

I used the prompt `prompt = "In: What action should the robot take to pick up the mug?\nOut:"` and got the output `Predicted 7-DOF action: [ 1.48869192e-05 -1.95259519e-02  1.83735840e-03  1.51873807e-02
 -5.10204509e-02 -9.51627977e-02  9.96078431e-01]` when I gave it a sample image of my nasa mug on my table.

 I'll need to figure out exactly what each number means.


## Project structure (planned)
vla-testing/
├── README.md           # This file
├── learning/           # PyTorch fundamentals exercises(MNIST, etc.)
├── scripts/            # Standalone experiment scripts
├── envs/               # Custom Isaac Lab environments
├── data/               # Demonstration collection
├── model/              # Policy architecture
└── notes/              # Paper notes, design decisions

## Roadmap

- [x] Day 1: Ubuntu dual-boot, NVIDIA driver
- [x] Day 2: Isaac Sim + Isaac Lab installed, smoke test passing
- [ ] Day 3 (Memorial Day): PyTorch fluency (MNIST), OpenVLA standalone inference
- [ ] Week 2: OpenVLA + Isaac Lab integration, zero-shot evaluation
- [ ] Week 3: LoRA fine-tuning on custom Isaac Lab task

