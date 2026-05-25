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

## Project structure (planned)
vla-thesis/
├── README.md           # This file
├── learning/           # PyTorch fundamentals exercises (MNIST, etc.)
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

