# VLA Thesis Research

Independent research toward M.S. thesis in vision-language-action (VLA) robotic manipulation.

## Goals

- Build foundational understanding of VLA architectures for robotic manipulation
- Reproduce and evaluate pretrained OpenVLA in Isaac Lab simulation
- Fine-tune VLA models on custom manipulation tasks via behavior cloning
- Develop a research portfolio to demonstrate independent capability


## Roadmap
- [x] Environment: Ubuntu dual-boot, NVIDIA driver, Isaac Sim + Isaac Lab smoke test
- [x] PyTorch foundations: quickstart, training loop fundamentals
- [x] OpenVLA quantized inference running locally on RTX 3070
- [X] Run SmolVLA; compare against OpenVLA outputs
- [X] Bridge to sim: feed an Isaac Lab camera frame into OpenVLA inference
- [ ] Closed loop: VLA actions driving the simulated Franka
- [ ] Quantized SmolVLA: measure success rate degradation when quantized to INT8/INT4
- [ ] Fine-tuning (LoRA) on a custom task — needs cloud GPU or upgraded hardware

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

 OpenVLA inference runs in a separate openvla conda env (Python 3.10) with era-pinned libraries — see requirements-openvla.txt.

## Research Logs
Moved to Logs.md
