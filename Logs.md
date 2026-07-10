## Research Logs

### 2026-07-10: Attempted closed loop OpenVLA; pivot to SmolVLA due to GPU mem constraint (8GB)
I attempted to get a basic flask server running to get a closed loop inference on quantized (4bit) OpenVLA with IsaacSim Franka and a cube. This failed because I kept running into memory issues having the quantized model running then attempting to run a headless IsaacSim instance to give the model server a frame to analyze. I even dumbed down the simulated camera resolution to 256x256, unplugged all but one monitor, closed all unnecessary apps, and it still wasn't enough. I made the decision to try and move to SmolVLA which should be better given that it seems to have been made with consumer hardware memory constraints in mind.


### 2026-07-08: Ran OpenVLA inference on IsaacSim simulated frame
 Grabbed a tutorial script from IsaacLab that shows how to get sensors to work, created `capture_frame.py` to get IsaacSim to simulate a Franka arm with a cube on a table, then set the camera to save a frame to the project. Then ran the inference to pull the frame and determine a movement tensor to pick up the cube.
 - Had to do a lot of fussing with the simulated camera (why does it ship with such high sensitivity when navigating the 3D world?!)
 - Had to realize that OpenVLA was trained on a certain angle for the camera meaning an optimal location was a good idea
 - Running the sim headless was best for my poor GPU
 - Changed the inference code prompt to `prompt = "In: Pick up the blue cube\nOut:"` even though the cube is actually multicolored, I just wanted to see what would happen. I got back an action tensor successfully, I just don't know what that would look like. (Predicted 7-DOF action: [-0.00065614 -0.0111082  -0.00154037  0.01391019  0.02668386 -0.05970324 0.])

 Achieved:
 Isaac Lab scene in Isaac Sim -> camera frame -> OpenVLA -> 7 DOF action tensor

### 2026-07-07: Ran OpenVLA inference (quantized for my RTX3070) on real photo; SmolVLA is next to compare
Loaded up and ran inference on OpenVLA (quantized so it fits on my 3070). Giving it a real image of a cup on my desk and then later comparing it to SmolVLA which fits better on my GPU.
- Had to fuss with a lot of mismatched libraries because I was trying to run a quantized model from 2024 on libraries from now.
- Worked with Claude to find that OpenVLA actually recommends to just have a dedicated conda env for era-correct libs
- Discovered that inputs need to match the weights (in my case my processor was creating an image tensor in 32 when the model weights are set to 16)
- Learned a bunch of things in the process so I'm blurry on some things like timm, how bitsandbytes works....

I used the prompt `prompt = "In: What action should the robot take to pick up the mug?\nOut:"` and got the output `Predicted 7-DOF action: [ 1.48869192e-05 -1.95259519e-02  1.83735840e-03  1.51873807e-02
 -5.10204509e-02 -9.51627977e-02  9.96078431e-01]` when I gave it a sample image of my nasa mug on my table.

 I'll need to figure out exactly what each number means.

 A few notes:
 OpenVLA inference runs in a separate openvla conda env (Python 3.10) with era-pinned libraries — see requirements-openvla.txt.

### 2026-07-03: PyTorch quickstart completed
Worked through the PyTorch quickstart tutorial (FashionMNIST classifier).
Built conceptual understanding of:
- Model definition with nn.Module, forward pass, and layer stacking
- Weights vs. biases and how each is learned via gradient descent
- Cross-entropy loss and softmax for classification
- Training vs. evaluation modes, batch size, epochs
- Why VRAM matters and how parameters map to memory

### 2026-5-24: Initial environment setup
- Ubuntu 22.04 dual-boot completed
- NVIDIA driver 580 verified via `nvidia-smi`
- conda + Python 3.11 env created (`isaaclab`)
- PyTorch CUDA confirmed: `torch.cuda.is_available() == True`, RTX 3070 detected
- Isaac Sim 5.1.0 installed via pip
- Isaac Lab cloned and installed
- Smoke test passed: `Isaac-Lift-Cube-Franka-v0` task launches and runs RL training
- Known noise: pip dependency conflicts on psutil, click, torchaudio — not blocking
- Observation: Isaac Sim startup is heavy on 3070; reducing monitor count helps