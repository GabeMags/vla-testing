"""Push the sim scene toward SmolVLA's SO-100 training distribution.

Import this from the closed-loop / preview scripts *after* AppLauncher has started.
"""

import math

# Isaac Lab imports are deferred into the functions that need them so this module can be
# imported before AppLauncher finishes booting the simulation app.

HORIZONTAL_APERTURE = 20.955  # cm, Isaac Lab default (35 mm spherical projector)


def focal_length_for_hfov(hfov_deg, horizontal_aperture=HORIZONTAL_APERTURE):
    """Isaac focal length (cm) giving a target horizontal FOV, so we can think in degrees."""
    return (horizontal_aperture / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


# --- third-person camera presets -------------------------------------------------------
#
# Scene geometry this is derived from (SoArm100LiftCubeEnvCfg):
#   robot base at env origin, table top at z=0, cube spawns at (0.2, 0.0, 0.015)
#   randomized x +/-0.1, y +/-0.2  ->  workspace is roughly x in [0.1, 0.3], y in [-0.2, 0.2].
# The arm reaches out along +x, so "front" means looking from +x back toward the base.
#
# eye/target are metres in env-local coordinates; hfov is horizontal FOV in degrees.
# A Logitech-class webcam is ~65-70 deg horizontal, which is what the community rigs use.

# Targets sit at z~0.12 rather than at the cube: aiming at the table dumped a third of the
# frame into empty foreground and clipped the arm off the top edge (preview_0).

CAMERA_PRESETS = {
    "front": dict(
        eye=(0.58, 0.00, 0.34),
        target=(0.14, 0.00, 0.10),
        hfov=68.0,
        note="Head-on across the table. Cleanest read of gripper height, weakest depth cue on y.",
    ),
    "front_right": dict(
        eye=(0.46, -0.30, 0.28),
        target=(0.14, 0.00, 0.12),
        hfov=68.0,
        note="3/4 view from the arm's right, ~0.47 m out at 20 deg. Median community SO-100 rig.",
    ),
    "front_left": dict(
        eye=(0.46, 0.30, 0.28),
        target=(0.14, 0.00, 0.12),
        hfov=68.0,
        note="Mirror of front_right; worth testing since the mix is not left/right symmetric.",
    ),
    "high_front": dict(
        eye=(0.42, -0.18, 0.38),
        target=(0.16, 0.00, 0.06),
        hfov=72.0,
        note="Steeper ~45 deg look-down. Better cube visibility, more foreshortened arm.",
    ),
    "low_front": dict(
        eye=(0.55, -0.22, 0.20),
        target=(0.13, 0.00, 0.12),
        hfov=68.0,
        note="Shallow ~10 deg angle, near table level. Common with laptop-webcam setups.",
    ),
    # Kept for A/B against the 2026-07-30 baseline run.
    "legacy": dict(
        eye=(-0.212, -1.409, 1.294),
        target=(0.00, 0.00, 0.10),
        hfov=47.2,
        note="The 2026-07-30 pose (Franka-era leftover). Baseline only.",
    ),
}

DEFAULT_PRESET = "front_right"

# Render 4:3 so SmolVLA's pad-resize reproduces the training letterbox (see module docstring).
DEFAULT_WIDTH, DEFAULT_HEIGHT = 640, 480


# --- wrist camera ----------------------------------------------------------------------
# Mounted on the `gripper` link, behind and above the jaws, tilted down so the grasp point
# lands near frame centre. At 0 deg tilt the jaws sat in the bottom corners and most of the
# frame was empty table (preview_0).

WRIST_CAM_POS = (0.0, -0.02, 0.045)
WRIST_CAM_TILT_DEG = 20.0
WRIST_HFOV = 85.0


def wrist_cam_quat(tilt_deg):
    """Quat (w, x, y, z) for a gripper-mounted camera looking down -y, tilted toward -z.

    convention="ros": the camera looks along its +Z, +Y is image-down, +X is image-right.
    Mapping cam +Z -> gripper (0, -cos t, -sin t) and cam +Y -> gripper (0, sin t, -cos t)
    gives a rotation matrix whose trace is -1 for every t, i.e. always a 180 deg turn, so
    w = 0 and the axis carries the tilt:

        q = (0, 0, (cos(t/2) + sin(t/2))/sqrt(2), -(cos(t/2) - sin(t/2))/sqrt(2))

    t = 0 recovers (0, 0, sqrt(2)/2, -sqrt(2)/2), the straight-down-the-jaw view.
    """
    half = math.radians(tilt_deg) / 2.0
    root2 = math.sqrt(2.0)
    return (
        0.0,
        0.0,
        (math.cos(half) + math.sin(half)) / root2,
        -(math.cos(half) - math.sin(half)) / root2,
    )


def disable_debug_vis(env_cfg):
    """Turn off frame-triad / goal-pose markers so they stay out of the camera frame.

    Must be called on the *config*, before gym.make.
    """
    turned_off = []
    ee_frame = getattr(env_cfg.scene, "ee_frame", None)
    if ee_frame is not None and getattr(ee_frame, "debug_vis", False):
        ee_frame.debug_vis = False
        turned_off.append("scene.ee_frame")

    commands = getattr(env_cfg, "commands", None)
    if commands is not None:
        for name in dir(commands):
            if name.startswith("_"):
                continue
            term = getattr(commands, name)
            if hasattr(term, "debug_vis") and term.debug_vis:
                term.debug_vis = False
                turned_off.append(f"commands.{name}")

    print(f"[scene_match] debug_vis off: {turned_off or 'nothing found'}")
    return turned_off


def add_scene_cameras(
    env_cfg,
    preset=DEFAULT_PRESET,
    preset2="front_left",
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    wrist=True,
    gripper_body="gripper",
):
    """Attach two third-person cameras and (optionally) a wrist camera to the scene config.

    smolvla_base declares three image slots with ``empty_cameras: 0``, so all three want a
    real tensor. ``preset``/``preset2`` fill two of them with distinct third-person views;
    the wrist cam fills the third. Pass ``preset2=None`` for the older 2-camera setup.

    The third-person poses here are placeholders; ``aim_external_cam`` sets them for real
    after reset, which is what lets us re-aim between presets without rebuilding the env.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    cfg = CAMERA_PRESETS[preset]

    def world_cam(name, preset_name):
        c = CAMERA_PRESETS[preset_name]
        return CameraCfg(
            prim_path="{ENV_REGEX_NS}/" + name,  # world-fixed, not parented to the robot
            update_period=0.0,  # render every step; the loop reads a frame every step
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length_for_hfov(c["hfov"]),
                focus_distance=400.0,
                horizontal_aperture=HORIZONTAL_APERTURE,
                # vertical_aperture=None -> derived from height/width, so pixels stay square.
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(pos=c["eye"], convention="ros"),
        )

    env_cfg.scene.external_cam = world_cam("external_cam", preset)
    if preset2:
        env_cfg.scene.external_cam2 = world_cam("external_cam2", preset2)

    if wrist:
        # Parented to the fixed jaw (`gripper` link; `jaw` is the moving one), like a real
        # wrist cam bracket. In that link's frame the jaw tip is along -y (the task's
        # ee_frame offset is pos=[0, -0.09, 0.01]), so the camera looks down -y.
        env_cfg.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + gripper_body + "/wrist_cam",
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length_for_hfov(WRIST_HFOV),  # wrist cams are wide
                focus_distance=400.0,
                horizontal_aperture=HORIZONTAL_APERTURE,
                clipping_range=(0.005, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=WRIST_CAM_POS,
                rot=wrist_cam_quat(WRIST_CAM_TILT_DEG),
                convention="ros",
            ),
        )

    print(
        f"[scene_match] cameras: external={width}x{height} @ {cfg['hfov']:.0f} deg hfov "
        f"(preset '{preset}'), external2={preset2 or 'off'}, wrist={'on' if wrist else 'off'}"
    )
    return env_cfg


def aim_external_cam(env, preset=DEFAULT_PRESET, cam_name="external_cam"):
    """Point a third-person camera at the workspace. Call after env.reset()."""
    import torch

    cfg = CAMERA_PRESETS[preset]
    device = env.unwrapped.device
    eye = torch.tensor([cfg["eye"]], device=device)
    target = torch.tensor([cfg["target"]], device=device)
    env.unwrapped.scene[cam_name].set_world_poses_from_view(eyes=eye, targets=target)

    dx, dy, dz = (e - t for e, t in zip(cfg["eye"], cfg["target"]))
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    elev = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    print(
        f"[scene_match] {cam_name} preset '{preset}': {dist:.2f} m out, "
        f"{elev:.0f} deg above -- {cfg['note']}"
    )
    return cfg


# --- arm color -------------------------------------------------------------------------
#
# so_arm100.urdf defines two materials:
#   "3d_printed" rgba(1.0, 0.82, 0.12)  <- saturated yellow, the OOD one
#   "sts3215"    rgba(0.1, 0.1, 0.1)    <- black servos, already realistic
# We repaint at runtime on the USD stage instead of editing the vendored URDF, so the
# external isaac_so_arm101 checkout stays clean and no URDF->USD reconversion is needed.

ARM_COLORS = {
    "white": (0.88, 0.88, 0.86),  # off-white PLA; pure 1.0 blows out under the dome light
    "black": (0.05, 0.05, 0.05),
    "yellow": (1.0, 0.82, 0.12),  # the URDF default, for A/B
}

# UsdPreviewSurface uses diffuseColor; MDL (OmniPBR) uses diffuse_color_constant.
_DIFFUSE_INPUTS = ("diffuseColor", "diffuse_color_constant", "base_color_constant")
_ROUGHNESS_INPUTS = ("roughness", "reflection_roughness_constant")


def _get_stage():
    try:
        import isaacsim.core.utils.stage as stage_utils

        return stage_utils.get_current_stage()
    except Exception:
        import omni.usd

        return omni.usd.get_context().get_stage()


def _shaders_under(stage, path_contains="/Robot"):
    from pxr import UsdShade

    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        if path_contains and path_contains not in str(prim.GetPath()):
            continue
        yield UsdShade.Shader(prim)


def list_arm_materials(path_contains="/Robot"):
    """Print every shader under the robot and its current diffuse colour.

    Run this first if a recolor silently does nothing -- the URDF importer may have
    sanitized the material names differently than we expect.
    """
    stage = _get_stage()
    found = []
    for shader in _shaders_under(stage, path_contains):
        path = str(shader.GetPath())
        color = None
        for name in _DIFFUSE_INPUTS:
            inp = shader.GetInput(name)
            if inp and inp.Get() is not None:
                color = (name, tuple(round(float(c), 3) for c in inp.Get()))
                break
        found.append((path, color))
        print(f"[scene_match]   {path}  ->  {color}")
    if not found:
        print(f"[scene_match] no shaders found under '{path_contains}'")
    return found


def recolor_arm(
    body_color="white",
    servo_color=None,
    body_match="printed",
    servo_match="sts3215",
    roughness=0.65,
    path_contains="/Robot",
):
    """Repaint the printed arm parts (and optionally the servos). Call after env.reset().

    Matching is by material-name substring, falling back to the URDF's source RGB. Note the
    substring is "printed", not "3d_printed": the URDF importer rejects a leading digit and
    logs `The path 3d_printed is not a valid usd path, modifying to a_d_printed`, so the
    literal URDF name never appears on the stage. "printed" matches either spelling, which
    also makes the recolor idempotent -- matching on the source RGB alone would fail on a
    second call, because by then the parts are no longer yellow.
    """
    from pxr import Gf

    body_rgb = ARM_COLORS.get(body_color, body_color)
    servo_rgb = ARM_COLORS.get(servo_color, servo_color) if servo_color else None

    # Fallback identification by the URDF's declared colours, in case names were mangled.
    src_colors = {"body": (1.0, 0.82, 0.12), "servo": (0.1, 0.1, 0.1)}

    def _close(a, b, tol=0.02):
        return a is not None and all(abs(float(x) - y) <= tol for x, y in zip(a, b))

    stage = _get_stage()
    touched = []

    for shader in _shaders_under(stage, path_contains):
        path = str(shader.GetPath())
        lowered = path.lower()

        diffuse_inp, current = None, None
        for name in _DIFFUSE_INPUTS:
            inp = shader.GetInput(name)
            if inp:
                diffuse_inp, current = inp, inp.Get()
                break
        if diffuse_inp is None:
            continue

        if body_match in lowered or _close(current, src_colors["body"]):
            target = body_rgb
        elif servo_rgb is not None and (servo_match in lowered or _close(current, src_colors["servo"])):
            target = servo_rgb
        else:
            continue

        diffuse_inp.Set(Gf.Vec3f(*target))
        if roughness is not None:
            for rname in _ROUGHNESS_INPUTS:
                rinp = shader.GetInput(rname)
                if rinp:
                    rinp.Set(float(roughness))  # matte PLA, not shiny plastic
                    break
        touched.append(path)

    if touched:
        print(f"[scene_match] recolored {len(touched)} shader(s) -> body={body_color}")
    else:
        print(
            "[scene_match] recolor matched nothing. Run list_arm_materials() to see the "
            "actual shader names/colors on this stage."
        )
    return touched
