"""Push the sim scene toward SmolVLA's SO-100/101 training distribution (SO-ARM101 copy).

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


# --- arm variants ------------------------------------------------------------------------
# SO-100 and SO-101 differ in three ways this module cares about:
#   * task id and link naming (`gripper` vs `gripper_link`)
#   * base spawn orientation: SO-100 is rotated +90 deg about z, SO-101 is unrotated
#   * which way the jaw points in the gripper link's own frame, which sets the wrist cam
#     aim. The lift task's ee_frame offset gives it away: SO-100 uses [0, -0.09, 0.01]
#     (jaw along -y), SO-101 uses [0.01, 0, -0.09] (jaw along -z).
#
# wrist_up is not a guess: it is -1 * the gripper joint's hinge axis, expressed in the
# gripper link's frame (rotate the joint's child-frame axis (0,0,1) by its URDF rpy).
# Sitting the camera on the hinge axis is what makes the two jaws appear side by side
# instead of one filling the frame. SO-100 rpy (0, pi, 0) -> hinge -z -> up +z; SO-101
# rpy (pi/2, 0, 0) -> hinge -y -> up +y.

ARMS = {
    "so100": dict(
        task="Isaac-SO-ARM100-Lift-Cube-Play-v0",
        gripper_body="gripper",
        wrist_forward=(0.0, -1.0, 0.0),
        wrist_up=(0.0, 0.0, 1.0),
    ),
    "so101": dict(
        task="Isaac-SO-ARM101-Lift-Cube-Play-v0",
        gripper_body="gripper_link",
        wrist_forward=(0.0, 0.0, -1.0),
        wrist_up=(0.0, 1.0, 0.0),
    ),
}
DEFAULT_ARM = "so101"


# --- wrist camera ----------------------------------------------------------------------
# Mounted on the gripper link, ahead of and above the jaws, tilted down so the grasp point
# lands near frame centre. At 0 deg tilt the jaws sat in the bottom corners and most of the
# frame was empty table (2026-08-03 firstpass preview).

WRIST_FWD_M = 0.02   # metres toward the jaw along `wrist_forward`
WRIST_UP_M = 0.045   # metres above the jaw along `wrist_up`
WRIST_CAM_TILT_DEG = 20.0
WRIST_HFOV = 85.0


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _mat_to_quat(x, y, z):
    """Rotation matrix given as its columns x|y|z -> quaternion (w, x, y, z)."""
    m = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


def look_quat(forward, up, tilt_deg=0.0):
    """ros-convention quat (w, x, y, z): look along `forward`, with `up` upright in frame.

    convention="ros" means the camera views along its own +Z, +Y is image-down and +X is
    image-right. `tilt_deg` rotates the view down toward -up, which is how the grasp point
    gets centred. Generalizes the old -y-only SO-100 derivation so SO-101 (jaw along -z)
    works from the same code.
    """
    f = _unit(forward)
    u = _unit(up)
    u = _unit(tuple(u[i] - _dot(u, f) * f[i] for i in range(3)))  # orthogonalize against f
    t = math.radians(tilt_deg)
    ct, st = math.cos(t), math.sin(t)
    z = _unit(tuple(f[i] * ct - u[i] * st for i in range(3)))     # view axis, tilted down
    up_t = _unit(tuple(f[i] * st + u[i] * ct for i in range(3)))
    y = tuple(-c for c in up_t)                                   # image-down
    x = _cross(y, z)
    return _mat_to_quat(x, y, z)


def wrist_cam_pose(arm=DEFAULT_ARM, tilt_deg=WRIST_CAM_TILT_DEG):
    """(pos, quat) for the wrist camera in the gripper link's frame, for either arm."""
    a = ARMS[arm]
    f, u = _unit(a["wrist_forward"]), _unit(a["wrist_up"])
    pos = tuple(f[i] * WRIST_FWD_M + u[i] * WRIST_UP_M for i in range(3))
    return pos, look_quat(f, u, tilt_deg)


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
    arm=DEFAULT_ARM,
    gripper_body=None,
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
        # Parented to the fixed jaw (the moving jaw is a separate link), like a real wrist
        # cam bracket. Which way it looks depends on the arm -- see ARMS / wrist_cam_pose.
        body = gripper_body or ARMS[arm]["gripper_body"]
        wrist_pos, wrist_rot = wrist_cam_pose(arm, WRIST_CAM_TILT_DEG)
        env_cfg.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body + "/wrist_cam",
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
                pos=wrist_pos,
                rot=wrist_rot,
                convention="ros",
            ),
        )

    print(
        f"[scene_match] cameras: external={width}x{height} @ {cfg['hfov']:.0f} deg hfov "
        f"(preset '{preset}'), external2={preset2 or 'off'}, "
        f"wrist={'on (' + arm + ')' if wrist else 'off'}"
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
