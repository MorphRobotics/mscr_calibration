# mscr_control — closed-loop MSCR tip circle via UR5e magnet

Drive the UR5e (with the magnet on its tool flange) so the **MSCR tip traces a
circle**, using live D435 tip feedback, a hand-eye transform, and the trained
inverse magnet model from `magnetcontrol_ws`.

This pairs **this workspace** (`mscr_calibration`: D435 tip sensing via
`mscr_tracker.py`) with the **magnetcontrol_ws** repo (UR5e + inverse model).

## Pieces
| file | role |
|------|------|
| `config.yaml` | all parameters (robot IP, model paths, circle, gains, **safety box**) |
| `se3.py` | SE(3) / pose helpers (UR rotvec ↔ matrix) |
| `inverse_model.py` | ONNX inverse model wrapper: desired tip-delta → magnet (TCP) position |
| `tip_sensor.py` | D435 color+depth stream + `MSCRTracker` → MSCR tip (camera frame, m) |
| `transforms.py` | compose hand-eye + tip + TCP → tip in the robot base frame |
| `handeye_calibrate.py` | eye-to-hand calibration (ChArUco) → `results/T_base_camera.npz` |
| `trace_circle.py` | the closed-loop circle controller (RTDE) |

## The three transforms (your explicit ask)
- **base ← tool0**  — from RTDE `getActualTCPPose()` (robot kinematics, always known).
- **camera ← tip**  — from `MSCRTracker` (`tip_xyz_mm`, color-camera frame).
- **base ← camera** — the missing one; **measured** by `handeye_calibrate.py`
  (it cannot be computed from nothing). The camera is fixed in the cell
  (eye-to-hand), so this is a constant. With it, `tip_base = T_base_camera · tip_camera`.

## Setup assumptions (verify for your rig)
1. **Magnet is on the UR5e tool flange** (the inverse model commands TCP pose = magnet
   position, matching `mscr_inv_control.py`).
2. **D435 is stationary** looking at the workspace (eye-to-hand).
3. Robot reachable at `robot.ip` (default `192.168.56.101`) via RTDE; `ur_rtde` installed.
4. Inverse model + norm come from `magnetcontrol_ws/.../magnet_control/` (paths in config).
5. The inverse model's tip-delta axes are the **base X/Y** (so default circle plane =
   `base_xy`), and `cat_tip_center` / `z_rel_mag` / `B_*` match `mscr_inv_control.py`.

## Run order
```bash
cd mscr_calibration/mscr_control

# 0. sanity (no hardware): prints the plan + simulated control loop
python3 trace_circle.py --dry-run

# 1. ONE-TIME camera->robot (hand-eye) calibration -> results/T_base_camera.npz
#    Pick ONE method:
#  (a) NO-MOUNT touch-point (recommended if you can't fix a board to the flange):
#      board lies fixed on the table; touch its corners with the TCP tip.
python3 handeye_touch.py            # set the pendant TCP to the pointer/magnet tip!
#  (b) Classic: temporarily clamp the checkerboard to the flange, jog varied poses.
python3 handeye_calibrate.py
#    Either way: check the printed residual < ~5 mm.

# 2. camera-only check: senses the tip, computes everything, does NOT move the robot
python3 trace_circle.py --no-move

# 3. FULL closed loop (robot moves; prints plan, asks to type 'yes')
python3 trace_circle.py
```

## Safety (read before step 3)
- **Set `robot.workspace_min/max`** in `config.yaml` to a box you trust for your
  cell. Every magnet command is **clamped** to it. The defaults bracket the model's
  magnet orbit (z ≈ −0.178 m); a wrong box silently flattens the trajectory.
- Motion is slow (`tcp_speed` 0.05 m/s); the script prints the plan and waits for
  `yes`; it always `stopScript()`s on exit. **Keep the e-stop in hand.**
- Start with a small `circle.radius_m` (default 5 mm — the tip workspace is small)
  and `revolutions: 1`.

## Tuning notes
- `control.kp` — proportional feedback on tip error. Start 0.6; raise for tighter
  tracking, lower if it oscillates. `control.lpf_alpha` smooths magnet motion.
- `control.settle_s` — the MSCR is compliant and slow; give it time to settle
  after each magnet move before re-sensing.
- The loop is **feedforward (inverse model) + proportional feedback**; gains need
  tuning on the real rig. Logs go to `results/trace_circle_log.csv`
  (desired vs measured tip, error, magnet position) for plotting.

## Known limitations
- Hand-eye must be (re)done if the camera or robot base is moved.
- The inverse model was trained for a specific MSCR/magnet; large deviations from
  its training workspace will track poorly. The `base_xy` plane is the well-trained
  one; other planes are allowed but unvalidated.
- Tip sensing (depth on a thin dark rod) can drop out; the loop retries and the
  tracker averages a few frames, but good lighting / depth help a lot.
