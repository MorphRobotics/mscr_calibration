# mscr_control — Handoff / Resume Notes

Working dir: `/home/dozie/mscr_calibration/mscr_control`
Pairs **this workspace** (`mscr_calibration`: D435 tip sensing) with the
**magnetcontrol_ws** repo (`/home/dozie/magnetcontrol_ws`: UR5e + inverse magnet model).

> ⚡ **CURRENT FRONTIER.** Closed-loop circle control is built but NOT the active
> task. The live goal is: **verify the UR5e can approach the MSCR tip from camera
> feedback.** Tip sensing works, hand-eye is done (residual a bit high), and
> `approach_tip.py --move` was about to be tested when the session ended (after
> fixing an RTDE double-connection bug). RESUME THERE — run `approach_tip.py --move`.

---

## Goal
Move the UR5e (magnet on its tool flange) so the **MSCR tip** does something
useful — ultimately trace a circle (closed loop), immediately just **approach the
tip** as a visual-feedback demo. Requires three transforms:
- **base←tool0**: from RTDE `getActualTCPPose()` (always known).
- **camera←tip**: from the D435 (`mscr_tracker.py`), color+depth.
- **base←camera**: hand-eye, MEASURED by `handeye_touch.py` (the hard part).

## Hardware / environment (all verified working this session)
- **UR5e at `192.168.1.102`**, mask `255.255.255.0`. Host on `enp3s0` =
  `192.168.1.10/24`. To bring up after reboot:
  `sudo ip link set enp3s0 up; sudo ip addr add 192.168.1.10/24 dev enp3s0`
  (NOT persistent — offered netplan, not yet done). The old `192.168.56.101` in
  magnetcontrol_ws was a dead sim/docker bridge — ignore it.
- `ur_rtde` (rtde_control/rtde_receive) installed → standalone control, no ROS.
- **D435i** on USB; color+depth. Emitter off by default; ON (200) for calibration.
- **TCP attachment: 97 mm** along tool +Z (`robot.tool_tip_offset_m: [0,0,0.097]`).
  Confirmed 97 mm (NOT 97 cm — that exceeds reach). Assumes pendant TCP is at the
  FLANGE; if you set the pendant TCP to the tip instead, set the offset to [0,0,0]
  (don't double-count).
- Preflight: `python check_hardware.py` → was 6/7 (only hand-eye missing).

## Files
| file | role | status |
|------|------|--------|
| `config.yaml` | all params (robot, camera, tracker, handeye, approach, circle) | live |
| `check_hardware.py` | preflight: link/ping/RTDE/dashboard/camera/hand-eye | works |
| `handeye_touch.py` | **no-mount** eye-to-hand calib: touch board corners w/ TCP | works; residual high |
| `handeye_calibrate.py` | classic eye-to-hand (board on flange) + `find_chessboard()` | works (unused path) |
| `tip_sensor.py` | `D435Camera` + `TipSensor` (wraps `mscr_tracker`) | works |
| `transforms.py` / `se3.py` | frame composition + SE3 helpers | works |
| `inverse_model.py` | ONNX inverse magnet model wrapper | works (offline) |
| `trace_circle.py` | closed-loop circle controller (dry-run validated) | built, untested on HW |
| `approach_tip.py` | **sense tip → base → reachability → guarded move** | RESUME HERE |
| `debug_board.py` / `debug_targets.py` / `debug_tip.py` | diagnostics | works |

## Calibration (hand-eye) — DONE but residual ~25 mm
`python handeye_touch.py` (checkerboard flat on table, robot CLEAR, then touch 5
corners with the TCP tip in freedrive). Key facts learned:
- **Checkerboard is 10×9 INNER corners @ 25 mm** (config `handeye.target`). The
  classic `findChessboardCorners` is FLAKY on this rippled/grazing board — we use
  `findChessboardCornersSB` (exhaustive) everywhere via `find_chessboard()`.
- **Corner 3D comes from DEPTH** (deproject), NOT solvePnP — solvePnP was wrong at
  the grazing angle (gave 1.3× scaled points → 35 mm residual). Emitter ON for calib.
- Targets = **4 image-extreme corners + center** (`pick_targets_px`), labelled
  FAR/NEAR-LEFT/RIGHT so they're unambiguous.
- **97 mm tool offset applied orientation-aware** (tip_base = T_base_tool0·offset).
- Camera CLOSES right after detection so its stream can't crash during touches.
- Result saved to `results/T_base_camera.npz`. Residual came out ~25 mm — usable
  for a coarse approach but NOT for precise circle control. The end-of-run
  `--- diagnosis ---` block tells you if point sets are congruent. To improve:
  flatter/more head-on board, touch more precisely, confirm pendant TCP, maybe more
  points. The `cam_points`/`base_points` are saved in the npz for analysis.

## Tip sensing — WORKS (after the right setup)
`python debug_tip.py`. Hard-won lessons:
- The MSCR rod is a thin DARK object → **returns no depth of its own**. It MUST have
  a **plain backdrop 10–20 cm behind it, within `tracker.depth_max_m` (0.6 m)** so
  the tracker can range it. Without a near backdrop: arc=0, tipZ=0, valid=False.
- Rod ~25–35 cm from camera, prominent, checkerboard REMOVED (it's a huge dark
  distractor that the tracker grabbed → arc=1796 mm garbage).
- `tracker:` config overrides `mscr_tracker.TrackerParams`: depth_min/max_m,
  dark_threshold, entry.
- **Tip end is chosen by image side**, not the tracker's flaky base/tip: 
  `tracker.tip_image_side: "top"` → tip = topmost centerline point (rod points up,
  base clamped at bottom). Implemented in `TipSensor._tip_from_result`.
- A depth sanity gate in `read_tip` rejects tips outside 0.1–1.2 m (junk like the
  earlier 6.4 m reading).
- Verified good: valid=True, tipZ≈0.21 m, arc≈117 mm (CONFIRM rod is ~117 mm, not
  45 — if 45, trace runs into the stand; tip is still correct either way).

## Approach the tip — the live task
`python approach_tip.py` (report only) gave a SENSIBLE result this session:
- tip(camera) [0.034,0.012,0.236] m; tip(base) [-0.108,-0.852,0.134] m; move 473 mm;
  **within safety limits: True**.
- The `robot.workspace_min/max` box is for the MAGNET circle (negative Z) and is
  WRONG for the approach (positive Z near the rod) — so approach uses its own
  `approach.workspace_min/max` box (now passes) + `approach.max_move_m` (0.6) +
  the robot's `isPoseWithinSafetyLimits`.
- **Just fixed**: `approach_tip.py` created TWO RTDE control connections (safety
  check + move) → "RTDE input registers already in use". Now uses ONE connection,
  reused and disconnected via `cleanup()`.

### RESUME: next action
```bash
cd ~/mscr_calibration/mscr_control
python approach_tip.py --move     # report, type 'yes', slow moveL to 5 cm above tip
```
Keep the e-stop in hand. The tip is at y≈-0.85 m (near reach limit) — if jerky/faults,
slide the rod closer to the robot base and re-run. If "registers in use" persists:
`pgrep -f approach_tip` (kill leftovers), wait ~10 s; last resort check the robot has
no EtherNet/IP/PROFINET/MODBUS enabled (Installation → Fieldbus).

## After the single approach works
- Offered but NOT built: a **continuous "follow the tip"** mode (re-sense + re-command
  so moving the rod by hand makes the robot track it) — a clearer responsive demo.
- The closed-loop **circle** (`trace_circle.py`) needs: a good (low-residual) hand-eye,
  the magnet workspace (`magnet_model.cat_tip_center` etc.) reconciled to THIS cell
  (the old values are from a different base frame — robot TCP was at +z while the
  model expects magnet at z≈-0.178), and gain tuning. Dry-run works; HW untested.

## Gotchas / repeats
- D435 "Device or resource busy" / "/dev/video6 not found" = stale USB after an
  unclean exit or a `hardware_reset`. `D435Camera` now self-heals (resets + retries).
  If stuck: wait, or replug USB.
- cv2 GUI windows crash in this env → all scripts are file-based (save PNGs to
  `results/`, no `imshow`). VS Code itself got unstable with big images open.
- Everything is in `mscr_control/` and NOT yet git-committed.
