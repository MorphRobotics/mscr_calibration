#!/usr/bin/env python3
"""Visualize what the MSCR tip tracker detects. Draws the detected centerline,
tip and base onto the color image (so we see it even when the tracker's own
debug overlay is empty), plus the depth-gated foreground used for detection."""
import numpy as np, cv2, yaml
from pathlib import Path
from tip_sensor import D435Camera, TipSensor

HERE = Path(__file__).parent
cfg = yaml.safe_load(open(HERE / "config.yaml"))
tcfg = cfg.get("tracker", {})
cam = D435Camera(cfg["camera"]["width"], cfg["camera"]["height"],
                 cfg["camera"]["fps"], cfg["camera"]["laser_power"])
tip = TipSensor(cam, tcfg)
out = HERE / "results"; out.mkdir(exist_ok=True)
print(f"tracker depth gate: {tcfg.get('depth_min_m',0.05)}-{tcfg.get('depth_max_m',0.8)} m, "
      f"dark_threshold={tcfg.get('dark_threshold',85)}")
print("Make sure the MSCR rod is in view and the checkerboard is REMOVED.\n")

color = depth = res = None
for i in range(10):
    color, depth = cam.get_frames()
    res = tip.tracker.process_frame(color, depth)
    cl = np.asarray(res.centerline_px).reshape(-1, 2)
    t_mm = np.asarray(res.tip_xyz_mm, float)
    print(f"frame {i}: valid={res.valid}  centerline_pts={len(cl)}  "
          f"arc={res.arc_length_mm:.0f}mm  tipZ={t_mm[2]/1000:.3f}m")
cam.close()

# draw the LAST frame's detection
vis = color.copy()
# depth gate visualization (greyed where outside the gate)
dm = depth.astype(float) * cam.depth_scale
zmin, zmax = tcfg.get("depth_min_m", 0.05), tcfg.get("depth_max_m", 0.8)
gate = ((dm > zmin) & (dm < zmax)).astype(np.uint8)
vis[gate == 0] = (vis[gate == 0] * 0.35).astype(np.uint8)   # dim out-of-gate regions

cl = np.asarray(res.centerline_px).reshape(-1, 2).astype(int)
if len(cl) >= 2:
    cv2.polylines(vis, [cl], False, (0, 255, 0), 2, cv2.LINE_AA)
    # mark the side-corrected TIP (red) chosen by tip_image_side
    side = tcfg.get("tip_image_side")
    col, row = cl[:, 0], cl[:, 1]
    end = {"top": int(np.argmin(row)), "bottom": int(np.argmax(row)),
           "left": int(np.argmin(col)), "right": int(np.argmax(col))}.get(side, -1)
    tip_px = tuple(cl[end])
    base_px = tuple(cl[0] if end == len(cl) - 1 else cl[-1])
    cv2.circle(vis, base_px, 6, (255, 0, 0), -1)           # base (blue)
    cv2.circle(vis, tip_px, 7, (0, 0, 255), -1)            # TIP (red)
    cv2.putText(vis, f"arc={res.arc_length_mm:.0f}mm  tip_side={side}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
    cv2.putText(vis, "NO ROD DETECTED", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imwrite(str(out / "tip_debug.png"), vis)
print("\nsaved -> results/tip_debug.png  (bright = inside depth gate; green = detected "
      "rod; red = tip)")
print("If 'NO ROD DETECTED' or the green line is on background, the rod isn't being "
      "isolated — tell me the arc/tipZ and what the bright (in-gate) region looks like.")
