#!/usr/bin/env python3
"""Minimal camera test — run this first to confirm frames arrive."""
import sys
print("Step 1: importing pyrealsense2...", flush=True)
import pyrealsense2 as rs
print("Step 2: importing numpy...", flush=True)
import numpy as np
print("Step 3: creating pipeline...", flush=True)

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
cfg.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 30)

print("Step 4: starting pipeline...", flush=True)
try:
    profile = pipeline.start(cfg)
except Exception as e:
    print(f"FAILED to start pipeline: {e}", flush=True)
    sys.exit(1)

print("Step 5: setting emitter...", flush=True)
try:
    ds = profile.get_device().first_depth_sensor()
    ds.set_option(rs.option.emitter_enabled, 1)
    ds.set_option(rs.option.laser_power, 60)
    print("  emitter ON at 60", flush=True)
except Exception as e:
    print(f"  emitter warning: {e}", flush=True)

print("Step 6: waiting for frames (will print 30 then stop)...", flush=True)
count = 0
try:
    while count < 30:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        lf = frames.get_infrared_frame(1)
        rf = frames.get_infrared_frame(2)
        if lf and rf:
            l = np.asanyarray(lf.get_data())
            r = np.asanyarray(rf.get_data())
            count += 1
            print(f"  frame {count:3d}  left={l.shape} min={l.min()} max={l.max()}"
                  f"  right={r.shape}", flush=True)
        else:
            print(f"  frame skipped (lf={bool(lf)} rf={bool(rf)})", flush=True)
finally:
    pipeline.stop()
    print(f"\nDone. Got {count} frames.", flush=True)
