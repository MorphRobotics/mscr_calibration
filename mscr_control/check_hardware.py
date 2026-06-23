#!/usr/bin/env python3
"""
check_hardware.py — preflight before trace_circle.py.

Checks, in order, and prints a PASS/FAIL summary:
  1. ethernet link to the robot is up (carrier on the wired iface)
  2. robot IP is pingable
  3. RTDE (30004) + dashboard (29999) TCP ports are open
  4. RTDE actually reads a TCP pose
  5. D435 enumerates and streams a color+depth frame
  6. hand-eye transform file loads

Nothing here moves the robot. Run:  python check_hardware.py
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).parent


def ok(label, passed, detail=""):
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
    return passed


def port_open(ip, port, timeout=2.0):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def main():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    ip = cfg["robot"]["ip"]
    results = []
    print(f"Preflight for UR5e @ {ip}\n")

    # 1. wired link
    link = False
    for iface in ("enp3s0", "eth0", "enp0s31f6"):
        p = Path(f"/sys/class/net/{iface}/operstate")
        if p.exists():
            state = p.read_text().strip()
            try:
                carrier = Path(f"/sys/class/net/{iface}/carrier").read_text().strip()
            except OSError:
                carrier = "?"
            link = (state == "up" and carrier == "1")
            results.append(ok(f"wired link ({iface})", link,
                              f"operstate={state} carrier={carrier}"))
            break
    else:
        results.append(ok("wired link", False, "no wired iface found"))

    # 2. ping
    pingable = subprocess.run(["ping", "-c", "1", "-W", "2", ip],
                              capture_output=True).returncode == 0
    results.append(ok(f"ping {ip}", pingable))

    # 3. ports
    rtde = port_open(ip, 30004)
    dash = port_open(ip, 29999)
    results.append(ok("RTDE port 30004", rtde))
    results.append(ok("dashboard port 29999", dash))

    # 4. RTDE read
    tcp_read = False
    if rtde:
        try:
            import rtde_receive
            r = rtde_receive.RTDEReceiveInterface(ip)
            pose = r.getActualTCPPose()
            tcp_read = pose is not None and len(pose) == 6
            results.append(ok("RTDE getActualTCPPose", tcp_read,
                              f"TCP={[round(v,3) for v in pose]}" if tcp_read else ""))
        except Exception as e:
            results.append(ok("RTDE getActualTCPPose", False, str(e)[:80]))
    else:
        results.append(ok("RTDE getActualTCPPose", False, "port closed"))

    # 5. D435
    cam_ok = False
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            results.append(ok("D435 present", False, "no RealSense device"))
        else:
            name = devs[0].get_info(rs.camera_info.name)
            from tip_sensor import D435Camera
            cam = D435Camera(cfg["camera"]["width"], cfg["camera"]["height"],
                             cfg["camera"]["fps"], cfg["camera"]["laser_power"])
            color, depth = cam.get_frames()
            cam.close()
            cam_ok = color is not None and depth is not None
            results.append(ok("D435 streams color+depth", cam_ok,
                              f"{name}, frame {color.shape}"))
    except Exception as e:
        results.append(ok("D435 streams color+depth", False, str(e)[:80]))

    # 6. hand-eye file
    he = cfg["handeye"]["transform_file"]
    he_path = HERE / he if not Path(he).is_absolute() else Path(he)
    he_ok = he_path.exists()
    results.append(ok("hand-eye transform present", he_ok,
                      str(he_path) if he_ok else "run handeye_calibrate.py"))

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    if not all(results):
        print("Resolve the FAILs above before running trace_circle.py "
              "(hand-eye can be done after the robot/camera pass).")


if __name__ == "__main__":
    main()
