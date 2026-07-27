"""Controller TF stability test.

Records VR controller pose data over time and plots:
  - Position (x, y, z) time-series  — raw vs One Euro filtered
  - Quaternion (x, y, z, w) time-series — raw vs filtered
  - Message arrival interval histogram (latency / jitter)
  - 3-D position trajectory — raw vs filtered

Usage:
    uv run python webxr/test_controller_stability.py --duration 30
    uv run python webxr/test_controller_stability.py --duration 30 --no-filter
    uv run python webxr/test_controller_stability.py --duration 60 --hand left \\
        --min-cutoff 1.0 --beta 0.007 --save plot.png
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from one_euro_filter import ControllerFilter
from vr_receiver import ControllerState, VRReceiver


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Record:
    """Raw and (optionally) filtered pose data for one controller."""

    t: list[float] = field(default_factory=list)           # elapsed time [s]
    # Raw
    pos_x: list[float] = field(default_factory=list)
    pos_y: list[float] = field(default_factory=list)
    pos_z: list[float] = field(default_factory=list)
    quat_x: list[float] = field(default_factory=list)
    quat_y: list[float] = field(default_factory=list)
    quat_z: list[float] = field(default_factory=list)
    quat_w: list[float] = field(default_factory=list)
    # Filtered (empty when --no-filter)
    fpos_x: list[float] = field(default_factory=list)
    fpos_y: list[float] = field(default_factory=list)
    fpos_z: list[float] = field(default_factory=list)
    fquat_x: list[float] = field(default_factory=list)
    fquat_y: list[float] = field(default_factory=list)
    fquat_z: list[float] = field(default_factory=list)
    fquat_w: list[float] = field(default_factory=list)
    # Misc
    grip: list[float] = field(default_factory=list)
    trigger: list[float] = field(default_factory=list)
    browser_t_ms: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect(
    receiver: VRReceiver,
    duration: float,
    poll_hz: float = 100.0,
    hands: list[str] = ("left", "right"),
    filter_kwargs: dict | None = None,
) -> dict[str, Record]:
    """Record controller data for *duration* seconds.

    Args:
        receiver: Running VRReceiver instance.
        duration: Recording length [s].
        poll_hz: Polling rate [Hz].
        hands: Which hands to record.
        filter_kwargs: Passed to ControllerFilter.  None disables filtering.

    Returns:
        Dict mapping hand name to Record.
    """
    records: dict[str, Record] = {h: Record() for h in hands}
    last_received: dict[str, float] = {h: 0.0 for h in hands}
    filters: dict[str, ControllerFilter | None] = {}
    for h in hands:
        filters[h] = ControllerFilter(**(filter_kwargs or {})) if filter_kwargs is not None else None

    t0 = time.monotonic()
    interval = 1.0 / poll_hz
    print(f"[test] Recording for {duration:.0f} s — move your controllers now …")

    while True:
        now = time.monotonic()
        elapsed = now - t0
        if elapsed >= duration:
            break

        for h in hands:
            state: ControllerState = receiver.get_left() if h == "left" else receiver.get_right()

            if state.received == last_received[h]:
                continue

            dt = elapsed - records[h].t[-1] if records[h].t else 0.0
            last_received[h] = state.received

            rec = records[h]
            rec.t.append(elapsed)
            rec.pos_x.append(state.position[0])
            rec.pos_y.append(state.position[1])
            rec.pos_z.append(state.position[2])
            rec.quat_x.append(state.quaternion_xyzw[0])
            rec.quat_y.append(state.quaternion_xyzw[1])
            rec.quat_z.append(state.quaternion_xyzw[2])
            rec.quat_w.append(state.quaternion_xyzw[3])
            rec.grip.append(state.grip)
            rec.trigger.append(state.trigger)
            rec.browser_t_ms.append(state.timestamp_ms)

            filt = filters[h]
            if filt is not None:
                fp, fq = filt.update(state.position, state.quaternion_xyzw, dt)
                rec.fpos_x.append(fp[0])
                rec.fpos_y.append(fp[1])
                rec.fpos_z.append(fp[2])
                rec.fquat_x.append(fq[0])
                rec.fquat_y.append(fq[1])
                rec.fquat_z.append(fq[2])
                rec.fquat_w.append(fq[3])

        remaining = interval - (time.monotonic() - now)
        if remaining > 0:
            time.sleep(remaining)

        bar_len = 40
        filled = int(bar_len * elapsed / duration)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {elapsed:5.1f}/{duration:.0f} s", end="", flush=True)

    print()
    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _arrival_intervals_ms(t: list[float]) -> np.ndarray:
    if len(t) < 2:
        return np.array([])
    return np.diff(t) * 1000.0


def _plot_channel(ax, t, raw, filtered, label, color_raw, color_filt):
    """Plot one channel raw + filtered on the same axes."""
    ax.plot(t, raw, color=color_raw, lw=0.6, alpha=0.45, label=f"{label} raw")
    if filtered:
        ax.plot(t, filtered, color=color_filt, lw=1.2, label=f"{label} filtered")


def plot_stability(
    records: dict[str, Record],
    save_path: str | None = None,
) -> None:
    hands = [h for h in ("left", "right") if h in records and len(records[h].t) > 1]
    if not hands:
        print("[test] No data recorded — was the Quest 2 connected?")
        return

    has_filter = bool(records[hands[0]].fpos_x)
    n_hands = len(hands)

    fig = plt.figure(figsize=(7 * n_hands, 22))
    title = "VR Controller TF Stability"
    if has_filter:
        title += "  (dashed = raw, solid = One Euro filtered)"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # raw colors per hand, filtered colors (brighter)
    raw_colors = {
        "left":  {"x": "#f1948a", "y": "#e67e22", "z": "#f7dc6f"},
        "right": {"x": "#85c1e9", "y": "#76d7c4", "z": "#a9cce3"},
    }
    filt_colors = {
        "left":  {"x": "#c0392b", "y": "#d35400", "z": "#d4ac0d"},
        "right": {"x": "#1a5276", "y": "#148f77", "z": "#2e4057"},
    }

    for col, hand in enumerate(hands):
        rec = records[hand]
        t = np.asarray(rec.t)
        rc = raw_colors[hand]
        fc = filt_colors[hand]
        filtered = has_filter

        # ---- 1. Position ------------------------------------------------
        ax_pos = fig.add_subplot(4, n_hands, col + 1)
        _plot_channel(ax_pos, t, rec.pos_x, rec.fpos_x if filtered else [], "x", rc["x"], fc["x"])
        _plot_channel(ax_pos, t, rec.pos_y, rec.fpos_y if filtered else [], "y", rc["y"], fc["y"])
        _plot_channel(ax_pos, t, rec.pos_z, rec.fpos_z if filtered else [], "z", rc["z"], fc["z"])
        ax_pos.set_title(f"{hand.capitalize()} — Position [m]")
        ax_pos.set_xlabel("time [s]")
        ax_pos.set_ylabel("position [m]")
        ax_pos.legend(loc="upper right", fontsize=7, ncol=2)
        ax_pos.grid(True, alpha=0.3)

        # ---- 2. Quaternion ----------------------------------------------
        ax_quat = fig.add_subplot(4, n_hands, n_hands + col + 1)
        quat_pairs = [
            ("qx", rec.quat_x, rec.fquat_x),
            ("qy", rec.quat_y, rec.fquat_y),
            ("qz", rec.quat_z, rec.fquat_z),
            ("qw", rec.quat_w, rec.fquat_w),
        ]
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (lbl, raw, fraw) in enumerate(quat_pairs):
            c = prop_cycle[i % len(prop_cycle)]
            ax_quat.plot(t, raw, color=c, lw=0.6, alpha=0.4, label=f"{lbl} raw")
            if filtered:
                ax_quat.plot(t, fraw, color=c, lw=1.2, label=f"{lbl} filt")
        # Quaternion norm sanity
        norms = np.sqrt(
            np.array(rec.quat_x)**2 + np.array(rec.quat_y)**2
            + np.array(rec.quat_z)**2 + np.array(rec.quat_w)**2
        )
        ax_quat.plot(t, norms, color="gray", lw=0.6, linestyle=":", label="|q|")
        ax_quat.set_title(f"{hand.capitalize()} — Quaternion (xyzw)")
        ax_quat.set_xlabel("time [s]")
        ax_quat.legend(loc="upper right", fontsize=7, ncol=2)
        ax_quat.grid(True, alpha=0.3)

        # ---- 3. Arrival interval histogram ------------------------------
        ax_hist = fig.add_subplot(4, n_hands, 2 * n_hands + col + 1)
        intervals = _arrival_intervals_ms(rec.t)
        if len(intervals) > 0:
            ax_hist.hist(intervals, bins=60, color=fc["x"], edgecolor="white", lw=0.3)
            median_ms = float(np.median(intervals))
            p95_ms = float(np.percentile(intervals, 95))
            ax_hist.axvline(median_ms, color="black", linestyle="--", lw=1.0,
                            label=f"median {median_ms:.1f} ms")
            ax_hist.axvline(p95_ms, color="red", linestyle=":", lw=1.0,
                            label=f"p95 {p95_ms:.1f} ms")
            ax_hist.set_title(
                f"{hand.capitalize()} — Arrival interval\n"
                f"n={len(rec.t)} msgs  fps≈{1000/median_ms:.1f}"
            )
            ax_hist.set_xlabel("interval [ms]")
            ax_hist.set_ylabel("count")
            ax_hist.legend(fontsize=8)
            ax_hist.grid(True, alpha=0.3)

        # ---- 4. 3-D trajectory ------------------------------------------
        ax3d = fig.add_subplot(4, n_hands, 3 * n_hands + col + 1, projection="3d")
        sc = ax3d.scatter(
            rec.pos_x, rec.pos_z, rec.pos_y,
            c=t, cmap="Reds" if hand == "left" else "Blues",
            s=3, alpha=0.4, label="raw",
        )
        if filtered:
            ax3d.plot(
                rec.fpos_x, rec.fpos_z, rec.fpos_y,
                color=fc["x"], lw=1.0, alpha=0.9, label="filtered",
            )
        fig.colorbar(sc, ax=ax3d, shrink=0.45, label="time [s]")
        ax3d.set_title(f"{hand.capitalize()} — 3-D trajectory")
        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("z [m]")
        ax3d.set_zlabel("y [m]")
        if filtered:
            ax3d.legend(fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[test] Plot saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------

def print_summary(records: dict[str, Record]) -> None:
    for hand, rec in records.items():
        n = len(rec.t)
        if n < 2:
            print(f"[{hand}] no data")
            continue
        intervals = _arrival_intervals_ms(rec.t)
        pos = np.column_stack([rec.pos_x, rec.pos_y, rec.pos_z])
        print(f"\n{'='*52}")
        print(f"  {hand.upper()} controller  ({n} msgs over {rec.t[-1]:.1f} s)")
        print(f"{'='*52}")
        print(f"  Rate     : {n / rec.t[-1]:.1f} msg/s")
        print(f"  Interval : median={np.median(intervals):.1f} ms  "
              f"p95={np.percentile(intervals, 95):.1f} ms  "
              f"max={intervals.max():.1f} ms")
        print(f"  Position : x=[{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]  "
              f"y=[{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]  "
              f"z=[{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]  [m]")
        norms = np.sqrt(
            np.array(rec.quat_x)**2 + np.array(rec.quat_y)**2
            + np.array(rec.quat_z)**2 + np.array(rec.quat_w)**2
        )
        print(f"  |quat|   : mean={norms.mean():.6f}  std={norms.std():.6f}  (≈1.0)")

        if rec.fpos_x:
            raw = np.column_stack([rec.pos_x, rec.pos_y, rec.pos_z])
            flt = np.column_stack([rec.fpos_x, rec.fpos_y, rec.fpos_z])
            rmse = float(np.sqrt(np.mean((raw - flt)**2)))
            print(f"  Filter   : pos RMSE vs raw = {rmse*1000:.2f} mm")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VR controller TF stability test")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--https-port", type=int, default=8443)
    parser.add_argument("--wss-port", type=int, default=8765)
    parser.add_argument("--cert", default="webxr/cert.pem")
    parser.add_argument("--key", default="webxr/key.pem")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Recording duration [s] (default: 30)")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both",
                        help="Which controller(s) to record (default: both)")
    parser.add_argument("--save", metavar="FILE",
                        help="Save plot to FILE instead of showing interactively")
    # One Euro Filter params
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable One Euro Filter (plot raw only)")
    parser.add_argument("--min-cutoff", type=float, default=1.0,
                        help="One Euro min cutoff frequency [Hz] (default: 1.0)")
    parser.add_argument("--beta", type=float, default=0.007,
                        help="One Euro speed coefficient (default: 0.007)")
    parser.add_argument("--d-cutoff", type=float, default=1.0,
                        help="One Euro derivative cutoff [Hz] (default: 1.0)")
    args = parser.parse_args()

    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    filter_kwargs = (
        None
        if args.no_filter
        else {"min_cutoff": args.min_cutoff, "beta": args.beta, "d_cutoff": args.d_cutoff}
    )

    receiver = VRReceiver(
        host=args.host,
        https_port=args.https_port,
        wss_port=args.wss_port,
        cert=args.cert,
        key=args.key,
    )
    receiver.start()

    if filter_kwargs:
        print(f"[test] One Euro Filter: min_cutoff={args.min_cutoff}  beta={args.beta}  d_cutoff={args.d_cutoff}")

    print(f"[test] Open https://<your-ip>:{args.https_port} on Quest 2, then move controllers.")
    print("[test] Waiting 3 s for browser to connect …")
    time.sleep(3.0)

    records = collect(receiver, duration=args.duration, hands=hands, filter_kwargs=filter_kwargs)
    print_summary(records)
    plot_stability(records, save_path=args.save)


if __name__ == "__main__":
    main()
