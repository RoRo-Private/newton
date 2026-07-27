"""Thread-safe WebSocket receiver for Quest 2 controller poses.

Newton example usage:

    from webxr.vr_receiver import VRReceiver

    # One Euro Filter enabled by default (recommended)
    receiver = VRReceiver(cert="webxr/cert.pem", key="webxr/key.pem")
    receiver.start()

    # each step:
    left  = receiver.get_left()    # filtered ControllerState
    right = receiver.get_right()

    left.position        # (x, y, z) in WebXR local space [m]
    left.quaternion_xyzw # (x, y, z, w)
    left.grip            # 0.0 (open) .. 1.0 (fully squeezed)

    # raw (unfiltered) access for debugging:
    left_raw = receiver.get_left(raw=True)
"""

from __future__ import annotations

import asyncio
import json
import ssl
import threading
from dataclasses import dataclass
from pathlib import Path


try:
    import websockets
except ImportError as e:
    raise ImportError("websockets package required: uv add websockets") from e


@dataclass
class ControllerState:
    """Latest state for one controller."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position in WebXR local reference space [m]."""
    quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Orientation quaternion (x, y, z, w)."""
    trigger: float = 0.0
    """Trigger (index finger) analog value: 0.0 – 1.0. Used for calibration."""
    grip: float = 0.0
    """Grip (middle finger) analog value: 0.0 = fully open, 1.0 = fully squeezed."""
    timestamp_ms: int = 0
    """Browser timestamp [ms] when the pose was captured."""
    received: float = 0.0
    """Python-side wall-clock time when the message arrived [s]."""


class VRReceiver:
    """HTTPS + WSS server that receives Quest 2 controller poses.

    Starts two services in background daemon threads:

    - HTTPS (port 8443): serves index.html to the Quest 2 browser
    - WSS   (port 8765): receives controller poses from the browser

    Call :meth:`start` once, then poll :meth:`get_left` / :meth:`get_right`
    each simulation step.  No separate server.py needed.

    One Euro Filter is applied to each incoming pose in real-time to reduce
    jitter while minimising lag during fast motion.  The filter resets
    automatically on browser reconnect.  Disable with ``filter=False`` or
    access raw poses via ``get_left(raw=True)``.

    Args:
        host: Interface to bind both servers to.
        https_port: Port for the HTTPS server (serves index.html).
        wss_port: Port for the WSS server (receives poses).
        cert: Path to TLS certificate PEM file.
        key: Path to TLS private key PEM file.
        filter: Enable One Euro Filter on incoming poses (default: True).
        min_cutoff: One Euro minimum cutoff frequency [Hz].  Lower value →
            stronger smoothing when stationary, more lag.
        beta: One Euro speed coefficient.  Higher value → less lag during
            fast motion.
        d_cutoff: Cutoff frequency [Hz] for the derivative low-pass filter.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        https_port: int = 8443,
        wss_port: int = 8765,
        cert: str | None = None,
        key: str | None = None,
        filter: bool = True,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._host = host
        self._https_port = https_port
        self._wss_port = wss_port
        self._cert = cert
        self._key = key
        self._filter_enabled = filter
        self._filter_kwargs = {"min_cutoff": min_cutoff, "beta": beta, "d_cutoff": d_cutoff}

        self._lock = threading.Lock()
        self._states: dict[str, ControllerState] = {
            "left": ControllerState(),
            "right": ControllerState(),
        }
        self._raw_states: dict[str, ControllerState] = {
            "left": ControllerState(),
            "right": ControllerState(),
        }
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start HTTPS + WSS servers in background daemon threads."""
        if self._thread is not None:
            return

        # HTTPS server (serves index.html) in its own thread
        if self._cert and self._key:
            import http.server
            serve_dir = str(Path(__file__).parent)

            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(self._cert, self._key)

            class _Handler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *a, **kw):
                    super().__init__(*a, directory=serve_dir, **kw)
                def log_message(self, *_):
                    pass

            httpd = http.server.HTTPServer((self._host, self._https_port), _Handler)
            httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)
            threading.Thread(target=httpd.serve_forever, daemon=True, name="VRReceiver-HTTPS").start()
            print(f"[VRReceiver] HTTPS on https://<your-ip>:{self._https_port} — open on Quest 2")

        # WSS server in async thread
        self._thread = threading.Thread(target=self._run, daemon=True, name="VRReceiver-WSS")
        self._thread.start()
        filt_info = (
            f"One Euro filter ON (min_cutoff={self._filter_kwargs['min_cutoff']}, "
            f"beta={self._filter_kwargs['beta']})"
            if self._filter_enabled
            else "One Euro filter OFF"
        )
        print(f"[VRReceiver] WSS on wss://{self._host}:{self._wss_port}  [{filt_info}]")

    def get_left(self, *, raw: bool = False) -> ControllerState:
        """Return a snapshot of the latest left controller state.

        Args:
            raw: If True, return the unfiltered pose (useful for debugging).

        Returns:
            ControllerState with filtered (or raw) position and orientation.
        """
        with self._lock:
            return self._raw_states["left"] if raw else self._states["left"]

    def get_right(self, *, raw: bool = False) -> ControllerState:
        """Return a snapshot of the latest right controller state.

        Args:
            raw: If True, return the unfiltered pose (useful for debugging).

        Returns:
            ControllerState with filtered (or raw) position and orientation.
        """
        with self._lock:
            return self._raw_states["right"] if raw else self._states["right"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        ssl_ctx: ssl.SSLContext | None = None
        if self._cert and self._key:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(self._cert, self._key)

        async with websockets.serve(self._handle, self._host, self._wss_port, ssl=ssl_ctx):
            await asyncio.Future()

    async def _handle(self, ws) -> None:
        import time

        print(f"[VRReceiver] browser connected {ws.remote_address}")

        # Per-connection filter instances — reset on every reconnect
        filters: dict[str, object] = {}
        if self._filter_enabled:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(__file__))
            from one_euro_filter import ControllerFilter
            filters = {
                "left": ControllerFilter(**self._filter_kwargs),
                "right": ControllerFilter(**self._filter_kwargs),
            }

        last_t: dict[str, float] = {"left": 0.0, "right": 0.0}

        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")
                if "left" in msg_type:
                    hand = "left"
                elif "right" in msg_type:
                    hand = "right"
                else:
                    continue

                p = data.get("p")
                q = data.get("q")
                if not (p and len(p) == 3 and q and len(q) == 4):
                    continue

                now = time.monotonic()
                position = (float(p[0]), float(p[1]), float(p[2]))
                quaternion = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

                raw_state = ControllerState(
                    position=position,
                    quaternion_xyzw=quaternion,
                    trigger=float(data.get("trigger", 0.0)),
                    grip=float(data.get("grip", 0.0)),
                    timestamp_ms=int(data.get("t", 0)),
                    received=now,
                )

                if self._filter_enabled:
                    dt = now - last_t[hand] if last_t[hand] > 0.0 else 0.0
                    last_t[hand] = now
                    fp, fq = filters[hand].update(position, quaternion, dt)
                    filtered_state = ControllerState(
                        position=(float(fp[0]), float(fp[1]), float(fp[2])),
                        quaternion_xyzw=(float(fq[0]), float(fq[1]), float(fq[2]), float(fq[3])),
                        trigger=raw_state.trigger,
                        grip=raw_state.grip,
                        timestamp_ms=raw_state.timestamp_ms,
                        received=now,
                    )
                else:
                    filtered_state = raw_state

                with self._lock:
                    self._raw_states[hand] = raw_state
                    self._states[hand] = filtered_state

        except Exception:
            pass
        finally:
            print(f"[VRReceiver] browser disconnected {ws.remote_address}")


# ---------------------------------------------------------------------------
# Standalone test mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="VRReceiver standalone test")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--https-port", type=int, default=8443)
    parser.add_argument("--wss-port", type=int, default=8765)
    parser.add_argument("--cert", default="webxr/cert.pem")
    parser.add_argument("--key", default="webxr/key.pem")
    parser.add_argument("--no-filter", action="store_true", help="Disable One Euro Filter")
    parser.add_argument("--min-cutoff", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.007)
    args = parser.parse_args()

    receiver = VRReceiver(
        host=args.host,
        https_port=args.https_port,
        wss_port=args.wss_port,
        cert=args.cert,
        key=args.key,
        filter=not args.no_filter,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
    )
    receiver.start()

    print("Waiting for Quest 2 — press Ctrl+C to quit\n")
    try:
        while True:
            l = receiver.get_left()
            r = receiver.get_right()
            print(
                f"\rL pos=({l.position[0]:+.3f},{l.position[1]:+.3f},{l.position[2]:+.3f})"
                f" grip={l.grip:.2f} | "
                f"R pos=({r.position[0]:+.3f},{r.position[1]:+.3f},{r.position[2]:+.3f})"
                f" grip={r.grip:.2f}",
                end="",
                flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDone.")
