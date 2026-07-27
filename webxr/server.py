"""WebXR relay server for Newton bimanual teleoperation.

Runs two services in a single async process:
  - HTTPS (port 8443): serves index.html to the Quest 2 browser
  - WSS   (port 8765): relays controller poses between browser and Newton

Usage:
    python webxr/server.py --cert cert.pem --key key.pem

Cert generation (mkcert, one-time):
    mkcert -install
    mkcert <your-local-ip>   # e.g. mkcert 192.168.0.10
    # produces <ip>+...cert.pem and <ip>+...key.pem
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import ssl
import threading
from pathlib import Path

import websockets

# ---------------------------------------------------------------------------
# WebSocket relay
# ---------------------------------------------------------------------------
_CLIENTS: set = set()


async def _ws_handler(ws) -> None:
    _CLIENTS.add(ws)
    addr = ws.remote_address
    print(f"[WSS] connected   {addr}  (total={len(_CLIENTS)})")
    try:
        async for msg in ws:
            try:
                t = json.loads(msg).get("type", "?")
            except Exception:
                t = "raw"
            print(f"[WSS] {addr} type={t} len={len(msg)}")
            others = _CLIENTS - {ws}
            if others:
                await asyncio.gather(*[c.send(msg) for c in others], return_exceptions=True)
    finally:
        _CLIENTS.discard(ws)
        print(f"[WSS] disconnected {addr}  (total={len(_CLIENTS)})")


async def _run_wss(host: str, port: int, ssl_ctx: ssl.SSLContext) -> None:
    async with websockets.serve(_ws_handler, host, port, ssl=ssl_ctx):
        print(f"[WSS] listening on wss://{host}:{port}")
        await asyncio.Future()


# ---------------------------------------------------------------------------
# HTTPS file server (serves index.html)
# ---------------------------------------------------------------------------
def _run_https(host: str, port: int, ssl_ctx: ssl.SSLContext, serve_dir: Path) -> None:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, fmt, *args):
            print(f"[HTTP] {fmt % args}")

    httpd = http.server.HTTPServer((host, port), Handler)
    httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)
    print(f"[HTTP] listening on https://{host}:{port}")
    httpd.serve_forever()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="WebXR relay server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--https-port", type=int, default=8443)
    parser.add_argument("--wss-port", type=int, default=8765)
    parser.add_argument("--cert", default="webxr/cert.pem", help="TLS certificate (PEM)")
    parser.add_argument("--key", default="webxr/key.pem", help="TLS private key (PEM)")
    args = parser.parse_args()

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(args.cert, args.key)

    serve_dir = Path(__file__).parent

    # HTTPS in a background thread (blocking HTTPServer)
    t = threading.Thread(
        target=_run_https,
        args=(args.host, args.https_port, ssl_ctx, serve_dir),
        daemon=True,
    )
    t.start()

    # WSS in the main async loop
    asyncio.run(_run_wss(args.host, args.wss_port, ssl_ctx))


if __name__ == "__main__":
    main()
