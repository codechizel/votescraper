"""Local HTTP server that receives parsed KanFocus vote data from the browser.

The browser JS worker fetches and parses KanFocus pages, then POSTs the
structured JSON to this server. Each vote is saved as a JSON file in the
cache directory.

Usage:
    uv run python scripts/kanfocus_receiver.py [--port PORT] [--session SESSION]

The server writes files to:
    data/kansas/{session}/.cache/kanfocus/{vote_num}_{year}_{chamber}.json
"""

import argparse
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


class VoteReceiver(BaseHTTPRequestHandler):
    cache_dir: Path  # Set by class factory

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")
            return

        # Handle batch: {"votes": {"1_2025_H": {...}, ...}}
        votes = data.get("votes", {})
        saved = 0
        for key, vote_data in votes.items():
            path = self.cache_dir / f"{key}.json"
            path.write_text(json.dumps(vote_data, ensure_ascii=False), encoding="utf-8")
            saved += 1

        # Handle status update
        status = data.get("status")
        if status:
            print(f"  [{status.get('stream', '?')}] fetched={status.get('fetched', 0)} "
                  f"votes={status.get('votes', 0)} empty={status.get('consecutiveEmpty', 0)}"
                  f"{' DONE' if status.get('done') else ''}")

        if saved:
            print(f"  Saved {saved} votes: {', '.join(sorted(votes.keys()))}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "https://kanfocus.com")
        self.end_headers()
        self.wfile.write(json.dumps({"saved": saved}).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "https://kanfocus.com")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass


def main():
    parser = argparse.ArgumentParser(description="KanFocus vote data receiver")
    parser.add_argument("--port", type=int, default=9847)
    parser.add_argument("--session", default="91st_2025-2026",
                        help="Session directory name (e.g. 91st_2025-2026)")
    args = parser.parse_args()

    base = Path("data/kansas") / args.session / ".cache" / "kanfocus"
    base.mkdir(parents=True, exist_ok=True)

    # Create handler class with cache_dir bound
    handler = type("Handler", (VoteReceiver,), {"cache_dir": base})

    server = HTTPServer(("127.0.0.1", args.port), handler)
    print(f"KanFocus receiver listening on http://127.0.0.1:{args.port}")
    print(f"Cache dir: {base}")
    print(f"Existing files: {len(list(base.glob('*.json')))}")
    print("Waiting for votes...")
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. Total cached: {len(list(base.glob('*.json')))} votes")
        server.server_close()


if __name__ == "__main__":
    main()
