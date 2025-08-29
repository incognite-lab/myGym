#!/usr/bin/env python3
"""
taskchecker.py
Lightweight interactive selector for task/config JSON files.

Default:
  - Scans the ./configs directory (non-recursive) for *.json
  - Lets you pick one (CLI or curses UI)
  - On Enter runs: python test.py --config <json_path> -g 1

Options:
  --root DIR        Directory to scan (default ./configs)
  --no-curses       Force plain CLI mode
  --dry-run         Show the command but do not execute
  --order alpha|date
  --no-gui          Use -g 0 instead of -g 1
  --print-only      Just list found JSON files
  Any extra args after '--' are appended to test.py invocation.
"""

import os
import sys
import argparse
import subprocess
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Entry:
    idx: int
    path: str
    rel: str
    mtime: float
    # dynamic flag (set at runtime if 'oracle' launch requested)
    oracle: bool = False

def scan_configs(root: str) -> List[Entry]:
    entries: List[Entry] = []
    if not os.path.isdir(root):
        return entries
    for i, name in enumerate(sorted(os.listdir(root))):
        if not name.lower().endswith(".json"):
            continue
        full = os.path.join(root, name)
        if os.path.isfile(full):
            entries.append(Entry(len(entries), full, name, os.path.getmtime(full)))
    return entries

def fmt_date(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"

def run_test(config_path: str, dry: bool, g_value: str, extra: List[str]):
    cmd = [sys.executable, "test.py", "--config", config_path, "-g", g_value]
    cmd += extra
    print(f"\nRunning: {' '.join(cmd)}")
    if dry:
        print("(dry-run) Not executing.")
        return
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] test.py exited with status {e.returncode}")

def cli_select(entries: List[Entry]) -> Optional[Entry]:
    print("Discovered JSON configs:")
    for e in entries:
        print(f"[{e.idx}] {e.rel}  ({fmt_date(e.mtime)})")
    print("Commands: <index>=run, o <index>=run with -ct oraculum, q=quit.")
    while True:
        raw = input("Select: ").strip()
        lower = raw.lower()
        if lower in ("q", "quit", "exit"):
            return None
        parts = raw.split()
        if len(parts) == 2 and parts[0].lower() == 'o' and parts[1].isdigit():
            idx = int(parts[1])
            if 0 <= idx < len(entries):
                sel = entries[idx]
                sel.oracle = True
                return sel
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(entries):
                return entries[idx]
        print("Invalid selection.")

def curses_select(entries: List[Entry], start_idx: int = 0) -> Optional[Entry]:
    try:
        import curses  # type: ignore
    except Exception:
        return cli_select(entries)

    def loop(stdscr):
        pos = min(start_idx, max(0, len(entries)-1))
        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            stdscr.addstr(0, 0, f"Configs ({len(entries)}) - Enter: run  o: run -ct oraculum  q: quit")
            max_visible = h - 2
            start = 0
            if pos >= max_visible:
                start = pos - max_visible + 1
            for i, e in enumerate(entries[start:start+max_visible]):
                line_idx = start + i
                prefix = ">" if line_idx == pos else " "
                line = f"{prefix} [{e.idx}] {e.rel} ({fmt_date(e.mtime)})"
                if line_idx == pos:
                    stdscr.attron(curses.A_REVERSE)
                    stdscr.addstr(i+1, 0, line[:w-1])
                    stdscr.attroff(curses.A_REVERSE)
                else:
                    stdscr.addstr(i+1, 0, line[:w-1])
            stdscr.refresh()
            ch = stdscr.getch()
            if ch in (ord('q'), 27):
                return None
            if ch in (curses.KEY_DOWN, ord('j')):
                if pos < len(entries)-1:
                    pos += 1
            elif ch in (curses.KEY_UP, ord('k')):
                if pos > 0:
                    pos -= 1
            elif ch in (10, 13):  # Enter
                return entries[pos]
            elif ch in (ord('o'), ord('O')):
                sel = entries[pos]
                sel.oracle = True
                return sel

    return curses.wrapper(loop)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./configs", help="Directory with JSON config files (default ./configs)")
    ap.add_argument("--no-curses", action="store_true", help="Disable curses UI (use simple CLI)")
    ap.add_argument("--dry-run", action="store_true", help="Do not execute test.py")
    ap.add_argument("--order", choices=["alpha","date"], default="alpha", help="Order listing")
    ap.add_argument("--no-gui", action="store_true", help="Use -g 0 instead of -g 1")
    ap.add_argument("--print-only", action="store_true", help="Print found JSONs and exit")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to test.py after --")
    args = ap.parse_args()

    entries = scan_configs(args.root)
    if args.order == "date":
        entries.sort(key=lambda e: e.mtime, reverse=True)
        for i, e in enumerate(entries):
            e.idx = i

    if args.print_only:
        for e in entries:
            print(e.path)
        return

    if not entries:
        print("No JSON configs found.")
        return

    # Selection
    selected = cli_select(entries) if args.no_curses else curses_select(entries)
    if not selected:
        print("No selection made.")
        return

    g_val = "0" if args.no_gui else "1"
    extra_args = list(args.extra)
    if selected.oracle:
        # Insert oracle parameters if not already present
        if '-ct' not in extra_args and '--ct' not in extra_args:
            extra_args += ['-ct', 'oraculum',]
        if '-ba' not in extra_args and '--ba' not in extra_args:
            extra_args += ['-ba', 'absolute_gripper',]
    run_test(selected.path, args.dry_run, g_val, extra_args)

if __name__ == "__main__":
    main()