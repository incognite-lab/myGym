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
import shutil
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from myGym.utils.helpers import get_robot_dict

@dataclass
class Entry:
    idx: int
    path: str
    rel: str
    mtime: float
    # dynamic flags
    oracle: bool = False
    keyboard: bool = False
    slider: bool = False
    random_mode: bool = False
    selected_robot: Optional[str] = None

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
    print("Commands: <index>=run, o <index>=run with -ct oraculum, k <index>=run with -ct keyboard, s <index>=run with -ct slider, r <index>=run with -ct random, v <index>=view file, e <index>=edit file, c <index>=choose robot, q=quit.")
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
        if len(parts) == 2 and parts[0].lower() == 'k' and parts[1].isdigit():
            idx = int(parts[1])
            if 0 <= idx < len(entries):
                sel = entries[idx]
                sel.keyboard = True
                return sel
        if len(parts) == 2 and parts[0].lower() == 's' and parts[1].isdigit():
            idx = int(parts[1])
            if 0 <= idx < len(entries):
                sel = entries[idx]
                sel.slider = True
                return sel
        if len(parts) == 2 and parts[0].lower() == 'r' and parts[1].isdigit():
            idx = int(parts[1])
            if 0 <= idx < len(entries):
                sel = entries[idx]
                sel.random_mode = True
                return sel
        if len(parts) == 2 and parts[0].lower() == 'v' and parts[1].isdigit():
            idx = int(parts[1])
            if 0 <= idx < len(entries):
                path = entries[idx].path
                print(f"--- {path} ---")
                try:
                    with open(path,'r') as f:
                        print(f.read())
                except Exception as ex:
                    print(f"[ERROR] {ex}")
                print("--- end file ---")
                continue
        if len(parts) == 2 and parts[0].lower() == 'e' and parts[1].isdigit():
            idx = int(parts[1])
            if 0 <= idx < len(entries):
                path = entries[idx].path
                editor = os.environ.get('EDITOR') or ('notepad' if os.name == 'nt' else 'nano')
                # fallback chain if editor not found
                if shutil.which(editor) is None:
                    for cand in ("nano","vi","notepad"):
                        if shutil.which(cand):
                            editor = cand
                            break
                try:
                    subprocess.call([editor, path])
                    # update mtime
                    entries[idx].mtime = os.path.getmtime(path)
                    print(f"Edited {path}")
                except Exception as ex:
                    print(f"[ERROR] Editing failed: {ex}")
            continue
        if len(parts) == 2 and parts[0].lower() == 'c' and parts[1].isdigit():
            idx = int(parts[1])
            if 0 <= idx < len(entries):
                entry = entries[idx]
                robots = sorted(get_robot_dict().keys())
                print("Select robot:")
                for i, rk in enumerate(robots):
                    print(f"[{i}] {rk}")
                while True:
                    choice = input("Robot index (or q to cancel): ").strip().lower()
                    if choice in ('q','quit','exit',''):
                        break
                    if choice.isdigit():
                        r_i = int(choice)
                        if 0 <= r_i < len(robots):
                            entry.selected_robot = robots[r_i]
                            return entry
                    print("Invalid robot selection.")
            continue
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
            stdscr.addstr(0, 0, f"Configs ({len(entries)}) - Enter: run  o: oraculum  k: keyboard  s: slider  r: random  v: view  e: edit  c: choose robot  q: quit")
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
            elif ch in (curses.KEY_UP,):  # removed 'k' to free for keyboard run
                if pos > 0:
                    pos -= 1
            elif ch in (10, 13):  # Enter
                return entries[pos]
            elif ch in (ord('o'), ord('O')):
                sel = entries[pos]
                sel.oracle = True
                return sel
            elif ch in (ord('k'), ord('K')):
                sel = entries[pos]
                sel.keyboard = True
                return sel
            elif ch in (ord('s'), ord('S')):
                sel = entries[pos]
                sel.slider = True
                return sel
            elif ch in (ord('r'), ord('R')):
                sel = entries[pos]
                sel.random_mode = True
                return sel
            elif ch in (ord('v'), ord('V')):
                # show file content
                path = entries[pos].path
                stdscr.clear()
                h, w = stdscr.getmaxyx()
                try:
                    with open(path,'r') as f:
                        lines = f.readlines()
                except Exception as ex:
                    lines = [f"[ERROR] {ex}\n"]
                stdscr.addstr(0,0, f"File: {path}"[:w-1])
                max_body = h - 2
                for i, line in enumerate(lines[:max_body]):
                    stdscr.addstr(1+i, 0, line.rstrip()[:w-1])
                if len(lines) > max_body:
                    stdscr.addstr(h-1, 0, f"-- truncated {len(lines)-max_body} more lines -- press any key --"[:w-1])
                else:
                    stdscr.addstr(h-1, 0, "Press any key to return"[:w-1])
                stdscr.refresh()
                stdscr.getch()
            elif ch in (ord('e'), ord('E')):
                # edit file in external editor
                path = entries[pos].path
                editor = os.environ.get('EDITOR') or ('notepad' if os.name == 'nt' else 'nano')
                if shutil.which(editor) is None:
                    for cand in ("nano","vi","notepad"):
                        if shutil.which(cand):
                            editor = cand
                            break
                curses.endwin()
                try:
                    subprocess.call([editor, path])
                    # update mtime
                    entries[pos].mtime = os.path.getmtime(path)
                    print(f"Edited {path}")
                except Exception as ex:
                    print(f"[ERROR] Editing failed: {ex}")
                input("Press Enter to return...")
                stdscr = curses.initscr()
                curses.curs_set(0)
            elif ch in (ord('c'), ord('C')):
                # robot selection sub-view
                robot_keys = sorted(get_robot_dict().keys())
                rpos = 0
                while True:
                    stdscr.clear()
                    h, w = stdscr.getmaxyx()
                    stdscr.addstr(0,0, "Select Robot (Enter=choose, q/Esc=cancel)")
                    max_vis = h - 2
                    start_r = 0
                    if rpos >= max_vis:
                        start_r = rpos - max_vis + 1
                    for i, rk in enumerate(robot_keys[start_r:start_r+max_vis]):
                        li = start_r + i
                        pref = '>' if li == rpos else ' '
                        line = f"{pref} {rk}"
                        if li == rpos:
                            stdscr.attron(curses.A_REVERSE)
                            stdscr.addstr(i+1,0,line[:w-1])
                            stdscr.attroff(curses.A_REVERSE)
                        else:
                            stdscr.addstr(i+1,0,line[:w-1])
                    stdscr.refresh()
                    rc = stdscr.getch()
                    if rc in (ord('q'), 27):
                        break
                    elif rc in (curses.KEY_DOWN, ord('j')) and rpos < len(robot_keys)-1:
                        rpos += 1
                    elif rc in (curses.KEY_UP, ord('k')) and rpos > 0:
                        rpos -= 1
                    elif rc in (10,13):
                        entries[pos].selected_robot = robot_keys[rpos]
                        return entries[pos]
    return curses.wrapper(loop)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./configs", help="Directory with JSON config files (default ./configs)")
    ap.add_argument("--no-curses", action="store_true", help="Disable curses UI (use simple CLI)")
    ap.add_argument("--dry-run", action="store_true", help="Do not execute test.py")
    ap.add_argument("--order", choices=["alpha","date"], default="date", help="Order listing")
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

    last_index = 0
    while True:
        selected = cli_select(entries) if args.no_curses else curses_select(entries, start_idx=last_index)
        if not selected:
            print("No selection made.")
            return
        last_index = selected.idx
        g_val = "0" if args.no_gui else "1"
        extra_args = list(args.extra)
        if selected.oracle:
            if '-ct' not in extra_args and '--ct' not in extra_args:
                extra_args += ['-ct', 'oraculum']
            if '-ba' not in extra_args and '--ba' not in extra_args:
                extra_args += ['-ba', 'absolute_gripper']
        if selected.keyboard:
            if '-ct' not in extra_args and '--ct' not in extra_args:
                extra_args += ['-ct', 'keyboard']
            if '-ba' not in extra_args and '--ba' not in extra_args:
                extra_args += ['-ba', 'step_gripper']
        if selected.slider:
            if '-ct' not in extra_args and '--ct' not in extra_args:
                extra_args += ['-ct', 'slider']
            if '-ba' not in extra_args and '--ba' not in extra_args:
                extra_args += ['-ba', 'joints_gripper']
        if selected.random_mode:
            if '-ct' not in extra_args and '--ct' not in extra_args:
                extra_args += ['-ct', 'random']
        if selected.selected_robot:
            if '-b' not in extra_args and '--b' not in extra_args:
                extra_args += ['-b', selected.selected_robot]
            selected.selected_robot = None  # reset after run
        run_test(selected.path, args.dry_run, g_val, extra_args)

if __name__ == "__main__":
    main()