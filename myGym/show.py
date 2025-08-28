#!/usr/bin/env python3
"""
Interactive launcher for test.py based on discovered config files (e.g. train.json).

Adds inline display of evaluation results (last episode & success_rate) if an
'evaluation_results.json' file is found in the same directory as train.json.
"""
import argparse
import os
import sys
import subprocess
import json
import shutil
from dataclasses import dataclass
from typing import List, Optional, Any
from datetime import datetime

@dataclass
class Entry:
    idx: int
    path: str
    rel: str
    last_episode: Optional[int] = None
    success_rate: Optional[float] = None

DEFAULT_TARGET_NAME = "train.json"
EVAL_RESULTS_FILENAME = "evaluation_results.json"


def _filter_episode_success(eval_path: str) -> list[str]:
    lines_out = []
    try:
        with open(eval_path, 'r') as f:
            raw = f.read().strip()
        # Try whole-file JSON first
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # fallback: newline separated JSON objects
            for ln in raw.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                ep = obj.get('episode') or obj.get('id') or obj.get('iter')
                sr = obj.get('success_rate') or obj.get('successRate') or obj.get('success') or obj.get('sr')
                if ep is not None and sr is not None:
                    lines_out.append(f"episode={ep} success_rate={sr}")
            return lines_out if lines_out else ["[INFO] No episode/success_rate pairs found."]
        def handle_container(container):
            if isinstance(container, list):
                for item in container:
                    if isinstance(item, dict):
                        ep = item.get('episode') or item.get('id') or item.get('iter')
                        sr = item.get('success_rate') or item.get('successRate') or item.get('success') or item.get('sr')
                        if ep is not None and sr is not None:
                            lines_out.append(f"episode={ep} success_rate={sr}")
            elif isinstance(container, dict):
                # Direct episode/success pair
                ep = container.get('episode') or container.get('last_episode') or container.get('id') or container.get('iter')
                sr = container.get('success_rate') or container.get('successRate') or container.get('success') or container.get('sr')
                if ep is not None and sr is not None:
                    lines_out.append(f"episode={ep} success_rate={sr}")
                # Nested lists
                for key in ('episodes','history','evaluations'):
                    if key in container and isinstance(container[key], list):
                        handle_container(container[key])
                # Dict-of-evaluations pattern
                for v in container.values():
                    if isinstance(v, dict):
                        vv_ep = v.get('episode') or v.get('id') or v.get('iter')
                        vv_sr = v.get('success_rate') or v.get('successRate') or v.get('success') or v.get('sr')
                        if vv_ep is not None and vv_sr is not None:
                            lines_out.append(f"episode={vv_ep} success_rate={vv_sr}")
        handle_container(parsed)
        return lines_out if lines_out else ["[INFO] No episode/success_rate pairs found."]
    except Exception as ex:
        return [f"[ERROR] {ex}"]


def _format_mtime(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%d/%m/%Y")
    except Exception:
        return "?"


def run_visualize(train_json_path: str):
    file_dir = os.path.dirname(os.path.abspath(train_json_path))
    parent_dir = os.path.dirname(file_dir)
    target_path = parent_dir if parent_dir else file_dir
    try:
        print(f"\n[VIS] Running visualization for base path {target_path} (from {train_json_path})")
        subprocess.run([sys.executable, 'visualize_results.py', '--pth', target_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[VIS][ERROR] visualize_results.py exited with {e.returncode}")
    except FileNotFoundError:
        print("[VIS][ERROR] visualize_results.py not found.")


def run_visualize_current(train_json_path: str):
    file_dir = os.path.dirname(os.path.abspath(train_json_path))
    try:
        print(f"\n[VIS] Running visualization for path {file_dir} (no parent exclusion)")
        subprocess.run([sys.executable, 'visualize_results.py', '--pth', file_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[VIS][ERROR] visualize_results.py exited with {e.returncode}")
    except FileNotFoundError:
        print("[VIS][ERROR] visualize_results.py not found.")


def _extract_eval_metrics(eval_path: str) -> tuple[Optional[int], Optional[float]]:
    """Robustly extract last episode index and success rate from evaluation_results.json.

    Supports formats:
      1. Single dict with keys 'episodes' (list of dicts) or 'history'.
      2. Single dict with scalar 'episode' / 'last_episode' and 'success_rate'.
      3. List of episode dicts directly.
      4. Newline-delimited JSON objects (each a dict) -> use last valid line.
      5. Mixed: list of numbers for success rates under 'success_rate_history'.
    """
    try:
        with open(eval_path, 'r') as f:
            raw = f.read().strip()
        last_ep: Optional[int] = None
        success: Optional[float] = None

        def norm_ep(v):
            try:
                return int(v)
            except Exception:
                return None
        def norm_sr(v):
            try:
                return float(v)
            except Exception:
                return None

        parsed = None
        # Try full-file JSON parse first
        try:
            parsed = json.loads(raw)
        except Exception:
            # Fallback: treat as newline-delimited JSON objects
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            for ln in reversed(lines):
                try:
                    obj = json.loads(ln)
                    parsed = obj
                    break
                except Exception:
                    continue
        if parsed is None:
            return None, None

        # Case: dict
        if isinstance(parsed, dict):
            # Direct keys first
            if 'success_rate' in parsed:
                success = norm_sr(parsed.get('success_rate'))
            elif 'successRate' in parsed:
                success = norm_sr(parsed.get('successRate'))
            # Historical success list
            if success is None:
                sr_hist = parsed.get('success_rate_history') or parsed.get('successRateHistory')
                if isinstance(sr_hist, list) and sr_hist:
                    success = norm_sr(sr_hist[-1])
            # Episodes style lists
            episodes = parsed.get('episodes') or parsed.get('history') or parsed.get('evaluations')
            if isinstance(episodes, list) and episodes:
                for item in reversed(episodes):
                    if isinstance(item, dict):
                        if last_ep is None:
                            last_ep = norm_ep(item.get('episode') or item.get('id') or item.get('iter'))
                        if success is None:
                            success = norm_sr(item.get('success_rate') or item.get('success') or item.get('sr'))
                        if last_ep is not None and success is not None:
                            break
            # Fallback scalar keys
            if last_ep is None:
                last_ep = norm_ep(parsed.get('last_episode') or parsed.get('episode') or parsed.get('epoch'))
            # NEW: handle dictionary-of-evaluations pattern (keys like evaluation_after_XXXX_steps)
            if last_ep is None and success is None:
                eval_blocks = []
                for k, v in parsed.items():
                    if isinstance(v, dict) and ('episode' in v or 'success_rate' in v or 'successRate' in v):
                        ep_candidate = norm_ep(v.get('episode'))
                        sr_candidate = norm_sr(v.get('success_rate') or v.get('successRate'))
                        if ep_candidate is not None:
                            eval_blocks.append((ep_candidate, sr_candidate))
                if eval_blocks:
                    # pick the block with the largest episode number
                    eval_blocks.sort(key=lambda x: (x[0] is None, x[0]))  # None episodes last
                    last_ep, success_candidate = eval_blocks[-1]
                    if success is None:
                        success = success_candidate
        # Case: list
        elif isinstance(parsed, list) and parsed:
            # Assume list of dicts or list of scalar success rates
            for item in reversed(parsed):
                if isinstance(item, dict):
                    if last_ep is None:
                        last_ep = norm_ep(item.get('episode') or item.get('id') or item.get('iter'))
                    if success is None:
                        success = norm_sr(item.get('success_rate') or item.get('success') or item.get('sr'))
                    if last_ep is not None and success is not None:
                        break
                else:  # scalar
                    if success is None:
                        success = norm_sr(item)
                        # episode index maybe implicit length
                        last_ep = len(parsed)
                        break
        return last_ep, success
    except Exception:
        return None, None


def scan(root: str, target_name: str, recursive: bool) -> List[Entry]:
    entries: List[Entry] = []
    i = 0
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            if target_name in filenames:
                full = os.path.join(dirpath, target_name)
                rel = os.path.relpath(full, root)
                eval_path = os.path.join(dirpath, EVAL_RESULTS_FILENAME)
                last_ep, succ = _extract_eval_metrics(eval_path) if os.path.isfile(eval_path) else (None, None)
                entries.append(Entry(i, full, rel, last_ep, succ))
                i += 1
    else:
        for f in os.listdir(root):
            full_dir = os.path.join(root, f)
            if os.path.isdir(full_dir):
                full = os.path.join(full_dir, target_name)
                if os.path.isfile(full):
                    eval_path = os.path.join(full_dir, EVAL_RESULTS_FILENAME)
                    last_ep, succ = _extract_eval_metrics(eval_path) if os.path.isfile(eval_path) else (None, None)
                    rel = os.path.relpath(full, root)
                    entries.append(Entry(i, full, rel, last_ep, succ))
                    i += 1
    return entries


def run_test(config_path: str, dry: bool, extra: List[str], g_value: Optional[str]):
    cmd = [sys.executable, 'test.py', '--config', config_path]
    # Only add -g if not already provided in extra
    if all(arg != '-g' for arg in extra) and g_value is not None:
        cmd += ['-g', g_value]
    cmd += extra
    print(f"\nRunning: {' '.join(cmd)}")
    if dry:
        print("(dry-run) Not executing.")
        return
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"test.py exited with non-zero status {e.returncode}")


def cli_select(entries: List[Entry]) -> Optional[Entry]:
    if not entries:
        print("No train.json files found.")
        return None
    print("Discovered configurations:")
    for e in entries:
        ep = _format_steps(e.last_episode)
        sr = _format_success(e.success_rate)
        dir_path = os.path.dirname(e.rel)
        dt = _format_mtime(e.path)
        print(f"[{e.idx}] {dir_path}  ({dt} {ep} {sr})")
    print("Type index to run, 's <index>' to show that evaluation_results.json, 'v <index>' to visualize, 'c <index>' to visualize (no parent), 'd <index>' to delete directory, or q to quit.")
    while True:
        raw = input("Select index / command: ").strip().lower()
        if not raw:
            continue
        if raw in ('q','quit','exit'):
            print("Exiting.")
            sys.exit(0)
        if raw.startswith('s'):
            parts = raw.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 0 <= idx < len(entries):
                    entry = entries[idx]
                    eval_path = os.path.join(os.path.dirname(entry.path), EVAL_RESULTS_FILENAME)
                    if os.path.isfile(eval_path):
                        print(f"--- {eval_path} (episode & success_rate) ---")
                        for line in _filter_episode_success(eval_path):
                            print(line)
                        print("--- end ---")
                    else:
                        print("[INFO] evaluation_results.json not found for this entry.")
                else:
                    print("Invalid index for show command.")
            else:
                print("Usage: s <index>")
            continue
        if raw.startswith('v'):
            parts = raw.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 0 <= idx < len(entries):
                    run_visualize(entries[idx].path)
                else:
                    print("Invalid index for visualize command.")
            else:
                print("Usage: v <index>")
            continue
        if raw.startswith('c'):
            parts = raw.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 0 <= idx < len(entries):
                    run_visualize_current(entries[idx].path)
                else:
                    print("Invalid index for visualize command.")
            else:
                print("Usage: c <index>")
            continue
        if raw.startswith('d'):
            parts = raw.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 0 <= idx < len(entries):
                    entry = entries[idx]
                    target_dir = os.path.dirname(entry.path)
                    confirm = input(f"Delete directory '{target_dir}'? This cannot be undone! (y/N): ").strip().lower()
                    if confirm == 'y':
                        try:
                            shutil.rmtree(target_dir)
                            print(f"Deleted {target_dir}")
                            # remove from list and reindex
                            entries[:] = [e for e in entries if os.path.exists(e.path)]
                            for i, e in enumerate(entries):
                                e.idx = i
                            # reprint list
                            print("Updated configurations:")
                            for e in entries:
                                ep = _format_steps(e.last_episode)
                                sr = _format_success(e.success_rate)
                                dir_path = os.path.dirname(e.rel)
                                dt = _format_mtime(e.path) if os.path.exists(e.path) else 'DELETED'
                                print(f"[{e.idx}] {dir_path}  ({dt} {ep} {sr})")
                        except Exception as ex:
                            print(f"[ERROR] Failed to delete: {ex}")
                else:
                    print("Invalid index for delete command.")
            else:
                print("Usage: d <index>")
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
        print("curses not available, falling back to CLI selection.")
        return cli_select(entries)
    if not entries:
        return None

    filter_text = ""
    filtered = entries
    pos = min(start_idx, max(0, len(entries)-1))

    def apply_filter():
        nonlocal filtered, pos
        if not filter_text:
            filtered = entries
        else:
            ft = filter_text.lower()
            filtered = [e for e in entries if ft in e.rel.lower()]
        if pos >= len(filtered):
            pos = max(0, len(filtered)-1)

    def draw(stdscr):
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0,0, f"Config selector - {len(filtered)}/{len(entries)} (filter: '{filter_text}')")
        stdscr.addstr(1,0, "Arrows: navigate  Enter: run  r: run+record  s: show eval  v: visualize  c: visualize(cur)  d: delete  /: start filter  BKSP: delete  ESC: clear filter  q: quit")
        max_visible = h - 3
        start = 0
        if pos >= max_visible:
            start = pos - max_visible + 1
        for i, e in enumerate(filtered[start:start+max_visible]):
            line_idx = start + i
            prefix = '>' if line_idx == pos else ' '
            ep = _format_steps(e.last_episode)
            sr = _format_success(e.success_rate)
            dir_path = os.path.dirname(e.rel)
            dt = _format_mtime(e.path)
            display = f"{prefix} [{e.idx}] {dir_path}  ({dt} {ep} {sr})"
            if line_idx == pos:
                stdscr.attron(curses.A_REVERSE)
                stdscr.addstr(i+3, 0, display[:w-1])
                stdscr.attroff(curses.A_REVERSE)
            else:
                stdscr.addstr(i+3, 0, display[:w-1])
        stdscr.refresh()

    def show_eval(stdscr, entry):
        stdscr.clear()
        eval_path = os.path.join(os.path.dirname(entry.path), EVAL_RESULTS_FILENAME)
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0,0, f"Eval file: {eval_path}"[:w-1])
        if os.path.isfile(eval_path):
            lines = _filter_episode_success(eval_path)
        else:
            lines = ["[INFO] evaluation_results.json not found."]
        max_body = h - 2
        for i, line in enumerate(lines[:max_body]):
            stdscr.addstr(1+i, 0, line[:w-1])
        if len(lines) > max_body:
            stdscr.addstr(h-1, 0, f"-- truncated {len(lines)-max_body} more lines -- press any key --"[:w-1])
        else:
            stdscr.addstr(h-1, 0, "Press any key to return"[:w-1])
        stdscr.refresh()
        stdscr.getch()

    def loop(stdscr):
        nonlocal pos, filter_text, entries
        curses.curs_set(0)
        while True:
            apply_filter()
            draw(stdscr)
            ch = stdscr.getch()
            if ch in (ord('q'), 27):
                if ch == 27 and filter_text:
                    filter_text = ""
                    continue
                return None, False
            elif ch in (curses.KEY_DOWN, ord('j')):
                if pos < len(filtered)-1:
                    pos += 1
            elif ch in (curses.KEY_UP, ord('k')):
                if pos > 0:
                    pos -= 1
            elif ch == ord('\n'):
                if filtered:
                    return filtered[pos], False
            elif ch == ord('r'):
                if filtered:
                    return filtered[pos], True
            elif ch == ord('s'):
                if filtered:
                    show_eval(stdscr, filtered[pos])
            elif ch == ord('v'):
                if filtered:
                    curses.endwin()
                    run_visualize(filtered[pos].path)
                    stdscr = curses.initscr()
                    curses.curs_set(0)
            elif ch == ord('c'):
                if filtered:
                    curses.endwin()
                    run_visualize_current(filtered[pos].path)
                    stdscr = curses.initscr()
                    curses.curs_set(0)
            elif ch == ord('d'):
                if filtered:
                    entry = filtered[pos]
                    target_dir = os.path.dirname(entry.path)
                    stdscr.clear()
                    stdscr.addstr(0,0, f"Delete '{target_dir}'? y/N: ")
                    stdscr.refresh()
                    c2 = stdscr.getch()
                    if c2 in (ord('y'), ord('Y')):
                        try:
                            curses.endwin()
                            shutil.rmtree(target_dir)
                            print(f"Deleted {target_dir}")
                        except Exception as ex:
                            print(f"[ERROR] Failed to delete: {ex}")
                        # rebuild entries list excluding deleted ones
                        entries = [e for e in entries if os.path.exists(e.path)]
                        for i, e2 in enumerate(entries):
                            e2.idx = i
                        # reset filtered & pos
                        filter_text = ""
                        pos = 0
                        stdscr = curses.initscr()
                        curses.curs_set(0)
            elif ch == ord('/'):
                filter_text = ""
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                if filter_text:
                    filter_text = filter_text[:-1]
            elif 32 <= ch <= 126:
                filter_text += chr(ch)

    selected, record_mode = curses.wrapper(loop)
    if selected is None:
        return None
    # attach attribute for record_mode
    setattr(selected, 'record_mode', record_mode)
    return selected


def _format_steps(steps: Optional[int]) -> str:
    if steps is None:
        return "Steps:-"
    return "Steps:" + format(steps, ",").replace(",", ".")


def _format_success(sr: Optional[float]) -> str:
    if sr is None:
        return "Success:-"
    return f"Success:{sr:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./trained_models', help='Root directory to scan')
    parser.add_argument('--name', default="train.json", help='Exact filename to match (default train.json)')
    parser.add_argument('--no-recursive', action='store_true', help='Disable recursive search')
    parser.add_argument('--no-curses', action='store_true', help='Force CLI mode (disable curses)')
    parser.add_argument('--dry-run', action='store_true', help='Do not execute test.py, just show command')
    parser.add_argument('--no-loop', action='store_true', help='Run once and exit (loop is default)')
    parser.add_argument('--once', action='store_true', help='Select & run first entry automatically (useful for scripting)')
    parser.add_argument('--print-only', action='store_true', help='Print discovered paths and exit')
    parser.add_argument('--order', choices=['alpha','date'], default='alpha', help='Order listing: alpha or date (newest first)')
    parser.add_argument('--no-gui', action='store_true', help='Run commands with GUI disabled (adds -g 0)')
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='Extra args passed to test.py after --')
    args = parser.parse_args()

    recursive = not args.no_recursive
    entries = scan(args.root, args.name, recursive)
    # ordering of entries
    if args.order == 'alpha':
        entries.sort(key=lambda e: e.rel.lower())
    else:  # date
        entries.sort(key=lambda e: os.path.getmtime(e.path), reverse=True)
    # reindex after sort
    for i, e in enumerate(entries):
        e.idx = i

    if args.print_only:
        for e in entries:
            print(e.path)
        return
    if not entries:
        print("No matching files found.")
        return

    if args.once:
        sel = entries[0]
        print(f"Auto-selecting first: {sel.rel}")
        g_value = '0' if args.no_gui else '1'
        run_test(sel.path, args.dry_run, args.extra, g_value)
        return

    last_index = 0
    while True:
        record_mode = False
        if args.no_curses:
            selected = cli_select(entries)
        else:
            selected = curses_select(entries, start_idx=last_index)
        if not selected:
            print("No selection made. Exiting.")
            return
        record_mode = getattr(selected, 'record_mode', False)
        last_index = selected.idx  # remember selection for next iteration
        ep = _format_steps(selected.last_episode)
        sr = _format_success(selected.success_rate)
        dt = _format_mtime(selected.path)
        print(f"Selected: [{selected.idx}] {os.path.dirname(selected.rel)}  ({dt} {ep} {sr})")
        extra_args = list(args.extra)
        if record_mode:
            if '--record' not in extra_args:
                extra_args += ['--record', '2']
            if '--eval_episodes' not in extra_args:
                extra_args += ['--eval_episodes', '5']
            if '--camera' not in extra_args:
                extra_args += ['--camera', '1']
        g_value = '0' if args.no_gui else '1'
        run_test(selected.path, args.dry_run, extra_args, g_value)
        if args.no_loop:
            break

if __name__ == "__main__":
    main()
