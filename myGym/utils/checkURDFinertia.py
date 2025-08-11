# Usage:
#   python inertia2urdf.py <urdf_file> [--tolerance 0.15] [--package-dirs pkga_path:pkgb_path] [--write-updated out.urdf]
#
# For every <link> having <inertial><mass><inertia> and at least one <mesh>, compute inertia from the mesh
# (with optional mesh scale) using trimesh, scale to the declared mass, and compare.
# If relative difference of any inertia component (ixx, iyy, izz, ixy, ixz, iyz) exceeds tolerance,
# a corrected inertial block is proposed. If --write-updated is given, an updated URDF is written
# (only for links whose inertia differs).
#
# Notes:
# - Inertia is computed about the mesh COM and we set <origin xyz="COM">. If existing inertial origin
#   differs we treat that as discrepancy.
# - package:// URIs are resolved by searching the provided --package-dirs paths (colon separated)
#   for the first directory containing the package name as a folder root.
#
import sys
import os
import math
import argparse
import xml.etree.ElementTree as ET
from copy import deepcopy

import trimesh

# ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf_file")
    ap.add_argument("--tolerance", type=float, default=0.15,
                    help="Relative tolerance (e.g. 0.15 = 15% difference allowed)")
    ap.add_argument("--package-dirs", default="",
                    help="Colon separated roots to search for packages referenced by package:// URIs (optional; autodiscovery used if omitted)")
    ap.add_argument("--write-updated", default=None,
                    help="If set, write an updated URDF with corrected inertials")
    ap.add_argument("--interactive", action="store_true",
                    help="Step through differing links; press Enter to advance, 'q' to quit")
    return ap.parse_args()

# NEW: build default search roots when --package-dirs not supplied

def build_default_pkg_dirs(urdf_file, max_parent_levels=6):
    roots = []
    urdf_dir = os.path.dirname(os.path.abspath(urdf_file))
    # add urdf directory and its parents up to limit
    cur = urdf_dir
    for _ in range(max_parent_levels):
        if cur and cur not in roots:
            roots.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # add ROS_PACKAGE_PATH entries if present
    ros_paths = os.environ.get("ROS_PACKAGE_PATH", "").split(":" )
    for p in ros_paths:
        if p and os.path.isdir(p) and p not in roots:
            roots.append(p)
    return roots

def resolve_mesh_path(uri, pkg_dirs):
    if uri.startswith("package://"):
        rest = uri[len("package://"):]
        parts = rest.split("/", 1)
        if len(parts) == 1:
            return None
        pkg, subpath = parts
        for root in pkg_dirs:
            candidate = os.path.join(root, pkg, subpath)
            if os.path.isfile(candidate):
                return candidate
        return None
    # relative path => relative to urdf file directory
    return uri

def load_trimesh(mesh_path, scale=None):
    m = trimesh.load(mesh_path, force='mesh')
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump(concatenate=True)
    if scale is not None:
        sx, sy, sz = scale
        m.apply_scale([sx, sy, sz])
    return m

def fmt6(x):
    return f"{x:.6e}"

def rel_diff(a, b):
    if a == 0 and b == 0:
        return 0
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom

def extract_scale(mesh_elem):
    scale_attr = mesh_elem.get("scale")
    if not scale_attr:
        return None
    try:
        vals = [float(v) for v in scale_attr.strip().split()]
        if len(vals) == 3:
            return vals
    except:
        pass
    return None

def get_first_mesh(link):
    # Prefer collision mesh, fallback to visual
    for tag in ("collision", "visual"):
        for elem in link.findall(tag):
            geom = elem.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is not None and mesh.get("filename"):
                return mesh
    return None

def parse_inertial(link):
    inertial = link.find("inertial")
    if inertial is None:
        return None
    mass_elem = inertial.find("mass")
    inertia_elem = inertial.find("inertia")
    origin_elem = inertial.find("origin")
    if mass_elem is None or inertia_elem is None:
        return None
    try:
        mass = float(mass_elem.get("value"))
    except:
        return None
    data = {
        "mass": mass,
        "ixx": float(inertia_elem.get("ixx")),
        "iyy": float(inertia_elem.get("iyy")),
        "izz": float(inertia_elem.get("izz")),
        "ixy": float(inertia_elem.get("ixy", "0")),
        "ixz": float(inertia_elem.get("ixz", "0")),
        "iyz": float(inertia_elem.get("iyz", "0")),
        "origin_xyz": (0.0,0.0,0.0),
        "origin_rpy": (0.0,0.0,0.0)
    }
    if origin_elem is not None:
        xyz = origin_elem.get("xyz","0 0 0").split()
        rpy = origin_elem.get("rpy","0 0 0").split()
        try:
            data["origin_xyz"] = tuple(float(v) for v in xyz)
            data["origin_rpy"] = tuple(float(v) for v in rpy)
        except:
            pass
    return data, inertial

def build_inertial_xml(mass, com, inertia_matrix):
    ixx = inertia_matrix[0,0]; iyy = inertia_matrix[1,1]; izz = inertia_matrix[2,2]
    ixy = -inertia_matrix[0,1]; ixz = -inertia_matrix[0,2]; iyz = -inertia_matrix[1,2]
    inertial = ET.Element("inertial")
    origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}")
    origin.set("rpy", "0 0 0")
    mass_el = ET.SubElement(inertial, "mass")
    mass_el.set("value", f"{mass:.6f}")
    inertia_el = ET.SubElement(inertial, "inertia")
    inertia_el.set("ixx", fmt6(ixx)); inertia_el.set("iyy", fmt6(iyy)); inertia_el.set("izz", fmt6(izz))
    inertia_el.set("ixy", fmt6(ixy)); inertia_el.set("ixz", fmt6(ixz)); inertia_el.set("iyz", fmt6(iyz))
    return inertial

def inertial_to_dict(el):
    # For building updated file while preserving order minimal changes
    return deepcopy(el)

def replace_inertial(link, new_inertial):
    old = link.find("inertial")
    if old is not None:
        link.remove(old)
    link.insert(0, new_inertial)

def main():
    args = parse_args()
    urdf_file = args.urdf_file
    # user provided dirs (can be empty string)
    user_pkg_dirs = [p for p in args.package_dirs.split(":") if p]
    pkg_dirs = user_pkg_dirs or build_default_pkg_dirs(urdf_file)
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    updated = False
    report = []
    link_results = []  # store per-link info for interactive mode
    report.append(f"[INFO] Using package search roots: {pkg_dirs}")
    for link in root.findall("link"):
        lname = link.get("name","(unnamed)")
        parsed = parse_inertial(link)
        if not parsed:
            continue
        inertial_data, inertial_xml = parsed
        mesh_elem = get_first_mesh(link)
        if mesh_elem is None:
            continue
        filename = mesh_elem.get("filename")
        scale = extract_scale(mesh_elem)
        mesh_path = resolve_mesh_path(filename, pkg_dirs)
        if mesh_path is None:
            msg = f"[SKIP] {lname}: cannot resolve mesh {filename}"
            report.append(msg)
            link_results.append({"name": lname, "type": "skip", "lines": [msg]})
            continue
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(os.path.dirname(urdf_file), mesh_path)
        if not os.path.isfile(mesh_path):
            msg = f"[SKIP] {lname}: mesh file not found {mesh_path}"
            report.append(msg)
            link_results.append({"name": lname, "type": "skip", "lines": [msg]})
            continue

        try:
            mesh = load_trimesh(mesh_path, scale)
        except Exception as e:
            msg = f"[SKIP] {lname}: failed to load mesh ({e})"
            report.append(msg)
            link_results.append({"name": lname, "type": "skip", "lines": [msg]})
            continue

        mass = inertial_data["mass"]
        if mesh.mass == 0:
            msg = f"[SKIP] {lname}: mesh has zero volume"
            report.append(msg)
            link_results.append({"name": lname, "type": "skip", "lines": [msg]})
            continue
        com = mesh.center_mass
        inertia_mat = mesh.moment_inertia * (mass / mesh.mass)

        comp = {
            "ixx": inertia_mat[0,0], "iyy": inertia_mat[1,1], "izz": inertia_mat[2,2],
            "ixy": -inertia_mat[0,1], "ixz": -inertia_mat[0,2], "iyz": -inertia_mat[1,2],
            "origin_xyz": tuple(com),
        }

        diffs = {}
        keys = ["ixx","iyy","izz","ixy","ixz","iyz"]
        exceed = False
        for k in keys:
            d = rel_diff(inertial_data[k], comp[k])
            diffs[k] = d
            if d > args.tolerance:
                exceed = True
        origin_diff = math.dist(inertial_data["origin_xyz"], comp["origin_xyz"])
        if origin_diff > 1e-4:
            exceed = True

        lines = []
        if exceed:
            new_inertial = build_inertial_xml(mass, com, inertia_mat)
            if args.write_updated:
                replace_inertial(link, new_inertial)
                updated = True
            header = f"[DIFF] {lname} inertia differs. Max rel diff: {max(diffs.values()):.3f} origin shift {origin_diff:.4f} m"
            lines.append(header)
            # build current and computed lines with per-component coloring if exceeded
            cur_parts = []
            comp_parts = []
            for k in keys:
                val = inertial_data[k]
                diff = diffs[k]
                colored_val = f"{RED}{val:.4e}{RESET}" if diff > args.tolerance else f"{val:.4e}"
                cur_parts.append(f"{k}={colored_val}({diff*100:.1f}%)")
                comp_parts.append(f"{k}={comp[k]:.4e}")
            lines.append("  Current: " + " ".join(cur_parts) + f" origin={inertial_data['origin_xyz']}")
            lines.append("  Computed: " + " ".join(comp_parts) + f" origin={tuple(round(v,6) for v in comp['origin_xyz'])}")
            lines.append("  Suggested <inertial> block:")
            inertial_str = ET.tostring(new_inertial, encoding="unicode")
            for line in inertial_str.strip().splitlines():
                lines.append("    " + line)
            link_results.append({"name": lname, "type": "diff", "lines": lines, "exceed": True})
        else:
            msg = f"[OK] {lname} within tolerance (max rel diff {max(diffs.values()):.3f})"
            lines.append(msg)
            link_results.append({"name": lname, "type": "ok", "lines": lines, "exceed": False})
            report.append(msg)

    if args.interactive:
        print("[INTERACTIVE] Press Enter to step through differing links, 'a' to show all, 'q' to quit")
        show_all = False
        for lr in link_results:
            if lr["type"] == "diff" or show_all:
                for line in lr["lines"]:
                    print(line)
                user = input("(Enter=next, a=show all remaining, q=quit) > ").strip().lower()
                if user == 'q':
                    break
                if user == 'a':
                    show_all = True
                    continue
            # if not diff and not show_all just skip silently
        print("[INTERACTIVE] Done.")
    else:
        # Non-interactive: print accumulated report (already included OK / SKIP lines); add diff lines
        printed = set()
        for lr in link_results:
            if lr["type"] == "diff":
                for line in lr["lines"]:
                    print(line)
        # already printed diff lines; print previously collected SKIP/OK in order
        # They were stored in report list; print them now (excluding duplicates)
        base_lines = [l for l in report if l.startswith("[INFO]") or l.startswith("[SKIP]") or l.startswith("[OK]")]
        if base_lines:
            print("\nSummary:")
            for l in base_lines:
                print(l)

    if args.write_updated and updated:
        ET.indent(tree, space="  ", level=0)
        tree.write(args.write_updated, encoding="utf-8", xml_declaration=True)
        print(f"\nUpdated URDF written to {args.write_updated}")
    elif args.write_updated:
        print("\nNo changes necessary; updated file not written.")

if __name__ == "__main__":
    main()