import trimesh
import sys

if len(sys.argv) != 3:
    print("Usage: python compute_inertia.py <mesh_file> <mass_kg>")
    sys.exit(1)

mesh_file = sys.argv[1]
mass = float(sys.argv[2])

# === LOAD MESH ===
mesh = trimesh.load(mesh_file)
if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)  # merge into single mesh if scene

# === GET CENTER OF MASS & MOMENT OF INERTIA (about origin) ===
com = mesh.center_mass
# moment_inertia is for unit density; scale to desired mass
inertia_matrix = mesh.moment_inertia * (mass / mesh.mass)

# Extract components for URDF
ixx = inertia_matrix[0, 0]
iyy = inertia_matrix[1, 1]
izz = inertia_matrix[2, 2]
ixy = -inertia_matrix[0, 1]  # URDF off-diagonals are negated
ixz = -inertia_matrix[0, 2]
iyz = -inertia_matrix[1, 2]

# === PRINT URDF BLOCK ===
print("<inertial>")
print(f'  <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>')
print(f'  <mass value="{mass}"/>')
print(f'  <inertia ixx="{ixx:.6e}" iyy="{iyy:.6e}" izz="{izz:.6e}" '
      f'ixy="{ixy:.6e}" ixz="{ixz:.6e}" iyz="{iyz:.6e}"/>')
print("</inertial>")