import pybullet as p
import os

# 1. Iniciar PyBullet sin interfaz gráfica
p.connect(p.DIRECT)

# 2. Rutas de archivos

import os

base_dir = os.path.dirname(os.path.abspath(__file__))

input_stl = os.path.join(
    base_dir,
    "robots",
    "S2",
    "meshes",
    "waist_yaw_link.STL"
)

output_obj = os.path.join(
    base_dir,
    "robots",
    "S2",
    "meshes",
    "part_vhacd.obj"
)

log_file = os.path.join(
    base_dir,
    "robots",
    "S2",
    "vhacd_log.txt"
)

# 3. Crear carpeta de salida si no existe
os.makedirs(os.path.dirname(output_obj), exist_ok=True)

import os

print("Input exists:", os.path.exists(input_stl))
print("Output folder exists:", os.path.exists(os.path.dirname(output_obj)))

# 4. Ejecutar VHACD
p.vhacd(
    input_stl,
    output_obj,
    log_file,

    # Calidad de descomposición
    resolution=10000,

    # Profundidad (más alto = más piezas)
    depth=20,

    # Qué tan preciso es con formas complejas
    concavity=0.0025,

    # Optimización interna
    planeDownsampling=4,
    convexhullDownsampling=4,

    # Ajustes finos (puedes dejarlos así)
    alpha=0.05,
    beta=0.05,
    gamma=0.001,

    # Otros parámetros
    pca=0,
    mode=0,

    # Máximo vértices por pieza convexa
    maxNumVerticesPerCH=64,

    # Ignorar piezas muy pequeñas
    minVolumePerCH=0.0001
)

# 5. Cerrar PyBullet
p.disconnect()

print("VHACD terminado correctamente")