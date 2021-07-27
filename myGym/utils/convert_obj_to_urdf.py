import os, sys, shutil
import argparse
from object2urdf import ObjectUrdfBuilder
import xml.etree.ElementTree as ET

"""
The structure of the result (before converting .obj files should be structured as follows,
the URDF prototype should aslo be inlcuded):

object_folder
├── obj
│   ├── object_1
│   │   ├── object_1.mtl
│   │   ├── object_1.obj
│   │   └── object_1_texture.jpg
│   ├── object_2
│   │   ├── object_2.mtl
│   │   ├── object_2.obj
│   │   └── object_2_texture.jpg
│   └── _prototype.urdf
└── urdf
    ├── object_1.urdf
    └── object_2.urdf
"""


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--object_folder", type=str, default=None,
     help="Path to the folder with .obj files")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_arguments()

    # Create URDF files
    object_folder = args.object_folder
    builder = ObjectUrdfBuilder(object_folder)
    builder.build_library(force_overwrite=True, decompose_concave=False, force_decompose=False, center = 'mass')

    # Prepare files to move
    source_files = os.listdir(object_folder)
    dest_path = os.path.join(os.path.dirname(object_folder), "urdf")
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    # Update paths to .obj and move
    for file in source_files:
        if file.endswith('.urdf') and "_prototype" not in file:
            # Update paths to .obj files in URDF
            urdf_base = ET.parse(os.path.join(object_folder, file)).getroot()
            collision_mesh = urdf_base.find('.//collision/geometry/mesh')
            collision_mesh.attrib["filename"] =  os.path.join("../obj", collision_mesh.attrib["filename"])
            visual_mesh = urdf_base.find('.//visual/geometry/mesh')
            visual_mesh.attrib["filename"] =  os.path.join("../obj", visual_mesh.attrib["filename"])
            updated_urdf = ET.tostring(urdf_base)
            with open(os.path.join(object_folder, file), "wb") as f:
                f.write(updated_urdf)

            # Move the updated URDF to ../urdf folder
            shutil.move(os.path.join(object_folder, file), os.path.join(dest_path, file))
