import pybullet as pb
import pybullet_utils.bullet_client as bc
import time
import numpy as np

"""
Simple pybullet collision detection-driven voxelizer. 
See the "voxelize" function for details. 
Run the test code from the myGym root folder "python -m myGym.utils.voxels"
"""


def _linspace(extends, resolution):
    """Returns linear spacing (indices and steps) for entered extends and resuolutions"""
    return (
        np.array(i)
        for i in zip(*(np.linspace(e[0], e[1], num=r, retstep=True) for e, r in zip(extends, resolution)))
    )


def voxelize(objects, extends, resolution, pb_client):
    """
    Voxelization of pybullet bodies in specific extends and resulution
    Takes a list of bullet bodies and iterates through the extends-defined box
    performing a collision detection on a moving voxel cube.

    Args:
        objects: list of pybullet bodies to voxelize
        extends: dimensions of a 3D box to voxelize in
        resolution: 3D list of resolution
        pb_client: pybullet client

    Returns:
        3D numpy array with voxel values (0.0 or 1.0)
    """
    indices, steps = _linspace(extends, resolution)
    voxels = np.full(resolution, 0.0)

    box = pb_client.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=pb_client.createCollisionShape(pb.GEOM_BOX, halfExtents=steps * 0.5),
    )
    zero_orn = pb.getQuaternionFromEuler([0, 0, 0])

    for i, x in enumerate(indices[0]):
        for j, y in enumerate(indices[1]):
            for k, z in enumerate(indices[2]):
                pb_client.resetBasePositionAndOrientation(box, posObj=[x, y, z], ornObj=zero_orn)
                if any(pb_client.getClosestPoints(box, o, 0) for o in objects):
                    voxels[i, j, k] = 1.0

    pb_client.removeBody(box)
    return voxels


def visualize(voxels, extends, pb_client):
    """Visualizes created voxels by adding a batch multibody of voxel cubes into the pybyllet world"""

    indices, steps = _linspace(extends, voxels.shape)

    batch_positions = [
        [x, y, z]
        for i, x in enumerate(indices[0])
        for j, y in enumerate(indices[1])
        for k, z in enumerate(indices[2])
        if voxels[i, j, k] == 1.0
    ]

    pb_client.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=pb_client.createCollisionShape(pb.GEOM_BOX, halfExtents=steps * 0.5),
        basePosition=[0, 0, 0],
        batchPositions=batch_positions,
    )


def test():
    p = bc.BulletClient(connection_mode=pb.GUI)
    p.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    floor = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=p.createCollisionShape(pb.GEOM_BOX, halfExtents=[2, 2, 0.2]),
        basePosition=[0, 0, -1.5],
    )
    duck = p.loadURDF("myGym/envs/objects/toys/urdf/bird.urdf", globalScaling=15.0)
    extends = [[-2, 2], [-2, 2], [-1.5, 2.5]]

    voxels = voxelize([floor, duck], extends, [50, 50, 50], p)
    visualize(voxels, extends, p)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    test()
