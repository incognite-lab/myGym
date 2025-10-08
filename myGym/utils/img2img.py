import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# ------------------------
# 1. Render an image in PyBullet
# ------------------------
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

# Camera setup
width, height = 512, 512
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[1, 1, 1],
    cameraTargetPosition=[0, 0, 0.5],
    cameraUpVector=[0, 0, 1],
)
proj_matrix = p.computeProjectionMatrixFOV(
    fov=60, aspect=1.0, nearVal=0.1, farVal=10.0
)

# Capture RGB image
_, _, rgb, _, _ = p.getCameraImage(
    width=width, height=height,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    renderer=p.ER_TINY_RENDERER
)

# Convert to PIL image
rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
rgb_img = Image.fromarray(rgb_array[:, :, :3])  # drop alpha

# Save debug image
rgb_img.save("pybullet_render.png")

# ------------------------
# 2. Load Stable Diffusion Img2Img
# ------------------------
model_id = "runwayml/stable-diffusion-v1-5"  # you can swap for SDXL if GPU is strong
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # use GPU if available

# ------------------------
# 3. Convert sim→real using img2img
# ------------------------
prompt = "A photorealistic robot standing in a lab environment, realistic lighting and textures"
negative_prompt = "cartoon, lowres, painting, blurry, unrealistic"

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=rgb_img,
    strength=0.6,   # how much to change (0.3 = preserve sim, 0.8 = more freedom)
    guidance_scale=7.5
).images[0]

# Save result
result.save("pybullet_to_real.png")
print("Saved: pybullet_to_real.png")
