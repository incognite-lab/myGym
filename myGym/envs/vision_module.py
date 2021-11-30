try:
    import torch
except:
    print("Torch doesn't work")
import sys
import numpy as np
import cv2
import random
import pkg_resources
currentdir = pkg_resources.resource_filename("myGym", "envs")

# import vision models YOLACT, VAE
sys.path.append(pkg_resources.resource_filename("myGym", "yolact_vision")) #may be moved somewhere else
try:
    from inference_tool import InfTool
except:
    print("Problem importing YOLACT.")
from myGym.vae.vis_helpers import load_checkpoint
from myGym.vae import  sample

class VisionModule:
    """
    Vision class that retrieves information from environment based on a visual subsystem (YOLACT, VAE) or ground truth

    Parameters:
        :param vision_src: (string) Source of information from environment (ground_truth, yolact, vae)
        :param env: (object) Environment, where the training takes place
        :param vae_path: (string) Path to a trained VAE in 2dvu reward type
        :param yolact_path: (string) Path to a trained Yolact in 3dvu reward type
        :param yolact_config: (string) Path to saved Yolact config obj or name of an existing one in the data/Config script or None for autodetection
    """
    def __init__(self, observation={}, env=None, vae_path=None, yolact_path=None, yolact_config=None):
        self.src = self.get_module_type(observation)
        self.env = env
        self.vae_embedder = None
        self.vae_imsize = None
        self.vae_path = vae_path
        self.yolact_path = yolact_path
        self.yolact_config = yolact_config
        self.obsdim = None
        self._initialize_network(self.src)
        self.mask = {}
        self.centroid = {}
        self.centroid_transformed = {}


    def get_module_type(self, observation):
        """
        Get source of the information from environment (ground_truth, yolact, vae)

        Returns:
            :return source: (string) Source of information
        """
        if observation["actual_state"] not in ["vae", "yolact", "voxel", "dope"]:
            src = "ground_truth_6D" if "6D" in observation["actual_state"] else "ground_truth"
        else:
            src = observation["actual_state"]
        return src

    def crop_image(self, img):
        """
        Crop image by 1/4 from each side

        Parameters:
            :param img: (list) Original image
        Returns:
            :return img: (list) Cropped image
        """
        dim1 = img.shape[0]
        crop1 = [int(dim1/4), int(dim1-(dim1/4))]
        dim2 = img.shape[1]
        crop2 = [int(dim2/4), int(dim2-(dim2/4))]
        img = img[crop1[0]:crop1[1], crop2[0]:crop2[1]]
        return img

    def get_obj_pixel_position(self, obj=None, img=None):
        """
        Get mask and centroid in pixel space coordinates of an object from 2D image

        Parameters:
            :param obj: (object) Object to find its mask and centroid
            :param img: (array) 2D input image to inference of vision model
        Returns:
            :return mask: (list) Mask of object
            :return centroid: (list) Centroid of object in pixel sprace coordinates
        """
        if self.src == "ground_truth" or self.src == "ground_truth_6D":
            pass
        elif self.src in ["dope", "yolact"]:
            if img is not None:
                if self.src == "yolact":
                    classes, class_names, scores, boxes, masks, centroids = self.inference_yolact(img)
                    if self.env.visualize == 1:
                        img_numpy = self.yolact_cnn.label_image(img)
                        cv2.imshow("Yolact(3dvs) inference", img_numpy)
                        cv2.waitKey(1)
                    try:
                        self.mask[obj.get_name()] = masks[class_names.index(obj.get_name())]
                        self.centroid[obj.get_name()] = centroids[class_names.index(obj.get_name())]
                        #print("{} was detected".format(obj.get_name()))
                    except:
                        if obj.get_name() not in self.mask.keys():
                            self.mask[obj.get_name()] = [[-1]]
                            self.centroid[obj.get_name()] = [-1,-1]
                        #print("{} not detected in present image".format(obj.get_name()))

                    return self.mask[obj.get_name()], self.centroid[obj.get_name()]
                elif self.src == "dope":
                    pass
                    # @TODO
            else:
                raise Exception("You need to provide image argument for segmentation")

    def get_obj_bbox(self, obj=None, img=None):
        """
        Get bounding box of an object from 2D image

        Parameters:
            :param obj: (object) Object to find its bounding box
            :param img: (array) 2D input image to inference of vision model
        Returns:
            :return bbox: (list) Bounding box of object
        """
        if self.src == "ground_truth" or "ground_truth_6D":
            if obj is not None:
                return obj.get_bounding_box()
            else:
                raise Exception("You need to provide obj argument to get gt bounding box")
        elif self.src in ["dope", "yolact"]:
            if img is not None:
                if self.src == "yolact":
                    classes, class_names, scores, boxes, masks, centroids = self.inference_yolact(img)
                    try:
                        bbox = boxes[class_names.index(obj.get_name())]
                    except:
                        bbox = []
                        print("Object not detected in present image")
                    return bbox
                elif self.src == "dope":
                    pass
                    # @TODO
            else:
                raise Exception("You need to provide image argument for bbox segmentation")
        else:
            raise Exception("{} module does not provide bounding boxes!".format(self.src))

    def get_obj_position(self, obj=None, img=None, depth=None):
        """
        Get object position in world coordinates of environment from 2D and depth image

        Parameters:
            :param obj: (object) Object to find its mask and centroid
            :param img: (array) 2D input image to inference of vision model
            :param depth: (array) Depth input image to inference of vision model
        Returns:
            :return position: (list) Centroid of object in world coordinates
        """
        if self.src == "ground_truth":
            if obj is not None:
                return list(obj.get_position())
            else:
                raise Exception("You need to provide obj argument to get gt position")
        elif self.src == "ground_truth_6D":
            if obj is not None:
                return (list(obj.get_position()) + list(obj.get_orientation()))
            else:
                raise Exception("You need to provide obj argument to get gt position")
        elif self.src in ["yolact", "dope"]:
            if img is not None:
                if self.src == "yolact":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    mask, centroid = self.get_obj_pixel_position(obj, img)
                    centroid_transformed = self.yolact_cnn.find_3d_centroids_(mask, depth, self.env.unwrapped.cameras[self.env.active_cameras].view_x_proj)
                    if centroid_transformed.size == 3:
                        self.centroid_transformed[obj.get_name()] = centroid_transformed
                        #print("{} was detected at {}".format(obj.get_name(),self.centroid_transformed[obj.get_name()]))
                    elif obj.get_name() not in self.centroid_transformed.keys():
                        self.centroid_transformed[obj.get_name()] = [10, 10, 10]
                        #print("{} was not detected, assign {}".format(obj.get_name(),self.centroid_transformed[obj.get_name()]))
                    else:
                        pass
                        #print("{} was not detected, assign previous {}".format(obj.get_name(),self.centroid_transformed[obj.get_name()]))
                    return list(self.centroid_transformed[obj.get_name()])
            else:
                raise Exception("You need to provide image argument to infer object position")
        return

    def get_obj_orientation(self, obj=None, img=None):
        """
        Get object orientation in world coordinates of environment from 2D image

        Parameters:
            :param obj: (object) Object to find its mask and centroid
            :param img: (array) 2D input image to inference of vision model
        Returns:
            :return orientation: (list) Orientation of object in world coordinates
        """
        if self.src == "ground_truth" or "ground_truth_6D":
            if obj is not None:
                return obj.get_orientation()
            else:
                raise Exception("You need to provide obj argument to get gt orientation")
        elif self.src in ["yolact", "dope"]:
            if img is not None:
                # @TODO
                pass
            else:
                raise Exception("You need to provide image argument to infer orientation")
        return

    def vae_generate_sample(self):
        """
        Generate image as a sample of VAE latent representation

        Returns:
            :return dec_img: Generated image from VAE latent representation
        """
        latent_z = torch.tensor([random.uniform(-2, 2) for _ in range(self.vae_embedder.n_latents)]).unsqueeze(0)
        decoded = self.vae_embedder.image_decoder(latent_z)
        img = decoded.squeeze(0).reshape(self.vae_imsize, self.vae_imsize, 3)
        dec_img = np.asarray((img * 255).cpu().detach(), dtype="uint8")
        return dec_img

    def encode_with_vae(self, imgs, task="reach", decode=0):
        """
        Encode the input image into an n-dimensional latent variable using VAE model

        Parameters:
            :param imgs: (list of arrays) Input images
            :param task: (string) Type of learned task (reach, push, ...)
            :param decode: (bool) Whether to decode encoded images from latent representation back to image array
        Returns:
            :return latent_z: (list) Latent representation of images
            :return dec_img: (list of arrays) Decoded images from latent representation back to image arrays
        """
        if self.src != "vae":
            raise Exception("Encoding can only be done with VAE module!")
        imgs_input = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img.shape[0] != self.vae_imsize:
                res = [0,450,100, 500] if task == "reach" else [60,390,160,480]
                img = cv2.resize(img[res[0]:res[1],res[2]:res[3]], (self.vae_imsize, self.vae_imsize))
            im = torch.tensor(img).type(torch.FloatTensor)
            im = im.reshape(img.shape[2], img.shape[0], img.shape[0]).unsqueeze(0)/255
            imgs_input = torch.cat((imgs_input, im), dim=0) if torch.is_tensor(imgs_input) else im
        latent_z = self.vae_embedder.infer(imgs_input)[0].detach().cpu()
        dec_img = sample.decode_images(self.vae_embedder, latent_z) if decode == 1 else []
        return latent_z.squeeze().tolist(), dec_img

    def inference_yolact(self, img):
        """
        Infere using YOLACT model

        Parameters:
            :param img: (array) Input 2D image
        Returns:
            :return classes: (list of ints) Classes IDs of detected objects
            :return class_names: (list of strings) Classes names of detected objects
            :return scores: (list of floats) Scores (confidence) of object detections
            :return boxes: (list of lists) Bounding boxes of detected objects
            :return masks: (list of lists) Masks of detected objects
            :return centroids: (list of lists) Centroids of detected objects in pixel space coordinates
        """
        classes, class_names, scores, boxes, masks, centroids = self.yolact_cnn.raw_inference(img)
        return classes, class_names, scores, boxes, masks, centroids

    def _initialize_network(self, network):
        """
        Initialize pre-trained vision model and define corresponding dimension of observation data

        Parameters:
            :param network: (string) Source of information from environment (yolact, vae)
        """
        if network == "vae":
            weights_pth = pkg_resources.resource_filename("myGym", self.vae_path)
            try:
                self.vae_embedder, imsize = load_checkpoint(weights_pth, use_cuda=True)
            except:
                raise Exception("For vae observation, you need to download pre-trained vision model and specify its path in config. Specified {} not found.".format(self.vae_path))
            self.vae_imsize = imsize
            self.obsdim = (2*self.vae_embedder.n_latents) + 3
        elif network == "yolact":
            weights = pkg_resources.resource_filename("myGym", self.yolact_path)
            if ".obj" in self.yolact_config:
                config = pkg_resources.resource_filename("myGym", self.yolact_config)
            try:
                self.yolact_cnn = InfTool(weights=weights, config=config, score_threshold=0.2)
            except:
                raise Exception("For yolact observations, you need to download pre-trained vision model and specify its path in config. Specified {} and {} not found.".format(self.yolact_path, self.yolact_config))
        return

