.. _vision:

Vision
======

MyGym enables you to use pre-trained vision models to extend the versatility of your training scenarios. 
The vision models can be used instead of ground truth data from simulator to retrieve information about 
the environment where robot performs its task. Vision models take simulator's camera data (RGB and/or depth image) 
as inputs to inference and return information about observed scene. Thanks to that, your training becomes 
independent on ground truth from simulator and can be therefore easier transfered to real robot tasks.

MyGym integrated two different vision modules - YOLACT and VAE - and you can alternate between ground truth 
and these when specifying the type of source of reward signal in *config file* or as a command line argument: 
**reward_type=** either **gt** (ground truth) or **3dvs** (YOLACT) or **2dvu** (VAE).

YOLACT
------
Mygym implements YOLACT [1]_ for instance segmantation. If **3dvs** is chosen for **reward_type**, the pre-trained YOLACT 
model is used to get observations from the environment. The input into YOLACT inference is RGB image rendered by 
the active camera, the inference results are masks and bounding boxes of detected objects. The vision module further 
calculates the position of centroids of detected objects in pixel space. Lastly, the vision module utilizes the depth 
image from the active camera to project the object's centroid into 3D worl coordinates. This way, the absolute 
position of task objects is obtained only from sensory data without any ground truth inputs.

The current pre-trained model can detect all :ref:`mygym_objects` and three of :ref:`mygym_robots` including 
their grippers (kuka, jaco, panda).

If you would like to train new YOLACT model, you can use prepared dataset generator available in myGym, 
see :ref:`dataset`. For instructions regarding training itself, visit `YOLACT`_ home page.

.. [1] Daniel Bolya, Chong Zhou, Fanyi Xiao, & Yong Jae Lee (2019). YOLACT: Real-time Instance Segmentation. In ICCV.
.. _YOLACT: https://github.com/dbolya/yolact 

VAE
---

The objective of an unsupervised version of the prepared tasks (reach
task, push task, pick and place etc.) is to minimize the difference
between the actual and goal scene images. To measure their difference,
we have implemented a variational autoencoder (VAE) that
compresses each image into an n-dimensional latent vector. Since the VAE
is optimized so that it preserves similarities among images also in the
latent space (scenes with objects close to each other will have their
encoded vectors also closer to each other), it is possible to measure
the euclidean distance between the encoded scenes and use it for reward
calculation - i.e., the smaller the euclidean distance between actual
and goal image, the higher the reward. Please note that the limitation of
using VAE is that it works conveniently only with 2D information - i.e.,
it is a very weak source of visual information in 3D tasks such as pick
and place.

We provide a pretrained VAE for some of the task scenarios, but we also
include code for training of your own VAE (including dataset
generation), so that you can create custom experiments. To learn how to train your robot with the pretrained weights, see :ref:`train_vae`

How to train a custom VAE
~~~~~~~~~~~~~~~~~~~~

You are free to train your own VAE with a custom set of objects, level
of randomisation, background scene or type of robot. Here we describe
how.

Generating a dataset
~~~~~~~~~~~~~~~~~~~~

To generate a VAE dataset, run the following script:

``python generate_dataset.py configs/dataset_vae.json``

All the dataset parameters shall be adjusted in
configs/dataset_vae.json. They are described in comments, so here we
highlight the most important ones:

-  **output_folder**: where to save the resulting dataset
-  **imsize**: the resulting square image size, that will be saved. We
   currently only support VAE architectures for imsize of 128 or 64. The
   cropping of the image is done atomatically and can be adjusted in the
   code.
-  **num_episodes**: corresponds to the overall number of images in the
   dataset (in case the make_shot_every_frame parameter is set to 1)
-  **random_arm_movement**: whether to move the robot randomly,
   otherwise it stays fixed in its default position
-  **used_class_names_quantity**: what kind of objects do you want to
   show in the scene and how often. The names correspond to the urdf
   object names in the envs/objects directory. The first number in each
   list corresponds to the frequency, i.e. 1 is a default frequency and
   values above 1 make the object appear more ofthen than the others.
-  **object_sampling_area**: set the area in which the selected objects
   will be sampled; the format is *xxyyzz*
-  **num_objects_range**: in each image taken, a random number of
   objects from this range will appear in the scene
-  **object_colors**: if you have a color randomizer enabled and want
   some objects to have fixed color, you can set it up here
-  **active_camera**: the viewpoint from which the scene will be
   captured. The number 1 defines the active camera that will be used.
   We currently only support one camera viewpoint for dataset
   generation.

Training VAE
~~~~~~~~~~~~

Once you have your dataset ready, you can continue with VAE training.
This is handled with the following script:

``python train_vae.py --config vae/config.ini --offscreen``

The –offscreen parameter turns off any kind of visualisation, so if you
want to see the progress, do not use it. Otherwise, all parameters can
be set in the config.ini file as follows:

-  **n_latents**: the dimensionality of the latent vector z
-  **batch_size**: choose any integer
-  **lr**: the learning rate
-  **beta**: size of the beta paremeter to induce disentanglement of the
   latent space as proposed in `this
   paper <https://openreview.net/forum?id=Sy2fzU9gl>`__
-  **img_size**: size of the square images to train on. Currently the
   only supported sizes are 64 or 128
-  **use_cuda**
-  **n_epochs**: the number of training epochs
-  **viz_every_n_epochs**: how often to save the image reconstruction to
   monitor the training progress
-  **annealing_epochs**: the number of epochs for which to gradually
   increase the impact of KLD term in the ELBO loss. See `this
   paper <https://arxiv.org/abs/1903.10145>`__ for more info
-  **log_interval**: how often to print out the log about the training
   progress in the console
-  **test_data_percentage**: the fraction of the training dataset that
   will be used as testing data
-  **dataset_path**: path to the dataset folder containing images

The trained VAE will be saved in the ciircgym/vae/trained_models/ folder,
along with the config used for the training and visualisations.

.. automodule:: myGym.envs.vision_module
  :members:

