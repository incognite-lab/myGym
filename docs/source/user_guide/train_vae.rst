.. _train_vae:

Train a robot - unsupervised vision 
=====================================


Besides training based on ground truth object positions, MyGym also enables training from image input only. This can be done either in a "supervised" way with YOLACT, or without supervision using a variational autoencoder (VAE).


Here we show you how to train with VAE. For this, we will use one of the five cameras that are available in the environment and a VAE that was pretrained on images from this specific viewpoint.
We then use VAE to generate goal scenes (images of a scene that we want) and to encode the current scene into a latent representation. During training, the objective is to minimise the (euclidean) distance between
the latent vectors of the actual and the goal scenes.


To train a robot using VAE, set in the config file:

``"reward_type": "2dvu",``


Setting up the reward type as "2dvu" means that it will automatically use the VAE to encode the images. Therefore, you also need to specify the path to the VAE model:

``"vae_path":"/vae/trained_models/vae_armtop_4/model_best.pth.tar",``


This is of course dependent on the task that you choose and also on the camera that you select (the VAE cannot generalise across different viewpoints).
We currently only support top view of the scene, corresponding to camera 4 (for other viewpoints, you need to train your own VAE -  :ref:`vision`):

``"camera": 4,``


To download the pretrained VAE model, please run:

    ``cd mygym``

    ``sh ./vae/download_weights.sh``



We also recommed to turn the visgym parameter off:

``"visgym": 0",``

and the actual and decoded (as seen by the VAE) images can be visualized using the parameter "visualize":

``"visualize":1,``

Start the training with this modified config file:

``python train.py --config my_config.json``

Alternatively, you can specify the above parameters in the command line:

.. code-block:: python

    python train.py --reward_type=2dvu,
    --vae_path=/vae/trained_models/vae_armtop_4/model_best.pth.tar
    --visgym=0 --visualize=1

After the training finishes, find your model in the logdir.



We have also implemented a specific version for the reach task and VAE, designed for the KUKA arm. The goal is to get the KUKA arm in a specific position, based on a goal image. There are no objects included in this task, only the arm moving. To try this scenario, run:

``python train.py --config configs/train_vae_reach.json``


To learn more about myGym's vision models, see :ref:`vision`.
