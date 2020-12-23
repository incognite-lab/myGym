.. _train_camera:

Train robot - supervised vision
===============================

MyGym enables you to use pre-trained vision models to extend the versatility of your training scenarios.
Each workspace has five virtual cameras, that observe the scene. You can use images from a camera
and pretrained YOLACT model to do image segmentation. This way you retrieve information about 
position of robot and task objects using camera images instead of using ground truth. The reward for 
training will be calculated based on data returned by vision.

.. figure:: ../../../myGym/images/workspaces/kuka_yolact.gif
   :alt: training yolact

To train a robot using image segmentation set in the config file:

``"reward_type": "3dvs",``

and specify the path to YOLACT preptrained model and model config:

``"yolact_path":"trained_models/weights_yolact_mygym_23/crow_base_15_266666.pth",``
``"yolact_config":"trained_models/weights_yolact_mygym_23/config_train_15.obj",``

.. note:: 
    If you do not have myGym's pretrained vision model, download it first:

    ``cd myGym`` 

    ``sh download_vision.sh``

We recommed to turn the visgym parameter off:

``"visgym": 0",``

You can visualize how the vision performs:

``"visualize":1,``

Start the training with this modified config file:

``python train.py --config my_config.json``

Alternatively, you can specify the above parameters in the command line:

.. code-block:: python

    python train.py --reward_type=3dvs 
    --yolact_path=trained_models/weights_yolact_mygym_23/crow_base_15_266666.pth 
    --yolact_config=trained_models/weights_yolact_mygym_23/config_train_15.obj 
    --visgym=0 --visualize=1

After the training finishes, find your model in the logdir.

To learn more about myGym's vision models, see :ref:`vision`.
