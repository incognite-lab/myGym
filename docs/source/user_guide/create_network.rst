.. _create_network:

Create custom network
=====================

MyGym toolkit is available in such form, that enables you to start with
your own training very easily and quickly. You can select one of the 
Reinforcement learning algorithms implemented by Stable baselines [1]_
or try to design your own.
This tutorial shows how to train a custom network in myGym with your own
implementation of a learning algortihm. We recommend you to get familiar
with the structure of `Stable baselines`_ implementations first.

To create your custom algorithm we recommend you to define it as a
child of an existing Stable baselines `algorithm`_ class. 
To create your custom policy class, follow these `instructions`_.

Choose a keyword for your custom algorithm and add new value to the 
dictionary of implemented combinations in `train.py <https://github.com/incognite-lab/myGym/blob/master/myGym/train.py>`_:

.. code-block:: python

    implemented_combos = {"myalgo": {*framework*: [MyAlgoClass, (MyPolicyClass, env), {*training kwargs*}]},
                            ...}

To start training with your custom network, set your new algorithm 
keyword as the *algo* parameter in the config file:

``"algo":"myalgo",``

or in the command line:

``python train.py --algo=myalgo``

The function *train* in the train.py script will read the value from 
the implemented_combos dictionary:

.. code-block:: python

    model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
    model_kwargs = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][2]

create a model, an instance of MyAlgoClass: 

.. code-block:: python

    model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0](*model_args, **model_kwargs)

and start with training:

.. code-block:: python

    model.learn(total_timesteps=arg_dict["steps"], callback=callbacks_list)

When finished, trained model is saved to the *lodgir*:

.. code-block:: python

    model.save(os.path.join(model_logdir, model_name))


.. [1] Hill, Ashley and Raffin, Antonin and Ernestus, Maximilian and Gleave, Adam and Kanervisto, Anssi and Traore, Rene and Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai (2018). Stable Baselines. Github repository.
.. _Stable baselines: https://github.com/hill-a/stable-baselines
.. _instructions: https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
.. _algorithm: https://stable-baselines.readthedocs.io/en/master/modules/base.html