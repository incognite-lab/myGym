import gym
from myGym import envs
import cv2
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env
import os, imageio
import numpy as np
import time
from numpy import matrix
import pybullet as p

clear = lambda: os.system('clear')

AVAILABLE_SIMULATION_ENGINES = ["mujoco", "pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["tensorflow", "pytorch"]

def test_env(env, arg_dict):
    debug_mode = True
    env.render("human")
    env.reset()
    joints = ['Joint1','Joint2','Joint3','Joint4','Joint5','Joint6','Joint7','Joint 8','Joint 9']
    jointparams = ['Jnt1','Jnt2','Jnt3','Jnt4','Jnt5','Jnt6','Jnt7','Jnt 8','Jnt 9']
    frictions = ['Friction 1','Friction 2','Friction 3','Friction 4','Friction 5','Friction 6','Friction 7','Friction 8','Friction 9']
    
    if debug_mode:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.resetDebugVisualizerCamera(2, 120, -30, [0, 1, 0])
            for i in range (env.action_space.shape[0]):
                joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i], env.action_space.high[i], 0)
    
    for e in range(10000):
        env.reset()
        for t in range(arg_dict["max_episode_steps"]):
            #action = env.action_space.sample()
            
            if t>=1:
                for i in range (env.action_space.shape[0]):
                    jointparams[i] = p.readUserDebugParameter(joints[i])
                    action.append(jointparams[i])
                #action = observation[10:17]
                #action = env.action_space.sample()
            else:
                action = [0,0,0,0,0,0,0]
                #action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if debug_mode:
                #print("Reward is {}, observation is {}".format(reward, observation))
                action = matrix(np.around(np.array(action),5))
                oaction = matrix(np.around(np.array(observation[10:17]),5))
                diff = matrix(np.around(np.array(action-oaction),5))
                print(f"Step:{t}")
                print (f"RAction:{action}")
                print(f"OAction:{oaction}")
                print(f"DAction:{diff}")
                p.addUserDebugText(f"DAction:{diff}",
                                        [-1, -1, 2], textSize=1.0, lifeTime=0.05, textColorRGB=[0.6, 0.0, 0.6])
                #time.sleep(.5)
                clear()
            action=[]
            if arg_dict["visualize"]:
                visualizations = [[],[]]
                env.render("human")
                for camera_id in range(len(env.cameras)):
                    camera_render = env.render(mode="rgb_array", camera_id=camera_id)
                    image = cv2.cvtColor(camera_render[camera_id]["image"], cv2.COLOR_RGB2BGR)
                    depth = camera_render[camera_id]["depth"]
                    image = cv2.copyMakeBorder(image, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    cv2.putText(image, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (0, 0, 0), 1, 0)
                    visualizations[0].append(image)
                    depth = cv2.copyMakeBorder(depth, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    cv2.putText(depth, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (0, 0, 0), 1, 0)
                    visualizations[1].append(depth)
                    
                if len(visualizations[0])%2 !=0:
                        visualizations[0].append(255*np.ones(visualizations[0][0].shape, dtype=np.uint8))
                        visualizations[1].append(255*np.ones(visualizations[1][0].shape, dtype=np.float32))
                fig_rgb = np.vstack((np.hstack((visualizations[0][0::2])),np.hstack((visualizations[0][1::2]))))
                fig_depth = np.vstack((np.hstack((visualizations[1][0::2])),np.hstack((visualizations[1][1::2]))))
                cv2.imshow('Camera RGB renders', fig_rgb)
                cv2.imshow('Camera depthrenders', fig_depth)
                cv2.waitKey(1)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def test_model(env, model=None, implemented_combos=None, arg_dict=None, model_logdir=None, deterministic=False):
    if arg_dict.get("model_path") is None and model is None:
        print("Path to the model using --model_path argument not specified. Testing random actions in selected environment.")
        test_env(env, arg_dict)
    else:
        try:
            if arg_dict["algo"] == "multi":
                model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
                model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(arg_dict["model_path"], env=model_args[1].env)
            else:
                model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(arg_dict["model_path"])
        except:
            if (arg_dict["algo"] in implemented_combos.keys()) and (arg_dict["train_framework"] not in list(implemented_combos[arg_dict["algo"]].keys())):
                err = "{} is only implemented with {}".format(arg_dict["algo"],list(implemented_combos[arg_dict["algo"]].keys())[0])
            elif arg_dict["algo"] not in implemented_combos.keys():
                err = "{} algorithm is not implemented.".format(arg_dict["algo"])
            else:
                err = "invalid model_path argument"
            raise Exception(err)

    images = []  # Empty list for gif images
    success_episodes_num = 0
    distance_error_sum = 0
    steps_sum = 0

    for e in range(arg_dict["eval_episodes"]):
        done = False
        obs = env.reset()
        is_successful = 0
        distance_error = 0
        while not done:
            steps_sum += 1
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            is_successful = not info['f']
            distance_error = info['d']

            if (arg_dict["record"] > 0) and (len(images) < 250):
                render_info = env.render(mode="rgb_array", camera_id = arg_dict["camera"])
                image = render_info[arg_dict["camera"]]["image"]
                images.append(image)
                print(f"appending image: total size: {len(images)}]")
            else:
                time.sleep(0.02)

        success_episodes_num += is_successful
        distance_error_sum += distance_error

    mean_distance_error = distance_error_sum / arg_dict["eval_episodes"]
    mean_steps_num = steps_sum // arg_dict["eval_episodes"]

    print("#---------Evaluation-Summary---------#")
    print("{} of {} episodes ({} %) were successful".format(success_episodes_num, arg_dict["eval_episodes"], success_episodes_num / arg_dict["eval_episodes"]*100))
    print("Mean distance error is {:.2f}%".format(mean_distance_error * 100))
    print("Mean number of steps {}".format(mean_steps_num))
    print("#------------------------------------#")

    model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
    file = open(os.path.join(model_logdir, "train_" + model_name + ".txt"), 'a')
    file.write("\n")
    file.write("#Evaluation results: \n")
    file.write("#{} of {} episodes were successful \n".format(success_episodes_num, arg_dict["eval_episodes"]))
    file.write("#Mean distance error is {:.2f}% \n".format(mean_distance_error * 100))
    file.write("#Mean number of steps {}\n".format(mean_steps_num))
    file.close()

    if arg_dict["record"] == 1:
        gif_path = os.path.join(model_logdir, "train_" + model_name + ".gif")
        imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=15)
        os.system('./utils/gifopt -O3 --lossy=5 -o {dest} {source}'.format(source=gif_path, dest=gif_path))
        print("Record saved to " + gif_path)
    elif arg_dict["record"] == 2:
        video_path = os.path.join(model_logdir, "train_" + model_name + ".avi")
        height, width, layers = image.shape
        out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))
        for img in images:
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        print("Record saved to " + video_path)


def main():
    parser = get_parser()
    arg_dict = get_arguments(parser)

    # Check if we chose one of the existing engines
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return

    model_logdir = os.path.dirname(arg_dict.get("model_path",""))
    env = configure_env(arg_dict, model_logdir, for_train=0)
    implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)

    test_model(env, None, implemented_combos, arg_dict, model_logdir)


if __name__ == "__main__":
    main()
