import gym
from myGym import envs
import cv2
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env
import os, imageio
import numpy as np
import time
import open3d as o3d


AVAILABLE_SIMULATION_ENGINES = ["mujoco", "pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["tensorflow", "pytorch"]

def test_env(env, arg_dict):
    env.render("human")
    #geometry = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Voxel', width=640, height=480, left=1100, top=0, visible=True)
    #vis.add_geometry(geometry)
    ctr=vis.get_view_control()
    env.reset()
    for e in range(10000):
        env.reset()

        for t in range(200):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # print("Reward is {}, observation is {}".format(reward, observation))

            if arg_dict["visualize"]:
                visualizations = [[],[]]
                env.render("human")
                camera_id=4
                camera_render = env.render(mode="rgb_array", camera_id=camera_id)
                image = cv2.cvtColor(camera_render[camera_id]["image"], cv2.COLOR_RGB2BGR)
                depth = camera_render[camera_id]["depth"]
                #image = cv2.copyMakeBorder(image, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                #cv2.putText(image, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                #            (0, 0, 0), 1, 0)
                #    visualizations[0].append(image)
                #    depth = cv2.copyMakeBorder(depth, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                #    cv2.putText(depth, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                #                (0, 0, 0), 1, 0)
                #    visualizations[1].append(depth)
                #    
                #if len(visualizations[0])%2 !=0:
                #        visualizations[0].append(255*np.ones(visualizations[0][0].shape, dtype=np.uint8))
                #        visualizations[1].append(255*np.ones(visualizations[1][0].shape, dtype=np.float32))
                #fig_rgb = np.vstack((np.hstack((visualizations[0][0::2])),np.hstack((visualizations[0][1::2]))))
                #fig_depth = np.vstack((np.hstack((visualizations[1][0::2])),np.hstack((visualizations[1][1::2]))))
                depth_image = np.stack((depth, depth, depth), axis=2)[:, :, :]
                depth_image = (255*depth_image).astype(np.uint8)
                #cv2.imshow('Camera RGB', image)
                winname='CameraD'
                cv2.namedWindow(winname)        # Create a named window
                cv2.moveWindow(winname, 1100,520) 
                cv2.imshow(winname, depth_image)
                #cv2.waitKey(1)
                rgbimg = o3d.geometry.Image((image).astype(np.uint8))
                depthimg = o3d.geometry.Image((depth_image).astype(np.uint8))
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbimg, depthimg)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                    o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=.00003)
                
                vis.add_geometry(voxel_grid)
                
                ctr.rotate(400,100-t*2)
                cv2.waitKey(delay=10)
                vis.poll_events()
                
                
                vis.update_renderer()
                
                #vis.run()
                
                vis.clear_geometries()
                #o3d.visualization.draw_geometries([voxel_grid])
                
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def test_model(env, model=None, implemented_combos=None, arg_dict=None, model_logdir=None, deterministic=True):
    model_path = arg_dict.get("model_path")
    if model_path is None and model is None:
        print("Path to the model using --model_path argument not specified. Testing random actions in selected environment.")
        test_env(env, arg_dict)
    else:
        try:
            model = []
            if isinstance(model_path, list):
                for m_path in model_path:
                    model.append(implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(m_path))
            else:
                model.append(implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(model_path))
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
        model_idx = 0
        done = False
        obs = env.reset()
        is_successful = 0
        distance_error = 0
        while not done:
            steps_sum += 1
            action, _state = model[model_idx].predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            is_successful = not info['f']
            distance_error = info['d']
            if len(model) > 1:
                model_idx = info['s']

            if (arg_dict["record"] > 0):# and (len(images) < 250):
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
    print("{} of {} episodes were successful".format(success_episodes_num, arg_dict["eval_episodes"]))
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
    model_path = arg_dict.get("model_path","")
    if isinstance(model_path, list):
        model_logdir = arg_dict['logdir']
    else:
        model_logdir = os.path.dirname(model_path)
    env = configure_env(arg_dict, model_logdir, for_train=0)
    implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)

    test_model(env, None, implemented_combos, arg_dict, model_logdir)


if __name__ == "__main__":
    main()
