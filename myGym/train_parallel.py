import os
import threading

algos = [' ppo ',' ppo2 ','sac','trpo']
api = '/home/michal/code/myGym/myGym'
configfile = 'configs/train.json'
script_path = api + '/train.py'

def train(algo):
    os.system('cd {api};python {script_path} --config {configfile} --algo {algos}'.format(script_path=script_path, api=api, configfile=configfile, algos=algo))
    
    

if __name__ == '__main__':
    threads = []
    for i, algo in enumerate(algos):
        thread = threading.Thread(target=train, args=(algo,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


    


# os.system("./train.py --task_type reach --algo ppo2 --reward_type gt --robot kuka -ba step --env_name CrowWorkspaceEnv-v0 --steps 300000 -to hammer -w tensorflow -g 0 -p pybullet  -r 0 -c 6 -dt euclidian -i 0 -t 1 -f 0 -d opengl")