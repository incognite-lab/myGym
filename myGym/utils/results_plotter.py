import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas, json
import os
from glob import glob
import time, gym, csv
from typing import Tuple, Dict, Any, List, Optional

# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
plt.rcParams['svg.fonttype'] = 'none'

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

class LoadMonitorResultsError(Exception):
    """
    Raised when loading the monitor log fails.
    """
    pass

class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: (gym.Env) The environment
    :param filename: (Optional[str]) the location to save a log file, can be None for no log
    :param allow_early_resets: (bool) allows the reset of the environment before it is done
    :param reset_keywords: (tuple) extra keywords for the reset call, if extra parameters are needed at reset
    :param info_keywords: (tuple) extra information to log, from the information return of environment.step
    """
    EXT = "monitor.csv"
    file_handler = None

    def __init__(self,
                 env: gym.Env,
                 filename: Optional[str],
                 allow_early_resets: bool = True,
                 reset_keywords=(),
                 info_keywords=()):
        super(Monitor, self).__init__(env=env)
        self.t_start = time.time()
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.file_handler = open(filename, "wt")
            self.file_handler.write('#%s\n' % json.dumps({"t_start": self.t_start, 'env_id': env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.file_handler,
                                         fieldnames=('r', 'l', 't') + reset_keywords + info_keywords)
            self.logger.writeheader()
            self.file_handler.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: (np.ndarray) the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, "
                               "wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError('Expected you to pass kwarg {} into reset'.format(key))
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """
        Step the environment with the given action

        :param action: (np.ndarray) the action
        :return: (Tuple[np.ndarray, float, bool, Dict[Any, Any]]) observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": eplen, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info['episode'] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def close(self):
        """
        Closes the environment
        """
        super(Monitor, self).close()
        if self.file_handler is not None:
            self.file_handler.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return: (int)
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return: ([float])
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return: ([int])
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return: ([float])
        """
        return self.episode_times

def get_monitor_files(path: str) -> List[str]:
    """
    get all the monitor files in the given path

    :param path: (str) the logging folder
    :return: ([str]) the log files
    """
    return glob(os.path.join(path, "*" + Monitor.EXT))

def load_results(path: str) -> pandas.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv`` and ``*monitor.json``

    :param path: (str) the directory path containing the log file(s)
    :return: (pandas.DataFrame) the logged data
    """
    # get both csv and (old) json files
    monitor_files = (glob(os.path.join(path, "*monitor.json")) + get_monitor_files(path))
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, path))
    data_frames = []
    headers = []
    for file_name in monitor_files:
        with open(file_name, 'rt') as file_handler:
            if file_name.endswith('csv'):
                first_line = file_handler.readline()
                assert first_line[0] == '#'
                header = json.loads(first_line[1:])
                data_frame = pandas.read_csv(file_handler, index_col=None)
                headers.append(header)
            elif file_name.endswith('json'):  # Deprecated json format
                episodes = []
                lines = file_handler.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                data_frame = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            data_frame['t'] += header['t_start']
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values('t', inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame['t'] -= min(header['t_start'] for header in headers)
    # data_frame.headers = headers  # HACK to preserve backwards compatibility
    return data_frame

def rolling_window(array, window):
    """
    apply a rolling window to a np.ndarray

    :param array: (np.ndarray) the input Array
    :param window: (int) length of the rolling window
    :return: (np.ndarray) rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1, var_2, window, func):
    """
    apply a function to the rolling window of 2 arrays

    :param var_1: (np.ndarray) variable 1
    :param var_2: (np.ndarray) variable 2
    :param window: (int) length of the rolling window
    :param func: (numpy function) function to apply on the rolling window on variable 2 (such as np.mean)
    :return: (np.ndarray, np.ndarray)  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2


def ts2xy(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    return x_var, y_var


def plot_curves(xy_list, xaxis, title):
    """
    plot the curves

    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    """

    plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()


def plot_results(dirs, num_timesteps, xaxis, task_name):
    """
    plot the results

    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int or None) only plot the points below this value
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: (str) the title of the task to plot
    """

    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name)


def main():
    """
    Example usage in jupyter-notebook

    .. code-block:: python

        from stable_baselines import results_plotter
        %matplotlib inline
        results_plotter.plot_results(["./log"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")

    Here ./log is a directory containing the monitor.csv files
    """
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs='*', default=['./log'])
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--xaxis', help='Varible on X-axis', default=X_TIMESTEPS)
    parser.add_argument('--task_name', help='Title of plot', default='Breakout')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(folder) for folder in args.dirs]
    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.task_name)
    plt.show()


if __name__ == '__main__':
    main()