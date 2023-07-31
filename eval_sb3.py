import argparse
import os
import json
import numpy as np
import pandas as pd
import copy
import sys

import numpy as np
from tqdm import tqdm
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, HumanOutputFormat, DEBUG
from stable_baselines3.sac import SAC

from gym.wrappers.time_limit import TimeLimit

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import probes
from IBM_env import IBMEnv
from Incompact3D_params import SolverParams
def create_env(config, horizon, n_env):

    def _init():
        cf = copy.deepcopy(config)
        cf['index'] = n_env

        return Monitor(TimeLimit(IBMEnv(cf), max_episode_steps=horizon))

    return _init



def read_force_output(cwd):
    
    force_f = open(os.path.join(cwd, 'eval/env_0/aerof6.dat'), 'r')
    force_vals = [line.split() for line in force_f.readlines()]
    drag_vals = [float(line[0]) for line in force_vals]
    lift_vals = [float(line[1]) for line in force_vals]

    return (np.array(drag_vals), np.array(lift_vals))


def one_run(duration, solver_params, baseline):
    nb_actuations = int(np.ceil(env_params['max_iter'] / solver_params.step_iter))
    output_dir = 'baseline_run/' if baseline_run else 'test_run/'

    if os.path.exists(output_dir):
        print(f'Test result directory already exists. Delete {os.path.join(os.getcwd(), output_dir)} before rerunning. Aborting...')
        exit()

    #os.makedirs(example_environment.cwd + 'snapshots')

    print("Running baseline") if baseline else print("Start simulation")
    state = example_environment.reset()
    dt = solver_params.dt
    simulation_duration = 51000 * dt

    action_step_size = simulation_duration / nb_actuations  # Duration of 1 train episode / actions in 1 episode
    action_steps = int(duration / action_step_size)
    solver_step_iter = int(action_step_size / dt)
    episode_forces = np.empty([action_steps*(solver_step_iter - 2), 2]) # drag idx 0, lift idx 1
    angles = np.empty([action_steps, 2])
    rewards = np.empty((action_steps,))
    t = range(action_steps)
    cwd = os.getcwd()
    #print(cwd)
    for iter in t:

        action, _ = agent.predict(state, deterministic=True) if not baseline else np.array(env_params['default_action'])
        state, rw, done, _ = example_environment.step(action)
        #print(state)
        #if os.path.exists(example_environment.cwd + 'snapshots/snapshot0000.vtr'):
            # Rename snapshots so the solver does not overwrite them
            #os.rename(example_environment.cwd + 'snapshots/snapshot0000.vtr', example_environment.cwd + f'snapshots/snapshot{iter}.vtr')

        (drag, lift) = read_force_output(cwd)
        #print(drag)
        episode_forces[iter*(solver_step_iter - 2):(iter + 1)*(solver_step_iter - 2), 0] = drag
        episode_forces[iter*(solver_step_iter - 2):(iter + 1)*(solver_step_iter - 2), 1] = lift
        angles[iter] = action
        rewards[iter] = rw

    print(f'Average drag: {np.mean(episode_forces[:, 0])}\nAverage lift: {np.mean(episode_forces[:, 1])}')

    df_forces = pd.DataFrame(episode_forces, columns=['Drag', 'Lift'])
    df_angles = pd.DataFrame(angles, columns=['Top Flap', 'Bottom Flap'])
    df_rewards = pd.DataFrame(rewards, columns=['Reward'])

    os.makedirs(output_dir)
    df_forces.to_csv(output_dir + 'forces.csv', sep=';', index=False)
    df_angles.to_csv(output_dir + 'angles.csv', sep=';', index=False)
    df_rewards.to_csv(output_dir + 'rewards.csv', sep=';', index=False)


# Argument Parsing
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reset-iterations", required=False, help="solver iterations for reset state generation. 0 for no reset generation", type=int, default=0)
ap.add_argument("-b", "--baseline-run", required=False, help="If true, compute values for a run with no control output", type=bool, default=False)
ap.add_argument("-t", "--restore", required=False, help="Directory from which to restore the model.", type=str, default=None)
ap.add_argument("-a", "--algorithm", required=False, help="Which algorithm to run, one of (SAC, TQC)", type=str)
args = vars(ap.parse_args())

params = json.load(open('params.json', 'r'))

reset_iterations = args["reset_iterations"]
baseline_run = args["baseline_run"]
restoredir = args["restore"]
algo = args["algorithm"]

cwd = os.getcwd()
exec_dir = os.path.join(cwd, 'Incompact3d_Flaps')
reset_dir = os.path.join(cwd, 'reset_env')

env_params = params['env_params']
solver_params = SolverParams(exec_dir)
probe_layout = probes.ProbeLayout(env_params['probe_layout'], solver_params)
probe_layout.generate_probe_layout_file(os.path.join(exec_dir, 'probe_layout.txt'))

if not reset_iterations == 0:
    IBMEnv.GenerateRestart(reset_dir, exec_dir, reset_iterations)

nb_actuations = int(np.ceil(env_params['max_iter'] / solver_params.step_iter))

config = json.load(open(os.path.join(cwd, 'params.json'), 'r'))

env_config = {
    'cwd': cwd,
    'exec_dir': exec_dir,
    'reset_dir': reset_dir,
    'probe_layout': probe_layout,
    'solver_params': solver_params,
    'env_params': env_params,
    'logdir': None,
    'index': 512,
    'eval': True
}

config['env_config'] = env_config


if __name__ == '__main__':
    saver_restore = '/home/jackyzhang/anaconda3/bin/IBM-Flaps-RL-trialsb3/rl-training/saver_data/TQC_Flaps_Model_FreeMoving_69120_steps.zip'
    vecnorm_path = '/home/jackyzhang/anaconda3/bin/IBM-Flaps-RL-trialsb3/rl-training/saver_data/TQC_Flaps_Model_FreeMoving_vecnormalize_69120_steps.pkl'
    example_environment = SubprocVecEnv([create_env(env_config, nb_actuations, i) for i in range(1)], start_method='spawn')
    example_environment = VecNormalize.load(venv=example_environment, load_path=vecnorm_path)
    agent = TQC.load(saver_restore)
    one_run(120, solver_params, baseline_run)

