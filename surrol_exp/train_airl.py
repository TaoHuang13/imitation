import argparse
from matplotlib import use
import numpy as np
import gym
import torch
import random
import os
from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import Trajectory
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from stable_baselines3.common.vec_env import DummyVecEnv

MAX_STEPS = 50
OBS_DICT = ['observation', 'desired_goal']

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--env', default='NeedlePick-v0')
    parser.add_argument('--num_demo', default='100', type=int)
    parser.add_argument('--work_dir', default='./surrol_exp/logs', type=str)
    parser.add_argument('--seed', default=1, type=int)

    args = parser.parse_args()
    return args

def flatten_obs(obs):
    obs_ = np.append(obs[OBS_DICT[0]], obs[OBS_DICT[1]])
    return obs_

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

class FlattenDictWrapper(gym.ObservationWrapper):
    """Flattens selected keys of a Dict observation space into
    an array.
    """
    def __init__(self, env, dict_keys):
        super(FlattenDictWrapper, self).__init__(env)
        self.dict_keys = dict_keys

        # Figure out observation_space dimension.
        size = 0
        for key in dict_keys:
            shape = self.env.observation_space.spaces[key].shape
            size += np.prod(shape)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

    def observation(self, observation):
        assert isinstance(observation, dict)
        obs = []
        for key in self.dict_keys:
            obs.append(observation[key].ravel())
        return np.concatenate(obs)

import imageio
import os
import numpy as np

class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, fps=10):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            # try:
            #     frame = env.render(
            #         mode='rgb_array',
            #         height=self.height,
            #         width=self.width,
            #     )
                
            # except:
            frame = env.render(
                mode='rgb_array',
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import shutil
import torch
import torchvision
import numpy as np
from termcolor import colored

FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'), ('step', 'S', 'int'),
            ('duration', 'D', 'time'), ('episode_reward', 'R', 'float'),
            ('batch_reward', 'BR', 'float'), ('actor_loss', 'A_LOSS', 'float'),
            ('critic_loss', 'CR_LOSS', 'float')
        ],
        'eval': [('step', 'S', 'int'), ('episode_reward', 'ER', 'float')]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb=True, config='rl'):
        self._log_dir = log_dir
        if use_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train.log'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, 'eval.log'),
            formating=FORMAT_CONFIG[config]['eval']
        )

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        if self._sw is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step):
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_image(self, key, image, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_image(key, image, step)

    def log_video(self, key, frames, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._eval_mg.dump(step, 'eval')

def main():
    args = parse_args()

    env = DummyVecEnv([lambda: FlattenDictWrapper(gym.make(args.env), OBS_DICT)] * 1)

    env.seed(args.seed)
    set_seed_everywhere(args.seed)

    exp_name = 'airl-' + args.env + '-s' + str(args.seed)
    args.work_dir = args.work_dir + '/' + exp_name

    make_dir(args.work_dir)

    video_dir = make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir)

    L = Logger(args.work_dir, use_tb=False)

    demoData = np.load('/home/zac/SurRoL/surrol/data/demo/data_' + args.env + '_random_' + str(args.num_demo) + '.npz', allow_pickle=True)
    demoData_obs = demoData['obs']
    demoData_acs = demoData['acs']
    demoData_infos = demoData['info']

    assert demoData_obs.shape[0] == args.num_demo

    trajectories = []
    for i in range(args.num_demo):
        trajectory = {'obs':[], 'acts':[], 'infos':[]}
        trajectory['obs'].append(flatten_obs(demoData_obs[i][0]))

        for j in range(MAX_STEPS):
            trajectory['obs'].append(flatten_obs(demoData_obs[i][j+1]))
            trajectory['acts'].append(demoData_acs[i][j])
            trajectory['infos'].append(demoData_infos[i][j])
            
        trajectories.append(Trajectory(**trajectory, terminal=True))

    transitions = flatten_trajectories(trajectories)

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=100,
        ent_coef=1e-2,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=2000,
        policy_kwargs={'net_arch': [256, 256, 256]}
    )

    reward_net = BasicShapedRewardNet(
        env.observation_space, env.action_space, normalize_input_layer=RunningNorm
    )

    airl_trainer = AIRL(
        demonstrations=trajectories,
        demo_batch_size=64,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=airl_trainer.policy
    )

    bc_trainer.train(n_epochs=50)
    reward_before_training, sr = evaluate_policy(airl_trainer.policy, env, video, 0, L, n_eval_episodes=20)
    print(f"Reward/SR before training: {reward_before_training}/{sr}")

    for i in range(50):
        airl_trainer.train(2000)
        reward_after_training, sr = evaluate_policy(airl_trainer.policy, env, video, i+1, L, n_eval_episodes=20)
        print(f"Reward after training: {reward_after_training}/{sr}")

if __name__ == '__main__':
    main()
    

