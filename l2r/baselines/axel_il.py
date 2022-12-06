# ========================================================================= #
# Filename:                                                                 #
#    random.py                                                              #
#                                                                           #
# Description:                                                              #
#    an agent that randomly chooses actions                                 #
# ========================================================================= #

from core.templates import AbstractAgent
from envs.env import RacingEnv

import cv2
import torch
from torchvision import models, transforms, utils
from PIL import Image
from baselines.imitation_learning.imitation_train import imitation_learn
import torchvision
import numpy as np

VELOCITY_IDX_LOW = 3
VELOCITY_IDX_HIGH = 6

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

class AxelILActionAgent(AbstractAgent):
    """Reinforcement learning agent that simply chooses random actions.

    :param dict training_kwargs: training keyword arguments
    """
    def __init__(self, training_kwargs):
        self.num_episodes = training_kwargs['num_episodes']
        print('Running Axel IL')
        vgg18 = torchvision.models.resnet18(pretrained = True)
        self.model_acc = imitation_learn(vgg18)
        self.model_acc.to(device=DEVICE)
        model_weight = "/home/mrsd-lab/Documents/IDL/project/axel_sim_l2r/l2r/baselines/imitation_learning/checkpoints/base_imit_steering_learning.pth"
        checkpoint = torch.load(model_weight)
        self.model_acc.load_state_dict(checkpoint['model_state_dict'])
        self.model_acc.eval()

    def race(self):
        """Demonstrative training method.
        """
        for e in range(self.num_episodes):
            print(f'Episode {e+1} of {self.num_episodes}')
            ep_reward = 0
            state, done = self.env.reset(), False

            while not done:
                proc_obs,velo = self.process_observation(state)
                action = self.select_action(proc_obs,velo)
                print('*********** {}'.format(action))
                state, reward, done, info = self.env.step(action)
                ep_reward += reward

            print(f'Completed episode with total reward: {ep_reward}')
            print(f'Episode info: {info}\n')

    def process_observation(self,state):
        obs = state['CameraFrontRGB']
        img = Image.fromarray(obs)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0., std=1.)
        ])
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        # o = torch.as_tensor(cv2.resize(o, (im_w, im_h)),
        #                     dtype=torch.float32, device=DEVICE)
        # o = o / 255
        
        # get velocity
        pose = state['pose']
        v = np.linalg.norm(pose[VELOCITY_IDX_LOW:VELOCITY_IDX_HIGH])
        return img,v
        
    def select_action(self, obs,velo):
        """Select a random action from the action space.

        :return: random action to take [steering,acceleration]
        :rtype: numpy array
        """
        steer_pred = self.model_acc(obs.to('cuda'))
        if velo>3.0:
            accel = 0.0
        else:
            accel = 1.0
        return np.array([steer_pred.detach().cpu().numpy()/10-1,accel])
        return self.env.action_space.sample()

    def create_env(self, env_kwargs, sim_kwargs):
        """Instantiate a racing environment

        :param dict env_kwargs: environment keyword arguments
        :param dict sim_kwargs: simulator setting keyword arguments
        """
        self.env = RacingEnv(
            max_timesteps=env_kwargs['max_timesteps'],
            obs_delay=env_kwargs['obs_delay'],
            training=False,
            not_moving_timeout=env_kwargs['not_moving_timeout'],
            controller_kwargs=env_kwargs['controller_kwargs'],
            reward_pol=env_kwargs['reward_pol'],
            reward_kwargs=env_kwargs['reward_kwargs'],
            action_if_kwargs=env_kwargs['action_if_kwargs'],
            pose_if_kwargs=env_kwargs['pose_if_kwargs'],
            cameras=env_kwargs['cameras']
        )

        self.env.make(
            level=sim_kwargs['racetrack'],
            multimodal=env_kwargs['multimodal'],
            driver_params=sim_kwargs['driver_params']
        )

        print('Environment created with observation space: ')
        for k, v in self.env.observation_space.spaces.items():
            print(f'\t{k}: {v}')
