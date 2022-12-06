# ========================================================================= #
# Filename:                                                                 #
#    il.py                                                                  #
#                                                                           #
# Description:                                                              #
#    imitation learning agent                                               #
# ========================================================================= #
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from core.templates import AbstractAgent
from envs.env import RacingEnv

from baselines.il.il_model import CILModel
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE


class ILAgent(AbstractAgent):
    """Reinforcement learning agent that simply chooses random actions.

    :param training_kwargs: training keyword arguments
    :type training_kwargs: dict
    """

    def __init__(self, model_params, training_kwargs):
        self.num_episodes = training_kwargs['num_episodes']
        self.normalize = transforms.Compose([
            #             transforms.Resize(256),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[125.61341389, 118.31236235, 114.9765454],
                                 std=[68.98788514, 64.9655252, 64.56587821])
        ])
        self.model = CILModel(model_params)
        # self.model = self.model.to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=training_kwargs['learning_rate'])
        self.mseLoss = nn.MSELoss()
        self.model = self.model.to(DEVICE)
        self.save_path = training_kwargs['save_path']
        self.checkpoint_name = training_kwargs['checkpoint']

        if training_kwargs['inference_only']:
            self.model.eval()

    def select_action(self, x, a):
        """Select an action
        """
        out = self.model(x, a)
        return out

    def il_train(self, data_loader, **il_kwargs):

        n_epochs = il_kwargs['n_epochs']
        eval_every = il_kwargs['eval_every']

        for i in range(n_epochs):

            print('Training: epoch {}'.format(i))

            for imgs, sensors, target in data_loader:
                '''
                Input for NN:
                    imgs: n x 3 x H x W
                    sensors: n x Dim
                Target: n x 2
                '''

                imgs, sensors, target = imgs.transpose(2, 3).type(torch.FloatTensor).to(DEVICE), \
                    sensors.to(DEVICE), target.to(DEVICE)

                assert imgs.shape == torch.Size(
                    [imgs.shape[0], 3, 512, 384]), "FATAL: unexpectd image shape"

                # The output(branches) is a list of 5 branches results
                # each branch is with size [120,3]
                self.model.zero_grad()

                # TODO: Match I/O
                out = self.model(imgs, sensors)

                loss = self.mseLoss(out, target)
                loss.backward()
                self.optimizer.step()

            if (i) % eval_every == 0:
                print('Eval / save, eval_every: {}'.format(eval_every))
                # self.eval()
                self.save_model(i)

    def eval(self):
        """
        evaluate the agent
        """
        print("Model evaluation")

        model_cpu = self.model.cpu()

        for e in range(self.num_episodes):
            print('=' * 10 + f' Episode {e+1} of {self.num_episodes} ' + '=' * 10)
            ep_reward, ep_timestep, best_ep_reward = 0, 0, 0
            obs = self.env.reset()
            obs, reward, done, info = self.env.step([0, 1])

            while not done:
                # print(obs)
                print(obs['pose'].shape)
                # (sensor, img) = obs
                img = (obs['CameraFrontRGB'])
                img = Image.fromarray(img)
                img = self.normalize(img)
                img = img.unsqueeze(0)

                sensor = obs['pose']

                action = model_cpu(img, torch.FloatTensor(sensor).unsqueeze(0))
                print(f"action {action}")
                action = torch.clamp(action, -1, 1)
                action = action.squeeze(0).detach().numpy()
                action[0] = action[0]*5
                obs, reward, done, info = self.env.step(
                    action)
                ep_reward += reward
                ep_timestep += 1

                # Save if best (or periodically)
                if (ep_reward > best_ep_reward and ep_reward > 250):
                    print(f'New best episode reward of {round(ep_reward,1)}!')
                    best_ep_reward = ep_reward
                    path_name = f'{self.save_path}il_episode_{e}_best.pt'
                    torch.save(self.model.state_dict(), path_name)

            print(f'Completed episode with total reward: {ep_reward}')
            print(f'Episode info: {info}\n')
    
    def eval_dl(self, dataloader):
            """
            evaluate the agent
            """
            print("Model evaluation")

            model_cpu = self.model.cpu()
            ep_reward, ep_timestep, best_ep_reward = 0, 0, 0
            obs = self.env.reset()
            obs, reward, done, info = self.env.step([0, 1])
            DEVICE = 'cpu'

            field = ['steering request', 
                     'gear request', 
                     'mode', 
                     'direction velocity in m/s',
                     'direction velocity in m/s',
                     'direction velocity in m/s',
                     'directional acceleration in m/s^2', 
                     'directional acceleration in m/s^2',
                     'directional acceleration in m/s^2',
                     'directional angular velocity', 
                     'directional angular velocity', 
                     'directional angular velocity',
                    'vehicle yaw', 
                    'vehicle pitch', 
                    'vehicle roll', 
                    'center of vehicle coordinates(y)',
                    'center of vehicle coordinates(x)',
                    'center of vehicle coordinates(z)',
                   ' wheel revolutions per minute',
                   ' wheel revolutions per minute',
                   ' wheel revolutions per minute',
                   ' wheel revolutions per minute',
                    'wheel braking', 
                    'wheel braking', 
                    'wheel braking', 
                    'wheel braking',
                    'wheel torque (per wheel)',
                    'wheel torque (per wheel)',
                    'wheel torque (per wheel)',
                    'wheel torque (per wheel)']
            
            for imgs, sensors, target in dataloader:
                        '''
                        Input for NN:
                            imgs: n x 3 x H x W
                            sensors: n x Dim
                        Target: n x 2
                        '''

                        imgs, sensors, target = imgs.transpose(2, 3).type(torch.FloatTensor).to(DEVICE), \
                            sensors.to(DEVICE), target.to(DEVICE)

                        # assert imgs.shape == torch.Size(
                            # [imgs.shape[0], 3, 512, 384]), "FATAL: unexpectd image shape"

                        # USE SIMULATOR IMAGES
                        img = (obs['CameraFrontRGB'])
                        # img = np.swapaxes(img,0,1)
                        # img = np.swapaxes(img,1,2)
                        # img = np.swapaxes(img,1,0) #c w h
                        img = Image.fromarray(img)
                        # img = torch.Tensor(img)
                        # img = torch.unsqueeze(img, 0) #1 c w h
                        # img = img.transpose(1, 3) 
                        img = self.normalize(img)
                        img = img.unsqueeze(0)
                        # print(f"img sim : {torch.unique(img)} {img.shape}")

                        # USE DATALOADER IMAGES
                        # img_pil = imgs[0]
                        # img_pil = torch.Tensor(img_pil)
                        # img_pil = torch.unsqueeze(img_pil, 0)
                        # img_single = img_single.transpose(1, 3)
                        
                        # img_single = torch.FloatTensor(img_single).unsqueeze(0)
                        # print(f"img {img.shape} img_single {img_single.shape}")
                        # img_single = self.normalize(img_pil)
                        action = model_cpu(img, torch.FloatTensor(obs['pose']).unsqueeze(0))


                        # SHOW IMAGE FROM DATALOADER
                        img_single = imgs[0] # 1 x c x h x w
                        img_single = img_single.squeeze(0) # c x h x w
                        img_single = img_single.transpose(0, 2)
                        img_single = img_single.detach().cpu().numpy()
                        # print(f"img {np.unique(img_single)}")
                        img_single = 255 * img_single.astype('uint8')
                        # print(f"img {img_single.shape}")
                        img_single = Image.fromarray(img_single)
                        # img_single.show()
                        
                        
                        # print(f"action {action[0]}")
                        # print(f"laoder data {sensors[0]}")
                        # print(f"sim data {obs['pose']}")
                        # imgs = imgs[0].squeeze(0)
                        # print(imgs.shape)
                        # img_pil = Image.fromarray(imgs.transpose(3,1))
                        # img_pil.show()

                        # img_pil = Image.fromarray(obs['CameraFrontRGB'])
                        # img_pil.show()
                        
                    #     print(imgs)
                    #     print("------------------------------------------")
                    #     img = torch.FloatTensor(Image.fromarray(obs['CameraFrontRGB'])).unsqueeze(
                    # 0).transpose(1, 3)
                    #     print(self.normalize(img))

                        # for i in range(30):
                        #     print(f"field {field[i]} data_loader {sensors[0][i]} sim_data {obs['pose'][i]} diff {abs(sensors[0][i] - obs['pose'][i])}")
                        print(f"action ; {action[0]}")
                        action = torch.clamp(action[0], -1, 1)
                        
                        obs, reward, done, info = self.env.step(
                            action.squeeze(0).detach().numpy())
                        
                        ep_reward += reward
                        ep_timestep += 1

            print(f'Completed episode with total reward: {ep_reward}')
            print(f'Episode info: {info}\n')

    def save_model(self, e):
        path_name = f'{self.save_path}il_episode_{e}.pt'
        torch.save(self.model.state_dict(), path_name)

    def load_model(self):
        path = f'{self.save_path}{self.checkpoint_name}'
        print(f"LOADING SAVED MODEL")
        self.model.load_state_dict(torch.load(path))

    def create_env(self, env_kwargs, sim_kwargs):
        """Instantiate a racing environment

        :param env_kwargs: environment keyword arguments
        :type env_kwargs: dict
        :param sim_kwargs: simulator setting keyword arguments
        :type sim_kwargs: dict
        """
        self.env = RacingEnv(
            max_timesteps=env_kwargs['max_timesteps'],
            obs_delay=env_kwargs['obs_delay'],
            not_moving_timeout=env_kwargs['not_moving_timeout'],
            controller_kwargs=env_kwargs['controller_kwargs'],
            reward_pol=env_kwargs['reward_pol'],
            reward_kwargs=env_kwargs['reward_kwargs'],
            action_if_kwargs=env_kwargs['action_if_kwargs'],
            #camera_if_kwargs=env_kwargs['camera_if_kwargs'],
            pose_if_kwargs=env_kwargs['pose_if_kwargs'],
            #logger_kwargs=env_kwargs['pose_if_kwargs'],
            cameras=env_kwargs['cameras']
        )

        self.env.make(
            level=sim_kwargs['racetrack'],
            multimodal=env_kwargs['multimodal'],
            driver_params=sim_kwargs['driver_params']
        )
        
        # self.env.make(
        #     level=sim_kwargs['racetrack'],
        #     multimodal=env_kwargs['multimodal'],
        #     driver_params=sim_kwargs['driver_params'],
        #     camera_params=sim_kwargs['camera_params'],
        #     sensors=sim_kwargs['active_sensors']
        # )
