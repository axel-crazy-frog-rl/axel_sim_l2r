'''
file: ddqn.py
author: Felix Yu
date: 2018-09-12
original: https://github.com/flyyufelix/donkey_rl/blob/master/donkey_rl/src/ddqn.py
'''
import os
import sys
import random
import argparse
#import signal

import numpy as np
#import gym
import cv2

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.initializers import normal, identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from racetracks.mapping import level_2_trackmap
import wandb
#import gym_donkeycar

#EPISODES = 3000
EPISODES = 3
img_rows, img_cols = 144, 144
# Convert image into Black and white
img_channels = 4 # We stack 4 frames

# Controller
T = 6

class DQNAgent:

    def __init__(self, state_size, action_space,racetrack, centerline_arr, train=True):
        self.t = 0
        self.max_Q = 0
        self.train = train
        
        # Get size of state and action
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = action_space

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if (self.train):
            self.epsilon = 0.02
            self.initial_epsilon = 0.02
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.001
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000
        
        self.config = {
            'batch_size': self.batch_size,
            'lr': self.learning_rate,
            'name' : 'ddqn',
            'episodes': EPISODES, 
        }


        # Create replay memory using deque
        self.memory = deque(maxlen=10000)

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()
        
        self.load_track(centerline_arr,racetrack)
        
        self.file = open('/home/mrsd-lab/Documents/learn-to-race/metrics/ddqn.txt', 'w')
        datarecord = "pct_complete" + "    " + "total_time" + "    " + "total_distance" + "    " + "average_speed_kph" + "    " + 'average_displacement_error' + "    " + 'trajectory_efficiency'+ "    " + 'trajectory_admissibility' + "    " + 'movement_smoothness' + "    " + 'timestep/sec' + '\n'
        self.file.write(datarecord)

    def load_track(self,centerline_arr, track_name=['VegasNorthRoad']):
        """Load trace track

        :param str track_name: 'VegasNorthRoad' or 'Thruxton'
        """
        assert len(track_name) == 1
        map_file, _ = level_2_trackmap(track_name[-1])

        # np.asarray(original_map['Racing'], dtype=np.float32).T
        raceline = centerline_arr
        self.race_x = raceline[:, 0]
        self.race_y = raceline[:, 1]
        self.raceline_length = self.race_x.shape[0]

        X_diff = np.concatenate([self.race_x[1:] - self.race_x[:-1],
                                 [self.race_x[0] - self.race_x[-1]]])
        Y_diff = np.concatenate([self.race_y[1:] - self.race_y[:-1],
                                 [self.race_y[0] - self.race_y[-1]]])
        self.race_yaw = np.arctan(Y_diff / X_diff)  # (L-1, n)
        self.race_yaw[X_diff < 0] += np.pi
        # Ensure the yaw is within [-pi, pi)
        self.race_yaw = (self.race_yaw + np.pi) % (2 * np.pi) - np.pi
        # pdb.set_trace()
        # self.race_yaw = np.asarray(original_map['RacingPsi'], dtype=np.float32)


    @staticmethod
    def unpack_state(state):
        state = state['pose']
        x = state[16]
        y = state[15]
        v = (state[4]**2 + state[3]**2 + state[5]**2)**0.5
        yaw = np.pi / 2 - state[12]
        # Ensure the yaw is within [-pi, pi)
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        return x, y, v, yaw
    
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        # 15 categorical bins for Steering angles
        model.add(Dense(15, activation="linear")) 

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        
        return model


    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        #print("++++++++++rgb shape: ",rgb.shape)
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def process_image(self, obs):
        obs = self.rgb2gray(obs)
        obs = cv2.resize(obs, (img_rows, img_cols))
        return obs
        

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        #print("random value+++++++++++++", self.action_space.sample()[0])
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()[0]       #(-1,1)
        else:
            #print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)
            #print("=======q value shape:",q_value.shape) (1,15)

            # Convert q array to steering value
            return linear_unbin(q_value[0])


    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore


    def train_replay(self):
        if len(self.memory) < self.train_start:
            return 0
        
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        loss = self.model.train_on_batch(state_t, targets)
        return loss


    def load_model(self, name):
        self.model.load_weights(name)


    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)
    
    def get_xref(self, idx, yaw, n_targets=T):
        """Get targets.

        :param int idx: index of the raceline the agent is nearest to
        :param float yaw: current vehicle heading
        :return: array of shape (4, n_targets)
        :rtype: numpy array
        """
        step_size = 8 # number of indicies b/w waypoints
        # TODO: Interval between waypoint have to change as well
        target_idxs = [(idx + step_size * t) %
                       self.raceline_length for t in range(1, 1 + n_targets)]
        target_x = [self.race_x[i] for i in target_idxs]
        target_y = [self.race_y[i] for i in target_idxs]

        target_yaw = np.array([self.race_yaw[i] for i in target_idxs])
        if np.any((target_yaw > 5 / 6 * np.pi) | (target_yaw < -5 / 6 * np.pi)):
            to2pi_flag = True
            target_yaw[target_yaw < 0] += 2 * np.pi
        else:
            to2pi_flag = False

        cdy = abs(target_yaw[-1] - yaw)
        mdy = abs(target_yaw[0] - target_yaw[-3])
        tdy = abs(target_yaw[0] - target_yaw[-1])
        self.vt = 50.0 # Arbitrary velo target
        vt = self.vt

        # crude velocity profile
        if tdy > 0.1 or cdy > 0.1:
            vt = self.vt * 0.9
        if tdy > 0.2 or cdy > 0.2:
            vt = self.vt * 0.7
        if tdy > 0.4 or cdy > 0.4:
            vt = self.vt * 0.5
        if tdy > 1.0 or mdy > 1.0:
            vt = self.vt * 0.3

        target_v = [vt] * n_targets

        return np.array([target_x, target_y, target_v, target_yaw], dtype=np.float32), to2pi_flag
        
# 
# 1. Train more with loss plots
# 2. Fix the throttle
# 3, Maybe add obstacles 
# 4. Try another RL policy



## Utils Functions ##

def linear_bin(a):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between -1 and 1

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.

    See Also
    --------
    linear_bin
    """
    if not len(arr) == 15:
        raise ValueError('Illegal array length, must be 15')
    b = np.argmax(arr)
    #print("======b value:", b)
    a = b * (2 / 14) - 1
    #print("=====afetr linear unbin:", a)
    return a



def run_ddqn(racetrack,env,args):
    '''
    run a DDQN training session, or test it's result, with the donkey simulator
    '''

    # only needed if TF==1.13.1
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=config)
    # K.set_session(sess)

    # Construct gym environment. Starts the simulator if path is given.
    #env = gym.make(args.env_name, exe_path=args.sim, port=args.port)

    # not working on windows...
    # def signal_handler(signal, frame):
    #     print("catching ctrl+c")
    #     env.unwrapped.close()
    #     sys.exit(0)

    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    # signal.signal(signal.SIGABRT, signal_handler)

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_space = env.action_space # Steering and Throttle
    #Box([-1.  0.], [1. 5.], (2,), float32)

    try:
        agent = DQNAgent(state_size, action_space, racetrack, env.centerline_arr, train=not args["test"])
        if agent.train:
            wandb.login(key="1dd96eb00b4613139aaed9f7c4440ea233b45a00") 
            # Create your wandb run
            run = wandb.init(
                name = agent.config["name"], ## Wandb creates random run names if you skip this field
                reinit = True, ### Allows reinitalizing runs when you re-run this cell
                #id = "asetlsif", ### Insert specific run id here if you want to resume a previous run
                #resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
                project = "deep_learning_project", ### Project should be created in your wandb account 
                config = agent.config ### Wandb Config for your run
            )
        
        throttle = args['throttle'] # Set throttle as constant value_
        episodes = []
        print("!!!!!!!!!!!!!!!!!!!loaded model path:", os.path.exists(args["loadmodel"]))
        if os.path.exists(args["loadmodel"]):
            print("!!!!!!!!!!!!!!!!!!!!load the saved model: ", args["loadmodel"])
            agent.load_model(args["loadmodel"])

        for e in range(EPISODES):
            print("Episode: ", e)

            done = False
            obs, info = env.reset()

            episode_len = 0
            train_loss = 0
            #print("****************obs:", obs["CameraFrontRGB"])
            #print("***********obs shape", obs["CameraFrontRGB"].shape) #(144,192,3)
            x_t = agent.process_image(obs["CameraFrontRGB"])
            

            s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*144*144*4       
            total_reward = 0
            next_obs = None
            yaw_diff = 0.0
            while not done:
                
                # if next_obs!=None:
                #     x, y, v, yaw = DQNAgent.unpack_state(next_obs)

                #     idx = info['track_idx']
                #     xref, to2pi_flag = agent.get_xref(idx, yaw)

                #     if to2pi_flag:
                #         if yaw < 0:
                #             yaw += 2 * np.pi
                #     x0 = [x, y, v, yaw]  # current state

                #     yaw_diff = (yaw-xref[3, 0])*180/np.pi
                #     #print(' Yaw diff {}'.format(yaw_diff))
                #     #print(f'v={v}')
                #     # print(f'Target: loc=({xref[0, 0]}, {xref[1, 0]}), v={xref[2, 0]}, yaw={xref[3, 0]*180/np.pi}')
                
                
                # Get action for the current state and go one step in environment
                steering = agent.get_action(s_t)
                throttle = args['throttle']
                # # throttle = (1.0-abs(steering))*args['throttle']
                # # print("========throttle: ",throttle)
                # if(abs(yaw_diff)<10 or v<2.0):
                #     throttle = args['throttle']
                # else:
                #     #print('Braking!!')
                #     throttle = -args['throttle']
                action = [steering, throttle]
                next_obs, reward, done, info = env.step(action)
                total_reward += reward

                x_t1 = agent.process_image(next_obs["CameraFrontRGB"])

                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
                s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #1x80x80x4

                # Save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(s_t, np.argmax(linear_bin(steering)), reward, s_t1, done)
                agent.update_epsilon()

                if agent.train:
                    loss = agent.train_replay()
                    train_loss += loss

                s_t = s_t1
                agent.t = agent.t + 1
                episode_len = episode_len + 1
                if agent.t % 30 == 0:
                    print("EPISODE",  e, "TIMESTEP", agent.t,"/ ACTION", action, "/ REWARD", reward, "/ EPISODE LENGTH", episode_len, "/ Q_MAX " , agent.max_Q)

                if done:

                    # Every episode update the target model to be same with model
                    agent.update_target_model()

                    episodes.append(e)
                    
                    train_loss /= episode_len
                    
                    

                    # Save model for each episode
                    if agent.train:
                        if e % 20 == 0 or e == EPISODES-1:
                            model_path = args["model_path"] + str(100+e) + ".h5"
                            agent.save_model(model_path)
                            print("!!!!!!!!!!!!!!!save to ",model_path)

                    # print("episode:", e, "  memory length:", len(agent.memory),
                    #     "  epsilon:", agent.epsilon, " episode length:", episode_len, " total reward:",total_reward)
                    print("\nEpisode {}/{}: \nTrain Loss {:.04f}\t Learning Rate {:.04f}\t Memory Length {:.04f}\t Epsilon {:.04f}\t Episode Length {:.04f}\t total_reward {:.04f}\t".format(
                        e + 1,
                        agent.config['episodes'],
                        train_loss,
                        K.eval(agent.model.optimizer.lr),
                        len(agent.memory),
                        agent.epsilon,
                        episode_len,
                        total_reward
                        ))
                    
                    print(f'[eval episode] {info}')
                    metrics = info["metrics"]
                    data_record = str(metrics["pct_complete"]) + "    " + str(metrics["total_time"]) + "    " + str(metrics["total_distance"]) + "    " + str(metrics["average_speed_kph"]) + "    " + str(metrics['average_displacement_error']) + "    " + str(metrics['trajectory_efficiency']) + "    " + str(metrics['trajectory_admissibility']) + "    " + str(metrics['movement_smoothness']) + "    " + str(metrics['timestep/sec'])  + '\n'
                    agent.file.write(data_record)
                    
                        # wandb.log({"loss":train_loss, 'Memory_Length': len(agent.memory), 'Total_Reward': total_reward,
                        #     'Episode_Length':episode_len, "learning_rate": K.eval(agent.model.optimizer.lr)})
        
        #run.finish()

    except KeyboardInterrupt:
        print("stopping run...")
    # finally:
    #     env.unwrapped.close()


# if __name__ == "__main__":

#     # Initialize the donkey environment
#     # where env_name one of:    
#     env_list = [
#        "donkey-warehouse-v0",
#        "donkey-generated-roads-v0",
#        "donkey-avc-sparkfun-v0",
#        "donkey-generated-track-v0"
#     ]

#     parser = argparse.ArgumentParser(description='ddqn')
#     parser.add_argument('--sim', type=str, default="manual", help='path to unity simulator. maybe be left at manual if you would like to start the sim on your own.')
#     parser.add_argument('--model', type=str, default="rl_driver.h5", help='path to model')
#     # parser.add_argument('--loadmodel', type=str, default="rl_driver_download.h5", help='path to model')
#     parser.add_argument('--loadmodel', type=str, default="train2/rl_driver_epoch1500.h5", help='path to model')
#     parser.add_argument('--test', action="store_true", help='agent uses learned model to navigate env')
#     parser.add_argument('--port', type=int, default=9091, help='port to use for websockets')
#     parser.add_argument('--throttle', type=float, default=0.3, help='constant throttle for driving')
#     parser.add_argument('--env_name', type=str, default='donkey-generated-roads-v0', help='name of donkey sim environment', choices=env_list)

#     args = parser.parse_args()

#     run_ddqn(args)

