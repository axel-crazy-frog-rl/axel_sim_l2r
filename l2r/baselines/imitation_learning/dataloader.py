import torch 
import os
import numpy as np 

class carDataLoader(torch.utils.data.Dataset):

    def __init__(self, data_path, episode): 

        self.data_path = data_path
        self.episode = episode

        """
            1. read all files (npz)
            2. load them all and store it in an array 
            2. From the __getitem__ function return image and control inputs
        """ 
        
        self.all_files = os.listdir(os.path.join(self.data_path, self.episode))

        self.length = len(self.all_files)
        self.imgs = []
        self.controls = []

        for i in range(self.length):
            transistion = np.load(os.path.join(self.data_path, self.episode, self.all_files[i]))
            img = transistion['img']
            control = transistion['action']
            offset = np.array([1, 1])
            control += offset
            control *= 10

            if img is not None and control is not None:
                self.imgs.append(img)
                self.controls.append(control)

            # self.imgs = np.array(self.imgs)
            # self.controls = np.array(self.controls)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):       
        img = self.imgs[ind]
        img = torch.tensor(np.swapaxes(img, 2, 0), dtype = torch.float)
        controls = torch.tensor(self.controls[ind], dtype = torch.float)

        if img.dim() != 3:
            print(self.all_files[ind])
        if controls.dim() != 1:
            print(self.all_files[ind])

        return (img, controls)
    


# train_dataset = carDataLoader(data_path = "/home/arjun/Desktop/fall23/idl/project/thruxton/train/", episode = "episode_0")
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, num_workers = 5, shuffle = True)

# for i, data in enumerate(train_loader):
#     print(f"img :  {data[0].shape}")
#     print(f"control : {data[1]}")