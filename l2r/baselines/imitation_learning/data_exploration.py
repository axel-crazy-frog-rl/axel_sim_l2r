import numpy as np 
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision
import torch 
from torch import nn, Tensor
dataset_dir = "/home/arjun/Desktop/fall23/idl/project/thruxton/train/episode_0/"
# sample_file = os.path.join(dataset_dir, "transitions_0.npz")

model = torchvision.models.resnet18(pretrained = True)
num_cnn_feat = model.fc.in_features

files = os.listdir(dataset_dir)
imgs = []
fig = plt.figure()
dataset_prune = []
l = []

from typing import Dict, Iterable, Callable

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features

resnet_features = FeatureExtractor(model, layers=["layer4"])

for f in files:
    transition = np.load(os.path.join(dataset_dir, f))
    l = [f, transition.files]
    # print(l)
    dataset_prune.append(l)
    img = transition['img'] # image from front camera
    img = np.swapaxes(img, 2, 0) # make into channel x h x w
    img = np.expand_dims(img, 0) # make into batch x channel x h x w

    data = transition['multimodal_data'] # (https://learn-to-race.readthedocs.io/en/latest/multimodal.html)
    action = transition['action'] # (steering, acc) : [[-1, 1], [-1, 1]] : [[left, right], [fwd, back]]

    if img.shape != (384, 512, 3):
        print(f)
    if action.shape != (2,):
        print(f)
    # print(f)
    # print(img.shape)
    # print(action.shape)
    # plt.title()
    # imgs.append([plt.imshow(img, animated = True)])

# anim = animation.ArtistAnimation(fig, imgs, interval = 500, blit = True, repeat = False, repeat_delay = 1000)
# print(f"Steeing : {action[0]} Acceleration {action[1]}")

# # plt.imshow(img)
# # plt.show()
# anim.save("episode1.mp4")