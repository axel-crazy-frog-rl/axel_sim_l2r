import torch 
import torchvision
from torchsummaryX import summary
from tqdm import tqdm
import numpy as np


import matplotlib.pyplot as plt

from dataloader import carDataLoader

device = 'cuda'
model = torchvision.models.resnet18(pretrained = True)
num_cnn_feat = model.fc.in_features

for feats in model.children():
    feats.requires_grad_(False)

vgg18 = torchvision.models.resnet18(pretrained = True)
# num_cnn_feat = model.fc.in_features

# for feats in model.children():
#     feats.requires_grad_(False)

class imitation_learn(torch.nn.Module):
    def __init__(self, cnn):
        super(imitation_learn, self).__init__()

        self.activation = {}

        self.backbone = cnn.eval()
        num_cnn_feat = cnn.fc.in_features

        self.model_linear = torch.nn.Sequential(
                                torch.nn.Linear(num_cnn_feat, 256),
                                torch.nn.GELU(),
                                torch.nn.Linear(256, out_features = 1) #doing this because output is between -1 and 1
                                )

    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def forward(self, x):        
        h3 = self.backbone.avgpool.register_forward_hook(self.getActivation('comp'))
        x = self.backbone(x)
        x = self.activation['comp']
        x = x.squeeze(2)
        x = x.squeeze(2)

        x = self.model_linear(x)

        return x

model = imitation_learn(vgg18)

model.to(device=device)
model_weight = "/home/arjun/Desktop/fall23/idl/project/src/base_imit_acc_learning.pth"
checkpoint = torch.load(model_weight)
model.load_state_dict(checkpoint['model_state_dict'])


model.eval()


test_dataset = carDataLoader(data_path = "/home/arjun/Desktop/fall23/idl/project/thruxton/val/", episode = "episode_2")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, num_workers = 12)

criterion = torch.nn.MSELoss()

test_loss = 0
cum_pred_controls = None
cum_actual_controls = None

batch_bar = tqdm(total = len(test_loader), dynamic_ncols = True, leave = True, position = 0, desc = 'test')

for i, data in enumerate(test_loader):
    img, controls = data
    img, controls = img.to(device=device), controls.to(device=device)
    pred = model(img)
    loss = criterion(pred.squeeze(1), controls[:,0])
    
    if cum_pred_controls is None:
        cum_pred_controls = pred.detach().cpu().numpy()
    else:
        cum_pred_controls = np.vstack((cum_pred_controls, pred.detach().cpu().numpy()))

    controls_expand = controls[:,0].reshape(-1, 1)
    if cum_actual_controls is None:
        cum_actual_controls = controls_expand.detach().cpu().numpy()
    else:
        cum_actual_controls = np.vstack((cum_actual_controls, controls_expand.detach().cpu().numpy()))


    test_loss += loss
    batch_bar.update()

test_loss /= len(test_loader)
print(test_loss.item())
# summary(model, torch.rand(17, 3, 384, 512))


import matplotlib.pyplot as plt

plt.figure()
plt.plot(cum_actual_controls[:, 0]/10 - 1, label = 'actual',color='green', marker='o', markersize=1)
plt.plot(cum_pred_controls[:,0]/10 - 1, label = 'predicted', color='red', marker='x', markersize=1)
plt.title("Steering control")
plt.legend()
plt.show()

# plt.figure()
# plt.plot(cum_actual_controls[:, 1], label = 'actual')
# plt.plot(cum_pred_controls[:, 1], label = 'predicted')
# plt.title("Acceleration control")
# plt.legend()
# plt.show()

