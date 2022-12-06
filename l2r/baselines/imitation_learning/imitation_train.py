import torch 
import torchvision
import os 
from torchsummaryX import summary
from tqdm import tqdm
import wandb
import numpy as np
from PIL import Image
from torchvision import models, transforms, utils

# from dataloader import carDataLoader

# from torchviz import make_dot

device = 'cuda'
# train_dataset = carDataLoader(data_path = "/home/arjun/Desktop/fall23/idl/project/thruxton/train/", episode = "episode_0")
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, num_workers = 10)

# val_dataset = carDataLoader(data_path = "/home/arjun/Desktop/fall23/idl/project/thruxton/train/", episode = "episode_1")
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, num_workers = 10)

# for i, data in enumerate(train_loader):
#     print(f"img :  {data[0].shape}")
#     print(f"control : {data[1]}")

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

class imitation_learn_acc(torch.nn.Module):
    def __init__(self, cnn):
        super(imitation_learn_acc, self).__init__()

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

# model = imitation_learn(vgg18)

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=0., std=1.)
# ])
# img = np.load("/home/arjun/Desktop/fall23/idl/project/thruxton/train/episode_0/transitions_1.npz")
# img = img['img']
# img = Image.fromarray(img)
# img = transform(img)
# img = torch.unsqueeze(img, 0)

# y = model(img)   
# make_dot(y, params=dict(list(model.named_parameters()) + [('x', img)])).render("model_output", cleanup = True, format = "png")


# summary(model.to(device=device), torch.rand(17, 3, 384, 512).to(device=device))

# learning_rate = 1e-4
# epoch = 200

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = torch.nn.MSELoss() #Defining Loss function 

# checkpoint = torch.load("base_imit_acc_learning.pth")

# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# def train(model, train_loader, optimizer, criterion, device = 'cuda'):
#     model.train()
#     batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True,
#                      leave=False, position=0, desc='Train')
#     train_loss = 0

#     #multi head training : https://gist.github.com/quanvuong/1b474dd6043a41c1258e5a279a0024c8


#     for i, data in enumerate(train_loader):

#         # TODO: Fill this with the help of your sanity check
#         img, controls = data
#         img, controls = img.to(device=device), controls.to(device=device)
        
#         optimizer.zero_grad()

#         pred = model(img)
#         loss = criterion(pred.squeeze(1), controls[:,1])
#         # loss_acceleration = criterion(pred_acc.squeeze(1), controls[:,1])

#         train_loss += loss #+ loss_acceleration

#         batch_bar.set_postfix(
#             loss_total=f"{train_loss/(i+1):.4f}",
#             loss_steer=f"{loss:.4f}",
#             lr=f"{optimizer.param_groups[0]['lr']}"
#         )

#         batch_bar.update()

#         loss.backward()
#         # model.zero_grad()
#         # loss_acceleration.backward()

#         optimizer.step()

#     # batch_bar.close()
#     train_loss /= len(train_loader)
    
#     return train_loss

# def eval(model, val_loader, criterion, device = 'cuda'):

#     model.eval()
#     batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True,
#                      leave=False, position=0, desc='Val')
#     val_loss = 0

#     for i, data in enumerate(val_loader):

#         # TODO: Fill this with the help of your sanity check
#         img, controls = data
#         img, controls = img.to(device=device), controls.to(device=device)
#         pred = model(img)
#         loss1 = criterion(pred.squeeze(1), controls[:,1])
#         # loss2 = criterion(acc.squeeze(1), controls[:,1])

#         val_loss += loss1 #+ loss2

#         batch_bar.set_postfix(
#             loss=f"{val_loss/(i+1):.4f}",
#             loss_steer=f"{loss1:.4f}",
#             lr=f"{optimizer.param_groups[0]['lr']}"
#         )

#         batch_bar.update()

#     # batch_bar.close()
#     val_loss /= len(val_loader)
    
#     return val_loss

# wandb.login(key="d9d9a797f003ed7a40866f3e05a007f21b007cdd")
# run = wandb.init(
#     name = "baseline acc model", ### Wandb creates random run names if you skip this field, we recommend you give useful names
#     reinit=False, ### Allows reinitalizing runs when you re-run this cell
#     project="IDL Project", ### Project should be created in your wandb account 
#     # config=config, ### Wandb Config for your run
# )

# best_valloss = 100
# for epoch in range(epoch):

#     train_loss = train(model, train_loader, optimizer, criterion)

#     curr_lr = float(optimizer.param_groups[0]['lr'])
    
#     # print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.08f}".format(
#     #     epoch + 1,
#     #     epoch,
#     #     train_acc,
#     #     train_loss,
#     #     curr_lr))
    
#     val_loss = eval(model, val_loader, criterion)
    
#     # scheduler.step()
# #    scheduler.step(val_acc)

#     print("Train Loss {:.04f}\t Val Loss {:.04f}".format(train_loss, val_loss))

#     wandb.log({"train_loss":train_loss, 'validation_loss': val_loss, "learning_Rate": curr_lr})
    
#     if val_loss <= best_valloss:
#       print("[INFO] Saving model")
#       torch.save({'epoch': epoch,
#               'model_state_dict': model.state_dict(),
#               'optimizer_state_dict': optimizer.state_dict(),
#               'train_loss': train_loss,
#               'val_loss' : val_loss},
#             "base_imit_acc_learning.pth")

#       best_valloss = val_loss
# #       wandb.save('checkpoint.pth')
# run.finish()

