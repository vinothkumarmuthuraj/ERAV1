import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision
# from torchscan import summary
from torch.optim.lr_scheduler import OneCycleLR

from utils import dataloader, plot_sample_data, trainer, evaluate_model#, train_transforms, test_transforms
from models import *
from utils import *

data_path = r"data"
batch_size = 512

trainloader,testloader, classes = dataloader(data_path, batch_size)#, train_transforms,test_transforms)

batch_data, batch_label = next(iter(testloader))
plot_sample_data(testloader)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = ResNet18().to(device)
summary(model, input_size=(3, 32, 32))
print(summary)

EPOCHS = 1
optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)  # large learning rate


scheduler = OneCycleLR(
        optimizer,
        max_lr=4.51E-02,
        steps_per_epoch=len(trainloader),
        epochs=EPOCHS,
        pct_start=1/EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )

trainer = trainer(model, device, optimizer, scheduler)


for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    trainer.train(trainloader)
    trainer.test(testloader)
    scheduler.step()

plot_misclassified_images(model, testloader, classes, device)
plot_grad_cam_images(model, testloader, classes, device)