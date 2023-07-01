import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Train data transformations
train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.dropout.coarse_dropout.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0.4733, mask_fill_value=None, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.4913, 0.4821, 0.4465), std=(0.2470, 0.2434, 0.2615),p=1.0),
        ToTensorV2(),
    ]
)

# Test data transformations
test_transforms = A.Compose(
    [
        A.Normalize(mean=(0.4913, 0.4821, 0.4465), std=(0.2470, 0.2434, 0.2615),p=1.0),
        ToTensorV2(),
    ]
)


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):

    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

def dataloader(data_path,batch_size,train_transforms,test_transforms):
    trainset = Cifar10SearchDataset(root=data_path, train=True,download=True, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

    testset = Cifar10SearchDataset(root=data_path, train=False, download=True, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)
    classes = trainset.classes
    return trainloader, testloader, classes

def plot_sample_data(dataloader):
    batch_data, batch_label = next(iter(dataloader))
    fig = plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(torch.permute(batch_data[i], (1, 2, 0)))
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

class trainer:
    def __init__(self,model,device,optimizer,scheduler):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def getcorrectpredcount(self,prediction, labels):
        return prediction.argmax(dim=1).eq(labels).sum().item()

    def train(self,train_loader):
        self.model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            correct += self.getcorrectpredcount(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(train_loader))

    def test(self,test_loader):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

                correct += self.getcorrectpredcount(output, target)

        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def evaluate_all_class(self,classes,test_loader):

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def evaluate_model(model, loader, device):
    cols, rows = 4, 6
    figure = plt.figure(figsize=(20, 20))
    for index in range(1, cols * rows + 1):
        k = np.random.randint(0, len(loader.dataset))  # random points from test dataset

        img, label = loader.dataset[k]  # separate the image and label
        img = img.unsqueeze(0)  # adding one dimention
        pred = model(img.to(device))  # Prediction

        figure.add_subplot(rows, cols, index)  # making the figure
        plt.title(f"Predcited label {pred.argmax().item()}\n True Label: {label}")  # title of plot
        plt.axis("off")  # hiding the axis
        plt.imshow(img.squeeze(), cmap="gray")  # showing the plot

    plt.show()


