import torch
import torchvision
import matplotlib.pyplot as plt

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
