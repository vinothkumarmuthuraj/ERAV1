import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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





