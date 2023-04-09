import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from torchvision import models
import torchvision.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from init_mnist import MNIST

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5))
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = torch.nn.Flatten()(out)   # 전결합층을 위해서 Flatten
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out

class CNNTrain():
    def __init__(self):
        use_cuda = False
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.mnist = MNIST()
        self.model = CNN()

    def model_train(self, force_train=False, learning_rate=0.001, train_epochs=15, batch_size=100):
        if(force_train == False and os.path.isfile("cnn_model/model.pt")):
            self.model_load()
            return None

        self.model.to(self.device)
        loss = torch.nn.CrossEntropyLoss().to(self.device) 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        total_batch = len(self.mnist.train_loader)
        print(f'Total Epochs : {total_batch}')

        for epoch in range(train_epochs):
            avg_cost = 0

            for X, Y in self.mnist.train_loader:
                # image is already size of (28x28), no reshape
                # label is not one-hot encoded
                X = X.to(self.device)
                Y = Y.to(self.device)

                optimizer.zero_grad()
                hypothesis = self.model(X)
                cost = loss(hypothesis, Y)
                cost.backward()
                optimizer.step()

                avg_cost += cost / total_batch
            print(f'[Epoch: {(epoch+1):>4}] cost = {avg_cost:>.9}')
        
        torch.save(self.model, "cnn_model/model.pt")
        print("CNN model saved")

    def model_load(self):
        self.model = torch.load("cnn_model/model.pt")
        print("CNN model loaded")

    def model_test(self, idx):
        with torch.no_grad():
            image, label = self.mnist.data_test[idx]
            image_torch = image.data.view(1, 1, 28, 28).float().to(self.device)
            predict = self.model(image_torch)
            print(f"Label: {label} Predict: {torch.argmax(predict)}")
            return image, label, predict

    def model_evaluate(self):
        test_loss = 0
        correct_cnt = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            self.model.eval()
            for image, label in self.mnist.test_loader:
                predict_logit = self.model(image)
                test_loss += criterion(predict_logit, label).item()
                predict = torch.argmax(predict_logit)
                correct_cnt += predict.eq(label.view_as(predict)).sum().item()
            print(f"Loss: {test_loss}, Correct: {(correct_cnt/len(self.mnist.test_loader))*100:.2f}% = {correct_cnt}/{len(self.mnist.test_loader)}")
            return test_loss, correct_cnt

# trainer = CNNTrain()
# trainer.model_train(False)
# trainer.model_test(99)
# test_loss, correct_cnt = trainer.model_evaluate()
