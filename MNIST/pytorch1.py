import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import torch.optim as optim
import matplotlib.pyplot as plt
BATCH_SIZE = 100
NUM_EPOCHS = 100

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=False)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

class Znet(nn.Module):
    def __init__(self):
        super(Znet, self).__init__() 
        self.conv1=nn.Conv2d(kernel_size=(3,3),in_channels=1,out_channels=32)
        self.pool1=nn.MaxPool2d(kernel_size=(2,2))
        self.conv2=nn.Conv2d(kernel_size=(3,3),in_channels=32,out_channels=64)
        self.pool2=nn.MaxPool2d(kernel_size=(2,2))
        self.drop1=nn.Dropout2d(0.3)
        # self.flat=nn.Linear(64*3*3,64*3*3)
        self.dense1=nn.Linear(1600,100)
        self.drop2=nn.Dropout2d(0.5)
        self.dense2=nn.Linear(100,10)
    def forward(self,x):
        x=nn.functional.relu(self.conv1(x))
        x=self.pool1(x)
        x=nn.functional.relu(self.conv2(x))
        x=self.pool2(x)
        x=self.drop1(x)
        x=self.flat2d(x)
        x=nn.functional.relu(self.dense1(x))
        x=self.drop2(x)
        # x=nn.functional.softmax(self.dense2(x))
        x=self.dense2(x)
        return(x)
    def flat2d(self,x):
        slist=x.size()[1:]  
        n=1
        for i in slist:
            n*=i
        x=x.view(-1,n)
        return(x)

net=Znet()
print(net)
params = list(net.parameters())
print(len(params))
for i in range(len(params)):
    print(params[i].size())
optimizer=optim.Adadelta(params=params,rho=0.95,eps=1e-8)
criterion=nn.CrossEntropyLoss()
rec_acc=list()
rec_loss=list()
rec_val_acc=list()
rec_val_loss=list()
for epoch in range(1,NUM_EPOCHS):
    train_loss=0.0
    train_acc=0
    cnt=0
    for images,labels in tqdm(train_loader):#把六万组数据分为128组并行，每组为128维
        cnt=cnt+1
        optimizer.zero_grad()
        outputs=net(images)
        # print(outputs.data.max(1,keepdim=True)[1])
        outlabels=outputs.data.max(1,keepdim=True)[1]
        train_acc+=outlabels.eq(labels.data.view_as(outlabels)).cpu().sum().item()
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    test_loss=0.0
    test_acc=0
    for images,labels in tqdm(test_loader):
        outputs=net(images)
        outlabels=outputs.data.max(1,keepdim=True)[1]
        test_loss+=criterion(outputs,labels).item()
        test_acc+=outlabels.eq(labels.data.view_as(outlabels)).cpu().sum().item()
    # print('No.%s,loss=%.6f,acc=%.6f'%(epoch,train_loss/len(train_loader),train_acc/len(train_dataset)))
    # print('No.%s,val_loss=%.6f,val_acc=%.6f'%(epoch,test_loss/len(test_loader),test_acc/len(test_dataset)))
    rec_acc.append(train_acc/len(train_dataset))
    rec_loss.append(train_loss/len(train_loader))
    rec_val_acc.append(test_acc/len(test_dataset))
    rec_val_loss.append(test_loss/len(test_loader))
    print('No.%s,loss=%.6f,acc=%.6f,val_loss=%.6f,val_acc=%.6f'%(epoch,train_loss/len(train_loader),train_acc/len(train_dataset),test_loss/len(test_loader),test_acc/len(test_dataset)))
print('Finished Training')
plt.plot(rec_acc)
plt.plot(rec_val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(rec_loss)
plt.plot(rec_val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
