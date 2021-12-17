import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time

########################################
# You can define whatever classes if needed
########################################
class resnet_block1(nn.Module):
    def __init__(self, channels, stride=1):
        super(resnet_block1, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.convolution_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.convolution_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        out = self.batch_norm1(x)
        out = self.relu1(out)
        out = self.convolution_1(out)
        out = self.batch_norm2(out)
        out = self.relu2(out)
        out = self.convolution_2(out)
        out += x
        return out


class resnet_block2(nn.Module):
    def __init__(self, channels, stride=1):
        super(resnet_block2, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.convolution_1 = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(2 * channels)
        self.relu2 = nn.ReLU()
        self.convolution_2 = nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.sc = nn.Sequential(nn.ReLU(), nn.BatchNorm2d(channels), nn.Conv2d(channels, 2 * channels, kernel_size=1, stride=2))

    def forward(self, x):
        out = self.batch_norm1(x)
        out = self.relu1(out)
        out = self.convolution_1(out)
        out = self.batch_norm2(out)
        out = self.relu2(out)
        out = self.convolution_2(out)
        out += self.sc(x)

        return out


class IdentityResNet(nn.Module):

    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
        ########################################
        # Implement the network
        # You can declare whatever variables
        ########################################
        self.convolution_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_stg1 = self.block_define(64, nblk_stage1, 1)
        self.res_stg2 = self.block_define(64, nblk_stage2, 2)
        self.res_stg3 = self.block_define(128, nblk_stage3, 3)
        self.res_stg4 = self.block_define(256, nblk_stage4, 4)
        self.avg_pooling = nn.AvgPool2d(4, stride=4)
        self.fully_connected_1 = nn.Linear(512, 10)

    ########################################
    # You can define whatever methods
    ########################################
    def block_define(self, p, n_block, n_stage):
        l = []
        cnt = 0

        if n_stage != 1:
            l.append(resnet_block2(p))
            while cnt < (n_block - 1):
                l.append(resnet_block1(2 * p))
                cnt += 1
        else:
            l.append(resnet_block1(p))
            while cnt < (n_block - 1):
                l.append(resnet_block1(p))
                cnt += 1

        return nn.Sequential(*l)

    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################
        out = self.convolution_1(x)
        out = self.res_stg1(out)
        out = self.res_stg2(out)
        out = self.res_stg3(out)
        out = self.res_stg4(out)
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected_1(out)
        return out


########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', dev)

########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 32

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
net.to(dev)

# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
# print(net)
# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)

        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()

        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)

        # set loss
        loss = criterion(outputs, labels)

        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()

        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end - t_start, ' sec')
            t_start = t_end

print('Finished Training')

# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    net.eval()
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' % (classes[i]), ': ',
          100 * class_correct[i] / class_total[i], '%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct) / sum(class_total)) * 100, '%')
