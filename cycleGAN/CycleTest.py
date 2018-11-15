from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils

from itertools import chain

from utils.dataset import DATASET
from utils.ImagePool import ImagePool
from model.Discriminator import Discriminator
from model.Generator import Generator

##########   DATASET   ###########
datasetA = DATASET(os.path.join("facades/train/",'A'),256,256,1)
datasetB = DATASET(os.path.join("facades/train/",'B'),256,256,1)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
        os.path.join("facades/",'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
]))
loader_A = torch.utils.data.DataLoader(dataset=datasetA,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=2)

loader_B = torch.utils.data.DataLoader(dataset=datasetB,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=2)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
num_workers=2)

ABPool = ImagePool(50)
BAPool = ImagePool(50)
###########   MODEL   ###########
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ndf = 64
ngf = 64
input_nc = 3
output_nc = 3
fineSize = 128
batchSize = 1
cuda = 1

D_A = Discriminator(input_nc,ndf)
D_B = Discriminator(output_nc,ndf)
G_AB = Generator(input_nc, output_nc, ngf)
G_BA = Generator(output_nc, input_nc, ngf)

G_AB.apply(weights_init)
G_BA.apply(weights_init)

D_A.apply(weights_init)
D_B.apply(weights_init)

if(cuda):
    D_A.cuda()
    D_B.cuda()
    G_AB.cuda()
    G_BA.cuda()

###########   LOSS & OPTIMIZER   ##########
criterionMSE = nn.L1Loss()
criterion = nn.MSELoss()
# chain is used to update two generators simultaneously
optimizerD_A = torch.optim.Adam(D_A.parameters(),lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
optimizerD_B = torch.optim.Adam(D_B.parameters(),lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
optimizerG = torch.optim.Adam(chain(G_AB.parameters(),G_BA.parameters()),lr=0.0002, betas=(0.5, 0.999))


real_A = torch.FloatTensor(batchSize, input_nc, fineSize, fineSize)
AB = torch.FloatTensor(batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(batchSize, output_nc, fineSize, fineSize)
BA = torch.FloatTensor(batchSize, output_nc, fineSize, fineSize)
label = torch.FloatTensor(batchSize)

if(cuda):
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    label = label.cuda()
    AB = AB.cuda()
    BA = BA.cuda()
    criterion.cuda()
    criterionMSE.cuda()

real_label = 1
fake_label = 0

###########   Testing    ###########
def test(niter):
    loader_train = iter(train_dataset)
    imgB,labelB = loader_train.next()
    print("imgB")
    print(imgB.size())
    print(imgB)
    loaderA = iter(loader_A)
    imgA = loaderA.next()
    print("imgA")
    print(imgA.size())
    real_A.data.resize_(imgA.size()).copy_(imgA)
#    real_A.data.resize_(imgA.size()).copy_(imgA).unsqueeze_(0)
    print("real_A")
    print(real_A.size())
    print(real_A)
    #real_B.data.resize_(imgB.size()).copy_(imgB)
    AB = G_AB(real_A)
    print("AB")
    print(AB.size())
    print(AB)
#   BA = G_BA(real_B)
    outA = D_A(real_A)
    print("outA")
    print(outA.size())
    print(outA)

    vutils.save_image(AB.data,'AB_niter_0_1.png',normalize=True)
#   vutils.save_image(BA.data,'BA_niter_%03d_1.png' % (niter),normalize=True)

test(3)
print("test end")
#print(D_A)
#print(D_B)
#print(G_AB)
#print(G_BA)