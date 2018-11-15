import torch.nn as nn
import torch
from Generator import InstanceNormalization

class Discriminator(nn.Module):
    def __init__(self,input_nc,ndf):
        super(Discriminator,self).__init__()
        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc,ndf,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 64 x 64
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 InstanceNormalization(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 32 x 32
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=1,padding=1),
                                 InstanceNormalization(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 31 x 31
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=1),
                                 nn.Sigmoid())
        # 30 x 30

    def forward(self,x):
        print("Discriminator forward start")
        print(x.size())
        out = self.layer1(x)
        print(out.size())
        out = self.layer2(out)
        print(out.size())
        out = self.layer3(out)
        print(out.size())
        out = self.layer4(out)
        print(out.size())
        print("Discriminator forward end")
        return out
