import torch
import torch.nn as nn



class AngleLossFcn(nn.Module):
    def __init__(self,mean=True):
        super(AngleLossFcn, self).__init__()
        self.mean=mean
    def forward(self, fake, real):

        exp = 0.000000000001
        ab = fake[:,0,:,:] * real[:,0,:,:] + fake[:,1,:,:] * real[:,1,:,:] + fake[:,2,:,:] * real[:,2,:,:]
        a = fake[:,0,:,:] * fake[:,0,:,:] + fake[:,1,:,:] * fake[:,1,:,:] + fake[:,2,:,:] * fake[:,2,:,:]
        a = torch.sqrt(a)
        b = real[:,0,:,:] * real[:,0,:,:] + real[:,1,:,:] * real[:,1,:,:] + real[:,2,:,:] * real[:,2,:,:]
        b = torch.sqrt(b)
        mul_ab = torch.mul(a,b)
        mul_ab = torch.add(mul_ab,exp)
        inputdata = torch.div(ab,mul_ab)
        angle = torch.abs(inputdata-1)
        if self.mean==True:
            angle = torch.mean(angle)
        return angle

class LossFcn(nn.Module):
    def __init__(self):
        super(LossFcn, self).__init__()
        self.color_loss = AngleLossFcn()
        self.pix_loss = nn.L1Loss()

    def forward(self, fake, real):
        loss = self.pix_loss(fake, real) + 0.1*self.color_loss(fake, real)
        return loss

