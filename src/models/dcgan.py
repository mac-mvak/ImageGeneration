import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_channels, gen_maps, photo_channels, **kwargs):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( latent_channels, gen_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_maps * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(gen_maps * 8, gen_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_maps * 4),
            nn.ReLU(),
            nn.ConvTranspose2d( gen_maps * 4, gen_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_maps * 2),
            nn.ReLU(),
            nn.ConvTranspose2d( gen_maps * 2, gen_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_maps),
            nn.ReLU(),
            nn.ConvTranspose2d( gen_maps, photo_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, photo_channels, dis_maps, **kwargs):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(photo_channels, dis_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dis_maps, dis_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_maps * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dis_maps * 2, dis_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_maps * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dis_maps * 4, dis_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_maps * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dis_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)


