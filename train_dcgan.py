import json
from src.models.dcgan import Generator, Discriminator
import torchvision.transforms as transforms
from src.utils import inf_loop, save_model
from tqdm import tqdm
import torch.nn as nn
from src.logger import Writer
from src.metrics import FIDCalc, create_gen_folder
from piq import FID
import torchvision.datasets as dset
import torch

dataroot='data'
image_size=64
torch.use_deterministic_algorithms(False) 
with open('configs/config_dcgan.json') as f:
    cfg = json.load(f)



dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=True, num_workers=5, drop_last=True)

dataloader = inf_loop(dataloader)


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

gen = Generator(**cfg["model"]["args"])
gen = gen.to(device)
dis = Discriminator(**cfg["model"]["args"])
dis = dis.to(device)

opt_gen = torch.optim.Adam(params=gen.parameters(), **cfg['optim_params'])
opt_dis = torch.optim.Adam(params=dis.parameters(), **cfg['optim_params'])

scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(opt_gen, gamma=0.999)
scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(opt_dis, gamma=0.999)
a = scheduler_gen.get_last_lr()
loss = nn.BCELoss()
writer = Writer('img_gen', 'big_run', cfg)

img_list = []
G_losses = []
D_losses = []
epochs = cfg['trainer']['epochs']
len_epoch = cfg['trainer']['len_epoch']
lat_size = cfg['model']['args']['latent_channels']
const_z = torch.randn(2000, lat_size, 1, 1, device=device)
fid_metric = FID().to(device)


for i in range(epochs):
    for j, batch in tqdm(enumerate(dataloader), total=len_epoch):
        # DisLoss
        opt_dis.zero_grad()
        real_ph = batch[0].to(device)
        n = real_ph.shape[0]
        true_labels = torch.ones(n, device=device)
        true_dis_out = dis(real_ph).squeeze()
        dis_loss_real = loss(true_dis_out, true_labels)
        dis_loss_real.backward()
        D_x = dis_loss_real.mean().item()

        z = torch.randn(n, lat_size, 1, 1, device=device)
        fake_ph = gen(z)
        fake_labels = torch.zeros(n, device=device)
        fake_dis_out = dis(fake_ph.detach()).squeeze()
        dis_loss_fake = loss(fake_dis_out, fake_labels)
        dis_loss_fake.backward()
        D_G_z1 = fake_dis_out.mean().item()
        errD = dis_loss_real.detach() + dis_loss_fake.detach()

        opt_dis.step()

        #GenLoss
        opt_gen.zero_grad()
        gen_output = dis(fake_ph).squeeze()
        loss_G = loss(gen_output, true_labels)
        loss_G.backward()

        D_G_z2 = gen_output.mean().item()
        opt_gen.step()

        writer.log({'errD': errD.item(), 
                    'errG': loss_G.item(),
                    'D_x': D_x,
                    'D_G_z1': D_G_z1,
                    'D_G_z2': D_G_z2,
                    'gen_lr': scheduler_gen.get_last_lr()[0],
                    'dis_lr': scheduler_dis.get_last_lr()[0]})
        if j + 1 == len_epoch:
            break
    scheduler_dis.step()
    scheduler_gen.step()
    if i % 5 == 0:
        save_model(i, gen, dis, opt_gen, opt_dis, cfg)
    ssim = create_gen_folder(gen, lat_size, device, const_z)
    fid = FIDCalc(fid_metric)
    writer.log({'SSIM': errD.item(), 
                    'FID': fid})

save_model(i, gen, dis, opt_gen, opt_dis, cfg)  


