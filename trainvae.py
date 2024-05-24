#  based on https://github.com/pytorch/examples/blob/main/vae/main.py
import os
import torch
import torch.utils.data
from os import mkdir
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter 


import time
from datetime import timedelta

# project modules
from utils import print, rndstr
from vae import VAE, IMAGE_SIZE, LATENT_DIM, CELEB_PATH, image_dim, celeb_transform

from data_512 import FFHQ_Data
from pathlib import Path



#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

EPOCHS = 7000  # number of training epochs
BATCH_SIZE = 40 #16  # for data loaders # you can increase the batch size if you specify the number of workers (30) to be close to the number of cores (56)
PRINT_EVERY = 1000
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print('EPOCHS', EPOCHS, 'BATCH_SIZE', BATCH_SIZE, 'device', device)

# for model and results
directory = f'vaemodels-{rndstr()}'
mkdir(directory)
print(directory)
writer = SummaryWriter(f'./{directory}/vae_512')

# load dataset
p = Path('../VAE/ffhq52000/')
ffhq_data = FFHQ_Data(data_dir= p, transform= transforms.ToTensor())
data_length = ffhq_data.__len__() 
print(f'no. of images in the entire dataset is:', data_length)

# Split the dataset into training and validation sets
train_size = int(0.8 *len(ffhq_data))#0.4, 0.5, 0.8
test_size = len(ffhq_data) - train_size
#generate a random number 
gen = torch.Generator()
gen.manual_seed(0) # this to make sure that the data is being splitted randomly in the same way each time we run the code 

train_dataset, test_dataset = random_split(ffhq_data, [train_size, test_size], generator=gen)#if we omit "generator=gen" then each time we run the code the random split will be different, and this will produce different training results 

print('no of training samples:', len(train_dataset))
print('no of testing samples:', len(test_dataset))

#train_dataset = CelebA(CELEB_PATH, transform=celeb_transform, download=True, split='train')
#test_dataset = CelebA(CELEB_PATH, transform=celeb_transform, download=True, split='valid') # or 'test'

# create train and test dataloaders
#train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=30, shuffle=True) #num_workers is 30 beacuse the number of cpu cores is 56 this is to make the training faster 
test_loader = DataLoader(dataset=test_dataset, batch_size= BATCH_SIZE, num_workers=30, shuffle =False)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)#1e-3 = 1 x 10^-3 =  0.001


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var):
    #change dimensions if you want to use xaing's MSE loss function 
    recon_x = recon_x.view(40, 3, -1)
    x_hat =  torch.flatten(x, start_dim=2)
    MSE = torch.mean(torch.sum((x_hat - recon_x)**2, dim=1), dim=-1).mean()
    
    #use the x, and recon_x that are in the arguments without any change 
    #MSE =F.mse_loss(recon_x, x.view(-1, image_dim))
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kld_weight = 0.00025
    loss = MSE + kld_weight * KLD  
    return loss


def train(epoch):
    model.train()
    train_epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):
        torch.cuda.empty_cache()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_epoch_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
            
    train_epoch_loss = train_epoch_loss / len(train_loader.dataset)
    writer.add_scalars("Epoch Loss", {'Training Loss': train_epoch_loss}, epoch)
    if epoch % PRINT_EVERY == 0:#will print after every time the epoch is a multiple of 100 (if the remainder is 0 then it is a multiple).
        print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch,  train_epoch_loss))

def test(epoch):
    model.eval()
    valid_epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            valid_epoch_loss += loss_function(recon_batch, data, mu, log_var).item()
            if i == 0:
                #save as a grid that has 8 ground truth images, and 8 reconstructed images
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)[:n]])
                save_image(comparison.cpu(),
                           f'{directory}/reconstruction_{str(epoch)}.png', nrow=n)
                
                #save single images 
                recon_batch = recon_batch.view(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
                save_image(data[0].detach().cpu(), f"./{directory}/GT_image_512_e{epoch}.png")#you need to go to the cpu to save the image 
                save_image(recon_batch[0].detach().cpu(), f"./{directory}/reconst_image_512_e{epoch}.png")



    valid_epoch_loss /= len(test_loader.dataset)
    writer.add_scalars("Epoch Loss", {'Validation Loss': valid_epoch_loss}, epoch)
    if epoch % PRINT_EVERY == 0:#will print after every time the epoch is a multiple of 100 (if the remainder is 0 then it is a multiple).
        print('====> Test set loss: {:.4f}'.format(valid_epoch_loss))

if __name__ == "__main__":
    print(f'epochs: {EPOCHS}')

    for epoch in range(1, EPOCHS + 1):
        start_time = time.monotonic()
        train(epoch)
        if epoch % PRINT_EVERY == 0:
            torch.save(model, f'{directory}/vae_model_{epoch}.pth')
        test(epoch)
        end_time = time.monotonic()
        print("training and validating this one epoch took:", timedelta(seconds= end_time - start_time))
        with torch.no_grad():
            sample = torch.randn(64, LATENT_DIM).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, IMAGE_SIZE, IMAGE_SIZE),
                       f'{directory}/sample_{str(epoch)}.png')
    
    writer.close()
    print("Done training ....!")
