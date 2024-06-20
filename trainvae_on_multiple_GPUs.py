#  based on https://github.com/pytorch/examples/blob/main/vae/main.py
import os
import torch
import torch.utils.data
from os import mkdir
from torch import optim
from torch.nn import functional as F
from torch.nn import DataParallel
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



NUM_EPOCHS = 1 #800  # number of training epochs
START_EPOCH = 0 # 0 means that this epoch is the first epoch in the training 
PRINT_EVERY = 1 #10
PREVIOUS_DIR = 'vaemodels-e1_70k_images'


BATCH_SIZE =  480#240#for single 80gb a100 gpu  #40  # for data loaders # you can increase the batch size if you specify the number of workers (30) to be close to the number of cores (56)
#also you need to check the gpu usage in 'on demand' to decide if you can pump up the batch size
# if the gpu usage is low then you can bump up the batch size 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print('EPOCHS', NUM_EPOCHS, 'BATCH_SIZE', BATCH_SIZE, 'device', device)

# for model and results
directory = f'vaemodels-{rndstr()}'
mkdir(directory)
print(directory)


# load dataset
#p = Path('../VAE/ffhq52000/')
p = Path('/scratch/malmaim/unpacked_ffhq/all/')
#ffhq_data = FFHQ_Data(data_dir= p, transform= transforms.ToTensor())
ffhq_data = FFHQ_Data(data_dir= p, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]))

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

#model_original is the original model 
#model is the parallel model 
#model = VAE().to(device) #use if you have only one gpu and you need to change every model_original to modle 
model_original = VAE().to(device)
if torch.cuda.device_count() > 1: #if there is more than one GPU we will use DataParellel to devide the data on the GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    gpu_list = list(range(torch.cuda.device_count()))
    model = DataParallel(model_original, device_ids = gpu_list) #this is a parallel model #model_original


optimizer = optim.Adam(model_original.parameters(), lr=0.0001)#1e-3 = 1 x 10^-3 =  0.001 #model_original


def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
    'epoch' : epoch,
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
}, filename)
    print(f"Epoch {epoch} | Training checkpoint saved at {filename}")

#----------------------------------------------------------------------------

def resume(model, optimizer, filename):#restart training from a specific epoch 
#resume the state of the model   
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print(f'Resume training from Epoch {epoch}')

#----------------------------------------------------------------------------

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var):
    b_size = x.shape[0] #this batch size if from the data loader that has all the data 
    #change dimensions if you want to use xaing's MSE loss function 
    recon_x = recon_x.view(b_size, 3, -1)
    x_hat =  torch.flatten(x, start_dim=2)
    MSE = torch.mean(torch.sum((x_hat - recon_x)**2, dim=1), dim=-1).mean()
    
    #use the x, and recon_x that are in the arguments without any change 
    #MSE =F.mse_loss(recon_x, x.view(-1, image_dim))
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())#correct
    kld_weight = 0.00025
    loss = MSE + kld_weight * KLD  
    return loss

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

def test(epoch):
    model.eval()
    valid_epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            valid_epoch_loss += loss_function(recon_batch, data, mu, log_var).item()
            if epoch % 1 == 0:# save the reconstruction grid every 100 epochs 
                if i == 0:
                    b_size = data.shape[0]
                    #save as a grid that has 8 ground truth images, and 8 reconstructed images
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(b_size, 3, IMAGE_SIZE, IMAGE_SIZE)[:n]])
                    save_image(comparison.cpu(),
                            f'{directory}/reconstruction_grid_{str(epoch)}.png', nrow=n)
                
                #save single images 
                #this block of code works with images size 512 
                #But it doesn't work with image size 150 
                # recon_batch = recon_batch.view(b_size, 3, IMAGE_SIZE, IMAGE_SIZE)
                # if epoch % PRINT_EVERY == 0:
                #     save_image(data[0].detach().cpu(), f"./{directory}/GroundTruth_image_512_e{epoch}.png")#you need to go to the cpu to save the image 
                #     save_image(recon_batch[0].detach().cpu(), f"./{directory}/reconst_image_512_e{epoch}.png")


    valid_epoch_loss /= len(test_loader.dataset)
    writer.add_scalars("Epoch Loss", {'Validation Loss': valid_epoch_loss}, epoch)
    if epoch % PRINT_EVERY == 0:#will print after every time the epoch is a multiple of 100 (if the remainder is 0 then it is a multiple).
        print('====> Test set loss: {:.4f}'.format(valid_epoch_loss))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    
    if START_EPOCH > 0:
        filename = f'{PREVIOUS_DIR}/vae_model_{START_EPOCH}.pth'
        resume(model, optimizer, filename) 
        writer = SummaryWriter(f'./{PREVIOUS_DIR}/vae_512')
    else: 
        writer = SummaryWriter(f'./{directory}/vae_512')

    for epoch in range(START_EPOCH+1, NUM_EPOCHS + 1):
        print(f"Start training for epoch # {epoch} ......... ")
        start_time = time.monotonic()
        train(epoch)
        
        test(epoch)
        end_time = time.monotonic()
        print("training and validating this one epoch took:", timedelta(seconds= end_time - start_time))
        if epoch % PRINT_EVERY == 0:
            #torch.save(model, f'{directory}/vae_model_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, f'{directory}/vae_model_{epoch}.pth')
            #sampling from the latent space 
            with torch.no_grad():
                sample = torch.randn(64, LATENT_DIM).to(device)
                sample = model_original.decode(sample).cpu()#model_original
                save_image(sample.view(64, 3, IMAGE_SIZE, IMAGE_SIZE),
                        f'{directory}/sample_{str(epoch)}.png')
        
    writer.close()
    print("Done training ....!")
