import Diffusion as dif
import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import random
import datetime



seq_from_fasta = []
for line_num, line in enumerate(open('fasta/train_data_from_AFDB_from100res_to200res.fasta')):
    if line_num%2 == 1:
        seq_from_fasta.append(line.strip())
        
        
 

res_num = 200
num_of_all_train_data = len(seq_from_fasta)
epochs = 1000
batch_size = 100 #int(num_of_all_train_data/epochs) 
print('batch size:',batch_size)
num_timesteps = 1000
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'



diffuser = dif.Diffuser(num_timesteps, device=device)
#model = dif.UNet()
model = dif.UNet_UNet_UNet()
#model = dif.UNet_SelfAtt_UNet()
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

losses = []

data_loader = []
for _ in range(100):
    random.shuffle(seq_from_fasta)
    batch = []
    for seq in seq_from_fasta[0:batch_size]:
        batch.append(dif.seq_embedder(seq))
    batch = torch.stack(batch)
    data_loader.append(batch)
    print('batch shape:', batch.shape)
data_loader = torch.stack(data_loader)


dt_now = str(datetime.datetime.now())
dt_now_list = dt_now.split()
dt_now = dt_now_list[0]+''+dt_now_list[1]

w = open(f'train_log/Loss_log_{dt_now}.txt','w')


for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    # generate samples every epoch ===================
    # images = diffuser.sample(model)
    # show_images(images)
    # ================================================
 
    for seq_vecs in tqdm(data_loader):
        try:
            optimizer.zero_grad()
            x = seq_vecs.to(device)
            #print(x.shape)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

            x_noisy, noise = diffuser.add_noise(x, t)
            #print(x_noisy.shape)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1
        except ValueError:
            None

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')
    w.write(f'Epoch {epoch} | Loss: {loss_avg}')



w.close()

#save_model
torch.save(model, f'pth/aa_seq_diffusion_{dt_now}.pth')



# generate samples
#images = diffuser.sample(model)

