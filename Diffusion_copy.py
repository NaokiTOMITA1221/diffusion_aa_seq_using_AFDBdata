import math
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

aa_seq_len = 200
dim_corr = 1
aa_kinds = 21
adjusting_dim = 3



def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v



def pos_encoding(timesteps, output_dim, device='cpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v
    
    
    
    


def seq_embedder(aa_seq):
  aa_one_hots_list = []
  aa_pos_encode_list = []

  aa_list = {'A':0, 'a':0,'C':1, 'c':1, 'D':2, 'd':2, 'E':3, 'e':3, 'F':4,'f':4, 'G':5, 'g':5, 'H':6, 'h':6, 'I':7, 'i':7, 'K':8, 'k':8, 'L':9, 'l':9, 'M':10, 'm':10, 'N':11, 'n':11, 'P':12, 'p':12, 'Q':13, 'q':13, 'R':14, 'r':14, 'S':15, 's':15, 'T':16, 't':16, 'V':17, 'v':17, 'W':18, 'w':18, 'Y':19, 'y':19, 'b':20}


 # aa_list = {'A':0, 'a':0, 'D':1, 'd':1, 'E':2, 'e':2, 'F':3,'f':3, 'G':4, 'g':4, 'H':5, 'h':5, 'I':6, 'i':6, 'K':7, 'k':7, 'L':8, 'l':8, 'M':9, 'm':9, 'N':10, 'n':10, 'P':11, 'p':11, 'Q':12, 'q':12, 'R':13, 'r':13, 'S':14, 's':14, 'T':15, 't':15, 'V':16, 'v':16, 'W':17, 'w':17, 'Y':18, 'y':18, 'b':19} #when C omittion
  for aa_ind, aa in enumerate(aa_seq):
      aa_emb = np.ones(aa_kinds+adjusting_dim)*1e-5 #pseudo count
      aa_emb[aa_list[aa]] = 1
        
      aa_pos_encode = _pos_encoding(aa_ind, aa_kinds+adjusting_dim)
        

      aa_one_hots_list.append(aa_emb)
      aa_pos_encode_list.append(aa_pos_encode)
      
  seq_onehot_and_pos = np.array([aa_one_hots_list, aa_pos_encode_list])
  seq_onehot_and_pos = torch.from_numpy(seq_onehot_and_pos.astype(np.float32)).clone()
      #print(seq_onehot.shape)
  return seq_onehot_and_pos

    
    
    
    
    
    
    

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape#N,C:batch size, channel num
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y
        
        
        
        
        
        
        
        
        
        

class UNet(nn.Module):
    def __init__(self, in_ch=2, time_embed_dim=1000):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x, timesteps):
        #print('original_x:',x.size())
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)
        
        x1 = self.down1(x, v)
        #print('x_after_down1:',x1.shape)
        x = self.maxpool(x1)
        #print('maxpooled_x1:',x.shape)
        x2 = self.down2(x, v)
        #print('x_after_down2:',x2.shape)
        x = self.maxpool(x2)
        #print('maxpooled_x2:',x.shape)

        x = self.bot1(x, v)

        x = self.upsample(x)
        #print(x.size())
        #x = self.dim_adjuster(x)
        #print(x.size())
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        #x = self.dim_adjuster(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x


class UNet_UNet_UNet(nn.Module):
    def __init__(self, in_ch=2, time_embed_dim=1000): #in_chにはアミノ酸配列情報と残基番号位置エンコーディングの情報が２チャンネル分入っている
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64*dim_corr, time_embed_dim)
        self.down2 = ConvBlock(64*dim_corr, 128*dim_corr, time_embed_dim)
        self.bot1 = ConvBlock(128*dim_corr, 256*dim_corr, time_embed_dim)
        self.up2 = ConvBlock((128 + 256)*dim_corr, 128*dim_corr, time_embed_dim)
        self.up1 = ConvBlock((128 + 64)*dim_corr, 64*dim_corr, time_embed_dim)
        self.out = nn.Conv2d(64*dim_corr, in_ch, 1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')



    def forward(self, x, timesteps):
        #print('original_x:',x.size())
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)
        
        x1 = self.down1(x, v)
        #print('x_after_down1:',x1.shape)
        x = self.maxpool(x1)
        #print('maxpooled_x1:',x.shape)
        x2 = self.down2(x, v)
        #print('x_after_down2:',x2.shape)
        x = self.maxpool(x2)
        #print('maxpooled_x2:',x.shape)

        x = self.bot1(x, v)

        x = self.upsample(x)
        #print(x.size())
        #x = self.dim_adjuster(x)
        #print(x.size())
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        #x = self.dim_adjuster(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        
        x1 = self.down1(x, v)
        #print('x_after_down1:',x1.shape)
        x = self.maxpool(x1)
        #print('maxpooled_x1:',x.shape)
        x2 = self.down2(x, v)
        #print('x_after_down2:',x2.shape)
        x = self.maxpool(x2)
        #print('maxpooled_x2:',x.shape)

        x = self.bot1(x, v)

        x = self.upsample(x)
        #print(x.size())
        #x = self.dim_adjuster(x)
        #print(x.size())
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        #x = self.dim_adjuster(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        
        
        x1 = self.down1(x, v)
        #print('x_after_down1:',x1.shape)
        x = self.maxpool(x1)
        #print('maxpooled_x1:',x.shape)
        x2 = self.down2(x, v)
        #print('x_after_down2:',x2.shape)
        x = self.maxpool(x2)
        #print('maxpooled_x2:',x.shape)

        x = self.bot1(x, v)

        x = self.upsample(x)
        #print(x.size())
        #x = self.dim_adjuster(x)
        #print(x.size())
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        #x = self.dim_adjuster(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x












class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)#ノイズスケジューリング（各時刻のbetaはスカラー）
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)#alpha_tの積算した値を全時刻分リストかしたもの
        
    def add_noise(self, x_0, t):#tはbatch size分だけ時刻がランダムに格納されている
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)

        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)

        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise #変数変換トリック
        return x_t, noise

    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std













    def sample(self, model, x_shape=(500, 2, aa_seq_len, aa_kinds+adjusting_dim)):
        ind_to_aa = {0:'A',1:'C',2:'D',3:'E',4:'F',5:'G',6:'H',7:'I',8:'K',9:'L',10:'M',11:'N',12:'P',13:'Q',14:'R',15:'S',16:'T',17:'V',18:'W',19:'Y',20:'b'}
       # ind_to_aa = {0:'A',1:'D',2:'E',3:'F',4:'G',5:'H',6:'I',7:'K',8:'L',9:'M',10:'N',11:'P',12:'Q',13:'R',14:'S',15:'T',16:'V',17:'W',18:'Y',19:'b'}#when C omittion
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)

        seq_list = []
        for seq_vec in x:
            seq = ''
            for aa_vec in seq_vec[0]:
                seq += ind_to_aa[int(torch.argmax(aa_vec[:aa_kinds]))]
            seq_list.append(seq)
        return seq_list


