
class UNet_SelfAtt_UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=200):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        #UNet 1
        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)
        
        self.self_attention = nn.MultiheadAttention(21, 7)#残基番号に関するシグナルを伝達させるためにUNetを2つ設け、その間にattentionをブッこむ
        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dim_adjuster = nn.Linear(20,21) #up samplingの後次元が揃わないので、それを調節する応急処置
        
    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = self.dim_adjuster(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        
        dim1, dim2, dim3, dim4 = x.shape

        Q = torch.reshape(x, (dim1, dim3, dim4))
        K = torch.reshape(x, (dim1, dim3, dim4))
        V = torch.reshape(x, (dim1, dim3, dim4))
        
        x, attention_weights = self.self_attention(Q,K,V)
        x = torch.reshape(x, (dim1, dim2, dim3, dim4))

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = self.dim_adjuster(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x

