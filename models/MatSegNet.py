import torch.nn as nn
from torchvision import models
import torch
import os
import yaml

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, padding=0), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, padding=0), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, padding=0), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, drop_p=0.3):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Dropout(drop_p),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
    def forward(self, x): return self.decode(x)

class MatSegNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.enc1 = nn.Sequential(*list(base.children())[:3])
        self.enc2 = nn.Sequential(*list(base.children())[3:5])
        self.enc3, self.enc4, self.enc5 = list(base.children())[5], list(base.children())[6], list(base.children())[7]
        self.up5, self.up4, self.up3, self.up2 = [nn.ConvTranspose2d(c, c//2, 2, 2) for c in [512, 256, 128, 64]]
        self.att4, self.att3, self.att2, self.att1 = [AttentionBlock(g, l, i) for g, l, i in [(256, 256, 128), (128, 128, 64), (64, 64, 32), (32, 64, 32)]]
        self.dec4, self.dec3, self.dec2, self.dec1 = [DecoderBlock(i, o, p) for i, o, p in [(512, 256, 0.4), (256, 128, 0.3), (128, 64, 0.2), (96, 32, 0.1)]]
        self.final_up = nn.ConvTranspose2d(32, 16, 2, 2)
        self.final_conv = DecoderBlock(16, 16, 0.0)
        self.final_mask, self.final_edge = nn.Conv2d(16, n_classes, 1), nn.Conv2d(16, 1, 1)

    def forward(self, x):
      
        e1 = self.enc1(x)
         
        e2 = self.enc2(e1)
    
        e3 = self.enc3(e2)
       
        e4 = self.enc4(e3)
     
        e5 = self.enc5(e4)
       
        
        d5 = self.up5(e5)
     
      
        
        d4 = self.dec4(torch.cat((self.att4(d5, e4), d5), 1))
        d4_up = self.up4(d4)
       
        d3 = self.dec3(torch.cat((self.att3(d4_up, e3), d4_up), 1))
        d3_up = self.up3(d3)
       
        
        d2 = self.dec2(torch.cat((self.att2(d3_up, e2), d3_up), 1))
        d2_up = self.up2(d2)
         
        d1 = self.dec1(torch.cat((self.att1(d2_up, e1), d2_up), 1))

           
        d_out = self.final_conv(self.final_up(d1))
        
  
        return self.final_mask(d_out), self.final_edge(d_out)




def get_model(n_classes=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    if n_classes is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        configs_dir = os.path.abspath(os.path.join(project_root, 'configs'))
        CONFIG_FILE_PATH = os.path.join(configs_dir, "MatSegNet.yaml")
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = yaml.safe_load(f)
        n_classes = config['num_classes']
       
    model = MatSegNet(n_classes=n_classes)
    model = model.to(device)
    return model
    
    
if __name__ == "__main__":
    model_default = get_model()
    model_custom = get_model(n_classes=5)
    print("Model loaded successfully!")