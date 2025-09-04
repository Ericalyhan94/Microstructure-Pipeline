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
        print("base enc1: ",*list(base.children())[:3])
        print("base enc2: ",*list(base.children())[3:5])
        print("base enc3: ",list(base.children())[5])
        print("base enc4: ",list(base.children())[6])
        print("base enc5: ",list(base.children())[7])
        
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
        print("---shape of x: ",(x.size()))
        e1 = self.enc1(x)
        print("---shape of e1: ",(e1.size()))
        e2 = self.enc2(e1)
        print("---shape of e2: ",(e2.size()))
        e3 = self.enc3(e2)
        print("---shape of e3: ",(e3.size()))
        e4 = self.enc4(e3)
        print("---shape of e4: ",(e4.size()))
        e5 = self.enc5(e4)
        print("---shape of e5: ",(e5.size()))
        
        d5 = self.up5(e5)
     
        print("---shape of d5: ",(d5.size()))
        
        d4 = self.dec4(torch.cat((self.att4(d5, e4), d5), 1))
        d4_up = self.up4(d4)
        print("---shape of d4: ",(d4.size()))
        print("---shape of d4_up: ",(d4_up.size()))
        
        d3 = self.dec3(torch.cat((self.att3(d4_up, e3), d4_up), 1))
        d3_up = self.up3(d3)
        print("---shape of d3: ",(d3.size()))
        print("---shape of d3_up: ",(d3_up.size()))
        
        
        d2 = self.dec2(torch.cat((self.att2(d3_up, e2), d3_up), 1))
        d2_up = self.up2(d2)
        print("---shape of d2: ",(d2.size()))
        print("---shape of d2_up: ",(d2_up.size()))
        
        d1 = self.dec1(torch.cat((self.att1(d2_up, e1), d2_up), 1))

        print("---shape of d1: ",(d1.size()))

        
        d_out = self.final_conv(self.final_up(d1))
        
        print("---shape of d_out: ",(d1.size()))
        
        print("---shape of final_mask: ",(self.final_mask(d_out).size()))
        print("---shape of final_edge: ",(self.final_edge(d_out).size()))
        
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