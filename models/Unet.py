import torch.nn as nn
from torchvision import models
import torch

import os
import sys
import yaml

class UpsamplingBlock(nn.Module):
    def __init__(self, up_in_channels, skip_in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_channels, up_in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(up_in_channels + skip_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, expansive_input, contractive_input):
        x = self.up(expansive_input)
        x = torch.cat([x, contractive_input], dim=1)
        x = self.conv(x)
        return x
        
        
class UnetMobileNetV2(nn.Module):
    def __init__(self, n_classes=1, dropout_rate=0.3):
        super().__init__()

        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        encoder_features = base_model.features
        print("encoder_features",encoder_features)
        # Create a parent module for the encoder
        self.encoder = nn.ModuleDict({
            's1': encoder_features[0:2],
            's2': encoder_features[2:4],
            's3': encoder_features[4:7],
            's4': encoder_features[7:14],
            'bottleneck': encoder_features[14:]
        })

        self.bottleneck_dropout = nn.Dropout2d(p=dropout_rate)

        # ---  Decoder ---
        self.d1 = UpsamplingBlock(1280, 96, 512)
        self.d2 = UpsamplingBlock(512, 32, 256)
        self.d3 = UpsamplingBlock(256, 24, 128)
        self.d4 = UpsamplingBlock(128, 16, 64)
        
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        
      
        s1_out = self.encoder.s1(x)
        
   
        
        s2_out = self.encoder.s2(s1_out)
        
       
        s3_out = self.encoder.s3(s2_out)
        
         
        s4_out = self.encoder.s4(s3_out)
        
          
        
        b_out = self.encoder.bottleneck(s4_out)
        b_out = self.bottleneck_dropout(b_out)
        
      
        
        d1_out = self.d1(b_out, s4_out)
 
        
        d2_out = self.d2(d1_out, s3_out)
 
        
        
        d3_out = self.d3(d2_out, s2_out)
  
        d4_out = self.d4(d3_out, s1_out)
 
        
        final_out = self.final_up(d4_out)
 
        
        outputs = self.outputs(final_out)
 
        
        return outputs
        
    
    
def get_model(n_classes=None, dropout_rate=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    if n_classes is None or dropout_rate is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        configs_dir = os.path.abspath(os.path.join(project_root, 'configs'))
        CONFIG_FILE_PATH = os.path.join(configs_dir, "Unet.yaml")
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = yaml.safe_load(f)

        if n_classes is None:
            n_classes = config['num_classes']
        
        if dropout_rate is None:

            dropout_rate = config.get('train', {}).get('dropout_rate', 0.3) 
    model = UnetMobileNetV2(n_classes=n_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    return model


if __name__ == "__main__":
    model_default = get_model(n_classes=None, dropout_rate=None)
    model_custom = get_model(n_classes=5, dropout_rate=0.5)
    print("model has been constructed")