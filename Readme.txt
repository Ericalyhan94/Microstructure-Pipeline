Segmentation_Pipeline/
├── configs/                 
│   └── config.py
│         └── MatSegNet.yaml
│         └── Segformer.yaml
│         └── Unet.yaml
├── data/                     
│   ├── SEM_images/           
│   ├── datasets/              
│  		└──bainite_set
│		└──martensite_set
│		└──training_set
│		└──validation_set
│		└──test_set
├── models/   
│   ├──MatSegNet.py
│   ├──Segformer.py
│   ├──Unet.py
├── output/                  
│   ├── checkpoints/
│         └──best_matsegnet.pth
│         └──best_segformer.pth
│         └──best_unet_mobilenetv2.pth
│         └──matsegnet.pth
│         └──segformer.pth
│         └──unet_mobilenetv2.pth
│   └── accuracy_output/     
├── src/            
│   ├── datasets/
│         └──preprocessing.py
│         └──load_data.py
│         └──checkpoints.py
│         └──training.py
│         └──visualization.py
├── scripts/
│         └──segment_images.py
│         └──train_test_split.py
│         └──train.py
│         └──visualize_results.py
│         └──carbide_morphology.py
│         └──size_aspect_ratio.py

├── .gitignore              
├── requirements.txt          
└── README.md     

