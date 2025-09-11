## For downloading dataset
```
cd scripts
python download_dataset.py
```

## For training ResNet-18
```
cd scripts
sbatch train_resnet.slurm
```

## For training ViT
```
cd scripts
sbatch train_vit.slurm
```

## Run live_demo for ResNet-18
```
python live_demo.py   --test_dir data/HumanActionRecognition/Structured/test   --resnet_csv Output/ResNet-18/resnet_test_predictions.csv   --vit_csv Output/ResNet-18/resnet_test_predictions.csv   --save_dir live_demo_outputs/ResNet-18   --num 25
```

## Generate a Demo-Video for ResNet-18
```
ffmpeg -y -framerate 4 -pattern_type glob -i 'live_demo_outputs/ResNet-18/*_demo.png' -c:v libx264 -pix_fmt yuv420p live_demo_outputs/live_demo_resnet.mp4
```

## Run live_demo for ViT
```
python live_demo.py   --test_dir data/HumanActionRecognition/Structured/test   --resnet_csv Output/ViT/vit_test_predictions.csv   --vit_csv Output/ViT/vit_test_predictions.csv   --save_dir live_demo_outputs/ViT   --num 25
```

## Generate a Demo-Video for ViT
```
ffmpeg -y -framerate 4 -pattern_type glob -i 'live_demo_outputs/ViT/*_demo.png' -c:v libx264 -pix_fmt yuv420p live_demo_outputs/live_demo_vit.mp4
```
