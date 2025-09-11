## Activate the conda environment
```
conda activate cvs_ass2
```

## For Downloading Dataset
```
cd Assignment_2/scripts
python download_data.py
```

## FasterRCNN
```
cd Assignment_2/scripts
sbatch FasterRCNN.slurm
```

## DETR
```
cd Assignment_2/scripts
sbatch DETR.slurm
```

## Grounding DINO
Download the weights from this link
```
cd GroundingDINO/Weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
bash scripts/DINO_eval.sh /home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017 Weights/checkpoint0011_4scale.pth
```

## DINO
Download the  DINO model checkpoint "checkpoint0011_4scale.pth" from this link https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_ in the Weights folder.
```
cd DINO/Weights
gdown https://drive.google.com/uc?id=1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_ --output checkpoint0011_4scale.pth

cd Assignment_2/scripts
conda activate cvs_ass2_dino
sbatch DINO.slurm
```
