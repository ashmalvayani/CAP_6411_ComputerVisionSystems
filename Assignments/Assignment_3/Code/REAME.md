## To download the dataset, run
python download_dataset.py

## Correct images for the dataset, because the kaggle data has uneven pixel distribution. So, downloading from the source
git clone https://github.com/lih627/CamVid.git
python original_data.py

## Create the conda environment
conda env create -f environment.yml

## or below if environment is created
conda activate cvs_ass3

## clone the GroundingDINO repo
git clone https://github.com/IDEA-Research/GroundingDINO.git

## follow the steps here to setup the GroundingDINO repo, or if the conda env was setup, it should be fine too
https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install

## Run the job to perform the evaluation
sbatch sam2.slurm
