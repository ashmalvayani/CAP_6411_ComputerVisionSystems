# For training a model, first activate the appropriate enviornment
conda env create -f environment_sophia.yml
# Activate the conda environment
conda activate cvs_ass5_qwen25vl 
# submit the batch job
sbatch finetune_sophiavl_lora.slurm
# For inference, run the following
sbatch infer_mathvista_sophiavl.slurm

# For InternVL
conda env create -f environment_internvl.yml
# Activate the conda environment
conda activate cvs_ass5_internvl
# submit the batch job
sbatch finetune_InternVL3_lora.slurm
# For inference, run the following
sbatch infer_mathvista_internvl.slurm