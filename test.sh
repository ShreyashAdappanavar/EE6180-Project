#!/bin/bash

## Run these commands in the same order to run the test.sh script:
# conda deactivate
# conda deactivate
# conda activate Object-Stitch
# export PYTHONPATH=$PYTHONPATH:$(pwd)/src/taming-transformers
# chmod +x test.sh
# bash test.sh

#### OR, JUST COPY THIS (REMEMBER TO CHANGE THE PATH WHERE YOUR ENV IS LOCATED)
# cd /home/gokul/Hier-Legal-Graph/mimic_dataset/mimiciv_dataset/2.2/Tempp/ObjectStitch-Image-Composition && conda deactivate && conda deactivate && conda activate Object-Stitch && export PYTHONPATH=$PYTHONPATH:$(pwd)/src/taming-transformers && chmod +x test.sh && bash test.sh

declare -a sample_steps_array=(100)
declare -a num_samples_array=(3)

# Run experiments for each configuration
for steps in "${sample_steps_array[@]}"; do
  for samples in "${num_samples_array[@]}"; do
    # Create a unique results directory for this configuration
    results_dir="results_steps${steps}_samples${samples}"
    mkdir -p "$results_dir"
    
    echo "Running experiment with sample_steps=$steps, num_samples=$samples"

    python3 scripts/inference.py \
      --outdir "$results_dir" \
      --testdir Generate_Masks \
      --num_samples "$samples" \
      --sample_steps "$steps" \
      --gpu 1

    echo "Completed experiment with sample_steps=$steps, num_samples=$samples"
    echo "Results saved in: $results_dir"
    echo "----------------------------------------"

  done
done

######################################################################
# source /home/gokul/Hier-Legal-Graph/mimic_dataset/mimiciv_dataset/2.2/Tempp/ObjectStitch-Image-Composition/Object-Stitch/bin/activate
# python scripts/inference.py \
# --outdir results \
# --testdir examples \
# --num_samples 3 \
# --sample_steps 50 \
# --gpu 0

######################################################################
# I need to run these commands before running the bash test.sh command

# cd /home/gokul/Hier-Legal-Graph/mimic_dataset/mimiciv_dataset/2.2/Tempp/ObjectStitch-Image-Composition && conda deactivate && conda deactivate && conda activate Object-Stitch && export PYTHONPATH=$PYTHONPATH:$(pwd)/src/taming-transformers && chmod +x test.sh && bash test.sh