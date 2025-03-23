#!/bin/bash

declare -a sample_steps_array=(500 50 5 1)
declare -a num_samples_array=(2 4)

# Run experiments for each configuration
for steps in "${sample_steps_array[@]}"; do
  for samples in "${num_samples_array[@]}"; do
    # Create a unique results directory for this configuration
    results_dir="results_steps${steps}_samples${samples}"
    mkdir -p "$results_dir"
    
    echo "Running experiment with sample_steps=$steps, num_samples=$samples"

    python scripts/inference.py \
      --outdir "$results_dir" \
      --testdir examples \
      --num_samples "$samples" \
      --sample_steps "$steps" \
      --gpu 0

    echo "Completed experiment with sample_steps=$steps, num_samples=$samples"
    echo "Results saved in: $results_dir"
    echo "----------------------------------------"

  done
done

######################################################################
# source ~/Hier-Legal-Graph/mimic_dataset/mimiciv_dataset/2.2/ObjectStitch-Image-Composition/Object-Stitch/bin/activate
# python scripts/inference.py \
# --outdir results \
# --testdir examples \
# --num_samples 3 \
# --sample_steps 50 \
# --gpu 0

######################################################################
# I need to run these commands before running the bash tesh.sh command
# conda deactivate
# conda deactivate
# source Object-Stitch/bin/activate
# export PYTHONPATH=$PYTHONPATH:$(pwd)/src/taming-transformers
#