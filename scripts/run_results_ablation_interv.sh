# The primary intended use of Pythia is research on the behavior, functionality, and limitations of large language models. 
# This suite is intended to provide a controlled setting for performing scientific experiments. 
# We also provide 154 checkpoints per model: 
#       initial step0, 10 log-spaced checkpoints step{1,2,4...512}, 
#       and 143 evenly-spaced checkpoints from step1000 to step143000.
# These checkpoints are hosted on Hugging Face as branches. 
# Note that branch 143000 corresponds exactly to the model checkpoint on the main branch of each model. (<-- we have this...)
#
# 
# Training procedure:
# All models were trained on the exact same data, in the exact same order. Each model saw 299,892,736,000 tokens during training,
# and 143 checkpoints for each model are saved every 2,097,152,000 tokens, spaced evenly throughout training, from step1000 to step143000
# (which is the same as main). 
# 
# In addition, we also provide frequent early checkpoints: step0 and step{1,2,4...512}.
# This corresponds to training for just under 1 epoch on the Pile for non-deduplicated models, and about 1.5 epochs on the deduplicated Pile.
# All Pythia models trained for 143000 steps at a batch size of 2M (2,097,152 tokens).


# ======= RUNNING IMPACT OF INTERVENTION ======
# CREATE A VERSION OF THIS SCRIPT WITH THE REVISION KWARG FOR PYTHIA MODELS
# EleutherAI/pythia-6.9b EleutherAI/pythia-6.9b-deduped 

# Higher priority =============== (Ava-s4)
# Note: "EleutherAI/pythia-intervention-6.9b-deduped" branches from the deduped one starting in epoch 134000 ----------------------------------------------------------------

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 0 133000 

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 1 134000
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-intervention-6.9b-deduped 2 134000 

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 3 140000
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-intervention-6.9b-deduped 4 140000 

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 5 143000
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-intervention-6.9b-deduped 6 143000 

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 7 0
# ============================================================================

# AVA-S2
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 0 1000 
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 1 10000
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 2 100000
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 3 125000 

# ----- models above this line are running

# OPTIONAL (LOWER PRIORITY) ==================
# nice to have for context.... (not not necessary)
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 1 1
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 2 4
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 3 16
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 4 256
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 5 512  # 1 epoch

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 6 1000  
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 7 5000

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 0 25000 
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 1 50000 
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results_ablation_interv.sh EleutherAI/pythia-6.9b-deduped 2 125000 




MODEL_NAME=$1
DEVICE=$2
STEP=$3
PROJECT_DIR=/home/cbelem/projects/pmi_project/experiments-iclr-2024
cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/code

echo model: $MODEL_NAME
echo device: $DEVICE
# ------------------------------
# GENERATED
# ------------------------------
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-words5/final-results/revised_templates.csv" --model_revision step$STEP

CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-words10/final-results/revised_templates.csv" --model_revision step$STEP
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-words20/final-results/revised_templates.csv" --model_revision step$STEP

# -------------------------------
# BASELINES
# -------------------------------
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-baselines/final-results/coref__Winobias__templates.dev.csv" --model_revision step$STEP
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-baselines/final-results/coref__Winobias__templates.test.csv" --model_revision step$STEP
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-baselines/final-results/coref__Winogender__templates.csv" --model_revision step$STEP
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-baselines/final-results/lm__StereoSet_pronouns_only.csv" --model_revision step$STEP



