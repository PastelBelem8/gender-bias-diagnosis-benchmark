# ======= RUNNING Pythia (FULLY TRAINED ON PILE) =======

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-70m 0
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-70m-deduped 1

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-160m 2
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-160m-deduped 3

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-410m 0
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-410m-deduped 1

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-1.4b 2
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-1.4b-deduped 2


# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-2.8b 3
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-2.8b-deduped 4


# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-6.9b 5
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-6.9b-deduped 6


# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/gpt-j-6b 7

# ======= RUNNING OPT (PARTIAL TRAINED ON PILE)  =======
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh facebook/opt-125m 0
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh facebook/opt-350m 0
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh facebook/opt-2.7b 1
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh facebook/opt-6.7b 2


# ======= RUNNING OTHERS (PARTIAL TRAINED ON PILE)  =======

# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh /extra/ucinlp1/llama-2/hf_models/7B 0,1,2
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh mosaicml/mpt-7b 3,4,5
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh mosaicml/mpt-30b 0,1,2,3

# bloom

# LARGER MODELS -- MAY BE TRICKY
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-12b 1,2,3
# conda activate py39; cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/scripts; ./run_results.sh EleutherAI/pythia-12b-deduped 6




MODEL_NAME=$1
DEVICE=$2
PROJECT_DIR=/home/cbelem/projects/pmi_project/experiments-iclr-2024
cd /home/cbelem/projects/pmi_project/experiments-iclr-2024/code

echo model: $MODEL_NAME
echo device: $DEVICE
# ------------------------------
# GENERATED
# ------------------------------
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-words5/final-results/revised_templates.csv"
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-words10/final-results/revised_templates.csv"
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-words20/final-results/revised_templates.csv"

# -------------------------------
# BASELINES
# -------------------------------
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-baselines/final-results/coref__Winobias__templates.dev.csv"
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-baselines/final-results/coref__Winobias__templates.test.csv"
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-baselines/final-results/coref__Winogender__templates.csv"
CUDA_VISIBLE_DEVICES=$DEVICE python -m run_results --model_name $MODEL_NAME --filename "$PROJECT_DIR/results-baselines/final-results/lm__StereoSet_pronouns_only.csv"


