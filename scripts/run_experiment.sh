PROJECT_DIR=/home/cbelem/projects/pmi_project/experiments-iclr-2024

cd ../code
python -m run_experiment --exp_config $PROJECT_DIR/configs/words5/prompt1_he.yml --placeholders_config $PROJECT_DIR/configs/placeholders.json
python -m run_experiment --exp_config $PROJECT_DIR/configs/words5/prompt1_she.yml --placeholders_config $PROJECT_DIR/configs/placeholders.json


python -m run_experiment --exp_config $PROJECT_DIR/configs/words10/prompt1_he.yml --placeholders_config $PROJECT_DIR/configs/placeholders.json
python -m run_experiment --exp_config $PROJECT_DIR/configs/words10/prompt1_she.yml --placeholders_config $PROJECT_DIR/configs/placeholders.json


python -m run_experiment --exp_config $PROJECT_DIR/configs/words20/prompt1_he.yml --placeholders_config $PROJECT_DIR/configs/placeholders.json
python -m run_experiment --exp_config $PROJECT_DIR/configs/words20/prompt1_she.yml --placeholders_config $PROJECT_DIR/configs/placeholders.json

cd ../scripts