PROJECT_DIR=/home/cbelem/projects/pmi_project/experiments-iclr-2024

cd ../code

# each prompt is going to be tried for 11 times
# python -m run_experiment_revision --placeholders_config $PROJECT_DIR/configs/placeholders.json --revision_config $PROJECT_DIR/configs/revision.yml --input_dir $PROJECT_DIR/results-words5/words\*/step3_filter_is_likely__\*he.csv --output_dir $PROJECT_DIR/results-words5/final-results --num_retries 44 2>&1 | tee final-results-5words.log
python -m run_experiment_revision --placeholders_config $PROJECT_DIR/configs/placeholders.json --revision_config $PROJECT_DIR/configs/revision.yml --input_dir $PROJECT_DIR/results-words10/words\*/step3_filter_is_likely__\*he.csv --output_dir $PROJECT_DIR/results-words10/final-results --num_retries 44 2>&1 | tee final-results-10words.log
python -m run_experiment_revision --placeholders_config $PROJECT_DIR/configs/placeholders.json --revision_config $PROJECT_DIR/configs/revision.yml --input_dir $PROJECT_DIR/results-words20/words\*/step3_filter_is_likely__\*he.csv --output_dir $PROJECT_DIR/results-words20/final-results --num_retries 44 2>&1 | tee final-results-20words.log

cd ../scripts