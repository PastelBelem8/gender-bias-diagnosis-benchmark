from openai_utils import get_completion_block_until_succeed
from run_experiment import *
from itertools import cycle

import os, glob, re, tqdm
import pandas as pd

import argparse, os, pickle


def read_cli_args():
    # Example use: python -m run_experiment_revision
    # --placeholders_config /home/cbelem/projects/pmi_project/experiments-iclr-2024/configs/placeholders.json
    # --revision_config /home/cbelem/projects/pmi_project/experiments-iclr-2024/configs/revision.yml
    # --input_dir /home/cbelem/projects/pmi_project/experiments-iclr-2024/results-words5/words*/step3_filter_is_likely__*he.csv
    # --output_dir /home/cbelem/projects/pmi_project/experiments-iclr-2024/results-words5/final-results
    # --num_tries 30
    # --dist_threshold 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--placeholders_config", required=True, type=str, help="Contains the definition of the placeholders to use in the parse of the templates.")
    parser.add_argument("--revision_config", required=True, type=str, help="Revision config file. Should contain a list of prompts to perform the edit.")
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--num_retries", default=40, type=int)
    parser.add_argument("--dist_threshold", default=0, type=int)

    args = parser.parse_args()
    return args


def is_word_in_template(data) -> bool:
    # contractions can be tricky so we'll account for that
    word, sentence = data["word"].lower(), data["sentence"].lower()
    return re.search(f"\\b{word}\\b", sentence) is not None

def is_likely_both(data) -> bool:
    dct = eval(data)
    return dct["male"] == "likely" and dct["female"] == "likely"


def run_revise_step(example: dict, num_retries: int, threshold: int, placeholders_configs: dict, experiment_configs: dict):
    conversion_configs, filtering_configs, revision_configs = [experiment_configs[name] for name in ("conversion", "filters", "revision")]

    template = example["template"]
    sentence = example["sentence"]
    target_word = example["target_word"]

    if target_word in ("he", "his", "him", "himself"):
        pronouns = "he/his/him/himself"
    else:
        pronouns = "she/her/her/herself"
    # ^Note: even if the sentence was created originally with female, we want to make sure that we can
    # Step 1.
    attempt = 0
    for prompt_name, configs in cycle(revision_configs.items()):
        if attempt > num_retries: break
        prompt = configs["prompt"]
        prompt = prompt.replace("{sentence}", sentence)
        prompt = prompt.replace("{word}", example["word"])
        prompt = prompt.replace("{target_word}", pronouns)

        configs = {k: v for k,v in configs.items() if k != "prompt"}
        response = get_completion_block_until_succeed(prompt, **configs)

        # Check for word, check for pronoun
        has_word = is_word_in_template({"word": example["word"], "sentence": response})

        if not has_word:  # If still doesn't have the word, just skip to the next round of editing
            attempt +=1
            continue

        new_example = example.copy()
        new_example["sentence"] = response
        new_example["has_word"] = has_word

        # Convert if necessary
        conv_mapping = conversion_configs.get("mapping", placeholders_configs["gender_to_placeholder"])
        conv_df, _ = run_convert_step(
            target_word=target_word,
            output_dir=None,
            input_df=pd.DataFrame({"sentence": [response]}),
            placeholder_mapping=conv_mapping,
            **conversion_configs,
        )
        new_example.update(conv_df.loc[0])

        if not new_example["has_placeholder"]: # If it contains the word, let's check the pronoun
            attempt +=1
            continue

        # Collect likely/unlikely
        for filter_name, filter_kwargs in filtering_configs.items():
             #print("Applying filter:", filter_name, "with kwargs:", filter_kwargs)
            filt_mapping = filter_kwargs.get("mapping", placeholders_configs["placeholder_to_gender"])
            filter_results, _ = run_filter_step(target_word=new_example["word"],
                                                templates=[template],
                                                placeholder_mapping=filt_mapping,
                                                output_dir=None, input_path=None,
                                                **filter_kwargs)
            new_example[filter_kwargs["colname"]] = filter_results[0]

        # print(f"New example after {attempt} using prompt '{prompt_name}': {prompt}")
        return new_example

    print(f"Couldn't revise the template for example (after {attempt}):", example["word"], "\t", example['template'])
    return None


def run_revise_experiment(args, **kwargs):
    # Dump configs in new directory
    with open(f"{args.output_dir}/configs_revise.yml", "wt") as f:
        yaml.dump(kwargs, f)

    # Load the files
    filepaths = sorted(glob.glob(args.input_dir))
    print(f"Found {len(filepaths)} at {args.input_dir}: \n-", "\n- ".join(filepaths))

    gen_df = pd.concat([pd.read_csv(f) for f in filepaths]).reset_index(drop=True)
    print("Loaded a total of", len(gen_df), "sentences...")

    num_need_revision = 0
    final_results = []
    failed_examples = []
    for _, example in tqdm.tqdm(gen_df.iterrows()):
        has_word = is_word_in_template({k: example[k] for k in ("word", "sentence")})
        has_pronoun = example["has_placeholder"]
        example["is_natural"] =  is_likely_both(example["likely_under"])

        if has_word and has_pronoun:
            example["has_word"] = has_word
            example["is_revised"] = False
            final_results.append(example)
        else:
            revised_example = run_revise_step(example, **kwargs)
            num_need_revision += 1

            if revised_example is not None:
                revised_example["is_natural"] = is_likely_both(example["likely_under"])
                revised_example["is_revised"] = True
                final_results.append(revised_example)
            else:
                failed_examples.append(example)

    print("Initial number of examples:", len(gen_df))
    print("Revised:", num_need_revision)
    print("Failed to revise:", len(failed_examples))
    print(f"Final well-defined templates: {len(final_results)/len(gen_df):.2%}")
    print(type(final_results), type(failed_examples))

    try:
        pd.DataFrame(final_results).to_csv(f"{args.output_dir}/revised_templates.csv")
    except:
        with open(f"{args.output_dir}/revised_templates.pkl", 'wb') as file:
            pickle.dump(final_results, file)
    try:
        pd.DataFrame(failed_examples).to_csv(f"{args.output_dir}/failed_templates.csv")
    except:
        with open(f"{args.output_dir}/failed_templates.pkl", 'wb') as file:
            pickle.dump(failed_examples, file)

    print_sep("FINISHED!!!")


if __name__ == "__main__":
    import json, yaml

    args = read_cli_args()
    print_sep("Loading placeholders")
    with open(args.placeholders_config, "r") as f:
        plc_configs = json.load(f)
    print("Loaded:", plc_configs)

    print_sep("Loading revision configs")
    with open(args.revision_config, "rt") as f:
        revision_configs = yaml.safe_load(f)
    print("Loaded:", revision_configs)

    os.makedirs(args.output_dir, exist_ok=True)
    run_revise_experiment(
        args,
        num_retries=args.num_retries,
        threshold=args.dist_threshold,
        placeholders_configs=plc_configs,
        experiment_configs=revision_configs,
    )
