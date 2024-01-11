import argparse, os, yaml
import pandas as pd

from collections import defaultdict

from model_utils import compute_perplexity, load_model
from templates import fill_template
from run_results import TEMPLATE_INFILS




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_path", required=True, type=str)
    args = parser.parse_args()

    with open(args.configs_path) as f:
        configs = yaml.safe_load(f)

    print("="*80)
    print(configs)
    print("="*80)

    model_name = configs.pop("model_name", "gpt2-xl")
    MODEL_FILENAME, MODEL, TOKENIZER, _ = load_model(model_name)
    print(MODEL.config.n_positions)

    output_dir = configs.pop("output_dir") + f"/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    print("Storing perplexity results at", output_dir)

    template_col = configs.pop("template_col", "template")
    filenames_by_exp = configs.pop("files")

    agg_results = defaultdict(list)
    for name, exp_configs in filenames_by_exp.items():
        results = defaultdict(list)
        print("-->", name, "\n", exp_configs)
        input_paths = exp_configs.pop("input_path")
        input_paths = [input_paths] if isinstance(input_paths, str) else input_paths

        for path in input_paths:
            data = pd.read_csv(path)
            templates = data[template_col].values.tolist()

            for gender in ("male", "female"):
                templates_modified = [fill_template(t, TEMPLATE_INFILS[gender]) for t in templates]

                ppls = compute_perplexity(
                    templates=templates_modified, model=MODEL, tokenizer=TOKENIZER, max_length=MODEL.config.n_positions, **exp_configs
                )

                agg_results["n"].append(len(templates))
                agg_results["mean_ppl"].append(ppls["mean_perplexity"])
                agg_results["std_ppl"].append(ppls["std_perplexity"])

                agg_results["name"].append(name)
                agg_results["gender"].append(gender)
                agg_results["input_path"].append(path)

                results["ppl"].extend(ppls["perplexities"])
                results["name"].extend([name] * len(templates_modified))
                results["gender"].extend([gender] * len(templates_modified))
                results["input_path"].extend([path] * len(templates_modified))

        results = pd.DataFrame(results)
        print("Computed individual ppls for", len(results), "sequences")
        results.to_csv(f"{output_dir}/ind_ppl__{name}.csv")
        print("\n\n\n")

    pd.DataFrame(agg_results).to_csv(f"{output_dir}/aggregate_ppls.csv")
