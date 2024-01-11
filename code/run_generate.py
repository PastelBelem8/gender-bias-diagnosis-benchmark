from transformers import AutoTokenizer, AutoModelForCausalLM
from model_utils import load_model
from typing import Dict, List

import argparse, yaml, os, time
import numpy as np
import pandas as pd
import torch
import tqdm

def init_seed(seed: int):
    """Set random seed to ensure reproducibility of results."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


def init_pad_token(tokenizer: AutoTokenizer):
    """Initialize the pad token to be the EOS token in case pad token is not defined
    for the specified tokenizer."""
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print(f"SETTING pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def generate__autoreg(
        prefixes: List[str],
        num_samples_per_prefix: int,
        batch_size: int,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        seed: int=None,
        add_bos_token: bool=True,
        **generation_kwargs,
    ) -> Dict[str, np.array]:
    """Autoregressively generates num_samples_per_prefix for each prefix in
    batches of batch_size."""
    init_seed(seed); init_pad_token(tokenizer);
    device = model.device

    # Add default generation kwargs (it will override users' definitions)
    generation_kwargs.update(
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Let's duplicate each example by num_samples_per_prefix
    # Note: associate a unique id to each of the prefixes so that we can
    # recover uniquely which sequences were generated wrt to each prefix.
    prefixes_ids = [i for i in range(len(prefixes))]
    prefixes = prefixes * num_samples_per_prefix
    prefixes_ids = prefixes_ids * num_samples_per_prefix

    generated_seqs, generated_seqs_scores = [], []
    generated_seqs_length = []
    for start in tqdm.tqdm(range(0, len(prefixes), batch_size)):
        end = min(start+batch_size, len(prefixes))

        batch = prefixes[start:end]
        if add_bos_token:
            batch = [tokenizer.bos_token + p for p in batch]

        batch_enc = tokenizer.batch_encode_plus(batch, return_tensors="pt", add_special_tokens=False, padding=True)
        input_ids = batch_enc.input_ids.to(device)
        attention_mask = batch_enc.attention_mask.to(device)

        # Generate sequences
        outputs = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
        sequences = outputs.sequences

        # Make sure the pad token is not accounted for in the loss
        targets = sequences.clone()
        targets[sequences == tokenizer.pad_token_id] = -100

        # Compute each continuation's probability
        mask = sequences == tokenizer.pad_token_id
        attention_mask = torch.ones_like(sequences)
        attention_mask[mask] = 0
        results = model(sequences, attention_mask=attention_mask, labels=sequences)
        batch_score = -results.loss.cpu().detach().numpy()

        # Based on the discussion at
        # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/20
        logits = torch.log_softmax(results.logits, dim=-1).detach()
        # collect the probability of the generated token
        # -- probability at index 0 corresponds to the token at index 1
        logits, input_ids = logits[:, :-1, :], sequences[:,1:,None]

        # Scores per token of the template
        batch_seq_scores = torch.gather(logits, 2, input_ids).squeeze(-1)

        # Sanity check of log scores computation
        _avg_loss = batch_seq_scores.mean(dim=-1).mean().item()
        assert np.abs(_avg_loss - batch_score) <= 1e-5, f"Loss does not match: (batch: {input_ids})), {_avg_loss} - {batch_score} > 1e-6"

        generated_seqs.extend(tokenizer.batch_decode(sequences, skip_special_tokens=True))
        generated_seqs_scores.extend(batch_seq_scores.sum(dim=-1).detach().cpu().numpy().tolist())
        generated_seqs_length.extend(attention_mask.sum(dim=-1).detach().cpu().numpy().tolist())

    return {
        "prefix": prefixes,
        "generated_seqs": generated_seqs,
        "generated_seqs_scores": generated_seqs_scores,
        "generated_seqs_num_tokens": generated_seqs_length,
        "_prefix_id": prefixes_ids,
    }


if __name__ == "__main__":
    #
    # Example use: python -m run_generate --model_name gpt2 --filename ../ --colnames M_template,F_template --generation_config ../configs/generate/greedy.yml --output_filename ../out
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--model_revision", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=32, type=int)

    parser.add_argument("--filename", required=True, type=str)
    parser.add_argument("--colnames", required=True, type=str)
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--output_filename", required=True, type=str)
    args = parser.parse_args()

    print("="*80)
    print(f"Starting Experiment\n[Experiment] Configs: {args}")
    print("="*80)

    # --------------------------------------------------------
    # Load the config file
    # --------------------------------------------------------
    with open(args.config_path, "r") as f:
        configs = yaml.safe_load(f)
    print("Loaded configs:", configs)

    # -------------------------------------------------------
    # Load data path
    # -------------------------------------------------------
    print("Reading dataset from:\n-->", args.filename)
    data = pd.read_csv(args.filename, index_col=0)

    output_dir, _, basename = args.output_filename.rpartition("/")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------
    # Load model
    # -------------------------------------------------------
    print("Loading model...")
    model_name, model, tokenizer, device = load_model(
        name=args.model_name,
        revision=args.model_revision,
        device=args.device,
    )

    output_path = f"{output_dir}/{model_name}__{basename}"
    # -------------------------------------------------------
    # Load decoding algorithms
    # -------------------------------------------------------
    colnames = args.colnames.split(",")
    for col in colnames:
        assert col in data.columns, f"Specified {col} is not in data columns {data.columns}\n\t(from file {args.filename})"


    prefixes_by_col = [data[col].values.tolist() for col in colnames]
    gen_kwargs = configs.pop("generation_kwargs", {})

    final_data = {}
    for prefixes, col in zip(prefixes_by_col, colnames):
        print(f"Processing {len(prefixes)}: {col}")
        start = time.time()
        # Copy the generation kwargs so that it can be re-used in the next round
        gen_kwargs = {k: v for k, v in gen_kwargs.items()}
        gen_results = generate__autoreg(
            prefixes=prefixes,
            model=model, tokenizer=tokenizer,
            batch_size=args.batch_size,
            **configs, **gen_kwargs,
        )
        end = time.time()
        print("\n\Generation duration:", (end - start) / 60, "min")
        final_data.update({f"{col}_{k}": v for k,v in gen_results.items()})

    print("Creating results file in", output_path)
    pd.DataFrame(final_data).to_csv(output_path, index=None)
    print("Done!")
