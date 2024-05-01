import pandas as pd
import json, glob

from typing import Dict, List


BASE_DIR = "../other_benchmarks"


def load_eec(filepath: str = None) -> List[str]:
    if filepath is None:
        filepath = f"{BASE_DIR}/sent_analysis/EEC/Equity-Evaluation-Corpus.csv"

    df = pd.read_csv(filepath)
    return df["Sentence"].values.tolist()


def load_winogender(filepath: str = None) -> List[str]:
    if filepath is None:
        filepath = f"{BASE_DIR}/coref/WinoGender.tsv"

    df = pd.read_csv(filepath, sep="\t", lineterminator="\n")
    return df["sentence"].values.tolist()


def load_winobias(filepath: str = None, split="dev") -> List[str]:
    if filepath is None:
        filepath = f"{BASE_DIR}/coref/WinoBias/*.txt.{split}"
        filepaths = sorted(glob.glob(filepath))
    else:
        filepaths = [filepath]

    def read_winobias_filepath(fp: str) -> List[str]:
        import re

        with open(fp) as f:
            lines = [l.strip() for l in f.readlines()]
            lines = [re.sub(r"^[0-9]{1,3} ", "", l) for l in lines]
            lines = [l.replace("[", "").replace("]", "") for l in lines]
            lines = [l for l in lines if l]
        return lines

    sentences = []
    for wino_fp in filepaths:
        sents = read_winobias_filepath(wino_fp)
        print("Read", wino_fp, "with", len(sents), "sentences...")
        sentences.extend(sents)

    return sentences


def load_bug(filepath: str = None) -> List[str]:
    if filepath is None:
        filepath = f"{BASE_DIR}/coref/BUG/full_bug.csv"

    df = pd.read_csv(filepath, index_col=0)
    return df["sentence_text"].values.tolist()


def load_crows(filepath: str = None) -> List[str]:
    if filepath is None:
        filepath = f"{BASE_DIR}/language_modeling/CrowS-pairs.csv"

    df = pd.read_csv(filepath, index_col=0)
    return df["sent_more"].values.tolist() + df["sent_less"].values.tolist()


def load_stereoset(filepath: str = None) -> List[str]:
    if filepath is None:
        filepath = f"{BASE_DIR}/language_modeling/StereoSet_dev.json"

    with open(filepath) as f:
        stereo_data = json.load(f)

    # We will pick the intrasentence task as it aims to measure
    # the bias at a sentence-level reasoning
    stereo_data_templates = []
    for example in stereo_data["data"]["intrasentence"]:
        example_sents = example["sentences"]
        example_sents = [s["sentence"] for s in example_sents]
        stereo_data_templates.extend(example_sents)

    return stereo_data_templates


def load_benchmarks() -> Dict[str, List[str]]:
    return {
        "SA/EEC": load_eec(),
        "COREF/WinoGender": load_winogender(),
        "COREF/Winobias": load_winobias(),
        "COREF/BUG": load_bug(),
        "LM/Crows-pairs": load_crows(),
        "LM/Stereoset": load_stereoset(),
    }
