# Notebook utilities

GROUP_PAIRED_WORDLIST = [
    ("she", "he"),
    ("her", "his"),
    ("her", "him"),
    ("hers", "his"),
    ("grandmother", "grandfather"),
    ("grandma", "grandpa"),
    ("stepmother", "stepfather"),
    ("stepmom", "stepdad"),
    ("mother", "father"),
    ("mom", "dad"),
    ("aunt", "uncle"),
    ("aunts", "uncles"),
    ("mummy", "daddy"),
    ("sister", "brother"),
    ("sisters", "brothers"),
    ("daughter", "son"),
    ("daughters", "sons"),
    ("female", "male"),
    ("females", "males"),
    ("feminine", "masculine"),
    ("woman", "man"),
    ("women", "men"),
    ("madam", "sir"),
    ("matriarchy", "patriarchy"),
    ("girl", "boy"),
    ("lass", "lad"),
    ("girls", "boys"),
    ("girlfriend", "boyfriend"),
    ("girlfriends", "boyfriends"),
    ("wife", "husband"),
    ("wives", "husbands"),
    ("queen", "king"),
    ("queens", "kings"),
    ("princess", "prince"),
    ("princesses", "princes"),
    ("lady", "lord"),
    ("ladies", "lords"),
]
# unpack the previous list into female, male
FEMALE_WORDS, MALE_WORDS = zip(*GROUP_PAIRED_WORDLIST)


def canonic_model_name(model_name: str) -> str:
    if "EleutherAI__" in model_name:
        model_name = model_name.replace("EleutherAI__", "")
    elif "facebook__" in model_name:
        model_name = model_name.replace("facebook__", "")
    elif "llama" in model_name:
        ix = model_name.index("llama")
        model_name = model_name[ix:].replace("__hf_models__", "-")
    elif "mosaicml__" in model_name:
        model_name = model_name.replace("mosaicml__", "")
        
    if "deduped" in model_name:
        model_name = model_name.replace("-deduped", " (D)")
    return model_name


def get_model_size(canonic_name: str) -> int:
    import re 
    val = re.search(r"(\d+(\.\d+)?)(b|B|m|M)", canonic_name)[0]
    const = 1_000 if val[-1] in ("b", "B") else 1        
    return float(val[:-1]) * const