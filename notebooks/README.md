We describe the different notebooks by functionality and how they were used to produce the plots in the paper.
A few notes on the terms used throughout these notebooks:

- You may see the words _target words_ or _group words_ or _gendered words_. These refer to the set of words that characterize individuals belonging to certain demographic groups. Like previous literature, we define thesse wordlists such that each word in one list has a corresponding parallel term (of the opposite group), such as ("mother", "father"), ("brother", "sister"). In our case, we focus on the following gendered pronouns: ("she", "he"), ("her", "his"), ("her", "him"), and ("herself", "himself").
- You will often seen the words _attribute words_ or _seed words_ or _words_, which refer to the target concept or association that we are interested in studying in the context of social bias. For instance, in Winobias and Winogender, the attribute words are the occupations and/or participant words used during the construction of the templates since this is what drives the creation of each template.


### Preprocessing notebooks

- [baseline__preprocessing](./Baseline__preprocessing.ipynb): preprocesses other benchmark original files, manipulating the files to obtain a similar structure that fits our proposed framework (e.g., containing "word", "template", "sentence").


### Benchmark generation notebooks

- [word selection - select attribute words](./WordSelection__Select_Attribute_Words.ipynb): carries out the first stage of the proposed framework and preprocesses the PILE vocabulary to narrow down the selection to English words only. 


### Evaluation notebooks

These notebooks carry out the post-processing of the byproducts of our framework, including tables, plots, and samples of the generated examples.

- [evaluate percentage of invalid original examples](./Evaluate__pct_invalid_original_examples.ipynb): collects the statistics of the generated benchmarks in terms of the number of sentences containing the pronoun, the exact word, among others. This notebook was run before and after running the regeneration step in the proposed framework using the following flow: (1) select attribute words, (2) 5 generate sentences for each attribute-group word pair, (3) collect information regarding whether generated sentences contain both the pronoun and the word form, (3a) if not, run the revision step, (3b) otherwise stop.

- [evaluate benchmark statistics](./Evaluate__Benchmark_statistics.ipynb): gathers simple statistics about the benchmark properties, including sentence length, number of pronouns, the position of the first and last pronouns occurring in sentences, number of words, among others. It also computes the number of gendered expressions per benchmark.

- [evaluate word-level](./Evaluate__PMI_word-level.ipynb): evaluates

- [evaluate sentence-level](./Evaluate__PMI_sentence-level.ipynb):

