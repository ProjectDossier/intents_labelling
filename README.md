# Intents Labelling project

[![DOI](https://researchdata.tuwien.ac.at/badge/DOI/10.48436/pp7xz-n9a06.svg)](https://doi.org/10.48436/pp7xz-n9a06)


## Installation 

Create [conda](https://docs.conda.io/en/latest/miniconda.html) environment:

```bash
$ conda create --name intents_labelling python==3.8.12
```

Activate the environment:

```bash
$ source activate intents_labelling
```

Use pip to install requirements:

```bash
(intents_labelling) $ pip install -r requirements.txt
```


Install `intents_labelling` package for development

```bash
(intents_labelling) $ pip install -e .
```

Install `spacy` language model:

```bash
(intents_labelling) $ python -m spacy download en_core_web_lg
```


List of movie titles can be found [here](https://github.com/fivethirtyeight/data/blob/master/bechdel/movies.csv).

Put all data files in `data/input/` directory.


## Usage 

Create a training set which will be a sample of ORCAS dataset. Filter out testset examples

```bash
(intents_labelling) $ python intents_labelling/create_train_file.py
```

Create snorkel annotations

```bash
(intents_labelling) $ python intents_labelling/main.py
```

Train Bert model

```bash
(intents_labelling) $ python intents_labelling/models/train_bert_classifier.py
```

