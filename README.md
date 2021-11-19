# Intents Labelling project


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
