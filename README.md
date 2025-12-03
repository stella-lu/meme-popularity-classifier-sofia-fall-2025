## Background

Sofia University, Artificial Intelligence class, Fall 2025.

Dataset was taken from here, and it is also saved as a JSON in this repo.
https://www.kaggle.com/datasets/sayangoswami/reddit-memes-dataset

See the slideshow or the youtube video (link in slideshow) for a walkthrough.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install tensorflow scikit-learn matplotlib pillow
```


## Run Training

Modify params in `train_meme_model.py`

```
source .venv/bin/activate
python train_meme_model.py
```

## Run Demo

Save images into folder `demo_memes`

```
source .venv/bin/activate
python demo_predict.py
```
