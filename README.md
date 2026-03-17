# Arabic sentiment classification
A compact machine learning project developed for Arabic sentiment classification using classical NLP baselines. The goal is to build a reproducible text classification pipeline with documented preprocessing, model comparison, and error analysis.


## Project Goal
This project investigates sentiment classification on Arabic social media text using TF-IDF features and linear classifiers. It is designed as a small but complete ML pipeline covering preprocessing, training, evaluation, and reproducibility.

## Dataset
- Dataset: ASTD / chosen Arabic sentiment dataset
- Task: Multi-class sentiment classification
- Input: Arabic text
- Labels: sentiment classes
- Split strategy: stratified 80/20 train-test split with `random_state=42`


## Preprocessing
The preprocessing pipeline:
- removed null rows
- removed duplicate texts
- removed URLs
- removed user mentions
- removed tatweel
- removed Arabic diacritics
- normalized selected Arabic character variants
- collapsed extra whitespace

Implementation file:
- `src/preprocess.py`
