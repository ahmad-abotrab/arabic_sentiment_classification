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


## Training
	TF-IDF + Logistic Regression
	• Accuracy: 0.8024
	• Macro F1: 0.7767

	TF-IDF + Linear SVM
	• Accuracy: 0.8010
	• Macro F1: 0.7751

    Initial Observation
	• Logistic Regression slightly outperformed Linear SVM on both accuracy and macro F1.
	• The strongest class was Negative, while Positive had lower recall, suggesting more confusion with other sentiment classes.

## Error Analysis

The best model in this project was **TF-IDF + Logistic Regression**.

After reviewing the confusion matrix and the misclassified examples, the main problem was clear: the model often mixed up **Neutral** and **Negative** tweets.

### Most common mistakes
- Neutral predicted as Negative: **13,103**
- Negative predicted as Neutral: **10,522**
- Positive predicted as Negative: **8,320**

### What this means
The model works better when the sentiment is clear and strongly expressed.

It struggles more with:
- short tweets
- unclear wording
- context-dependent meaning
- tweets where neutral and negative words look similar

### Main takeaway
The biggest weakness of the model is separating **Neutral** from **Negative**.  
This is likely because TF-IDF depends on surface words, so it has trouble with subtle meaning and limited context.


## Additional Analysis

To improve interpretability, I added:
- a normalized confusion matrix to compare class-level error rates
- an analysis of the top predictive words for each sentiment class
- a chart of the most frequent misclassification pairs
- a visual comparison of the baseline models

These additions helped explain both the strengths and the weaknesses of the best-performing model.

# Installation project
 * git clone project
 * open data folder and unzip Compressed data then should be three part 
   * origin dataset (Arabic sentiment dataset)
   * cleaned dataset
   * folder clean contain (train.csv , test.csv)
 * create virtual environment (that depends on your machine)
 * pip3 install requirements.txt
 * finally you can try anything in this project
