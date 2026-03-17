import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "cleaned" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "cleaned" / "test.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

results_df = test_df.copy()
results_df["predicted_label"] = y_pred
results_df["is_correct"] = results_df["label"] == results_df["predicted_label"]

misclassified_df = results_df[results_df["is_correct"] == False].copy()
misclassified_df.to_csv(OUTPUT_DIR / "misclassified.csv", index=False)

top_errors = (
    misclassified_df.groupby(["label", "predicted_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

top_errors.to_csv(OUTPUT_DIR / "top_error_pairs.csv", index=False)

sample_errors = []
for _, row in top_errors.head(10).iterrows():
    true_label = row["label"]
    pred_label = row["predicted_label"]

    examples = misclassified_df[
        (misclassified_df["label"] == true_label) &
        (misclassified_df["predicted_label"] == pred_label)
    ].head(5)

    for _, ex in examples.iterrows():
        sample_errors.append({
            "true_label": true_label,
            "predicted_label": pred_label,
            "text": ex["text"]
        })

sample_errors_df = pd.DataFrame(sample_errors)
sample_errors_df.to_csv(OUTPUT_DIR / "sample_error_examples.csv", index=False)

summary = {
    "total_test_samples": int(len(test_df)),
    "total_misclassified": int(len(misclassified_df)),
    "top_error_pairs": top_errors.head(10).to_dict(orient="records")
}

with open(OUTPUT_DIR / "error_analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Saved:")
print("- confusion_matrix.png")
print("- misclassified.csv")
print("- top_error_pairs.csv")
print("- sample_error_examples.csv")
print("- error_analysis_summary.json")