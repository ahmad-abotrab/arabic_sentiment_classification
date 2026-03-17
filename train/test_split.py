from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    cleaned_df,
    test_size=0.2,
    random_state=42,
    stratify=cleaned_df["label"]
)

train_df.to_csv("../data/cleaned/train.csv", index=False)
test_df.to_csv("../data/cleaned/test.csv", index=False)

print(train_df.shape, test_df.shape)
print(train_df["label"].value_counts(normalize=True))
print(test_df["label"].value_counts(normalize=True))