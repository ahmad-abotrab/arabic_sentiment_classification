## Error Analysis Summary

- Total test samples: 235,483
- Total misclassified samples: 46,527

### Top error pairs
- Neutral → Negative: 13,103
- Negative → Neutral: 10,522
- Positive → Negative: 8,320
- Positive → Neutral: 6,411

### Interpretation
The strongest class was Negative, while the main weakness of the model was distinguishing Neutral from Negative. This indicates that short or ambiguous tweets, as well as tweets with overlapping lexical sentiment cues, remain difficult for TF-IDF based linear models.