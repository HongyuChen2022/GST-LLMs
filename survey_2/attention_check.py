import pandas as pd

attention_checks = pd.DataFrame({
    "short_text": [
        'Please select "3: Neutral" for this question.',
        'Please select "1: Very Feminine" for this question.',
        'Please select "5: Very Masculine" for this question.',
    ],
    "expected_answer": [3, 1, 5],
})

attention_checks.to_csv("attention_checks.csv", index=False)

print("Created attention_checks.csv")