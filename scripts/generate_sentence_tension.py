# scripts/generate_sentence_tension.py

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize
from pathlib import Path
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("vader_lexicon")

def generate_sentence_tension(input_txt="alice-text.txt",
                              output_csv="data_processed/events/alice_sentence_tension.csv"):

    text_path = Path(input_txt)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = text_path.read_text(encoding="utf-8")

    # Split into sentences
    sentences = sent_tokenize(text)

    # Sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    rows = []
    for i, sent in enumerate(sentences):
        scores = sia.polarity_scores(sent)
        tension = abs(scores["compound"])

        # ---- NEW: Convert tension to label ----
        if tension < 0.20:
            label = 0      # calm
        elif tension < 0.50:
            label = 1      # medium
        else:
            label = 2      # high

        rows.append([i, sent, tension, label])

    df = pd.DataFrame(rows, columns=["sentence_idx", "sentence", "tension", "label"])

    df.to_csv(out_path, index=False)
    print(f"Saved tension + labels file → {out_path}")
    return df


if __name__ == "__main__":
    generate_sentence_tension()
