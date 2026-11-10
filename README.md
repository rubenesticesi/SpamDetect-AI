# SpamDetect AI — SMS Spam Detection using NLP and Machine Learning

Academic project: detection of unwanted SMS messages through exploratory data analysis, TF-IDF, supervised classifiers and a local Flask web application.

## Team members

- Member 1: [Name]
- Member 2: [Name]
- Member 3: [Name]
- Member 4: [Name]

## Dataset

Reference: [SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), loaded with `datasets.load_dataset('ucirvine/sms_spam')`. **Important:** Confirm the exact dataset and pretrained model indicated in the course links before final submission.

## Project layout

- `notebook/01_spam_detection.ipynb`: reproducible EDA and experiments.
- `train.py`: cross-validation on training data, model selection, held-out evaluation and artifact export.
- `app/app.py`: Flask interface.
- `app/templates/index.html`: user interface.
- `models/`: generated classifier (not tracked).
- `reports/`: model metrics (generated).
- `tests/`: basic automated checks.

## Setup

Python 3.10–3.12 recommended. Use a fresh virtual environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
python app/app.py
```

Navigate to http://127.0.0.1:5000 . For notebook exploration run `jupyter notebook`.

## Methodology

1. Validate classes, missing data and duplicates before splitting.
2. Stratified 80/20 train/test split.
3. Pipeline with TF-IDF (unigrams and bigrams) and three estimators: MultinomialNB, LogisticRegression, LinearSVC.
4. Five-fold stratified CV **on training only** selects the estimator by positive-class F1.
5. Evaluate once on held-out test, reporting Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision and a confusion matrix.
6. Flask inference loads the locally trained model.

## Pretrained model

A possible reference is [`mrm8488/bert-tiny-finetuned-sms-spam-detection`](https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection). The course's model link has not been supplied. Because the reference model may have trained on the SMS Spam Collection, its scores on that dataset do not establish independent generalization.

## Results

No experimental scores are claimed in advance. Run `python train.py` and inspect `reports/test_metrics.json` and `reports/cross_validation.csv`.

## Limitations and safety

English-language historical SMS data may not generalize to current phishing/spam or to Spanish texts. False positives and false negatives are possible. Do not treat predictions as security guarantees. Never load untrusted pickle/joblib artifacts.
