"""Train and select an SMS spam classifier using CV on training data only."""
from pathlib import Path
import json
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent

def main():
    ds = load_dataset('ucirvine/sms_spam')['train'].to_pandas().rename(columns={'sms':'message'})
    ds = ds.dropna(subset=['message','label']).copy()
    ds['message'] = ds['message'].astype(str)
    conflicts = ds.groupby('message')['label'].nunique()
    if (conflicts > 1).any():
        raise ValueError('Identical messages with contradictory labels: review dataset')
    ds = ds.drop_duplicates(subset=['message'])
    x_train, x_test, y_train, y_test = train_test_split(ds.message, ds.label.astype(int), test_size=0.2, random_state=42, stratify=ds.label)
    candidates = {
        'MultinomialNB': MultinomialNB(),
        'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        'LinearSVC': LinearSVC(class_weight='balanced', random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparisons = []
    for name, estimator in candidates.items():
        p = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),max_features=10000,sublinear_tf=True)),('model',estimator)])
        scores = cross_validate(p, x_train, y_train, cv=cv, scoring={'f1':'f1','precision':'precision','recall':'recall'}, n_jobs=-1)
        comparisons.append({'model':name,'cv_f1_mean':float(scores['test_f1'].mean()),'cv_precision_mean':float(scores['test_precision'].mean()),'cv_recall_mean':float(scores['test_recall'].mean())})
    ranked = sorted(comparisons, key=lambda v:v['cv_f1_mean'], reverse=True)
    name = ranked[0]['model']
    best = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),max_features=10000,sublinear_tf=True)),('model',candidates[name])])
    best.fit(x_train,y_train)
    pred = best.predict(x_test)
    scores = best.predict_proba(x_test)[:,1] if hasattr(best,'predict_proba') else best.decision_function(x_test)
    result = {'model':name,'accuracy':accuracy_score(y_test,pred),'precision_spam':precision_score(y_test,pred,zero_division=0),'recall_spam':recall_score(y_test,pred,zero_division=0),'f1_spam':f1_score(y_test,pred,zero_division=0),'roc_auc':roc_auc_score(y_test,scores),'average_precision':average_precision_score(y_test,scores),'confusion_matrix':confusion_matrix(y_test,pred).tolist()}
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'reports').mkdir(exist_ok=True)
    joblib.dump(best,ROOT/'models'/'spam_classifier.joblib')
    pd.DataFrame(ranked).to_csv(ROOT/'reports'/'cross_validation.csv', index=False)
    (ROOT/'reports'/'test_metrics.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
    print('Model selected by training CV:', name)
    print('Test metrics:',json.dumps(result,indent=2))
    print(classification_report(y_test,pred,target_names=['HAM','SPAM'],zero_division=0))

if __name__ == '__main__':main()
