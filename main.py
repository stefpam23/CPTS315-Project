import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import preprocess_function  # Assuming you have created a separate file for preprocessing
import matplotlib.pyplot as plt

def plot_predictions(comment_id, predictions_df):
    comment = test.loc[test['id'] == comment_id, 'comment_text'].values[0]
    probabilities = predictions_df.loc[predictions_df['id'] == comment_id, toxicity_categories].values[0]
    
    fig, ax = plt.subplots()
    ax.barh(toxicity_categories, probabilities)
    ax.set_xlabel('Probability')
    ax.set_title(f'Toxicity probabilities for comment ID {comment_id}')
    plt.show()

    print("Comment text:")
    print(comment)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['processed_text'] = Parallel(n_jobs=-1)(delayed(preprocess_function.preprocess)(text) for text in train['comment_text'])
test['processed_text'] = Parallel(n_jobs=-1)(delayed(preprocess_function.preprocess)(text) for text in test['comment_text'])

X_train, X_val, y_train, y_val = train_test_split(train['processed_text'], train.iloc[:, 2:], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

toxicity_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def train_model(category):
    model = LogisticRegression()
    model.fit(X_train_vec, y_train[category])
    y_pred_prob = model.predict_proba(X_val_vec)[:, 1]
    roc_auc = roc_auc_score(y_val[category], y_pred_prob)
    return model, roc_auc

models_and_scores = Parallel(n_jobs=-1)(delayed(train_model)(category) for category in toxicity_categories)

models = [model for model, _ in models_and_scores]
roc_auc_scores = [score for _, score in models_and_scores]

print("ROC AUC scores:", roc_auc_scores)

X_all = train['processed_text']
y_all = train.iloc[:, 2:]
X_all_vec = vectorizer.fit_transform(X_all)

for category, model in zip(toxicity_categories, models):
    model.fit(X_all_vec, y_all[category])

X_test = test['processed_text']
X_test_vec = vectorizer.transform(X_test)
test_predictions = [model.predict_proba(X_test_vec)[:, 1] for model in models]

test_predictions_df = pd.DataFrame(np.array(test_predictions).T, columns=toxicity_categories)
test_predictions_df['id'] = test['id']
test_predictions_df = test_predictions_df[['id'] + toxicity_categories]
test_predictions_df.to_csv('test_predictions.csv', index=False)

# Replace '00001cee341fdb12' with a comment ID from the test dataset
comment_id = '00001cee341fdb12'
plot_predictions(comment_id, test_predictions_df)
