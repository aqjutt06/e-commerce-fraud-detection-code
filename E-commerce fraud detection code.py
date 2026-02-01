import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# Load Data
df = pd.read_csv('E-Commerce Fraud Detection.csv')
print(df.describe())
print(df.head(25))

# Mean of numeric fill, Mode of objects fill
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])
df.drop_duplicates(inplace=True)

# time data extraction and dropping uneccesary features
df['transaction_time'] = pd.to_datetime(df['transaction_time'])
df['transaction_hour'] = df['transaction_time'].dt.hour
df['transaction_day_of_week'] = df['transaction_time'].dt.dayofweek
df_clean = df.drop(columns=['transaction_id', 'user_id', 'transaction_time'])

# taking sample of 5000 rows
sns.set(style="whitegrid")
df_sample = df_clean.sample(n=5000, random_state=42)

# Histograms
df_clean.select_dtypes(include=np.number).hist(figsize=(15, 12), bins=20, color='teal', edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

# Pair Plot
pair_cols = ['amount', 'account_age_days', 'avg_amount_user', 'is_fraud']
sns.pairplot(df_sample[pair_cols], hue='is_fraud', palette='husl', diag_kind='kde')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_clean.select_dtypes(include=[np.number]).corr(), cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Distribution Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_clean, x='amount', hue='is_fraud', fill=True, common_norm=False, palette='crest')
plt.xscale('log')
plt.title('Distribution of Transaction Amount (Log Scale)')
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_fraud', y='account_age_days', data=df_clean, palette='Set2')
plt.title('Account Age vs Fraud Status')
plt.show()

# Bar chart
plt.figure(figsize=(8, 5))
sns.countplot(x='is_fraud', data=df_clean, palette='viridis')
plt.title('Class Balance: Legitimate (0) vs Fraud (1)')
plt.show()

# Classification algorithm
le = LabelEncoder()
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = le.fit_transform(df_clean[col])

X = df_clean.drop(columns=['is_fraud'])
y = df_clean['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature identification using mutual info
mi_scores = mutual_info_classif(X_train.sample(50000, random_state=42), y_train.loc[X_train.sample(50000, random_state=42).index], random_state=42)
mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=mi_series.head(10).values, y=mi_series.head(10).index, palette='viridis')
plt.title('Top 10 Features (Mutual Info)')
plt.show()

# Logistic regression model
print("\n--- Logistic Regression Results ---")
log_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_log))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()


# Decision tree model
print("\n--- Decision Tree Results ---")
decision_model = DecisionTreeClassifier(random_state=42)
decision_model.fit(X_train, y_train)

y_pred_decision = decision_model.predict(X_test)
y_prob_decision = decision_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_decision))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_decision), annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.show()


# Random Forest model
print("\n--- Random Forest Results ---")
random_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
random_model.fit(X_train, y_train)

y_pred_random = random_model.predict(X_test)
y_prob_random = random_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_random))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_random), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.show()

# ROC curve of all models
plt.figure(figsize=(10, 8))

fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
auc_log = auc(fpr_log, tpr_log)
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.2f})')

fpr_decision, tpr_decision, _ = roc_curve(y_test, y_prob_decision)
auc_decision = auc(fpr_decision, tpr_decision)
plt.plot(fpr_decision, tpr_decision, label=f'Decision Tree (AUC = {auc_decision:.2f})')

fpr_random, tpr_random, _ = roc_curve(y_test, y_prob_random)
auc_random = auc(fpr_random, tpr_random)
plt.plot(fpr_random, tpr_random, label=f'Random Forest (AUC = {auc_random:.2f})')

plt.plot([0, 1], [0, 1], 'k--') # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

