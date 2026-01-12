import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# LOAD DATA

df = pd.read_csv("creditcard.csv")
df.fillna(df.mean(), inplace=True)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


# SCALE (ONLY FOR LR)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#MODEL 1: LOGISTIC REGRESSION

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

cm_lr = confusion_matrix(y_test, y_pred_lr)

print("\nLOGISTIC REGRESSION REPORT")
print(classification_report(y_test, y_pred_lr))


# MODEL 2: RANDOM FOREST

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\nRANDOM FOREST REPORT")
print(classification_report(y_test, y_pred_rf))


# CONFUSION MATRICES (VISUALISED)

fig, axes = plt.subplots(1, 2, figsize=(10,4))

axes[0].imshow(cm_lr)
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm_lr[i, j], ha="center", va="center")

axes[1].imshow(cm_rf)
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm_rf[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()




