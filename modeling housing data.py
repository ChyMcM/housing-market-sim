import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

df_preds = pd.read_csv(r"C:\Users\mitsu\Downloads\MIS\MIS\work\experimentation\oof_preds_h12_5pct.csv")  # adjust filename to match your run

y_true = df_preds["oof_true"].values
y_score = df_preds["oof_proba"].values

# Compute Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
pr_auc = auc(recall, precision)

# Plot
plt.figure(figsize=(7,5))
plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall (catching crashes)")
plt.ylabel("Precision (accuracy of warnings)")
plt.title("Precision–Recall Curve for Crash Prediction")
plt.legend()
plt.grid(True)
plt.show()