
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def classification_metrics(y_true,y_pred,y_prob=None):
    return {
        "Accuracy":accuracy_score(y_true,y_pred),
        "Precision":precision_score(y_true,y_pred,zero_division=0),
        "Recall":recall_score(y_true,y_pred,zero_division=0),
        "F1":f1_score(y_true,y_pred,zero_division=0),
        "ROC_AUC":roc_auc_score(y_true,y_prob) if y_prob is not None else np.nan,
        "NonPotable_Recall":recall_score(y_true,y_pred,pos_label=0,zero_division=0),
    }

def save_selected_evaluation(y_true,y_pred,y_prob,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    pd.DataFrame(cm,index=["Actual Non-Potable","Actual Potable"],columns=["Predicted Non-Potable","Predicted Potable"]).to_csv(out/"confusion_matrix.csv")
    fig,ax=plt.subplots(figsize=(6.5,5.2)); im=ax.imshow(cm,cmap="Blues")
    fig.colorbar(im,ax=ax); ax.set_xticks([0,1],["Non-Potable","Potable"]); ax.set_yticks([0,1],["Non-Potable","Potable"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix — Selected Model")
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=190); plt.close(fig)
    if y_prob is not None:
        fpr,tpr,_=roc_curve(y_true,y_prob); auc=roc_auc_score(y_true,y_prob)
        fig,ax=plt.subplots(figsize=(6.5,5.2)); ax.plot(fpr,tpr,label=f"AUC = {auc:.3f}"); ax.plot([0,1],[0,1],"--",label="No-skill reference")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve — Selected Model"); ax.legend()
        fig.tight_layout(); fig.savefig(out/"roc_curve.png",dpi=190); plt.close(fig)
    pd.DataFrame([classification_metrics(y_true,y_pred,y_prob)]).to_csv(out/"test_metrics.csv",index=False)
