
from pathlib import Path
import json, sys, platform, time
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import load_dataset, detect_target, dataset_summary
from preprocessing import build_preprocessor
from model_selection import selection_f1, SELECTION_METRIC
from model_selection import non_potable_recall
from eda import run_eda
from evaluate import save_selected_evaluation

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/water_potability.csv"
MODEL_DIR=ROOT/"models"; OUT=ROOT/"outputs"; EDA=OUT/"eda"; EVAL=OUT/"evaluation"
for p in [MODEL_DIR,OUT,EDA,EVAL]: p.mkdir(parents=True,exist_ok=True)
RANDOM_STATE=42
TEST_SIZE=.20

def iqr_outlier_counts(df, features):
    rows=[]
    for c in features:
        s=df[c].dropna()
        q1,q3=s.quantile([.25,.75]); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr
        rows.append({"column":c,"q1":q1,"q3":q3,"iqr":iqr,"lower_fence":lo,"upper_fence":hi,
                     "outlier_count":int(((s<lo)|(s>hi)).sum()),"outlier_percent":float(((s<lo)|(s>hi)).mean()*100)})
    return pd.DataFrame(rows)

def main():
    start=time.time()
    df=load_dataset(DATA)
    target=detect_target(df)
    if set(df[target].dropna().unique()) != {0,1}:
        raise ValueError(f"Expected binary 0/1 target; found {sorted(df[target].dropna().unique())}")
    summary=dataset_summary(df,target)
    print("\n=== AquaGuard Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Target: {target}")
    print("Class distribution:", summary["class_distribution"])
    print("Missing values:", {k:v for k,v in summary["missing_values"].items() if v})
    print("Duplicate rows:", summary["duplicate_rows"])

    # Data quality artifacts
    numeric=df.select_dtypes(include="number").columns.tolist()
    features=[c for c in df.columns if c!=target]
    quality=pd.DataFrame({
        "column":df.columns,
        "dtype":[str(df[c].dtype) for c in df.columns],
        "missing_count":[int(df[c].isna().sum()) for c in df.columns],
        "missing_percent":[float(df[c].isna().mean()*100) for c in df.columns],
        "unique_values":[int(df[c].nunique(dropna=True)) for c in df.columns],
        "constant_column":[bool(df[c].nunique(dropna=False)<=1) for c in df.columns],
    })
    quality.to_csv(EDA/"data_quality_summary.csv",index=False)
    df.describe().T.to_csv(EDA/"descriptive_statistics.csv")
    iqr_outlier_counts(df,[c for c in numeric if c!=target]).to_csv(EDA/"outlier_analysis.csv",index=False)

    # Remove exact duplicates only if present; none are removed silently.
    clean=df.drop_duplicates().copy()
    cleaning_log={
        "original_shape":list(df.shape),
        "duplicate_rows_found":int(df.duplicated().sum()),
        "duplicates_removed":int(df.duplicated().sum()),
        "shape_after_duplicate_handling":list(clean.shape),
        "missing_value_treatment":"Median imputation for numeric features and most-frequent imputation for categorical features inside the training pipeline. No rows removed for missingness.",
        "outlier_treatment":"Outliers were analyzed using IQR fences and retained because extreme measurements can be legitimate observations; no blanket row deletion was applied.",
        "impossible_value_checks":"pH values outside [0,14] and negative numeric measurements were checked. Findings are recorded below."
    }
    impossible={}
    for c in features:
        if pd.api.types.is_numeric_dtype(clean[c]):
            impossible[c+"_negative_count"]=int((clean[c]<0).sum())
    if "ph" in clean.columns:
        impossible["ph_outside_0_14"]=int(((clean["ph"].dropna()<0)|(clean["ph"].dropna()>14)).sum())
    cleaning_log["impossible_value_counts"]=impossible
    (OUT/"cleaning_log.json").write_text(json.dumps(cleaning_log,indent=2))

    X=clean.drop(columns=[target]); y=clean[target].astype(int)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)

    # EDA on complete labelled dataset
    eda_info=run_eda(clean,target,EDA,train_df=X_train)

    # PCA analysis: impute + scale numeric training features only
    num_cols=X_train.select_dtypes(include="number").columns.tolist()
    pca_pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("pca",PCA())])
    pca_pipe.fit(X_train[num_cols])
    pca=pca_pipe.named_steps["pca"]
    evr=pca.explained_variance_ratio_
    cum=np.cumsum(evr)
    n80=int(np.argmax(cum>=.80)+1)
    n90=int(np.argmax(cum>=.90)+1)
    pca_summary={"numeric_features":num_cols,"components":len(evr),"explained_variance_ratio":[float(x) for x in evr],
                 "cumulative_explained_variance":[float(x) for x in cum],"components_for_80_percent":n80,
                 "components_for_90_percent":n90}
    (OUT/"pca_summary.json").write_text(json.dumps(pca_summary,indent=2))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5)); plt.plot(range(1,len(evr)+1),evr,marker="o"); plt.xlabel("Principal Component"); plt.ylabel("Explained Variance Ratio"); plt.title("PCA Explained Variance by Component"); plt.tight_layout(); plt.savefig(EDA/"09_pca_explained_variance.png",dpi=190); plt.close()
    plt.figure(figsize=(8,5)); plt.plot(range(1,len(cum)+1),cum,marker="o"); plt.axhline(.80,linestyle="--",label="80%"); plt.axhline(.90,linestyle="--",label="90%"); plt.xlabel("Number of Components"); plt.ylabel("Cumulative Explained Variance"); plt.title("PCA Cumulative Explained Variance"); plt.legend(); plt.tight_layout(); plt.savefig(EDA/"10_pca_cumulative_variance.png",dpi=190); plt.close()
    coords=pca_pipe.transform(X_train[num_cols])[:,:2]
    pca_df=pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"Potability":y_train.to_numpy()})
    plt.figure(figsize=(8,6))
    for label in [0,1]:
        z=pca_df[pca_df.Potability==label]
        plt.scatter(z.PC1,z.PC2,s=16,alpha=.45,label=("Non-Potable" if label==0 else "Potable"))
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("2D PCA Projection of Training Samples"); plt.legend(); plt.tight_layout(); plt.savefig(EDA/"11_pca_2d.png",dpi=190); plt.close()

    pre=build_preprocessor(X_train)
    models={
        "Logistic Regression":LogisticRegression(max_iter=2000,random_state=RANDOM_STATE),
        "K-Nearest Neighbors":KNeighborsClassifier(n_neighbors=15),
        "Decision Tree":DecisionTreeClassifier(max_depth=8,min_samples_leaf=5,random_state=RANDOM_STATE),
        "Random Forest":RandomForestClassifier(n_estimators=400,max_depth=None,min_samples_leaf=2,n_jobs=-1,random_state=RANDOM_STATE,class_weight="balanced"),
        "Support Vector Machine":SVC(C=1.0,kernel="rbf",probability=True,random_state=RANDOM_STATE,class_weight="balanced"),
        "XGBoost":XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.05,subsample=.85,colsample_bytree=.85,
                                objective="binary:logistic",eval_metric="logloss",random_state=RANDOM_STATE,n_jobs=-1)
    }
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
    rows=[]; fitted={}
    print("\n=== 5-Fold Training Cross-Validation ===")
    for name,clf in models.items():
        pipe=Pipeline([("preprocessor",pre),("classifier",clf)])
        yp=cross_val_predict(pipe,X_train,y_train,cv=cv,method="predict",n_jobs=None)
        yprob=cross_val_predict(pipe,X_train,y_train,cv=cv,method="predict_proba",n_jobs=None)[:,1]
        metrics={
            "Model":name,
            "Accuracy":accuracy_score(y_train,yp),
            "Precision":precision_score(y_train,yp,zero_division=0),
            "Recall":recall_score(y_train,yp,zero_division=0),
            "F1":f1_score(y_train,yp,zero_division=0),
            "ROC_AUC":roc_auc_score(y_train,yprob),
            "NonPotable_Recall":non_potable_recall(y_train,yp),
        }
        metrics["Selection_Metric"]=metrics["F1"]
        rows.append(metrics)
        fitted[name]=pipe
        print(name, {k:round(v,4) for k,v in metrics.items() if k not in ["Model"]})
    comp=pd.DataFrame(rows).sort_values("Selection_Metric",ascending=False).reset_index(drop=True)
    comp.to_csv(OUT/"model_comparison.csv",index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6))
    x=np.arange(len(comp)); width=.18
    for i,col in enumerate(["Accuracy","Precision","Recall","F1","ROC_AUC"]):
        ax.bar(x+(i-2)*width,comp[col],width,label=col)
    ax.set_xticks(x,comp["Model"],rotation=25,ha="right"); ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_title("Cross-Validated Model Comparison")
    ax.legend(ncol=3); fig.tight_layout(); fig.savefig(EDA/"12_model_comparison.png",dpi=190); plt.close(fig)
    best_name=comp.iloc[0]["Model"]
    print("\nSelected model:",best_name)

    best_pipe=fitted[best_name]
    best_pipe.fit(X_train,y_train)
    joblib.dump(best_pipe,MODEL_DIR/"best_model.pkl")

    ytest=best_pipe.predict(X_test)
    yprob=best_pipe.predict_proba(X_test)[:,1] if hasattr(best_pipe,"predict_proba") else None
    test_metrics={
        "Accuracy":float(accuracy_score(y_test,ytest)),
        "Precision":float(precision_score(y_test,ytest,zero_division=0)),
        "Recall":float(recall_score(y_test,ytest,zero_division=0)),
        "F1":float(f1_score(y_test,ytest,zero_division=0)),
        "ROC_AUC":float(roc_auc_score(y_test,yprob)) if yprob is not None else None,
        "NonPotable_Recall":float(non_potable_recall(y_test,ytest)),
    }
    save_selected_evaluation(y_test,ytest,yprob,EVAL)
    report=classification_report(y_test,ytest,target_names=["Non-Potable","Potable"],output_dict=True,zero_division=0)
    pd.DataFrame(report).transpose().to_csv(EVAL/"classification_report.csv")

    # Feature ranges from training data for app inputs
    ranges={}
    medians={}
    for c in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[c]):
            s=X_train[c].dropna()
            ranges[c]={"min":float(s.min()),"max":float(s.max()),"median":float(s.median()),
                       "p05":float(s.quantile(.05)),"p95":float(s.quantile(.95))}
            medians[c]=float(s.median())
        else:
            vals=X_train[c].dropna().astype(str).value_counts().to_dict()
            ranges[c]={"categories":list(vals.keys()),"most_frequent":next(iter(vals),None)}
    pd.DataFrame([{"feature":c,**v} for c,v in ranges.items()]).to_csv(OUT/"feature_ranges.csv",index=False)

    # Held-out test predictions
    pred=pd.DataFrame({"Actual_Potability":y_test.to_numpy(),"Predicted_Potability":ytest})
    if yprob is not None: pred["Potable_Probability"]=yprob
    pred.to_csv(OUT/"predictions.csv",index=False)

    # Data-derived live-monitoring baseline: the observed sample with the highest
    # model probability of class 1 in the supplied labelled dataset. This is only
    # a demonstration starting point; it is not a recommended water profile.
    all_probs=best_pipe.predict_proba(X)[:,1]
    baseline_idx=int(np.argmax(all_probs))
    baseline_sample={c:(float(X.iloc[baseline_idx][c]) if pd.api.types.is_numeric_dtype(X[c]) else str(X.iloc[baseline_idx][c])) for c in X.columns}
    baseline_info={"source":"supplied dataset","row_index":baseline_idx,"actual_label":int(y.iloc[baseline_idx]),
                   "model_potable_probability":float(all_probs[baseline_idx]),"sample":baseline_sample}
    (OUT/"monitoring_baseline.json").write_text(json.dumps(baseline_info,indent=2))

    # model metadata
    pyver=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    metadata={
        "project_name":"AquaGuard — Real-Time Water Safety Monitoring & Classification System",
        "problem_type":"Binary Classification","task":"Predict whether a water sample is potable based on supplied water-quality measurements.",
        "target_column":target,"target_mapping":{"0":"Non-Potable","1":"Potable"},
        "selected_model":best_name,"selection_metric":"F1",
        "selection_rule":"Select the model with the highest 5-fold cross-validated F1-score on the training split; F1 balances precision and recall and avoids selecting a degenerate classifier that predicts only one class.",
        "validation_test_strategy":{"test_size":TEST_SIZE,"random_state":RANDOM_STATE,"stratified":True,"cv":"5-fold StratifiedKFold on training split","test_set_used_only_for_final_selected_model_evaluation":True},
        "input_features":features,"engineered_features":[],"feature_engineering_note":"No additional domain-derived features were added; the supplied measurements are retained to avoid unnecessary transformations and leakage.",
        "preprocessing_steps":["Numeric median imputation","Numeric StandardScaler","Categorical most-frequent imputation","Categorical OneHotEncoder(handle_unknown='ignore')","All preprocessing is stored inside the persisted pipeline."],
        "pca_analysis":pca_summary,
        "model_metrics":{"cross_validated_comparison":comp.to_dict(orient="records"),"held_out_test":test_metrics},
        "dataset":{"original_shape":list(df.shape),"cleaned_shape":list(clean.shape),"training_shape":[len(X_train),X_train.shape[1]],"test_shape":[len(X_test),X_test.shape[1]],"missing_values":summary["missing_values"],"duplicates":summary["duplicate_rows"]},
        "training_timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version":pyver,"package_versions":{"pandas":pd.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"joblib":joblib.__version__},
        "xgboost_version":__import__("xgboost").__version__,
        "feature_ranges":ranges,
        "training_medians":medians,
        "runtime_seconds":round(time.time()-start,2)
    }
    (OUT/"model_config.json").write_text(json.dumps(metadata,indent=2))
    print("\n=== Held-out Test Metrics ===")
    print({k:round(v,4) if isinstance(v,float) else v for k,v in test_metrics.items()})
    print(f"Runtime: {time.time()-start:.1f}s")

if __name__=="__main__":
    main()
