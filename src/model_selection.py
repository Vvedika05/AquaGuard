from sklearn.metrics import make_scorer, f1_score

def selection_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)
SELECTION_SCORER = make_scorer(selection_f1)
SELECTION_METRIC = "F1"


def non_potable_recall(y_true,y_pred):
    from sklearn.metrics import recall_score
    return recall_score(y_true,y_pred,pos_label=0,zero_division=0)
