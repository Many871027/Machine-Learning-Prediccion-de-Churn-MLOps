from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_model_configs(ratio_neg_pos=1.0):
    """
    Returns the estimator and a highly optimized hyperparameter grid 
    to iterate quickly while preventing class imbalance via SMOTE.
    """
    return {
        'DecisionTree': {
            'estimator': ImbPipeline([('resampler', None), ('classifier', DecisionTreeClassifier(random_state=42))]),
            'param_grid': {
                'resampler': [SMOTE(random_state=42)],
                'classifier__max_depth': [5],
                'classifier__class_weight': ['balanced']
            }
        },
        'RandomForest': {
            'estimator': ImbPipeline([('resampler', None), ('classifier', RandomForestClassifier(random_state=42))]),
            'param_grid': {
                'resampler': [SMOTE(random_state=42)],
                'classifier__n_estimators': [100],
                'classifier__max_depth': [5],
                'classifier__class_weight': ['balanced']
            }
        },
        'XGBoost': {
            'estimator': ImbPipeline([('resampler', None), ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))]),
            'param_grid': {
                'resampler': [SMOTE(random_state=42)],
                'classifier__max_depth': [4],
                'classifier__scale_pos_weight': [ratio_neg_pos]
            }
        },
        'KNN': {
            'estimator': ImbPipeline([('resampler', None), ('classifier', KNeighborsClassifier())]),
            'param_grid': {
                'resampler': [SMOTE(random_state=42)],
                'classifier__n_neighbors': [5]
            }
        },
        'SVM': {
            'estimator': ImbPipeline([('resampler', None), ('classifier', SVC(probability=True, random_state=42))]),
            'param_grid': {
                'resampler': [SMOTE(random_state=42)],
                'classifier__class_weight': ['balanced']
            }
        },
        'NaiveBayes': {
            'estimator': ImbPipeline([('resampler', None), ('classifier', GaussianNB())]),
            'param_grid': {
                'resampler': [SMOTE(random_state=42)]
            }
        }
    }
