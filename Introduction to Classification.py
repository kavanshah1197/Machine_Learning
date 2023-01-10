#%% Import modules
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_recall_curve

# %% Fetch dataset
walldata = fetch_openml(name='wall-robot-navigation', version=3)
pd.DataFrame(walldata.data, columns=["V1","V2","V3","V4"]).describe()

#%% Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(walldata.data.astype(np.float32), walldata.target.astype(np.float32), test_size=0.2, random_state=42)
X_train_df = pd.DataFrame(X_test, columns=["V1","V2","V3","V4"])
X_train_df.describe()
pd.Series(y_test).value_counts()
y_train_right = (y_train == 2) | (y_train == 3)
y_test_right = (y_test == 2) | (y_test == 3)

# %% Nearest Centroid Classifier
centroid_clf = NearestCentroid()
centroid_clf.fit(X=X_train, y=y_train)
centroid_clf.score(X=X_train, y=y_train)
centroid_clf_confusion_matrix = confusion_matrix(y_train,centroid_clf.predict(X_train))
plot_confusion_matrix(centroid_clf, X_train, y_train)

# %% Nearest Centroid Classifier Pipeline with feature scaling
centroid_clf_scaled = Pipeline([('scaler', StandardScaler()),('centroid', NearestCentroid())])
centroid_clf_scaled.fit(X_train, y_train)
centroid_clf_scaled.score(X_train, y_train)
centroid_clf_scaled_confusion_matrix = confusion_matrix(y_train,centroid_clf_scaled.predict(X_train))
plot_confusion_matrix(centroid_clf_scaled, X_train, y_train)
confusion_matrix_improvement = np.zeros((4,4))
confusion_matrix_non_zero = centroid_clf_confusion_matrix!=0

#%% Nearest Centroid Classifier Pipeline with Poly features
centroid_clf_scaled_poly = Pipeline([('poly', PolynomialFeatures()), ('scaler', StandardScaler()),('centroid', NearestCentroid())])
centroid_clf_scaled_poly.fit(X_train, y_train)
centroid_clf_scaled_poly.score(X_train, y_train)

# %% Nearest Centroid Classifier Pipeline Grid Search
parameters = [{'poly__degree':[1,2,3,5,7,10]}]
centroid_clf_scaled_poly = Pipeline([('poly', PolynomialFeatures()),('scaler', StandardScaler()),('centroid', NearestCentroid())])
grid_centroid_clf = GridSearchCV(centroid_clf_scaled_poly,param_grid=parameters,return_train_score=True,scoring='accuracy')
grid_centroid_clf.fit(X_train, y_train)
grid_centroid_clf.best_params_
grid_centroid_clf.best_estimator_.score(X_train, y_train)
plot_confusion_matrix(grid_centroid_clf.best_estimator_, X_train, y_train)
grid_centroid_clf.best_estimator_.score(X_test, y_test)

# %% Support Vector Classifier - Precision recall and ROC curves
from sklearn.svm import SVC
svc_clf = SVC(kernel='linear')
svc_clf.fit(X_train, y_train_right)
svc_clf.score(X_train, y_train_right)
plot_precision_recall_curve(svc_clf,X_train,y_train_right)
decision_scores = svc_clf.decision_function(X_train)
pr, re, thr = precision_recall_curve(y_train_right,probas_pred=decision_scores)
np.interp(0.9,pr,re)
plot_roc_curve(svc_clf,X_train,y_train_right)

# %% Support Vector Classifier Pipeline Grid Search
svc_pipe = Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='rbf'))])
parameters = {'svc__gamma':[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000],'svc__C':[0.1,1.0,10.0]}
svc_grid_clf = GridSearchCV(svc_pipe,param_grid=parameters,scoring='accuracy',return_train_score=True)
svc_grid_clf.fit(X_train,y_train_right)
decision_scores = svc_grid_clf.decision_function(X_train)
pr, re, thr = precision_recall_curve(y_train_right,probas_pred=decision_scores)
plot_precision_recall_curve(svc_grid_clf,X_train,y_train_right)
plot_roc_curve(svc_grid_clf,X_train,y_train_right)
results = svc_grid_clf.cv_results_
for mean_train, mean_score, params in zip(results["mean_train_score"], results["mean_test_score"],results["params"]):
    print(mean_train-mean_score, params)
svc_grid_clf.score(X_test,y_test_right)
svc_grid_clf.best_estimator_.fit(X_test,y_test)
svc_grid_clf.best_estimator_.score(X_test,y_test)
