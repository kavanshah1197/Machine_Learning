#%%
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

#%% IMPORTING THE DATA
pos_path = os.path.join("..", "dataset", "cracks", "Positive")
neg_path = os.path.join("..", "dataset", "cracks", "Negative")
pix_dim = 16
all_imgs = np.zeros((40000,pix_dim*pix_dim), dtype=np.uint8)
all_labels = np.hstack((np.zeros(20000),np.ones(20000))).astype(np.uint8)
# %%
for i in range(20000):
    im = Image.open(os.path.join(neg_path, f"0000{i+1}.jpg"[-9:])).convert("L")
    im = im.resize((pix_dim,pix_dim))
    all_imgs[i,:] = np.array(im).reshape(1,-1)
    if (i+1)%1000 == 0:
        print(f"Done importing {i+1} images")
for i in range(20000):
    try:
        im = Image.open(os.path.join(pos_path, f"0000{i+1}.jpg"[-9:])).convert("L")
    except FileNotFoundError:
        im = Image.open(os.path.join(pos_path, f"0000{i+1}_1.jpg"[-11:])).convert("L")
    im = im.resize((pix_dim,pix_dim))
    all_imgs[i+20000,:] = np.array(im).reshape(1,-1)
    if (i+1)%1000 == 0:
        print(f"Done importing {i+1} images") 

X_train, X_test, y_train, y_test = train_test_split(all_imgs, all_labels, test_size=0.2, random_state=42)
X_train_s = X_train[:5000]
y_train_s = y_train[:5000]

# %% Nearest Centroid Classifier
centroid_clf = NearestCentroid()
centroid_clf.fit(X_train, y_train)
print(f"Vanilla nearest centroid classifier training score: {centroid_clf.score(X_train, y_train)}")

centroid_pipe = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("clf", NearestCentroid())])

centroid_param_grid = [{"poly__degree":[1,2]}]

centroid_gs = GridSearchCV(centroid_pipe, param_grid=centroid_param_grid, scoring='accuracy', cv=5, return_train_score=True)
centroid_gs.fit(X_train, y_train)
centroid_best = centroid_gs.best_estimator_
print(centroid_gs.best_score_)
print(centroid_best.score(X_train, y_train))
print(centroid_best.score(X_test, y_test))


# %% Quadratic Discriminant Analysis
qda_clf = QuadraticDiscriminantAnalysis(reg_param=0.0, store_covariance=True)
qda_clf.fit(X_train, y_train)
print(qda_clf.score(X_train, y_train))
#%%
qda_pipe = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("clf", QuadraticDiscriminantAnalysis())
])
qda_param_grid = [{"poly__degree":[1], "clf__reg_param":[0.0,0.1,0.2]}]
qda_gs = GridSearchCV(qda_pipe, param_grid=qda_param_grid, cv=5, scoring='accuracy', return_train_score=True)
qda_gs.fit(X_train, y_train)
qda_best = qda_gs.best_estimator_
print(qda_gs.best_score_)
print(qda_best.score(X_train, y_train))
print(qda_best.score(X_test, y_test))
qda_cv_results = qda_gs.cv_results_
for mean_train, mean_score, params in zip(qda_cv_results["mean_train_score"], qda_cv_results["mean_test_score"], qda_cv_results["params"]):
    print(mean_train, mean_score, params)

# %% Linear Discriminant Analysis
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)
print(lda_clf.score(X_train, y_train))
lda_pipe = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("clf", LinearDiscriminantAnalysis())
])
lda_param_grid = [{"poly__degree":[1], "clf__shrinkage":[0.0,0.1,0.2]}]
lda_gs = GridSearchCV(lda_pipe, param_grid=lda_param_grid, cv=5, scoring='accuracy', return_train_score=True)
lda_gs.fit(X_train, y_train)
lda_best = lda_gs.best_estimator_
print(lda_gs.best_score_)
print(lda_best.score(X_train, y_train))
print(lda_best.score(X_test, y_test))
lda_cv_results = lda_gs.cv_results_
for mean_train, mean_score, params in zip(lda_cv_results["mean_train_score"], lda_cv_results["mean_test_score"], lda_cv_results["params"]):
    print(mean_train, mean_score, params)

#%% K Nearest Neighbours
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_train, y_train))
knn_pipe = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])
knn_param_grid = [{"poly__degree":[1], "clf__n_neighbors":[1,5,10]}]
knn_gs = GridSearchCV(knn_pipe, param_grid=knn_param_grid, cv=5, scoring='accuracy', return_train_score=True)
knn_gs.fit(X_train, y_train)
knn_best = knn_gs.best_estimator_
print(knn_gs.best_score_)
print(knn_best.score(X_train, y_train))
print(knn_best.score(X_test, y_test))
knn_cv_results = knn_gs.cv_results_
for mean_train, mean_score, params in zip(knn_cv_results["mean_train_score"], knn_cv_results["mean_test_score"], knn_cv_results["params"]):
    print(mean_train, mean_score, params)

#%% Logistic Regression
log_clf = LogisticRegression(solver='liblinear')
log_clf.fit(X_train_s, y_train_s)
print(log_clf.score(X_train_s, y_train_s))
#%%
log_pipe = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver='liblinear'))
])
log_param_grid = [{"poly__degree":[2], "clf__C":[0.1,0.5,1.0]}]
log_gs = GridSearchCV(log_pipe, param_grid=log_param_grid, cv=5, scoring='accuracy', return_train_score=True)
log_gs.fit(X_train_s, y_train_s)
log_best = log_gs.best_estimator_
print(log_gs.best_score_)
print(log_best.score(X_train_s, y_train_s))
# print(log_best.score(X_test, y_test))
log_cv_results = log_gs.cv_results_
for mean_train, mean_score, params in zip(log_cv_results["mean_train_score"], log_cv_results["mean_test_score"], log_cv_results["params"]):
    print(mean_train, mean_score, params)

#%% Support Vector Classifier
svm_clf = SVC(kernel="sigmoid")
svm_clf.fit(X_train_s, y_train_s)
print(svm_clf.score(X_train_s, y_train_s))
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf"))
])
svm_param_grid = [{"clf__gamma":[0.16,0.162]}]
svm_gs = GridSearchCV(svm_pipe, param_grid=svm_param_grid, cv=5, scoring='accuracy', return_train_score=True)
svm_gs.fit(X_train, y_train)
svm_best = svm_gs.best_estimator_
print(svm_gs.best_score_)
print(svm_best.score(X_train, y_train))
# print(svm_best.score(X_test, y_test))
svm_cv_results = svm_gs.cv_results_
for mean_train, mean_score, params in zip(svm_cv_results["mean_train_score"], svm_cv_results["mean_test_score"], svm_cv_results["params"]):
    print(mean_train, mean_score, params)

#%% Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train_s, y_train_s)
print(dt_clf.score(X_train_s, y_train_s))
dt_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(criterion="entropy", splitter="best"))
])
dt_param_grid = [{"clf__ccp_alpha":[0.0,0.01,0.02]}]
dt_gs = GridSearchCV(dt_pipe, param_grid=dt_param_grid, cv=5, scoring='accuracy', return_train_score=True)
dt_gs.fit(X_train_s, y_train_s)
dt_best = dt_gs.best_estimator_
print(dt_gs.best_score_)
print(dt_best.score(X_train_s, y_train_s))
# print(svm_best.score(X_test, y_test))
dt_cv_results = dt_gs.cv_results_
for mean_train, mean_score, params in zip(dt_cv_results["mean_train_score"], dt_cv_results["mean_test_score"], dt_cv_results["params"]):
    print(mean_train, mean_score, params)

#%% Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_s, y_train_s)
print(rf_clf.score(X_train_s, y_train_s))
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier())
])
rf_param_dist = [{"clf__n_estimators":[50,75,100],
    "clf__criterion":["gini","entropy"],
    "clf__max_depth":[10,20,30],
    "clf__max_features":["sqrt", "log2"],
    "clf__":[]}]
rf_rs = RandomizedSearchCV(rf_pipe, param_distributions=rf_param_dist, cv=5, scoring='accuracy', return_train_score=True, n_iter=20, n_jobs=-1)
rf_rs.fit(X_train_s, y_train_s)
rf_best = rf_rs.best_estimator_
print(rf_rs.best_score_)
print(rf_best.score(X_train_s, y_train_s))
# print(svm_best.score(X_test, y_test))
rf_cv_results = rf_rs.cv_results_
for mean_train, mean_score, params in zip(rf_cv_results["mean_train_score"], rf_cv_results["mean_test_score"], rf_cv_results["params"]):
    print(mean_train, mean_score, params)
