#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
np.random.seed(42)
# %%
mean_a = np.array([0.,1.])
cov_a = np.array([[0.1,0.],[0.,0.1]])
ax, ay = np.random.multivariate_normal(mean_a, cov_a, 10).T
# %%
mean_b = np.array([1.,0.])
cov_b = np.array([[0.1,0.],[0.,0.1]])
bx, by = np.random.multivariate_normal(mean_b, cov_b, 10).T
# %%
plt.plot(ax, ay, 'x')
plt.plot(bx, by, 'o')
plt.axis('equal')
fig = plt.gcf()
# %%
def save_fig(fig, figname, tight_layout=True):
    PROJ_ROOT_DIR = '.'
    path = os.path.join(PROJ_ROOT_DIR, "images", figname + ".png")
    print("Saving figure " + figname)
    if tight_layout:
        plt.tight_layout()
    fig.savefig(path, format='png', dpi=300)
# %%
save_fig(fig,'Synth')
# %%
a = np.hstack((ax.reshape(-1,1), ay.reshape(-1,1)))
b = np.hstack((bx.reshape(-1,1), by.reshape(-1,1)))
X = np.vstack((a,b))
# %%
y = np.hstack((np.zeros(10), np.ones(10)))
# %%
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
clf.fit(X,y)
# %%
print(clf.score(X,y))
print(clf.centroids_)
# %%
from sklearn.linear_model import Perceptron
clf = Perceptron()
clf.fit(X,y)
# %%
print(clf.score(X,y))
print(clf.coef_)
# %%
h = 0.02
x_min, x_max = np.min(X[:,0])-1, np.max(X[:,0])+1
y_min, y_max = np.min(X[:,1])-1, np.max(X[:,1])+1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
# %%
Z = clf.predict(np.c_[xx.reshape(-1,1), yy.reshape(-1,1)])
# %%
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['cyan', 'orange'])
# %%
Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light,shading='auto')

plt.plot(ax, ay, 'x')
plt.plot(bx, by, 'o')

plt.axis('equal')
# %%
X = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
y = np.array([0,1,1,0])
# %%
plt.plot(X[~y.astype(bool),0],X[~y.astype(bool),1], 'x')
plt.plot(X[y.astype(bool),0],X[y.astype(bool),1], 'o')
# %%
clf = Perceptron()
clf.fit(X,y)
# %%
print(clf.score(X,y))
print(clf.coef_)
# %%
h = 0.02
x_min, x_max = np.min(X[:,0])-1, np.max(X[:,0])+1
y_min, y_max = np.min(X[:,1])-1, np.max(X[:,1])+1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))

Z = clf.predict(np.c_[xx.reshape(-1,1), yy.reshape(-1,1)])
Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light,shading='auto')
plt.plot(X[~y.astype(bool),0],X[~y.astype(bool),1], 'x')
plt.plot(X[y.astype(bool),0],X[y.astype(bool),1], 'o')
plt.axis('equal')
# %%
clf = NearestCentroid()
clf.fit(X,y)
# %%
print(clf.score(X,y))
# %%
h = 0.02
x_min, x_max = np.min(X[:,0])-1, np.max(X[:,0])+1
y_min, y_max = np.min(X[:,1])-1, np.max(X[:,1])+1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))

Z = clf.predict(np.c_[xx.reshape(-1,1), yy.reshape(-1,1)])
Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light,shading='auto')
plt.plot(X[~y.astype(bool),0],X[~y.astype(bool),1], 'x')
plt.plot(X[y.astype(bool),0],X[y.astype(bool),1], 'o')
plt.axis('equal')
# %%
X = np.c_[X, np.array([[0.],[1.],[1.],[0.]])]
print(X)
# %%
clf = Perceptron()
clf.fit(X,y)
print(clf.score(X,y))
print(clf.coef_)
print(clf.intercept_)
# %%
h = 0.02
xx, yy = np.meshgrid(np.arange(0,1,h),np.arange(0,1,h))
w = clf.coef_[0,:]
alpha=clf.intercept_
z = (-alpha - w[0]*xx - w[1]*yy)/(w[2])
ax = plt.axes(projection='3d')
ax.plot_surface(xx,yy,z)

x0=np.array([0.,1.])
y0=np.array([0.,1.])
z0=np.array([0.,0.])
x1=np.array([1.,0.])
y1=np.array([0.,1.])
z1=np.array([1.,1.])
ax.plot3D(x0,y0,z0,'x')
ax.plot3D(x1,y1,z1,'o')
# %%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)
# %%
X, y = mnist["data"].to_numpy(), mnist["target"].to_numpy()
# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# %%
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index,:], y_train[shuffle_index]
# %%
y_train_5 = (y_train==5).astype(int)
y_test_5 = (y_test==5).astype(int)
# %%
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(X_train, y_train_5)
# %%
sgd_clf.predict(X[35014,:].reshape(1,-1))
# %%
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
# %%
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
# %%
from sklearn.linear_model import Perceptron
per_clf = Perceptron()
per_clf_train_pred = cross_val_predict(per_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, per_clf_train_pred)
# %%
