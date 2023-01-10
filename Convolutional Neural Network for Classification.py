#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

#%% IMPORTING THE DATA
pos_path = os.path.join("..", "dataset", "cracks", "Positive")
neg_path = os.path.join("..", "dataset", "cracks", "Negative")
pix_dim = 32
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

X_tv, X_test, y_tv, y_test = train_test_split(all_imgs, all_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.2, shuffle=False)


# %% Scaling, converting to Pytorch tensor
X_train_cnn = X_train/255.
X_val_cnn = X_val/255.
X_test_cnn = X_test/255.

X_trainf = torch.from_numpy(X_train_cnn).float()
X_valf = torch.from_numpy(X_val_cnn).float()
X_testf = torch.from_numpy(X_test_cnn).float()
y_train = torch.from_numpy(y_train).long()
y_val = torch.from_numpy(y_val).long()
y_test = torch.from_numpy(y_test).long()

X_train = X_trainf.view(-1,1,pix_dim,pix_dim)
X_val = X_valf.view(-1,1,pix_dim,pix_dim)
X_test = X_testf.view(-1,1,pix_dim,pix_dim)
plt.imshow(X_train[0][0], cmap="Greys", interpolation='bicubic')
print(y_train[0])

#%% Defining training parameters
bs = 64
train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
val_ds = TensorDataset(X_val, y_val)
val_dl = DataLoader(val_ds, batch_size=len(val_ds))
test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size=len(test_ds))
learning_rate = 1e-3
epochs=25

# %% Defining network architecture
torch.manual_seed(52)
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,5)
        self.pool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(10,20,5)
        self.fc1 = torch.nn.Linear(500,250)
        self.fc2 = torch.nn.Linear(250,2)
    def forward(self,x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(-1,500)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = torch.nn.CrossEntropyLoss()

#%% Getting dense layer input size
# conv1 = torch.nn.Conv2d(1,10,5)
# pool = torch.nn.MaxPool2d(2)
# conv2 = torch.nn.Conv2d(10,20,5)

# print(X_train[0:1].shape)
# print(conv1(X_train[0:1]).shape)
# print(pool(conv1(X_train[0:1])).shape)
# print(F.relu(pool(conv1(X_train[0:1]))).shape)
# print(conv2(F.relu(pool(conv1(X_train[0:1])))).shape)
# print(pool(conv2(F.relu(pool(conv1(X_train[0:1]))))).shape)

# img_out = F.relu(pool(conv2(F.relu(pool(conv1(X_train[0:1]))))))

# print(img_out.shape)
# print(np.prod(img_out.shape))

# %% Training loop with early stopping
breaker = False
size = len(train_dl.dataset)
for epoch in range(epochs):
    for batch, (X,y) in enumerate(train_dl):
        pred = model(X)
        loss = loss_fn(pred, y)

        y_pred = torch.argmax(pred.data,1)
        correct = (y==y_pred).sum().item()/len(y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (epoch%1 == 0) and (batch%100 == 0):
        #     print(f"Epoch {epoch}, Batch {batch}: Training Loss = {loss.item()} Accuracy = {correct}")
        
    with torch.no_grad():
        for Xv, yv, in val_dl:
            pred_val = model(Xv)
            # loss_value = loss_fn(pred, y).item()
            y_pred_val = torch.argmax(pred_val.data,1)
            correct_val = (yv==y_pred_val).sum().item()/len(y_pred_val)
            print(f"Val accuracy = {correct_val}")
            if correct_val>0.993:
                breaker = True
                break
    if breaker:
        break
# %% Training accuracy
with torch.no_grad():
    pred_train = model(X_train)
    # loss_value = loss_fn(pred, y).item()
    y_pred_train = torch.argmax(pred_train.data,1)
    correct_train = (y_train==y_pred_train).sum().item()/len(y_pred_train)
    print(f"Val accuracy = {correct_train}")

#%% Test accuracy
with torch.no_grad():
    pred_test = model(X_test)
    # loss_value = loss_fn(pred, y).item()
    y_pred_test = torch.argmax(pred_test.data,1)
    correct_test = (y_test==y_pred_test).sum().item()/len(y_pred_test)
    print(f"Val accuracy = {correct_test}")