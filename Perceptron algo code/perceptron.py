import numpy as np
import pandas as pd
import random
#train
df_train=pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\CA1data\train.data",header=None)
df_train = df_train.rename(columns={0: 'X1', 1: 'X2', 2: 'X3' , 3: 'X4', 4: 'Yact'})

#train for 12
x_train12=[]
y_train12=[]
df_train12=df_train.loc[df_train["Yact"].isin(["class-1","class-2"])]
df_train12.loc[:, 'Yact'].replace({'class-1': 1, 'class-2': -1}, inplace=True)
x_train12=df_train12[["X1","X2","X3","X4"]]
y_train12=df_train12["Yact"].tolist()
x_train12 = np.array(x_train12, dtype = float)
y_train12 = np.array(y_train12)

#train for 23
x_train23=[]
y_train23=[]
df_train23=df_train.loc[df_train["Yact"].isin(["class-2","class-3"])]
df_train23.loc[:, 'Yact'].replace({'class-2': 1, 'class-3': -1}, inplace=True)
x_train23=df_train23[["X1","X2","X3","X4"]]
y_train23=df_train23["Yact"].tolist()
x_train23 = np.array(x_train23, dtype = float)
y_train23 = np.array(y_train23)

#train for 13
x_train13=[]
y_train13=[]
df_train13=df_train.loc[df_train["Yact"].isin(["class-1","class-3"])]
df_train13.loc[:, 'Yact'].replace({'class-1': 1, 'class-3': -1}, inplace=True)
x_train13=df_train13[["X1","X2","X3","X4"]]
y_train13=df_train13["Yact"].tolist()
x_train13 = np.array(x_train13, dtype = float)
y_train13 = np.array(y_train13)

# 1 vs rest approach train
x_trainrest1=[]
y_trainrest1=[]
df_trainrest1=df_train.loc[df_train["Yact"].isin(["class-1","class-2","class-3"])]
df_trainrest1.loc[:, 'Yact'].replace({'class-1': 1, 'class-3': -1, 'class-2': -1}, inplace=True)
x_trainrest1=df_trainrest1[["X1","X2","X3","X4"]]
y_trainrest1=df_trainrest1["Yact"].tolist()
x_trainrest1=np.array(x_trainrest1,dtype=float)
y_trainrest1=np.array(y_trainrest1)

# 2 vs rest approach train
x_trainrest2=[]
y_trainrest2=[]
df_trainrest2=df_train.loc[df_train["Yact"].isin(["class-1","class-2","class-3"])]
df_trainrest2.loc[:, 'Yact'].replace({'class-1': -1, 'class-3': -1, 'class-2': 1}, inplace=True)
x_trainrest2=df_trainrest2[["X1","X2","X3","X4"]]
y_trainrest2=df_trainrest2["Yact"].tolist()
x_trainrest2=np.array(x_trainrest2,dtype=float)
y_trainrest2=np.array(y_trainrest2)

# 3 vs rest approach train
x_trainrest3=[]
y_trainrest3=[]
df_trainrest3=df_train.loc[df_train["Yact"].isin(["class-1","class-2","class-3"])]
df_trainrest3.loc[:, 'Yact'].replace({'class-1': -1, 'class-3': 1, 'class-2': -1}, inplace=True)
x_trainrest3=df_trainrest3[["X1","X2","X3","X4"]]
y_trainrest3=df_trainrest3["Yact"].tolist()
x_trainrest3=np.array(x_trainrest3,dtype=float)
y_trainrest3=np.array(y_trainrest3)


#test
df_test=pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\CA1data\test.data",header=None)
df_test = df_test.rename(columns={0: 'X1', 1: 'X2', 2: 'X3' , 3: 'X4', 4: 'Yact'})

#test12
x_test12=[]
y_test12=[]
df_test12=df_test.loc[df_test["Yact"].isin(["class-1","class-2"])]
x_test12=df_test12[["X1","X2","X3","X4"]]
df_test12.loc[:, 'Yact'].replace({'class-1': 1, 'class-2': -1}, inplace=True)
y_test12=df_test12["Yact"].tolist()
x_test12 = np.array(x_test12, dtype = float)
y_test12 = np.array(y_test12)

#test23
x_test23=[]
y_test23=[]
df_test23=df_test.loc[df_test["Yact"].isin(["class-2","class-3"])]
x_test23=df_test23[["X1","X2","X3","X4"]]
df_test23.loc[:, 'Yact'].replace({'class-2': 1, 'class-3': -1}, inplace=True)
y_test23=df_test23["Yact"].tolist()
x_test23 = np.array(x_test23, dtype = float)
y_test23 = np.array(y_test23)

#test13
x_test13=[]
y_test13=[]
df_test13=df_test.loc[df_test["Yact"].isin(["class-1","class-3"])]
x_test13=df_test13[["X1","X2","X3","X4"]]
df_test13.loc[:, 'Yact'].replace({'class-1': 1, 'class-3': -1}, inplace=True)
y_test13=df_test13["Yact"].tolist()
x_test13 = np.array(x_test13, dtype = float)
y_test13 = np.array(y_test13)


# 1 vs rest approach test
x_testrest1=[]
y_testrest1=[]
df_testrest1=df_test.loc[df_test["Yact"].isin(["class-1","class-2","class-3"])]
df_testrest1.loc[:, 'Yact'].replace({'class-1': 1, 'class-3': -1, 'class-2': -1}, inplace=True)
x_testrest1=df_testrest1[["X1","X2","X3","X4"]]
y_testrest1=df_testrest1["Yact"].tolist()
x_testrest1=np.array(x_testrest1,dtype=float)
y_testrest1=np.array(y_testrest1)

# 2 vs rest approach test
x_testrest2=[]
y_testrest2=[]
df_testrest2=df_test.loc[df_test["Yact"].isin(["class-1","class-2","class-3"])]
df_testrest2.loc[:, 'Yact'].replace({'class-1': -1, 'class-3': -1, 'class-2': 1}, inplace=True)
x_testrest2=df_testrest2[["X1","X2","X3","X4"]]
y_testrest2=df_testrest2["Yact"].tolist()
x_testrest2=np.array(x_testrest2,dtype=float)
y_testrest2=np.array(y_testrest2)

# 3 vs rest approach test
x_testrest3=[]
y_testrest3=[]
df_testrest3=df_test.loc[df_test["Yact"].isin(["class-1","class-2","class-3"])]
df_testrest3.loc[:, 'Yact'].replace({'class-1': -1, 'class-3': 1, 'class-2': -1}, inplace=True)
x_testrest3=df_testrest3[["X1","X2","X3","X4"]]
y_testrest3=df_testrest3["Yact"].tolist()
x_testrest3=np.array(x_testrest3,dtype=float)
y_testrest3=np.array(y_testrest3)

# Q2
def perceptronTrain(X,Y,maxiter=20):
    w = np.zeros(X.shape[1]).transpose()
    b = 0
    for Iter in range(maxiter):
        for i, x in enumerate(X):
            a = np.dot(w, x) + b
            if Y[i] * a <= 0:
                    w += Y[i] * x
                    b = b + Y[i]
    return w, b


def perceptronTest(X,w,b):
    a = np.dot(X, w) + b
    return np.sign(a)

def perceptron_train_l2(X, Y,l, maxiters=20):
    w = np.zeros(X.shape[1])
    b = 0
    for Iter in range(maxiters):
        for i, x in enumerate(X):
            a = np.dot(w, x) + b
            if Y[i] * a <= 0:
              w = (1-(2*l))*w + Y[i] * x
              b = b + Y[i]
            else:
              w = random.uniform(1-(2*l),1-(2*l))*w
              b = b
    return w, b

def accuracy(x_test,y_test,w,b):
     summation=sum(y_test == perceptronTest(x_test,w,b))
     acc=summation*100/len(y_test)
     print("The Accuracy is",acc)

# Q3
print(" The accuracy for class-1 and class-2")
w,b=perceptronTrain(x_train12,y_train12)
accuracy(x_test12,y_test12,w,b)

print(" The accuracy for class-2 and class-3")
w,b=perceptronTrain(x_train23,y_train23)
accuracy(x_test23,y_test23,w,b)

print(" The accuracy for class-1 and class-3")
w,b=perceptronTrain(x_train13,y_train13)
accuracy(x_test13,y_test13,w,b)

# Q4

print("The accuracy for class-1 vs rest")
w,b=perceptronTrain(x_trainrest1,y_trainrest1)
accuracy(x_testrest1,y_testrest1,w,b)

print("The accuracy for class-2 vs rest")
w,b=perceptronTrain(x_trainrest2,y_trainrest2)
accuracy(x_testrest2,y_testrest2,w,b)

print("The accuracy for class-3 vs rest")
w,b=perceptronTrain(x_trainrest3,y_trainrest3)
accuracy(x_testrest3,y_testrest3,w,b)


# Q5
# 1 vs rest using l2
L = [ 0.01, 0.1, 1.0, 10.0, 100.0]
print("Below are the accuracies for class1 vs rest using l2 regularization")
for l in L:
    w,b=perceptron_train_l2(x_trainrest1,y_trainrest1,l)
    accuracy(x_testrest1,y_testrest1,w,b)

print("below are the accuracies for class2 vs rest using l2 regularization")
for l in L:
    w,b=perceptron_train_l2(x_trainrest2,y_trainrest2,l)
    accuracy(x_testrest2,y_testrest2,w,b)

print("below are the accuracies for class3 vs rest using l2 regularization")
for l in L:
    w,b=perceptron_train_l2(x_trainrest3,y_trainrest3,l)
    accuracy(x_testrest3,y_testrest3,w,b)

