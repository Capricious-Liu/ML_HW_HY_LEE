import pandas as pd
import numpy as np
from math import log,floor

def load_data(train_data_path, train_label_path):
    X_train = pd.read_csv(train_data_path,sep=',',header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path,sep=',', header=0)
    Y_train = np.array(Y_train.values)

    return (X_train,Y_train)

def normalize(X_all):
    mu = (sum(X_all)/X_all.shape[0])
    sigma = np.std(X_all,axis=0)
    mu = np.tile(mu,(X_all.shape[0],1))
    sigma = np.tile(sigma,(X_all.shape[0],1))
    X_all_normed = (X_all-mu)/sigma

    return X_all_normed

def _shuffle(X,Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def sigmoid(z):
    res = 1/(1.0+np.exp(-z))
    return np.clip(res,1e-8,1-(1e-8))

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size*(percentage)))

    X_all,Y_all = _shuffle(X_all,Y_all)

    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def valid(w,b,X_valid,Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid,np.transpose(w))+b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()/valid_data_size)))


def train(X_all, Y_all):
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all,Y_all,valid_set_percentage)

    w = np.zeros((106,))
    b = np.zeros((1,))
    l_rate = 0.001
    batch_size = 32
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size/batch_size))
    epoch_num = 1000
    save_param_iter = 50

    total_loss = 0.0
    for epoch in range(1,epoch_num):
        if (epoch) % save_param_iter == 0:
            print('epoch avg loss = %f' % (total_loss/(float(save_param_iter)*train_data_size)))
            total_loss = 0.0
            valid(w,b,X_valid,Y_valid)

        X_train, Y_train = _shuffle(X_train,Y_train)

        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w))+b
            y = sigmoid(z)

            cross_entropy = -1*(np.dot(np.squeeze(Y),np.log(y))+np.dot((1-np.squeeze(Y)),np.log(1-y)))
            total_loss += cross_entropy

            # temp = np.squeeze(Y)-y
            # temp2 = temp.reshape((batch_size,1))
            # temp3 = X*temp2

            w_grad = np.mean(-1*X*(np.squeeze(Y)-y).reshape((batch_size,1)),axis=0)
            b_grad = np.mean(-1*(np.squeeze(Y)-y))

            w = w - l_rate*w_grad
            b = b - l_rate*b_grad
    return

if __name__=='__main__':
    train_data_path = r'./data/X_train.csv'
    train_label_path = r'./data/Y_train.csv'
    X_all, Y_all = load_data(train_data_path,train_label_path)
    X_all = normalize(X_all)

    train(X_all,Y_all)