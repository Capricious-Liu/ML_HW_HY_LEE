from HW2.1_Logistic_Regression import split_valid_set,sigmoid,load_data,normalize
import numpy as np

def valid(X_valid,Y_valid,mu1,mu2,shared_sigma,N1,N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot((mu1-mu2),sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(
        float(N1) / N2)
    a = np.dot(w,x)+b
    y = sigmoid(a)
    y_ = np.round(y)
    result = (np.squeeze(Y_valid)==y_)
    print('Valid acc = %f' % (float(result.sum())/result.shape[0]))
    return

def train(X_all,Y_all):
    valid_set_percentage = 0.1
    X_train,Y_train,X_valid,Y_valid = split_valid_set(X_all,Y_all,valid_set_percentage)

    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    # means of Gaussian Models
    mu1 = np.zeros((106,))
    mu2 = np.zeros((106,))

    for i in range(train_data_size):
        if Y_train[i]==1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    # variances of Gaussian Models
    sigma1 = np.zeros((106, 106))
    sigma2 = np.zeros((106, 106))
    for i in range(train_data_size):
        if Y_train[i]==1:
            sigma1 += np.dot(np.transpose([X_train[i]-mu1]),[X_train[i]-mu1])
        else:
            sigma2 += np.dot(np.transpose([X_train[i]-mu1]),[X_train[i]-mu1])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1)/train_data_size)*sigma1+(float(cnt2)/train_data_size)*sigma2
    N1 = cnt1
    N2 = cnt2

    valid(X_valid,Y_valid,mu1,mu2,shared_sigma,N1,N2)

if __name__=='__main__':
    train_data_path = r'./data/X_train.csv'
    train_label_path = r'./data/Y_train.csv'
    X_all, Y_all = load_data(train_data_path, train_label_path)
    X_all = normalize(X_all)
    train(X_all,Y_all)