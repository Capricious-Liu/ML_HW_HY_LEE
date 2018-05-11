import csv
import numpy as np
import random
import math

def readCSV():
    data = []
    for i in range(18):
        data.append([])
    text = open('data/train.csv','r',encoding='big5')
    row = csv.reader(text,delimiter=',')

    n_row = 0
    for r in row:
        if n_row!=0:
            for i in range(3,27):
                if r[i] != 'NR':
                    data[(n_row-1)%18].append(float(r[i]))
                else:
                    data[(n_row-1)%18].append(float(0))
        n_row+=1
    text.close()
    return data

def parseData(data):
    x = []
    y = []
    for i in range(12):
        for j in range(471):
            x.append([])
            for t in range(18):
                for s in range(9):
                    x[471*i+j].append(data[t][480*i+j+s])
            y.append(data[9][480*i+j+s])
    x = np.array(x)
    y = np.array(y)
    x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    return x,y

def initVariables(len):
    w = np.zeros(len)
    # l_rate = 0.0000000001
    # adagrad -> less rely on l_rate
    l_rate = 1
    repeat = 10000
    return w,l_rate,repeat

def training(x,y,w,l_rate,repeat):
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))

    print(x.shape)
    print(w.shape)

    for i in range(repeat):
        hypo = np.dot(x,w)
        loss = hypo - y
        cost = np.sum(loss**2)/len(x)
        cost_a = math.sqrt(cost)
        gra = np.dot(x_t,loss)
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w -= l_rate*gra/ada
        print('iteration:%d | Cost:%f   '% (i,cost_a))


if __name__=="__main__":
    data = readCSV()
    x,y = parseData(data)
    w,l_rate,repeat = initVariables(len(x[0]))
    training(x,y,w,l_rate,repeat)