import csv
import numpy as np
def readCSV():
    data = []
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
    x = np.array()
    y = np.array()
    x = np.concatenate((np.ones(x.shape[0],1),x),axis=1)
    return x,y

def training(x,y):


if __name__=="__main__":
    data = readCSV()
    x,y = parseData(data)