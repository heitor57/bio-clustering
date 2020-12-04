import numpy as np


def precision(actual,predicted):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(predicted)):
        for j in range(len(predicted)):
            if j>i:
                if predicted[i]==predicted[j]:
                    if actual[i]==actual[j]:
                        tp+=1
                    else:
                        fp+=1
                else:
                    if actual[i]==actual[j]:
                        fn+=1
                    else:
                        tn+=1
