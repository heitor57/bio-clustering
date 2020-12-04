import numpy as np

def clustering_tfpn(actual,predicted):
    tfpn_ = {
        'tp':0,
        'fp':0,
        'tn':0,
        'fn':0,
    }
    
    for i in range(len(predicted)):
        for j in range(len(predicted)):
            if j>i:
                if predicted[i]==predicted[j]:
                    if actual[i]==actual[j]:
                        tfpn_['tp']+=1
                    else:
                        tfpn_['fp']+=1
                else:
                    if actual[i]==actual[j]:
                        tfpn_['fn']+=1
                    else:
                        tfpn_['tn']+=1
    return tfpn_

def precision(tp,fp):
    return tp/(tp+fp)
