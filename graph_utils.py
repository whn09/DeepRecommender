import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report  


def draw_ks(v, n, p, diff, pass_ratio, npass_ratio, ks, ks_x, filename):
    x = v
    plt.figure()
    plt.plot(x,n,'-', color='r', label='N')
    plt.plot(x,p,'-', color='g', label='P')
    plt.plot(x,diff,'-', color='b', label='KS')
    plt.plot(x,pass_ratio,'-', color='c', label=u'Pass')
    plt.plot(x,npass_ratio,'-', color='k', label=u'NPass')
    plt.vlines(ks_x, -1, 1, colors = "y", linestyles = "dashed")
    plt.xlim(v[0], v[-1])
    plt.ylim(-1,1)
    plt.legend()
    plt.title('ks='+str(round(ks,2))+' x='+str(round(ks_x,2)))
    plt.savefig(filename)
    #plt.show()
    

def sort_by_0(s):
    return s[0]

    
def ks(y_test, y_score, filename):
    if len(y_test) != len(y_score):
        print('ERROR! y_test != y_score')
        return
    v = []
    n = []
    p = []
    diff = []
    pass_ratio = []
    npass_ratio = []
    psum = 0
    nsum = 0
    values = {}
    for i in range(len(y_test)):
        value = int(y_test[i])
        if value == 1:
            psum += 1
        elif value == 0:
            nsum += 1
        else:
            print('ERROR! y_test != 0 or 1')
            return
        if y_score[i] not in values:
            values[y_score[i]] = [0, 0]
        values[y_score[i]][value] += 1
        
    values_list = []
    for ki,vi in values.items():
        values_list.append([ki, vi[0], vi[1]])
    values_list = sorted(values_list, key=sort_by_0)
    
    n_pre_sum = 0.0
    p_pre_sum = 0.0
    for value in values_list:
        v.append(value[0])
        try:
            n_rate = (nsum-n_pre_sum)/nsum
        except:
            n_rate = 0
        try:
            p_rate = (psum-p_pre_sum)/psum
        except:
            p_rate = 0
        n.append(n_rate)
        p.append(p_rate)
        diff.append(n_rate-p_rate)
        n_pre_sum += value[1]
        p_pre_sum += value[2]
        pass_ratio.append((nsum-n_pre_sum+psum-p_pre_sum)/(nsum+psum))
        npass_ratio.append((nsum-n_pre_sum)/(nsum+psum))
    
    ks = max(min(diff), max(diff), key=abs)
    print('ks:', ks)
    ks_x = 0
    for i in range(len(diff)):
        if diff[i] == ks:
            ks_x = v[i]
    #print('v:',len(v))
    #print('n:',len(n))
    #print('p:',len(p))
    #print('diff:',len(diff))
    print('psum:', psum)
    print('nsum:', nsum)
    draw_ks(v, n, p, diff, pass_ratio, npass_ratio, ks, ks_x, filename)
    

def draw_roc(fpr, tpr, roc_auc, filename):
    plt.figure()  
    lw = 2  
    plt.figure(figsize=(10,10))  
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('ROC')  
    plt.legend(loc="lower right")
    plt.savefig(filename)
    #plt.show() 
  
  
def roc(y_test, y_score, filename):
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr,tpr)
    print('AUC:', roc_auc)
    draw_roc(fpr, tpr, roc_auc, filename)
    
    
def f1score(y_test, y_score):
    max_y = int(max(y_test))
    print('max_y:',max_y)
    print('max(y_score):',max(y_score))
    for i in range(1, 10):
        thres = round(i*0.1, 2)
        y_pred = []
        for j in range(len(y_score)):
            for k in range(0, max_y+1):
                if k==0:
                    if y_score[j] < k+thres:
                        y_pred.append(k)
                        break
                elif k==max_y:
                    if y_score[j] >= k-1+thres:
                        y_pred.append(k)
                        break
                else:
                    if y_score[j] < k+thres and y_score[j] >= k-1+thres:
                        y_pred.append(k)
                        break
        print('max(y_pred):', max(y_pred))
        C=confusion_matrix(y_test, y_pred)
        R=classification_report(y_test, y_pred)
        print('thres:', thres, 'confusion_matrix:')
        print(C)
        print('thres:', thres, 'classification_report:')
        print(R)