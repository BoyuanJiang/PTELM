####### test my code ########################
from PTELM import *
from utils import *
import numpy as np
from sklearn.utils import shuffle
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
import os
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import ParameterGrid
from scipy.stats import zscore

data_path = "./data/rcv1rcv2aminigoutte/SP"


def ELM_test(dim, source_x, source_y, target_x, target_y, target_nums, random_state):
    ##data perpare
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = load_text_dataset(dim, source_x, source_y, target_x,
                                                                                   target_y, source_nums=100,
                                                                                   target_nums=target_nums,
                                                                                   random_state=random_state)
    train_y_s = LabelTransform(train_y_s)
    train_y_t = LabelTransform(train_y_t)
    test_y = LabelTransform(test_y)

    ##train model
    net = ELM(train_x_t, train_y_t, test_x, test_y, 2000)
    net.ParamInit()
    net.Activation('relu')
    net.TrainELM('Lp', 0.01)
    net.TrainAccuracy('relu')
    net.TestAccuracy('relu')
    net.printf()
    return net.TestAcc * 100

def run_svm(dim,source_x,source_y,target_x, target_y, target_nums, random_state):
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = load_text_dataset(dim,source_x,source_y,target_x, target_y,source_nums=100,target_nums=target_nums,random_state=random_state)
    pca = PCA(40)
    pcaed=pca.fit_transform(np.row_stack([train_x_s,train_x_t,test_x]))
    train_x_s, train_x_t, test_x = pcaed[0:len(train_x_s),:], pcaed[len(train_x_s):len(train_x_s)+len(train_x_t),:],pcaed[-len(test_x):,:]
    clf = SVC()
    clf.fit(train_x_t, train_y_t)
    acc = clf.score(test_x, test_y)
    print acc
    return acc

def PTELM_test(f, dim, source_x, source_y, target_x, target_y, c0, c1, c2, nodes, method, target_nums, random_state=0):
    ##data perpare
    # Amazon=io.loadmat('amazon_zscore_SURF_L10.mat')
    # Caltech=io.loadmat('Caltech10_zscore_SURF_L10.mat')
    # Amazon = io.loadmat('amazon_decaf.mat')
    # # Caltech = io.loadmat('caltech_decaf.mat')
    # Amazon = io.loadmat('./data/Office-Caltech/surf/zscore/amazon_zscore_SURF_L10.mat')
    # Webcam = io.loadmat('./data/Office-Caltech/surf/zscore/webcam_zscore_SURF_L10.mat')
    # Amazon_x=Amazon.get('Xt')
    # Amazon_y=Amazon.get('Yt')
    # Webcam_x=Webcam.get('Xt')
    # Webcam_y = Webcam.get('Yt')
    # Amazon_Y=LabelTransform(Amazon_y)
    # Webcam_Y=LabelTransform(Webcam_y)
    # Webcam_x,Webcam_Y,test_x,test_y=RandomSplit(Webcam_x,Webcam_Y,0.1)
    # train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = get_dataset(base_path, "amazon", "webcam", 20, 3)
    # train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = get_off_cal("./data/Office-Caltech/surf/caltech_SURF_L10.mat","./data/Office-Caltech/surf/dslr_SURF_L10.mat",8,3,random_state)

    print("!!!!!!! " + source_name + "-->" + target_name + " !!!!!!!!!!")
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = load_text_dataset(dim, source_x, source_y, target_x,
                                                                                   target_y, source_nums=100,
                                                                                   target_nums=target_nums,
                                                                                   random_state=random_state)
    # train_x_s = preprocessing.normalize(train_x_s)
    # tmp = np.row_stack([train_x_s,train_x_t, test_x])
    # tmp = np.row_stack([train_x_t, test_x])
    train_y_s = LabelTransform(train_y_s)
    train_y_t = LabelTransform(train_y_t)
    test_y = LabelTransform(test_y)
    pca = PCA(dim)
    pcaed=pca.fit_transform(np.row_stack([train_x_s,train_x_t,test_x]))
    train_x_s, train_x_t, test_x = pcaed[0:len(train_x_s),:], pcaed[len(train_x_s):len(train_x_s)+len(train_x_t),:],pcaed[-len(test_x):,:]
    # pca_t = PCA(target_dim)
    # if reduc == "pca":
    #     tmp= pca_s.fit_transform(tmp)
    #     # paced = pca_t.fit_transform(tmp)
    #     train_x_s, train_x_t, test_x = tmp[0:len(train_x_s), :], tmp[len(train_x_s):len(train_x_s)+60, :], tmp[-len(test_x):,:]
    # else:
    #     raise NotImplementedError
    accs = []
    tmp = []
    for _ in range(3):
        if method == 1:
            net = PTELM1(train_x_s, train_x_t, train_y_s, train_y_t, test_x, test_y, nodes)
        elif method == 2:
            net = PTELM2(train_x_s, train_x_t, train_y_s, train_y_t, test_x, test_y, nodes)
        elif method == 0:
            net = PTELM(train_x_s, train_x_t, train_y_s, train_y_t, test_x, test_y, nodes)
        else:
            raise NotImplementedError
        net.ParamInit()
        net.Activation('relu')
        net.TrainPTELM(c0, c1, c2, 2)
        net.TestPTELM('relu')
        net.printf()
        f.write("seed:%d, Acc:%.2f\n" % (random_state, net.TestAcc))
        tmp.append(net.TestAcc)
    # f.write("%s-->%s, Acc:%.2f, Std:%.2f\n" % (source_name, targer_name, np.mean(accs)*100,np.std(accs)*100/np.sqrt(len(accs))))
    return np.mean(tmp) * 100


if __name__ == '__main__':
    # ELM_test()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--c0", type=float, default=1)
    parser.add_argument("--c1", type=int, default=30)
    parser.add_argument("--c2", type=int, default=10)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--t", type=int, default=20)


    args = parser.parse_args()
    c0, c1, c2, nodes, method, target_nums, dim = args.c0, args.c1, args.c2, args.n, args.m, args.t, args.d
    import datetime

    source_name = "FR"
    target_name = "SP"
    # now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    # log_path = os.path.join("./result/text/search",
    #                         now + "c1-" + str(c1) + "c2-" + str(c2) + "nodes-" + str(nodes) + "method-" + str(
    #                             method) + " dim-" + str(dim) + " target_nums-" + str(target_nums)+source_name)
    # os.mkdir(log_path)

    source_path = os.path.join(data_path, "Index_" + source_name + "-SP")
    target_path = os.path.join(data_path, "Index_" + target_name + "-SP")
    source_x, source_y = load_svmlight_file(source_path)
    target_x, target_y = load_svmlight_file(target_path)
    source_x, target_x = source_x.A, target_x.A

    EN_SP = [23, 26, 27, 50, 67, 68, 61, 62, 58, 80, 88, 3, 18, 33, 41, 42, 9, 21, 53, 85]
    FR_SP = [23, 61, 26, 50, 62, 27, 42, 45, 74, 59, 67, 68, 73, 75, 85, 87, 88, 10, 96, 28]
    GR_SP = [26, 59, 62, 14, 50, 61, 3, 9, 16, 27, 41, 42, 58, 79, 85, 33, 6, 28, 87, 53]
    IT_SP = [62, 26, 85, 23, 28, 33, 81, 59, 61, 78, 27, 43, 45, 58, 77, 90, 91, 0, 3, 6]
    seed_dict = {"EN": EN_SP,"FR":FR_SP,"GR":GR_SP,"IT":IT_SP}
    # param_grid = {"c":[40,60,80,100],"nodes":[500,1000,1500,2000],"dim":[30,60,120,240,360]}
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    log_path = os.path.join("/tmp", now)
    # log_path = os.path.join("./result/text/5", now)
    os.mkdir(log_path)
        # for param in list(ParameterGrid(param_grid)):
    accs = []
    with open(os.path.join(log_path, "c0-"+str(c0)+"c1-" + str(c1) + "c2-" + str(c2) + "nodes-" + str(nodes) + "method-" + str(
            method) + " dim-" + str(dim) + " target_nums-" + str(target_nums) + source_name + ".txt"), "w") as f:
        for i in seed_dict[source_name]:
            # print(i)
            acc=PTELM_test(f,dim,source_x,source_y,target_x, target_y,c0,c1,c2,nodes,method,target_nums=target_nums,random_state=i)
            # acc = ELM_test(dim, source_x, source_y, target_x, target_y, target_nums, random_state=i)
            # acc = run_svm(dim, source_x, source_y, target_x, target_y, target_nums,random_state=i)
            accs.append(acc)
        f.write("Acc:%.1f,Std:%.1f\n" % (np.mean(accs),np.std(accs)/np.sqrt(len(accs))))
        print("acc:%.2f,std:%.2f\n" % (np.mean(accs),np.std(accs)/np.sqrt(len(accs))))
