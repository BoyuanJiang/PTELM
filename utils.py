#########utils function##########
###############################

import numpy as np
from numpy.random import shuffle
import os
from scipy import io
import sklearn.utils as skutil
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA


# transform Label vector to Label matrix
def LabelTransform(Y):
    num = len(np.unique(Y))
    LabelMat = np.zeros((len(Y), num))
    for i in range(len(Y)):
        LabelMat[i, Y[i] - 1] = 1
    return LabelMat


def RandomSplit(X, Y, Ratio):
    m = X.shape[0]
    n = int(m * Ratio)
    ind = range(m)
    shuffle(ind)
    X = X[ind, :]
    Y = Y[ind, :]
    X1 = X[0:n, :]
    X2 = X[n:, :]
    Y1 = Y[0:n, :]
    Y2 = Y[n:, :]
    return X1, Y1, X2, Y2


def get_dataset(base_path, source, target, source_nums=20, target_nums=3):
    source_base_path = os.path.join(base_path, source)
    target_base_path = os.path.join(base_path, target)
    source_classes = os.listdir(source_base_path)
    target_classes = os.listdir(target_base_path)
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = [], [], [], [], [], []
    for idx, source_class in enumerate(source_classes):
        tmp = os.listdir(os.path.join(source_base_path, source_class))
        shuffle(tmp)
        for iidx, item in enumerate(tmp):
            if iidx == source_nums:
                break;
            else:
                train_x_s.append(io.loadmat(os.path.join(source_base_path, source_class, item)).get('histogram'))
                train_y_s.append(idx + 1)
    for idx, target_class in enumerate(target_classes):
        tmp = os.listdir(os.path.join(target_base_path, target_class))
        shuffle(tmp)
        for iidx, item in enumerate(tmp):
            if iidx < target_nums:
                train_x_t.append(io.loadmat(os.path.join(target_base_path, target_class, item)).get('histogram'))
                train_y_t.append(idx + 1)
            else:
                test_x.append(io.loadmat(os.path.join(target_base_path, target_class, item)).get('histogram'))
                test_y.append(idx + 1)
    return np.squeeze(np.array(train_x_s)), np.squeeze(np.array(train_y_s, dtype=int)), \
           np.squeeze(np.array(train_x_t)), np.squeeze(np.array(train_y_t, dtype=int)), \
           np.squeeze(np.array(test_x)), np.squeeze(np.array(test_y, dtype=int))


def get_off_cal(source_path, target_path, source_nums, target_nums, random_state):
    source = io.loadmat(source_path)
    target = io.loadmat(target_path)
    source_x = source.get("fts")
    source_y = source.get("labels")
    target_x = target.get("fts")
    target_y = target.get("labels")
    source_x, source_y = skutil.shuffle(source_x, source_y, random_state=random_state)
    target_x, target_y = skutil.shuffle(target_x, target_y, random_state=random_state)
    source_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    target_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = [], [], [], [], [], []
    for i in range(len(source_y)):
        if source_dict[int(source_y[i])] < source_nums:
            source_dict[int(source_y[i])] += 1
            train_x_s.append(source_x[i])
            train_y_s.append(source_y[i])

    for i in range(len(target_y)):
        if target_dict[int(target_y[i])] < target_nums:
            target_dict[int(target_y[i])] += 1
            train_x_t.append(target_x[i])
            train_y_t.append(target_y[i])
        else:
            test_x.append(target_x[i])
            test_y.append(target_y[i])

    return np.squeeze(np.array(train_x_s)), np.squeeze(np.array(train_y_s, dtype=int)), \
           np.squeeze(np.array(train_x_t)), np.squeeze(np.array(train_y_t, dtype=int)), \
           np.squeeze(np.array(test_x)), np.squeeze(np.array(test_y, dtype=int))


def load_mmdt_split(split_file, source_path, target_path):
    source = io.loadmat(source_path)
    target = io.loadmat(target_path)
    split = io.loadmat(split_file)
    trains_s = split.get("train")['source'][0, 0][0]
    trains_t = split.get("train")['target'][0, 0][0]
    test_t = split.get("test")['target'][0, 0][0]
    trains_s_x, trains_s_y, trains_t_x, trains_t_y, test_t_x, test_t_y = [], [], [], [], [], []
    for i in range(20):
        trains_s_x.append(source.get("fts")[trains_s[i] - 1, :])
        trains_s_y.append(source.get("labels")[trains_s[i] - 1, :])
        trains_t_x.append(target.get("fts")[trains_t[i] - 1, :])
        trains_t_y.append(target.get("labels")[trains_t[i] - 1, :])
        test_t_x.append(target.get("fts")[test_t[i] - 1, :])
        test_t_y.append(target.get("labels")[test_t[i] - 1, :])
    return trains_s_x, trains_s_y, trains_t_x, trains_t_y, test_t_x, test_t_y


def load_text_dataset(dim,source_x,source_y,target_x, target_y, source_nums, target_nums, random_state=0):
    source_x, source_y = skutil.shuffle(source_x, source_y, random_state=random_state)
    target_x, target_y = skutil.shuffle(target_x, target_y, random_state=random_state)
    source_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    target_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = [], [], [], [], [], []
    # pca=PCA(dim)
    # pcaed=pca.fit_transform(np.row_stack([source_x,target_x]))
    # source_x,target_x = pcaed[0:len(source_y),:],pcaed[-len(target_y):,:]
    # pca = PCA(dim)
    # target_x=pca.fit_transform(target_x)
    for i in range(len(source_y)):
        if source_dict[int(source_y[i])] < source_nums:
            source_dict[int(source_y[i])] += 1
            train_x_s.append(source_x[i])
            train_y_s.append(source_y[i])
    for i in range(len(target_y)):
        if target_dict[int(target_y[i])] < target_nums:
            target_dict[int(target_y[i])] += 1
            train_x_t.append(target_x[i])
            train_y_t.append(target_y[i])
        else:
            test_x.append(target_x[i])
            test_y.append(target_y[i])
    return np.squeeze(np.array(train_x_s)), np.squeeze(np.array(train_y_s, dtype=int)), \
           np.squeeze(np.array(train_x_t)), np.squeeze(np.array(train_y_t, dtype=int)), \
           np.squeeze(np.array(test_x)), np.squeeze(np.array(test_y, dtype=int))

def seed_to_mat(name,source_x, source_y, target_x, target_y, source_nums, target_nums, seed_dict):
    train_ss, train_ts, test_ss, test_ts = np.zeros((20,),np.object),np.zeros((20,),np.object),np.zeros((20,),np.object),np.zeros((20,),np.object)
    for idx,seed in enumerate(seed_dict):
        source_index = np.array([i for i in range(1,len(source_y)+1)])
        target_index = np.array([i for i in range(1,len(target_y)+1)])
        _, _, source_index = skutil.shuffle(source_x, source_y, source_index, random_state=seed)
        _, _, target_index = skutil.shuffle(target_x, target_y, target_index, random_state=seed)
        source_dict = {i:0 for i in range(1,len(np.unique(source_y))+1)}
        target_dict = {i:0 for i in range(1,len(np.unique(target_y))+1)}
        train_s, train_t, test_s, test_t = [], [], [], []
        for i in range(len(source_y)):
            if source_dict[int(source_y[i])] < source_nums:
                source_dict[int(source_y[i])] += 1
                train_s.append(source_index[i])
            else:
                test_s.append(source_index[i])
        for i in range(len(target_y)):
            if target_dict[int(target_y[i])] < target_nums:
                target_dict[int(target_y[i])] += 1
                train_t.append(target_index[i])
            else:
                test_t.append(target_index[i])
        train_ss[idx]=train_s
        train_ts[idx]=train_t
        test_ss[idx]=test_s
        test_ts[idx]=test_t
    train = {"source":train_ss,"target":train_ts}
    test = {"source":test_ss,"target":test_ts}
    param = {"num_train_source": source_nums, "num_train_target": target_nums, "num_trials": 20}
    save_dict = {"train":train,"test":test,"param":param}
    io.savemat("SameCategory_"+name+"_20RandomTrials_10Categories.mat",save_dict)


def load_usps_dataset(dim,source_x,source_y,target_x, target_y, source_nums, target_nums, random_state=0):
    source_x, source_y = skutil.shuffle(source_x, source_y, random_state=random_state)
    target_x, target_y = skutil.shuffle(target_x, target_y, random_state=random_state)
    source_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,7:0,8:0,9:0,10:0}
    target_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,7:0,8:0,9:0,10:0}
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = [], [], [], [], [], []
    pca=PCA(dim)
    source_x = pca.fit_transform(source_x)
    # pcaed=pca.fit_transform(np.row_stack([source_x,target_x]))
    # source_x,target_x = pcaed[0:len(source_x),:], pcaed[len(source_x):,:]
    pca = PCA(dim)
    target_x=pca.fit_transform(target_x)
    for i in range(len(source_y)):
        if source_dict[int(source_y[i])] < source_nums:
            source_dict[int(source_y[i])] += 1
            train_x_s.append(source_x[i])
            train_y_s.append(source_y[i])
    for i in range(len(target_y)):
        if target_dict[int(target_y[i])] < target_nums:
            target_dict[int(target_y[i])] += 1
            train_x_t.append(target_x[i])
            train_y_t.append(target_y[i])
        else:
            test_x.append(target_x[i])
            test_y.append(target_y[i])
    save_dict = {"X_s":np.array(train_x_s),"X_t":np.array(train_x_t),"Y_s":np.array(train_y_s, dtype=int),"Y_t":np.array(train_y_t, dtype=int),"X_test":np.array(test_x),"test_y":np.array(test_y, dtype=int)}
    io.savemat(str(random_state)+".mat",save_dict)
    return np.squeeze(np.array(train_x_s)), np.squeeze(np.array(train_y_s, dtype=int)), \
           np.squeeze(np.array(train_x_t)), np.squeeze(np.array(train_y_t, dtype=int)), \
           np.squeeze(np.array(test_x)), np.squeeze(np.array(test_y, dtype=int))

def load_coil_dataset(dim,source_x,source_y,target_x, target_y, source_nums, target_nums, random_state=0):
    source_x, source_y = skutil.shuffle(source_x, source_y, random_state=random_state)
    target_x, target_y = skutil.shuffle(target_x, target_y, random_state=random_state)
    source_dict = {i:0 for i in range(1,21)}
    target_dict = {i:0 for i in range(1,21)}
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = [], [], [], [], [], []
    pca=PCA(dim)
    source_x = pca.fit_transform(source_x)
    # pcaed=pca.fit_transform(np.row_stack([source_x,target_x]))
    # source_x,target_x = pcaed[0:len(source_x),:], pcaed[len(source_x):,:]
    pca = PCA(dim)
    target_x=pca.fit_transform(target_x)
    for i in range(len(source_y)):
        if source_dict[int(source_y[i])] < source_nums:
            source_dict[int(source_y[i])] += 1
            train_x_s.append(source_x[i])
            train_y_s.append(source_y[i])
    for i in range(len(target_y)):
        if target_dict[int(target_y[i])] < target_nums:
            target_dict[int(target_y[i])] += 1
            train_x_t.append(target_x[i])
            train_y_t.append(target_y[i])
        else:
            test_x.append(target_x[i])
            test_y.append(target_y[i])
    return np.squeeze(np.array(train_x_s)), np.squeeze(np.array(train_y_s, dtype=int)), \
           np.squeeze(np.array(train_x_t)), np.squeeze(np.array(train_y_t, dtype=int)), \
           np.squeeze(np.array(test_x)), np.squeeze(np.array(test_y, dtype=int))

if __name__ == '__main__':
    # base_path="./data/office31/surf"
    # train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y= get_dataset(base_path,"amazon","webcam",20,3)
    trains_s_x, trains_s_y, trains_t_x, trains_t_y, test_t_x, test_t_y = load_mmdt_split(
        "./data/Office-Caltech/surf/mmdtsplit/SameCategory_amazon-caltech_20RandomTrials_10Categories.mat")
