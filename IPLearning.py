##########IP Learning###########
from numpy import *
from numpy.matlib import repmat


def ExpDistribution(len):
    target=random.exponential(scale=1,size=(1,len))
    return target

def WeibullDistribution(len):
    target=random.weibull(1,len)
    return target


def IPLearning(actural_out):
    actural_out=sort(actural_out,1)
    a=zeros((len(actural_out)))
    b=zeros((len(actural_out)))
    for i in range(actural_out.shape[0]):
        M=vstack((actural_out[i,:],ones((actural_out[i,:].shape)))).T
        T=ExpDistribution(actural_out.shape[1])
        #T=WeibullDistribution(actural_out.shape[1])
        T=sort(T/(1.0001*T.max()))
        T=log(1.0/T-1)
        temp=mat(dot(M.T,M))
        # temp=temp+0.01*eye(temp.shape[0])
        temp=array(temp.I)
        v=dot(dot(temp,M.T),T.T)
        a[i]=v[0];b[i]=v[1]
    return a,b


def IPActivation(X,a,b):
    A=repmat(a.reshape(len(a),1),1,X.shape[1])
    B=repmat(b.reshape(len(b),1),1,X.shape[1])
    H=1/(1+exp(A*X+B))
    return H

def IPL(X):
    a,b=IPLearning(X)
    H=IPActivation(X,a,b)
    return H
