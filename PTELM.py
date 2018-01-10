## By chenchao 2017-12-20
## mail: chench@zju.edu.cn

from numpy import *
from scipy import io
from numpy.linalg import *
import time
from utils import *
from IPLearning import *
from numba import jit


class ELM:
    def __init__(self,train_x,train_y,test_x,test_y,NumofHiddenNodes):
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
        self.n_feature=self.train_x.shape[1]
        self.num=NumofHiddenNodes
        self.n_class=self.train_y.shape[1]
        self.runtime=time.time()


    def ParamInit(self):
        self.InputWeighws=2*random.random((self.num,self.n_feature))-1
        self.H=dot(self.InputWeighws,self.train_x.T)


    def Activation(self,type):
        if (type == 'sigmoid'):
            self.H=1/(1+exp(self.H))
        if (type == 'tanh'):
            self.H=(exp(self.H)-exp(-self.H))/(exp(self.H)+exp(-self.H))
        if (type == 'relu'):
            ind =self.H>0
            self.H=self.H*ind
        if (type=='IPL'):
            self.H=IPL(self.H)


    def TrainELM(self,type,coeff):
        if (type=='None'):
            temp=mat(dot(self.H,self.H.T))
            temp=temp.I
            self.OutputWeights=dot(dot(temp,self.H),self.train_y)
            self.OutputWeights=array(self.OutputWeights)
        if (type=='Lp'):
            temp=mat(dot(self.H,self.H.T))
            temp=temp+eye(temp.shape[0])/coeff
            temp=temp.I
            self.OutputWeights = dot(dot(temp,self.H),self.train_y)
            self.OutputWeights = array(self.OutputWeights)
        self.runtime = time.time()-self.runtime


    def TestELM(self,type):
        self.H = dot(self.InputWeighws, self.test_x.T)
        self.Activation(type)
        self.Actual_y=dot(self.H.T,self.OutputWeights)


    def TrainAccuracy(self,type):
        self.H = dot(self.InputWeighws, self.train_x.T)
        self.Activation(type)
        self.Actual_y=dot(self.H.T, self.OutputWeights)
        out_y = argmax(self.Actual_y, 1)+1
        actual_y = argmax(self.train_y,1)+1
        ind = where(out_y==actual_y)
        self.TrainAcc = floor(len(ind[0]))/len(self.train_y)



    def TestAccuracy(self,type):
        self.H=dot(self.InputWeighws,self.test_x.T)
        self.Activation(type)
        self.Actual_y=dot(self.H.T,self.OutputWeights)
        out_y=argmax(self.Actual_y,1)+1
        actual_y = argmax(self.test_y, 1) + 1
        ind=where(out_y==actual_y)
        self.TestAcc=floor(len(ind[0]))/len(self.test_y)

    def printf(self):
        print 'train accuracy:'
        print self.TrainAcc
        print 'test accuracy:'
        print self.TestAcc
        print 'run time(s):'
        print self.runtime

####################################################################################
####################################################################################
####################################################################################

class PTELM0:

    def __init__(self,Xs,Xt,Ys,Yt,test_x,test_y,NumOfHiddenNodes):
        self.Xs=Xs
        self.Xt=Xt
        self.Ys=Ys
        self.Yt=Yt
        self.test_x=test_x
        self.test_y=test_y
        self.n_feature1=self.Xs.shape[1]
        self.n_feature2=self.Xt.shape[1]
        self.num=NumOfHiddenNodes
        self.n_class=self.Ys.shape[1]


    def ParamInit(self):
        self.runtime=time.time()
        self.InputWeights1=2*random.random((self.num,self.n_feature1))-1
        self.InputWeights2=2*random.random((self.num,self.n_feature2))-1
        self.Hs=mat(self.InputWeights1)*mat(self.Xs.T)
        self.Ht=mat(self.InputWeights2)*mat(self.Xt.T)
        self.M=eye(self.num)
        self.A=zeros((self.num,self.n_class))
        #################################################
        self.Ms=eye(self.num)
        self.Mt=eye(self.num)
        self.OutputWeights=random.random((self.num,self.Ys.shape[1]))



    def Activation(self,type):
        self.Hs=array(self.Hs)
        self.Ht=array(self.Ht)
        if (type == 'sigmoid'):
            self.Hs=1/(1+exp(self.Hs))
            self.Ht=1/(1+exp(self.Ht))
        if (type == 'tanh'):
            self.Hs=(exp(self.Hs)-exp(-self.Hs))/(exp(self.Hs)+exp(-self.Hs))
            self.Ht=(exp(self.Ht)-exp(-self.Ht))/(exp(self.Ht)+exp(-self.Ht))
        if (type == 'relu'):
            ind1=self.Hs>0
            self.Hs=self.Hs*ind1
            ind2= self.Ht>0
            self.Ht=self.Ht*ind2
        if (type=='IPL'):
            self.Hs=IPL(self.Hs)
            self.Ht=IPL(self.Ht)



    def TrainPTELM(self,C1,C2,iter):
        k=0
        C2=0
        self.Hs=mat(self.Hs)
        self.Ht=mat(self.Ht)
        self.M=mat(self.M)
        self.Ys=mat(self.Ys)
        self.Yt=mat(self.Yt)
        I=mat(eye(self.Hs.shape[0]))
        while (k<iter):
            temp1=self.Hs*self.Hs.T+self.M.T*self.Ht*self.Ht.T*self.M+C1*I+C2*(self.M-I).T*(self.M-I)
            temp2=self.Hs*self.Ys+self.M.T*self.Ht*self.Yt
            self.OutputWeights1=pinv(temp1)*temp2
            temp3=self.Ht*self.Ht.T+C2*I
            temp4=self.Ht*self.Yt*self.OutputWeights1.T+C2*self.OutputWeights1*self.OutputWeights1.T
            temp5=self.OutputWeights1*self.OutputWeights1.T
            self.M=pinv(temp3)*temp4*pinv(temp5)
            k+=1
        self.OutputWeights2=self.M*self.OutputWeights1
        self.runtime = time.time()-self.runtime


    def TestPTELM(self,type):
        self.H=dot(self.InputWeights2,self.test_x.T)
        self.Activation(type)
        self.Actual_y = array(self.H.T*self.OutputWeights2)
        out_y=argmax(self.Actual_y,1)+1
        actual_y=argmax(self.test_y,1)+1
        ind = where(out_y==actual_y)
        self.TestAcc=floor(len(ind[0]))/len(self.test_y)


    def printf(self):
        print ('Test Accuracy:#######',self.TestAcc, 'run time:',self.runtime)


############################################################################
############################################################################
## define BetaT=M*BetaS+A

class PTELM(PTELM0):


    def TrainPTELM(self,C1,C2,iter):
        k=0
        self.Hs=mat(self.Hs)
        self.Ht=mat(self.Ht)
        self.M=mat(self.M)
        self.Ys=mat(self.Ys)
        self.Yt=mat(self.Yt)
        I=mat(eye(self.Hs.shape[0]))
        while (k<iter):
            temp1=self.Hs*self.Hs.T+self.M.T*self.Ht*self.Ht.T*self.M+C1*I+C2*self.M.T*self.M
            temp2=self.Hs*self.Ys+self.M.T*self.Ht*self.Yt-self.M.T*self.Ht*self.Ht.T*self.A-C2*self.M.T*self.A
            self.OutputWeights1=pinv(temp1)*temp2
            temp3=self.Ht*self.Yt*self.OutputWeights1.T-self.Ht*self.Ht.T*self.A*self.OutputWeights1.T-C2*self.A*self.OutputWeights1.T
            self.M=pinv(self.Ht*self.Ht.T+C2*I)*temp3*pinv(self.OutputWeights1*self.OutputWeights1.T)
            temp4=self.Ht*self.Yt-self.Ht*self.Ht.T*self.M*self.OutputWeights1-C2*self.M*self.OutputWeights1
            self.A=pinv(self.Ht*self.Ht.T+C2*I)*temp4
            k+=1
        self.OutputWeights2=self.M*self.OutputWeights1+self.A
        self.runtime = time.time()-self.runtime


####################################################################################
####################################################################################
##BetaT=Mt*Beta0  BetaS=Ms*Beta0
class PTELM1(PTELM0):
    def TrainPTELM(self,C1,C2,iter):
        k=0
        self.Hs=mat(self.Hs)
        self.Ht=mat(self.Ht)
        self.Ms=mat(self.Ms)
        self.Mt=mat(self.Mt)
        self.Ys=mat(self.Ys)
        self.Yt=mat(self.Yt)
        I=mat(eye(self.Hs.shape[0]))
        while (k < iter):
            temp1=self.Ms.T*self.Hs*self.Hs.T*self.Ms+self.Mt.T*self.Ht*self.Ht.T*self.Mt+C1*self.Ms.T*self.Ms+C2*self.Mt.T*self.Mt
            temp2=self.Ms.T*self.Hs*self.Ys+self.Mt.T*self.Ht*self.Yt
            self.OutputWeights=pinv(temp1)*temp2
            self.Ms=pinv(self.Hs*self.Hs.T*C1*I)*self.Hs*self.Ys*self.OutputWeights.T*pinv(self.OutputWeights*self.OutputWeights.T)
            self.Mt=pinv(self.Ht*self.Ht.T*C2*I)*self.Ht*self.Yt*self.OutputWeights.T*pinv(self.OutputWeights*self.OutputWeights.T)
            k += 1
        self.OutputWeights2=self.Mt*self.OutputWeights
        self.runtime=time.time()-self.runtime


#################################################################################
#################################################################################
## add L_21 regularization on the outputweights1

class PTELM2(PTELM0):
    def TrainPTELM(self,C,C1,C2,iter):
        k=0
        self.Hs=mat(self.Hs)
        self.Ht=mat(self.Ht)
        self.Ys=mat(self.Ys)
        self.Yt=mat(self.Yt)
        self.M=mat(self.M)
        I=mat(eye(self.Hs.shape[0]))
        self.OutputWeights1=random.random((self.num,self.n_class))
        while(k<iter):
            n=1
            self.D=I
            tol=1
            while(n<100 and tol>1e-10):
                OldOutputweights=self.OutputWeights1
                temp1=C*self.Hs*self.Hs.T+self.M.T*self.Ht*self.Ht.T*self.M+C1*self.D+C2*self.M.T*self.M
                temp2=C*self.Hs*self.Ys+self.M.T*self.Ht*self.Yt
                self.OutputWeights1=pinv(temp1)*temp2
                self.UpdateD()
                tol=norm(OldOutputweights-self.OutputWeights1)
                n+=1
            temp3=self.Ht*self.Ht.T+C2*I
            temp4=self.Ht*self.Yt*self.OutputWeights1.T
            temp5=self.OutputWeights1*self.OutputWeights1.T
            self.M=pinv(temp3)*temp4*pinv(temp5)
            k+=1
        self.OutputWeights2=self.M*self.OutputWeights1
        self.runtime=time.time()-self.runtime



    def UpdateD(self):
        for i in range(self.D.shape[0]):
            self.D[i,i]=1.0/(2*norm(self.OutputWeights1[i,:]))




#################################################################################
#################################################################################
## add L_21 regularization on both the outputweights1 and outputweights2

class PTELM3(PTELM0):
    def TrainPTELM(self, C1, C2, iter):
        k=0
        self.Hs = mat(self.Hs)
        self.Ht = mat(self.Ht)
        self.Ys = mat(self.Ys)
        self.Yt = mat(self.Yt)
        self.M = mat(self.M)
        I = mat(eye(self.Hs.shape[0]))
        while (k<iter):
            n=1
            num=100
            self.D=I
            self.G=I
            while(n<num):
                temp1=self.Hs*self.Hs.T+self.M.T*self.Ht*self.Ht.T*self.M+C1*self.D+C2*self.M.T*self.G*self.M
                temp2=self.Hs*self.Ys+self.M.T*self.Ht*self.Yt
                self.OutputWeights1 = pinv(temp1) * temp2
                self.UpdateD()
                # self.UpDateG()
                n+=1
            n=1
            while (n<num):
                temp3=self.Ht*self.Ht.T+C2*self.G
                temp4=self.Ht*self.Yt*self.OutputWeights1.T
                temp5=self.OutputWeights1*self.OutputWeights1.T
                self.M=pinv(temp3)*temp4*pinv(temp5)
                self.UpDateG()
                n+=1
            k+=1
        self.OutputWeights2 = self.M*self.OutputWeights1
        self.runtime = time.time()-self.runtime

    def UpdateD(self):
        for i in range(self.D.shape[0]):
            self.D[i,i]=1.0/(2*norm(self.OutputWeights1[i,:]))


    def UpDateG(self):
        temp=self.M*self.OutputWeights1
        for i in range(self.G.shape[0]):
            self.G[i,i]=1.0/(2*norm(temp[i,:]))



###################################################################################
#####################################################################################
## Unsupervised domain adaptation via parameter transfering


class USPTELM(PTELM0):

    def GraphLaplacian(self,sigma):
        self.L=zeros((self.Ys.shape[0],self.Ys.shape[0]))
        D=zeros(self.L.shape)
        W=SimilarityMat(self.Ys,sigma)
        for i in range(W.shape[0]):
            D[i,i]=W[i,:].sum()
        self.L=D-W


    def TrainPTELM(self,C1,C2,iter):
        k = 0
        self.Hs = mat(self.Hs)
        self.Ht = mat(self.Ht)
        self.M = mat(self.M)
        self.Ys = mat(self.Ys)
        self.Yt = mat(self.Yt)
        I = mat(eye(self.Hs.shape[0]))
        self.GraphLaplacian(self,0.1)
        while (k < iter):
            temp1=I+self.M.T*self.M+C1*self.Hs*self.Hs.T+C2*self.M.T*self.Ht*self.L.T*self.Ht.T*self.M
            self.OutputWeights1=C1*pinv(temp1)*self.Hs*self.Ys
            temp2=self.Ht*self.L.T*self.Ht.T*self.M*self.OutputWeights1*self.OutputWeights1.T
            self.M=0
            k += 1
        self.OutputWeights2 = self.M * self.OutputWeights1 + self.A
        self.runtime=time.time()-self.runtime

















