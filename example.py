from KFilter1 import KFilter1
import numpy as np

nrow = 100
z = np.reshape(np.random.randn(nrow),(nrow,1,1))
A = np.ones((nrow,1))
Inp = np.ones((nrow,0))

u = 0
mu0 = 0
sigma0 = 1

def linn(par):
     # function to calculate likelihood
    Phi = np.array([[par[0]]]) 
    cQ = np.array([[par[1]]])
    cR = np.array([[par[2]]])

    Ups = np.array([[0]])

    kf = KFilter1(nrow,z,A,mu0,sigma0,Phi,Ups,np.array([[0]]),cQ,cR,Inp)

    return kf['like']
init_par =[1,1,1]
print(linn(init_par))
    
