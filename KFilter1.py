import numpy as np
import math

def KFilter1(num,z,A,mu0,Sigma0,Phi,Ups,Gam,cQ,cR,inp):
    # Note: Must give cholesky decomp: cQ = np.linalg.choleske(Q),cR = np.linalg.choleske(R)
    Q = np.dot(cQ.T,cQ)
    R = np.dot(cR.T,cR)
    
    
    # z is num by q (time=row series=col) 
    # input is num by r (use 0 if not needed) 
    # A is an array with dim=c(q,p,num)
    # Ups is p by r (use 0 if not needed) Gam is q by r (use 0 if not needed)
    # R is q by q 
    # mu0 is p by 1 
    # Sigma0, Phi, Q are p by p,   Constant.
    Phi,z,inp = np.array(Phi),np.array(z), np.array(inp)
    pdim = Phi.shape[0]
    qdim = z.shape[1]  #
    rdim = inp.shape[1]  # 
    
    if np.max(np.abs(Ups)) == 0:
        Ups = np.zeros((pdim,rdim))
    if np.max(np.abs(Gam)) == 0:
        Gam = np.zeros((qdim,rdim))
    
    xp, xf = np.zeros([num,pdim,1]), np.zeros([num,pdim,1])  # Store the results
    Pp,Pf = np.zeros([num,pdim,pdim]), np.zeros([num,pdim,pdim])
    innov,sig = np.zeros([num,qdim,1]), np.zeros([num,qdim,qdim]) # Innovations, and the covariance matrix
    
    # Initialize
    x0,Sigma0 = np.array(mu0),np.array(Sigma0)
    xp[0] = np.dot(Phi,mu0) + np.dot(Ups,inp[0]) 
    Pp[0] = np.dot(np.dot(Phi, Sigma0),Phi.T) + Q
    
    A_cur = A[0]
    sigtemp = np.dot(np.dot(A_cur,Pp[0]),A_cur.T) + R
    sig[0] = (sigtemp.T +sigtemp) / 2
    
    sigInv = np.linalg.inv(sig[0])
    
    K = np.dot(np.dot(Pp[0], A_cur.T), sigInv)
    innov[0] = z[0] - np.dot(A_cur,xp[0]) - np.dot(Gam,inp[0])
    
    ## update
    xf[0] = xp[0] + np.dot(K,innov[0])
    Pf[0] = Pp[0] - np.dot(np.dot(K,A_cur),Pp[0])
    
    den = (1/np.sqrt(np.linalg.det(sig[0]))/(2*math.pi)**.5)
    if den == 0:
        den = den+10**(-300)
        print('Zer0 Encountered')
    like = -np.log(den) -( -.5 * np.dot(np.dot(innov[0].T,sigInv),innov[0]))  #-log(Likelihood)
    
    for i in range(1,num):
        xp[i] = np.dot(Phi,xf[i-1]) + np.dot(Ups,inp[i])
        Pp[i] = np.dot(np.dot(Phi, Pf[i-1]),Phi.T) + Q
        
        A_cur = A[i]
        sigtemp = np.dot(np.dot(A_cur,Pp[i]),A_cur.T) + R
        sig[i] = (sigtemp.T +sigtemp) / 2

        sigInv = np.linalg.inv(sig[i])

        K = np.dot(np.dot(Pp[i], A_cur.T), sigInv)
        innov[i] = z[i] - np.dot(A_cur,xp[i]) - np.dot(Gam,inp[i])

        ## update
        xf[i] = xp[i] + np.dot(K,innov[i])
        Pf[i] = Pp[i] - np.dot(np.dot(K,A_cur),Pp[i])
           
        den = (1/np.sqrt(np.linalg.det(sig[i]))/(2*math.pi)**.5)
        like = like - np.log(den) -( -.5 * np.dot(np.dot(innov[i].T,sigInv),innov[i])) #-log(Likelihood), very small zero!!!
    # Output
    re = {}
    re['like'] = like
    re['xp'] = xp
    re['xf'] = xf
    re['Pp'] = Pp
    re['Pf'] = Pf
    re['innov'] = innov
    re['Sig'] = sig          
           
    return re