import numpy as np
import torch 

def CSBM(N,alpha,d,l,mu,matrix='BE',genX=True):
    # Generate contextual stochastic block model data
    #
    # Inputs
    #   N:           Number of data points.
    #   alpha:       Sample ratio, alpha=1/gamma=F/N and F is the dimension of the feature. 
    #   d:           Averaged degree
    #   l:           Graph SNR
    #   mu:          Feature SNR
    #   matrix:      Options: 
    #                BN:  binary non-symmetric adjacency matrix
    #                BS:  binary symmetric adjacency matrix
    #                GN:  Gaussian non-symmetric adjacency matrix
    #                GS:  Gaussian symmetric adjacency matrix
                    

    hN=int(N/2)
    F=int(alpha*N)
    y=np.ones(hN)
    y=np.concatenate([y,-y],axis=0).reshape(N,1)
    if genX:
        u=np.random.randn(F,1)/np.sqrt(F)
        Omega=np.random.randn(N,F)
        X=np.sqrt(mu/N)*y@u.T+Omega/np.sqrt(F)
    else:
        X=None
    if matrix=='GS':
        Arand=np.random.randn(N,N)/np.sqrt(N)
        A=np.triu(Arand, k=1)
        A=A+A.T
        A[range(N),range(N)]=Arand[range(N),range(N)]*np.sqrt(2)
        Amean=y@y.T*l/N
        A=A+Amean
    if matrix=='GN':
        Amean=y@y.T*l/N
        Arand=np.random.randn(N,N)/np.sqrt(N)
        A=Arand+Amean
    if matrix=='BS':
        cin=(d+l*np.sqrt(d))/N
        cout=(d-l*np.sqrt(d))/N
        Aones=np.ones([hN,hN])
        A_p=np.concatenate([np.concatenate([cin*Aones,cout*Aones],axis=1),np.concatenate([cout*Aones,cin*Aones],axis=1)],axis=0)
        A_p=np.triu(A_p, k=0)
        A=torch.bernoulli(torch.tensor(A_p))
        A=(A+A.T).to(torch.long).numpy()
        A[A>1]=1
        A=A/np.sqrt(d)
    if matrix=='BN':
        cin=(d+l*np.sqrt(d))/N
        cout=(d-l*np.sqrt(d))/N
        Aones=np.ones([hN,hN])
        A_p=np.concatenate([np.concatenate([cin*Aones,cout*Aones],axis=1),np.concatenate([cout*Aones,cin*Aones],axis=1)],axis=0)
        A=torch.bernoulli(torch.tensor(A_p))
        A=A.to(torch.long).numpy()
        A=A/np.sqrt(d)
    return A,X,y


def LS(A,X,y,r,tau=1):
    #return train acc, test acc,  train risk, test risk, loss
    F=X.shape[1]
    N=X.shape[0]
    
    if tau==1:
        Q=A@X
        temp=Q.T@Q+r*np.eye(F)
        if r==0 and tau*N/F<=1:
            w=np.linalg.pinv(temp)@Q.T@y
        else:
            w=np.linalg.inv(temp)@Q.T@y
        h=Q@w
        return np.sum(y==np.sign(h))/N,0, np.linalg.norm(y-h)**2/N,0,(np.linalg.norm(y-h)**2+r*np.linalg.norm(w)**2)/N
    else:
        r=r*tau
        id1=np.random.choice(N, int((tau)*N), replace=False)
        id2=np.array(list(set(range(N))^set(list(id1))))
        A1=A[id1,:]
        A2=A[id2,:]
        Q1=A1@X
        Q2=A2@X
        y1=y[id1]
        y2=y[id2]
        temp=Q1.T@Q1+r*np.eye(F)
        if r==0 and tau*N/F<=1:
            w=np.linalg.pinv(temp)@Q1.T@y1
        else:
            w=np.linalg.inv(temp)@Q1.T@y1
        #h1=np.sign(A1@X@w)
        #h2=np.sign(A2@X@w)
        h1=Q1@w
        h2=Q2@w
        return np.sum(np.sign(h1)==y1)/(tau*N), np.sum(np.sign(h2)==y2)/(N-N*tau),np.linalg.norm(y1-h1)**2/(tau*N),np.linalg.norm(y2-h2)**2/(N-N*tau),(np.linalg.norm(y1-h1)**2+r*np.linalg.norm(w)**2)/N/tau

