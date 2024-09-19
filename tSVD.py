import numpy as np
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from KrylovSchur import *
import torch 




def fftKilmer(A):
    A = np.fft.fft(A, axis=2)
    return(A)


def ifftKilmer(A):
    A = np.fft.ifft(A, axis=2)
    return(A)


# def tProduct(A,B):
#     Abar = fftKilmer(A)
#     Bbar = fftKilmer(B)
#     n1,ell,n3=A.shape
#     n2=A.shape[1]
#     Cbar = np.zeros((n1,n2,n3))
#     mid=n3//2
#     for i in range(mid +1):
#         Cbar[:,:,i]=np.dot(Abar[:,:,i],Bbar[:,:,i])
#         Cbar[:,:,mid+i-1] =np.matrix.conjugate(Cbar[:,:,mid-i-2]) 
#     C=ifftKilmer(Cbar)
#     return np.real(C)
def tProduct(A,B):
    Abar = fftKilmer(A)
    Bbar = fftKilmer(B)
    n1,ell,n3=A.shape
    n2=B.shape[1]
    Cbar = np.zeros((n1,n2,n3),dtype = 'complex_')
    for i in range(n3):
        Cbar[:,:,i]=np.dot(Abar[:,:,i],Bbar[:,:,i])
    C=ifftKilmer(Cbar)
    return np.real(C)


def our_svd(B,k, m, maxIt,tol):
    n1,n2=B.shape
    if n1<n2:
        A=np.dot(B,B.conj().T)
        v1=np.random.rand(n1)
        [Q, H, isC, flag, nc, ni] = KrylovSchur( A,v1, n1, k, m, maxIt,tol)
        H=H.real[:-1,:]
        H=np.round(H)
        H=np.diag(H)
        Q=Q[:,:-1]
        H.flags.writeable = True
        ##################################
        for i in range(k):
            if (H[i]<0):
                H[i]=-H[i]
                Q[:,i]=-Q[:,i]
        ####################################
        # print(H)

        sigma=np.sqrt(H)    
        eig=np.diag(sigma)
        # print(Q.shape,eig.shape)###############################################

        V=np.dot(B.conj().T,np.dot(Q,np.diag(1/sigma)))
        return Q,eig,V
    else:
        A=np.dot(B.conj().T,B)
        v1=np.random.rand(n2)
        [Q, H, isC, flag, nc, ni] = KrylovSchur( A,v1, n2, k, m, maxIt,tol)
        H=H.real[:-1,:]
        H=np.round(H)
        H=np.diag(H)
        Q=Q[:,:-1]
        H.flags.writeable = True
        ##################################
        for i in range(k):
            if (H[i]<0):
                H[i]=-H[i]
                Q[:,i]=-Q[:,i]
        ####################################
        # print(H)
        sigma=np.sqrt(H)    
        eig=np.diag(sigma)
        # print(Q.shape,eig.shape)##################################################
        V=np.dot(B,np.dot(Q,np.diag(1/sigma)))
        return V, eig ,Q           


def scipy_svd(B,k):
    svd = TruncatedSVD(k)
    X_transformed = svd.fit_transform(B)
    U = X_transformed / svd.singular_values_
    Sigma_matrix = np.diag(svd.singular_values_)
    VT = svd.components_
    return U,Sigma_matrix,VT


def tSVD(A,k):
    Abar = fftKilmer(A)
    n1,n2,n3=A.shape
    Ubar=np.zeros((n1,k ,n3),dtype = 'complex_')
    Sbar=np.zeros((k,k ,n3),dtype = 'complex_')
    Vbar=np.zeros((n2,k ,n3),dtype = 'complex_')

    for i in range(Abar.shape[-1]):
        # Ubar[:,:,i],Sbar[:,:,i],Vbar[:,:,i]= scipy_svd(Abar[:,:,i],k)
        Ubar[:,:,i],Sb,Vbar[:,:,i]= torch.svd_lowrank(torch.tensor(Abar[:,:,i]), q=k)
        Sbar[:,:,i]=np.diag(Sb)

    U = ifftKilmer(Ubar)
    S = ifftKilmer(Sbar)
    Vt = ifftKilmer(Vbar)
    return U,S,Vt


def our_tSVD(A,k,m, maxIt,tol):
    Abar = fftKilmer(A)
    n1,n2,n3=A.shape
    Ubar=np.zeros((n1,k ,n3),dtype = 'complex_')
    Sbar=np.zeros((k,k ,n3))
    Vbar=np.zeros((n2,k ,n3),dtype = 'complex_')

    for i in range(Abar.shape[-1]):
        Ubar[:,:,i],Sbar[:,:,i],Vbar[:,:,i]= our_svd(Abar[:,:,i],k, m, maxIt,tol)
    U = ifftKilmer(Ubar)
    S = ifftKilmer(Sbar)
    Vt = ifftKilmer(Vbar)
    return U,S,Vt


def kilmer_transposs(A):
    [n1, n2, n3] = A.shape
    Atrans = np.zeros((n2, n1, n3),dtype = 'complex_')
    Atrans[:, :, 0] = np.transpose(A[:, :, 0])
    for k in range(1, n3):
        Atrans[:, :, k] = np.transpose(A[:, :, n3-k])

    return Atrans