import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import *
from numpy.linalg import *
import pandas as pa

# --- Generowanie danych ---
A = randint(-200 , 100, [6, 3]) 
W = randn(A.shape[1],A.shape[0])
B = randint(-200 , 100 , [1 , A.shape[1]])

def err ( W , A ,  i ):  # funkcja błędu 
    I = np.eye(A.shape[0] ,A.shape[1])
    return I [i , : ] - W @ A[: , i ] 

zm = W 
norma = []
while norm(err(zm ,A ,min(A.shape[0],A.shape[1])-1) , 1) > 0.000001 :
    for itr in range(min(A.shape[0],A.shape[1])):
        alfa = 0.00001
        blad = err ( zm , A ,  itr ).reshape([A.shape[1],1])
        X  = A[:,itr].reshape([1,A.shape[0]])
        zm = zm + alfa*(blad@X)
        norma.append(norm(zm.T@(B.T),2))
        
t = range(len(norma))
plt.plot(t,norma )

print(zm)

