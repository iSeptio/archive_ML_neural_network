import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import *
from numpy.linalg import *
import pandas as pa
# --- zad 2 --- #
def ss(x1 ,x2):# Funkcja 
    return np.sin(2*x1 + x2)

class Neuron(object): 
    
    def __init__(self):
        
        self.Wejscie = 2 #ilość wejść
        self.Wyjscie = 1 # ilość wyjść
        self.Ukryte  = 5 # ilość neuronów w warstwie ukrytej 
        
        self.waga_1 = randn(self.Ukryte,self.Wejscie) # WAGI  
        self.waga_2 = randn(self.Wyjscie,self.Ukryte) # WAGI
        
        self.b_1 = np.zeros((self.Ukryte,1)) # WAGI
        self.b_2 = np.zeros((self.Wyjscie,1)) # WAGI
    
    def sigmoida(self,x): # funkcja sigmoidalna
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def impuls(self,x): # sieć naturalna 
        
        self.imp_1 = self.waga_1@x+self.b_1 # wyliczamy wartość dla poszczególnych neuronów w warstwie ukrytej 
        self.war_1 = self.sigmoida(self.imp_1) # obliczana jest syigmoida dla  neuronu w warstwie ukrytej 
        
        self.imp_2 = self.waga_2@self.war_1+self.b_2 # obliczana jest wartość wyjścia Y 
        self.war_2 = self.sigmoida(self.imp_2) # obliczana jest syigmoida dla wyjścia Y 
        
        return self.war_2
    
    def wsteczna_propagacja(self,x,y): # algorytm wstecznej propagacji 
        
        self.L = x.shape[1]
        self.err_2 = self.impuls(x)-y
        self.w_2 = self.err_2@self.war_1.T*(1/self.L)
        self.b_02 =np.sum(self.err_2,axis=1)*(1/self.L)
        
        self.err_1 = np.multiply(self.waga_2.T@self.err_2,(1-np.power(self.war_1,2)))
        self.w_1 = self.err_1@x.T*(1/self.L)
        self.b_01 = np.sum(self.err_1 , axis=1)*(1/self.L)
        
        return self.w_1, self.w_2 ,self.b_01 ,self.b_02
    
    def aktualizacja_wag(self,x,y): # aktualizacja wag 
        
        w1,w2,b1,b2 = self.wsteczna_propagacja(x,y)
        
        self.waga_1 += -0.5*w1
        self.waga_2 += -0.5*w2
        self.b_1 += -0.5*b1
        self.b_2 += -0.5*b2

# --- Dane ---
X = [] 
X.append(rand(100)*6 - 3)
X.append(rand(100)*4 - 1)
X = np.matrix(X)
Y = np.sin(2 * X[0] + X[1])

# --- Uczenie sieci na 100000 iteracji ---
N= Neuron()  
blad= []
for i in range(10000):
    blad.append(np.sum(((N.impuls(X)-Y)@(N.impuls(X)-Y).T)))
    N.aktualizacja_wag(X,Y)

plt.plot(blad)

x1 = np.linspace(-3 ,3, 50)
y1 = np.linspace(-1 ,3, 50)
x1, y1 = np.meshgrid(x1,y1)
z =  np.sin(2*x1 + y1)to 

H = Axes3D(plt.figure(1))

H.plot_surface(x1 , y1, z , cmap = 'Greens' ,alpha=0.8)
H.scatter(X[0],X[1],N.impuls(X),c='red',marker='^')
plt.show()
        
    
    