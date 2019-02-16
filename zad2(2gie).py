import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import csv

BETA = 0.15
MOMENTUM = 0.005
ALFA = 0.01

def m_fun(x1, x2):
    return np.sin(2*x1 + x2)

number = 11

x1 = np.linspace(-3.0, 3.0, num=number)
x2 = np.linspace(-1.0, 3.0, num=number)

X = np.stack(np.meshgrid(x1, x2), -1).reshape(-1,2)

y = m_fun(X[:,0], X[:,1])
y = y.reshape((X.shape[0],1))

def sigm(x):
    return (1-np.exp(-BETA*x))/(1+np.exp(-BETA*x))

def der(x):
    return BETA*(1 - (x ** 2))
    
def run(X, wages_0, wages_1):
    l0 = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
    l1 = sigm(np.dot(l0,wages_0))
    l1 = np.concatenate((l1, np.ones((l1.shape[0],1))), axis=1)
    l2 = sigm(np.dot(l1,wages_1))
    return(l2)

np.random.seed(1)

hid = 5

wages_0 = 2*np.random.random((X.shape[1] + 1, hid)) - 1
wages_1 = 2*np.random.random((hid + 1,1)) - 1

errs = []

serrs = []
l0 = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)

l1_old_delta = 0
l2_old_delta = 0

for epok in range(100000):

    l1 = sigm(np.dot(l0,wages_0))
    l1 = np.concatenate((l1, np.ones((l1.shape[0],1))), axis=1)
    l2 = sigm(np.dot(l1,wages_1))

    l2_error = y - l2

    l2_delta = l2_error*der(l2)
    l1_error = l2_delta.dot(wages_1.T)
    
    l1_delta = l1_error * der(l1)

    d1 = l1.T.dot(l2_delta)
    d2 = l0.T.dot(l1_delta)[:,:-1]

    wages_1 += ALFA * d1
    wages_0 += ALFA * d2

    wages_1 += MOMENTUM * l2_old_delta
    wages_0 += MOMENTUM * l1_old_delta

    l1_old_delta = d2
    l2_old_delta = d1

    errs.append(np.mean(np.abs(l2_error)))

    if epok % 10000 == 0:
        print(str(epok) + " : błąd = " + str(errs[-1]) + ", delta_w = " + str(np.min(d1)))
        serrs.append([epok, errs[-1]])
    if epok > 10:
        if abs(np.min(d1)) < 0.00001:
            break
"""
with open('bledy.csv', 'w') as csvfile:
    fieldnames = ['Epoki', 'Błedy']
    file = csv.DictWriter(csvfile, fieldnames=fieldnames)

    file.writeheader()
    for e in serrs:
        file.writerow({fieldnames[0] : e[0], fieldnames[1] : e[1]})
"""
errs = errs[1:]

plt.plot(errs)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

nx1 = np.linspace(-3.0, 3.0, num=121)
nx2 = np.linspace(-1.0, 3.0, num=121)

mX, mY = np.meshgrid(nx1 , nx2)
Z1 = m_fun(mX, mY)
Z2 = m_fun(mX, mY)

for i in range(121):
    for j in range(121):
        Z2[i,j] = run(np.vstack((nx1[j], nx2[i])).T, wages_0, wages_1)

#ploting
x1 = np.linspace(-3 ,3, 50)
y1 = np.linspace(-1 ,3, 50)
x1, y1 = np.meshgrid(x1,y1)
z =  np.sin(2*x1 + y1)

H = Axes3D(plt.figure(1))

H.plot_surface(x1 , y1, z , cmap = 'Greens' ,alpha=0.3)
H.plot_surface(mX, mY, Z2 , cmap = 'Blues' ,alpha=0.5)
plt.show()
        

