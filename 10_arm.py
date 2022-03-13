import numpy as np
import matplotlib.pyplot as plt

k=10
Q_A= np.random.normal(0,1,k)
reward= np.zeros([1,k]) 
a_max = np.argmax(Q_A)

print(a_max) 


for i in range(len(Q_A)):

    reward= np.random.normal(Q_A[i],1,50)
    axis = np.ones([1,len(reward)])*i

    plt.scatter(axis,reward,marker='.',color='blue')


plt.scatter(range(k), Q_A, color='red')
plt.show()