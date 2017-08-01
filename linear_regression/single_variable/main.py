from csv import reader
import numpy as np
import pdb; 
import matplotlib.pyplot as plt
data = np.loadtxt(open('ex1data1.txt', 'rb'), delimiter = ',')

x = data[:,0]
y = data[:,1]
m = len(y) #number of training examples
print m
iterations = 1500
alpha = 0.01
x= np.hstack((np.ones((97,1)),x.reshape(m,1)))
y = y.reshape(m,1)
theta = np.zeros((1,2))

def compute_cost(x,y,theta):
	predictions = sum((x*theta).transpose()).reshape(m,1)

	sqr_error = (predictions - y)**2

	cost = 1.0/(2.0*m) * sum(sqr_error)
	return cost

def gradientDescent(x,y,theta,alpha,iterations):
	J_history = []
	for i in range(iterations):
		x1 = x[:,1].reshape(m,1)
		theta0 = theta[0][0]
		theta1 = theta[0][1]
		h = theta0 + theta1*x1
		h.reshape(m,1)
		theta0 = theta0 - (alpha * (1/float(m)) *sum(h - y))
		theta1 = theta1 - (alpha * (1/float(m)) *sum(((h-y)*x1)))
		theta[0][0] = theta0
		theta[0][1] = theta1
		cost = compute_cost(x,y,theta)
		J_history.append(cost)
		
	plt.plot(J_history)	
	plt.show()
	return [theta, J_history]

a = gradientDescent(x,y,theta, alpha, iterations)
print(a)


