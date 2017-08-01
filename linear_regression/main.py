import matplotlib.pyplot as plt
from math import sqrt
from random import seed
from random import randrange
from csv 
def load_csv():

	file = open('insurance.csv')
	#convert the csv file into a usable list
	data  = file.read().replace(',', '.').split()[2:]
	x = []
	y =[]
	position = 0
	for i in data:
		if position%2 == 0:
			x.append(float(i))
		else:
			y.append(float(i))
		position = position + 1
	return[x,y]
#calculate the mean of the numbers

def mean(values):
	return sum(values) / float(len(values))
# calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

x = [row[0] for row in dataset]
y = [row[1] for row in dataset]

mean_x, mean_y = mean(x), mean(y)
var_x, var_y = variance(x,mean_x), variance(y, mean_y)

print('x stats: mean = %.3f variance = %.3f' % (mean_x, var_x))
print('y stats: mean = %.3f variance = %.3f' % (mean_y, var_y))


#plt.plot(data()[0],data()[1], 'ro')
#plt.show()

#calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] -mean_x) * (y[i] - mean_y)
	return covar

covar = covariance(x, mean_x,y,mean_y)
print('Covarience: %.3f' % (covar))

#calculate the coefficient 
def coefficient(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean)/variance(x,x_mean)
	b0 = y_mean - (b1*x_mean)
	return [b0,b1]

def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficient(train)
	for row in test:
		yhat = b0 +b1*row[0]
		predictions.append(yhat)
	return predictions
#calculate root mean squired error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += prediction_error**2
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

#evaluate regression algorithm on training data

def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
	print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse

#test simple linear regression
rmse = evaluate_algorithm(dataset, simple_linear_regression)
print('RMSE: %.3f' % (rmse)) 