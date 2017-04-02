from numpy import array, arange, argmin, sum, mean, var, size, zeros,\
	where, histogram
from numpy.random import normal
from matplotlib.pyplot import figure, plot, hist, bar, xlabel, ylabel,\
    title, show, savefig

# Source: http://www.neuralengine.org//res/histogram.html
# Based on these publications:
# Shimazaki H. and Shinomoto S., A method for selecting the bin size of a time histogram Neural Computation (2007) Vol. 19(6), 1503-1527
# Shimazaki H. and Shinomoto S., A recipe for optimizing a time-histogram Neural Information Processing Systems (2007) Vol. 19, 1289-1296
# pdf versions available online (at previous link)


#x = normal(0, 100, 1e2) # Generate n pseudo-random numbers whit(mu,sigma,n)
x = [4.37,3.87,4.00,4.03,3.50,4.08,2.25,4.70,1.73,4.93,1.73,4.62,\
3.43,4.25,1.68,3.92,3.68,3.10,4.03,1.77,4.08,1.75,3.20,1.85,\
4.62,1.97,4.50,3.92,4.35,2.33,3.83,1.88,4.60,1.80,4.73,1.77,\
4.57,1.85,3.52,4.00,3.70,3.72,4.25,3.58,3.80,3.77,3.75,2.50,\
4.50,4.10,3.70,3.80,3.43,4.00,2.27,4.40,4.05,4.25,3.33,2.00,\
4.33,2.93,4.58,1.90,3.58,3.73,3.73,1.82,4.63,3.50,4.00,3.67,\
1.67,4.60,1.67,4.00,1.80,4.42,1.90,4.63,2.93,3.50,1.97,4.28,\
1.83,4.13,1.83,4.65,4.20,3.93,4.33,1.83,4.53,2.03,4.18,4.43,\
4.07,4.13,3.95,4.10,2.27,4.58,1.90,4.50,1.95,4.83,4.12]

x_max = max(x)
x_min = min(x)
N_MIN = 4   #Minimum number of bins (integer)
N_MAX = 50  #Maximum number of bins (integer)
N = arange(N_MIN, N_MAX) # #of Bins
D = (x_max - x_min) / N    #Bin size vector
C = zeros(shape=(size(D), 1))
C = zeros(size(D))

#Computation of the cost function
for i in xrange(size(N)):
	ki = histogram(x, bins=N[i])
	ki = ki[0]
	k = mean(ki) #Mean of event count
	v = var(ki)  #Variance of event count
	C[i] = (2 * k - v) / (D[i]**2) #The cost Function
#Optimal Bin Size Selection

idx = argmin(C)
cmin = C[idx]
optD = D[idx]

print '#', cmin, N[idx], optD

fig = figure()
if 0: # Two ways of plotting histogram.
	# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist
	hist(x, bins=N[idx])
else:
	# http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
	counts, bins = histogram(x, bins=N[idx])
	bar(bins[:-1], counts, width=optD)
title("Histogram")
ylabel("Frequency")
xlabel("Value")
show()#savefig('Hist.png')         
fig = figure()
plot(D, C, '.b', optD, cmin, '*r')
show()#savefig('Fobj.png')

