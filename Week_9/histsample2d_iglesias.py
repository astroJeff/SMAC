from numpy import array, arange, argmin, sum, mean, var, size, zeros,	where, histogram
from numpy.random import normal
from matplotlib.pyplot import figure, plot, hist, bar, xlabel, ylabel,title, show, savefig
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = normal(0, 100, 1e4) # Generate n pseudo-random numbers whit(mu,sigma,n)
y = normal(0, 100, 1e4) # Generate n pseudo-random numbers whit(mu,sigma,n)



x_max = max(x)
x_min = min(x)

y_max = max(y)
y_min = min(y)


Nx_MIN = 1   #Minimum number of bins in x (integer)
Nx_MAX = 100  #Maximum number of bins in x (integer)

Ny_MIN = 1 #Minimum number of bins in y (integer)
Ny_MAX = 100  #Maximum number of bins in y (integer)


Nx = arange(Nx_MIN, Nx_MAX) # #of Bins
Ny = arange(Ny_MIN, Ny_MAX) # #of Bins

Dx = (x_max - x_min) / Nx    #Bin size vector
Dy = (y_max - y_min) / Ny    #Bin size vector

Dxy=[]
for i in Dx:    #Bin size vector
    a=[]
    for j in Dy:    #Bin size vector        
        a.append((i,j))    
    Dxy.append(a)        
Dxy=array( Dxy, dtype=[('x', float),('y', float)]) #matrix of bin size vector 


Cxy=zeros(np.shape(Dxy))

Cxy__Dxy_plot=[] #save data to plot in scatterplot x,y,z 


#Computation of the cost function to x and y
for i in xrange(size(Nx)):
    for j in xrange(size(Ny)):
        ki = np.histogram2d(x,y, bins=(Nx[i],Ny[j]))
        ki = ki[0]   #The mean and the variance are simply computed from the event counts in all the bins of the 2-dimensional histogram.
        k = mean(ki) #Mean of event count
        v = var(ki)  #Variance of event count                     
        Cxy[i,j] = (2 * k - v) / ( (Dxy[i,j][0]*Dxy[i,j][1])**2 )  #The cost Function 
        
                            #(Cxy      , Dx          ,  Dy)        
        Cxy__Dxy_plot.append((Cxy[i,j] , Dxy[i,j][0] , Dxy[i,j][1]))#Save result of cost function to scatterplot
 
Cxy__Dxy_plot = np.array( Cxy__Dxy_plot , dtype=[('Cxy', float),('Dx', float), ('Dy', float)])  #Save result of cost function to scatterplot
 
#Optimal Bin Size Selection

#combination of i and j that produces the minimum cost function
idx_min_Cxy=np.where(Cxy == np.min(Cxy)) #get the index of the min Cxy

Cxymin=Cxy[idx_min_Cxy[0][0],idx_min_Cxy[1][0]] #value of the min Cxy

print sum(Cxy==Cxymin) #check if there is only one min value

optDxy=Dxy[idx_min_Cxy[0][0],idx_min_Cxy[1][0]]#get the bins size pairs that produces the minimum cost function

optDx=optDxy[0]
optDy=optDxy[1]

idx_Nx=idx_min_Cxy[0][0]#get the index in x that produces the minimum cost function
idx_Ny=idx_min_Cxy[1][0]#get the index in y that produces the minimum cost function

print '#', Cxymin, Nx[idx_Nx], optDx
print '#', Cxymin, Ny[idx_Ny], optDy


#PLOTS

#plot histogram2d
fig = figure()
H, xedges, yedges = np.histogram2d(x, y,bins=[Nx[idx_Nx],Ny[idx_Ny]])
Hmasked = np.ma.masked_where(H==0,H)
plt.imshow( Hmasked.T,extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ] ,interpolation='nearest',origin='lower',aspect='auto',cmap=plt.cm.Spectral) 
plt.ylabel("y")
plt.xlabel("x")
plt.colorbar().set_label('z') 
plt.show()

#plot scatterplot3d to Dx,Dy and Cxy
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x=Cxy__Dxy_plot['Dx']
y=Cxy__Dxy_plot['Dy']
z =Cxy__Dxy_plot['Cxy']
ax.scatter(x, y, z, c=z, marker='o')

ax.set_xlabel('Dx')
ax.set_ylabel('Dy')
ax.set_zlabel('Cxy')
plt.draw()

ax.scatter( [optDx], [optDy],[Cxymin], marker='v', s=150,c="red")
ax.text(optDx, optDy,Cxymin, "Cxy min", color='red')
plt.draw()
plt.show()





