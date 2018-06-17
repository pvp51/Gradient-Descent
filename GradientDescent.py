import sys
import math
import random

def dotproduct(x, y):
    if(len(x) == len(y)):
        dp = 0
        for i in range(0, len(x), 1):
            dp += x[i]*y[i]
    return dp

datafile = sys.argv[1]
f = open(datafile)
data = []
i=0
l=f.readline()
################
##Read Data
################
while(l != ''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    l2.append(1)
    data.append(l2)
    l=f.readline()

rows = len(data)
cols = len(data[0])
f.close()

################
##Read Labels
################
labelfile = sys.argv[2]
f = open(labelfile)
trainlabels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()
while(l != ''):
    a = l.split()
    trainlabels[int(a[1])] = int(a[0])
    if(trainlabels[int(a[1])] == 0):
        trainlabels[int(a[1])] = -1
    l = f.readline()
    n[int(a[0])] += 1

############
## Read eta
############
eta = float(sys.argv[3])

##Initialize w 
w=[]
for i in range(0,cols):
    w.append(0.002*random.random()-0.001)

##################
##Gradient descent
##################

#eta = 0.0001
dellf = []
for j in range(0, cols, 1):
    dellf.append(0)

prevObjective = 10000000
obj = prevObjective - 10

while(prevObjective - obj > eta):
    prevObjective = obj
    for j in range(0, cols, 1):
        dellf[j] = 0

    for i in range(0, rows, 1):
        if(trainlabels.get(i) != None):
            dp = dotproduct(w, data[i])
            for j in range(0, cols, 1):
                dellf[j] += (trainlabels.get(i) - dp)*data[i][j]

    for j in range(0, cols, 1):
        w[j] += eta * dellf[j]

    ## Calculating error ##
    error = 0
    for i in range(0, rows, 1):
        if(trainlabels.get(i) != None):
            error += (trainlabels.get(i) - dotproduct(w, data[i]))**2
    obj = error
    print ("Objective is : ", error)

print("w: ", w)
wlength = math.sqrt(w[0]**2 + w[1]**2)
dist_to_origin = abs(w[2])/wlength
print("Distance to origin: ", dist_to_origin)

###########################
## Clasify unlabeled points
###########################

for i in range(0, rows, 1):
    if(trainlabels.get(i) == None):
        dp = dotproduct(w, data[i])
        if (dp < 0):
            print("0 ", i)
        else:
            print("1 ", i)