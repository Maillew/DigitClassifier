#have an activation function, so we arent just spamming linear combinations of input function
#relU is more impactful than sigmoid, according to PhD lady
#for the final output layer, we use a softmax function, so that we are able to convert numeric values to probabilities
#soft max makes it so that "negative" values make sense; not a negative probability, just a rly rly small probability instead

#making our neural network "learn"
#back propagation is like chain rule going backwards, to find out how much of the previous inputs affected our output
#our "correct" output should be an array of all 0s, and then one 1 which is the correct digit
#make sure to get derivative of activation function too

#have a learning rate; how much we should adjust the values, based on derivative
# new = old - learning rate * derivative

#repeat over and over



#backpropagation in detail:

"""
for a single example:
there are some outputs that will be close, and some that are more inaccurate; we want to adjust the inaccurate ones more, and the 
more accurate ones less

to adjust the value of a node, we can adjust the things that feed into said node;
1. weights that scale the activation inputs
2. activation inputs themselves (previous nodes)
3. bias

Bias
- simple enough, we are just changing the threshold

Weights
- to get the most change, we change the weights that are associated with the LARGEST activation inputs

Activation
- to get the most change, we change the activation values, associated with the LARGEST weights
- note that we cant directly change the activation, since we can only change weight and biases
- instead, note these changes down, and then apply them recursively backwards later

repeat this process across ALL test data, to get an "average" change that we need to apply to the nodes
recurse backwards
- small details are left as Stochastic Gradient Descent


GETTING INTO THE MATHEMATICSSSSS
- read the 3b1b blog lol

"""

"""
784 input nodes, 10 output nodes
2 hidden layers, 64 nodes each
cost of example is sum of all squared differences of output layer
cost of a data set is average cost of all examples in the data set

for SGD, compute the average change for each output node, and then use this avg to backprop backwards

"""
import numpy as np

test_data = open("mnist_test.txt")
validation_data = open("validation_tests.txt")
# we should get ALL the test cases, and then shuffle it lol

def getTestCase(): #converts the mnist data text file, into an array of the inputs, and array of the corresponding
    output = test_data.readline()
    input = test_data.readline()
    input = input[6:]
    pixels = np.array(list(map(int,input.split())))

    output = output[6:]
    output = list(map(int,output.split()))
    probabilities = np.zeros(10)
    probabilities[output[0]] = 1
    return pixels,probabilities

def getVTestCase(): #converts the mnist data text file, into an array of the inputs, and array of the corresponding
    output = validation_data.readline()
    input = validation_data.readline()
    input = input[6:]
    pixels = np.array(list(map(int,input.split())))

    output = output[6:]
    output = list(map(int,output.split()))
    probabilities = np.zeros(10)
    probabilities[output[0]] = 1
    return pixels,probabilities

def relU(x):
    return 1.0/(1.0+np.exp(-x))

def dRelU(x):
    return relU(x)*(1-relU(x))

def softmax(outputLayer): #returns the probabilities of each digit being the answer
    total = sum(np.exp(outputLayer))
    return np.exp(outputLayer)/total


def forward(W1, B1, W2, B2, W3, B3, input): #still need to fix this lolsies
    Z1 = np.zeros(32)
    A1 = np.zeros(32)
    for j in range (32):
        for i in range (784):
            Z1[j] += W1[i][j]*input[i]
        Z1[j]+=B1[j]
        A1[j] = relU(Z1[j])

    Z2 = np.zeros(32)
    A2 = np.zeros(32)
    for j in range (32):
        for i in range (32):
            Z2[j] += W2[i][j]*A1[i]
        Z2[j]+=B2[j]
        A2[j] = relU(Z2[j])

    Z3 = np.zeros(10)
    A3 = np.zeros(10)
    for j in range (10):
        for i in range (32):
            Z3[j] += W3[i][j]*A2[i]
        Z3[j]+=B3[j]
        A3[j] = Z3[j]
    A3 = softmax(A3)
    return Z1, A1, Z2, A2, Z3, A3

#be careful of cost derivative; order matters. node - output
#problem is rn is that we overflow?? and that produces an error in the derivative
#try sigmoid??

def backward(Z1,A1,Z2,A2,Z3,A3, input, output):
    dC0_dak3 = np.zeros(10)
    for i in range (10):
        dC0_dak3[i] = 2 * (A3[i]-output[i])
    dak3_dzk3 = np.zeros(10)
    for i in range (10):
        dak3_dzk3[i] = dRelU(Z3[i])
    dzk3_dwjk3 = np.zeros((32,10))
    for j in range (32):
        for k in range (10):
            dzk3_dwjk3[j][k] = A2[j]
    dzk3_dbk3 = np.zeros(10)
    for k in range(10):
        dzk3_dbk3[k] = 1.0

    dwjk3 = np.zeros((32,10))
    dbk3 = np.zeros(10)
    for j in range (32):
        for k in range (10):
            dwjk3[j][k] += dC0_dak3[k]*dak3_dzk3[k]*dzk3_dwjk3[j][k]
            dbk3[k] = dC0_dak3[k]*dak3_dzk3[k]*dzk3_dbk3[k]


    dC0_daj2 = np.zeros(32)
    for j in range (32):
        for k in range (10):
            dC0_daj2[j]+= 2*(A3[k]-output[k])*dRelU(Z3[k])*iW3[j][k]
    daj2_dzj2 = np.zeros(32)
    for j in range (32):
        daj2_dzj2[j] = dRelU(Z2[j])
    dzj2_dwij2 = np.zeros((32,32))
    for i in range (32):
        for j in range (32):
            dzj2_dwij2[i][j] = A1[i]
    dzj2_dbj2 = np.zeros(32)
    for j in range (32):
        dzj2_dbj2[j] = 1
    
    dwij2 = np.zeros((32,32))
    dbj2 = np.zeros(32)
    for i in range (32):
        for j in range (32):
            dwij2[i][j] += dC0_daj2[j]*daj2_dzj2[j]*dzj2_dwij2[i][j]
            dbj2[j] = dC0_daj2[j]*daj2_dzj2[j]*dzj2_dbj2[j]
            

    dC0_dai1 = np.zeros(32)
    for i in range (32):
        for j in range (32):
            dC0_dai1[i]+=dC0_daj2[j] * dRelU(Z2[j]) * iW2[i][j]
    dai1_dzi1 = np.zeros(32)
    for i in range (32):
        dai1_dzi1[i] = dRelU(Z1[i])
    dzi1_dwini1 = np.zeros((784,32))
    for l in range (784):
        for i in range (32):
            dzi1_dwini1[l][i] = input[l]
    dzi1_dbi1 = np.zeros(32)
    for i in range (32):
        dzi1_dbi1[i] = 1
    
    dwini1 = np.zeros((784,32))
    dbi1 = np.zeros(32)
    for l in range (784):
        for i in range (32):
            dwini1[l][i] += dC0_dai1[i]*dai1_dzi1[i]*dzi1_dwini1[l][i]
            dbi1[i] = dC0_dai1[i]*dai1_dzi1[i]*dzi1_dbi1[i]
            
    
    return dwini1, dbi1, dwij2, dbj2, dwjk3, dbk3


def tweakParams(dW1, dB1, dW2, dB2, dW3, dB3):
    global iW1, iB1, iW2, iB2, iW3, iB3
    iW1 -= alpha*dW1
    iB1 -= alpha*dB1
    iW2 -= alpha*dW2
    iB2 -= alpha*dB2
    iW3 -= alpha*dW3
    iB3 -= alpha*dB3    
    return  iW1,iB1,iW2,iB2,iW3,iB3

def getDigit(output):
    mx,v = 0,0
    for i in range(10):
        if output[i]>mx:
            mx = output[i]
            v = i
    return v
def getCost(A3, output):
    sum = 0.0
    for i in range (10):
        sum += (A3[i]-output[i])**2
    return sum

alpha = 0.1 #learning rate
epoch = 60000 #learning iterations; training set has 60k, testing set has 10k
validationEpoch = 10000
#batchSize = 100 # #test cases for stochastic gradient descent
iW1 = np.random.rand(784,32)-0.5 #each of the 32 nodes, takes 784 inputs
iB1 = np.random.rand(32)-0.5 #each node only has one bias
iW2 = np.random.rand(32,32)-0.5
iB2 = np.random.rand(32)-0.5
iW3 = np.random.rand(32,10)-0.5# output layer has 10 nodes, each one taking in 32 inputs from prev layer
iB3 = np.random.rand(10)-0.5
#input is row, output is col

def init_GD(): #print the cost function
    global iW1, iB1, iW2, iB2, iW3, iB3
    for i in range (1,epoch+1):
        input,output = getTestCase()
        input = input.astype(float)
        input/=255.0
        Z1, A1, Z2, A2, Z3, A3 = forward(iW1,iB1,iW2,iB2,iW3,iB3, input) #from 3b1b, z is before normalization, a is after
        dW1, dB1, dW2, dB2, dW3, dB3 = backward(Z1,A1,Z2,A2,Z3,A3, input, output) #backward prop, to get deriv changes
        iW1,iB1,iW2,iB2,iW3,iB3 = tweakParams(dW1, dB1, dW2, dB2, dW3, dB3)
        #print("forward: ", Z1, A1, Z2, A2, Z3, A3)
        # print(i)
        # print("dW1:", np.max(dW1), np.min(dW1)) #derivatives are always 0 for some reason wtff
        # print("dB1:", np.max(dB1), np.min(dB1))
        # print("dW2:", np.max(dW2), np.min(dW2))
        # print("dB2:", np.max(dB2), np.min(dB2))
        # print("dW3:", np.max(dW3), np.min(dW3))
        # print("dB3:", np.max(dB3), np.min(dB3))
        #print("weights: ", iW1,iB1,iW2,iB2,iW3,iB3)
        #print(getCost(A3,output)) #cost of 0.9 is pure random

def run_validation():
    correct = 0 #how many do we correctly identify
    global iW1, iB1, iW2, iB2, iW3, iB3
    for i in range (1, validationEpoch+1):
        input,output = getVTestCase() #need to make a second get test case for validation data
        input = input.astype(float)
        input/=255.0
        Z1, A1, Z2, A2, Z3, A3 = forward(iW1,iB1,iW2,iB2,iW3,iB3, input) #from 3b1b, z is before normalization, a is after
        if output[getDigit(softmax(A3))] == 1:
            correct+=1
    print(correct)
    print(correct/validationEpoch)

init_GD()
run_validation()
# 951/1000
# 9332/10000
# 9338/10000


#consider shuffling the training test data? 

test_data.close()
