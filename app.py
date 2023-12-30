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
# we should get ALL the test cases, and then shuffle it lol

def getTestCase(): #converts the mnist data text file, into an array of the inputs, and array of the corresponding
    output = test_data.readline()
    input = test_data.readline()
    input = input[6:]
    pixels = list(map(int,input.split()))

    output = output[6:]
    output = list(map(int,output.split()))
    probabilities = [0]*10
    probabilities[output[0]] = 1
    return [pixels,probabilities]

def relU(x):
    return max(0,x)

def dRelU(x):
    return x>0

def softmax(outputLayer): #returns the probabilities of each digit being the answer
    total = sum(np.exp(outputLayer))
    return np.exp(outputLayer)/total


W1 = np.random.rand(784,32)-0.5 #each of the 32 nodes, takes 784 inputs
B1 = np.random.rand(32,1)-0.5 #each node only has one bias
W2 = np.random.rand(32,32)-0.5
B2 = np.random.rand(32,1)-0.5
W3 = np.random.rand(32,10)-0.5# output layer has 10 nodes, each one taking in 32 inputs from prev layer
B3 = np.random.rand(10,1)-0.5
#input is row, output is col

alpha = 1 #learning rate
epoch = 1000 #learning iterations
#batchSize = 100 # #test cases for stochastic gradient descent

def forward(W1, B1, W2, B2, W3, B3, input):
    Z1 = W1.dot(input) + B1
    A1 = relU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = relU(Z2)
    Z3 = W3.dot(A2) + B3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

#be careful of cost derivative; order matters. node - output
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
        dzk3_dbk3[k] = 1
    
    dC0_daj2 = np.zeros(32)
    for j in range (32):
        for k in range (10):
            dC0_daj2[j]+= 2*(A3[k]-output[k])*dRelU(Z3[k])*W3[j][k]
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
    
    dC0_dai1 = np.zeros((32))
    for i in range (32):
        for j in range (32):
            dC0_dai1[i]+=dC0_daj2[j] * dRelU(Z2[j]) * W2[i][j]
    dai1_dzi1 = np.zeros(32)
    for i in range (32):
        dai1_dzi1 = dRelU(Z1[i])
    dzi1_dwini1 = np.zeros((784,32))
    for l in range (784):
        for i in range (32):
            dzi1_dwini1[l][i] = input[l]
    dzi1_dbi1 = np.zeros(32)
    for i in range (32):
        dzi1_dbi1[i] = 1
    



    return 1


def tweakParams(dW1, dB1, dW2, dB2, dW3, dB3):
    W1 -= alpha*dW1
    B1 -= alpha*dB1
    W2 -= alpha*dW2
    B2 -= alpha*dB2
    W3 -= alpha*dW3
    B3 -= alpha*dB3    
    return  W1,B1,W2,B2,W3,B3

def execTestCase(input, output):
    for i in range (1,epoch+1):
        Z1, A1, Z2, A2, Z3, A3 = forward(W1,B1,W2,B2,W3,B3, input) #from 3b1b, z is before normalization, a is after
        dW1, dB1, dW2, dB2, dW3, dB3 = backward(Z1,A1,Z2,A2,Z3,A3, input, output) #backward prop, to get deriv changes
        W1,B1,W2,B2,W3,B3 = tweakParams(dW1, dB1, dW2, dB2, dW3, dB3)
        





print(getTestCase())

test_data.close()

