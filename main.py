import numpy as np
iris_full = np.genfromtxt('iris.data', delimiter=",",dtype="U75")

class Perceptron(object):
    # train_x n x m matrix representing n input feature vectors of 
    # dimensionality m-1 (1 extra column for the ones -> augmented vector)
    
    # Ws n x m matrix representing m weight vectors of dimensionality 
    # n equal to the dimensionality of feature vectors + 1 (for bias)
    
    def __init__(self,train_x,train_y,learning_rate,n_output):
        self.Ws = np.ones((train_x.shape[1]+1,n_output))
        self.train_x = np.ones((train_x.shape[0],train_x.shape[1]+1))
        self.train_x[:,1:self.train_x.shape[1]] = train_x
        self.train_y = train_y
        self.learning_rate = learning_rate
        
    def output(self,X,Ws):
        return np.dot(X,Ws)    
    
    def train(self):
        delta_ws = []
        for k in range(0,self.Ws.shape[1]):
            ws = self.Ws[:,k]
            delta_w = []
            for i in range(0,self.ws.shape[0]):
                grad = 0
                for j in range(0,self.train_x.shape[0]):
                    x_i = self.train_x[j,i]
                    x = self.train_x[j,:]
                    dotted = np.dot(x, ws) 
                    grad += 2*(dotted-y[k,j])*x_i
                delta_w.append(-1*learning_rate*grad)
            delta_ws.append(delta_w)
            
        for k in range(0,len(delta_ws)):
            for i in range(0,len(delta_ws[k])):
                self.Ws[i,k] += delta_ws[k][i]
            
    
