import torch
import numpy
import math
import pickle
from objectiveFunctionRandom import objFunction

class GP:
    def __init__(self, kernal, noise):
        self.kernal = kernal
        self.noise = noise
    
    def fit(self, X, Y):
        '''
        returns nothing, just initializes the weights of the GP
        '''
        with torch.no_grad():
            self.X = X
            self.Y = Y
            self.K = self.kernal(self.X,self.X)
            self.K = self.K + (self.noise*torch.eye(X.shape[0]))
            self.L = torch.linalg.cholesky(self.K)
            #need to unsqeeze to make dim fit, cholesky expects two 2D tensors
            self.a = torch.cholesky_solve(self.Y.unsqueeze(1),self.L)
            #Cholskey is an efficent way to solve stuff if stuff is positive
    
    
    def predict(self, points):
        '''
        returns the values and covarience^2
        '''
        with torch.no_grad():
            # print(self.X.shape,points.shape)
            K_points = self.kernal(self.X, points)
            # print(K_points.shape)
            values = K_points.t() @ self.a
            temp = torch.cholesky_solve(K_points,self.L)
            covarience = self.noise-K_points.t() @ temp
            return values, covarience

def UCB(values, covarience, k):
    '''
    NOTE: the varience is not the squared varience
    '''
    return values + k(covarience)

# dont get it
def expected_improvement(values, varience, best_values,k=0):
    with torch.no_grad():
        improvement = values-best_values-k
        Z = improvement / varience
        Z = torch.nan_to_num(Z)
        ei = improvement * torch.distributions.Normal(0, 1).cdf(Z) + varience * torch.exp(-0.5 * Z**2) / math.sqrt(2 * math.pi)
        return ei

#assume normalized bounds
def bayesian_opt(\
    game, iterations, numSamples, bounds, scaling, unscaling, setparamsDict, \
    changeparams, kernal, guesses=4, aqFunct=0, k=1, victoryCutoff=0.7, noise=0.1, \
    load_save=False):
    '''
    RETURNED: bestArgs, bestValue\n
    aqFunct{0->EI, 1->UCB}\n
    Saves every bayesian iteration, to load from a preivious iteration set load_save to True
    '''
    with torch.no_grad():
        path = str(setparamsDict['directory'])
        if(load_save==False):
            initialArgs = torch.rand(guesses, len(changeparams))
            unscaledArgs = [(unscaling(v.tolist(),bounds)) for v in (initialArgs)]
            print(initialArgs)
            print(unscaledArgs)
            
            setparamsDict['directory'] = setparamsDict['directory']+'Guess'
            
            valuesArgs = []
            for x in unscaledArgs:
                args1 = dict(zip(changeparams,x))
                args1.update(setparamsDict)
                print('args:\n', args1)
                noRandom_args1 = args1.copy()
                noRandom_args1['dirichlet_epsilon']=0
                with torch.enable_grad():
                    valuesArgs.append(objFunction(game=game,args1=noRandom_args1, victoryCutoff=victoryCutoff))
            valuesArgs = torch.tensor(valuesArgs)
            
            bestValue = valuesArgs.max()
            bestArgs = unscaledArgs[valuesArgs.argmax()]
            bestIndex = -1
            
            gp = GP(kernal=kernal, noise=noise)
            start =0
        else:
            f = open("./Data/BayesianSave/save.txt", "rb")
            data = pickle.load(f)
            f.close()
            unscaledArgs = data[0]
            valuesArgs = data[1]
            bestValue = data[2]
            bestArgs = data[3]
            bestIndex = data[4]
            gp = data[5]
            start = data[6]
        for zx in range(start,iterations):
            print('bayesian it: ', zx)
            setparamsDict['directory'] = path+str(zx)
            scaledArgs = torch.stack([torch.tensor(scaling(v,bounds)) for v in (unscaledArgs)])
            gp.fit(scaledArgs, valuesArgs)
            
            #Save stuff for pausing
            #save unscaledArgs, valuesArgs, bestValue, bestArgs, bestIndex, gp, zx
            print('saving...',end='')
            f = open("./Data/BayesianSave/save.txt", "wb")
            stuff = [unscaledArgs, valuesArgs, bestValue, bestArgs, bestIndex, gp, zx]
            pickle.dump(stuff,f)
            f.close()
            print('SAVED')

            # check the dimensions of samples, I dont get how gp.predict works on this
            # for the correct output of a single vector value and varience
            samples = torch.rand(numSamples, len(changeparams))
            values, covarience = gp.predict(samples)
            # unsquare the varience, only take the diagonal part for the varince of the points with themselves
            covarience = torch.diag(covarience).sqrt()
            
            if(aqFunct==0):
                choicesIndex = expected_improvement(values, covarience, best_values=bestValue)
            elif(aqFunct==1):
                choicesIndex = UCB(values, covarience, k)
            choicesIndex = choicesIndex.argmax()
            newArgs = unscaling(samples[choicesIndex].tolist(),bounds)
            args1 = dict(zip(changeparams,newArgs))
            args1.update(setparamsDict)
            
            print(args1)
            noRandom_args1 = args1.copy()
            noRandom_args1['dirichlet_epsilon']=0
            with torch.enable_grad():
                newValue = objFunction(game=game,args1=noRandom_args1, victoryCutoff=victoryCutoff)
            if(newValue>bestValue):
                bestValue = newValue
                bestArgs = newArgs
                bestIndex = zx
            valuesArgs = torch.cat((valuesArgs,torch.tensor([newValue])))
            unscaledArgs.append(newArgs)
            print(unscaledArgs)
        return bestArgs, bestValue, bestIndex

def Matern52(x1, x2, var=1, length=1):
    with torch.no_grad():
        r = torch.cdist(x1,x2)
        # r = torch.linalg.vector_norm(x1-x2)
        return (var)*(1+(5**1/2)*(r/length)+(5)*(r**2)/((3)*(length)))*(torch.exp((-r)*(5**1/2)/length))

def squared_exp(x1, x2, var=1, length=1):
    with torch.no_grad():
        r = torch.cdist(x1,x2)
        # r = torch.linalg.vector_norm(x1-x2)
        return var*(torch.exp((-0.5)*(r**2)/(length**2)))

    