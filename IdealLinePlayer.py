import random
import numpy
import Cycles

class IdealLinePlayer():
    # grundy = list( {} for i in range(9))
    # grundy[0]['-']=0
    # grundy[1]['>-']=0
    # grundy[2]['<-']=0
    # grundy[3]['->']=0
    # grundy[4]['-<']=0
    # grundy[5]['>->']=1
    # grundy[6]['<-<']=1
    # grundy[7]['<->']=0
    # grundy[8]['>-<']=0
    # grundy[0]['--']=0
    # grundy[1]['>--']=1
    # grundy[2]['<--']=1
    # grundy[3]['-->']=1
    # grundy[4]['--<']=1
    # grundy[5]['>-->']=0
    # grundy[6]['<--<']=0
    # grundy[7]['<-->']=1
    # grundy[8]['>--<']=1
    grundy = {}
    grundy['-']=0
    grundy['>-']=0
    grundy['<-']=0
    grundy['->']=0
    grundy['-<']=0
    grundy['>->']=1
    grundy['<-<']=1
    grundy['<->']=0
    grundy['>-<']=0
    grundy['--']=0
    grundy['>--']=1
    grundy['<--']=1
    grundy['-->']=1
    grundy['--<']=1
    grundy['>-->']=0
    grundy['<--<']=0
    grundy['<-->']=1
    grundy['>--<']=1
    def __init__(self):
        pass
    def mex(self, vector):
        if(len(vector)==0):
            return 0
        vector = numpy.sort(vector)
        if(vector[0]!=0):
            return 0
        print(vector)
        for i in range(1,len(vector)):
            if(vector[i-1]+1!= vector[i] and vector[i -1]!= vector[i]):
                return vector[i -1]+1
        return vector[vector.size-1]+1
    def stateString(self, n):
        cases = [[],[],[],[],[],[],[],[],[]]
        # print(cases)
        string = '>'+('-' * (n-1))
        cases[1].append(self.grundy[string[0:n]])
        cases[5].append(self.grundy[string[0:n]+'>'])
        cases[8].append(self.grundy[string[0:n]+'<'])
        string = '<'+('-' * (n-1))
        cases[2].append(self.grundy[string[0:n]])
        cases[6].append(self.grundy[string[0:n]+'<'])
        cases[7].append(self.grundy[string[0:n]+'>'])
        # print(cases)
        for i in range(1,n-1):
            string=('-'*i)+'>'+('-'*(n-i-1))
            cases[0].append(self.grundy[string[0:i+1]]^self.grundy[string[i:n]])
            # print((self.grundy[string[0:i+1]]^self.grundy[string[i:n]]))
            cases[1].append(self.grundy['>'+string[0:i+1]]^self.grundy[string[i:n]])
            cases[2].append(self.grundy['<'+string[0:i+1]]^self.grundy[string[i:n]])
            cases[3].append(self.grundy[string[0:i+1]]^self.grundy[string[i:n]+'>'])
            cases[4].append(self.grundy[string[0:i+1]]^self.grundy[string[i:n]+'<'])
            cases[5].append(self.grundy['>'+string[0:i+1]]^self.grundy[string[i:n]+'>'])
            cases[6].append(self.grundy['<'+string[0:i+1]]^self.grundy[string[i:n]+'<'])
            cases[7].append(self.grundy['<'+string[0:i+1]]^self.grundy[string[i:n]+'>'])
            cases[8].append(self.grundy['>'+string[0:i+1]]^self.grundy[string[i:n]+'<'])
            string=('-'*i)+'<'+('-'*(n-i-1))
            cases[0].append(self.grundy[string[0:i+1]]^self.grundy[string[i:n]])
            cases[1].append(self.grundy['>'+string[0:i+1]]^self.grundy[string[i:n]])
            cases[2].append(self.grundy['<'+string[0:i+1]]^self.grundy[string[i:n]])
            cases[3].append(self.grundy[string[0:i+1]]^self.grundy[string[i:n]+'>'])
            cases[4].append(self.grundy[string[0:i+1]]^self.grundy[string[i:n]+'<'])
            cases[5].append(self.grundy['>'+string[0:i+1]]^self.grundy[string[i:n]+'>'])
            cases[6].append(self.grundy['<'+string[0:i+1]]^self.grundy[string[i:n]+'<'])
            cases[7].append(self.grundy['<'+string[0:i+1]]^self.grundy[string[i:n]+'>'])
            cases[8].append(self.grundy['>'+string[0:i+1]]^self.grundy[string[i:n]+'<'])
        string=('-'*(n-1))+'>'
        cases[3].append(self.grundy[string[0:n]])
        cases[5].append(self.grundy['>'+string[0:n]])
        cases[7].append(self.grundy['<'+string[0:n]])
        string=('-'*(n-1))+'<'
        cases[4].append(self.grundy[string[0:n]])
        cases[6].append(self.grundy['<'+string[0:n]])
        cases[8].append(self.grundy['>'+string[0:n]])
        string=('-'*(n))
        # print('cases0',cases[0])
        self.grundy[string] = self.mex(cases[0])
        self.grundy['>'+string] = self.mex(cases[1])
        self.grundy['<'+string] = self.mex(cases[2])
        self.grundy[string+'>'] = self.mex(cases[3])
        self.grundy[string+'<'] = self.mex(cases[4])
        self.grundy['>'+string+'>'] = self.mex(cases[5])
        self.grundy['<'+string+'<'] = self.mex(cases[6])
        self.grundy['<'+string+'>'] = self.mex(cases[7])
        self.grundy['>'+string+'<'] = self.mex(cases[8])
        
    def action(self, state):
        n = state.shape[0] #num of edge 1 less
        for i in range(3,n):
            self.stateString(i)
        print(self.grundy)