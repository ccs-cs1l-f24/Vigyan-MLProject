#adj matrix, cycles matrix
import numpy
def LineGraph(n):
    adj = numpy.zeros((n,n))
    index=0
    for i in range(1,n-1):
        adj[i][i-1]=1
        adj[i][i+1]=1
    adj[0][1]=1
    adj[n-1][n-2]=1
    
    return adj, numpy.array([])
def CircleGraph(n):
    adj, _ = LineGraph(n)
    adj[0][n-1]=1
    adj[n-1][0]=1
    valid = numpy.zeros(n)
    for i in range(n):
        valid[i]=i
    return adj, valid
def GraphGraph(n):
    pass