import torch
import numpy
import torch.version
import Cycles
import LineCycleMaker
import IdealLinePlayer
import objectiveFunctionRandom
import BayesianOptimization
import pickle
import MCTSParallelRandom
import SampleGenerator

if __name__ == '__main__':

    # adj_matrix = numpy.array([
    #     [0,1,1,1],
    #     [1,0,1,0],
    #     [1,1,0,1],
    #     [1,0,1,0]
    # ])
    # valid_cycles = numpy.array([
    #     [0,1,2],
    #     [3,2,0]
    # ])
    adj_matrix = numpy.array( [
        [0,1,0,0,0,0,1],
        [1,0,1,0,0,0,1],
        [0,1,0,1,0,1,0],
        [0,0,1,0,1,0,0],
        [0,0,0,1,0,1,0],
        [0,0,1,0,1,0,1],
        [1,1,0,0,0,1,0]
    ] )

    valid_cycles = [
        [0,1,6],
        [6,1,2,5],
        [5,2,3,4]
    ]
    game = Cycles.Cycles(adj_matrix=adj_matrix, valid_cycles=valid_cycles)
    # args = {'lr': (0.1913), 'weight_decay': (0.9423), 'num_resBlocks': 2, 'num_hidden': 16, 'C': (2.3832), 'num_searches': 10, 'num_selfPlay_iterations': 116, 'num_epochs': 3, 'batch_size': 20, 'temperature': (7.4322), 'dirichlet_epsilon': (0.9403), 'dirichlet_alpha': (0.1353), 'num_parallel_games': 80, 'num_iterations': 3, 'check_ai': True, 'directory': './Data/BayesianModels/0'}


    # args = {'lr': (0.0159), 'weight_decay': (0.8363), 'num_resBlocks': 1, 'num_hidden': 28, 'C': (0.9153), 'num_searches': 15, 'num_selfPlay_iterations': 109, 'num_epochs': 1, 'batch_size': 28, 'temperature': (3.7989), 'dirichlet_epsilon': (0.4572), 'dirichlet_alpha': (0.7367), 'num_parallel_games': 116, 'num_iterations': 3, 'check_ai': True, 'directory': './Data/BayesianModels/0'}
    args = {
            'lr':0.002,
            'weight_decay':0.0001,
            'num_resBlocks': 9,
            'num_hidden': 58,
            'C' : 2.7265064418315887,
            'num_searches': 44,
            'num_iterations': 4,
            'num_selfPlay_iterations': 973,
            'num_epochs': 6,
            'batch_size': 37,
            'temperature' : 3.9412047266960144,
            'dirichlet_epsilon': 0.238503098487854,
            'dirichlet_alpha': 0.05893164873123169,
            'num_parallel_games': 128,
            'check_ai':True,
            'directory': "./Data/Manual/B",
        }

    # print(objectiveFunctionRandom.objFunction(game, args, 0.5))

    bounds = [
        [0.0001,0.5,0], # lr
        [0.00001,0.1,0], # weight_decay
        [5,14,1], # num_resBlocks
        [16,128,1], # num_hidden
        [0.5,10,0], # C
        [8,128,1], # num_searches
        [512,1024,1], # num_selfPlay_iterations
        [1,12,1], # num_epochs
        [16,64,1], # batch_size
        [1,10,0], # temperature
        [0,1,0], # dirichlet_epsilon
        [0,1,0], # dirichlet_alpha
    ]
    def scaling(args, bounds):
        newArgs = []
        for i, arg in enumerate(args):
            l = bounds[i][0]; u =bounds[i][1]
            newArgs.append((arg-l)/(u-l))
        return newArgs
    def unscaling(args, bounds):
        newArgs = []
        for i, arg in enumerate(args):
            l = bounds[i][0]; u =bounds[i][1]
            if(bounds[i][2]==1):
                newArgs.append(int((u-l)*(arg)+l))
            else:
                newArgs.append((u-l)*(arg)+l)    
        return newArgs

    f = open("./Data/BayesianSave/save.txt", "rb")
    data = pickle.load(f)
    f.close()
    unscaledArgsFULL = data[0]
    unscaledArgs = unscaledArgsFULL[0:5]
    valuesArgsFULL = data[1]
    valuesArgs = valuesArgsFULL[0:5]
    # print('unscal',unscaledArgs)
    # print()
    # gp = data[5]
    gp = BayesianOptimization.GP(BayesianOptimization.Matern52,0.1)
    scaledArgs = torch.stack([torch.tensor(scaling(v,bounds)) for v in (unscaledArgs)])
    # print('scal',scaledArgs)
    gp.fit(scaledArgs, valuesArgs,device=torch.device('cpu'))

    s = SampleGenerator.SampleGenerator(gp=gp,nthreads_=16)
    values, covarience, samples = s.mc_para(10, len(unscaledArgs[0]))
    print(values)
    print(covarience)
    print(samples)


    # ranPoints= torch.rand(8, len(unscaledArgs[0]))

    # for i in range(len(ranPoints)):
    #     points = ranPoints[i]
    #     # print(points)
    #     # print(gp.X[0])
    #     K_points = gp.kernal((gp.X), points.unsqueeze(0).to(torch.float32))
    #     # print('shape of k points',K_points.shape)
    #     values = K_points.t() @ (gp.a)
    #     temp = torch.cholesky_solve(K_points,(gp.L))
    #     # print('shape of K_points.t',K_points.T.shape,'otherone:',K_points.t().shape,'tempshape:',temp.shape)
    #     covarience = (gp.noise+1)-(K_points.T) @ (temp)
    #     print('before covar of',i,(K_points.T) @ (temp))
    #     print('covar of i:',i,covarience)
    #     print('values of',i,values)
    # batchVal, batchCovar = gp.predict(ranPoints)
    # print('batch covar:',batchCovar)
    # print('batch covar:',torch.diag(batchCovar).sqrt())
    # print('batch values:',batchVal)

    # t = torch.Tensor([[1]])
    # print(t)
    # t = torch.squeeze(t)
    # print(t)

    # print(BayesianOptimization.Matern52(torch.ones((1,1))+13,torch.ones((1,1))+13))

    # r = MCTSParallelRandom.Node(game=game,args={'C':2},state=0)
    # r.visit_count=1
    # n1 = MCTSParallelRandom.Node(game=game,args={'C':2},state=0)
    # n2 = MCTSParallelRandom.Node(game=game,args={'C':2},state=0)
    # n3 = MCTSParallelRandom.Node(game=game,args={'C':2},state=0)
    # n4 = MCTSParallelRandom.Node(game=game,args={'C':2},state=0)
    # n1.prior_input=0.7
    # n2.prior_input=0.2
    # n3.prior_input=0.1
    # n4.prior_input=0.0

    # children = [n1,n2,n3,n4]
    # r.children = children

    # for i in range(100):
    #     print(r.select().prior_input)