import torch
import numpy
import torch.version
import Cycles
import LineCycleMaker
import IdealLinePlayer
import objectiveFunctionRandom
import BayesianOptimization
import pickle

from multiprocessing.pool import Pool
from multiprocessing import Lock
from threading import current_thread
from time import sleep
import time
global rng #

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
    [0.0001,1,0], # lr
    [0.00001,1,0], # weight_decay
    [1,3,1], # num_resBlocks
    [16,32,1], # num_hidden
    [0.5,6,0], # C
    [10,20,1], # num_searches
    [64,128,1], # num_selfPlay_iterations
    [1,4,1], # num_epochs
    [16,32,1], # batch_size
    [1,10,0], # temperature
    [0,1,0], # dirichlet_epsilon
    [0,1,0], # dirichlet_alpha
    [64,128,1] # num_parallel_games
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
unscaledArgs = data[0]
valuesArgs = data[1]
bestValue = data[2]
bestArgs = data[3]
bestIndex = data[4]
gp = data[5]



# nthreads_ = 64 # number threads
# # initialize the worker process
# def init_worker(lock):
#     # get the current thread
#     thread = current_thread()
#     # report the name of the current thread
#     with lock:
#         seeds = pickle.load(open("seeds.p", "rb"))
#         s = seeds.pop()
#         #debug
#         info = f"Initializing thread {thread.native_id} :"
#         info = info + f" seed entropy {s.entropy}"
#         info = info + f" and spawn_key {s.spawn_key}."
#         print(info, flush=True)
#         global rng
#         rng = numpy.random.default_rng(s)
#         pickle.dump(seeds, open("seeds.p", "wb"))
# # task executed in a worker process
# def sample(p):
#     c = 0.25
#     n = int(p / c)
#     X = numpy.random.normal(0,1,(p,n))
#     S = X @ X.T / n
#     vals, vecs = eigsh (S, 1, which='LA')
#     h = vecs[:,-1]
#     r = max(h)
#     #debug
#     thread = current_thread()
#     m_seed = rng._bit_generator.seed_seq.entropy
#     t_seed = rng._bit_generator.seed_seq.spawn_key
#     info = f'Thread {thread.native_id}:'
#     info = info + f' seed = {m_seed} -- {t_seed}'
#     info = info + f' | r = {r}'
#     print(info, flush = True)
#     return(r)
# def mc_para(m = 10, d = 100, main_seed=0):
#     seeds = numpy.random.SeedSequence(main_seed)
#     child_seeds = seeds.spawn(nthreads_)
#     pickle.dump(child_seeds, open("seeds.p", "wb"))
#     lock = Lock()
#     pool = Pool(initializer=init_worker, initargs = (lock,), processes = nthreads_)
#     sleep(1)
#     print()
#     # issues task to the process pool
#     result = pool.starmap(sample, zip(numpy.repeat(d,m)))
#     # wait for tasks to complete
#     pool.close()
#     pool.join()
#     # process pool is closed automatically
#     return(result)
# if __name__ == '__main__':
#     #st = time.process_time()
#     m = 10000
#     d = 100
#     st = time.time()
#     result = mc_para(m, d)
#     #et = time.process_time() - st
#     et = time.time() - st
#     print("Multi -core sim. completed.")
#     est = numpy.mean(result)
    
#     st_ = time.time()
#     rng = numpy.random.default_rng(0)
#     ary = numpy.array(list(map(sample, numpy.repeat(d,m))))
#     et_ = time.time() - st_
#     print("Done with sigle core sim")
#     print(f"Estimate = {est}")
#     print(f"Time for multi -core sim. = {et}")
#     print(f"Time for single core sim. = {et_}")
#     print(numpy.mean(ary))


print(torch.__version__)
print('normal gpu way')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if(device.type=='cpu'):
    print('wadwadwad')
st__ = time.time()
samples = torch.rand(100000, len(unscaledArgs[0]),device=device)
print('samples is on',samples.get_device())
values, covarience = gp.predict(samples,device)
et__ = time.time() - st__
print(f'time gpu: {et__}')
print(f'highest value in the function: {values.max()}')


# initialArgs = torch.zeros(4, 13)
# unscaledArgs = [(unscaling(v.tolist(),bounds)) for v in (initialArgs)]
# scaledArgs = torch.stack([torch.tensor(scaling(v,bounds)) for v in (unscaledArgs)])


# values = torch.tensor([.1,.2,.3])
# args = torch.stack([torch.tensor([0.1, 0.1, 6, 67, 1.1, 110, 295, 9, 23, 3.1, 0.1, 0.1, 126]),
#         torch.tensor([0.2, 0.2, 6, 67, 1.2, 110, 295, 9, 23, 3.2, 0.2, 0.2, 126]),
#         torch.tensor([0.3, 0.3, 6, 67, 1.3, 110, 295, 9, 23, 3.3, 0.3, 0.3, 126])])

# gp = BayesianOptimization.GP(kernal=BayesianOptimization.Matern52, noise=0.1)
# gp.fit(args, values)

# samples = torch.rand(1, 13)
# values, covarience = gp.predict(samples)

# print(unscaledArgs)
# print(scaledArgs)
# print(values, covarience)

# # f = open("./Data/BayesianSave/test.txt", "wb")
# # stuff = [unscaledArgs, scaledArgs, gp]
# # pickle.dump(stuff,f)
# # f.close()
# print('check:')
# f = open("./Data/BayesianSave/test.txt", "rb")
# data = pickle.load(f)
# f.close()
# print(data[0])
# print(data[1])
# gpp = data[2]

# vvalues, ccovarience = gpp.predict(samples)
# print(vvalues, ccovarience)

# newValue = BayesianOptimization.objFunction(game=game,args1=args, victoryCutoff=0.5)
# print(newValue)