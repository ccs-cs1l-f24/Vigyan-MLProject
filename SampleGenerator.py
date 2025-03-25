import torch
import numpy
import pickle
import multiprocessing

from multiprocessing.pool import Pool
from multiprocessing import Lock
from threading import current_thread
from time import sleep
import time


#There is inefficent code somewhere, idk where though lol
#Big Shoutout to my Professor Alex Shkolnik for gving the sample code for this file!
class SampleGenerator:
    def __init__(self, gp, nthreads_=16):
        self.gp = gp
        self.nthreads_ = nthreads_
        
    def init_worker(self, lock):
        # get the current thread
        with lock:
            seeds = pickle.load(open("seeds.p", "rb"))
            s = seeds.pop()
            global rng
            rng = numpy.random.default_rng(s)
            pickle.dump(seeds, open("seeds.p", "wb"))
            
    def sample(self, p):
        with torch.no_grad():
            points = numpy.random.uniform(0,1,p)
            points = torch.from_numpy(points)
            K_points = self.gp.kernal((self.gp.X), points.unsqueeze(0).to(torch.float32))
            values = K_points.t() @ (self.gp.a)
            temp = torch.cholesky_solve(K_points,(self.gp.L))
            covarience = (1+self.gp.noise)-(K_points.T) @ (temp)
            return values, covarience, points
    def mc_para(self, m = 10, d = 10, main_seed=0):
        if(multiprocessing.get_start_method()!='forkserver'):
            multiprocessing.set_start_method('forkserver')
        seeds = numpy.random.SeedSequence(main_seed)
        child_seeds = seeds.spawn(self.nthreads_)
        pickle.dump(child_seeds, open("seeds.p", "wb"))
        lock = Lock()
        pool = Pool(initializer=self.init_worker, initargs = (lock,), processes = self.nthreads_)
        sleep(1)
        # print()
        # issues task to the process pool
        valuescovar = pool.starmap(self.sample, zip(numpy.repeat(d,m))) #repeats d, m times
        with torch.no_grad():
            values = torch.stack([torch.squeeze(pair[0]) for pair in valuescovar])
            covar = torch.stack([torch.squeeze(pair[1]) for pair in valuescovar])
            points = torch.stack([torch.squeeze(pair[2]) for pair in valuescovar])
        # wait for tasks to complete
        pool.close()
        pool.join()
        # process pool is closed automatically
        return values,covar,points
