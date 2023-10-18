import os
import time
import json
import ctypes
import numpy as np
import pandas as pd
import multiprocessing as multi
import matplotlib.pyplot as plt
from case_n1 import CASE
'''
multi-scenario differential evolution
'''
class OPT(CASE):
    def __init__(self):
        super().__init__()
        
    #algorithm parameter
    def set_params(self,**params):
        #general parameter
        self.thread=params['thread_num']
        self.maxiter=params['max_iter']
        self.iter=0
        self.popsize=params['pop_size']
        self.num=params['scenario_num']
        self.index=np.array([[i,j]for i in range(self.num) for j in range(self.popsize)])
        #algorithm parameter
        self.mutaterate=params['mutaterate']
        self.crossrate=params['crossrate']
        self.f=params['weight_factor']
        self.k=params['global_factor']

        self.load_constraint()

    def load_constraint(self):
        #initialize population
        maxmin=pd.read_csv(self.main_path+'range.csv')#bounds
        self.high=np.array(maxmin['high'])
        self.low=np.array(maxmin['low'])
        self.xrange=self.high-self.low
        self.dim=len(self.high)
        self.pop=np.random.uniform(low=self.low,high=self.high,size=(self.popsize,self.dim))
        if not os.path.exists(self.main_path+'record\\'):
            os.mkdir(self.main_path+'record\\')

        print('------------------------------------------------')
        print('range-high:')
        print(self.high)
        print('range-low:')
        print(self.low)
        print('x num:',self.dim)
        print('please ensure the range is correct')

    def pop_encode(self):#normalize
        self.pop=(self.pop-self.low)/self.xrange
        self.nextpop=(self.nextpop-self.low)/self.xrange

    def pop_decode(self):
        self.pop=self.pop*self.xrange+self.low
        self.nextpop=self.nextpop*self.xrange+self.low

    def mutate(self):#polynomial mutation
        for i in range(0,self.popsize):
            r=np.random.random(size=self.dim)
            b=np.zeros([self.dim])
            ind1=np.where(r<=0.5)[0]
            ind2=np.where(r>0.5)[0]
            b[ind1]=(2*r[ind1])**(1/(self.mutaterate+1))-1
            b[ind2]=1-(2*(1-r[ind2]))**(1/(self.mutaterate+1))
            self.nextpop[i]=self.nextpop[i]+b#*self.xrange

            maxi=np.where((self.nextpop[i]>1)==True)[0]
            self.nextpop[i,maxi]=np.random.random(len(maxi))
            lowi=np.where((self.nextpop[i]<0)==True)[0]
            self.nextpop[i,lowi]=np.random.random(len(lowi))
    
    def cross(self):#simulated binary crossover
        for i in range(0,self.popsize,2):
            r=np.random.random(size=self.dim)
            b=np.zeros([self.dim])
            ind1=np.where(r<=0.5)[0]
            ind2=np.where(r>0.5)[0]
            b[ind1]=(2*r[ind1])**(1/(1+self.crossrate))
            b[ind2]=(1/(2-2*r[ind2]))**(1/(1+self.crossrate))
            self.nextpop[i]=0.5*((1+b)*self.pop[i]+(1-b)*self.pop[i+1])
            self.nextpop[i+1]=0.5*((1-b)*self.pop[i]+(1+b)*self.pop[i+1])
            
            maxi=np.where((self.nextpop[i]>1)==True)[0]
            self.nextpop[i,maxi]=np.random.random(len(maxi))
            lowi=np.where((self.nextpop[i]<0)==True)[0]
            self.nextpop[i,lowi]=np.random.random(len(lowi))
            maxi=np.where((self.nextpop[i+1]>1)==True)[0]
            self.nextpop[i+1,maxi]=np.random.random(len(maxi))
            lowi=np.where((self.nextpop[i+1]<0)==True)[0]
            self.nextpop[i+1,lowi]=np.random.random(len(lowi))

    def select(self):#1vs1
        loser=np.where((self.next_eval>self.now_eval)==True)[0]
        self.nextpop[loser]=self.pop[loser]
        self.next_eval[loser]=self.now_eval[loser]

    def step(self):#one step
        self.nextpop=self.pop.copy()
        #cross & mutate
        self.pop_encode()
        self.cross()
        self.mutate()
        self.pop_decode()
        #evaluation
        self.multi_simu(self.index)
        try:
            self.next_eval=self.evaluate()
        except:
            print('------------------------------------------------')
            print('parents replaced dead offsprings')
            ind,ind2=self.check_result()
            self.nextpop[ind]=self.pop[ind]
            self.multi_simu(ind2)
            self.next_eval=self.evaluate()
        #population evolve
        self.select()
        
    def update(self):#update variable
        self.pop=self.nextpop.copy()
        self.now_eval=self.next_eval.copy()

    def run(self):
        start_time=time.time()
        print('------------------------------------------------')
        print('start opt-algo running')
        self.read_scenario()
        self.read_target()
        print('------------------------------------------------')
        print('start init No.0 generation ')
        self.init_pop()
        print('------------------------------------------------')
        print('start evolve')
        self.best_eval=[np.min(self.now_eval)]
        self.best_one=[np.argmin(self.now_eval)]
        self.eval_record=[self.now_eval]

        self.stop_flag=0
        while self.iter<self.maxiter:
            print('------------------------------------------------')
            print('now No.'+str(self.iter+1)+' generation')
            self.iter+=1
            self.step()
            self.stop_flag=self.stop_flag+1 if np.min(self.next_eval)==np.min(self.now_eval) else 0
            self.update()
            self.save()
            print('No.'+str(self.iter)+' best eval score:',np.min(self.now_eval))
            if self.stop_flag>=10:
                break

        end_time=time.time()
        print('------------------------------------------------')
        print(str(self.iter)+' iters finished in time(s):',round(end_time - start_time,3))
        print('opt-algo finished')
        print('------------------------------------------------')

if __name__ == '__main__':
    np.random.seed(2023)
    a=OPT()
    a.set_path('E:\\0-Work\\Data\\4-optm2\\')#end with \\
    a.set_config(csv_list = {
                            "BoilerList": "301.Boiler",
                            "TeesList": "304.TeeJoint",
                            "CrossList": "305.CrossJoint",
                            "PipeList": "306.Pipe",
                            "UserMList": "319.TerminalM"
                            })
    a.set_params(
                thread_num=20,
                scenario_num=1,
                max_iter=9999,
                pop_size=500,

                mutaterate=1,
                crossrate=2,
                weight_factor=0.6,
                global_factor=0.1,
                )
    # a.load('pop_100.csv')
    a.run()
    a.history()