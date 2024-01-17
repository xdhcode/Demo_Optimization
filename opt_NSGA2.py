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
NSGA-II algorithm is developed baesd on https://github.com/haris989/NSGA-II
'''
class OPT(CASE):
    def __init__(self):
        super().__init__()

    def evaluate(self):#multi-objective function
        self.read_result_json()
        error=abs(self.result-self.target).reshape([self.popsize,self.num,-1])#axis0:individuals axis1:scenarios axis2:sensors
        #axis=1:sensors axis=2:scenarios
        error=np.sum(error,axis=self.func)#2D array
        print('evaluated')
        return -error#max is best
    
    def save(self):
        self.eval_record.append(self.now_eval.tolist())
        pd.DataFrame(self.eval_record).T.to_csv(self.record_path+'pop_eval_record.csv',index=True)
        pd.DataFrame(self.pop).to_csv(self.record_path+'pop_'+str(self.iter)+'.csv',index=True)

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
    
    def popencode(self):##normalize
        self.pop=(self.pop-self.low)/self.xrange
        self.nextpop=(self.nextpop-self.low)/self.xrange

    def popdecode(self):
        self.pop=self.pop*self.xrange+self.low
        self.nextpop=self.nextpop*self.xrange+self.low
    
    def fnds(self,eval):#fast non-dominated sort
        num=len(eval)
        S=[[] for i in range(0,num)]#index of dominated individuals
        front = [[]]#individuals for each front
        n=[0 for i in range(0,num)]#num of dominators
        rank = [0 for i in range(0, num)]#front index of individuals
        for p in range(0,num):#for each individual
            S[p]=[]#individuals dominated by p
            n[p]=0#num of those dominate p
            for q in range(0,num):#compare p with each individual q
                if not (eval[p]==eval[q]).all():
                    if (eval[p]>=eval[q]).all():
                        if q not in S[p]:
                            S[p].append(q)
                    elif (eval[p]<=eval[q]).all():
                        n[p] = n[p] + 1
            if n[p]==0:#if p is not being dominated
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)#then p joins the first front
        i = 0
        while(front[i] != []):#if this front has individuals
            Q=[]#new front
            for p in front[i]:#for each individual in this front
                for q in S[p]:#for each individual dominated by p
                    n[q] =n[q] - 1#when n[q]=0, all individuals dominating q had been called
                    if(n[q]==0):
                        rank[q]=i+1#front rank of q is Fi+1
                        if q not in Q:
                            Q.append(q)
            i = i+1
            front.append(Q)
        #front includes index of individuals
        #rank includes front ranks of individuals
        del front[-1]
        return front

    def crowding_distance(self,front_i):
        front_i=np.array(front_i)
        distance=[0 for i in range(0,len(front_i))]
        distance[0],distance[-1]=999,999

        func_num=self.big_eval.shape[1]
        for i in range(func_num):#for each objective function
            score_front=self.big_eval[front_i,i]
            sorted=front_i[np.argsort(score_front)].tolist()#ascending order
            for k in range(1,len(front_i)-1):
                distance[k] = distance[k]+(
                    self.big_eval[sorted[k+1],i]
                    -self.big_eval[sorted[k-1],i])/(np.max(self.big_eval[:,i])
                                                    -np.min(self.big_eval[:,i]))
        return distance#return crowding distance of current front
    
    def sort_cd(self,front,distance):#sort by crowding distance
        new_order=[]
        for i in range(len(distance)):
            front_i=np.array(front[i])
            newfront=front_i[np.argsort(distance[i])[::-1]].tolist()
            new_order+=newfront
        self.nextpop=self.bigpop[new_order[:self.popsize]]
        self.next_eval=self.big_eval[new_order[:self.popsize]]
    
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
    
    def select(self):
        self.bigpop=np.concatenate([self.pop,self.nextpop],axis=0)
        self.big_eval=np.concatenate([self.now_eval,self.next_eval],axis=0)
        front=self.fnds(self.big_eval)
        distance_list=[]
        for i in range(len(front)):
            distance_list.append(self.crowding_distance(front[i]))
        self.sort_cd(front,distance_list)

    def step(self):#one step
        self.nextpop=self.pop.copy()
        #cross & mutate
        self.popencode()
        self.cross()
        self.mutate()
        self.popdecode()
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
        
    def show(self):
        front=self.fnds(self.next_eval)
        self.front=front[0]
        for i in range(0,len(self.front)):
            plt.plot(-self.next_eval[i],label=str(i))
        plt.xlabel('Functions')
        plt.ylabel('Evaluation')
        plt.legend(loc='upper right')
        plt.show()

    def update(self):#update variable
        self.pop=self.nextpop.copy()
        self.now_eval=self.next_eval.copy()

    def run(self):
        start_time=time.time()
        print('------------------------------------------------')
        print('start opt-algo running')
        self.read_snenario()
        self.read_target()
        print('------------------------------------------------')
        print('start init No.0 generation ')
        self.init_pop()
        print('------------------------------------------------')
        print('start evolve')
        self.eval_record=[self.now_eval.tolist()]

        while self.iter<self.maxiter:
            print('------------------------------------------------')
            print('now No.'+str(self.iter+1)+' generation')
            self.iter+=1
            self.step()
            self.update()
            self.save()
            print('No.'+str(self.iter)+' best eval score:',np.max(np.sum(self.big_eval,axis=1)))

        end_time=time.time()
        print('------------------------------------------------')
        print(str(self.maxiter)+' iters finished in time(s):',round(end_time - start_time,3))
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
                scenario_num=3,
                max_iter=9999,
                pop_size=500,

                mutaterate=1,
                crossrate=2
                )
    # a.load('pop_100.csv')
    a.run()
    a.history()
    a.show()