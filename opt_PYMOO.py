import os
import time
import json
import ctypes
import numpy as np
import pandas as pd
import multiprocessing as multi
import matplotlib.pyplot as plt
from json_tool import JST

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import Problem

class OPT(Problem,JST):
    def __init__(self):
        super().__init__()
        JST.__init__(self)
        np.random.seed(2023)

     #hydraulic simulation
    def set_path(self,path):#network path
        self.main_path=path
        self.csv_path=self.main_path+'netcsv_1\\'
        self.record_path=self.main_path+'record\\'

    def set_config(self,csv_list):#network config
        self.csv_list=csv_list
        self.csv_json=self.csv2json()
        self.config_json=json.dumps(self.config).encode()
        self.result_json=multi.Manager().dict()

    def simu(self,index):#single simulation
        s_ind=index[0]#scenario index
        i_ind=index[1]#individual index
        #set the scenario
        csv=self.csv_json.copy()
        scenario1=self.scenario1[s_ind].tolist()
        scenario2=self.scenario2[s_ind].tolist()
        csv['UserMList'] = list(map(lambda dict, value: {**dict, 'm': value}, csv['UserMList'], scenario1))
        csv['BoilerList'] = list(map(lambda dict, value: {**dict, 'm_1': value}, csv['BoilerList'], scenario2))
        csv['BoilerList'] = list(map(lambda dict, value: {**dict, 'm_2': value}, csv['BoilerList'], scenario2))
        #set roughness of pipelines
        optparam=(self.nextpop[i_ind]*1e-6).tolist()#unit:micrometer to meter
        csv['PipeList'] = list(map(lambda dict, value: {**dict, 'roughness': value}, csv['PipeList'], optparam))
        #call hydraulic simulation dll
        simulator = ctypes.WinDLL(self.main_path+"Systemr32.dll")
        simulator.Calculate.restype = ctypes.c_char_p
        simu_info = eval(simulator.Calculate(json.dumps(csv).encode(),self.config_json))
        #record results
        if simu_info['state'] == 1:
            self.result_json[str(s_ind)+'_'+str(i_ind)]=self.get_result_json(simu_info)
        else:
            self.result_json[str(s_ind)+'_'+str(i_ind)]='dead'
            print('------------------------------------------------')
            print(str(index)+" Error")
            if self.iter!=0:
                print('iter '+str(self.iter)+':'+str(index)+' dead')
                print('------------------------------------------------')

    def multi_simu(self,ind):#multi-thread simulation
        time1=time.time()
        print('multi simu start, pooling')
        try:
            pool = multi.Pool(processes=self.thread)
            print('multi simu running, relax~')
            pool.map(self.simu,ind)
            pool.close()
            pool.join()
        except Exception as e:
            print('------------------------------------------------')
            print('pool error:', e)
            print('------------------------------------------------')
        finally:
            time2=time.time()
            print("multi simu finished in:",round(time2-time1,3))

    def get_result_json(self,input):
        temp1,temp2=[],[]
        for i in range(len(input['UserMList'])):
            temp1.append(input['UserMList'][i]['p_1'])
            temp2.append(input['UserMList'][i]['p_2'])
        return temp1+temp2
 
    #objective function
    def read_scenario(self):#read scenarios
        cat1=pd.DataFrame()#heat station
        cat2=pd.DataFrame()#heat source
        for i in range(self.num):
            userm=pd.read_csv(self.main_path+'netcsv_'+str(i+1)+'\\319.TerminalM.csv',usecols=[6]).T
            cat1=pd.concat([cat1,userm],axis=0)
            boilm=pd.read_csv(self.main_path+'netcsv_'+str(i+1)+'\\301.Boiler.csv',usecols=[6]).T
            cat2=pd.concat([cat2,boilm],axis=0)
        self.scenario1=cat1.values#2D array, scenario for each row
        self.scenario2=cat2.values

    def read_target(self):#read true values
        cat=pd.DataFrame()
        for i in range(self.num):
            userp1=pd.read_csv(self.main_path+'target_'+str(i+1)+'\\319.TerminalMResult.csv',usecols=[2]).T.reset_index(drop=True)
            userp2=pd.read_csv(self.main_path+'target_'+str(i+1)+'\\319.TerminalMResult.csv',usecols=[6]).T.reset_index(drop=True)
            cat=pd.concat([cat,userp1,userp2],axis=1)
        self.target=cat.values#1D array
    
    def read_result_json(self):
        temp1=[]
        for j in range(self.popsize):
            temp2=[]
            for i in range(self.num):
                temp2+=self.result_json[str(i)+'_'+str(j)]
            temp1.append(temp2)
        self.result=np.array(temp1)

    def check_result(self):#check dead individuals
        ind=[]#individual index
        for j in range(self.popsize):
            for i in range(self.num):
                if self.result_json[str(i)+'_'+str(j)]=='dead':
                    ind.append(j)
        if len(ind)!=0:
            ind=np.unique(ind)#1D array
        ind2=np.array([[i,j]for i in range(self.num) for j in ind])
        print('checked result')
        return ind,ind2#index of dead

    def evaluate1(self):#objective function
        self.read_result_json()
        error=np.sum(abs(self.result-self.target),axis=1)#abs error
        print('evaluated')
        return error#1D array

    def save(self):#save population and evaluation
        self.eval_record.append(self.eval)
        self.best_eval.append(np.min(self.eval))
        self.best_one.append(np.argmin(self.eval))
        info=pd.DataFrame([self.best_one,self.best_eval]).T
        info.columns=['best_one','best_eval']
        info.to_csv(self.record_path+'pop_best.csv',index=True)
        pd.DataFrame(self.eval_record).T.to_csv(self.record_path+'pop_eval_record.csv',index=True)
        pd.DataFrame(self.pop).to_csv(self.record_path+'pop_'+str(self.iter)+'.csv',index=True)
    
    #algorithm parameter
    def set_params(self,**params):
        #general parameter
        self.thread=params['thread_num']
        self.num=params['scenario_num']
        self.mode=params['muilti_thread']
        self.load_constraint()

    def load_constraint(self):
        #initialize population
        maxmin=pd.read_csv(self.main_path+'range.csv')#bounds
        self.high=np.array(maxmin['high']).astype(float)
        self.low=np.array(maxmin['low']).astype(float)
        self.xrange=self.high-self.low
        self.dim=len(self.high)
        if not os.path.exists(self.main_path+'record\\'):
            os.mkdir(self.main_path+'record\\')
        print(self.high)
        print(self.low)
        weight=np.array(maxmin['valid']).ravel()
        self.keyindex=np.where(weight>=0.01)[0]
        self.initx=np.array(maxmin['init']).astype(float)

        self.n_var=len(self.keyindex)
        self.xu=self.high[self.keyindex]
        self.xl=self.low[self.keyindex]
        self.n_obj=1
        self.n_ieq_constr=0

        self.read_scenario()
        self.read_target()
        self.best_eval=[]
        self.best_one=[]
        self.eval_record=[]
        self.iter=0 if self.mode else -1
        
        self.bigx=[]
        self.bigy=[]
        self.bigz=[]

    def _evaluate(self, x, out, *args, **kwargs):
        if self.mode:#multi-thread
            print('------------------------------------------------')
            print('now No.'+str(self.iter)+' generation')
            self.popsize=len(x)
            self.pop=np.ones([self.popsize,self.dim])*self.initx
            self.pop[:,self.keyindex]=x.copy()
            print('popsize:',x.shape)
            index=np.array([[i,j]for i in range(self.num) for j in range(self.popsize)])
            self.multi_simu(index)
            try:
                self.eval=self.evaluate1()
            except Exception as e:
                print('------------------------------------------------')
                print('error:',e)
                print('parents replaced dead offsprings')
                ind,ind2=self.check_result()
                self.pop[ind]=np.random.uniform(low=self.low,high=self.high,size=(len(ind),self.dim))
                self.multi_simu(ind2)
                self.eval=self.evaluate1()
            out["F"] = self.eval
            self.save()
            print('No.'+str(self.iter)+' best eval score:',np.min(self.eval))
            self.iter+=1
            print('==================================================================================================')
            print('n_gen  |  n_eval  |     f_avg     |     f_min     |     sigma     | min_std  | max_std  |   axis  ')
            print('==================================================================================================')
        else:
            self.pop=x.copy()
            self.popsize=len(self.pop)
            index=np.array([[i,j]for i in range(self.num) for j in range(self.popsize)])
            self.multi_simu(index)
            try:
                self.eval=self.evaluate1()
            except Exception as e:
                print('------------------------------------------------')
                print('error:',e)
                print('parents replaced dead offsprings')
                ind,ind2=self.check_result()
                self.pop[ind]=np.random.uniform(low=self.low,high=self.high,size=(len(ind),self.dim))
                self.multi_simu(ind2)
                self.eval=self.evaluate1()
            out["F"] = self.eval

            for i in range(len(x)):
                self.bigx.append(x[i])
                self.bigy.append(self.eval[i])
            self.iter+=1
            if self.iter%(self.popsize//2)==0:
                iter=self.iter//(self.popsize//2)
                pd.DataFrame(self.bigx).to_csv(self.record_path+'pop_'+str(iter)+'.csv',index=True)
                self.bigz.append([np.argmin(self.bigy),np.min(self.bigy)])
                info=pd.DataFrame(self.bigz)
                info.columns=['best_one','best_eval']
                info.to_csv(self.record_path+'pop_best.csv',index=True)
                self.bigx=[]
                self.bigy=[]

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)
    
if __name__ == '__main__':
    start_time=time.time()
    
    print('opt start!')
    problem=OPT()
    problem.set_path('E:\\0-Work\\Data\\4-optm-compare\\')##end with \\
    problem.set_config(csv_list = {
                            "BoilerList": "301.Boiler",
                            "TeesList": "304.TeeJoint",
                            "CrossList": "305.CrossJoint",
                            "PipeList": "306.Pipe",
                            "UserMList": "319.TerminalM"
                            })
    problem.set_params(
                muilti_thread=True,#G3PCX-False
                scenario_num=32,
                scenario_num=1,
                )
    
    # algorithm = G3PCX(pop_size=popsize)
    algorithm = CMAES(x0=np.random.uniform(low=problem.low[problem.keyindex],high=problem.high[problem.keyindex],size=len(problem.keyindex)),
                    sigma=0.25,
                    tolfun=5000,
                    # tolx=1,
                    popsize=30,
                    restarts=9,
                    restart_from_best='False',
                    bipop=True)
    # algorithm = GA(
                    # pop_size=500,
                    # eliminate_duplicates=True)
    # algorithm = NelderMead()
    # algorithm = DE(
    #     pop_size=500,
    #     sampling=LHS(),
    #     variant="DE/rand/1/bin",
    #     CR=0.3,
    #     dither="vector",
    #     jitter=False
    # )
    # algorithm = PSO(
    #                 pop_size=10,
    #                 )

    res = minimize(problem,
                algorithm,
                # ('n_gen', 9900),
                seed=1,
                verbose=True,
                )
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    end_time=time.time()
    print('------------------------------------------------')
    print('All finished in time(s):',round(end_time - start_time,3))