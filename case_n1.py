import os
import time
import json
import ctypes
import numpy as np
import pandas as pd
import multiprocessing as multi
import matplotlib.pyplot as plt
from json_tool import JST
'''
set the input & output of one specific network
'''
class CASE(JST):
    def __init__(self):
        super().__init__()

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
        print('------------------------------------------------')
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
            userm=pd.read_csv(self.main_path+'netcsv_'+str(i+1)+'\\'+self.csv_list['UserMList']+'.csv',usecols=[6]).T
            cat1=pd.concat([cat1,userm],axis=0)
            boilm=pd.read_csv(self.main_path+'netcsv_'+str(i+1)+'\\'+self.csv_list['BoilerList']+'.csv',usecols=[6]).T
            cat2=pd.concat([cat2,boilm],axis=0)
        self.scenario1=cat1.values#2D array, scenario for each row
        self.scenario2=cat2.values

    def read_target(self):#read true values
        cat=pd.DataFrame()
        for i in range(self.num):
            userp1=pd.read_csv(self.main_path+'target_'+str(i+1)+'\\'+self.csv_list['UserMList']+'Result.csv',usecols=[2]).T.reset_index(drop=True)
            userp2=pd.read_csv(self.main_path+'target_'+str(i+1)+'\\'+self.csv_list['UserMList']+'Result.csv',usecols=[6]).T.reset_index(drop=True)
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

    def evaluate(self):#objective function
        self.read_result_json()
        error=np.sum(abs(self.result-self.target),axis=1)#abs error
        print('evaluated')
        return error#1D array

    def save(self):#save population and evaluation
        self.eval_record.append(self.now_eval)
        self.best_eval.append(np.min(self.now_eval))
        self.best_one.append(np.argmin(self.now_eval))
        info=pd.DataFrame([self.best_one,self.best_eval]).T
        info.columns=['best_one','best_eval']
        info.to_csv(self.record_path+'pop_best.csv',index=True)
        pd.DataFrame(self.eval_record).T.to_csv(self.record_path+'pop_eval_record.csv',index=True)
        pd.DataFrame(self.nextpop).to_csv(self.record_path+'pop_'+str(self.iter)+'.csv',index=True)
        
    def load(self,file=''):#load population
        if file=='':
            pass
        else:
            pop=np.array(pd.read_csv(self.record_path+file,index_col=0))
            if len(pop)>=len(self.pop):
                self.pop=pop[:len(self.pop)]
            elif len(pop)<len(self.pop):
                self.pop[:len(pop)]=pop

    def trend(self):#pop trending
        fig, axs = plt.subplots(int(self.maxiter/10), 10, figsize=(20, 10))
        axs = axs.flatten()
        for i in range(self.maxiter):
            for row in self.pops[str(i+1)]:
                axs[i].plot(row)
            axs[i].set_xlabel('x_dim')
            axs[i].set_ylabel('value')
            axs[i].set_ylim(np.min(self.low),np.max(self.high))
            axs[i].set_title(f'iter {i+1}')
        plt.tight_layout()
        plt.show()

    def history(self):#best evaluation trending
        plt.clf()
        plt.plot(list(range(len(self.best_eval))),self.best_eval)
        plt.title('best_eval_score')
        plt.xlabel('iter')
        plt.ylabel('eval_score')
        plt.show()

    def init_pop(self):#initialize population
        self.nextpop=self.pop.copy()
        self.multi_simu(self.index)
        flag=1
        while flag<=5:
            try:
                self.now_eval=self.evaluate()
                print('------------------------------------------------')
                print('initialize population success')
                break
            except Exception as e:
                print('------------------------------------------------')
                print('re-initialize dead individual:',flag)
                print(e)
                ind,ind2=self.check_result()
                self.nextpop[ind]=np.random.uniform(low=self.low,high=self.high,size=(len(ind),self.dim))#re-initialize
                self.multi_simu(ind2)
                flag+=1
            finally:
                pass
        pd.DataFrame(self.nextpop).to_csv(self.main_path+'record\\pop_0.csv',index=True)