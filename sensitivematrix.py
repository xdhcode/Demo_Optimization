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
sensitive matrix (sensors on pipe roughness)
'''
class SEN(JST):
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
        #set roughness of pipelines
        optparam=(self.pop[i_ind]*1e-6).tolist()#unit:micrometer to meter
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
    
    def read_scenario(self):
        cat1=pd.DataFrame()
        cat2=pd.DataFrame()
        for i in range(self.num):
            pipe=pd.read_csv(self.main_path+'netcsv_'+str(i+1)+'\\'+self.csv_list['PipeList']+'.csv',usecols=[3]).T
            userp1=pd.read_csv(self.main_path+'netcsv_'+str(i+1)+'\\'+self.csv_list['UserMList']+'.csv',usecols=[6]).T.reset_index(drop=True)
            cat1=pd.concat([cat1,pipe],axis=0)
            cat2=pd.concat([cat2,userp1],axis=0)
        self.scenario1=cat1.values#2D array, scenario for each row
        self.popsize=pipe.shape[1]#pipe_num
        self.dim=userp1.shape[1]#sensor_num
        self.index=np.array([[i,j]for i in range(self.num) for j in range(self.popsize)])
        self.pop=self.baserough*(np.ones([self.popsize,self.popsize])+self.ratio*np.eye(self.popsize))

    def read_target(self):#base scenario
        csv=self.csv_json.copy()
        roughlist=self.baserough*np.ones(self.popsize)
        optparam=(roughlist*1e-6).tolist()
        csv['PipeList'] = list(map(lambda dict, value: {**dict, 'roughness': value}, csv['PipeList'], optparam))
        simulator = ctypes.WinDLL(self.main_path+"Systemr32.dll")
        simulator.Calculate.restype = ctypes.c_char_p
        simu_info = eval(simulator.Calculate(json.dumps(csv).encode(),self.config_json))
        path=self.main_path+'target_1\\'
        if not os.path.exists(path):
            os.mkdir(path)
        if simu_info['state'] == 1:
            print("Success: base condition saved")
            self.json2csv(simu_info, path)
        else:
            print("Error: "+simu_info['message'])
        #read result
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
    
    def evaluate(self):#calculate change rates
        self.read_result_json()
        error=(self.result-self.target)/self.target
        self.senmat=pd.DataFrame(error)
        self.senmat.columns=['sensor_'+str(i+1) for i in range(self.dim*2)]
        self.senmat.to_csv(self.record_path+'senmat_'+str(self.baserough)+'-'+str(self.ratio)+'.csv',index=True)
        print('evaluated')

    def set_params(self,**params):
        #general parameter
        self.thread=params['thread_num']
        self.num=params['scenario_num']
        self.baserough=params['base_roughness']
        self.ratio=params['roughness_increment']

        if not os.path.exists(self.main_path+'record\\'):
            os.mkdir(self.main_path+'record\\')

    def run(self):
        start_time=time.time()
        print('------------------------------------------------')
        print('start sensitivity matrix running')
        self.read_scenario()
        self.read_target()
        self.multi_simu(self.index)
        self.evaluate()
        end_time=time.time()
        print('All finished in time(s):',round(end_time - start_time,3))
        print('------------------------------------------------')

if __name__ == '__main__':
    np.random.seed(2023)
    a=SEN()
    a.set_path('E:\\0-Work\\Data\\4-optm-sen\\')#end with \\
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
                base_roughness=1000,
                roughness_increment=0.1
                )
    a.run()



