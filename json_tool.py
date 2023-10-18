import pandas as pd
import json
import os
'''
hydraulic simulation module (json)
private information denoted as *
'''
class JST():
    def __init__(self):
        #hydraulic config
        self.config={
                "config_id": "*",
                "project_id": "*",
                "topology_id": "*",
                "CalcList": [{
                    "print": 0,
                    "heatLossLimitM": 0.05,
                    "heatLossSet": 0.0,
                    "timeStep": 0.1,
                    "minP": 0.001,
                    "threadCount": 1,
                    "errPLimit": 0.1,
                    "errMLimit": 0.1,
                    "loopCount": 1000000,
                    "pInit": 1000000.0,
                    "tInit": 330
                    }]}
    def csv2json(self):
        #header of network component
        tp,k1,k2=["tp_code","tp_id"],["conn_code_1","conn_port_1"],["conn_code_2","conn_port_2"]
        name0=tp+k1+k2
        name={
            "BoilerList":tp+["calc_type"]+k1+["p_1","m_1","t_1"]+k2+["p_2","m_2","t_2"],
            "HeaterRList":name0+["r","t_1",'t_2'],
            "HeaterMList":name0+["m","t_1",'t_2'],
            "HeaterRMList":name0+["ra2","ra1","ra0","t_1",'t_2'],
            "UserRList":name0+["dp","dt","dq","calc_type"],
            "UserMList":name0+["m","dt","dq","calc_type"],
            "UserRMList":name0+["a2","a1","a0","dt","dq","calc_type"],
            "PlugList":tp+["conn_code_1","conn_port_1"],
            "TwowaysList":name0,
            "TeesList":name0+["conn_code_3","conn_port_3"],
            "CrossList":name0+["conn_code_3","conn_port_3","conn_code_4","conn_port_4"],
            "ValveList":name0+["on_off"],
            "ValvePRList":name0+["dp"],
            "ValveRLList":name0+["d100","cv","open_ness"],
            "ValveLineList":name0+["d100","kvs","rv","open_ness"],
            "ValveEPList":name0+["d100","kvs","rv","open_ness"],
            "PumpOLList":name0+["a2","a1","a0","rated_fre","current_fre"],
            "PumpOLGList":name0+["a2","a1","a0","rated_fre","current_fre","pump_num"],
            "PumpSPList":name0+["a2","a1","a0","rated_fre","current_fre","cpp_p"],
            "PumpSPGList":name0+["a2","a1","a0","rated_fre","current_fre","cpp_p","pump_num"],
            "PipeList":tp+["inside","wall_thickness","length","roughness"]+k1+["tp_height_1"]+k2+["tp_height_2","heat_loss"]
            }
        #initialize json
        InputJson = {"calc_id":"calc_id","project_id":"project_id","topology_id":"topology_id","data_time":"data_time",
                    "BoilerList":[],"HeaterRList":[],"HeaterMList":[],"HeaterRMList":[],
                    "UserRList":[],"UserMList":[],"UserRMList":[],"ExStationRList":[],"ExStationRMList":[],
                    "PlugList":[],"TwowaysList":[],"TeesList":[],"CrossList":[],
                    "ValveList":[],"ValvePRList":[],"ValveRLList":[],"ValveLineList":[],"ValveEPList":[],"ConnectorList":[],
                    "PumpOLList":[],"PumpOLGList":[],"PumpSPList":[],"PumpSPGList":[],"PipeList":[]}
        
        for key in self.csv_list.keys():
            topoPDT=pd.read_csv(self.csv_path+self.csv_list[key]+'.csv',encoding='gbk')
            topoPDT.columns=name[key]
            if key in ["UserRList","UserMList","UserRMList"]:
                topoPDT["calc_type"]=topoPDT["calc_type"].astype(int)
            if key in ["ValveList"]:
                topoPDT["on_off"]=topoPDT["on_off"].astype(int)
            if key in ["HeaterRList","HeaterMList","HeaterRMList"]:
                topoPDT['t_2']=topoPDT['t_1']
            InputJson[key] = topoPDT.apply(lambda row: dict(zip(topoPDT.columns, row)), axis=1).tolist()
        return  InputJson
    
    def save_json(self,j):#save json
        with open(self.main_path+'csv.json','w') as f:
            json.dump(j,f,indent=4)
        print('json saved')
        
    def json2csv(self, info0, path):#save simulation result as csv
        if info0["state"] == 1:
            if not os.path.exists(path):
                os.mkdir(path)
            for key,value in self.csv_list.items():
                pd.DataFrame(info0[key]).to_csv(path+value+'Result.csv',index=False)
        else:
            print('state error code:',info0["state"])