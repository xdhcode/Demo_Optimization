# Public Optimization Code
## Purpose
This project aims to find the **true** pipeline roughness of **large** district heating networks using heuristic optimization algorthims.
## File intro
### [json_tool.py](https://github.com/xdhcode/Public_Optimization/blob/main/json_tool.py)
Parameter for calling hydraulic simulation software.
### [case_n1.py](https://github.com/xdhcode/Public_Optimization/blob/main/case_n1.py)
The data input and output of one specific network. 
### [opt_DE.py](https://github.com/xdhcode/Public_Optimization/blob/main/opt_DE.py)
+ Search pipeline roughness using differential evolution.
+ Decision variable: The roughness of each pipeline.
+ Objective function: The sum of absolute error of all pressure sensors of all scenarios.
### [opt_NSGA2.py](https://github.com/xdhcode/Public_Optimization/blob/main/opt_NSGA2.py)
+ Search pipeline roughness using NSGA-II based on https://github.com/haris989/NSGA-II.
+ Decision variable: The roughness of each pipeline.
+ Objective function: The sum of absolute error of all pressure sensors. One function for one scenario.
### [opt_CMAES.py](https://github.com/xdhcode/Public_Optimization/blob/main/opt_CMAES.py)
+ Search pipeline roughness using python library [cmaes](https://github.com/CyberAgentAILab/cmaes).
+ Decision variable: The roughness of each pipeline.
+ Objective function: The sum of absolute error of all pressure sensors of all scenarios.
### [opt_PYMOOS.py](https://github.com/xdhcode/Public_Optimization/blob/main/opt_PYMOO.py)
+ Search pipeline roughness using python library [pymoo](https://pymoo.org/).
+ Decision variable: The roughness of each pipeline.
+ Objective function: The sum of absolute error of all pressure sensors of all scenarios.
### [sensitivematrix.py](https://github.com/xdhcode/optimization_library/blob/main/sensitivematrix.py)
+ Calculate the sensitive matrix of one network.
