
This system GRENA builds on top of ETH Robustness Analyzer for Neural Networks [(ERAN)](https://github.com/eth-sri/eran) system.

Requirements 
------------
Installation of ERAN system

Our tailored solver 
------------
Our tailored LP solving is implemented under /tf_verify/ml_bound_solver

Usage
-------------
To run the experiments in the papaer, please use the four python scripts.

To run network M\_6x256 for solving comparison, please remember to update respective field to your OWN PATH:
```
python script_M6x256_solve.py
```

To run network M\_ConvMed for solving comparision, use the first script shown below; for property verification, use the second script:
```
python script_Mconvmed_solve.py
python script_Mconvmed_verify.py
```

To run network C\_Resnet4b for solving comparision, use:
```
python script_Cresnet_solve.py
```
Our solver implementation currently faces some OverflowError: (34, 'Numerical result out of range') problem; and GUROBI solver cannot terminate with 24 hours, thus we eliminate this result in the paper.