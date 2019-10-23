# All-Transfer Deep Learning (ATDL) Code
Reimplementation of ATDL series using TensorFlow. 

# Dependencies
- Python 3.6.3::Anaconda Custom (64-bit)
- TensorFlow 1.10.0
- pandas 0.20.3
- numpy 1.14.2

# Usage
Before executing this code, please make ./data and ./resuls directories, and ./data/source.csv and ./data/target.csv. source.csv and target.csv are the souce and target domain data. The format of these data files is: 

`<label>,<value1>,<value2> ...`

After making these files, please execute a following command.

`python main.py [method type]`

method type:
- 0 : Full scratch training
- 1 : Conventional transfer learning (reuse layers except for an output layer)
- 2 : Counting-based ATDL [1]
- 3 : Mean-based ATDL [2]
- 4 : Mean-modified ATDL [3]

This code executes K-fold cross validation (All hyperparamers including K are defined in macro.py). After finishing, the result is saved in ./results.

# References
[1]:Sawada et.al, "[Transfer learning method using multi-prediction deep boltzmann machines for a small scale dataset](http://www.mva-org.jp/Proceedings/2015USB/papers/05-21.pdf)", MVA, 2015.

[2]:Sawada et.al, "[All-Transfer Learning for Deep Neural Networks and its Application to Sepsis Classification](https://arxiv.org/abs/1711.04450)", ECAI, 2016.

[3]:Sawada et.al, "[Improvement in Classification Performance Based on Target Vector Modification for All-Transfer Deep Learning](https://www.mdpi.com/2076-3417/9/1/128)", Applied Sciences, 2019.


# Copyright
Copyright (c) 2019 Yoshihide Sawada
Released under the MIT license
https://opensource.org/licenses/mit-license.php
