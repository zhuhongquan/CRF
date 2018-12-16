# CRF
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/:
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./:
        CRF.py: 代码(未进行特征抽取优化)
        CRF_v2.py: 代码(特征抽取优化)
    ./README.md: 使用说明

## 二、运行
### 1.运行环境
    python 3
### 2.运行方法
    各个参数
    'train_data_file': 'data/train.conll', #训练集文件,大数据改为'../big_data/train.conll'
    'dev_data_file': 'data/dev.conll',     #开发集文件,大数据改为'../big_data/dev.conll'
    'test_data_file': 'data/dev.conll',    #测试集文件,大数据改为'../big_data/test.conll'
    'iterator': 100,                          # 最大迭代次数
    'batchsize': 1,                           # 批次大小
    'shuffle': False,                         # 每次迭代是否打乱数据
    'exitor': 10,                             # 连续多少个迭代没有提升就退出
    'regulization': False,                    # 是否正则化
    'step_opt': False,                        # 是否步长优化
    'eta': 0.5,                               # 初始步长
    'C': 0.0001                               # 正则化系数,regulization为False时无效
    
### 3.参考结果
注：</br>
1.迭代次数均从0开始计算。</br>
2.每次迭代时间为train/test/dev的时间总和。</br>
3.由于正则化效果不明显，故未给出正则化实验结果。
#### (1)小数据测试
```
训练集：data/train.conll
测试集：data/test.conll
开发集：data/dev.conll
```
| partial-feature |初始步长| 步长优化 | 迭代次数 | train准确率 | dev准确率 | 时间/迭代 |
| :-------------: |:------:|:-------:|:-------:| :--------: |:--------:|:--------:|
|        ×        |   0.5  |   ×    | 41/51    |    100%   |   88.62%  |    75s   |
|        ×        |   0.5  |   √    |  28/38   |    100%   |   88.67%  |    75s   |
|        √        |   0.5  |   ×    |  25/35   |    100%   |  88.99%   |    20s   |
|        √        |   0.5  |   √    | 12/22    |    100%   |  88.96%   |    20s   |

#### (2)大数据测试
```
训练集：big-data/train.conll
测试集：big-data/test.conll
开发集：big-data/dev.conll
```
| partial-feature |初始步长| 步长优化 | 迭代次数 | train准确率 | dev准确率 | 时间/迭代 |
| :-------------: |:------:|:-------:|:-------:| :--------: |:--------:|:--------:|
|        ×       |   0.5  |   ×    |         |          |         |       |
|        ×       |   0.5  |   √    |         |          |         |       |
|        √       |   0.5  |   ×    |         |          |         |       |
|        √       |   0.5  |   √    |         |          |         |       |
