# QPLMTS_Vapour
基于QPLMTS算法的边缘计算场景下的任务调度器

## QPLMTS算法
一种融合 Q-Learning、优先级列表与线性加权法的调度长度、服务成本多目标协同优化任务调度算法。
面向静态DAG任务集，将边缘计算场景下的任务调度问题分为**任务排序、节点绑定**两阶段，具体步骤如下：
1. 以任务排序、节点绑定为两个阶段，基于 Upward Rank 评估任务的紧要程度
2. 引入时间差分的强化学习机制进行任务优先级排序
3. 采用 β 权重因子对各优化目标加权打分

## QPLMTS_Vapour调度器
### 运行环境
- 语言环境：Python3.6+
- 依赖环境：matplotlib、sympy、pymysql、numpy、tqdm

### 运行方法
- 为QPL类生成对象
- 采用数据集发生器/边缘计算系统API初始化DAG任务数据、边缘节点数据
- 运行QPLMTS算法核心模块，产生调度结果
- 执行实验分析模块，分析调度结果

### 主要函数与功能
#### 数据预处理区
1. **数据集发生器：** ReadDataSet_Auto(Scale=任务集规模,EdgeNum=边缘节点数,fat=任务链宽度)
  - 基于Daggen产生对应规模(Scale)、链宽度(fat)的静态DAG任务集
  - 产生边缘节点集

#### QPLMTS算法核心区
2. **Upward Rank值计算模块：** CalUR()
  - 基于任务计算量、任务间通信成本，计算所有任务的Upward Rank值
  - 递归结构+LRU_Cache高效计算
3. **调度Q表产生模块：** Q_Process(IterCount=Q表迭代次数)
  - 以任务Upward Rank为Q-Learning Reward，产生调度Q表
  - 给出Reward变化过程记录表
4. **任务分发次序产生模块：** CalCTPS(CurrentTask=入口任务)
  - 根据最大Q值原则，基于收敛的调度Q表，计算任务分发次序CTPS
5. **边缘节点绑定模块：** AllocateNode(CTPS=任务分发次序,a1=MakeSpan的β权重因子,a2=ServiceCost的β权重因子)
  - 基于CTPS，针对每个任务进行节点打分，依次遍历所有任务，产生边缘节点-任务绑定表
  - 边缘节点-任务绑定表数据上行MySQL

#### 一体化调度区
6. **QPLMTS算法一步调度模块：** QPLMTS(a1=MakeSpan的β权重因子,a2=ServiceCost的β权重因子,IterCount=Q表迭代次数)
  - 直接产生一次QPLMTS调度结果

7. **HEFT算法一步调度模块：** HEFT()
  - 直接产生一次HEFT（经典算法）调度结果

#### 实验分析区
8. **学习率实验模块：** ParaOpt_LearningRate(IterCount=Q表迭代次数)
  - 执行学习率参数设计实验，产生学习率图表

9. **β权重因子实验模块：** ParaOpt_Weight(IterCount=Q表迭代次数,Accuracy=β权重因子变化步长)
  - 执行β权重因子参数设计实验，产生β权重因子图表
 
10. **Q表迭代次数实验模块：** ParaOpt_Q(Scale=任务集规模列表,Curve=是否绘制稳定性曲线,IterCount=Q表迭代次数)
  - 执行Q表迭代次数实验，产生Q表迭代次数图表
 
#### 功能模块
11. **MySQL操作模块：** __InputMysql(mode=数据模式,List=数据列表)
12. **更新结果模块：** UpdateResult(List=数据列表)
13. **最早完成时间计算模块：** __ETFT(Task=任务,Edge=边缘节点)、__ETIT(Task=任务,Edge=边缘节点)、__EEAT(Edge=边缘节点)
14. **服务成本计算模块：** __Cost(Task=任务,Edge=边缘节点)
