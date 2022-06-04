from collections import defaultdict
from functools import lru_cache #缓存加速（动态规划原理）
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False

class TaskScheduling:
    '''类：QOMTS算法模块'''
    ## 强化学习参数
    #State-Action-Reward参数
    states = None # 状态集：任务名称
    actions = None # 动作集：（下一）任务名称
    rewards = None # 回报：UR值
    QTable = defaultdict(dict)  #Q表
    #学习参数
    epsilon = 0.9   # 贪婪度 greedy
    alpha = 0.8     # 学习率
    gamma = 0.8     # 奖励递减值
    ## DAG任务与节点参数
    Tasks = {}
    ComCosts = defaultdict(dict)
    Edges = {} #边缘节点
    
    #边缘节点分配参数
    BenchMarkPrice = 0.0002   #云服务单位价格
    Prices = {} #单位时间的算力价格
    __EdgeFreeTime = {i:0 for i in Edges}
    __TaskFinishTime = {i:0 for i in Tasks}
    
    #数据库信息
    host = '127.0.0.1'
    user = 'root'
    password = '*'
    database = 'vapour',
    
    def __init__(self): 
        '''函数：初始化'''
        pass
    
    def ReadDataSet_Auto(self,Scale=20,EdgeNum=5,fat=3):
        '''函数：自动生成任务、边缘节点'''
        import os 
        import random
        
        self.Tasks = {}
        self.ComCosts = defaultdict(dict)
        
        ## 1. 生成DAG任务
        #提前生成Tasks
        for i in range(1,Scale+1):
            self.Tasks[i] = [-1,[],[]]
        DAG_raw = os.popen(f'./daggen -n {Scale} --dot -fat {fat}').read().split('\n')[3:-2]
        #print(DAG_raw)
        for dag in DAG_raw:
            Task = dag.split('[size')[0].strip()
            Cost = int(dag.split('="')[1].split('"')[0])/10000000   #缩小化

            #解析任务
            # 通信链
            if (Task.find('->') != -1):
                self.ComCosts[int(Task.split('->')[0].strip())][int(Task.split('->')[1].strip())] = Cost
                # 同时修改前驱后继任务
                self.Tasks[int(Task.split('->')[0].strip())][1].append(int(Task.split('->')[1].strip()))  #添加后继任务
                self.Tasks[int(Task.split('->')[1].strip())][2].append(int(Task.split('->')[0].strip()))  #添加前驱任务
            # 任务点
            else:
                self.Tasks[int(Task)][0] = Cost  #后继，前驱（留空）
        
        ## 2.生成边缘节点
        self.Edges = {}
        for i in range(1,EdgeNum+1):
            self.Edges[i] = random.randint(50,100)  #算力生成器
        
        self.init_price()
        #数据上库
        Tasks = []
        for i in self.Tasks:
            Tasks.append([i,self.Tasks[i][0]])
        Edges = []
        for i in self.Edges:
            Edges.append([i,self.Edges[i]])
        self.__InputMysql(mode='Tasks',List=Tasks)
        self.__InputMysql(mode='Edges',List=Edges)
        
        
    def ReadDataSet(self,mode='file',Scale=20):
        import pickle
        
        '''函数：读取任务数据集'''
        if (mode=='file'):
            ## 从TaskInfo.db读取任务
            with open('TaskInfo.db','rb') as f:
                self.Tasks,self.ComCosts= pickle.load(f)
                f.close()
        elif (mode=='manual'):
            # 静态DAG任务
            ## Task[任务名称]-> (任务计算量,[后继任务],[前驱任务])
            ## ComCost[任务1][任务2]
            from collections import defaultdict

            # 手动生成任务
            ## 输入：
            n = int(input('请输入任务数：'))
            for i in range(n):
                Name,ComputCost = map(int,input('名称 计算代价：(仅限数字)').strip().split())
                PriorTask = list(map(int,input('该任务的前序任务：').strip().split()))
                SuccTask = list(map(int,input('该任务的后序任务：').strip().split()))
                PriorComCost = input('抵达前序任务的通信成本：').strip().split()

                self.Task[Name] = (ComputCost,SuccTask,PriorTask)  #计算代价，前序，后序

                c = 0
                for j in PriorTask:
                    self.ComCost[j][Name] = float(PriorComCost[c])
                    c += 1

            Storage = (self.Task,self.ComCost)

            with open('TaskInfo.db','wb') as f:
                pickle.dump(Storage,f)
                f.close()

            print('√ 任务信息序列化至TaskInfo.db')
    
    def init_price(self):
        import math
        
        '''函数：价格初始化'''
        self.Prices[list(self.Edges.keys())[0]] = self.BenchMarkPrice
        for Edge in self.Edges:
            self.Prices[Edge] = self.Prices[list(self.Edges.keys())[0]] * (self.Edges[Edge]/self.Edges[list(self.Edges.keys())[0]])* (1+math.log(self.Edges[Edge]/self.Edges[list(self.Edges.keys())[0]]))
        
    def __init_QTable(self):
        '''函数：初始化Q表'''
        #初始化Q表
        self.QTable = defaultdict(dict)
        for i in self.states:
            for j in self.actions:
                self.QTable[i][j] = 0

    def __get_valid_actions(self,state,PassTasks):
        '''函数：取当前状态下的合法动作集合'''
        #取出所有经过节点的后继节点，已经包括了自己
        VaildActions = set()

        #遍历所有不在PassTasks的节点
        for OutTask in list(set(self.states) - PassTasks):
            if (PassTasks.intersection(set(self.Tasks[OutTask][2])) == set(self.Tasks[OutTask][2])): #扫描它的前序节点，如果都在PassTasks中，则可以
                VaildActions.add(OutTask)

        return list(VaildActions)

    def __isAllZero(self,a):
        for i in a:
            if (i != 0):
                return False
        return True

    @lru_cache(None)
    def __UR(self,TaskName):
        '''函数：递归法计算任务UR'''
        MaxUrgenValue = 0  #考虑到最后一个任务
        for SuccTaskName in self.Tasks[TaskName][1]:
            ## 最大值替代
            MaxUrgenValue = max(MaxUrgenValue,self.__UR(SuccTaskName) + float(self.ComCosts[TaskName][SuccTaskName]))  #通信代价

        return MaxUrgenValue + float(self.Tasks[TaskName][0])
    
    def CalUR(self):
        '''函数：UR值计算'''
        UR_Results = {}  #存储各任务UR值
        #遍历每个任务
        for TaskName in self.Tasks:
            #求任务的Upward Rank
            UR_Results[TaskName] = self.__UR(TaskName)#计算所得UR值
        
        return UR_Results

    def Q_Process(self,IterCount=400):
        '''函数：强化学习Q过程'''
        import random
        from tqdm import tqdm
        
        RewardShow = []  #记录收敛情况
        
        #初始化状态
        self.actions = list(self.Tasks.keys())
        self.states = list(self.Tasks.keys())
        self.rewards = self.CalUR()
        self.__init_QTable()
        
        '''函数：Q表收敛过程'''
        for i in tqdm(range(IterCount),desc='Q表收敛'):
            #选取初始位置：改进
            current_state = 1

            PassTasks = set()  #已经包含的任务
            PassTasks.add(current_state)

            # 迭代到无路可走
            ## 中止条件：没有后继任务
            ## 一次for，一次完整序列
            while(len(self.__get_valid_actions(current_state,PassTasks)) != 0):
                # ⭐选择动作
                if (random.uniform(0,1) > self.epsilon) or (self.__isAllZero(self.QTable[current_state].values())):
                    current_action = random.choice(self.__get_valid_actions(current_state,PassTasks))
                else:  #如果达到预期贪婪值，就选择Q最大
                    current_action = max(self.QTable[current_state].items(),key=lambda x:x[1])[0]

                # 计算得到下一个状态
                next_state = current_action

                # 计算下一个状态的所有Q值，取最大值
                next_state_q_values = [self.QTable[next_state][i] for i in self.__get_valid_actions(next_state,PassTasks)]        

                #更新Q
                self.QTable[current_state][current_action] += self.alpha * (self.rewards[next_state] + self.gamma * max(next_state_q_values) - self.QTable[current_state][current_action])
                # 进入下一个状态
                current_state = next_state

                PassTasks.add(current_state)  #加入包含列表

            #计算q表的整体reward，统计收敛情况
            AllReward = 0
            for i in self.QTable:
                for j in self.QTable[i]:
                    AllReward += self.QTable[i][j]

            RewardShow.append(AllReward)

        return RewardShow

    #⭐问题
    def CalCTPS(self,CurrentTask = 1):
        '''函数：生成任务分发次序'''
        
        CTPS = [CurrentTask]
        CTPS_Hash = set([CurrentTask])

        while(len(CTPS) != len(self.states)):
            #排序当前任务的状态转移序列
            ActionList = sorted(self.QTable[CurrentTask].items(),key=lambda x:x[1],reverse=True)
            #逐一取出
            for Action in ActionList:
                #判断有没有
                if (Action[0] not in CTPS_Hash):  #不在，添加
                    CTPS.append(Action[0])
                    CTPS_Hash.add(Action[0])

                    #转移状态
                    CurrentTask = Action[0]
                    break
            
        return CTPS
    
    '''最早完成时间模块'''
    def __ETFT(self,Task,Edge):
        '''函数：计算最早完成时间'''
        return self.Tasks[Task][0]/self.Edges[Edge]+self.__ETIT(Task,Edge)

    def __ETIT(self,Task,Edge):
        '''函数：计算任务最早发起执行时间'''
        #遍历前驱任务，选出最大的
        MaxPredecessorTask = -1

        #如果没有前驱任务，直接执行
        for PredTask in self.Tasks[Task][2]:
            MaxPredecessorTask = max(self.__TaskFinishTime[PredTask] + self.ComCosts[PredTask][Task],MaxPredecessorTask)

        return max(self.__EEAT(Edge),MaxPredecessorTask)

    def __EEAT(self,Edge):
        '''函数：计算节点最早进入空闲的时间'''
        return self.__EdgeFreeTime[Edge]

    def __Cost(self,Task,Edge):
        '''函数：服务成本代价
           任务总计算时间*价格'''
        return (self.Tasks[Task][0]/self.Edges[Edge])*self.Prices[Edge]
    
    def AllocateNode(self,CTPS,a1=0.5,a2=0.5):
        '''函数：线性加权法分配边缘节点'''
        #任务完成时刻表（相对时间/秒）
        self.__TaskFinishTime = {i:-1 for i in self.Tasks}
        self.__EdgeFreeTime = {i:0 for i in self.Edges}

        AllocateList = []  #节点分配表（(任务,节点)）
        ServiceCost = 0

        # 遍历每个任务
        for Task in CTPS:
            # 前序任务是否完成
            ## 判断前序任务完成情况
            
            #节点分数列表
            EdgeList = []
            #遍历每个节点
            for EdgeName in self.Edges:
                #节点打分
                EdgeScores = -a1*self.__ETFT(Task,EdgeName)-a2*self.__Cost(Task,EdgeName)
                EdgeList.append((EdgeName,EdgeScores))

            #选出一个节点
            EdgeList = sorted(EdgeList,key=lambda x:x[1],reverse=True)

            ChoiceEdge = EdgeList[0][0]

            #更新：任务完成时间
            TaskStartTime = self.__ETIT(Task,ChoiceEdge)
            self.__TaskFinishTime[Task] = TaskStartTime + self.Tasks[Task][0]/self.Edges[ChoiceEdge]
            
            #更新：服务成本
            ServiceCost += self.__Cost(Task,ChoiceEdge)
            
            #更新：节点最早空闲时间
            self.__EdgeFreeTime[ChoiceEdge] = self.__TaskFinishTime[Task]
            
            #任务、节点、当前MakeSpan、当前ServiceCost
            AllocateList.append([Task,ChoiceEdge,TaskStartTime,self.__TaskFinishTime[Task],ServiceCost])
        
        self.__InputMysql(mode='allocation',List=AllocateList)

            #print('WARNING: 跳过写入Mysql过程')
        #print('√ 第三阶段：产生任务-边缘节点映射完成')
        return AllocateList
    
    def QOMTS(self,a1=0.85,a2=0.15,IterCount=1000):
        '''函数：QOMTS调度'''
        
        self.Q_Process(IterCount)
        CTPS = self.CalCTPS()
        AllocateList = self.AllocateNode(CTPS,a1=a1,a2=a2)
        
        #输出调度映射表
        
        #print('任务调度规则如下：')
        #for i in AllocateList:
            #print(f'''INFO: 任务"{i[0]}" → 边缘节点"{i[1]}"，任务执行模拟用时{round(i[2],2)}s''')
        
        return AllocateList  #倒数第二个即为MakeSpan值
    
    ## 结果导出模块
    def __InputMysql(self,mode=None,List=None):
        import pymysql 
        
        db = pymysql.connect(host=self.host,
                     user=self.user,
                     password=self.password,
                     database=self.database,
                     charset='utf8mb4') #连接本地数据库
        cursor = db.cursor() #获取操作游标：操作数据库
        
        if (mode=='allocation'):
            Paras = '%s,%s,%s,%s,%s'
            Table = 'qomts'
        elif (mode=='Tasks'):
            Paras = '%s,%s'
            Table = 'TasksInfo'
        elif (mode=='Edges'):
            Paras = '%s,%s'
            Table = 'EdgesInfo'
        elif (mode=='result'):
            Paras = '%s,%s,%s'
            Table = 'Result'

        #清空表
        cursor.execute(f'truncate {Table}')
        cursor.executemany(f'INSERT INTO {Table} values ({Paras});',List)
        db.commit()
        db.close()
    
    ## 实验部分
    #参数设计
    def ParaOpt_Q(self,Scale=[],Curve=True,IterCount=400):
        from sympy import Symbol,diff
        import numpy as np
        
        plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
        plt.rcParams["axes.unicode_minus"]=False
        
        ##1. 强化学习机制参数分析
        print(f'INFO: 进行强化学习机制参数分析...')
        print(f'INFO: 1. 迭代次数与收敛性')
        # 迭代次数分析
        # 多个迭代次数——多条线
        x1 = [i+1 for i in range(IterCount)]
        #图片信息
        #plt.xlabel("迭代次数/次")
        #plt.ylabel("Reward总量")
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("迭代次数/次")
        ax1.set_ylabel("Reward总量")
        
        #重复三次过程
        for i in range(len(Scale)):
            #self.ReadDataSet_Auto(Scale=i) #生成DAG任务
            #导入数据集
            self.Tasks = Scale[i][0]
            self.ComCosts = Scale[i][1]
            
            RewardAll=self.Q_Process(IterCount)

            #plt.plot(x1,RewardAll,label=f'第{i+1}次收敛')  #绘制总Q值
            ax1.plot(x1, RewardAll,label=f'{len(Scale[i][0])}任务数')
        plt.legend(loc='best')#图例
        
        #拟合函数
        p1 = np.poly1d(np.polyfit(x1, RewardAll, 4)) #使用次数合成多项式

        x = Symbol('x')
        p1_express = p1[0]+p1[1]*x+p1[2]*x**2+p1[3]*x**3+p1[4]*x**4
        diffy = diff(p1_express,x)
        y_diffy = []

        for i in x1:
            y_diffy.append(float(diffy.subs(x,i)))
        
        if (Curve is True):
            #plt.plot(x1,y_diffy,label='收敛波动率')
            ax2 = ax1.twinx()
            ax2.set_ylabel("波动率/%")
            ax2.plot(x1,y_diffy,color='purple',label="收敛波动率")

            plt.legend(loc='best')
        plt.title(f"调度Q表收敛趋势")
        plt.savefig("迭代次数-收敛性.svg")
    
    def ParaOpt_LearningRate(self,IterCount = 500):
        LearningRateRange = range(20,101,10)
        for i in LearningRateRange:
            self.alpha = i/100
            RewardAll = self.Q_Process(IterCount=IterCount)
            
            plt.plot(list(range(1,IterCount+1)),RewardAll,label=f'{i}学习率')
        plt.legend(loc='best')
        plt.xlabel('迭代次数')
        plt.ylabel('Reward总量')
        plt.title(f"调度Q表学习率参数")
        plt.savefig("学习率.svg")
        
    def ParaOpt_Weight(self,IterCount=400,Accuracy=5):
        self.Q_Process(IterCount)
        
        ##2. MakeSpan、ServiceCost权重系数
        AccuracyRange = range(10,91,Accuracy)
        x_2 = [i/100 for i in AccuracyRange]
        y_2 = []  #[(MakeSpan,ServiceCost)]
        for i in AccuracyRange:
            CTPS = self.CalCTPS()
            AllocateList = self.AllocateNode(CTPS,i/100,1-i/100)
            y_2.append((AllocateList[-1][2],AllocateList[-1][3]))
            
        #画图
        k1 = list(zip(*y_2))[0]#线1的纵坐标
        k2 = list(zip(*y_2))[1]#线2的纵坐标

        fig, ax1 = plt.subplots()
        ax1.plot(x_2, k1, color="orange",marker='o',label="MakeSpan")
        ax1.set_xlabel("MakeSpan权重比")
        ax1.set_ylabel("MakeSpan")
        plt.legend(loc='upper left')#图例

        ax2 = ax1.twinx()
        ax2.plot(x_2, k2, color="purple",marker='o',label="ServiceCost")
        ax2.set_ylabel("ServiceCost")
        plt.title('MakeSpan-ServiceCost权重因子分配')
        plt.legend(loc='upper right')#图例
        plt.savefig("权重因子分配.svg")
        
        
    #HEFT算法
    def HEFT(self):
        #UR值确定任务序列
        TaskList = list(zip(*sorted(self.CalUR().items(),reverse=True,key=lambda x:x[1])))[0]
        
        #计算分配节点的时间
        return self.AllocateNode(TaskList,a1=1,a2=0)  #仅MakeSpan
    
    #更新结果
    def UpdateResult(self,List):
        self.__InputMysql(mode='result',List=List)