# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 09:20:42 2022

@author: Tao
"""
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
from sklearn.metrics import r2_score
from scipy.optimize import least_squares
import pandas as pd
import datetime
from pyecharts import options as opts
from pyecharts.charts import Map, Timeline

def get_data(country,t_start_str,t_end_str):

    data = pd.read_csv('owid-covid-data.csv')
    location = data['location']
    for i in range(0,len(location)):
        if location[i] == country:
            location_start = i
            break
    i=0
    for i in range(0,len(location)):
        if location[i] == country:
            location_end = i
    data = data[location_start:location_end+1]
    #索引替换
    data = data.reset_index(drop=True) 
    #提取出总人口
    data_population = data['population']
    population = data_population[1]
    #提取出有用的几列
    data = data[['continent','location','date','total_cases','new_cases','total_deaths','new_deaths','people_fully_vaccinated',
                 'people_fully_vaccinated_per_hundred']]
    #缺失值
    for k in range(0,data.shape[1]):
        if pd.isnull(data.iloc[0,k]) == True:
            data.iloc[0,k] = 0
    data.fillna(axis=0,method='ffill',inplace = True)
    
    #计算各种数据
        #计算new recover
    new_recover = []                            
    for h in range(0,len(data['date'])):
        if h<10:
            new_recover.append(0)
        else:
            nr = data['new_cases'][h-10]-data['new_deaths'][h]
            new_recover.append(nr)
    data.insert(loc=len(data.columns), column='new_recover', value=new_recover)
        #计算total recover
    total_recover = []                           
    for a in range(0,len(data['date'])):
        tr = sum(new_recover[0:a+1])
        total_recover.append(tr)
    data.insert(loc=len(data.columns), column='total_recover', value=total_recover)
        #计算R
    R = []
    for b in range(0,len(data['date'])):
        r = data['total_deaths'][b]+data['people_fully_vaccinated'][b]+data['total_recover'][b]
        R.append(r)
    data.insert(loc=len(data.columns), column='R', value=R)
        #计算I 可能出现负的情况
    I = []
    for c in range(0,len(data['date'])):
        i_ = data['total_cases'][c]-data['total_deaths'][c]-data['total_recover'][c]
        I.append(i_)
    for i_0 in range(0,len(I)):
        if I[i_0] < 0:
            I[i_0]=0
    data.insert(loc=len(data.columns), column='I', value=I)
        #计算E
    E = []
    for d in range(0,len(data['date'])-4):
        e = I[d+4]
        E.append(e)
    E.append(0)
    E.append(0)
    E.append(0)
    E.append(0)
    data.insert(loc=len(data.columns), column='E', value=E) 
        #计算lamda
    lamda = []
    E_0 = 0
    t_ = np.arange(1, len(data['date'])+1, 1)
    for f in range(0,len(data['date'])):
        if E[f] == 0:
            lamda.append(0)
        else:
            E_0 = E[f]
            break #会有0出现的情况
    if E_0 != 0 : #如果找到了   
        for f_ in range(f,len(data['date'])):
                lamda_ = ((np.log(E[f_])-np.log(E_0))/t_[f_])+0.35
                lamda.append(lamda_)
    else:
        for f_ in range(f,len(data['date'])):
            lamda_ = 0
            lamda.append(lamda_)
        
    for l_0 in range(0,len(lamda)):
        if lamda[l_0] == float('-inf'):
            lamda[l_0]=0
    data.insert(loc=len(data.columns), column='lamda', value=lamda) 
    
    #第一列插入
    insert_list = []
    for j in range(1,len(data['date'])+1):
        insert_list.append(j)
    data.insert(loc=0, column='t', value=insert_list)
    
    data_date = data['date']
    for p in range(0,len(data['date'])):
        if data_date[p]==t_start_str:
            t_start = p
            break
 
    for q in range(0,len(data['date'])):
        if data_date[q]==t_end_str:
            t_end = q+1
            break
    
    data = data[t_start:t_end]
    data = data.reset_index(drop=True) 
    #把负值变为0
    return data,population

def fit_predict(country,t_start_str,t_end_str,days):
    data , population= get_data(country,t_start_str,t_end_str)
    
    def trans(date):#将日期转换为所需要的索引 注意定义END时要加一因为左闭右开
        data_date = data['date']
        for i in range(0,len(data_date)):
            if data_date[i]==date:
                t = i
                break
        return t
    
    t_start = trans(t_start_str)
    t_end = trans(t_end_str)+1
    q2 = 0.1
    delta = 0.25    
    number = population# 总人数
    I0 = data['I'][t_start] # 患病者的初值
    E0 = data['E'][t_start]         # 潜伏者的初值
    r0 = data['R'][t_start] # 被治愈的人的初值，一开始可设为0 
    Q0=number*0.1
    S0 = number-I0-E0-r0-Q0  # 易感者的初值
    Y0 = (S0, E0, I0, Q0, r0)  # 微分方程组的初值
    
    def lamda(t):#拟合lamda(t)
        list = data['I']
        list = list.to_list()
        tEnd_ = len(list)+1  # 拟合日期长度,使用所有数据拟合
        t_ = np.arange(1, tEnd_, 1)  # (start,stop,step) 左闭右开
        data_lamda = data["lamda"]
        data_lamda = np.array(data_lamda)
        r = np.polyfit(t_, data_lamda, 3)
        return r[0]*t**3+r[1]*t**2+r[2]*t**1+r[3]-(0.1-q2)
    
    def u(t):#拟合u(t)
        list = data['I']
        list = list.to_list()
        tEnd_ = len(list)+1  # 拟合日期长度,使用所有数据拟合
        data_u = data["people_fully_vaccinated_per_hundred"]
        t_start_u = len(data_u)
        for i in range(0,len(data_u)):
            if data_u[i]!=0:
                t_start_u = i
                break
        if t_start_u != len(data_u): #如果找到了开始时间则正常返回
            t_ = np.arange(t_start_u+1 ,tEnd_, 1)  # (start,stop,step)
            data_u = data_u[t_start_u:]
            data_u = np.array(data_u)
            r = np.polyfit(t_, data_u, 3)                
            return (r[0]*t**3+r[1]*t**2+r[2]*t**1+r[3])*0.01
        else:#否则返回0
            return 0*t
    
    def SEIR(y, t, lamda ,delta, mu ,q1 ,q2 ,q3 , a, b):  # SEIR 模型
        S,E,I,Q,R = y 
        dS_dt = -q1*S -lamda(t)*(S/number)*I+a*Q  # ds/dt = -lamda*s*i  #a*Q#
        dE_dt = lamda(t)*(S/number)*I - delta*E - q2*E  # de/dt = lamda*s*i - delta*e
        dI_dt = delta*E - mu*I-q3*I # di/dt = delta*e - mu*i
        dQ_dt = q1*S+q2*E+q3*I-b*Q-a*Q
        dR_dt = mu*I+b*Q
        return np.array([dS_dt,dE_dt,dI_dt,dQ_dt,dR_dt])
    
    def SEIR_v(y, t, lamda ,delta, mu ,q1 ,q2 ,q3 ,a, b ,u, p1,p2):#, p2
        S,E,I,Q,R = y 
        dS_dt =  -q1*S -lamda(t)*(S/number)*I-p1*u(t)*S+a*Q +p2*R
        dE_dt = lamda(t)*(S/number)*I - delta*E - q2*E  # 
        dI_dt = delta*E - mu*I-q3*I # 
        dQ_dt = q1*S+q2*E+q3*I-b*Q-a*Q
        dR_dt = mu*I+p1*u(t)*S-p2*R+b*Q
        return np.array([dS_dt,dE_dt,dI_dt,dQ_dt,dR_dt])
    
    def error(pr,t,y,Y0_1):
        q1,q3,a,b,mu = pr[0],pr[1],pr[2],pr[3],pr[4]
        ySEIR = odeint(SEIR, Y0_1, t, args=(lamda,delta,mu,q1,q2,q3,a,b)) #注意这里的初值
        return (ySEIR[:,2]-y)**2
    
    def error_v(pr,t,y,Y0_1):
        q1,q3,a,b,mu,p1,p2 = pr[0],pr[1],pr[2],pr[3],pr[4],pr[5],pr[6]
        ySEIR = odeint(SEIR_v, Y0_1, t, args=(lamda,delta,mu,q1,q2,q3,a,b,u,p1,p2)) #注意这里的初值 ,p2
        return (ySEIR[:,2]-y)**2
    
    def stage(t_start_str,t_end_str,Y0): #xxxx/x/xx
        p0=[0.261427,0.16088,0.1,0.280726,0.8]
        bounds= ([0,0,0,0,0],[1,1,1,1,1])
        t_start = trans(t_start_str)
        t_end = trans(t_end_str)+1
        y_real = np.array(data["I"][t_start:t_end])#获取阶段数据
        t_train=np.arange(t_start+1, t_end+1, 1)#这里的序列仅为下面的最小二乘训练而用
        Para=least_squares(error,p0,args=(t_train,y_real,Y0),bounds = bounds) 
        Para = Para.x
        q1,q3,a,b,mu = Para[0],Para[1],Para[2],Para[3],Para[4]
        t_predict = np.arange(t_start+1, t_end+1, 1)#这里的序列是拟合的时间
        y_predict = odeint(SEIR, Y0, t_predict, args=(lamda,delta,mu,q1,q2,q3,a,b))   
        r2 = r2_score(y_real,y_predict[:,2])#计算阶段一r^2 
        Y0_next=y_predict[t_end-t_start-1,:]#输出下一个阶段的初值即此阶段的最后一天的值
        return  y_real,y_predict,Para,r2,Y0_next
    
    def stage_v(t_start_str,t_end_str,Y0): #xxxx/xx/xx
        p0=[0.5,0.5,0.1,0.5,0.5,0.002,0.8]
        bounds_v = ([0,0,0,0,0,0,0],[1,1,1,1,1,1,1])
        t_start = trans(t_start_str)
        t_end = trans(t_end_str)+1
        y_real = np.array(data["I"][t_start:t_end])#获取阶段数据 这里是作为索引
        t_train=np.arange(t_start+1, t_end+1, 1)#这里的序列仅为下面的最小二乘训练而用 这里作为计算
        Para=least_squares(error_v,p0,args=(t_train,y_real,Y0),bounds = bounds_v) 
        pr = Para.x
        q1,q3,a,b,mu,p1,p2 = pr[0],pr[1],pr[2],pr[3],pr[4],pr[5],pr[6]
        t_predict = np.arange(t_start+1, t_end+1, 1)#这里的序列是预测的时间 这里作为计算
        y_predict = odeint(SEIR_v, Y0, t_predict, args=(lamda,delta,mu,q1,q2,q3,a,b,u,p1,p2)) 
        r2 = r2_score(y_real,y_predict[:,2])#计算阶段一r^2
        Y0_next=y_predict[t_end-t_start-1,:]
        return  y_real,y_predict,pr,r2,Y0_next

    y_real,y_fit,Para_1,r2,Y0_next=stage_v(t_start_str,t_end_str,Y0)

    q1=Para_1[0]
    q3=Para_1[1]
    a=Para_1[2]
    b=Para_1[3]
    mu= Para_1[4]
    p1=Para_1[5]
    p2=Para_1[6]
    
    R0_list = []
    i=0
    for i in range(0,len(data['lamda'])):
        R0 = (data['lamda'][i]*0.25)/(0.35*(mu+q3))
        R0_list.append(R0)
    
    t_predict_30 = np.arange(t_start+1, t_end+days+1, 1)
    y_predict_30 = odeint(SEIR_v, Y0, t_predict_30, args=(lamda,delta,mu,q1,q2,q3,a,b,u,p1,p2))
    I_predict = y_predict_30[:,2]
    I_fit = y_fit[:,2]
    I_real = y_real
    
#    I_predict = pd.DataFrame(I_predict)    
#    I_fit = pd.DataFrame(I_fit)
#    I_real = pd.DataFrame(I_real)
    i=0
    for i in range(0,len(I_predict)-len(I_fit)):
        I_fit = np.append(I_fit,np.nan) 
        I_real = np.append(I_real,np.nan)
        R0_list.append(np.nan)
    result_I = pd.DataFrame(data = None, columns = ['I_real','I_fit','I_predict','R0'])
    result_I['I_real'] = I_real
    result_I['I_fit'] = I_fit
    result_I['I_predict'] = I_predict
    result_I['R0'] = R0_list
    
    result_para = {'Para':Para_1}
    result_para = pd.DataFrame(result_para)
    t_end_str_datetime  = datetime.datetime.strptime(t_end_str,'%Y-%m-%d') 
    delta30 = datetime.timedelta(days=days)
    t_end_str_datetime_30 = t_end_str_datetime+delta30 #这是datetime结构需要转化为str
    t_end_str_datetime_30 = datetime.datetime.strftime(t_end_str_datetime_30,"%Y-%m-%d")#转换为str
    result_I.index = pd.date_range(t_start_str,t_end_str_datetime_30)
    return result_I ,r2, result_para

def SEIQR(country_list,t_start_str,t_end_str,days):
    
    t_end_str_datetime  = datetime.datetime.strptime(t_end_str,'%Y-%m-%d') 
    delta30 = datetime.timedelta(days=days)
    t_end_str_datetime_30 = t_end_str_datetime+delta30 #这是datetime结构需要转化为str
    t_end_str_datetime_30 = datetime.datetime.strftime(t_end_str_datetime_30,"%Y-%m-%d")#转换为str
    
    t_30 = pd.date_range(t_start_str,t_end_str_datetime_30)
    
    data_predict_country = pd.DataFrame(data = None, columns = country_list)

#计算并将各国结果合并为一个dataframe
    for country in country_list:
        try:
            result_I, r2,result_para = fit_predict(country,t_start_str,t_end_str,days)
            I_predict = result_I['I_predict']
            data_predict_country[country] = I_predict
        except Exception as e:
            print(e)
            pass
        continue
    data_predict_country.index = t_30  
    data_predict_country = data_predict_country.fillna(0)
    data_predict_country = data_predict_country.astype(int)#取整的问题
#    max_number = max(data_predict_country.max())
#绘图
    t = Timeline()
    for i in t_30:
        draw_data = list(data_predict_country.loc[i])
        for j in range(0,len(country_list)):#国家名矫正
            if country_list[j] == 'South Korea':
                country_list[j] = 'Korea'
        c = (
        Map()
        .add(None, [list(z) for z in zip(country_list, draw_data)], "world",is_map_symbol_show = False)#countrylist需要修正
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="World"),
            visualmap_opts=opts.VisualMapOpts(max_=500000 ))#尺度的问题
        )
        date = datetime.datetime.strftime(i,"%Y-%m-%d")
        t.add(c, date)#只提取年月日
    t.add_schema(
        symbol = 'rect',# 设置标记时间；
        #orient = 'vertical',
         symbol_size = '3.5',# 标记大小;
        play_interval = 40,# 播放时间间隔；
        control_position = 'left',# 控制位置;
        is_loop_play = False,
        linestyle_opts = opts.LineStyleOpts(width = 5,type_ = 'solid'),#(193,205,205)
        label_opts = opts.LabelOpts(    font_size = 12,
                                        font_style = 'normal',
                                        font_weight = 'bold',
                                        font_family ='Time New Roman',
                                    position = 'bottom',
                                    interval = 20
                                        )
    )
    t.render("SEIQR_map.html")
    return

def fit_predict_multi(country,t_start_str_list,t_end_str_list,days):#若为连续则上一个结束时间为下一个开始时间
    data , population= get_data(country,t_start_str_list[0],t_end_str_list[-1])
    
    def trans(date):#将日期转换为所需要的索引 注意定义END时要加一因为左闭右开
        data_date = data['date']
        for i in range(0,len(data_date)):
            if data_date[i]==date:
                t = i
                break
        return t

    q2 = 0.1
    delta = 0.25    
    number = population# 总人数
    
    def lamda(t):#拟合lamda(t)
        list = data['I']
        list = list.to_list()
        tEnd_ = len(list)+1  # 拟合日期长度,使用所有数据拟合
        t_ = np.arange(1, tEnd_, 1)  # (start,stop,step) 左闭右开
        data_lamda = data["lamda"]
        data_lamda = np.array(data_lamda)
        r = np.polyfit(t_, data_lamda, 3)
        return r[0]*t**3+r[1]*t**2+r[2]*t**1+r[3]-(0.1-q2)
    
    def u(t):#拟合u(t)
        list = data['I']
        list = list.to_list()
        tEnd_ = len(list)+1  # 拟合日期长度,使用所有数据拟合
        data_u = data["people_fully_vaccinated_per_hundred"]
        t_start_u = len(data_u)
        for i in range(0,len(data_u)):
            if data_u[i]!=0:
                t_start_u = i
                break
        if t_start_u != len(data_u): #如果找到了开始时间则正常返回
            t_ = np.arange(t_start_u+1 ,tEnd_, 1)  # (start,stop,step)
            data_u = data_u[t_start_u:]
            data_u = np.array(data_u)
            r = np.polyfit(t_, data_u, 3)                
            return (r[0]*t**3+r[1]*t**2+r[2]*t**1+r[3])*0.01
        else:#否则返回0
            return 0*t
    
    def SEIR_v(y, t, lamda ,delta, mu ,q1 ,q2 ,q3 ,a, b ,u, p1,p2):#, p2
        S,E,I,Q,R = y 
        dS_dt =  -q1*S -lamda(t)*(S/number)*I-p1*u(t)*S+a*Q +p2*R
        dE_dt = lamda(t)*(S/number)*I - delta*E - q2*E  # 
        dI_dt = delta*E - mu*I-q3*I # 
        dQ_dt = q1*S+q2*E+q3*I-b*Q-a*Q
        dR_dt = mu*I+p1*u(t)*S-p2*R+b*Q
        return np.array([dS_dt,dE_dt,dI_dt,dQ_dt,dR_dt])
    
    def error_v(pr,t,y,Y0_1):
        q1,q3,a,b,mu,p1,p2 = pr[0],pr[1],pr[2],pr[3],pr[4],pr[5],pr[6]
        ySEIR = odeint(SEIR_v, Y0_1, t, args=(lamda,delta,mu,q1,q2,q3,a,b,u,p1,p2)) #注意这里的初值 ,p2
        return (ySEIR[:,2]-y)**2
    
    
    def stage_v(t_start_str,t_end_str,Y0): #xxxx/xx/xx
        data_lamda = data['lamda']
        R0_list = []
        p0=[0.5,0.5,0.1,0.5,0.5,0.002,0.8]
        bounds_v = ([0,0,0,0,0,0,0],[1,1,1,1,1,1,1])
        t_start = trans(t_start_str)
        t_end = trans(t_end_str)+1
        y_real = np.array(data["I"][t_start:t_end])#获取阶段数据 这里是作为索引
        t_train=np.arange(t_start+1, t_end+1, 1)#这里的序列仅为下面的最小二乘训练而用 这里作为计算
        Para=least_squares(error_v,p0,args=(t_train,y_real,Y0),bounds = bounds_v) 
        pr = Para.x
        q1,q3,a,b,mu,p1,p2 = pr[0],pr[1],pr[2],pr[3],pr[4],pr[5],pr[6]
        t_predict = np.arange(t_start+1, t_end+1, 1)#这里的序列是预测的时间 这里作为计算
        y_predict = odeint(SEIR_v, Y0, t_predict, args=(lamda,delta,mu,q1,q2,q3,a,b,u,p1,p2)) 
        for i in range(t_start,t_end):
            R0 = (data_lamda[i]*0.25)/(0.35*(mu+q3))
            R0_list.append(R0)
        r2 = r2_score(y_real,y_predict[:,2])#计算阶段一r^2
        Y0_next=y_predict[t_end-t_start-1,:]
        return  y_real,y_predict,pr,r2,R0_list,Y0_next
    
    I0 = data['I'][0] # 患病者的初值
    E0 = data['E'][0]         # 潜伏者的初值
    r0 = data['R'][0] # 被治愈的人的初值，一开始可设为0 
    Q0=number*0.1
    S0 = number-I0-E0-r0-Q0  # 易感者的初值
    Y0 = (S0, E0, I0, Q0, r0)  # 微分方程组的初值
    
    I_fit_r = []
    I_real_r = []
    columns = []
    r2_list = []
    R0_list_r= []
    for k in range(0,len(t_start_str_list)):
        columns.append(k)
    result_para = pd.DataFrame(data = None, columns = columns)
    result_Y0 = pd.DataFrame(data = None, columns = columns)
    for k in range(0,len(t_start_str_list)):
        t_start_str = t_start_str_list[k]
        t_end_str = t_end_str_list[k]
        y_real,y_fit,Para_1,r2,R0_list,Y0_next=stage_v(t_start_str,t_end_str,Y0)
        r2_list.append(r2)
        R0_list_r = R0_list_r+R0_list
        Y0 = Y0_next
        result_Y0[k] = list(Y0)
        I_fit = y_fit[:,2]
        I_real = y_real
        I_fit_r = np.append(I_fit_r,I_fit)
        I_real_r = np.append(I_real_r,I_real)
        result_para[k] = Para_1        
        
    t_start_predict = trans(t_start_str_list[-1])
    t_end_predict = trans(t_end_str_list[-1])
    Para_last = result_para.iloc[:,-1]
    q1=Para_last[0]
    q3=Para_last[1]
    a=Para_last[2]
    b=Para_last[3]
    mu= Para_last[4]
    p1=Para_last[5]
    p2=Para_last[6]
    t_predict_30 = np.arange(t_start_predict+1, t_end_predict+days+1, 1)
    y_predict_30 = odeint(SEIR_v, result_Y0.iloc[:,-2], t_predict_30, args=(lamda,delta,mu,q1,q2,q3,a,b,u,p1,p2))
    I_predict = y_predict_30[:,2]
    len_add = trans(t_end_str_list[-1])-trans(t_start_str_list[-1])
    I_predcit_add = I_predict[len_add:]
    I_predict_r = np.append(I_fit_r ,I_predcit_add)
    i=0
    for i in range(0,len(I_predict_r)-len(I_fit_r)):
        I_fit_r = np.append(I_fit_r,np.nan) 
        I_real_r = np.append(I_real_r,np.nan)
        R0_list_r.append(np.nan)
    
    result_I = pd.DataFrame(data = None, columns = ['I_real','I_fit','I_predict','R0'])
    result_I['I_real'] = I_real_r
    result_I['I_fit'] = I_fit_r
    result_I['I_predict'] = I_predict_r
    result_I['R0'] = R0_list_r
    t_end_str_datetime  = datetime.datetime.strptime(t_end_str_list[-1],'%Y-%m-%d') 
    delta30 = datetime.timedelta(days=days)
    t_end_str_datetime_30 = t_end_str_datetime+delta30 #这是datetime结构需要转化为str
    t_end_str_datetime_30 = datetime.datetime.strftime(t_end_str_datetime_30,"%Y-%m-%d")#转换为str
    result_I.index = pd.date_range(t_start_str_list[0],t_end_str_datetime_30)
    return result_I ,r2_list, result_para