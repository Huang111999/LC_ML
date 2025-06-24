import os
import re
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator

def calculate_c_index(real, predict, label, last_prob):
    
    index_all = 0
    index_true = 0
    for i in range(len(real)):
        for j in range(len(predict)):
            if label[i] == 0 and label[j] == 0:
                continue
            elif label[i] == 0 and label[j] == 1 and real[i] < real[j]:
                continue
            elif label[i] == 1 and label[j] == 0 and real[i] > real[j]:
                continue
            elif predict[i] == predict[j]:
                index_all += 1
                if (real[i] > real[j] and last_prob[i] > last_prob[j]) or (real[i] < real[j] and last_prob[i] < last_prob[j]):
                    index_true += 1
            elif i != j:
                index_all += 1
                if (real[i] > real[j] and predict[i] > predict[j]) or (real[i] < real[j] and predict[i] < predict[j]):
                    index_true += 1
    
    C_index = index_true / index_all
    
    return(C_index)
    
def get_AUC(real_what, predict, time):

    a = real_what.copy()
    a[a <= time-0.5] = 0
    a[a > 0] = 1
    
    value = roc_auc_score(a, predict)

    return(value)

def split_data(in_data, os_month, test_ratio = 0.2, new_method = False):
    
    raw_data = in_data.sort_values(by='OS(month)').copy()
    raw_data = raw_data[["性别(男1女0)","年龄","年龄≤60","吸烟史","包年","侧肺 左1 右2","肺叶上中1下2","切除方式（亚肺叶1，肺叶2）"
                   ,"病理级别","侵犯肺膜","脉管侵犯","肿瘤最大径","肿瘤最大径（》2cm 1，余0）","N",'T',
                   "分期2","CEA（》5为1）","BMI","BMI（<18.5 为1，18.5-24为2，余3）", "是否复发", "DFS(month)","是否死亡","OS(month)"]]
    
    if new_method:
        data_survive = raw_data[(raw_data['OS(month)'] >= os_month)]
        data_dead = raw_data[(raw_data['OS(month)'] < os_month) & (raw_data['是否死亡'] == 1)]
        data_dir = os.path.join(f'{os_month}_month_survival', 'data')
    else:
        data_survive = raw_data[raw_data['是否死亡'] == 0]
        data_dead = raw_data[raw_data['是否死亡'] == 1]
        data_dir = os.path.join(f'input_data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 打印可用数据量
    print(f'生存量{data_survive.shape[0]}, 死亡量{data_dead.shape[0]}, 总量{data_survive.shape[0] + data_dead.shape[0]}')
    
    # 取出test数据
    data_survive_test = data_survive.iloc[::int(1/test_ratio)]
    data_dead_test = data_dead.iloc[::int(1/test_ratio)]
    data_test = pd.concat([data_survive_test, data_dead_test])
    # 将index设置为"新编号"列, 保存test数据
    data_test = data_test.reset_index()
    data_test['index'] = data_test.index
    data_test.to_excel(os.path.join(data_dir, 'data_test.xlsx'), index = False)
    
    # 剩余的为train和val数据
    data_survive_train = data_survive.drop(data_survive_test.index)
    data_dead_train = data_dead.drop(data_dead_test.index)
    
    # 合并data_train
    data_train = pd.concat([data_survive_train, data_dead_train])
    # 将index设置为"index"列
    data_train = data_train.sample(frac = 1).reset_index()
    data_train['index'] = data_train.index
    # 重新将“编号”列设置为index
    data_train = data_train.set_index('编号')
    data_survive_train = data_train.loc[data_survive_train.index]
    data_dead_train = data_train.loc[data_dead_train.index]
    
    # 5折交叉验证划分train和val数据，分别保存到input_data
    for fold in range(0, 5):
        
        data_survive_val_fold = data_survive_train.iloc[fold::5]
        data_dead_val_fold = data_dead_train.iloc[fold::5]
        
        data_survive_train_fold = data_survive_train.drop(data_survive_val_fold.index)
        data_dead_train_fold = data_dead_train.drop(data_dead_val_fold.index)
        
        data_survive_val_fold['train'] = 0
        data_dead_val_fold['train'] = 0
        data_survive_train_fold['train'] = 1
        data_dead_train_fold['train'] = 1
        
        # 合并
        data_fold = pd.concat([data_survive_train_fold, data_dead_train_fold, data_survive_val_fold, data_dead_val_fold])
        
        # 保存
        data_fold.to_excel(os.path.join(data_dir, f'data_fold{fold}.xlsx'))
        
    return data_dir

def load_train_data(data_path):
    data = {}
    for data_file in os.listdir(data_path):
        if not 'fold' in data_file:
            continue
        fold_num = int(data_file.split('_fold')[-1].split('.')[0])
        current_data = pd.read_excel(os.path.join(data_path, data_file))
        data[fold_num] = current_data.set_index('index')
        
    return data

def get_predict_suvr(value, step):
    
    value2 = value.copy()
    
    value2[value2 >= 0.5] = 1
    value2[value2 < 0.5] = 0
    
    time_se = np.sum(value2, axis=1, dtype=int)
    
    predict_time = []
    for i in range(len(time_se)):
        if time_se[i] == len(step):
            predict_time.append(step[-1])
        else:
            upper = value[i,time_se[i]-1]
            lower = value[i,time_se[i]]
            t1 = step[time_se[i]-1]
            t2 = step[time_se[i]]
    
            t = (0.5 - upper) / (lower - upper) * (t2 - t1) + t1
            
            predict_time.append(t)
    
    return(predict_time)

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def train(data, model_DFS, model_OS, model_name, lasso_param=0.005):

    feature_columns = ["性别(男1女0)","年龄","吸烟史","包年","侧肺 左1 右2","肺叶上中1下2","切除方式（亚肺叶1，肺叶2）"
                       ,"病理级别","侵犯肺膜","脉管侵犯","肿瘤最大径","N",'T',
                       "分期2","CEA（》5为1）","BMI"]
    
    ouput_dir = f"output_{model_name}"
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
        
    if model_name == 'Lasso':
        output_result = f"{ouput_dir}/result.{model_name}.{str(lasso_param).replace('.','_')}.xlsx"
        output_QA = f"{ouput_dir}/QA.{model_name}.{str(lasso_param).replace('.','_')}.xlsx"
    else:
        output_result = f"{ouput_dir}/result.{model_name}.xlsx"
        output_QA = f"{ouput_dir}/QA.{model_name}.xlsx"

    # 初始化
    data_len = data[0].shape[0]
        
    real_recu = np.zeros(data_len, dtype=int)
    real_dead = np.zeros(data_len, dtype=int)
    real_DFS = np.zeros(data_len, dtype=int)
    real_OS = np.zeros(data_len, dtype=int)
    predict_DFS = np.zeros(data_len)
    predict_OS = np.zeros(data_len)
    last_prob_DFS = np.zeros(data_len)
    last_prob_OS = np.zeros(data_len)
    risk_DFS = np.zeros([data_len])
    risk_OS = np.zeros([data_len])
    dt_DFS = np.dtype([('recu', bool), ('DFS', np.float64)])
    dt_OS = np.dtype([('dead', bool), ('OS', np.float64)])
    all_Y_DFS = np.array([[(False, 0) for x in range(data_len)]], dtype=dt_DFS).reshape(-1)
    all_Y_OS = np.array([[(False, 0) for x in range(data_len)]], dtype=dt_OS).reshape(-1)
    
    times = np.asarray([24,30,36,42,48,54,60,66,72], dtype=int)
    DFS_times = {}
    OS_times = {}
    for i in range(len(times)):
        DFS_times["DFS_" + str(times[i])] = np.zeros(data_len)
        OS_times["OS_" + str(times[i])] = np.zeros(data_len)

    for i in range(5):
        
        print(i)

        # 数据划分
        current_data = data[i]

        current_train_data = current_data.loc[current_data['train'] == 1]
        current_val_data = current_data.loc[current_data['train'] == 0]

        current_train_data['T'] = current_train_data['T'].map({"1a":1,"1b":1,"1c":1,"2a":2,"2b":2,"3":3,3:3})
        current_val_data['T'] = current_val_data['T'].map({"1a":1,"1b":1,"1c":1,"2a":2,"2b":2,"3":3,3:3})

        # index
        index = np.asarray(current_val_data.index)

        # X
        train_X = current_train_data[feature_columns]
        val_X = current_val_data[feature_columns]

        # X 标准化
        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)

        # Y
        train_DFS = current_train_data['DFS(month)']
        train_OS = current_train_data['OS(month)']
        val_DFS = current_val_data['DFS(month)']
        val_OS = current_val_data['OS(month)']

        real_DFS[index] = val_DFS
        real_OS[index] = val_OS

        train_dead = current_train_data['是否死亡']
        train_recu = current_train_data['是否复发']
        val_dead = current_val_data['是否死亡']
        val_recu = current_val_data['是否复发']

        real_recu[index] = val_recu
        real_dead[index] = val_dead

        dt_DFS = np.dtype([('recu', bool), ('DFS', np.float64)])
        dt_OS = np.dtype([('dead', bool), ('OS', np.float64)])

        train_Y_DFS = np.array([[(train_recu[x], train_DFS[x]) for x in current_train_data.index]], dtype=dt_DFS).reshape(-1)
        train_Y_OS = np.array([[(train_recu[x], train_OS[x]) for x in current_train_data.index]], dtype=dt_OS).reshape(-1)
        val_Y_DFS = np.array([[(val_recu[x], val_DFS[x]) for x in current_val_data.index]], dtype=dt_DFS).reshape(-1)
        val_Y_OS = np.array([[(val_recu[x], val_OS[x])  for x in current_val_data.index]], dtype=dt_OS).reshape(-1)
        
        all_Y_DFS[index] = val_Y_DFS
        all_Y_OS[index] = val_Y_OS

        # train
        model_DFS.fit(train_X, train_Y_DFS)
        model_OS.fit(train_X, train_Y_OS)

        step_DFS = model_DFS.unique_times_
        step_DFS = [int(x) for x in step_DFS]
        step_OS = model_OS.unique_times_
        step_OS = [int(x) for x in step_OS]
        
        if model_name == 'Lasso':
            predict_DFS_array = model_DFS.predict_survival_function(val_X, alpha=lasso_param, return_array=True)
            predict_OS_array = model_OS.predict_survival_function(val_X, alpha=lasso_param, return_array=True)
        else:
            predict_DFS_array = model_DFS.predict_survival_function(val_X, return_array=True)
            predict_OS_array = model_OS.predict_survival_function(val_X, return_array=True)

        predict_time_DFS = get_predict_suvr(predict_DFS_array, step_DFS)
        predict_time_OS = get_predict_suvr(predict_OS_array, step_OS)

        predict_DFS[index] = predict_time_DFS
        predict_OS[index] = predict_time_OS
        last_prob_DFS[index] = predict_DFS_array[:,-1]
        last_prob_OS[index] = predict_OS_array[:,-1]
        
        if model_name == 'Lasso':
            step_predict_DFS = model_DFS.predict_survival_function(val_X, alpha=lasso_param)
            step_predict_OS = model_OS.predict_survival_function(val_X, alpha=lasso_param)
            risk_DFS[index] = model_DFS.predict(val_X, alpha=lasso_param)
            risk_OS[index] = model_OS.predict(val_X, alpha=lasso_param)
        else:
            step_predict_DFS = model_DFS.predict_survival_function(val_X)
            step_predict_OS = model_OS.predict_survival_function(val_X)
            risk_DFS[index] = model_DFS.predict(val_X)
            risk_OS[index] = model_OS.predict(val_X)

        for j in range(len(times)):
            DFS_times["DFS_" + str(times[j])][index] = [step_predict_DFS[x](times[j]) for x in range(len(step_predict_DFS))]
            OS_times["OS_" + str(times[j])][index] = [step_predict_OS[x](times[j]) for x in range(len(step_predict_OS))]

    
    result = pd.DataFrame()
    result['real_recu'] = real_recu
    result['real_dead'] = real_dead
    result['real_DFS'] = real_DFS
    result['real_OS'] = real_OS
    for i in range(len(times)):
        result['DFS_' + str(times[i])] = DFS_times['DFS_' + str(times[i])]
        result['OS_' + str(times[i])] = OS_times['OS_' + str(times[i])]
    result['risk_DFS'] = risk_DFS
    result['risk_OS'] = risk_OS
    result.to_excel(output_result, index=False)

    # C-index
    DFS_c_index, _, _, _, _ = concordance_index_censored(np.asarray(real_recu, dtype=bool), real_DFS, risk_DFS)
    OS_c_index, _, _, _, _ = concordance_index_censored(np.asarray(real_dead, dtype=bool), real_OS, risk_OS)

    print("C-index:", DFS_c_index, OS_c_index)


    AUC_DFS, mean_auc = cumulative_dynamic_auc(all_Y_DFS, all_Y_DFS, risk_DFS, times)
    AUC_OS, mean_auc = cumulative_dynamic_auc(all_Y_OS, all_Y_OS, risk_OS, times)

    df = pd.DataFrame()

    for i in range(len(times)):
        df['DFS_AUC_' + str(times[i])] = [AUC_DFS[i]]
        df['OS_AUC_' + str(times[i])] = [AUC_OS[i]]
    df['DFS_c_index'] = [DFS_c_index]
    df['OS_c_index'] = [OS_c_index]
    df = df.T
    
    print(df)
    
    df.to_excel(output_QA)

def test(model_DFS, model_OS, model_name, lasso_param=0.005):

    feature_columns = ["性别(男1女0)","年龄","吸烟史","包年","侧肺 左1 右2","肺叶上中1下2","切除方式（亚肺叶1，肺叶2）"
                       ,"病理级别","侵犯肺膜","脉管侵犯","肿瘤最大径","N",'T',
                       "分期2","CEA（》5为1）","BMI"]
    
    ouput_dir = f"output_test_{model_name}"
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
        
    if model_name == 'Lasso':
        output_result = f"{ouput_dir}/result.{model_name}.{str(lasso_param).replace('.','_')}.xlsx"
        output_QA = f"{ouput_dir}/QA.{model_name}.{str(lasso_param).replace('.','_')}.xlsx"
    else:
        output_result = f"{ouput_dir}/result.{model_name}.xlsx"
        output_QA = f"{ouput_dir}/QA.{model_name}.xlsx"
        
    train_data = pd.read_excel('input_data/data_fold0.xlsx')
    test_data = pd.read_excel('input_data/data_test.xlsx')

    # 初始化
    data_len = test_data.shape[0]
    real_recu = np.zeros(data_len, dtype=int)
    real_dead = np.zeros(data_len, dtype=int)
    real_DFS = np.zeros(data_len, dtype=int)
    real_OS = np.zeros(data_len, dtype=int)
    predict_DFS = np.zeros(data_len)
    predict_OS = np.zeros(data_len)
    last_prob_DFS = np.zeros(data_len)
    last_prob_OS = np.zeros(data_len)
    risk_DFS = np.zeros([data_len])
    risk_OS = np.zeros([data_len])
    dt_DFS = np.dtype([('recu', bool), ('DFS', np.float64)])
    dt_OS = np.dtype([('dead', bool), ('OS', np.float64)])
    all_Y_DFS = np.array([[(False, 0) for x in range(data_len)]], dtype=dt_DFS).reshape(-1)
    all_Y_OS = np.array([[(False, 0) for x in range(data_len)]], dtype=dt_OS).reshape(-1)
    times = np.asarray([24,30,36,42,48,54,60,66,72], dtype=int)
    DFS_times = {}
    OS_times = {}
    for i in range(len(times)):
        DFS_times["DFS_" + str(times[i])] = np.zeros(data_len)
        OS_times["OS_" + str(times[i])] = np.zeros(data_len)

    current_train_data = train_data
    current_val_data = test_data

    current_train_data['T'] = current_train_data['T'].map({"1a":1,"1b":1,"1c":1,"2a":2,"2b":2,"3":3,3:3})
    current_val_data['T'] = current_val_data['T'].map({"1a":1,"1b":1,"1c":1,"2a":2,"2b":2,"3":3,3:3})

    # index
    index = np.asarray(current_val_data.index)

    # X
    train_X = current_train_data[feature_columns]
    val_X = current_val_data[feature_columns]

    # X 标准化
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)

    # Y
    train_DFS = current_train_data['DFS(month)']
    train_OS = current_train_data['OS(month)']
    val_DFS = current_val_data['DFS(month)']
    val_OS = current_val_data['OS(month)']

    real_DFS[index] = val_DFS
    real_OS[index] = val_OS

    train_dead = current_train_data['是否死亡']
    train_recu = current_train_data['是否复发']
    val_dead = current_val_data['是否死亡']
    val_recu = current_val_data['是否复发']

    real_recu[index] = val_recu
    real_dead[index] = val_dead

    dt_DFS = np.dtype([('recu', bool), ('DFS', np.float64)])
    dt_OS = np.dtype([('dead', bool), ('OS', np.float64)])

    train_Y_DFS = np.array([[(train_recu[x], train_DFS[x]) for x in current_train_data.index]], dtype=dt_DFS).reshape(-1)
    train_Y_OS = np.array([[(train_recu[x], train_OS[x]) for x in current_train_data.index]], dtype=dt_OS).reshape(-1)
    val_Y_DFS = np.array([[(val_recu[x], val_DFS[x]) for x in current_val_data.index]], dtype=dt_DFS).reshape(-1)
    val_Y_OS = np.array([[(val_recu[x], val_OS[x])  for x in current_val_data.index]], dtype=dt_OS).reshape(-1)

    all_Y_DFS[index] = val_Y_DFS
    all_Y_OS[index] = val_Y_OS

    # train
    model_DFS.fit(train_X, train_Y_DFS)
    model_OS.fit(train_X, train_Y_OS)

    step_DFS = model_DFS.unique_times_
    step_DFS = [int(x) for x in step_DFS]
    step_OS = model_OS.unique_times_
    step_OS = [int(x) for x in step_OS]

    if model_name == 'Lasso':
        predict_DFS_array = model_DFS.predict_survival_function(val_X, alpha=lasso_param, return_array=True)
        predict_OS_array = model_OS.predict_survival_function(val_X, alpha=lasso_param, return_array=True)
    else:
        predict_DFS_array = model_DFS.predict_survival_function(val_X, return_array=True)
        predict_OS_array = model_OS.predict_survival_function(val_X, return_array=True)

    predict_time_DFS = get_predict_suvr(predict_DFS_array, step_DFS)
    predict_time_OS = get_predict_suvr(predict_OS_array, step_OS)

    predict_DFS[index] = predict_time_DFS
    predict_OS[index] = predict_time_OS
    last_prob_DFS[index] = predict_DFS_array[:,-1]
    last_prob_OS[index] = predict_OS_array[:,-1]

    if model_name == 'Lasso':
        step_predict_DFS = model_DFS.predict_survival_function(val_X, alpha=lasso_param)
        step_predict_OS = model_OS.predict_survival_function(val_X, alpha=lasso_param)
        risk_DFS[index] = model_DFS.predict(val_X, alpha=lasso_param)
        risk_OS[index] = model_OS.predict(val_X, alpha=lasso_param)
    else:
        step_predict_DFS = model_DFS.predict_survival_function(val_X)
        step_predict_OS = model_OS.predict_survival_function(val_X)
        risk_DFS[index] = model_DFS.predict(val_X)
        risk_OS[index] = model_OS.predict(val_X)

    for j in range(len(times)):
        DFS_times["DFS_" + str(times[j])][index] = [step_predict_DFS[x](times[j]) for x in range(len(step_predict_DFS))]
        OS_times["OS_" + str(times[j])][index] = [step_predict_OS[x](times[j]) for x in range(len(step_predict_OS))]

    
    result = pd.DataFrame()
    result['real_recu'] = real_recu
    result['real_dead'] = real_dead
    result['real_DFS'] = real_DFS
    result['real_OS'] = real_OS
    for i in range(len(times)):
        result['DFS_' + str(times[i])] = DFS_times['DFS_' + str(times[i])]
        result['OS_' + str(times[i])] = OS_times['OS_' + str(times[i])]
    result['risk_DFS'] = risk_DFS
    result['risk_OS'] = risk_OS
    result.to_excel(output_result, index=False)

    # C-index
    DFS_c_index, _, _, _, _ = concordance_index_censored(np.asarray(real_recu, dtype=bool), real_DFS, risk_DFS)
    OS_c_index, _, _, _, _ = concordance_index_censored(np.asarray(real_dead, dtype=bool), real_OS, risk_OS)

    print("C-index:", DFS_c_index, OS_c_index)


    AUC_DFS, mean_auc = cumulative_dynamic_auc(all_Y_DFS, all_Y_DFS, risk_DFS, times)
    AUC_OS, mean_auc = cumulative_dynamic_auc(all_Y_OS, all_Y_OS, risk_OS, times)

    df = pd.DataFrame()

    for i in range(len(times)):
        df['DFS_AUC_' + str(times[i])] = [AUC_DFS[i]]
        df['OS_AUC_' + str(times[i])] = [AUC_OS[i]]
    df['DFS_c_index'] = [DFS_c_index]
    df['OS_c_index'] = [OS_c_index]
    df = df.T
    
    print(df)
    
    df.to_excel(output_QA)

def plot_calibration_curve(test = False):

    tag = 'OS'
    
    test_path = '_test' if test else ''

    result_Cox = pd.read_excel(f"output{test_path}_Cox/result.Cox.xlsx")
    result_RSF = pd.read_excel(f"output{test_path}_RSF/result.RSF.xlsx")
    result_Lasso = pd.read_excel(f"output{test_path}_Lasso/result.Lasso.0_005.xlsx")
    result_GB = pd.read_excel(f"output{test_path}_GB/result.GB.xlsx")

    times = np.asarray([24,30,36,42,48,54,60,66,72], dtype=int)

    if tag == 'DFS':
        real_data = result_RSF['real_recu']
        show_y = "recurrence"
    else:
        real_data = result_RSF['real_dead']
        show_y = 'dead'

    for time in times:

        predict_RSF = result_RSF[tag + '_' + str(time)]
        predict_Lasso = result_Lasso[tag + '_' + str(time)]
        predict_Cox = result_Cox[tag + '_' + str(time)]
        predict_GB = result_GB[tag + '_' + str(time)]
        #predict_data = result["risk_" + tag]



        RSF_true, RSF_pred = calibration_curve(1 - real_data, predict_RSF, n_bins=4,  strategy="uniform")
        Cox_true, Cox_pred = calibration_curve(1 - real_data, predict_Cox, n_bins=4,  strategy="uniform")
        GB_true, GB_pred = calibration_curve(1 - real_data, predict_GB, n_bins=4,  strategy="uniform")
        Lasso_true, Lasso_pred = calibration_curve(1 - real_data, predict_Lasso, n_bins=4,  strategy="uniform")
        #quantile  uniform

        if time % 12 == 0:
            show_year = str(int(time / 12))
        else:
            show_year = str(time / 12)


        plt.figure(figsize=(6,4))
        plt.plot(Cox_pred, Cox_true, label='Cox', c = 'tomato', lw = 3)
        plt.plot(RSF_pred, RSF_true, label='RSF', c = 'limegreen', lw = 3)  
        plt.plot(GB_pred, GB_true, label='GB', c = 'purple', lw = 3)  
        plt.plot(Lasso_pred, Lasso_true, label='Lasso', c = 'deepskyblue', lw = 3)  
        plt.scatter(Cox_pred, Cox_true, s = 50, c = 'tomato' )
        plt.scatter(RSF_pred, RSF_true, s = 50, c = 'limegreen' )
        plt.scatter(GB_pred, GB_true, s = 50, c = 'purple' )
        plt.scatter(Lasso_pred, Lasso_true, s = 50, c = 'deepskyblue' )
        plt.plot([0,1],[0,1], '--', c='grey', label = 'Perfectly calibrated')
        plt.legend(loc='lower right')

        plt.xlabel("Predicted probability of " + show_year +  "-years " + tag, fontsize=12)
        plt.ylabel("Fraction of " + show_y, fontsize=12)
        plt.savefig(f"plot{test_path}/calibration_" + tag + "_" + str(time) + ".png", dpi=300,bbox_inches='tight')
        plt.savefig(f"plot{test_path}/calibration_" + tag + "_" + str(time) + ".pdf",bbox_inches='tight')
        

def plot_DCA(test = False):
    
    test_path = '_test' if test else ''
    
    result_Cox = pd.read_excel(f"output{test_path}_Cox/result.Cox.xlsx")
    result_RSF = pd.read_excel(f"output{test_path}_RSF/result.RSF.xlsx")
    result_GB = pd.read_excel(f"output{test_path}_GB/result.GB.xlsx")
    result_Lasso = pd.read_excel(f"output{test_path}_Lasso/result.Lasso.0_005.xlsx")

    
    tag = 'DFS'

    times = np.asarray([24,30,36,42,48,54,60,66,72], dtype=int)
    
    
    if tag == 'DFS':
        y_label = np.asarray(result_RSF['real_recu'])
    else:
        y_label = np.asarray(result_RSF['real_dead'])
        
        
    for time in times:
        
        y_pred_Cox = 1 - np.asarray(result_Cox[tag + '_' + str(time)])
        y_pred_RSF = 1 - np.asarray(result_RSF[tag + '_' + str(time)])
        y_pred_GB = 1 - np.asarray(result_GB[tag + '_' + str(time)])
        y_pred_Lasso = 1 - np.asarray(result_Lasso[tag + '_' + str(time)])
        
        #y_pred_score = np.arange(0, 1, 0.001)
        #y_label = np.array([1]*25 + [0]*25 + [0]*450 + [1]*25 + [0]*25+ [1]*25 + [0]*25 + [1]*25 + [0]*25+ [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*50 + [1]*125)
    
        thresh_group = np.arange(0,1,0.01)
        net_benefit_Cox = calculate_net_benefit_model(thresh_group, y_pred_Cox, y_label)
        net_benefit_RSF = calculate_net_benefit_model(thresh_group, y_pred_RSF, y_label)
        net_benefit_GB = calculate_net_benefit_model(thresh_group, y_pred_GB, y_label)
        net_benefit_Lasso = calculate_net_benefit_model(thresh_group, y_pred_Lasso, y_label)
        net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
        
        #ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
        
        if time % 12 == 0:
            show_year = str(int(time / 12))
        else:
            show_year = str(time / 12)
        
        
        plt.figure(figsize=(6,4))
        plt.plot(thresh_group, net_benefit_Cox, color = 'tomato', label = 'Cox')
        plt.plot(thresh_group, net_benefit_RSF, color = 'limegreen', label = 'RSF')
        plt.plot(thresh_group, net_benefit_GB, color = 'purple', label = 'GB')
        plt.plot(thresh_group, net_benefit_Lasso, color = 'deepskyblue', label = 'Lasso')
        plt.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
        plt.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
    
        #Fill，显示出模型较于treat all和treat none好的部分
        #y2 = np.maximum(net_benefit_all, 0)
        #y1 = np.maximum(net_benefit_model, y2)
        #ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)
    
        #Figure Configuration， 美化一下细节
        plt.xlim(0,1)
        if tag == 'DFS':
            plt.ylim(-0.05, 0.2)
        elif tag == 'OS':
            plt.ylim(-0.05, 0.1)
        #ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
        plt.xlabel('Threshold Probability', fontsize= 12)
        plt.ylabel('Net Benefit', fontsize= 12)
        plt.grid('major')
        #plt.spines['right'].set_color((0.8, 0.8, 0.8))
        #plt.spines['top'].set_color((0.8, 0.8, 0.8))
        plt.legend(loc = 'upper right', fontsize= 12)
        plt.title("DCA for predicting " + show_year + "-year " + tag, fontsize= 12)
        
        # fig.savefig('fig1.png', dpi = 300)
        #plt.show()
        #plt.savefig("plot/DCA_" + tag + "_" + str(time) + ".png", dpi=300,bbox_inches='tight')
        plt.savefig(f"plot{test_path}/DCA_" + tag + "_" + str(time) + ".pdf", bbox_inches='tight')
        
def plot_kaplan_meier(test = False):

    test_path = '_test' if test else ''

    result = pd.read_excel(f"output{test_path}_Lasso/result.Lasso.0_005.xlsx")
    data = pd.read_excel("早期肺癌670例(1).xlsx")

    tag = 'OS'

    if tag == 'DFS':
        index1 = list(result.loc[result['risk_DFS'] >= 0].index)
        index2 = list(result.loc[result['risk_DFS'] < 0].index)
        event = np.asarray(data['是否复发'])
        time = np.asarray(data['DFS(month)'])
    else:
        index1 = list(result.loc[result['risk_OS'] >= 0].index)
        index2 = list(result.loc[result['risk_OS'] < 0].index)
        event = np.asarray(data['是否死亡'])
        time = np.asarray(data['OS(month)'])

    event = np.asarray(event, dtype=bool)

    plt.figure(figsize = (7,5))
    x1, y1, conf_int1 = kaplan_meier_estimator(event[index1], time[index1], conf_type="log-log")
    x2, y2, conf_int2 = kaplan_meier_estimator(event[index2], time[index2], conf_type="log-log")
    plt.step(x1, y1, where="post", c='tomato', label='High risk group')
    plt.fill_between(x1, conf_int1[0], conf_int1[1], alpha=0.25, step="post", facecolor='tomato')
    plt.step(x2, y2, where="post", c='deepskyblue', label='Low risk group')
    plt.fill_between(x2, conf_int2[0], conf_int2[1], alpha=0.25, step="post", facecolor='deepskyblue')
    plt.ylim(0, 1.02)
    plt.xlim(0, 93)
    plt.legend(loc='lower right')
    plt.xlabel(tag + "(mouth)")
    plt.ylabel("Overall Survival Probability")
    plt.savefig(f"plot{test_path}/kaplan_meier_" + tag + ".pdf", dpi=300,bbox_inches='tight')
    
def plot_AUC(test = False):
    
    test_path = '_test' if test else ''

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.metrics import confusion_matrix


    result_Cox = pd.read_excel(f"output{test_path}_Cox/QA.Cox.xlsx", index_col = 0)
    result_RSF = pd.read_excel(f"output{test_path}_RSF/QA.RSF.xlsx", index_col = 0)
    result_Lasso = pd.read_excel(f"output{test_path}_Lasso/QA.Lasso.0_005.xlsx", index_col = 0)
    result_GB = pd.read_excel(f"output{test_path}_GB/QA.GB.xlsx", index_col = 0)

    times = np.asarray([24,30,36,42,48,54,60,66,72], dtype=int)


    y_Cox_DFS = np.zeros(len(times))
    y_Cox_OS = np.zeros(len(times))
    y_RSF_DFS = np.zeros(len(times))
    y_RSF_OS = np.zeros(len(times))
    y_Lasso_DFS = np.zeros(len(times))
    y_Lasso_OS = np.zeros(len(times))
    y_GB_DFS = np.zeros(len(times))
    y_GB_OS = np.zeros(len(times))

    for i in range(len(times)):
        y_Cox_DFS[i] = result_Cox.T['DFS_AUC_' + str(times[i])]
        y_Cox_OS[i] = result_Cox.T['OS_AUC_' + str(times[i])]
        y_RSF_DFS[i] = result_RSF.T['DFS_AUC_' + str(times[i])]
        y_RSF_OS[i] = result_RSF.T['OS_AUC_' + str(times[i])]
        y_Lasso_DFS[i] = result_Lasso.T['DFS_AUC_' + str(times[i])]
        y_Lasso_OS[i] = result_Lasso.T['OS_AUC_' + str(times[i])]
        y_GB_DFS[i] = result_GB.T['DFS_AUC_' + str(times[i])]
        y_GB_OS[i] = result_GB.T['OS_AUC_' + str(times[i])]




    plt.figure(figsize=(6,4))
    plt.plot(times, y_Cox_DFS, label='Cox', c = 'tomato', lw = 3)
    plt.plot(times, y_RSF_DFS, label='RSF', c = 'limegreen', lw = 3) 
    plt.plot(times, y_GB_DFS, label='GB', c = 'purple', lw = 3)   
    plt.plot(times, y_Lasso_DFS, label='Lasso', c = 'deepskyblue', lw = 3) 
    plt.scatter(times, y_Cox_DFS, s = 50, c = 'tomato' )
    plt.scatter(times, y_RSF_DFS, s = 50, c = 'limegreen' )
    plt.scatter(times, y_GB_DFS, s = 50, c = 'purple')   
    plt.scatter(times, y_Lasso_DFS, s = 50, c = 'deepskyblue' )
    plt.legend(loc='lower left', fontsize=12) 
    plt.xlabel("DFS(month)", fontsize=12)
    plt.ylabel("Time-dependent AUC", fontsize=12) 
    plt.savefig(f"plot{test_path}/AUC_DFS.pdf", dpi=300,bbox_inches='tight')



    plt.figure(figsize=(6,4))
    plt.plot(times, y_Cox_OS, label='Cox', c = 'tomato', lw = 3)
    plt.plot(times, y_RSF_OS, label='RSF', c = 'limegreen', lw = 3) 
    plt.plot(times, y_GB_OS, label='GB', c = 'purple', lw = 3)    
    plt.plot(times, y_Lasso_OS, label='Lasso', c = 'deepskyblue', lw = 3) 
    plt.scatter(times, y_Cox_OS, s = 50, c = 'tomato' )
    plt.scatter(times, y_RSF_OS, s = 50, c = 'limegreen' )
    plt.scatter(times, y_GB_OS, s = 50, c = 'purple')   
    plt.scatter(times, y_Lasso_OS, s = 50, c = 'deepskyblue' )
    plt.legend(loc='lower left', fontsize=12) 
    plt.xlabel("OS(month)", fontsize=12)
    plt.ylabel("Time-dependent AUC", fontsize=12) 
    plt.savefig(f"plot{test_path}/AUC_OS.pdf", dpi=300,bbox_inches='tight')
    

# 划分数据
split_data(pd.read_excel('早期肺癌670例(1).xlsx', index_col=0), os_month = 60, new_method = False)

# 读取数据
data = load_train_data('input_data/')

# 5fold

# RSF
train(data = data, 
      model_DFS = RandomSurvivalForest(n_estimators=1000, n_jobs=-1, random_state=20),
      model_OS = RandomSurvivalForest(n_estimators=1000, n_jobs=-1, random_state=20),
      model_name = 'RSF')

# Cox
train(data = data, 
      model_DFS = CoxPHSurvivalAnalysis(), 
      model_OS = CoxPHSurvivalAnalysis(), 
      model_name = 'Cox')

# Lasso
for lasso_param in [0.005, 0.05]:
    train(data = data, 
          model_DFS = CoxnetSurvivalAnalysis(alphas=[lasso_param], l1_ratio=1.0, fit_baseline_model=True),
          model_OS =CoxnetSurvivalAnalysis(alphas=[lasso_param], l1_ratio=1.0, fit_baseline_model=True),
          model_name = 'Lasso', 
          lasso_param = lasso_param)

# GB
train(data = data, 
      model_DFS = GradientBoostingSurvivalAnalysis(n_estimators=1000, random_state=20), 
      model_OS = GradientBoostingSurvivalAnalysis(n_estimators=1000, random_state=20), 
      model_name = 'GB')

# test

# RSF
test(model_DFS = RandomSurvivalForest(n_estimators=1000, n_jobs=-1, random_state=20),
      model_OS = RandomSurvivalForest(n_estimators=1000, n_jobs=-1, random_state=20),
      model_name = 'RSF')

# Cox
test(model_DFS = CoxPHSurvivalAnalysis(), 
      model_OS = CoxPHSurvivalAnalysis(), 
      model_name = 'Cox')

# Lasso
for lasso_param in [0.005, 0.05]:
    test(model_DFS = CoxnetSurvivalAnalysis(alphas=[lasso_param], l1_ratio=1.0, fit_baseline_model=True),
          model_OS =CoxnetSurvivalAnalysis(alphas=[lasso_param], l1_ratio=1.0, fit_baseline_model=True),
          model_name = 'Lasso', 
          lasso_param = lasso_param)

# GB
test(model_DFS = GradientBoostingSurvivalAnalysis(n_estimators=1000, random_state=20), 
      model_OS = GradientBoostingSurvivalAnalysis(n_estimators=1000, random_state=20), 
      model_name = 'GB')

if not os.path.exists('plot'):
    os.makedirs('plot')
if not os.path.exists('plot_test'):
    os.makedirs('plot_test')

plot_calibration_curve(test = False)
plot_calibration_curve(test = True)

plot_DCA(test = False)
plot_DCA(test = True)

plot_kaplan_meier(test = False)
plot_kaplan_meier(test = True)

plot_AUC(test = False)
plot_AUC(test = True)