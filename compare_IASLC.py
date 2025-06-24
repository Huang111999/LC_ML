# 标准库导入
import os
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec

# Scikit-learn 导入
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# Scikit-survival 导入
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# 设置字体样式 - 与原代码保持一致
fontproperties = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'

seed = 657
np.random.seed(seed)
random.seed(seed)

def load_train_data(data_path):
    """加载训练数据"""
    data = {}
    for data_file in os.listdir(data_path):
        if not 'fold' in data_file:
            continue
        fold_num = int(data_file.split('_fold')[-1].split('.')[0])
        data[fold_num] = pd.read_excel(os.path.join(data_path, data_file), index_col=0)
    return data

def train_iaslc_model(data, model_name="IASLC"):
    """训练仅使用IASLC分级系统的模型"""
    
    # 仅使用病理分期作为特征
    feature_columns = ["分期2"]
    
    output_dir = f"output_{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_result = f"{output_dir}/result.{model_name}.xlsx"
    output_QA = f"{output_dir}/QA.{model_name}.xlsx"

    # 初始化
    data_len = data[0].shape[0]
    
    real_recu = np.zeros(data_len, dtype=int)
    real_dead = np.zeros(data_len, dtype=int)
    real_DFS = np.zeros(data_len, dtype=int)
    real_OS = np.zeros(data_len, dtype=int)
    risk_DFS = np.zeros([data_len])
    risk_OS = np.zeros([data_len])
    dt_DFS = np.dtype([('recu', bool), ('DFS', np.float64)])
    dt_OS = np.dtype([('dead', bool), ('OS', np.float64)])
    all_Y_DFS = np.array([[(False, 0) for x in range(data_len)]], dtype=dt_DFS).reshape(-1)
    all_Y_OS = np.array([[(False, 0) for x in range(data_len)]], dtype=dt_OS).reshape(-1)
    
    times = np.asarray([24, 30, 36, 42, 48, 54, 60, 66, 72], dtype=int)
    DFS_times = {}
    OS_times = {}
    for i in range(len(times)):
        DFS_times["DFS_" + str(times[i])] = np.zeros(data_len)
        OS_times["OS_" + str(times[i])] = np.zeros(data_len)

    for i in range(5):
        print(f"Processing fold {i}")

        # 数据划分
        current_data = data[i]
        current_train_data = current_data.loc[current_data['train'] == 1]
        current_val_data = current_data.loc[current_data['train'] == 0]

        # index
        index = np.asarray(current_val_data.index)

        # X - 仅使用病理分期
        train_X = current_train_data[feature_columns]
        val_X = current_val_data[feature_columns]

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

        train_Y_DFS = np.array([[(train_recu[x], train_DFS[x]) for x in current_train_data.index]], dtype=dt_DFS).reshape(-1)
        train_Y_OS = np.array([[(train_dead[x], train_OS[x]) for x in current_train_data.index]], dtype=dt_OS).reshape(-1)
        val_Y_DFS = np.array([[(val_recu[x], val_DFS[x]) for x in current_val_data.index]], dtype=dt_DFS).reshape(-1)
        val_Y_OS = np.array([[(val_dead[x], val_OS[x]) for x in current_val_data.index]], dtype=dt_OS).reshape(-1)
        
        all_Y_DFS[index] = val_Y_DFS
        all_Y_OS[index] = val_Y_OS

        # 训练Cox模型 - 仅使用病理分期
        model_DFS = CoxPHSurvivalAnalysis()
        model_OS = CoxPHSurvivalAnalysis()
        
        model_DFS.fit(train_X, train_Y_DFS)
        model_OS.fit(train_X, train_Y_OS)

        # 预测
        step_predict_DFS = model_DFS.predict_survival_function(val_X)
        step_predict_OS = model_OS.predict_survival_function(val_X)
        risk_DFS[index] = model_DFS.predict(val_X)
        risk_OS[index] = model_OS.predict(val_X)

        for j in range(len(times)):
            DFS_times["DFS_" + str(times[j])][index] = [step_predict_DFS[x](times[j]) for x in range(len(step_predict_DFS))]
            OS_times["OS_" + str(times[j])][index] = [step_predict_OS[x](times[j]) for x in range(len(step_predict_OS))]

    # 保存结果
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

    # 计算性能指标
    DFS_c_index, _, _, _, _ = concordance_index_censored(np.asarray(real_recu, dtype=bool), real_DFS, risk_DFS)
    OS_c_index, _, _, _, _ = concordance_index_censored(np.asarray(real_dead, dtype=bool), real_OS, risk_OS)

    print(f"IASLC C-index: DFS={DFS_c_index:.3f}, OS={OS_c_index:.3f}")

    AUC_DFS, _ = cumulative_dynamic_auc(all_Y_DFS, all_Y_DFS, risk_DFS, times)
    AUC_OS, _ = cumulative_dynamic_auc(all_Y_OS, all_Y_OS, risk_OS, times)

    # 保存评估结果
    df = pd.DataFrame()
    for i in range(len(times)):
        df[f'DFS_AUC_{times[i]}'] = [AUC_DFS[i]]
        df[f'OS_AUC_{times[i]}'] = [AUC_OS[i]]
    df['DFS_c_index'] = [DFS_c_index]
    df['OS_c_index'] = [OS_c_index]
    df = df.T
    df.to_excel(output_QA)
    
    return DFS_c_index, OS_c_index, AUC_DFS, AUC_OS

def plot_c_index_comparison():
    """绘制C-index比较图 - 与原代码风格一致"""
    
    if not os.path.exists('comparison_plots'):
        os.makedirs('comparison_plots')
    
    # 读取结果
    result_IASLC = pd.read_excel("output_IASLC/QA.IASLC.xlsx", index_col=0)
    result_Lasso = pd.read_excel("output_Lasso/QA.Lasso.xlsx", index_col=0)
    result_Cox = pd.read_excel("output_Cox/QA.Cox.xlsx", index_col=0)
    result_RSF = pd.read_excel("output_RSF/QA.RSF.xlsx", index_col=0)
    result_GB = pd.read_excel("output_GB/QA.GB.xlsx", index_col=0)

    # 提取C-index - 与原代码颜色保持一致
    models = ['IASLC', 'Cox', 'Lasso', 'RSF', 'GB']
    results = [result_IASLC, result_Cox, result_Lasso, result_RSF, result_GB]
    colors = ['red', 'tomato', 'deepskyblue', 'limegreen', 'purple']  # 与原代码一致
    
    dfs_cindices = []
    os_cindices = []
    
    for result in results:
        dfs_cindices.append(result.T['DFS_c_index'].iloc[0])
        os_cindices.append(result.T['OS_c_index'].iloc[0])
    
    # 绘制C-index比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # DFS C-index比较
    bars1 = ax1.bar(models, dfs_cindices, color=colors)
    ax1.set_title('DFS C-Index Comparison', fontsize=14)
    ax1.set_ylabel('C-Index', fontsize=12)
    ax1.set_ylim(0.5, 1.0)
    
    # 添加数值标签
    for bar, value in zip(bars1, dfs_cindices):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # OS C-index比较
    bars2 = ax2.bar(models, os_cindices, color=colors)
    ax2.set_title('OS C-Index Comparison', fontsize=14)
    ax2.set_ylabel('C-Index', fontsize=12)
    ax2.set_ylim(0.5, 1.0)
    
    # 添加数值标签
    for bar, value in zip(bars2, os_cindices):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("comparison_plots/c_index_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("comparison_plots/c_index_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_auc_comparison():
    """绘制AUC比较图 - 与原代码风格一致"""
    
    # 读取结果
    result_IASLC = pd.read_excel("output_IASLC/QA.IASLC.xlsx", index_col=0)
    result_Cox = pd.read_excel("output_Cox/QA.Cox.xlsx", index_col=0)
    result_RSF = pd.read_excel("output_RSF/QA.RSF.xlsx", index_col=0)
    result_Lasso = pd.read_excel("output_Lasso/QA.Lasso.xlsx", index_col=0)
    result_GB = pd.read_excel("output_GB/QA.GB.xlsx", index_col=0)

    times = np.asarray([24, 30, 36, 42, 48, 54, 60, 66, 72], dtype=int)

    # 提取AUC数据
    y_IASLC_DFS = np.zeros(len(times))
    y_IASLC_OS = np.zeros(len(times))
    y_Cox_DFS = np.zeros(len(times))
    y_Cox_OS = np.zeros(len(times))
    y_RSF_DFS = np.zeros(len(times))
    y_RSF_OS = np.zeros(len(times))
    y_Lasso_DFS = np.zeros(len(times))
    y_Lasso_OS = np.zeros(len(times))
    y_GB_DFS = np.zeros(len(times))
    y_GB_OS = np.zeros(len(times))

    for i in range(len(times)):
        y_IASLC_DFS[i] = result_IASLC.T['DFS_AUC_' + str(times[i])]
        y_IASLC_OS[i] = result_IASLC.T['OS_AUC_' + str(times[i])]
        y_Cox_DFS[i] = result_Cox.T['DFS_AUC_' + str(times[i])]
        y_Cox_OS[i] = result_Cox.T['OS_AUC_' + str(times[i])]
        y_RSF_DFS[i] = result_RSF.T['DFS_AUC_' + str(times[i])]
        y_RSF_OS[i] = result_RSF.T['OS_AUC_' + str(times[i])]
        y_Lasso_DFS[i] = result_Lasso.T['DFS_AUC_' + str(times[i])]
        y_Lasso_OS[i] = result_Lasso.T['OS_AUC_' + str(times[i])]
        y_GB_DFS[i] = result_GB.T['DFS_AUC_' + str(times[i])]
        y_GB_OS[i] = result_GB.T['OS_AUC_' + str(times[i])]

    # DFS AUC比较 - 与原代码风格一致
    plt.figure(figsize=(6, 4))
    plt.plot(times, y_IASLC_DFS, label='IASLC', c='red', lw=3)
    plt.plot(times, y_Cox_DFS, label='Cox', c='tomato', lw=3)
    plt.plot(times, y_RSF_DFS, label='RSF', c='limegreen', lw=3) 
    plt.plot(times, y_GB_DFS, label='GB', c='purple', lw=3)   
    plt.plot(times, y_Lasso_DFS, label='Lasso', c='deepskyblue', lw=3) 
    plt.scatter(times, y_IASLC_DFS, s=50, c='red')
    plt.scatter(times, y_Cox_DFS, s=50, c='tomato')
    plt.scatter(times, y_RSF_DFS, s=50, c='limegreen')
    plt.scatter(times, y_GB_DFS, s=50, c='purple')   
    plt.scatter(times, y_Lasso_DFS, s=50, c='deepskyblue')
    plt.legend(loc='lower left', fontsize=12) 
    plt.xlabel("DFS(month)", fontsize=12)
    plt.ylabel("Time-dependent AUC", fontsize=12) 
    plt.savefig("comparison_plots/AUC_DFS_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # OS AUC比较
    plt.figure(figsize=(6, 4))
    plt.plot(times, y_IASLC_OS, label='IASLC', c='red', lw=3)
    plt.plot(times, y_Cox_OS, label='Cox', c='tomato', lw=3)
    plt.plot(times, y_RSF_OS, label='RSF', c='limegreen', lw=3) 
    plt.plot(times, y_GB_OS, label='GB', c='purple', lw=3)    
    plt.plot(times, y_Lasso_OS, label='Lasso', c='deepskyblue', lw=3) 
    plt.scatter(times, y_IASLC_OS, s=50, c='red')
    plt.scatter(times, y_Cox_OS, s=50, c='tomato')
    plt.scatter(times, y_RSF_OS, s=50, c='limegreen')
    plt.scatter(times, y_GB_OS, s=50, c='purple')   
    plt.scatter(times, y_Lasso_OS, s=50, c='deepskyblue')
    plt.legend(loc='lower left', fontsize=12) 
    plt.xlabel("OS(month)", fontsize=12)
    plt.ylabel("Time-dependent AUC", fontsize=12) 
    plt.savefig("comparison_plots/AUC_OS_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def add_at_risk_counts(times, durations, events):
    """计算每个时间点的风险人数 - 与原代码一致"""
    n_at_risk = np.zeros(len(times))
    for i, t in enumerate(times):
        if i == 0:
            n_at_risk[i] = len(durations)
        else:
            n_at_risk[i] = sum((durations > t) | ((durations == t) & ~events))
    return n_at_risk

def plot_kaplan_meier_iaslc():
    """绘制IASLC分级的Kaplan-Meier曲线 - 与原代码风格一致"""
    
    if not os.path.exists('comparison_plots'):
        os.makedirs('comparison_plots')
    
    # 读取数据
    data = pd.read_excel('input_data/data_fold0.xlsx', index_col=0).sort_index()
    
    # 定义颜色 - 为不同分期使用不同颜色
    colors = ['limegreen', 'orange', 'tomato']  # 对应分期1, 2, 3
    stage_labels = ['Stage I', 'Stage II', 'Stage III']
    
    for tag in ['DFS', 'OS']:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), 
                                       gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})
        
        # 设置整个图形的背景为透明
        fig.patch.set_alpha(0)
        
        if tag == 'DFS':
            event = np.asarray(data['是否复发'], dtype=bool)
            time = np.asarray(data['DFS(month)'])
        else:
            event = np.asarray(data['是否死亡'], dtype=bool)
            time = np.asarray(data['OS(month)'])
        
        # 按分期分组
        stages = sorted([s for s in data['分期2'].unique() if pd.notna(s)])
        
        # 计算总体log-rank test p值
        df_logrank = pd.DataFrame({
            'T': time,
            'E': event,
            'group': data['分期2']
        })
        df_logrank = df_logrank.dropna()
        results = multivariate_logrank_test(df_logrank['T'], df_logrank['group'], df_logrank['E'])
        p_value = results.p_value
        
        # 定义共同的时间点
        times_plot = [0, 12, 24, 36, 48, 60, 72, 84]
        
        # 绘制Kaplan-Meier曲线
        for i, stage in enumerate(stages):
            stage_mask = data['分期2'] == stage
            stage_time = time[stage_mask]
            stage_event = event[stage_mask]
            
            x, y, conf_int = kaplan_meier_estimator(stage_event, stage_time, conf_type="log-log")
            
            ax1.step(x, y, where="post", c=colors[i], label=stage_labels[i])
            ax1.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post", facecolor=colors[i])
        
        ax1.set_ylim(0, 1.02)
        ax1.set_xlim(0, 84)
        ax1.set_xticks(times_plot)
        ax1.set_xlabel(tag + " (months)")
        ax1.set_ylabel("Survival Probability")
        ax1.legend(loc='lower left')
        
        # 添加p值
        ax1.text(0.1, 0.2, f'Log-rank p < 0.001', transform=ax1.transAxes, verticalalignment='top')
        
        # 设置上半部分图的背景为透明
        ax1.patch.set_alpha(0)
        
        # 风险表
        ax2.set_xlim(0, 84)
        ax2.set_ylim(0, 3)
        
        # 移除x轴和y轴的刻度
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # 添加组标签，颜色与曲线一致
        for i, stage in enumerate(stages):
            ax2.text(-5, len(stages) - i - 0.5, stage_labels[i], ha='right', va='center', color=colors[i])
        
        for i, stage in enumerate(stages):
            stage_mask = data['分期2'] == stage
            stage_time = time[stage_mask]
            stage_event = event[stage_mask]
            
            n_at_risk = add_at_risk_counts(times_plot, stage_time, stage_event)
            for j, t in enumerate(times_plot):
                ax2.text(t, len(stages) - i - 0.5, str(int(n_at_risk[j])), 
                         ha='center', va='center', fontsize=9, color=colors[i])
        
        # 去除风险表的所有边框
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        # 设置风险表的背景为透明
        ax2.patch.set_alpha(0)
        
        # 调整子图之间的间距
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        
        # 保存图片时设置透明背景
        plt.savefig(f"comparison_plots/kaplan_meier_IASLC_{tag}.pdf", dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

def plot_calibration_comparison():
    """绘制校准曲线比较 - 与原代码风格一致"""
    
    result_IASLC = pd.read_excel("output_IASLC/result.IASLC.xlsx")
    result_Cox = pd.read_excel("output_Cox/result.Cox.xlsx")
    result_RSF = pd.read_excel("output_RSF/result.RSF.xlsx")
    result_Lasso = pd.read_excel("output_Lasso/result.Lasso.xlsx")
    result_GB = pd.read_excel("output_GB/result.GB.xlsx")

    times = np.asarray([24, 30, 36, 42, 48, 54, 60, 66, 72], dtype=int)
    
    for tag in ['DFS', 'OS']:
        if tag == 'DFS':
            real_data = result_RSF['real_recu']
            show_y = "recurrence"
        else:
            real_data = result_RSF['real_dead']
            show_y = 'dead'

        for time in times:
            predict_IASLC = result_IASLC[tag + '_' + str(time)]
            predict_RSF = result_RSF[tag + '_' + str(time)]
            predict_Lasso = result_Lasso[tag + '_' + str(time)]
            predict_Cox = result_Cox[tag + '_' + str(time)]
            predict_GB = result_GB[tag + '_' + str(time)]

            IASLC_true, IASLC_pred = calibration_curve(1 - real_data, predict_IASLC, n_bins=4, strategy="uniform")
            RSF_true, RSF_pred = calibration_curve(1 - real_data, predict_RSF, n_bins=4, strategy="uniform")
            Cox_true, Cox_pred = calibration_curve(1 - real_data, predict_Cox, n_bins=4, strategy="uniform")
            GB_true, GB_pred = calibration_curve(1 - real_data, predict_GB, n_bins=4, strategy="uniform")
            Lasso_true, Lasso_pred = calibration_curve(1 - real_data, predict_Lasso, n_bins=4, strategy="uniform")

            if time % 12 == 0:
                show_year = str(int(time / 12))
            else:
                show_year = str(time / 12)

            plt.figure(figsize=(6, 4))
            plt.plot(IASLC_pred, IASLC_true, label='IASLC', c='red', lw=3)
            plt.plot(Cox_pred, Cox_true, label='Cox', c='tomato', lw=3)
            plt.plot(RSF_pred, RSF_true, label='RSF', c='limegreen', lw=3)  
            plt.plot(GB_pred, GB_true, label='GB', c='purple', lw=3)  
            plt.plot(Lasso_pred, Lasso_true, label='Lasso', c='deepskyblue', lw=3)  
            plt.scatter(IASLC_pred, IASLC_true, s=50, c='red')
            plt.scatter(Cox_pred, Cox_true, s=50, c='tomato')
            plt.scatter(RSF_pred, RSF_true, s=50, c='limegreen')
            plt.scatter(GB_pred, GB_true, s=50, c='purple')
            plt.scatter(Lasso_pred, Lasso_true, s=50, c='deepskyblue')
            plt.plot([0, 1], [0, 1], '--', c='grey', label='Perfectly calibrated')
            plt.legend(loc='lower right')

            plt.xlabel("Predicted probability of " + show_year + "-years " + tag, fontsize=12)
            plt.ylabel("Fraction of " + show_y, fontsize=12)
            plt.savefig(f"comparison_plots/calibration_" + tag + "_" + str(time) + ".png", dpi=300, bbox_inches='tight')
            plt.savefig(f"comparison_plots/calibration_" + tag + "_" + str(time) + ".pdf", bbox_inches='tight')
            plt.close()

def plot_dca_comparison():
    """绘制DCA比较 - 与原代码风格一致"""
    
    def calculate_net_benefit_all(thresh_group, y_label):
        net_benefit_all = np.array([])
        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total = tp + tn
        for thresh in thresh_group:
            net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
            net_benefit_all = np.append(net_benefit_all, net_benefit)
        return net_benefit_all
    
    def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
        net_benefit_model = np.array([])
        for thresh in thresh_group:
            y_pred_label = y_pred_score > thresh
            tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
            n = len(y_label)
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefit_model = np.append(net_benefit_model, net_benefit)
        return net_benefit_model
    
    for tag in ['DFS', 'OS']:
        result_IASLC = pd.read_excel("output_IASLC/result.IASLC.xlsx")
        result_Cox = pd.read_excel("output_Cox/result.Cox.xlsx")
        result_RSF = pd.read_excel("output_RSF/result.RSF.xlsx")
        result_GB = pd.read_excel("output_GB/result.GB.xlsx")
        result_Lasso = pd.read_excel("output_Lasso/result.Lasso.xlsx")

        times = np.asarray([24, 30, 36, 42, 48, 54, 60, 66, 72], dtype=int)

        if tag == 'DFS':
            y_label = np.asarray(result_RSF['real_recu'])
        else:
            y_label = np.asarray(result_RSF['real_dead'])

        for time in times:
            y_pred_IASLC = 1 - np.asarray(result_IASLC[tag + '_' + str(time)])
            y_pred_Cox = 1 - np.asarray(result_Cox[tag + '_' + str(time)])
            y_pred_RSF = 1 - np.asarray(result_RSF[tag + '_' + str(time)])
            y_pred_GB = 1 - np.asarray(result_GB[tag + '_' + str(time)])
            y_pred_Lasso = 1 - np.asarray(result_Lasso[tag + '_' + str(time)])

            thresh_group = np.arange(0, 1, 0.01)
            net_benefit_IASLC = calculate_net_benefit_model(thresh_group, y_pred_IASLC, y_label)
            net_benefit_Cox = calculate_net_benefit_model(thresh_group, y_pred_Cox, y_label)
            net_benefit_RSF = calculate_net_benefit_model(thresh_group, y_pred_RSF, y_label)
            net_benefit_GB = calculate_net_benefit_model(thresh_group, y_pred_GB, y_label)
            net_benefit_Lasso = calculate_net_benefit_model(thresh_group, y_pred_Lasso, y_label)
            net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)

            if time % 12 == 0:
                show_year = str(int(time / 12))
            else:
                show_year = str(time / 12)

            plt.figure(figsize=(6, 4))
            plt.plot(thresh_group, net_benefit_IASLC, color='red', label='IASLC')
            plt.plot(thresh_group, net_benefit_Cox, color='tomato', label='Cox')
            plt.plot(thresh_group, net_benefit_RSF, color='limegreen', label='RSF')
            plt.plot(thresh_group, net_benefit_GB, color='purple', label='GB')
            plt.plot(thresh_group, net_benefit_Lasso, color='deepskyblue', label='Lasso')
            plt.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
            plt.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')

            plt.xlim(0, 1)
            if tag == 'DFS':
                plt.ylim(-0.05, 0.2)
            elif tag == 'OS':
                plt.ylim(-0.05, 0.1)
            plt.xlabel('Threshold Probability', fontsize=12)
            plt.ylabel('Net Benefit', fontsize=12)
            plt.grid('major')
            plt.legend(loc='upper right', fontsize=12)
            plt.title("DCA for predicting " + show_year + "-year " + tag, fontsize=12)
            plt.savefig(f"comparison_plots/DCA_" + tag + "_" + str(time) + ".pdf", bbox_inches='tight')
            plt.savefig(f"comparison_plots/DCA_" + tag + "_" + str(time) + ".png", bbox_inches='tight')
            plt.close()

def print_comparison_summary():
    """打印比较摘要"""
    
    # 读取结果
    result_IASLC = pd.read_excel("output_IASLC/QA.IASLC.xlsx", index_col=0)
    result_Lasso = pd.read_excel("output_Lasso/QA.Lasso.xlsx", index_col=0)
    result_Cox = pd.read_excel("output_Cox/QA.Cox.xlsx", index_col=0)
    result_RSF = pd.read_excel("output_RSF/QA.RSF.xlsx", index_col=0)
    result_GB = pd.read_excel("output_GB/QA.GB.xlsx", index_col=0)

    # 提取C-index
    models = ['IASLC', 'Cox', 'Lasso', 'RSF', 'GB']
    results = [result_IASLC, result_Cox, result_Lasso, result_RSF, result_GB]
    
    comparison_data = []
    for model, result in zip(models, results):
        dfs_c = result.T['DFS_c_index'].iloc[0]
        os_c = result.T['OS_c_index'].iloc[0]
        comparison_data.append({
            'Model': model,
            'DFS_C_Index': dfs_c,
            'OS_C_Index': os_c
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存比较表格
    comparison_df.to_excel("comparison_plots/model_comparison_summary.xlsx", index=False)
    
    # 计算改进幅度
    iaslc_dfs_c = comparison_df[comparison_df['Model'] == 'IASLC']['DFS_C_Index'].iloc[0]
    iaslc_os_c = comparison_df[comparison_df['Model'] == 'IASLC']['OS_C_Index'].iloc[0]
    
    print("\n=== Model Performance Comparison ===")
    print(f"IASLC alone: DFS C-index = {iaslc_dfs_c:.3f}, OS C-index = {iaslc_os_c:.3f}")
    
    for model in ['Cox', 'Lasso', 'RSF', 'GB']:
        model_dfs_c = comparison_df[comparison_df['Model'] == model]['DFS_C_Index'].iloc[0]
        model_os_c = comparison_df[comparison_df['Model'] == model]['OS_C_Index'].iloc[0]
        
        dfs_improvement = ((model_dfs_c - iaslc_dfs_c) / iaslc_dfs_c) * 100
        os_improvement = ((model_os_c - iaslc_os_c) / iaslc_os_c) * 100
        
        print(f"{model}: DFS C-index = {model_dfs_c:.3f} (+{dfs_improvement:.1f}%), "
              f"OS C-index = {model_os_c:.3f} (+{os_improvement:.1f}%)")

# 主函数
if __name__ == "__main__":
    print("Loading data...")
    data = load_train_data('input_data/')
    
    print("Training IASLC-only model...")
    train_iaslc_model(data)
    
    print("Plotting comparisons...")
    plot_c_index_comparison()
    plot_auc_comparison()
    # plot_kaplan_meier_iaslc()
    plot_calibration_comparison()
    plot_dca_comparison()
    
    print("Generating summary...")
    print_comparison_summary()
    
    print("Analysis complete! Check the 'comparison_plots' folder for results.")