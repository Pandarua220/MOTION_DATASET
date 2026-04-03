import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import shap
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             cohen_kappa_score, recall_score, precision_score, 
                             matthews_corrcoef)
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
# 引入 utils (确保 utils.py 在同一目录下)
from utils import load_concatenated_features_combined

"""
ML classifier and SHAP analysis for brf
"""

# ===================== 配置区域 =====================

CV_N_SPLITS = 9 # cross-validation folds
shap_to_analysis = 2 # win_r for shap analysis


# model list to be run

# win_r_set = [0, 1, 2, 3, 4, 5]
# shap_analysis = True
# models_to_run = ["brf"]

win_r_set = [2]
shap_analysis = False
models_to_run = ["svm", "lda", "brf"]

# main output directory
base_out_dir = f"../result/classify_multi_models_{CV_N_SPLITS}fold"
os.makedirs(base_out_dir, exist_ok=True)

# feature channel group configuration 
channel_groups = {
    "All": range(22),
}

# label mapping
label_binary = {0: "Sleep", 1: "Wake"}
label_three_class = {0: "NREM", 1: "REM", 2: "Wake"}

# body part groups (for SHAP aggregation)
body_part_groups = {
    'head': [0, 9, 18],
    'left_arm': [1, 2, 3, 4],
    'left_leg': [5, 6, 7, 8],
    'right_arm': [10, 11, 12, 13],
    'right_leg': [14, 15, 16, 17],
    'torso': [19, 20],
    'whole': [21]
}

# model configuration
MODEL_CONFIG = {
    "svm": {
        "desc": "Support Vector Machine",
        "build_pipeline": lambda: make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            StandardScaler(),
            # use RBF kernel and balanced class weights
            SVC(kernel='rbf', class_weight='balanced', random_state=42)
        )
    },
    
    "lda": {
        "desc": "Linear Discriminant Analysis",
        "build_pipeline": lambda: make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            StandardScaler(),
            LinearDiscriminantAnalysis()
        )
    },
    
    "brf": {
        "desc": "Balanced Random Forest",
        "build_pipeline": lambda: make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            StandardScaler(),
            BalancedRandomForestClassifier(
                n_estimators=500,
                sampling_strategy='auto',
                replacement=True,
                n_jobs=-1,
                random_state=42
            )        
        )
    },
}


# ===================== Plotting and Export =====================

def plot_cm(cm, labels, path, title_suffix=""):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True', xlabel='Predicted',
           title=f'Confusion Matrix {title_suffix}')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



# ===================== 特征索引映射 (308特征) =====================

def get_feature_mapping(total_features=308):
    """
    映射逻辑：
    - 前 154 个特征：22个通道的 Optical Flow (每个通道7个特征)
    - 后 154 个特征：22个通道的 Displacement (每个通道7个特征)
    """
    num_channels = 22
    feats_per_ch = 7  # 308 / 2 / 22
    op_offset = 0
    kp_offset = 154
    
    mapping = {}
    # BODY_PART_GROUPS 定义见 utils 或配置
    for part_name, channels in body_part_groups.items():
        op_indices, kp_indices = [], []
        for ch in channels:
            # OP 索引范围
            s_op = op_offset + (ch * feats_per_ch)
            op_indices.extend(range(s_op, s_op + feats_per_ch))
            # KP 索引范围
            s_kp = kp_offset + (ch * feats_per_ch)
            kp_indices.extend(range(s_kp, s_kp + feats_per_ch))
        
        mapping[f"{part_name}_OP"] = op_indices
        mapping[f"{part_name}_KP"] = kp_indices
    return mapping

# ===================== 交叉验证 + SHAP 核心函数 =====================
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ===================== 1. 绘图辅助函数 =====================

def plot_grouped_beeswarm(all_shap_values_class, X_processed, feature_mapping, class_name, task_name, out_dir):
    """
    绘制按身体部位分组的 SHAP 蜂群图 (13组)
    """
    # 定义 13 组映射关系
    grouped_indices = {}
    parts = ['head', 'left_arm', 'left_leg', 'right_arm', 'right_leg', 'torso']
    for part in parts:
        grouped_indices[f"{part} (OP)"] = feature_mapping[f"{part}_OP"]
        grouped_indices[f"{part} (KP)"] = feature_mapping[f"{part}_KP"]
    
    # 合并 Whole 的 OP 和 KP 作为第 13 组
    grouped_indices["whole (Merged)"] = feature_mapping["whole_OP"] + feature_mapping["whole_KP"]
    
    group_names = list(grouped_indices.keys())
    num_samples = all_shap_values_class.shape[0]
    num_groups = len(group_names)
    
    # 构造聚合矩阵
    aggregated_shap = np.zeros((num_samples, num_groups))
    aggregated_X = np.zeros((num_samples, num_groups))
    
    for i, (name, indices) in enumerate(grouped_indices.items()):
        # SHAP 值求和反映总影响力，特征值取平均反映该部位运动强度
        aggregated_shap[:, i] = np.sum(all_shap_values_class[:, indices], axis=1)
        aggregated_X[:, i] = np.mean(X_processed[:, indices], axis=1)
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        aggregated_shap, 
        aggregated_X, 
        feature_names=group_names, 
        plot_type="dot", 
        show=False,
        max_display=13
    )
    plt.title(f"Grouped SHAP Beeswarm - {task_name} ({class_name})", fontsize=16)
    plt.xlabel("SHAP Value (Sum of Group Impact)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"RF_beeswarm_grouped_{task_name}_{class_name}.png"), dpi=300)
    plt.close()

# ===================== 2. 核心评估函数 =====================

def evaluate_model_with_cv_shap(x, y, groups, label_map, build_pipe, n_splits, out_dir, task_name):
    """
    执行交叉验证并在每一折计算 SHAP 值，最后生成柱状图和聚合蜂群图
    """
    gkf = GroupKFold(n_splits=n_splits)
    num_classes = len(label_map)
    
    # 获取特征索引映射 (308特征)
    feat_mapping = get_feature_mapping(x.shape[1])
    
    # 初始化存储所有样本 SHAP 值的空间
    all_shap_values = [np.zeros_like(x) for _ in range(num_classes)]
    true_list, pred_list = [], []

    print(f"  [{task_name}] 开始 {n_splits} 折交叉验证并计算 SHAP...")
    
    for fold, (tr, te) in enumerate(gkf.split(x, y, groups)):
        pipe = build_pipe()
        pipe.fit(x[tr], y[tr])
        
        # 记录预测结果
        pred_list.extend(pipe.predict(x[te]))
        true_list.extend(y[te])
        
        # 提取随机森林模型和预处理步骤
        clf = pipe.named_steps['balancedrandomforestclassifier']
        imputer = pipe.named_steps['simpleimputer']
        scaler = pipe.named_steps['standardscaler']
        
        # 对测试折进行相同的预处理
        x_te_proc = scaler.transform(imputer.transform(x[te]))
        
        # 计算该折的 SHAP 值
        explainer = shap.TreeExplainer(clf)
        fold_shap = explainer.shap_values(x_te_proc)
        
        # 根据 SHAP 返回格式填充全局数组
        for cls_idx in range(num_classes):
            all_shap_values[cls_idx][te] = fold_shap[:, :, cls_idx]

        print(f"    - 第 {fold+1} 折 SHAP 计算完成")

    # --- 后处理：计算全量数据的预处理特征用于蜂群图颜色显示 ---
    
    # 1. 先进行缺失值填充，得到一个副本 x_imputed
    full_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    x_imputed = full_imputer.fit_transform(x)
    
    # 2. 使用切片直接对填充后的数据进行取负处理 (处理 sleep probability)
    # 选中所有样本 (:)，从索引 6 开始，每隔 7 个取一个特征
    x_imputed[:, 6::7] = -1 * x_imputed[:, 6::7]
    
    # 3. 进行标准化
    full_scaler = StandardScaler()
    x_full_proc = full_scaler.fit_transform(x_imputed)

    # --- 聚合、保存与绘图 ---
    shap_export = {"body_part_feature_importance": {}}
    raw_influence = {}

    for cls_idx, label_name in label_map.items():
        # 1. 计算特征平均绝对影响 (用于柱状图)
        mean_abs_influence = np.mean(np.abs(all_shap_values[cls_idx]), axis=0)
        raw_influence[f"class_{cls_idx}_raw"] = np.mean(all_shap_values[cls_idx], axis=0)
        
        # 2. 柱状图聚合 (Level: Body Part)
        part_importance = {}
        for part_key, indices in feat_mapping.items():
            part_importance[part_key] = float(np.mean(mean_abs_influence[indices]))
        shap_export["body_part_feature_importance"][f"class_{cls_idx}"] = part_importance
        
        # 3. 绘制柱状图
        # plot_shap_bars_python(part_importance, label_name, task_name, out_dir)
        
        # 4. 绘制聚合蜂群图 (13组)
        print(f"    - 正在绘制 {label_name} 的身体部位聚合蜂群图...")
        plot_grouped_beeswarm(
            all_shap_values[cls_idx], 
            x_full_proc, 
            feat_mapping, 
            label_name, 
            task_name, 
            out_dir
        )

    # 保存结果到 .mat 文件
    savemat(os.path.join(out_dir, f"{task_name}_cv_shap_results.mat"), 
            {"shap": shap_export, "raw_influence": raw_influence, "all_shap": all_shap_values, "all_X": x_full_proc})
    
    # 计算最终性能指标
    y_true = np.array(true_list)
    y_pred = np.array(pred_list)
    classes = sorted(label_map.keys())
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "conf_mat": confusion_matrix(y_true, y_pred, labels=classes),
        "recall": recall_score(y_true, y_pred, average=None, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
    }
# ===================== 主逻辑 =====================

def evaluate_model(x, y, groups, label_map, build_pipe, n_splits):
    """通用模型评估函数"""
    gkf = GroupKFold(n_splits=n_splits)
    true_list, pred_list = [], []
    
    for tr, te in tqdm(gkf.split(x, y, groups), total=n_splits, desc="Evaluating Model"):
        pipe = build_pipe()
        pipe.fit(x[tr], y[tr])
        pred = pipe.predict(x[te])
        true_list.extend(y[te])
        pred_list.extend(pred)
        
    y_true = np.array(true_list)
    y_pred = np.array(pred_list)
    classes = sorted(label_map.keys())
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "conf_mat": confusion_matrix(y_true, y_pred, labels=classes),
        "recall": recall_score(y_true, y_pred, average=None, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
        "y_true": y_true, "y_pred": y_pred
    }

def main():
    # 对每个通道组进行分析
    for grp_name, chs in channel_groups.items():
        print(f"\n{'='*30}\nProcessing Channel Group: {grp_name}\n{'='*30}")
        
        for win_r in win_r_set:
            # 1. 加载数据 (只加载一次，供所有模型使用)
            # 1.1 二分类数据
            xb, yb, gb, sb = load_concatenated_features_combined(
                win_r, list(chs), "binary"
            )
            
            # 1.2 三分类数据
            xt, yt, gt, st = load_concatenated_features_combined(
                win_r, list(chs), "three_class"
            )
        
            # 遍历每个模型
            for model_key in models_to_run:
                print(f"\n>>> Running Model: {MODEL_CONFIG[model_key]['desc']} ({model_key})")
                
                # 创建该模型的输出目录
                model_out_dir = os.path.join(base_out_dir, f"{model_key}_r{win_r}", grp_name)
                os.makedirs(model_out_dir, exist_ok=True)
                
                
                # --- Binary Task ---
                if sb["Valid"] > 0:
                    print(f"  [Binary Task] Evaluating...")
                    task_name = f"Binary"
                    
                    # SHAP 分析 (仅限 BRF)
                    if model_key == "brf" and win_r == shap_to_analysis and shap_analysis:
                        shap_dir = os.path.join(model_out_dir, "shap")
                        if not os.path.exists(shap_dir):
                            os.makedirs(shap_dir, exist_ok=True)
                        res_bin = evaluate_model_with_cv_shap(xb, yb, gb, label_binary, MODEL_CONFIG[model_key]["build_pipeline"], CV_N_SPLITS, shap_dir, task_name)
                    else:
                        res_bin = evaluate_model(xb, yb, gb, label_binary, MODEL_CONFIG[model_key]["build_pipeline"], CV_N_SPLITS)
                    
                    print(f"    Accuracy: {res_bin['accuracy']:.4f}, Kappa: {res_bin['kappa']:.4f}")
                    
                    # 保存结果
                    
                    savemat(os.path.join(model_out_dir, f"{task_name}_results.mat"), {"res": res_bin})
                    plot_cm(res_bin["conf_mat"], list(label_binary.values()), 
                            os.path.join(model_out_dir, f"{task_name}_cm.png"), 
                            title_suffix=f"({model_key.upper()})")
                    
                    
                # --- Three Class Task ---
                if st["Valid"] > 0:
                    print(f"  [Three Class Task] Evaluating...")
                    task_name = f"ThreeClass"
                    # SHAP 分析 (仅限 BRF)
                    if model_key == "brf" and win_r == shap_to_analysis and shap_analysis:
                        shap_dir = os.path.join(model_out_dir, "shap")
                        if not os.path.exists(shap_dir):
                            os.makedirs(shap_dir, exist_ok=True)
                        res_tri = evaluate_model_with_cv_shap(xt, yt, gt, label_three_class, MODEL_CONFIG[model_key]["build_pipeline"], CV_N_SPLITS, shap_dir, task_name)
                    else:
                        res_tri = evaluate_model(xt, yt, gt, label_three_class, MODEL_CONFIG[model_key]["build_pipeline"], CV_N_SPLITS)
                    print(f"    Accuracy: {res_tri['accuracy']:.4f}, Kappa: {res_tri['kappa']:.4f}")
                    
                    # 保存结果
                    
                    savemat(os.path.join(model_out_dir, f"{task_name}_results.mat"), {"res": res_tri})
                    plot_cm(res_tri["conf_mat"], list(label_three_class.values()), 
                            os.path.join(model_out_dir, f"{task_name}_cm.png"),
                            title_suffix=f"({model_key.upper()})")
                    
                    

if __name__ == "__main__":
    main()