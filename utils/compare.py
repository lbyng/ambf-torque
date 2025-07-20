import numpy as np
import pickle

def compare_datasets(dataset1_path, dataset2_path):
    """比较两个数据集的差异"""
    
    # 加载两个数据集
    with open(dataset1_path, 'rb') as f:
        data1 = pickle.load(f)
    
    with open(dataset2_path, 'rb') as f:
        data2 = pickle.load(f)
    
    print("数据集比较分析")
    print("=" * 50)
    
    # 1. 基本形状比较
    print("1. 数据形状比较:")
    print(f"   数据集1: X_train={data1['X_train'].shape}, y_train={data1['y_train'].shape}")
    print(f"   数据集2: X_train={data2['X_train'].shape}, y_train={data2['y_train'].shape}")
    print()
    
    # 2. 特征名称比较
    print("2. 特征名称比较:")
    print(f"   数据集1特征: {data1['feature_names']}")
    print(f"   数据集2特征: {data2['feature_names']}")
    features_same = data1['feature_names'] == data2['feature_names']
    print(f"   特征名称相同: {features_same}")
    print()
    
    # 3. 数值精确比较
    print("3. 数值精确比较:")
    
    # 训练数据比较
    X_train_equal = np.array_equal(data1['X_train'], data2['X_train'])
    y_train_equal = np.array_equal(data1['y_train'], data2['y_train'])
    
    print(f"   X_train完全相同: {X_train_equal}")
    print(f"   y_train完全相同: {y_train_equal}")
    
    if not X_train_equal:
        # 计算差异
        diff = np.abs(data1['X_train'] - data2['X_train'])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"   X_train最大差异: {max_diff:.2e}")
        print(f"   X_train平均差异: {mean_diff:.2e}")
        
        # 找出差异最大的特征
        feature_max_diffs = np.max(diff, axis=(0, 1))
        for i, (name, max_diff) in enumerate(zip(data1['feature_names'], feature_max_diffs)):
            if max_diff > 1e-10:  # 只显示有显著差异的特征
                print(f"   特征 {name}: 最大差异 {max_diff:.2e}")
    
    if not y_train_equal:
        # 目标值差异
        y_diff = np.abs(data1['y_train'] - data2['y_train'])
        y_max_diff = np.max(y_diff)
        y_mean_diff = np.mean(y_diff)
        
        print(f"   y_train最大差异: {y_max_diff:.2e}")
        print(f"   y_train平均差异: {y_mean_diff:.2e}")
    
    print()
    
    # 4. 标准化器比较
    print("4. 标准化器比较:")
    
    # X标准化器
    scaler_X_mean_equal = np.allclose(data1['scaler_X'].mean_, data2['scaler_X'].mean_, atol=1e-10)
    scaler_X_scale_equal = np.allclose(data1['scaler_X'].scale_, data2['scaler_X'].scale_, atol=1e-10)
    
    print(f"   X_scaler均值相同: {scaler_X_mean_equal}")
    print(f"   X_scaler方差相同: {scaler_X_scale_equal}")
    
    if not scaler_X_mean_equal:
        mean_diff = np.abs(data1['scaler_X'].mean_ - data2['scaler_X'].mean_)
        print(f"   X_scaler均值最大差异: {np.max(mean_diff):.2e}")
        
        # 显示每个特征的差异
        for i, (name, diff) in enumerate(zip(data1['feature_names'], mean_diff)):
            if diff > 1e-10:
                print(f"     特征 {name}: 均值差异 {diff:.2e}")
    
    if not scaler_X_scale_equal:
        scale_diff = np.abs(data1['scaler_X'].scale_ - data2['scaler_X'].scale_)
        print(f"   X_scaler缩放最大差异: {np.max(scale_diff):.2e}")
        
        # 显示每个特征的差异
        for i, (name, diff) in enumerate(zip(data1['feature_names'], scale_diff)):
            if diff > 1e-10:
                print(f"     特征 {name}: 缩放差异 {diff:.2e}")
    
    # y标准化器
    scaler_y_mean_equal = np.allclose(data1['scaler_y'].mean_, data2['scaler_y'].mean_, atol=1e-10)
    scaler_y_scale_equal = np.allclose(data1['scaler_y'].scale_, data2['scaler_y'].scale_, atol=1e-10)
    
    print(f"   y_scaler均值相同: {scaler_y_mean_equal}")
    print(f"   y_scaler方差相同: {scaler_y_scale_equal}")
    
    print()
    
    # 5. 统计信息比较
    print("5. 数据统计信息比较:")
    
    X1_stats = {
        'mean': np.mean(data1['X_train']),
        'std': np.std(data1['X_train']),
        'min': np.min(data1['X_train']),
        'max': np.max(data1['X_train'])
    }
    
    X2_stats = {
        'mean': np.mean(data2['X_train']),
        'std': np.std(data2['X_train']),
        'min': np.min(data2['X_train']),
        'max': np.max(data2['X_train'])
    }
    
    print(f"   数据集1 X_train统计: {X1_stats}")
    print(f"   数据集2 X_train统计: {X2_stats}")
    
    # 计算统计差异
    for key in X1_stats:
        diff = abs(X1_stats[key] - X2_stats[key])
        print(f"   {key}差异: {diff:.2e}")
    
    print()
    
    # 6. 详细的逐特征分析
    print("6. 逐特征详细分析:")
    
    for i, name in enumerate(data1['feature_names']):
        feature1 = data1['X_train'][:, :, i].flatten()
        feature2 = data2['X_train'][:, :, i].flatten()
        
        # 统计信息
        mean1, std1 = np.mean(feature1), np.std(feature1)
        mean2, std2 = np.mean(feature2), np.std(feature2)
        
        # 差异
        diff = np.abs(feature1 - feature2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # 只显示有显著差异的特征
        if max_diff > 1e-10 or abs(mean1 - mean2) > 1e-10:
            print(f"   特征 {name}:")
            print(f"     数据集1: 均值={mean1:.6f}, 标准差={std1:.6f}")
            print(f"     数据集2: 均值={mean2:.6f}, 标准差={std2:.6f}")
            print(f"     最大差异: {max_diff:.2e}, 平均差异: {mean_diff:.2e}")
    
    print()
    print("=" * 50)
    print("分析完成!")

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='比较两个预处理数据集的差异')
    parser.add_argument('--dataset1', type=str, required=True, help='第一个数据集路径')
    parser.add_argument('--dataset2', type=str, required=True, help='第二个数据集路径')
    
    args = parser.parse_args()
    
    compare_datasets(args.dataset1, args.dataset2)

# 运行命令示例:
# python compare.py --dataset1 ./data/features/dataset.pkl --dataset2 ./data/features/dataset-1.pkl