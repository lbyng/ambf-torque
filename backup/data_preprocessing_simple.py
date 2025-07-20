import os
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import argparse

# ========== 数据平滑处理类 ==========
class DataSmoother:
    """数据平滑处理工具类"""
    
    def __init__(self, method='savgol', **kwargs):
        """
        初始化平滑器
        
        Parameters:
        -----------
        method : str
            平滑方法：'savgol', 'gaussian', 'butterworth', 'moving_average'
        **kwargs : dict
            各方法的参数
        """
        self.method = method
        self.params = kwargs
        
    def smooth(self, data, axis=0):
        """
        对数据进行平滑处理
        
        Parameters:
        -----------
        data : np.ndarray
            输入数据
        axis : int
            沿哪个轴进行平滑（默认为0，即时间轴）
            
        Returns:
        --------
        smoothed_data : np.ndarray
            平滑后的数据
        """
        if self.method == 'savgol':
            return self._savgol_smooth(data, axis)
        elif self.method == 'gaussian':
            return self._gaussian_smooth(data, axis)
        elif self.method == 'butterworth':
            return self._butterworth_smooth(data, axis)
        elif self.method == 'moving_average':
            return self._moving_average_smooth(data, axis)
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")
    
    def _savgol_smooth(self, data, axis=0):
        """Savitzky-Golay滤波器"""
        window_length = self.params.get('window_length', 11)
        polyorder = self.params.get('polyorder', 3)
        
        # 确保窗口长度是奇数
        if window_length % 2 == 0:
            window_length += 1
            
        # 确保窗口长度不超过数据长度
        data_length = data.shape[axis]
        if window_length > data_length:
            window_length = data_length if data_length % 2 == 1 else data_length - 1
            
        # 确保多项式阶数小于窗口长度
        if polyorder >= window_length:
            polyorder = window_length - 1
            
        return savgol_filter(data, window_length, polyorder, axis=axis)
    
    def _gaussian_smooth(self, data, axis=0):
        """高斯滤波器"""
        sigma = self.params.get('sigma', 1.0)
        return gaussian_filter1d(data, sigma=sigma, axis=axis)
    
    def _butterworth_smooth(self, data, axis=0):
        """Butterworth低通滤波器"""
        order = self.params.get('order', 4)
        cutoff_freq = self.params.get('cutoff_freq', 0.1)  # 归一化频率 (0-1)
        
        # 设计滤波器
        b, a = butter(order, cutoff_freq, btype='low')
        
        # 应用滤波器
        if axis == 0:
            if data.ndim == 1:
                return filtfilt(b, a, data)
            else:
                smoothed = np.zeros_like(data)
                for i in range(data.shape[1]):
                    smoothed[:, i] = filtfilt(b, a, data[:, i])
                return smoothed
        else:
            raise NotImplementedError("Butterworth filter only supports axis=0")
    
    def _moving_average_smooth(self, data, axis=0):
        """移动平均平滑"""
        window_size = self.params.get('window_size', 5)
        
        if axis == 0:
            if data.ndim == 1:
                return np.convolve(data, np.ones(window_size)/window_size, mode='same')
            else:
                smoothed = np.zeros_like(data)
                for i in range(data.shape[1]):
                    smoothed[:, i] = np.convolve(data[:, i], 
                                                  np.ones(window_size)/window_size, 
                                                  mode='same')
                return smoothed
        else:
            raise NotImplementedError("Moving average only supports axis=0")

# ========== 特征工程函数 ==========
def create_minimal_features(data):
    """
    创建最小特征集：只包含 jpos, jvel, control_command
    
    Parameters:
    -----------
    data : np.ndarray
        原始数据
        
    Returns:
    --------
    features : np.ndarray
        处理后的最小特征集
    feature_names : list
        特征名称列表
    """
    # 提取基础特征
    jpos = data[:, 13:16]      # 关节位置 [0:3]
    jvel = data[:, 162:165]    # 关节速度 [0:3]
    jpos_d = data[:, 194:197]  # 关节期望位置 [0:3]
    
    # 计算控制命令
    control_command = jpos_d - jpos  # 控制命令 = 期望位置 - 实际位置
    
    # 组合特征：只包含最核心的9个特征
    features = np.hstack([
        jpos,            # 3 features - 关节实际位置
        jvel,            # 3 features - 关节速度
        control_command  # 3 features - 控制命令
    ])
    
    # 特征名称
    feature_names = [
        'jpos[0]', 'jpos[1]', 'jpos[2]',
        'jvel[0]', 'jvel[1]', 'jvel[2]',
        'ctrl_cmd[0]', 'ctrl_cmd[1]', 'ctrl_cmd[2]'
    ]
    
    return features, feature_names

def create_original_features(data):
    """
    创建原始特征（不包含控制命令）- 保留用于兼容性
    """
    feature_select = np.concatenate([
        np.arange(13, 16),   # jpos[0:3]
        np.arange(162, 165), # jvel[0:3]
        np.arange(194, 197), # jpos_d[0:3]
    ])
    
    features = data[:, feature_select]
    
    feature_names = [
        'jpos[0]', 'jpos[1]', 'jpos[2]',
        'jvel[0]', 'jvel[1]', 'jvel[2]',
        'jpos_d[0]', 'jpos_d[1]', 'jpos_d[2]'
    ]
    
    return features, feature_names

# 删除 calculate_derived_features 函数，不再需要

def create_time_sequences(features, targets, sequence_length=10, step=1):
    """
    创建时序数据序列
    
    Parameters:
    -----------
    features : np.ndarray
        特征数据 (n_samples, n_features)
    targets : np.ndarray
        目标数据 (n_samples, n_targets)
    sequence_length : int
        序列长度
    step : int
        步长（用于数据增强）
        
    Returns:
    --------
    X_sequences : np.ndarray
        时序特征 (n_sequences, sequence_length, n_features)
    y_sequences : np.ndarray
        时序目标 (n_sequences, n_targets)
    """
    n_samples = features.shape[0]
    n_features = features.shape[1]
    n_targets = targets.shape[1]
    
    # 计算序列数量
    n_sequences = (n_samples - sequence_length) // step + 1
    
    # 初始化序列数组
    X_sequences = np.zeros((n_sequences, sequence_length, n_features))
    y_sequences = np.zeros((n_sequences, n_targets))
    
    # 创建序列
    for i in range(n_sequences):
        start_idx = i * step
        end_idx = start_idx + sequence_length
        
        X_sequences[i] = features[start_idx:end_idx]
        y_sequences[i] = targets[end_idx - 1]  # 预测序列最后一个时刻的tau
    
    return X_sequences, y_sequences

class DataProcessor:
    """数据预处理主类"""
    
    def __init__(self, config):
        """
        初始化数据处理器
        
        Parameters:
        -----------
        config : dict
            处理配置
        """
        self.config = config
        self.sequence_length = config.get('sequence_length', 10)
        self.sequence_step = config.get('sequence_step', 1)
        
        # 目标索引
        self.target_indices = np.arange(114, 117)  # tau[0:3]
        
        print(f"Data processor initialized:")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Using minimal features: jpos, jvel, control_command")
        print(f"  Smoothing: enabled")
    
    def apply_smoothing(self, data):
        """应用数据平滑 - 简化版本，总是启用"""
        print("Applying data smoothing...")
        data_smoothed = data.copy()
        
        # 定义数据索引
        jvel_indices = np.arange(162, 165)  # jvel[0:3]
        mvel_indices = np.arange(146, 149)  # mvel[0:3]
        tau_indices = np.arange(114, 117)   # tau[0:3]
        
        # 平滑关节速度
        smoother_jvel = DataSmoother(method='savgol', window_length=11, polyorder=3)
        data_smoothed[:, jvel_indices] = smoother_jvel.smooth(data_smoothed[:, jvel_indices])
        print(f"  Applied smoothing to joint velocities")
        
        # 平滑电机速度
        smoother_mvel = DataSmoother(method='savgol', window_length=11, polyorder=3)
        data_smoothed[:, mvel_indices] = smoother_mvel.smooth(data_smoothed[:, mvel_indices])
        print(f"  Applied smoothing to motor velocities")
        
        # 平滑扭矩
        smoother_tau = DataSmoother(method='gaussian', sigma=1.0)
        data_smoothed[:, tau_indices] = smoother_tau.smooth(data_smoothed[:, tau_indices])
        print(f"  Applied smoothing to torques")
        
        return data_smoothed
    
    def extract_features(self, data):
        """提取最小特征集"""
        print(f"Extracting minimal features (jpos, jvel, control_command)...")
        
        # 直接使用最小特征集
        features, feature_names = create_minimal_features(data)
        
        print(f"Total features: {len(feature_names)}")
        return features, feature_names
    
    def process_single_file(self, file_path):
        """处理单个文件"""
        print(f"Processing: {os.path.basename(file_path)}")
        
        # 加载数据
        data = np.loadtxt(file_path, delimiter=',')
        print(f"  Loaded data shape: {data.shape}")
        
        # 应用平滑
        data = self.apply_smoothing(data)
        
        # 提取特征
        features, feature_names = self.extract_features(data)
        
        # 提取目标
        targets = data[:, self.target_indices]
        
        # 处理NaN值
        nan_mask = np.isnan(features).any(axis=1) | np.isnan(targets).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"  Removing {nan_mask.sum()} samples with NaN values")
            features = features[~nan_mask]
            targets = targets[~nan_mask]
        
        # 创建时序序列
        if self.sequence_length > 1:
            print(f"  Creating time sequences (length={self.sequence_length}, step={self.sequence_step})")
            X_seq, y_seq = create_time_sequences(
                features, targets, 
                self.sequence_length, 
                self.sequence_step
            )
            print(f"  Generated {X_seq.shape[0]} sequences")
        else:
            # 非时序模式
            X_seq = features[:, np.newaxis, :]  # 添加时间维度以保持一致性
            y_seq = targets
        
        return X_seq, y_seq, feature_names
    
    def process_dataset(self, data_dir, train_split=0.8, val_split=0.1):
        """
        处理整个数据集 - 按轨迹分割，避免跨文件窗口问题
        
        Parameters:
        -----------
        data_dir : str
            数据目录路径
        train_split : float
            训练集比例
        val_split : float
            验证集比例
            
        Returns:
        --------
        dataset : dict
            处理后的数据集
        """
        print(f"Processing dataset from: {data_dir}")
        print("=" * 50)
        
        # 查找所有CSV文件
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        csv_files.sort()
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        print(f"Found {len(csv_files)} files:")
        for f in csv_files:
            print(f"  - {f}")
        
        # 按轨迹分割文件 - 前几个文件作为测试集，后面作为训练集
        n_files = len(csv_files)
        n_test_files = max(1, int(n_files * (1 - train_split - val_split)))  # 至少1个测试文件
        n_val_files = max(1, int(n_files * val_split))  # 至少1个验证文件
        n_train_files = n_files - n_test_files - n_val_files
        
        # 重新分配：前面的文件作为测试集
        test_files = csv_files[:n_test_files]
        val_files = csv_files[n_test_files:n_test_files + n_val_files]
        train_files = csv_files[n_test_files + n_val_files:]
        
        print(f"\nFile split (test files first):")
        print(f"  Test files: {len(test_files)} ({len(test_files)/n_files*100:.1f}%) - {test_files}")
        print(f"  Val files: {len(val_files)} ({len(val_files)/n_files*100:.1f}%) - {val_files}")
        print(f"  Train files: {len(train_files)} ({len(train_files)/n_files*100:.1f}%) - {train_files}")
        
        # 分别处理训练、验证、测试文件 - 每个文件独立创建时序窗口
        def process_file_group_safe(file_list, group_name):
            print(f"\nProcessing {group_name} files (safe mode - no cross-file windows)...")
            X_list = []
            y_list = []
            feature_names = None
            
            for file_name in file_list:
                file_path = os.path.join(data_dir, file_name)
                print(f"  Processing: {os.path.basename(file_path)}")
                
                # 加载单个文件数据
                data = np.loadtxt(file_path, delimiter=',')
                print(f"    Loaded data shape: {data.shape}")
                
                # 应用平滑
                data = self.apply_smoothing(data)
                
                # 提取特征（每个文件独立处理）
                features, names = self.extract_features(data)
                targets = data[:, self.target_indices]
                
                # 处理NaN值
                nan_mask = np.isnan(features).any(axis=1) | np.isnan(targets).any(axis=1)
                if nan_mask.sum() > 0:
                    print(f"    Removing {nan_mask.sum()} samples with NaN values")
                    features = features[~nan_mask]
                    targets = targets[~nan_mask]
                
                # 在单个文件内创建时序序列（避免跨文件问题）
                if self.sequence_length > 1:
                    print(f"    Creating time sequences (length={self.sequence_length}, step={self.sequence_step})")
                    X_seq, y_seq = create_time_sequences(
                        features, targets, 
                        self.sequence_length, 
                        self.sequence_step
                    )
                    print(f"    Generated {X_seq.shape[0]} sequences from this file")
                else:
                    X_seq = features[:, np.newaxis, :]
                    y_seq = targets
                
                # 添加到列表
                X_list.append(X_seq)
                y_list.append(y_seq)
                
                if feature_names is None:
                    feature_names = names
            
            if X_list:
                X_combined = np.vstack(X_list)
                y_combined = np.vstack(y_list)
                print(f"  {group_name} total shape: X={X_combined.shape}, y={y_combined.shape}")
                return X_combined, y_combined, feature_names
            else:
                return None, None, feature_names
        
        # 处理各组文件（每个文件独立创建窗口）
        X_train, y_train, feature_names = process_file_group_safe(train_files, "Training")
        X_val, y_val, _ = process_file_group_safe(val_files, "Validation")
        X_test, y_test, _ = process_file_group_safe(test_files, "Test")
        
        # 处理可能的空集合
        if X_val is None:
            print("Warning: No validation files, using part of training data")
            n_train = X_train.shape[0]
            split_idx = int(n_train * 0.9)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        if X_test is None:
            print("Warning: No test files, using part of training data")
            n_train = X_train.shape[0]
            split_idx = int(n_train * 0.9)
            X_test = X_train[split_idx:]
            y_test = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        print(f"\nFinal dataset split:")
        print(f"  Train samples: {X_train.shape[0]} (from later files)")
        print(f"  Val samples: {X_val.shape[0]} (from middle files)")
        print(f"  Test samples: {X_test.shape[0]} (from early files)")
        
        # 标准化特征和目标
        print("Standardizing features and targets...")
        
        # 特征标准化 (对每个特征维度分别标准化)
        scaler_X = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        scaler_X.fit(X_train_flat)
        
        X_train_scaled = scaler_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # 目标标准化
        scaler_y = StandardScaler()
        scaler_y.fit(y_train)
        
        y_train_scaled = scaler_y.transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)
        
        # 创建数据集字典
        dataset = {
            'X_train': X_train_scaled,
            'y_train': y_train_scaled,
            'X_val': X_val_scaled,
            'y_val': y_val_scaled,
            'X_test': X_test_scaled,
            'y_test': y_test_scaled,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_names': feature_names,
            'config': self.config,
            'metadata': {
                'n_samples': X_train.shape[0] + X_val.shape[0] + X_test.shape[0],
                'n_features': len(feature_names),
                'n_targets': 3,
                'sequence_length': self.sequence_length,
                'train_samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'test_samples': X_test.shape[0],
                'train_files': train_files,
                'val_files': val_files,
                'test_files': test_files,
                'files_processed': csv_files,
                'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'split_method': 'by_trajectory_safe_windows_test_first'
            }
        }
        
        return dataset
    
    def save_dataset(self, dataset, output_path):
        """保存处理后的数据集"""
        print(f"Saving processed dataset to: {output_path}")
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 保存配置信息
        config_path = output_path.replace('.pkl', '_info.txt')
        with open(config_path, 'w') as f:
            f.write("PROCESSED DATASET INFORMATION\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Processing time: {dataset['metadata']['processing_time']}\n")
            f.write(f"Number of samples: {dataset['metadata']['n_samples']}\n")
            f.write(f"Number of features: {dataset['metadata']['n_features']}\n")
            f.write(f"Sequence length: {dataset['metadata']['sequence_length']}\n")
            f.write(f"Feature type: minimal (jpos, jvel, control_command)\n")
            f.write(f"Smoothing: enabled\n")
            f.write(f"Data split: train={dataset['metadata']['train_samples']}, ")
            f.write(f"val={dataset['metadata']['val_samples']}, ")
            f.write(f"test={dataset['metadata']['test_samples']}\n\n")
            
            f.write("Configuration:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nFiles processed:\n")
            f.write("-" * 20 + "\n")
            for file in dataset['metadata']['files_processed']:
                f.write(f"  {file}\n")
            
            f.write("\nFeature names:\n")
            f.write("-" * 20 + "\n")
            for i, name in enumerate(dataset['feature_names']):
                f.write(f"  {i:2d}: {name}\n")
        
        print(f"Dataset info saved to: {config_path}")
        
        # 计算文件大小
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"Dataset file size: {file_size:.2f} MB")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='机器人扭矩数据预处理 - 简化版')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='原始数据目录路径')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出文件路径 (.pkl)')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='时序序列长度')
    parser.add_argument('--sequence_step', type=int, default=1,
                        help='时序序列步长')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    
    args = parser.parse_args()
    
    # 创建处理配置 - 简化版
    config = {
        'sequence_length': args.sequence_length,
        'sequence_step': args.sequence_step,
        'train_split': args.train_split,
        'val_split': args.val_split
    }
    
    print("Robot Torque Data Preprocessing - Simplified")
    print("=" * 50)
    print(f"Input directory: {args.data_dir}")
    print(f"Output file: {args.output_path}")
    print(f"Configuration: {config}")
    print("Features: jpos, jvel, control_command (9 features total)")
    print("Smoothing: enabled")
    print()
    
    # 创建数据处理器
    processor = DataProcessor(config)
    
    # 处理数据集
    dataset = processor.process_dataset(
        args.data_dir, 
        args.train_split, 
        args.val_split
    )
    
    # 保存数据集
    processor.save_dataset(dataset, args.output_path)
    
    print("\n" + "=" * 40)
    print("Data preprocessing completed successfully!")
    print(f"Processed dataset saved to: {args.output_path}")
    print(f"Dataset shape: X={dataset['X_train'].shape}, y={dataset['y_train'].shape}")
    print("=" * 40)

if __name__ == "__main__":
    main()