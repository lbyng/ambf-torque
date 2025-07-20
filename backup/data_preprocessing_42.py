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
def create_extended_features(data):
    """
    创建扩展特征（包含关节期望位置，pos_error会在derived features中添加）
    
    Parameters:
    -----------
    data : np.ndarray
        原始数据
        
    Returns:
    --------
    features : np.ndarray
        处理后的特征
    feature_names : list
        特征名称列表
    """
    # 提取需要的原始特征
    jpos = data[:, 13:16]      # 关节位置 [0:3]
    jvel = data[:, 162:165]    # 关节速度 [0:3]
    pos_d = data[:, 76:82]     # 笛卡尔期望位置
    mpos = data[:, 130:133]    # 电机位置 [0:3]
    mvel = data[:, 146:149]    # 电机速度 [0:3]
    jpos_d = data[:, 194:197]  # 关节期望位置 [0:3]
    
    # 组合特征（不包含control_command，因为会在derived features中作为pos_error添加）
    features = np.hstack([
        jpos,            # 3 features - 关节实际位置
        jvel,            # 3 features - 关节速度
        pos_d,           # 6 features - 笛卡尔期望位置
        mpos,            # 3 features - 电机位置
        mvel,            # 3 features - 电机速度
        jpos_d           # 3 features - 关节期望位置
    ])
    
    # 特征名称
    feature_names = [
        'jpos[0]', 'jpos[1]', 'jpos[2]',
        'jvel[0]', 'jvel[1]', 'jvel[2]',
        'pos_d[0]', 'pos_d[1]', 'pos_d[2]', 'pos_d[3]', 'pos_d[4]', 'pos_d[5]',
        'mpos[0]', 'mpos[1]', 'mpos[2]',
        'mvel[0]', 'mvel[1]', 'mvel[2]',
        'jpos_d[0]', 'jpos_d[1]', 'jpos_d[2]'
    ]
    
    return features, feature_names

def create_original_features(data):
    """
    创建原始特征（不包含控制命令）
    """
    feature_select = np.concatenate([
        np.arange(13, 16),   # jpos[0:3]
        np.arange(162, 165), # jvel[0:3]
        np.arange(76, 82),   # pos_d
        np.arange(130, 133), # mpos[0:3]
        np.arange(146, 149), # mvel[0:3]
        np.arange(194, 197), # jpos_d[0:3]
    ])
    
    features = data[:, feature_select]
    
    feature_names = [
        'jpos[0]', 'jpos[1]', 'jpos[2]',
        'jvel[0]', 'jvel[1]', 'jvel[2]',
        'pos_d[0]', 'pos_d[1]', 'pos_d[2]', 'pos_d[3]', 'pos_d[4]', 'pos_d[5]',
        'mpos[0]', 'mpos[1]', 'mpos[2]',
        'mvel[0]', 'mvel[1]', 'mvel[2]',
        'jpos_d[0]', 'jpos_d[1]', 'jpos_d[2]'
    ]
    
    return features, feature_names

def calculate_derived_features(data, dt=0.001):
    """
    计算衍生特征（加速度、误差积分等）
    
    Parameters:
    -----------
    data : np.ndarray
        原始数据
    dt : float
        时间步长
        
    Returns:
    --------
    derived_features : dict
        衍生特征字典
    """
    jpos = data[:, 13:16]
    jvel = data[:, 162:165]
    jpos_d = data[:, 194:197]
    
    derived = {}
    
    # 1. 关节加速度 (通过数值微分)
    jacc = np.zeros_like(jvel)
    jacc[1:] = (jvel[1:] - jvel[:-1]) / dt
    jacc[0] = jacc[1]  # 第一个点使用第二个点的值
    derived['jacc'] = jacc
    
    # 2. 位置误差
    pos_error = jpos_d - jpos
    derived['pos_error'] = pos_error
    
    # 3. 速度误差 (期望速度为0)
    vel_error = -jvel  # 假设期望速度为0
    derived['vel_error'] = vel_error
    
    # 4. 误差变化率
    error_rate = np.zeros_like(pos_error)
    error_rate[1:] = (pos_error[1:] - pos_error[:-1]) / dt
    error_rate[0] = error_rate[1]
    derived['error_rate'] = error_rate
    
    # 5. 累积误差 (积分项)
    error_integral = np.cumsum(pos_error * dt, axis=0)
    derived['error_integral'] = error_integral
    
    # 6. 运动状态特征
    motion_magnitude = np.linalg.norm(jvel, axis=1, keepdims=True)
    derived['motion_magnitude'] = np.tile(motion_magnitude, (1, 3))
    
    # 7. 加速度变化率 (jerk)
    jerk = np.zeros_like(jacc)
    jerk[1:] = (jacc[1:] - jacc[:-1]) / dt
    jerk[0] = jerk[1]
    derived['jerk'] = jerk
    
    return derived

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
        self.feature_type = config.get('feature_type', 'control_command')
        self.smoothing_config = config.get('smoothing_config', None)
        self.sequence_length = config.get('sequence_length', 10)
        self.sequence_step = config.get('sequence_step', 1)
        self.include_derived = config.get('include_derived', True)
        self.dt = config.get('dt', 0.001)  # 时间步长
        
        # 目标索引
        self.target_indices = np.arange(114, 117)  # tau[0:3]
        
        print(f"Data processor initialized:")
        print(f"  Feature type: {self.feature_type}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Include derived features: {self.include_derived}")
        print(f"  Smoothing enabled: {self.smoothing_config is not None}")
    
    def apply_smoothing(self, data):
        """应用数据平滑"""
        if self.smoothing_config is None:
            return data
        
        print("Applying data smoothing...")
        data_smoothed = data.copy()
        
        # 定义数据索引
        jvel_indices = np.arange(162, 165)  # jvel[0:3]
        mvel_indices = np.arange(146, 149)  # mvel[0:3]
        tau_indices = np.arange(114, 117)   # tau[0:3]
        
        # 平滑关节速度 (jvel)
        if 'jvel' in self.smoothing_config:
            config = self.smoothing_config['jvel'].copy()
            method = config.pop('method')
            smoother = DataSmoother(method=method, **config)
            data_smoothed[:, jvel_indices] = smoother.smooth(data_smoothed[:, jvel_indices])
            print(f"  Applied smoothing to joint velocities")
        
        # 平滑电机速度 (mvel)
        if 'mvel' in self.smoothing_config:
            config = self.smoothing_config['mvel'].copy()
            method = config.pop('method')
            smoother = DataSmoother(method=method, **config)
            data_smoothed[:, mvel_indices] = smoother.smooth(data_smoothed[:, mvel_indices])
            print(f"  Applied smoothing to motor velocities")
        
        # 平滑扭矩 (tau)
        if 'tau' in self.smoothing_config:
            config = self.smoothing_config['tau'].copy()
            method = config.pop('method')
            smoother = DataSmoother(method=method, **config)
            data_smoothed[:, tau_indices] = smoother.smooth(data_smoothed[:, tau_indices])
            print(f"  Applied smoothing to torques")
        
        return data_smoothed
    
    def extract_features(self, data):
        """提取特征"""
        print(f"Extracting {self.feature_type} features...")
        
        if self.feature_type == 'control_command':
            base_features, feature_names = create_extended_features(data)
        else:
            base_features, feature_names = create_original_features(data)
        
        # 添加衍生特征
        if self.include_derived:
            print("Computing derived features...")
            derived = calculate_derived_features(data, self.dt)
            
            # 组合所有特征
            all_features = [base_features]
            derived_names = []
            
            for name, feature in derived.items():
                all_features.append(feature)
                if feature.shape[1] == 3:  # 关节特征
                    derived_names.extend([f'{name}[{i}]' for i in range(3)])
                else:  # 其他特征
                    derived_names.extend([f'{name}[{i}]' for i in range(feature.shape[1])])
            
            features = np.hstack(all_features)
            feature_names = feature_names + derived_names
            
            print(f"  Added {len(derived_names)} derived features")
        else:
            features = base_features
        
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
        
        # 按轨迹分割文件
        n_files = len(csv_files)
        n_train_files = int(n_files * train_split)
        n_val_files = int(n_files * val_split)
        n_test_files = n_files - n_train_files - n_val_files
        
        train_files = csv_files[:n_train_files]
        val_files = csv_files[n_train_files:n_train_files + n_val_files]
        test_files = csv_files[n_train_files + n_val_files:]
        
        print(f"\nFile split:")
        print(f"  Train files: {len(train_files)} ({len(train_files)/n_files*100:.1f}%)")
        print(f"  Val files: {len(val_files)} ({len(val_files)/n_files*100:.1f}%)")
        print(f"  Test files: {len(test_files)} ({len(test_files)/n_files*100:.1f}%)")
        
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
        print(f"  Train samples: {X_train.shape[0]}")
        print(f"  Val samples: {X_val.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        
        # 检查数据范围（在标准化前）
        print(f"\nData range check before standardization:")
        for i, name in enumerate(feature_names):
            feature_data = X_train[:, :, i]
            min_val, max_val = np.min(feature_data), np.max(feature_data)
            if abs(max_val) > 1000 or abs(min_val) > 1000:
                print(f"  WARNING: {name} has extreme values: [{min_val:.2f}, {max_val:.2f}]")
        
        # 标准化特征和目标
        print("Standardizing features and targets...")
        
        # 特征标准化 (对每个特征维度分别标准化)
        scaler_X = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        scaler_X.fit(X_train_flat)
        
        # 检查标准化器的健康状况
        scale_min, scale_max = np.min(scaler_X.scale_), np.max(scaler_X.scale_)
        print(f"Scaler scale range: [{scale_min:.6f}, {scale_max:.6f}]")
        if scale_max / scale_min > 10000:
            print("  WARNING: Large scale difference detected!")
            problematic_features = np.where(scaler_X.scale_ > scale_min * 1000)[0]
            for idx in problematic_features:
                print(f"    Problem feature {idx}: {feature_names[idx]} (scale: {scaler_X.scale_[idx]:.2e})")
        
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
                'split_method': 'by_trajectory_safe_windows'
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
            f.write(f"Feature type: {self.feature_type}\n")
            f.write(f"Include derived features: {self.include_derived}\n")
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
    parser = argparse.ArgumentParser(description='机器人扭矩数据预处理')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='原始数据目录路径')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出文件路径 (.pkl)')
    parser.add_argument('--feature_type', type=str, default='control_command',
                        choices=['control_command', 'original'],
                        help='特征类型')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='时序序列长度')
    parser.add_argument('--sequence_step', type=int, default=1,
                        help='时序序列步长')
    parser.add_argument('--include_derived', action='store_true', default=True,
                        help='是否包含衍生特征')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--enable_smoothing', action='store_true',
                        help='启用数据平滑')
    
    args = parser.parse_args()
    
    # 配置平滑参数
    smoothing_config = None
    if args.enable_smoothing:
        smoothing_config = {
            'jvel': {'method': 'savgol', 'window_length': 11, 'polyorder': 3},
            'mvel': {'method': 'savgol', 'window_length': 11, 'polyorder': 3},
            'tau': {'method': 'gaussian', 'sigma': 1.0}
        }
    
    # 创建处理配置
    config = {
        'feature_type': args.feature_type,
        'sequence_length': args.sequence_length,
        'sequence_step': args.sequence_step,
        'include_derived': args.include_derived,
        'smoothing_config': smoothing_config,
        'dt': 0.001,  # 时间步长，根据实际情况调整
        'train_split': args.train_split,
        'val_split': args.val_split
    }
    
    print("Robot Torque Data Preprocessing")
    print("=" * 40)
    print(f"Input directory: {args.data_dir}")
    print(f"Output file: {args.output_path}")
    print(f"Configuration: {config}")
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