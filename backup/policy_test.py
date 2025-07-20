import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# 添加平滑处理所需的导入
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# 设置matplotlib不显示图像
plt.ioff()
import matplotlib
matplotlib.use('Agg')

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


def smooth_test_data(data, smooth_config=None, verbose=True):
    """
    对测试数据进行平滑处理
    
    Parameters:
    -----------
    data : np.ndarray
        原始数据
    smooth_config : dict
        平滑配置
    verbose : bool
        是否打印详细信息
        
    Returns:
    --------
    data_smoothed : np.ndarray
        平滑后的数据
    """
    if smooth_config is None:
        return data
    
    # 复制数据以避免修改原始数据
    data_smoothed = data.copy()
    
    # 定义数据索引
    jvel_indices = np.arange(162, 165)  # jvel[0:3]
    mvel_indices = np.arange(146, 149)  # mvel[0:3]
    tau_indices = np.arange(114, 117)   # tau[0:3]
    
    if verbose:
        print("\nApplying data smoothing to test data...")
        print("-" * 40)
    
    # 平滑关节速度 (jvel)
    if 'jvel' in smooth_config:
        config = smooth_config['jvel'].copy()
        method = config.pop('method')
        smoother = DataSmoother(method=method, **config)
        original = data_smoothed[:, jvel_indices].copy()
        data_smoothed[:, jvel_indices] = smoother.smooth(data_smoothed[:, jvel_indices])
        
        if verbose:
            change = np.sqrt(np.mean((data_smoothed[:, jvel_indices] - original)**2))
            print(f"Joint velocities (jvel) smoothed: RMSE change = {change:.6f}")
    
    # 平滑电机速度 (mvel)
    if 'mvel' in smooth_config:
        config = smooth_config['mvel'].copy()
        method = config.pop('method')
        smoother = DataSmoother(method=method, **config)
        original = data_smoothed[:, mvel_indices].copy()
        data_smoothed[:, mvel_indices] = smoother.smooth(data_smoothed[:, mvel_indices])
        
        if verbose:
            change = np.sqrt(np.mean((data_smoothed[:, mvel_indices] - original)**2))
            print(f"Motor velocities (mvel) smoothed: RMSE change = {change:.6f}")
    
    # 平滑扭矩 (tau) - 仅用于目标值
    if 'tau' in smooth_config:
        config = smooth_config['tau'].copy()
        method = config.pop('method')
        smoother = DataSmoother(method=method, **config)
        original = data_smoothed[:, tau_indices].copy()
        data_smoothed[:, tau_indices] = smoother.smooth(data_smoothed[:, tau_indices])
        
        if verbose:
            change = np.sqrt(np.mean((data_smoothed[:, tau_indices] - original)**2))
            print(f"Torques (tau) smoothed: RMSE change = {change:.6f}")
    
    if verbose:
        print("-" * 40)
    
    return data_smoothed

# ========== 特征创建函数（支持控制命令）==========
def create_features_with_control_command(data):
    """
    创建包含控制命令的特征
    
    Parameters:
    -----------
    data : np.ndarray
        原始数据
        
    Returns:
    --------
    features : np.ndarray
        处理后的特征，包含控制命令
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
    
    # 计算控制命令（控制误差）
    control_command = jpos_d - jpos  # 期望位置 - 实际位置
    
    # 组合特征
    features = np.hstack([
        jpos,            # 3 features - 关节实际位置
        jvel,            # 3 features - 关节速度
        control_command, # 3 features - 控制命令（NEW!）
        pos_d,           # 6 features - 笛卡尔期望位置
        mpos,            # 3 features - 电机位置
        mvel             # 3 features - 电机速度
    ])
    
    # 特征名称
    feature_names = [
        'jpos[0]', 'jpos[1]', 'jpos[2]',
        'jvel[0]', 'jvel[1]', 'jvel[2]',
        'ctrl_cmd[0]', 'ctrl_cmd[1]', 'ctrl_cmd[2]',  # 控制命令
        'pos_d[0]', 'pos_d[1]', 'pos_d[2]', 'pos_d[3]', 'pos_d[4]', 'pos_d[5]',
        'mpos[0]', 'mpos[1]', 'mpos[2]',
        'mvel[0]', 'mvel[1]', 'mvel[2]'
    ]
    
    return features, feature_names

# ========== 原有的类定义 ==========
class TorqueNet(nn.Module):
    """PyTorch神经网络模型"""
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate=0.1):
        super(TorqueNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TorqueModelTester:
    """扭矩模型测试器"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.smoothing_applied = False
        self.smoothing_config = None
        self.feature_type = 'original'  # 默认为原始特征
        self.feature_names = None
        
        # 加载模型
        self.load_model(model_path)
        
        # 根据模型类型设置特征索引
        if self.feature_type == 'control_command':
            # 使用控制命令特征，不需要索引
            self.feature_select = None
        else:
            # 原始特征索引
            self.feature_select = np.concatenate([
                np.arange(13, 16),   # jpos[0:3]
                np.arange(162, 165), # jvel[0:3]
                np.arange(76, 82),   # pos_d
                np.arange(130, 133), # mpos[0:3]
                np.arange(146, 149), # mvel[0:3]
                np.arange(194, 197), # jpos_d[0:3]
            ])
        
        # 目标索引
        self.target_indices = np.arange(114, 117)  # tau[0:3]
        
        # 关节名称
        self.joint_names = [
            'Joint 1',
            'Joint 2', 
            'Joint 3'
        ]
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        print(f"Loading model from: {model_path}")
        
        try:
            # 尝试安全加载（PyTorch 2.6+的新安全机制）
            import torch.serialization
            # 添加sklearn的StandardScaler到安全全局变量列表
            torch.serialization.add_safe_globals([
                'sklearn.preprocessing._data.StandardScaler',
                'sklearn.preprocessing.data.StandardScaler'
            ])
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e1:
            print(f"Safe loading failed, trying alternative method...")
            try:
                # 备用方法：明确设置weights_only=False
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            except Exception as e2:
                print(f"Alternative loading failed, trying legacy method...")
                try:
                    # 最后的备用方法：使用上下文管理器
                    from sklearn.preprocessing._data import StandardScaler
                    with torch.serialization.safe_globals([StandardScaler]):
                        checkpoint = torch.load(model_path, map_location=self.device)
                except Exception as e3:
                    print(f"All loading methods failed:")
                    print(f"  Method 1 error: {e1}")
                    print(f"  Method 2 error: {e2}")
                    print(f"  Method 3 error: {e3}")
                    raise RuntimeError("Cannot load model file. Please check PyTorch and sklearn versions.")
        
        # 检查模型文件格式
        if 'feature_dim' in checkpoint:
            # 完整的checkpoint文件
            feature_dim = checkpoint['feature_dim']
            target_dim = checkpoint['target_dim']
            hidden_layers = checkpoint['hidden_layers']
            
            # 重建模型
            self.model = TorqueNet(feature_dim, target_dim, hidden_layers)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 加载scaler
            self.scaler_x = checkpoint['scaler_x']
            self.scaler_y = checkpoint['scaler_y']
            
            # 检查是否有平滑配置
            if 'smoothing_applied' in checkpoint:
                self.smoothing_applied = checkpoint['smoothing_applied']
                self.smoothing_config = checkpoint.get('smoothing_config', None)
                
                if self.smoothing_applied:
                    print("\n⚠️  This model was trained with data smoothing enabled!")
                    print("Smoothing configuration:")
                    for data_type, config in self.smoothing_config.items():
                        print(f"  {data_type}: {config}")
                    print("The same smoothing will be applied to test data.\n")
            
            # 检查特征类型
            if 'feature_type' in checkpoint:
                self.feature_type = checkpoint['feature_type']
                if self.feature_type == 'control_command':
                    print("\n✅ This model uses CONTROL COMMAND features!")
                    print("Features include: jpos_d - jpos (control error)")
                    self.feature_names = checkpoint.get('feature_names', None)
            
            print(f"Model loaded successfully with {feature_dim} input features and {target_dim} outputs")
            print(f"Feature type: {self.feature_type}")
            print(f"Scaler information loaded successfully")
            
        else:
            # 只有权重的文件 (best_model.pth)
            print("Warning: Loading best_model.pth which only contains weights.")
            print("Using default model architecture...")
            
            # 使用默认架构参数（与训练脚本保持一致）
            feature_dim = 21  # 3+3+6+3+3+3 = 21个特征
            target_dim = 3    # 3个关节的扭矩
            hidden_layers = [600, 500, 400]
            
            # 重建模型
            self.model = TorqueNet(feature_dim, target_dim, hidden_layers)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            
            # 警告：没有scaler信息
            print("ERROR: best_model.pth doesn't contain scaler information!")
            print("Please use the complete model file: motor_torque_model_run_*.pth")
            print("Available model files should be in the models/ directory")
            
            # 列出可用的完整模型文件
            model_dir = os.path.dirname(model_path)
            available_models = [f for f in os.listdir(model_dir) 
                              if (f.startswith('motor_torque_model_run_') or 
                                  f.startswith('motor_torque_model_ctrl_cmd_run_')) and 
                              f.endswith('.pth')]
            
            if available_models:
                print("\nAvailable complete model files:")
                for model in available_models:
                    print(f"  - {os.path.join(model_dir, model)}")
                print("\nPlease use one of these files instead.")
            
            raise ValueError("Cannot proceed without scaler information. Please use a complete model file.")
    
    def load_test_data(self, test_file_path):
        """加载测试数据"""
        print(f"Loading test data from: {test_file_path}")
        
        data = np.loadtxt(test_file_path, delimiter=',')
        
        # 如果模型训练时使用了平滑，这里也要应用相同的平滑
        if self.smoothing_applied and self.smoothing_config:
            print("Applying smoothing to test data (same as training)...")
            data = smooth_test_data(data, self.smoothing_config, verbose=False)
        
        # 根据特征类型提取特征
        if self.feature_type == 'control_command':
            # 使用控制命令特征
            X_test, _ = create_features_with_control_command(data)
        else:
            # 使用原始特征索引
            X_test = data[:, self.feature_select]
        
        # 提取目标
        y_test = data[:, self.target_indices]
        
        # 处理NaN值
        nan_mask = np.isnan(X_test).any(axis=1) | np.isnan(y_test).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"Removing {nan_mask.sum()} samples with NaN values")
            X_test = X_test[~nan_mask]
            y_test = y_test[~nan_mask]
        
        print(f"Test data loaded: {X_test.shape[0]} samples")
        print(f"Feature dimension: {X_test.shape[1]}")
        
        return X_test, y_test
    
    def predict(self, X):
        """模型预测"""
        # 标准化输入
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 预测
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
        
        # 反标准化输出
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
    
    def calculate_percentage_errors(self, y_true, y_pred):
        """计算百分比误差"""
        # 避免除零错误，对于接近零的值使用小的epsilon
        epsilon = 1e-6
        
        # 绝对百分比误差
        abs_percentage_errors = []
        relative_percentage_errors = []
        
        for i in range(y_true.shape[1]):  # 对每个关节
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]
            
            # 方法1: 相对于真实值的百分比误差
            # 对于接近零的值，使用epsilon避免除零
            denominators = np.where(np.abs(true_vals) < epsilon, epsilon, np.abs(true_vals))
            rel_errors = np.abs(pred_vals - true_vals) / denominators * 100
            
            # 方法2: 相对于数据范围的百分比误差
            data_range = np.max(true_vals) - np.min(true_vals)
            if data_range < epsilon:
                data_range = epsilon
            range_errors = np.abs(pred_vals - true_vals) / data_range * 100
            
            abs_percentage_errors.append(range_errors)
            relative_percentage_errors.append(rel_errors)
        
        return abs_percentage_errors, relative_percentage_errors
    
    def detailed_analysis(self, y_true, y_pred, output_dir):
        """详细分析预测结果"""
        results = {}
        
        print("\n" + "="*60)
        print("DETAILED TORQUE PREDICTION ANALYSIS")
        print("="*60)
        
        # 整体性能
        overall_mse = mean_squared_error(y_true, y_pred)
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_rmse = np.sqrt(overall_mse)
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  RMSE: {overall_rmse:.6f} Nm")
        print(f"  MAE:  {overall_mae:.6f} Nm")
        print(f"  MSE:  {overall_mse:.6f}")
        
        results['overall'] = {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'mse': overall_mse
        }
        
        # 计算百分比误差
        abs_percentage_errors, rel_percentage_errors = self.calculate_percentage_errors(y_true, y_pred)
        
        # 每个关节的详细分析
        print(f"\nPER-JOINT ANALYSIS:")
        print("-" * 50)
        
        joint_results = []
        
        for i in range(3):  # 3个关节
            joint_name = self.joint_names[i]
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]
            
            # 基本统计
            joint_mse = mean_squared_error(true_vals, pred_vals)
            joint_mae = mean_absolute_error(true_vals, pred_vals)
            joint_rmse = np.sqrt(joint_mse)
            joint_max_error = np.max(np.abs(true_vals - pred_vals))
            
            # 数据范围
            true_min, true_max = np.min(true_vals), np.max(true_vals)
            true_range = true_max - true_min
            true_mean = np.mean(true_vals)
            true_std = np.std(true_vals)
            
            # 百分比误差统计
            abs_pct_errors = abs_percentage_errors[i]
            rel_pct_errors = rel_percentage_errors[i]
            
            # 过滤异常值（相对误差超过1000%的点，通常是真值接近零）
            valid_rel_mask = rel_pct_errors < 1000
            filtered_rel_errors = rel_pct_errors[valid_rel_mask]
            
            # 计算异常值统计
            outlier_count = np.sum(rel_pct_errors >= 1000)
            outlier_percentage = outlier_count / len(rel_pct_errors) * 100
            
            print(f"\n{joint_name}:")
            print(f"  Data Range: [{true_min:.4f}, {true_max:.4f}] Nm (range: {true_range:.4f} Nm)")
            print(f"  Mean ± Std: {true_mean:.4f} ± {true_std:.4f} Nm")
            print(f"  ")
            print(f"  Absolute Errors:")
            print(f"    RMSE: {joint_rmse:.6f} Nm")
            print(f"    MAE:  {joint_mae:.6f} Nm") 
            print(f"    Max Error: {joint_max_error:.6f} Nm")
            print(f"  ")
            print(f"  Percentage Errors (relative to data range):")
            print(f"    Mean: {np.mean(abs_pct_errors):.2f}%")
            print(f"    Median: {np.median(abs_pct_errors):.2f}%")
            print(f"    95th percentile: {np.percentile(abs_pct_errors, 95):.2f}%")
            print(f"    Max: {np.max(abs_pct_errors):.2f}%")
            print(f"  ")
            if len(filtered_rel_errors) > 0:
                print(f"  Percentage Errors (relative to true value):")
                print(f"    Mean: {np.mean(filtered_rel_errors):.2f}%")
                print(f"    Median: {np.median(filtered_rel_errors):.2f}%")
                print(f"    95th percentile: {np.percentile(filtered_rel_errors, 95):.2f}%")
                print(f"    Outliers (>1000%): {outlier_count} ({outlier_percentage:.1f}%)")
            
            # 误差分布统计
            error_thresholds = [0.01, 0.02, 0.05, 0.1]  # Nm
            print(f"  ")
            print(f"  Error Distribution:")
            for threshold in error_thresholds:
                within_threshold = np.sum(np.abs(true_vals - pred_vals) <= threshold)
                percentage = within_threshold / len(true_vals) * 100
                print(f"    Within ±{threshold:.3f} Nm: {within_threshold}/{len(true_vals)} ({percentage:.1f}%)")
            
            # 保存关节结果
            joint_result = {
                'name': joint_name,
                'rmse': joint_rmse,
                'mae': joint_mae,
                'max_error': joint_max_error,
                'data_range': true_range,
                'mean_abs_pct_error': np.mean(abs_pct_errors),
                'median_abs_pct_error': np.median(abs_pct_errors),
                'p95_abs_pct_error': np.percentile(abs_pct_errors, 95),
                'true_range': [true_min, true_max],
                'true_mean': true_mean,
                'true_std': true_std,
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_percentage
            }
            
            if len(filtered_rel_errors) > 0:
                joint_result.update({
                    'mean_rel_pct_error': np.mean(filtered_rel_errors),
                    'median_rel_pct_error': np.median(filtered_rel_errors),
                    'p95_rel_pct_error': np.percentile(filtered_rel_errors, 95),
                    'max_rel_pct_error': np.max(rel_pct_errors)  # 使用所有值的最大值
                })
            else:
                # 如果所有值都是异常值，提供默认值
                joint_result.update({
                    'mean_rel_pct_error': np.mean(rel_pct_errors),
                    'median_rel_pct_error': np.median(rel_pct_errors),
                    'p95_rel_pct_error': np.percentile(rel_pct_errors, 95),
                    'max_rel_pct_error': np.max(rel_pct_errors)
                })
            
            joint_results.append(joint_result)
        
        results['joints'] = joint_results
        
        # 生成可视化
        self.create_visualizations(y_true, y_pred, results, output_dir)
        
        # 保存详细报告
        self.save_detailed_report(results, output_dir)
        
        return results
    
    def create_visualizations(self, y_true, y_pred, results, output_dir):
        """创建可视化图表"""
        
        # 1. 预测 vs 真实值散点图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Predicted vs True Torque Values', fontsize=16)
        
        for i in range(3):
            ax = axes[i]
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]
            
            # 散点图
            ax.scatter(true_vals, pred_vals, alpha=0.5, s=1)
            
            # 完美预测线
            min_val = min(np.min(true_vals), np.min(pred_vals))
            max_val = max(np.max(true_vals), np.max(pred_vals))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax.set_xlabel('True Torque (Nm)')
            ax.set_ylabel('Predicted Torque (Nm)')
            ax.set_title(f'{self.joint_names[i]}\nRMSE: {results["joints"][i]["rmse"]:.4f} Nm')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加相关系数
            correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
            ax.text(0.05, 0.95, f'R²: {correlation**2:.4f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_vs_true_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 误差分布直方图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Error Distribution Analysis', fontsize=16)
        
        for i in range(3):
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]
            errors = pred_vals - true_vals
            abs_errors = np.abs(errors)
            
            # 绝对误差分布
            axes[0, i].hist(abs_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0, i].axvline(results['joints'][i]['mae'], color='red', linestyle='--', 
                              label=f'MAE: {results["joints"][i]["mae"]:.4f} Nm')
            axes[0, i].set_xlabel('Absolute Error (Nm)')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].set_title(f'{self.joint_names[i]} - Absolute Error Distribution')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # 相对误差分布
            axes[1, i].hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[1, i].axvline(0, color='red', linestyle='-', linewidth=2, label='Zero Error')
            axes[1, i].axvline(np.mean(errors), color='orange', linestyle='--', 
                              label=f'Mean Bias: {np.mean(errors):.4f} Nm')
            axes[1, i].set_xlabel('Error (Nm)')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_title(f'{self.joint_names[i]} - Error Distribution (Bias)')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box Plot - 误差分布箱线图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Error Distribution Box Plots', fontsize=16)
        
        # 准备数据
        abs_errors_all = []
        errors_all = []
        abs_pct_errors_all = []
        rel_pct_errors_all = []
        
        joint_labels = []
        
        for i in range(3):
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]
            errors = pred_vals - true_vals
            abs_errors = np.abs(errors)
            
            # 计算百分比误差
            data_range = np.max(true_vals) - np.min(true_vals)
            abs_pct_errors = abs_errors / data_range * 100
            
            epsilon = 1e-6
            denominators = np.where(np.abs(true_vals) < epsilon, epsilon, np.abs(true_vals))
            rel_pct_errors = abs_errors / denominators * 100
            
            abs_errors_all.append(abs_errors)
            errors_all.append(errors)
            abs_pct_errors_all.append(abs_pct_errors)
            rel_pct_errors_all.append(rel_pct_errors)
            joint_labels.append(f'Joint {i+1}')
        
        # 绝对误差box plot
        bp1 = axes[0, 0].boxplot(abs_errors_all, tick_labels=joint_labels, patch_artist=True)
        axes[0, 0].set_title('Absolute Error Distribution')
        axes[0, 0].set_ylabel('Absolute Error (Nm)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 着色
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        
        # 误差分布 box plot (带正负)
        bp2 = axes[0, 1].boxplot(errors_all, tick_labels=joint_labels, patch_artist=True)
        axes[0, 1].set_title('Error Distribution (with bias)')
        axes[0, 1].set_ylabel('Error (Nm)')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        # 百分比误差 box plot (相对于数据范围)
        bp3 = axes[1, 0].boxplot(abs_pct_errors_all, tick_labels=joint_labels, patch_artist=True)
        axes[1, 0].set_title('Percentage Error (relative to data range)')
        axes[1, 0].set_ylabel('Percentage Error (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
        
        # 百分比误差 box plot (相对于真实值) - 显示真实的所有数据
        bp4 = axes[1, 1].boxplot(rel_pct_errors_all, tick_labels=joint_labels, patch_artist=True, 
                                 showfliers=True, flierprops=dict(marker='o', markersize=0.5, alpha=0.3))
        axes[1, 1].set_title('Percentage Error (relative to true value, all data)')
        axes[1, 1].set_ylabel('Percentage Error (%)')
        axes[1, 1].set_yscale('log')  # 使用对数尺度来处理极大值
        axes[1, 1].grid(True, alpha=0.3)
        
        for patch, color in zip(bp4['boxes'], colors):
            patch.set_facecolor(color)
        
        # 添加说明文本
        max_errors = [np.max(rel_errors) for rel_errors in rel_pct_errors_all]
        info_text = f"Max errors: {', '.join([f'{max_err:.0f}%' for max_err in max_errors])}"
        axes[1, 1].text(0.02, 0.98, info_text, transform=axes[1, 1].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        verticalalignment='top', fontsize=9)
        
        # 添加统计信息文本
        stats_text = ""
        for i in range(3):
            stats_text += f"Joint {i+1}: "
            stats_text += f"Median AE: {np.median(abs_errors_all[i]):.4f} Nm, "
            stats_text += f"Median %E: {np.median(abs_pct_errors_all[i]):.2f}%\n"
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_boxplots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 时间序列预测图（可选择显示全部或采样）
        show_all_samples = len(y_true) <= 5000  # 如果数据量小于5000，显示全部
        
        if show_all_samples:
            sample_size = len(y_true)
            title_suffix = f"(All {sample_size} samples)"
        else:
            sample_size = min(2000, len(y_true))  # 增加到2000个点
            title_suffix = f"(First {sample_size}/{len(y_true)} samples)"
        
        indices = np.arange(sample_size)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Time Series Prediction Comparison {title_suffix}', fontsize=16)
        
        for i in range(3):
            ax = axes[i]
            ax.plot(indices, y_true[:sample_size, i], label='True', color='blue', alpha=0.7)
            ax.plot(indices, y_pred[:sample_size, i], label='Predicted', color='red', alpha=0.7)
            ax.fill_between(indices, 
                           y_true[:sample_size, i] - results['joints'][i]['mae'],
                           y_true[:sample_size, i] + results['joints'][i]['mae'],
                           alpha=0.2, color='gray', label=f'±MAE band')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Torque (Nm)')
            ax.set_title(f'{self.joint_names[i]} - Time Series Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 综合Box Plot对比图
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 创建所有关节的绝对误差对比
        all_abs_errors = []
        all_labels = []
        
        for i in range(3):
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]
            abs_errors = np.abs(pred_vals - true_vals)
            all_abs_errors.append(abs_errors)
            all_labels.append(f'{self.joint_names[i]}\n(MAE: {results["joints"][i]["mae"]:.4f} Nm)')
        
        bp = ax.boxplot(all_abs_errors, tick_labels=all_labels, patch_artist=True, 
                       showfliers=True, flierprops=dict(marker='o', markersize=0.8, alpha=0.4))
        
        # 美化
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax.set_title('Absolute Error Distribution Comparison Across Joints\n(All data points, no filtering)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Absolute Error (Nm)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加统计线和更多统计信息
        for i, abs_errors in enumerate(all_abs_errors):
            mean_error = np.mean(abs_errors)
            median_error = np.median(abs_errors)
            max_error = np.max(abs_errors)
            
            # 均值线
            ax.hlines(mean_error, i+0.8, i+1.2, colors='red', linestyles='solid', linewidth=2)
            ax.text(i+1, mean_error, f'  Mean: {mean_error:.4f}', va='center', fontsize=9)
            
            # 在图下方添加统计信息
            stats_text = f'Max: {max_error:.4f}\nQ99: {np.percentile(abs_errors, 99):.4f}'
            ax.text(i+1, ax.get_ylim()[1]*0.8, stats_text, ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_error_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization plots saved to: {output_dir}")
        print("Generated plots:")
        print("  - prediction_vs_true_scatter.png")
        print("  - error_distribution.png") 
        print("  - error_boxplots.png")
        print("  - time_series_comparison.png")
        print("  - comprehensive_error_boxplot.png")
    
    def save_detailed_report(self, results, output_dir):
        """保存详细报告"""
        report_path = os.path.join(output_dir, 'detailed_test_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("TORQUE PREDICTION MODEL - DETAILED TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Feature Type: {self.feature_type.upper()}\n")
            
            if self.feature_type == 'control_command':
                f.write("\nFeature Configuration:\n")
                f.write("  Using control command features (jpos_d - jpos)\n")
                if self.feature_names:
                    f.write("  Feature list:\n")
                    for i, name in enumerate(self.feature_names):
                        f.write(f"    {i}: {name}\n")
            
            # 记录平滑信息
            if self.smoothing_applied:
                f.write(f"\nData Smoothing: ENABLED\n")
                f.write("Smoothing Configuration:\n")
                for data_type, config in self.smoothing_config.items():
                    f.write(f"  {data_type}: {config}\n")
            else:
                f.write(f"\nData Smoothing: DISABLED\n")
            
            f.write("\n")
            
            # 整体性能
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            f.write(f"RMSE: {results['overall']['rmse']:.6f} Nm\n")
            f.write(f"MAE:  {results['overall']['mae']:.6f} Nm\n")
            f.write(f"MSE:  {results['overall']['mse']:.6f}\n\n")
            
            # 每个关节的详细结果
            f.write("PER-JOINT DETAILED ANALYSIS:\n")
            f.write("-" * 30 + "\n\n")
            
            for joint_result in results['joints']:
                f.write(f"{joint_result['name']}:\n")
                f.write(f"  Data Range: [{joint_result['true_range'][0]:.4f}, {joint_result['true_range'][1]:.4f}] Nm\n")
                f.write(f"  Data Range Span: {joint_result['data_range']:.4f} Nm\n")
                f.write(f"  Mean ± Std: {joint_result['true_mean']:.4f} ± {joint_result['true_std']:.4f} Nm\n")
                f.write(f"  \n")
                f.write(f"  Absolute Performance:\n")
                f.write(f"    RMSE: {joint_result['rmse']:.6f} Nm\n")
                f.write(f"    MAE:  {joint_result['mae']:.6f} Nm\n")
                f.write(f"    Max Error: {joint_result['max_error']:.6f} Nm\n")
                f.write(f"  \n")
                f.write(f"  Percentage Errors (relative to data range):\n")
                f.write(f"    Mean: {joint_result['mean_abs_pct_error']:.2f}%\n")
                f.write(f"    Median: {joint_result['median_abs_pct_error']:.2f}%\n")
                f.write(f"    95th percentile: {joint_result['p95_abs_pct_error']:.2f}%\n")
                f.write(f"  \n")
                f.write(f"  Percentage Errors (relative to true value, all data):\n")
                f.write(f"    Mean: {joint_result['mean_rel_pct_error']:.2f}%\n")
                f.write(f"    Median: {joint_result['median_rel_pct_error']:.2f}%\n")
                f.write(f"    95th percentile: {joint_result['p95_rel_pct_error']:.2f}%\n")
                f.write(f"    Max: {joint_result['max_rel_pct_error']:.2f}%\n")
                f.write(f"    Values > 1000%: {joint_result['outlier_count']} ({joint_result['outlier_percentage']:.1f}%)\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"Detailed report saved to: {report_path}")
    
    def test_model(self, test_data_paths, output_base_dir="torque_test_results"):
        """测试模型在多个测试数据集上的性能"""
        
        # 创建输出目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if self.feature_type == 'control_command':
            output_dir = f"{output_base_dir}_ctrl_cmd_{timestamp}"
        else:
            output_dir = f"{output_base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nTesting model on {len(test_data_paths)} datasets...")
        print(f"Results will be saved to: {output_dir}")
        
        all_results = []
        
        for i, test_path in enumerate(test_data_paths):
            print(f"\n{'='*60}")
            print(f"Testing on dataset {i+1}/{len(test_data_paths)}: {os.path.basename(test_path)}")
            print(f"{'='*60}")
            
            try:
                # 加载测试数据
                X_test, y_test = self.load_test_data(test_path)
                
                # 预测
                print("Running predictions...")
                y_pred = self.predict(X_test)
                
                # 创建数据集特定的输出目录
                dataset_dir = os.path.join(output_dir, f"dataset_{i+1}_{os.path.basename(test_path).replace('.csv', '')}")
                os.makedirs(dataset_dir, exist_ok=True)
                
                # 详细分析
                results = self.detailed_analysis(y_test, y_pred, dataset_dir)
                results['dataset_name'] = os.path.basename(test_path)
                results['dataset_index'] = i + 1
                
                all_results.append(results)
                
            except Exception as e:
                print(f"Error processing {test_path}: {e}")
                continue
        
        # 生成汇总报告
        if all_results:
            self.generate_summary_report(all_results, output_dir)
        
        print(f"\n{'='*60}")
        print("Testing completed!")
        print(f"All results saved to: {output_dir}")
        print(f"{'='*60}")
        
        return all_results
    
    def generate_summary_report(self, all_results, output_dir):
        """生成汇总报告"""
        summary_path = os.path.join(output_dir, 'summary_report.txt')
        
        with open(summary_path, 'w') as f:
            f.write("TORQUE PREDICTION MODEL - SUMMARY TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Feature Type: {self.feature_type.upper()}\n")
            f.write(f"Number of datasets tested: {len(all_results)}\n")
            
            # 记录平滑信息
            if self.smoothing_applied:
                f.write(f"\nData Smoothing: ENABLED\n")
                f.write("Note: The same smoothing configuration used in training was applied to all test data.\n")
            else:
                f.write(f"\nData Smoothing: DISABLED\n")
            
            f.write("\n")
            
            # 整体汇总
            overall_maes = [result['overall']['mae'] for result in all_results]
            overall_rmses = [result['overall']['rmse'] for result in all_results]
            
            f.write("OVERALL PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average MAE across all datasets: {np.mean(overall_maes):.6f} ± {np.std(overall_maes):.6f} Nm\n")
            f.write(f"Average RMSE across all datasets: {np.mean(overall_rmses):.6f} ± {np.std(overall_rmses):.6f} Nm\n\n")
            
            # 每个数据集的结果
            f.write("PER-DATASET RESULTS:\n")
            f.write("-" * 20 + "\n\n")
            
            for result in all_results:
                f.write(f"Dataset {result['dataset_index']}: {result['dataset_name']}\n")
                f.write(f"  Overall MAE: {result['overall']['mae']:.6f} Nm\n")
                f.write(f"  Overall RMSE: {result['overall']['rmse']:.6f} Nm\n")
                
                for joint_result in result['joints']:
                    f.write(f"  {joint_result['name']}: MAE={joint_result['mae']:.4f} Nm, ")
                    f.write(f"Avg %Error={joint_result['mean_abs_pct_error']:.2f}%\n")
                f.write("\n")
            
            # 关节级汇总
            f.write("PER-JOINT SUMMARY:\n")
            f.write("-" * 20 + "\n\n")
            
            for joint_idx in range(3):
                joint_name = self.joint_names[joint_idx]
                joint_maes = [result['joints'][joint_idx]['mae'] for result in all_results]
                joint_pct_errors = [result['joints'][joint_idx]['mean_abs_pct_error'] for result in all_results]
                
                f.write(f"{joint_name}:\n")
                f.write(f"  Average MAE: {np.mean(joint_maes):.6f} ± {np.std(joint_maes):.6f} Nm\n")
                f.write(f"  Average %Error: {np.mean(joint_pct_errors):.2f} ± {np.std(joint_pct_errors):.2f}%\n\n")
        
        print(f"Summary report saved to: {summary_path}")

def main():
    """主测试函数"""
    print("Torque Model Testing Script (With Control Command Support)")
    print("=" * 50)
    
    # 检查PyTorch版本并提供建议
    print(f"PyTorch version: {torch.__version__}")
    if hasattr(torch, '__version__') and float(torch.__version__.split('.')[0]) >= 2 and float(torch.__version__.split('.')[1]) >= 6:
        print("⚠️  Detected PyTorch 2.6+: Using enhanced security loading")
    
    # 配置路径
    model_path = input("Enter path to trained model (.pth file): ").strip()
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # 检查模型文件类型并提供建议
    if 'best_model.pth' in model_path:
        print(f"\n⚠️  Warning: You selected 'best_model.pth'")
        print("This file only contains model weights, not the complete training information.")
        
        model_dir = os.path.dirname(model_path)
        
        # 查找完整的模型文件
        complete_models = [f for f in os.listdir(model_dir) 
                          if (f.startswith('motor_torque_model_run_') or 
                              f.startswith('motor_torque_model_ctrl_cmd_run_')) and 
                          f.endswith('.pth')]
        
        if complete_models:
            complete_models.sort()
            print(f"\n✅ Found complete model files in the same directory:")
            for i, model in enumerate(complete_models):
                full_path = os.path.join(model_dir, model)
                model_type = "CONTROL COMMAND" if "ctrl_cmd" in model else "ORIGINAL"
                print(f"  {i+1}. {full_path} [{model_type}]")
            
            choice = input(f"\nDo you want to use one of these instead? (1-{len(complete_models)}, or 'n' to continue): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(complete_models):
                model_path = os.path.join(model_dir, complete_models[int(choice)-1])
                print(f"Using: {model_path}")
            elif choice.lower() != 'n':
                print("Invalid choice. Exiting...")
                return
    
    test_data_dir = input("Enter path to test data directory: ").strip()
    if not os.path.exists(test_data_dir):
        print(f"Test data directory not found: {test_data_dir}")
        return
    
    # 查找测试文件
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]
    if not test_files:
        print(f"No CSV files found in {test_data_dir}")
        return
    
    test_files.sort()
    test_paths = [os.path.join(test_data_dir, f) for f in test_files]
    
    print(f"\nFound {len(test_files)} test files:")
    for i, f in enumerate(test_files):
        print(f"  {i+1}. {f}")
    
    # 询问是否要测试所有文件
    if len(test_files) > 5:
        choice = input(f"\nTest all {len(test_files)} files? (y/n, default=y): ").strip().lower()
        if choice == 'n':
            # 让用户选择特定文件
            indices = input("Enter file numbers to test (e.g., 1,3,5 or 1-5): ").strip()
            try:
                if '-' in indices:
                    start, end = map(int, indices.split('-'))
                    selected_indices = list(range(start-1, end))
                else:
                    selected_indices = [int(x.strip())-1 for x in indices.split(',')]
                
                test_paths = [test_paths[i] for i in selected_indices if 0 <= i < len(test_paths)]
                print(f"Selected {len(test_paths)} files for testing")
            except:
                print("Invalid selection. Testing all files...")
    
    try:
        # 初始化测试器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        tester = TorqueModelTester(model_path, device)
        
        # 运行测试
        results = tester.test_model(test_paths)
        
        print("\nTesting completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print(f"\nTroubleshooting tips:")
        print(f"1. Make sure you're using a complete model file")
        print(f"   - Original features: motor_torque_model_run_*.pth")
        print(f"   - Control command features: motor_torque_model_ctrl_cmd_run_*.pth")
        print(f"2. Check that the test data files are in the correct format")
        print(f"3. Ensure the model was trained with the same feature configuration")
        
        # 检查PyTorch版本相关问题
        if "weights_only" in str(e) or "UnpicklingError" in str(e):
            print(f"4. PyTorch version compatibility issue detected.")
            print(f"   Current PyTorch: {torch.__version__}")
            print(f"   Try: pip install torch==2.1.0 (or another compatible version)")
            print(f"   Or: pip install --upgrade torch scikit-learn")

if __name__ == "__main__":
    main()