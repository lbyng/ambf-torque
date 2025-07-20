import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 添加平滑处理所需的导入
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

import utils.visualizer as visualizer

# 设置matplotlib不显示图像
plt.ioff()  # 关闭交互模式
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 创建输出目录结构
def create_output_directories():
    """创建输出目录结构"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = f"motor_torque_ctrl_cmd_results_{timestamp}"
    
    dirs = {
        'base': base_dir,
        'models': os.path.join(base_dir, 'models'),
        'plots': os.path.join(base_dir, 'plots'),
        'data_viz': os.path.join(base_dir, 'plots', 'data_visualization'),
        'training': os.path.join(base_dir, 'plots', 'training_history'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Output directories created under: {base_dir}")
    return dirs

# 设置数据路径
path_data = './data/'

def auto_detect_files(data_path):
    """自动检测和分类数据文件"""
    import glob
    
    # 获取所有CSV文件
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    all_files = [os.path.basename(f) for f in all_files]
    
    print(f"Found {len(all_files)} CSV files in {data_path}")
    
    # 根据文件名模式分类
    train_files = []
    test_files = []
    other_files = []
    
    for file in all_files:
        if any(pattern in file for pattern in ['_yz_', '_02_', '_03_', '_05_', '_016_', '_025_']):
            train_files.append(file)
        elif any(pattern in file for pattern in ['rand1200', 'rand_']):
            test_files.append(file)
        else:
            other_files.append(file)
    
    # 按文件名排序
    train_files.sort()
    test_files.sort()
    other_files.sort()
    
    print(f"\nAutomatically categorized files:")
    print(f"Training files ({len(train_files)}):")
    for f in train_files:
        print(f"  - {f}")
    
    print(f"\nTest files ({len(test_files)}):")
    for f in test_files:
        print(f"  - {f}")
    
    if other_files:
        print(f"\nOther files ({len(other_files)}):")
        for f in other_files:
            print(f"  - {f}")
        
        print(f"\nHow to handle 'other' files?")
        print("1. Add to training set")
        print("2. Add to test set") 
        print("3. Ignore them")
        choice = input("Enter choice (1/2/3, default=3): ").strip()
        
        if choice == '1':
            train_files.extend(other_files)
            print("Added other files to training set")
        elif choice == '2':
            test_files.extend(other_files)
            print("Added other files to test set")
        else:
            print("Ignoring other files")
    
    return train_files, test_files

# 自动检测文件
print("=== Automatic File Detection ===")
try:
    file_train_list_, file_test_list_ = auto_detect_files(path_data)
    
    if not file_train_list_ and not file_test_list_:
        print("Auto-detection failed. Using default file patterns...")
        file_train_list_ = ['data_record_yz_05.csv', 'data_record_yz_03.csv', 
                           'data_record_yz_025.csv', 'data_record_yz_02.csv', 
                           'data_record_yz_016.csv']
        file_test_list_ = ['data_record_rand1200_hm0.csv', 'data_record_rand1200_hm1.csv', 
                          'data_record_rand1200_hm2.csv', 'data_record_rand1200_hm3.csv', 
                          'data_record_rand1200_hm4.csv', 'data_record_rand1200_hm5.csv']
        
except Exception as e:
    print(f"Auto-detection error: {e}")
    file_train_list_ = []
    file_test_list_ = []

# 训练参数
hidden_layers = [600, 500, 400]
learning_rate = 0.0005
batch_size = 1024
epochs = 10
repeat_times = 1

# 正则化参数
weight_decay = 1e-4  # L2正则化
dropout_rate = 0.1   # Dropout

# 目标变量：前3个关节的扭矩
target_indices = np.arange(114, 117)  # tau[0:3]
target_dim = target_indices.size

print(f"\nTarget dimension: {target_dim}")
print(f"Target indices (tau[0:3]): {target_indices}")

# ========== 新增：特征创建函数 ==========
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
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class TorqueTrainer:
    """训练器类"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def prepare_data(self, X, y):
        """准备数据"""
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # 转换为PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        return X_tensor, y_tensor
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, weight_decay, output_dirs):
        """训练模型"""
        # 准备数据
        X_train_tensor, y_train_tensor = self.prepare_data(X_train, y_train)
        
        # 验证数据使用相同的scaler
        X_val_scaled = self.scaler_x.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
        
        # 训练历史
        history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        print(f"Starting training on {self.device}...")
        
        # 创建训练日志文件
        log_path = os.path.join(output_dirs['logs'], 'training_log.txt')
        with open(log_path, 'w') as log_file:
            log_file.write(f"Training started on {self.device}\n")
            log_file.write(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}\n")
            log_file.write(f"Weight decay: {weight_decay}\n\n")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_mae += F.l1_loss(outputs, y_batch).item()
            
            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_mae = F.l1_loss(val_outputs, y_val_tensor).item()
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_path = os.path.join(output_dirs['models'], 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                with open(log_path, 'a') as log_file:
                    log_file.write(f"Early stopping at epoch {epoch+1}\n")
                break
            
            # 打印和记录进度
            if (epoch + 1) % 20 == 0 or epoch < 10:
                current_lr = optimizer.param_groups[0]['lr']
                progress_msg = (f"Epoch [{epoch+1}/{epochs}] - "
                              f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}, "
                              f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}, "
                              f"LR: {current_lr:.2e}")
                print(progress_msg)
                
                with open(log_path, 'a') as log_file:
                    log_file.write(progress_msg + '\n')
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(best_model_path))
        
        # 保存训练历史到文件
        history_path = os.path.join(output_dirs['logs'], 'training_history.txt')
        with open(history_path, 'w') as f:
            f.write("Epoch,Train_Loss,Val_Loss,Train_MAE,Val_MAE\n")
            for i in range(len(history['train_loss'])):
                f.write(f"{i+1},{history['train_loss'][i]:.6f},{history['val_loss'][i]:.6f},"
                       f"{history['train_mae'][i]:.6f},{history['val_mae'][i]:.6f}\n")
        
        print(f"Training log saved to: {log_path}")
        print(f"Training history saved to: {history_path}")
        print(f"Best model saved to: {best_model_path}")
        
        return history
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
        
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred

# ========== 数据平滑处理类和函数 ==========
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


def smooth_robot_data(data_train, smooth_config=None, verbose=True):
    """
    对机器人数据进行平滑处理
    
    Parameters:
    -----------
    data_train : np.ndarray
        原始训练数据
    smooth_config : dict
        平滑配置，包含各数据类型的平滑参数
    verbose : bool
        是否打印详细信息
        
    Returns:
    --------
    data_smoothed : np.ndarray
        平滑后的数据
    """
    if smooth_config is None:
        # 默认平滑配置
        smooth_config = {
            'jvel': {
                'method': 'savgol',
                'window_length': 11,
                'polyorder': 3
            },
            'mvel': {
                'method': 'savgol',
                'window_length': 11,
                'polyorder': 3
            },
            'tau': {
                'method': 'gaussian',
                'sigma': 1.5
            }
        }
    
    # 复制数据以避免修改原始数据
    data_smoothed = data_train.copy()
    
    # 定义数据索引
    jvel_indices = np.arange(162, 165)  # jvel[0:3]
    mvel_indices = np.arange(146, 149)  # mvel[0:3]
    tau_indices = np.arange(114, 117)   # tau[0:3]
    
    if verbose:
        print("\nApplying data smoothing...")
        print("-" * 40)
    
    # 平滑关节速度 (jvel)
    if 'jvel' in smooth_config:
        config = smooth_config['jvel'].copy()  # 复制配置以避免修改原始字典
        method = config.pop('method')  # 提取method并从config中移除
        smoother = DataSmoother(method=method, **config)
        original_jvel = data_smoothed[:, jvel_indices].copy()
        data_smoothed[:, jvel_indices] = smoother.smooth(data_smoothed[:, jvel_indices])
        
        if verbose:
            rmse_change = np.sqrt(np.mean((data_smoothed[:, jvel_indices] - original_jvel)**2))
            print(f"Joint velocities (jvel) smoothed using {method}")
            print(f"  RMSE change: {rmse_change:.6f}")
            print(f"  Max change: {np.max(np.abs(data_smoothed[:, jvel_indices] - original_jvel)):.6f}")
    
    # 平滑电机速度 (mvel)
    if 'mvel' in smooth_config:
        config = smooth_config['mvel'].copy()  # 复制配置以避免修改原始字典
        method = config.pop('method')  # 提取method并从config中移除
        smoother = DataSmoother(method=method, **config)
        original_mvel = data_smoothed[:, mvel_indices].copy()
        data_smoothed[:, mvel_indices] = smoother.smooth(data_smoothed[:, mvel_indices])
        
        if verbose:
            rmse_change = np.sqrt(np.mean((data_smoothed[:, mvel_indices] - original_mvel)**2))
            print(f"\nMotor velocities (mvel) smoothed using {method}")
            print(f"  RMSE change: {rmse_change:.6f}")
            print(f"  Max change: {np.max(np.abs(data_smoothed[:, mvel_indices] - original_mvel)):.6f}")
    
    # 平滑扭矩 (tau)
    if 'tau' in smooth_config:
        config = smooth_config['tau'].copy()  # 复制配置以避免修改原始字典
        method = config.pop('method')  # 提取method并从config中移除
        smoother = DataSmoother(method=method, **config)
        original_tau = data_smoothed[:, tau_indices].copy()
        data_smoothed[:, tau_indices] = smoother.smooth(data_smoothed[:, tau_indices])
        
        if verbose:
            rmse_change = np.sqrt(np.mean((data_smoothed[:, tau_indices] - original_tau)**2))
            print(f"\nTorques (tau) smoothed using {method}")
            print(f"  RMSE change: {rmse_change:.6f}")
            print(f"  Max change: {np.max(np.abs(data_smoothed[:, tau_indices] - original_tau)):.6f}")
    
    if verbose:
        print("-" * 40)
        print("Data smoothing completed!")
    
    return data_smoothed


# ========== 原有的辅助函数 ==========
def load_data():
    """加载数据"""
    print("Loading training data...")
    
    # 检查数据目录是否存在
    if not os.path.exists(path_data):
        print(f"Data directory {path_data} not found!")
        print("Please create the directory and place your CSV files there.")
        return None, None
    
    # 加载训练数据
    data_train = None
    loaded_train_files = 0
    
    for i, file_training in enumerate(file_train_list_):
        file_path = os.path.join(path_data, file_training)
        if os.path.exists(file_path):
            try:
                print(f"Loading {file_training}...")
                data = np.loadtxt(file_path, delimiter=',')
                
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                print(f"  Loaded: {data.shape} (rows: {data.shape[0]}, cols: {data.shape[1]})")
                
                if data_train is None:
                    data_train = data
                else:
                    data_train = np.vstack((data_train, data))
                loaded_train_files += 1
                
            except Exception as e:
                print(f"  Error loading {file_training}: {e}")
        else:
            print(f"  Warning: {file_path} not found, skipping...")
    
    if data_train is None:
        raise FileNotFoundError("No training data files found or loaded successfully!")
    
    print(f"Successfully loaded {loaded_train_files} training files")
    print(f"Combined training data shape: {data_train.shape}")
    
    # 加载测试数据
    print(f"\nLoading test data...")
    data_test_list = []
    loaded_test_files = 0
    
    for file_test in file_test_list_:
        file_path = os.path.join(path_data, file_test)
        if os.path.exists(file_path):
            try:
                print(f"Loading {file_test}...")
                data = np.loadtxt(file_path, delimiter=',')
                
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                    
                print(f"  Loaded: {data.shape}")
                data_test_list.append(data)
                loaded_test_files += 1
                
            except Exception as e:
                print(f"  Error loading {file_test}: {e}")
        else:
            print(f"  Warning: {file_path} not found, skipping...")
    
    if not data_test_list:
        print("Warning: No test data files found!")
        print("Creating test set from training data...")
        split_idx = int(0.8 * len(data_train))
        test_portion = data_train[split_idx:]
        data_train = data_train[:split_idx]
        data_test_list = [test_portion]
        print(f"Split training data: train={data_train.shape}, test={test_portion.shape}")
    else:
        print(f"Successfully loaded {loaded_test_files} test files")
    
    # 数据维度检查
    expected_cols = 252
    if data_train.shape[1] != expected_cols:
        print(f"Warning: Expected {expected_cols} columns, but got {data_train.shape[1]}")
        print("This might affect feature indexing. Please verify your data format.")
    
    return data_train, data_test_list

def evaluate_model(trainer, X_test, y_test):
    """评估模型"""
    y_pred = trainer.predict(X_test)
    
    # 计算整体指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # 每个关节的指标
    joint_names = ['Joint 1 (Shoulder Rotation)', 'Joint 2 (Shoulder Pitch)', 'Joint 3 (Insertion)']
    joint_rmse = []
    joint_mae = []
    joint_max_error = []
    
    for i in range(3):
        joint_mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        joint_rmse.append(np.sqrt(joint_mse))
        joint_mae.append(mean_absolute_error(y_test[:, i], y_pred[:, i]))
        joint_max_error.append(np.max(np.abs(y_test[:, i] - y_pred[:, i])))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'joint_rmse': joint_rmse,
        'joint_mae': joint_mae,
        'joint_max_error': joint_max_error,
        'joint_names': joint_names,
        'y_pred': y_pred
    }

def save_results_summary(results_list, output_dirs):
    """保存结果总结"""
    summary_path = os.path.join(output_dirs['logs'], 'results_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("MOTOR TORQUE PREDICTION RESULTS SUMMARY (WITH CONTROL COMMANDS)\n")
        f.write("="*60 + "\n\n")
        
        # 写入每次运行的结果
        for i, result in enumerate(results_list):
            f.write(f"Run {i+1} Results:\n")
            f.write(f"  Training time: {result['training_time']:.2f} seconds\n")
            
            for test_idx, test_result in enumerate(result['test_results']):
                f.write(f"  Test Set {test_idx+1}:\n")
                f.write(f"    Overall RMSE: {test_result['rmse']:.6f} Nm\n")
                f.write(f"    Overall MAE:  {test_result['mae']:.6f} Nm\n")
                
                for j, joint_name in enumerate(test_result['joint_names']):
                    rmse = test_result['joint_rmse'][j]
                    mae = test_result['joint_mae'][j]
                    max_err = test_result['joint_max_error'][j]
                    f.write(f"    {joint_name}:\n")
                    f.write(f"      RMSE: {rmse:.4f} Nm, MAE: {mae:.4f} Nm, Max Error: {max_err:.4f} Nm\n")
            f.write("\n")
        
        # 计算和写入平均结果
        if len(results_list) > 1:
            f.write("AVERAGE RESULTS ACROSS ALL RUNS:\n")
            f.write("-" * 30 + "\n")
            
            num_test_sets = len(results_list[0]['test_results'])
            for test_idx in range(num_test_sets):
                rmse_values = [results_list[run]['test_results'][test_idx]['rmse'] for run in range(len(results_list))]
                mae_values = [results_list[run]['test_results'][test_idx]['mae'] for run in range(len(results_list))]
                
                avg_rmse = np.mean(rmse_values)
                std_rmse = np.std(rmse_values)
                avg_mae = np.mean(mae_values)
                std_mae = np.std(mae_values)
                
                f.write(f"Test Set {test_idx + 1}:\n")
                f.write(f"  RMSE: {avg_rmse:.6f} ± {std_rmse:.6f} Nm\n")
                f.write(f"  MAE:  {avg_mae:.6f} ± {std_mae:.6f} Nm\n\n")
    
    print(f"Results summary saved to: {summary_path}")

# ========== 修改后的主函数 ==========
def main():
    """主函数"""
    print("Starting Motor Torque Prediction with Control Commands...")
    
    # 创建输出目录
    output_dirs = create_output_directories()
    
    # 加载数据
    try:
        data_train, data_test_list = load_data()
        print(f"Training data shape: {data_train.shape}")
        print(f"Number of test datasets: {len(data_test_list)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("请确保数据文件存在于正确的路径中")
        return
    
    
    # Data visualization
    visualizer.visualize_data(data_train, output_dirs)
    visualizer.visualize_control_commands(data_train, output_dirs)
    
    
    # ========== 添加数据平滑处理 ==========
    # 配置平滑参数
    smooth_config = {
        'jvel': {
            'method': 'savgol',      # Savitzky-Golay滤波器适合保持信号形状
            'window_length': 11,     # 窗口长度（必须是奇数）
            'polyorder': 3           # 多项式阶数
        },
        'mvel': {
            'method': 'savgol',      # 电机速度也使用Savitzky-Golay
            'window_length': 11,
            'polyorder': 3
        },
        'tau': {
            'method': 'gaussian',    # 高斯滤波器提供更平滑的结果
            'sigma': 1.5            # 标准差控制平滑程度
        }
    }
    
    # 询问用户是否应用平滑
    apply_smooth = input("\nApply data smoothing to velocities and torques? (y/n, default=y): ").strip().lower()
    
    if apply_smooth != 'n':
        print("\n" + "="*50)
        print("APPLYING DATA SMOOTHING")
        print("="*50)
        
        # 应用平滑处理到原始数据
        data_train = smooth_robot_data(data_train, smooth_config, verbose=True)
        data_test_list = [smooth_robot_data(data_test, smooth_config, verbose=False) 
                         for data_test in data_test_list]
        
        # 保存平滑配置
        smooth_config_path = os.path.join(output_dirs['logs'], 'smoothing_config.txt')
        with open(smooth_config_path, 'w') as f:
            f.write("DATA SMOOTHING CONFIGURATION\n")
            f.write("="*30 + "\n\n")
            for data_type, config in smooth_config.items():
                f.write(f"{data_type}:\n")
                for key, value in config.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        print(f"Smoothing configuration saved to: {smooth_config_path}")
    else:
        print("Skipping data smoothing...")
    # ========== 平滑处理结束 ==========
    
    # 创建特征（使用控制命令）
    print("\n" + "="*50)
    print("CREATING FEATURES WITH CONTROL COMMANDS")
    print("="*50)
    
    X_train, feature_names = create_features_with_control_command(data_train)
    y_train = data_train[:, target_indices]
    
    # 为测试数据创建特征
    X_test_list = []
    y_test_list = []
    for data_test in data_test_list:
        X_test, _ = create_features_with_control_command(data_test)
        y_test = data_test[:, target_indices]
        X_test_list.append(X_test)
        y_test_list.append(y_test)
    
    print(f"\nFeature configuration with control command:")
    print(f"Total features: {X_train.shape[1]}")
    print(f"Feature breakdown:")
    print(f"  - Joint positions (jpos): 3 features")
    print(f"  - Joint velocities (jvel): 3 features")
    print(f"  - Control commands (jpos_d - jpos): 3 features  [NEW!]")
    print(f"  - Cartesian desired position (pos_d): 6 features")
    print(f"  - Motor positions (mpos): 3 features")
    print(f"  - Motor velocities (mvel): 3 features")
    print(f"  Total: 21 features")
    
    # 保存特征配置信息
    feature_config_path = os.path.join(output_dirs['logs'], 'feature_configuration.txt')
    with open(feature_config_path, 'w') as f:
        f.write("FEATURE CONFIGURATION (WITH CONTROL COMMANDS)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total features: {X_train.shape[1]}\n\n")
        f.write("Feature list:\n")
        for i, name in enumerate(feature_names):
            f.write(f"  {i}: {name}\n")
        f.write("\nControl command features (indices 6-8) are computed as: jpos_d - jpos\n")
    
    # 处理NaN值
    nan_mask = np.isnan(X_train).any(axis=1) | np.isnan(y_train).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"Removing {nan_mask.sum()} samples with NaN values...")
        X_train = X_train[~nan_mask]
        y_train = y_train[~nan_mask]
    
    print(f"\nFinal training data:")
    print(f"Features shape: {X_train.shape}")
    print(f"Targets shape: {y_train.shape}")
    
    # 数据分割
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 存储结果
    results_list = []
    
    # 设置特征维度
    feature_dim = X_train.shape[1]  # 现在是21维而不是原来的特征维度
    
    # 多次训练
    for run in range(repeat_times):
        print(f"\n{'='*50}")
        print(f"Training Run {run + 1}/{repeat_times}")
        print(f"{'='*50}")
        
        # 设置随机种子
        torch.manual_seed(10 * run + 3)
        np.random.seed(10 * run + 3)
        
        # 创建模型和训练器
        model = TorqueNet(feature_dim, target_dim, hidden_layers, dropout_rate)
        trainer = TorqueTrainer(model, device)
        
        print(f"Model created with {feature_dim} input features and {target_dim} output targets")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练模型
        start_time = time.time()
        history = trainer.train(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            output_dirs=output_dirs
        )
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # 绘制并保存训练历史
        visualizer.plot_training_history(history, run + 1, output_dirs)
        
        # 在测试集上评估
        test_results = []
        for i, (X_test, y_test) in enumerate(zip(X_test_list, y_test_list)):
            # 处理测试集的NaN值
            test_nan_mask = np.isnan(X_test).any(axis=1) | np.isnan(y_test).any(axis=1)
            if test_nan_mask.sum() > 0:
                print(f"Removing {test_nan_mask.sum()} NaN samples from test set {i+1}")
                X_test = X_test[~test_nan_mask]
                y_test = y_test[~test_nan_mask]
            
            result = evaluate_model(trainer, X_test, y_test)
            test_results.append(result)
            
            print(f"Test Set {i+1} Results:")
            print(f"  Overall RMSE: {result['rmse']:.6f} Nm")
            print(f"  Overall MAE:  {result['mae']:.6f} Nm")
            
            print(f"  Per-joint results:")
            for j in range(3):
                joint_name = result['joint_names'][j]
                rmse = result['joint_rmse'][j]
                mae = result['joint_mae'][j]
                max_err = result['joint_max_error'][j]
                print(f"    {joint_name}:")
                print(f"      RMSE: {rmse:.4f} Nm, MAE: {mae:.4f} Nm, Max Error: {max_err:.4f} Nm")
        
        # 保存模型（包含平滑信息和特征配置）
        model_path = os.path.join(output_dirs['models'], f'motor_torque_model_ctrl_cmd_run_{run+1}.pth')
        save_dict = {
            'model_state_dict': trainer.model.state_dict(),
            'scaler_x': trainer.scaler_x,
            'scaler_y': trainer.scaler_y,
            'feature_dim': feature_dim,
            'target_dim': target_dim,
            'hidden_layers': hidden_layers,
            'smoothing_applied': apply_smooth != 'n',
            'smoothing_config': smooth_config if apply_smooth != 'n' else None,
            'feature_type': 'control_command',  # 标记使用了控制命令特征
            'feature_names': feature_names
        }
        torch.save(save_dict, model_path)
        print(f"Model saved to: {model_path}")
        
        results_list.append({
            'run': run + 1,
            'training_time': training_time,
            'test_results': test_results,
            'history': history
        })
    
    # 保存结果总结
    save_results_summary(results_list, output_dirs)
    
    # 汇总结果显示
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY (WITH CONTROL COMMANDS)")
    print(f"{'='*50}")
    
    # 计算平均性能
    if len(X_test_list) > 0:
        for test_idx in range(len(X_test_list)):
            rmse_values = [results_list[run]['test_results'][test_idx]['rmse'] for run in range(repeat_times)]
            mae_values = [results_list[run]['test_results'][test_idx]['mae'] for run in range(repeat_times)]
            
            avg_rmse = np.mean(rmse_values)
            std_rmse = np.std(rmse_values)
            avg_mae = np.mean(mae_values)
            std_mae = np.std(mae_values)
            
            print(f"Test Set {test_idx + 1}:")
            print(f"  RMSE: {avg_rmse:.6f} ± {std_rmse:.6f} Nm")
            print(f"  MAE:  {avg_mae:.6f} ± {std_mae:.6f} Nm")
    
    print(f"\nAll results saved to: {output_dirs['base']}")
    print("Training completed successfully with Control Commands!")

if __name__ == "__main__":
    main()