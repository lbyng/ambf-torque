import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

# 设置matplotlib
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# ========== 简单的LSTM模型（与训练脚本相同）==========
class SimpleLSTM(nn.Module):
    """简单的LSTM扭矩预测模型"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=3):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])
        return output

# ========== 丰富的可视化测试器 ==========
class EnhancedTester:
    """增强的测试器，包含丰富的可视化"""
    
    def __init__(self, model_path, output_dir='./test_results'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.joint_names = ['Joint 1', 'Joint 2', 'Joint 3']
        self.joint_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载模型"""
        print(f"Loading model: {model_path}")
        
        # 修复PyTorch 2.6+的安全加载问题
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 重建模型
        input_dim = checkpoint['input_dim']
        self.model = SimpleLSTM(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载标准化器
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        
        print(f"Model loaded successfully! Input dimension: {input_dim}")
    
    def test(self, dataset):
        """测试模型并生成可视化"""
        print("\nStarting test...")
        
        # 准备测试数据
        X_test = torch.FloatTensor(dataset['X_test']).to(self.device)
        y_test_scaled = dataset['y_test']
        
        print(f"Test data: {X_test.shape}")
        
        # 预测
        with torch.no_grad():
            y_pred_scaled = self.model(X_test).cpu().numpy()
        
        # 反标准化
        y_test = self.scaler_y.inverse_transform(y_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # 计算指标
        results = self.calculate_metrics(y_test, y_pred)
        
        # 生成所有可视化
        self.create_all_visualizations(y_test, y_pred, results)
        
        return results
    
    def calculate_metrics(self, y_test, y_pred):
        """计算详细指标"""
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"\nOverall Test Results:")
        print(f"  RMSE: {rmse:.6f} Nm")
        print(f"  MAE:  {mae:.6f} Nm")
        print(f"  MSE:  {mse:.6f}")
        
        # 每个关节的详细指标
        joint_results = []
        print(f"\nDetailed Results by Joint:")
        
        for i in range(3):
            joint_mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            joint_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            joint_rmse = np.sqrt(joint_mse)
            
            # 计算相关系数
            correlation = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
            r_squared = correlation ** 2
            
            # 详细的百分比误差分析
            abs_errors = np.abs(y_test[:, i] - y_pred[:, i])
            
            # 1. 相对于数据范围的百分比误差
            data_range = np.max(y_test[:, i]) - np.min(y_test[:, i])
            range_pct_errors = (abs_errors / data_range) * 100
            
            # 2. 相对于真实值的百分比误差
            epsilon = 1e-6  # 避免除零
            true_vals_safe = np.where(np.abs(y_test[:, i]) < epsilon, epsilon, np.abs(y_test[:, i]))
            relative_pct_errors = (abs_errors / true_vals_safe) * 100
            
            # 3. 相对于均值的百分比误差
            data_mean = np.abs(np.mean(y_test[:, i]))
            if data_mean < epsilon:
                data_mean = epsilon
            mean_pct_errors = (abs_errors / data_mean) * 100
            
            # 4. 相对于标准差的归一化误差
            data_std = np.std(y_test[:, i])
            if data_std < epsilon:
                data_std = epsilon
            normalized_errors = abs_errors / data_std
            
            joint_result = {
                'name': self.joint_names[i],
                'rmse': joint_rmse,
                'mae': joint_mae,
                'mse': joint_mse,
                'r_squared': r_squared,
                
                # 百分比误差统计
                'range_pct_mean': np.mean(range_pct_errors),
                'range_pct_median': np.median(range_pct_errors),
                'range_pct_95': np.percentile(range_pct_errors, 95),
                'range_pct_max': np.max(range_pct_errors),
                
                'relative_pct_mean': np.mean(relative_pct_errors),
                'relative_pct_median': np.median(relative_pct_errors),
                'relative_pct_95': np.percentile(relative_pct_errors, 95),
                
                'mean_pct_mean': np.mean(mean_pct_errors),
                'mean_pct_median': np.median(mean_pct_errors),
                
                'normalized_error_mean': np.mean(normalized_errors),
                'normalized_error_median': np.median(normalized_errors),
                
                # 数据特征
                'data_range': data_range,
                'data_mean': np.mean(y_test[:, i]),
                'data_std': data_std,
                'data_min': np.min(y_test[:, i]),
                'data_max': np.max(y_test[:, i])
            }
            joint_results.append(joint_result)
            
            print(f"\n  {self.joint_names[i]}:")
            print(f"    Absolute Metrics:")
            print(f"      RMSE: {joint_rmse:.6f} Nm")
            print(f"      MAE:  {joint_mae:.6f} Nm")
            print(f"      R²:   {r_squared:.8f}")
            
            print(f"    Data Characteristics:")
            print(f"      Range: [{np.min(y_test[:, i]):.4f}, {np.max(y_test[:, i]):.4f}] Nm")
            print(f"      Mean:  {np.mean(y_test[:, i]):.4f} Nm")
            print(f"      Std:   {data_std:.4f} Nm")
            
            print(f"    Percentage Errors:")
            print(f"      Relative to Range:")
            print(f"        Mean: {np.mean(range_pct_errors):.4f}%")
            print(f"        Median: {np.median(range_pct_errors):.4f}%")
            print(f"        95th percentile: {np.percentile(range_pct_errors, 95):.4f}%")
            print(f"        Max: {np.max(range_pct_errors):.4f}%")
            
            print(f"      Relative to True Value:")
            print(f"        Mean: {np.mean(relative_pct_errors):.4f}%")
            print(f"        Median: {np.median(relative_pct_errors):.4f}%")
            print(f"        95th percentile: {np.percentile(relative_pct_errors, 95):.4f}%")
            
            print(f"      Relative to Data Mean:")
            print(f"        Mean: {np.mean(mean_pct_errors):.4f}%")
            print(f"        Median: {np.median(mean_pct_errors):.4f}%")
            
            print(f"    Normalized Error (relative to std):")
            print(f"      Mean: {np.mean(normalized_errors):.4f}σ")
            print(f"      Median: {np.median(normalized_errors):.4f}σ")
            
            # 误差分布统计
            print(f"    Error Distribution:")
            thresholds_abs = [0.0001, 0.0005, 0.001, 0.005]  # 绝对误差阈值 (Nm)
            for threshold in thresholds_abs:
                within_count = np.sum(abs_errors <= threshold)
                within_pct = (within_count / len(abs_errors)) * 100
                print(f"      Within ±{threshold:.4f} Nm: {within_count}/{len(abs_errors)} ({within_pct:.1f}%)")
            
            thresholds_pct = [0.1, 0.5, 1.0, 5.0]  # 百分比误差阈值 (%)
            for threshold in thresholds_pct:
                within_count = np.sum(range_pct_errors <= threshold)
                within_pct = (within_count / len(range_pct_errors)) * 100
                print(f"      Within {threshold:.1f}% of range: {within_count}/{len(range_pct_errors)} ({within_pct:.1f}%)")
        
        return {
            'overall': {'rmse': rmse, 'mae': mae, 'mse': mse},
            'joints': joint_results,
            'y_true': y_test,
            'y_pred': y_pred
        }
    
    def create_all_visualizations(self, y_true, y_pred, results):
        """创建所有可视化图表"""
        print(f"\nGenerating visualization plots...")
        
        # 1. 预测vs真实值散点图
        self.plot_prediction_scatter(y_true, y_pred, results)
        
        # 2. 时间序列对比图
        self.plot_time_series(y_true, y_pred, results)
        
        # 3. 误差分析图
        self.plot_error_analysis(y_true, y_pred, results)
        
        # 4. 误差分布直方图
        self.plot_error_distribution(y_true, y_pred, results)
        
        # 5. 模型性能总结图
        self.plot_performance_summary(results)
        
        # 6. 相关性热力图
        self.plot_correlation_heatmap(y_true, y_pred)
        
        # 7. 百分比误差详细分析
        self.plot_percentage_error_analysis(y_true, y_pred, results)
        
        print(f"All plots saved to: {self.output_dir}")
    
    def plot_prediction_scatter(self, y_true, y_pred, results):
        """预测vs真实值散点图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Predicted vs True Values Comparison', fontsize=16, fontweight='bold')
        
        for i in range(3):
            ax = axes[i]
            
            # 散点图
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=1, color=self.joint_colors[i])
            
            # 完美预测线
            min_val = min(np.min(y_true[:, i]), np.min(y_pred[:, i]))
            max_val = max(np.max(y_true[:, i]), np.max(y_pred[:, i]))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect Prediction')
            
            # 设置标签和标题
            ax.set_xlabel('True Torque (Nm)')
            ax.set_ylabel('Predicted Torque (Nm)')
            ax.set_title(f'{self.joint_names[i]}\nRMSE: {results["joints"][i]["rmse"]:.4f} Nm\nR²: {results["joints"][i]["r_squared"]:.4f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置相等的坐标轴范围
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '1_prediction_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_series(self, y_true, y_pred, results):
        """时间序列对比图"""
        # 选择前2000个点进行显示
        n_samples = min(2000, len(y_true))
        indices = np.arange(n_samples)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Time Series Prediction Comparison (First {n_samples} samples)', fontsize=16, fontweight='bold')
        
        for i in range(3):
            ax = axes[i]
            
            # 绘制真实值和预测值
            ax.plot(indices, y_true[:n_samples, i], label='True Values', color='blue', alpha=0.7, linewidth=1)
            ax.plot(indices, y_pred[:n_samples, i], label='Predicted Values', color='red', alpha=0.7, linewidth=1)
            
            # 填充误差带
            errors = np.abs(y_pred[:n_samples, i] - y_true[:n_samples, i])
            ax.fill_between(indices, 
                           y_true[:n_samples, i] - results['joints'][i]['mae'],
                           y_true[:n_samples, i] + results['joints'][i]['mae'],
                           alpha=0.2, color='gray', label=f'±MAE Band')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Torque (Nm)')
            ax.set_title(f'{self.joint_names[i]} - Time Series Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '2_time_series.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self, y_true, y_pred, results):
        """误差分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detailed Error Analysis', fontsize=16, fontweight='bold')
        
        for i in range(3):
            errors = y_pred[:, i] - y_true[:, i]
            abs_errors = np.abs(errors)
            
            # 上排：绝对误差随时间变化
            ax1 = axes[0, i]
            window_size = 100
            if len(abs_errors) > window_size:
                moving_mae = []
                for j in range(len(abs_errors) - window_size + 1):
                    moving_mae.append(np.mean(abs_errors[j:j+window_size]))
                
                ax1.plot(moving_mae, color=self.joint_colors[i], alpha=0.8)
                ax1.axhline(results['joints'][i]['mae'], color='red', linestyle='--', 
                           label=f'Average MAE: {results["joints"][i]["mae"]:.4f}')
                ax1.set_title(f'{self.joint_names[i]} - Moving Average Error (Window: {window_size})')
                ax1.set_ylabel('Absolute Error (Nm)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 下排：误差vs真实值
            ax2 = axes[1, i]
            ax2.scatter(y_true[:, i], errors, alpha=0.6, s=1, color=self.joint_colors[i])
            ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
            ax2.set_xlabel('True Torque (Nm)')
            ax2.set_ylabel('Prediction Error (Nm)')
            ax2.set_title(f'{self.joint_names[i]} - Error Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '3_error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_distribution(self, y_true, y_pred, results):
        """误差分布直方图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Error Distribution Statistics', fontsize=16, fontweight='bold')
        
        for i in range(3):
            errors = y_pred[:, i] - y_true[:, i]
            abs_errors = np.abs(errors)
            
            # 上排：绝对误差分布
            ax1 = axes[0, i]
            ax1.hist(abs_errors, bins=50, alpha=0.7, color=self.joint_colors[i], edgecolor='black')
            ax1.axvline(results['joints'][i]['mae'], color='red', linestyle='--', linewidth=2,
                       label=f'MAE: {results["joints"][i]["mae"]:.4f}')
            ax1.set_xlabel('Absolute Error (Nm)')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{self.joint_names[i]} - Absolute Error Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 下排：带正负的误差分布
            ax2 = axes[1, i]
            ax2.hist(errors, bins=50, alpha=0.7, color=self.joint_colors[i], edgecolor='black')
            ax2.axvline(0, color='red', linestyle='-', linewidth=2, label='Zero Error')
            ax2.axvline(np.mean(errors), color='orange', linestyle='--', linewidth=2,
                       label=f'Mean Bias: {np.mean(errors):.4f}')
            ax2.set_xlabel('Prediction Error (Nm)')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'{self.joint_names[i]} - Error Distribution (with Bias)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '4_error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_summary(self, results):
        """模型性能总结图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
        
        # 准备数据
        joint_names = [result['name'] for result in results['joints']]
        rmse_values = [result['rmse'] for result in results['joints']]
        mae_values = [result['mae'] for result in results['joints']]
        r_squared_values = [result['r_squared'] for result in results['joints']]
        pct_error_values = [result['range_pct_mean'] for result in results['joints']]  # 修复字段名
        
        # 1. RMSE对比
        ax1 = axes[0, 0]
        bars1 = ax1.bar(joint_names, rmse_values, color=self.joint_colors)
        ax1.set_title('RMSE Comparison by Joint')
        ax1.set_ylabel('RMSE (Nm)')
        for i, v in enumerate(rmse_values):
            ax1.text(i, v + max(rmse_values)*0.01, f'{v:.4f}', ha='center', va='bottom')
        ax1.grid(True, alpha=0.3)
        
        # 2. MAE对比
        ax2 = axes[0, 1]
        bars2 = ax2.bar(joint_names, mae_values, color=self.joint_colors)
        ax2.set_title('MAE Comparison by Joint')
        ax2.set_ylabel('MAE (Nm)')
        for i, v in enumerate(mae_values):
            ax2.text(i, v + max(mae_values)*0.01, f'{v:.4f}', ha='center', va='bottom')
        ax2.grid(True, alpha=0.3)
        
        # 3. R²对比
        ax3 = axes[1, 0]
        bars3 = ax3.bar(joint_names, r_squared_values, color=self.joint_colors)
        ax3.set_title('R² Comparison by Joint')
        ax3.set_ylabel('R² (Coefficient of Determination)')
        ax3.set_ylim(0, 1)
        for i, v in enumerate(r_squared_values):
            ax3.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
        ax3.grid(True, alpha=0.3)
        
        # 4. 百分比误差对比
        ax4 = axes[1, 1]
        bars4 = ax4.bar(joint_names, pct_error_values, color=self.joint_colors)
        ax4.set_title('Average Percentage Error by Joint')
        ax4.set_ylabel('Percentage Error (%)')
        for i, v in enumerate(pct_error_values):
            ax4.text(i, v + max(pct_error_values)*0.01, f'{v:.2f}%', ha='center', va='bottom')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '5_performance_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self, y_true, y_pred):
        """相关性热力图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('True vs Predicted Values Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 计算相关性矩阵
        true_corr = np.corrcoef(y_true.T)
        pred_corr = np.corrcoef(y_pred.T)
        
        # 真实值相关性
        ax1 = axes[0]
        sns.heatmap(true_corr, annot=True, cmap='coolwarm', center=0,
                    xticklabels=self.joint_names, yticklabels=self.joint_names,
                    ax=ax1, cbar_kws={'label': 'Correlation Coefficient'})
        ax1.set_title('Inter-Joint Correlation (True Torques)')
        
        # 预测值相关性
        ax2 = axes[1]
        sns.heatmap(pred_corr, annot=True, cmap='coolwarm', center=0,
                    xticklabels=self.joint_names, yticklabels=self.joint_names,
                    ax=ax2, cbar_kws={'label': 'Correlation Coefficient'})
        ax2.set_title('Inter-Joint Correlation (Predicted Torques)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '6_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_percentage_error_analysis(self, y_true, y_pred, results):
        """详细的百分比误差分析图"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Detailed Percentage Error Analysis', fontsize=16, fontweight='bold')
        
        for i in range(3):
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]
            abs_errors = np.abs(true_vals - pred_vals)
            
            # 计算不同类型的百分比误差
            data_range = np.max(true_vals) - np.min(true_vals)
            data_mean = np.abs(np.mean(true_vals))
            
            range_pct_errors = (abs_errors / data_range) * 100
            epsilon = 1e-6
            true_vals_safe = np.where(np.abs(true_vals) < epsilon, epsilon, np.abs(true_vals))
            relative_pct_errors = (abs_errors / true_vals_safe) * 100
            mean_pct_errors = (abs_errors / (data_mean if data_mean > epsilon else epsilon)) * 100
            
            # 第一行：相对于数据范围的百分比误差
            ax1 = axes[0, i]
            ax1.hist(range_pct_errors, bins=50, alpha=0.7, color=self.joint_colors[i], edgecolor='black')
            ax1.axvline(np.mean(range_pct_errors), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(range_pct_errors):.4f}%')
            ax1.axvline(np.median(range_pct_errors), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(range_pct_errors):.4f}%')
            ax1.set_xlabel('Percentage Error (% of data range)')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{self.joint_names[i]} - Error % of Range\nRange: {data_range:.4f} Nm')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 第二行：相对于真实值的百分比误差 (对数尺度)
            ax2 = axes[1, i]
            # 限制显示范围，避免极值影响可视化
            filtered_errors = relative_pct_errors[relative_pct_errors <= 1000]  # 过滤掉极大值
            
            ax2.hist(filtered_errors, bins=50, alpha=0.7, color=self.joint_colors[i], edgecolor='black')
            ax2.axvline(np.mean(filtered_errors), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(filtered_errors):.4f}%')
            ax2.axvline(np.median(filtered_errors), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(filtered_errors):.4f}%')
            ax2.set_xlabel('Percentage Error (% of true value)')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'{self.joint_names[i]} - Error % of True Value\n(Filtered: <1000%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 第三行：累积误差分布 (CDF)
            ax3 = axes[2, i]
            sorted_range_errors = np.sort(range_pct_errors)
            cumulative_prob = np.arange(1, len(sorted_range_errors) + 1) / len(sorted_range_errors)
            
            ax3.plot(sorted_range_errors, cumulative_prob * 100, color=self.joint_colors[i], linewidth=2)
            ax3.axvline(np.percentile(range_pct_errors, 50), color='orange', linestyle='--', 
                       label=f'50th percentile: {np.percentile(range_pct_errors, 50):.4f}%')
            ax3.axvline(np.percentile(range_pct_errors, 90), color='red', linestyle='--',
                       label=f'90th percentile: {np.percentile(range_pct_errors, 90):.4f}%')
            ax3.axvline(np.percentile(range_pct_errors, 95), color='darkred', linestyle='--',
                       label=f'95th percentile: {np.percentile(range_pct_errors, 95):.4f}%')
            
            ax3.set_xlabel('Percentage Error (% of range)')
            ax3.set_ylabel('Cumulative Probability (%)')
            ax3.set_title(f'{self.joint_names[i]} - Cumulative Error Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '7_percentage_error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建误差阈值分析表格图
        self.plot_error_threshold_analysis(y_true, y_pred)
    
    def plot_error_threshold_analysis(self, y_true, y_pred):
        """误差阈值分析表格"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle('Error Threshold Analysis - Percentage of Samples Within Thresholds', 
                     fontsize=16, fontweight='bold')
        
        # 定义阈值
        abs_thresholds = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]  # Nm
        pct_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]  # %
        
        # 创建表格数据
        table_data = []
        
        # 添加表头
        headers = ['Threshold Type', 'Threshold Value'] + [f'{joint}' for joint in self.joint_names]
        
        # 绝对误差阈值
        for threshold in abs_thresholds:
            row = ['Absolute Error', f'{threshold:.4f} Nm']
            for i in range(3):
                abs_errors = np.abs(y_true[:, i] - y_pred[:, i])
                within_count = np.sum(abs_errors <= threshold)
                within_pct = (within_count / len(abs_errors)) * 100
                row.append(f'{within_pct:.1f}%')
            table_data.append(row)
        
        # 添加分隔行
        table_data.append(['', ''] + [''] * 3)
        
        # 百分比误差阈值（相对于数据范围）
        for threshold in pct_thresholds:
            row = ['Range Percentage', f'{threshold:.2f}%']
            for i in range(3):
                abs_errors = np.abs(y_true[:, i] - y_pred[:, i])
                data_range = np.max(y_true[:, i]) - np.min(y_true[:, i])
                pct_errors = (abs_errors / data_range) * 100
                within_count = np.sum(pct_errors <= threshold)
                within_pct = (within_count / len(pct_errors)) * 100
                row.append(f'{within_pct:.1f}%')
            table_data.append(row)
        
        # 创建表格
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        # 美化表格
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置分隔行样式
        sep_row_idx = len(abs_thresholds) + 1
        for i in range(len(headers)):
            table[(sep_row_idx, i)].set_facecolor('#E0E0E0')
        
        # 隐藏坐标轴
        ax.set_axis_off()
        
        # 添加说明文本
        explanation = (
            "This table shows the percentage of predictions that fall within various error thresholds.\n"
            "• Absolute Error: Raw error in Nm\n"
            "• Range Percentage: Error as percentage of the data range for each joint\n"
            "• Higher percentages indicate better model performance"
        )
        
        plt.figtext(0.1, 0.02, explanation, fontsize=10, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '8_error_threshold_table.png'), dpi=300, bbox_inches='tight')

def load_dataset(dataset_path):
    """加载数据集"""
    print("Loading dataset: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset loaded successfully!")
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Enhanced LSTM Model Testing with Visualization')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    print("Enhanced LSTM Model Testing with Visualization")
    print("=" * 50)
    
    try:
        # 加载数据集
        dataset = load_dataset(args.dataset_path)
        
        # 创建增强测试器并测试
        tester = EnhancedTester(args.model_path, args.output_dir)
        results = tester.test(dataset)
        
        print("\n" + "=" * 50)
        print("Testing and visualization completed!")
        print(f"All results saved to: {args.output_dir}")
        print("Generated plots:")
        print("  1_prediction_scatter.png - Predicted vs True values comparison")
        print("  2_time_series.png - Time series prediction performance")
        print("  3_error_analysis.png - Detailed error analysis")
        print("  4_error_distribution.png - Error distribution statistics")
        print("  5_performance_summary.png - Model performance summary")
        print("  6_correlation_heatmap.png - Inter-joint correlations")
        print("  7_percentage_error_analysis.png - Detailed percentage error analysis")
        print("  8_error_threshold_table.png - Error threshold analysis table")
        print("=" * 50)
        
    except Exception as e:
        print(f"Testing error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# ========== 简单的LSTM模型（与训练脚本相同）==========
class SimpleLSTM(nn.Module):
    """简单的LSTM扭矩预测模型"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=3):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])
        return output

# ========== 简单测试器 ==========
class SimpleTester:
    """简化的测试器"""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载模型"""
        print(f"加载模型: {model_path}")
        
        # 修复PyTorch 2.6+的安全加载问题
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 重建模型
        input_dim = checkpoint['input_dim']
        self.model = SimpleLSTM(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载标准化器
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        
        print(f"模型加载成功! 输入维度: {input_dim}")
    
    def test(self, dataset):
        """测试模型"""
        print("\n开始测试...")
        
        # 准备测试数据
        X_test = torch.FloatTensor(dataset['X_test']).to(self.device)
        y_test_scaled = dataset['y_test']
        
        print(f"测试数据: {X_test.shape}")
        
        # 预测
        with torch.no_grad():
            y_pred_scaled = self.model(X_test).cpu().numpy()
        
        # 反标准化
        y_test = self.scaler_y.inverse_transform(y_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"\n测试结果:")
        print(f"  RMSE: {rmse:.6f} Nm")
        print(f"  MAE:  {mae:.6f} Nm")
        print(f"  MSE:  {mse:.6f}")
        
        # 每个关节的结果
        joint_names = ['关节1', '关节2', '关节3']
        print(f"\n各关节详细结果:")
        for i in range(3):
            joint_mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            joint_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            joint_rmse = np.sqrt(joint_mse)
            
            print(f"  {joint_names[i]}: RMSE={joint_rmse:.6f} Nm, MAE={joint_mae:.6f} Nm")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'y_true': y_test,
            'y_pred': y_pred
        }

def load_dataset(dataset_path):
    """加载数据集"""
    print(f"加载数据集: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"数据集加载成功!")
    return dataset

def main():
    parser = argparse.ArgumentParser(description='简单LSTM模型测试')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='测试数据集路径')
    
    args = parser.parse_args()
    
    print("简单LSTM模型测试")
    print("=" * 30)
    
    try:
        # 加载数据集
        dataset = load_dataset(args.dataset_path)
        
        # 创建测试器并测试
        tester = SimpleTester(args.model_path)
        results = tester.test(dataset)
        
        print("\n" + "=" * 30)
        print("测试完成!")
        print("=" * 30)
        
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


'''
python3 tester.py \
    --model_path './simple_models/simple_lstm_model_20250720_202349.pth' \
    --dataset_path './data/features/dataset.pkl' \
    --output_dir './test_results'
'''