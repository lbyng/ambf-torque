import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_data(data_train, output_dirs):
    """可视化数据并保存到本地"""
    print("Visualizing data for main 3 joints...")
    print(f"Data shape: {data_train.shape}")
    
    # 时间序列
    if data_train.shape[1] > 12:
        time_line = data_train[:, 12]
    else:
        time_line = np.arange(len(data_train))
    
    # 前3个关节的数据
    jpos_main = data_train[:, 13:16]    # 主要关节位置
    tau_main = data_train[:, 114:117]   # 主要关节扭矩
    jvel_main = data_train[:, 162:165]  # 主要关节速度
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('RAVEN Robot Main 3 Joints Data Visualization', fontsize=16)
    
    joint_names = ['Joint 1 (Shoulder Rotation)', 'Joint 2 (Shoulder Pitch)', 'Joint 3 (Insertion)']
    
    # 绘制关节位置、扭矩、速度
    for i in range(3):
        # 位置
        axes[0, i].plot(time_line, jpos_main[:, i], 'b-', alpha=0.7)
        axes[0, i].set_title(f'{joint_names[i]}\nPosition')
        axes[0, i].set_ylabel('Position')
        axes[0, i].grid(True, alpha=0.3)
        
        # 扭矩
        axes[1, i].plot(time_line, tau_main[:, i], 'r-', alpha=0.7)
        axes[1, i].set_title(f'Torque (Nm)')
        axes[1, i].set_ylabel('Torque (Nm)')
        axes[1, i].grid(True, alpha=0.3)
        
        # 速度
        axes[2, i].plot(time_line, jvel_main[:, i], 'g-', alpha=0.7)
        axes[2, i].set_title(f'Velocity')
        axes[2, i].set_xlabel('Time')
        axes[2, i].set_ylabel('Velocity')
        axes[2, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(output_dirs['data_viz'], 'main_joints_overview.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像释放内存
    print(f"Data visualization saved to: {save_path}")
    
    

def visualize_control_commands(data_train, output_dirs):
    jpos = data_train[:, 13:16]
    jpos_d = data_train[:, 194:197]
    control_command = jpos_d - jpos
    tau = data_train[:, 114:117]
    
    # 时间序列
    if data_train.shape[1] > 12:
        time_line = data_train[:, 12]
    else:
        time_line = np.arange(len(data_train))
    
    # 创建图表
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Control Commands Analysis', fontsize=16)
    
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3']
    
    for i in range(3):
        # 左列：控制命令和扭矩的关系
        ax_left = axes[i, 0]
        ax_left.scatter(control_command[:, i], tau[:, i], alpha=0.5, s=1)
        ax_left.set_xlabel('Control Command (rad)')
        ax_left.set_ylabel('Torque (Nm)')
        ax_left.set_title(f'{joint_names[i]} - Control Command vs Torque')
        ax_left.grid(True, alpha=0.3)
        
        # 添加相关系数
        correlation = np.corrcoef(control_command[:, i], tau[:, i])[0, 1]
        ax_left.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax_left.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 右列：时间序列对比
        ax_right = axes[i, 1]
        
        # 使用双y轴
        ax_right2 = ax_right.twinx()
        
        # 绘制控制命令
        line1 = ax_right.plot(time_line, control_command[:, i], 'b-', 
                             alpha=0.7, label='Control Command')
        ax_right.set_ylabel('Control Command (rad)', color='b')
        ax_right.tick_params(axis='y', labelcolor='b')
        
        # 绘制扭矩
        line2 = ax_right2.plot(time_line, tau[:, i], 'r-', 
                              alpha=0.7, label='Torque')
        ax_right2.set_ylabel('Torque (Nm)', color='r')
        ax_right2.tick_params(axis='y', labelcolor='r')
        
        ax_right.set_xlabel('Time')
        ax_right.set_title(f'{joint_names[i]} - Time Series')
        ax_right.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_right.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    save_path = os.path.join(output_dirs['data_viz'], 'control_commands_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Control commands visualization saved to: {save_path}")
    
    # 打印控制命令的统计信息
    print("\nControl Commands Statistics:")
    for i in range(3):
        cmd = control_command[:, i]
        print(f"  Joint {i+1}:")
        print(f"    Range: [{cmd.min():.4f}, {cmd.max():.4f}] rad")
        print(f"    Mean: {cmd.mean():.4f} rad")
        print(f"    Std: {cmd.std():.4f} rad")
        print(f"    Correlation with torque: {np.corrcoef(cmd, tau[:, i])[0, 1]:.3f}")
        
        
def plot_training_history(history, run_num, output_dirs):
    """绘制并保存训练历史"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title(f'Model Loss - Run {run_num}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train MAE', color='blue')
    plt.plot(history['val_mae'], label='Validation MAE', color='red')
    plt.title(f'Model MAE - Run {run_num}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(output_dirs['training'], f'training_history_run_{run_num}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")
    