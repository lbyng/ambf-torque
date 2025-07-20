import pickle
import numpy as np
import os

# 加载数据集
with open('./data/features/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# 反标准化测试数据
y_test_scaled = dataset['y_test']
y_test = dataset['scaler_y'].inverse_transform(y_test_scaled)

# 找到异常值的位置
joint1_data = y_test[:, 0]
max_idx = np.argmax(joint1_data)
max_value = joint1_data[max_idx]

print(f"异常值信息:")
print(f"  值: {max_value:.6f} Nm")
print(f"  在测试集中的索引: {max_idx}")

# 获取测试文件列表
test_files = dataset['metadata']['test_files']
print(f"\n测试集包含的文件:")
for i, file in enumerate(test_files):
    print(f"  {i}: {file}")

# 现在需要反向追踪这个样本来自哪个文件
# 由于每个文件创建的序列数量不同，需要重新计算

print(f"\n开始定位异常值来源文件...")

# 重新处理测试文件，计算每个文件的序列数量
data_dir = input("请输入原始数据目录路径: ").strip()
sequence_length = dataset['metadata']['sequence_length']

cumulative_sequences = 0
found_file = None

for file_idx, file_name in enumerate(test_files):
    file_path = os.path.join(data_dir, file_name)
    
    try:
        # 加载文件
        data = np.loadtxt(file_path, delimiter=',')
        
        # 计算这个文件能产生多少个序列
        if len(data) >= sequence_length:
            n_sequences = len(data) - sequence_length + 1
        else:
            n_sequences = 0
        
        print(f"文件 {file_name}: {len(data)} 行数据 → {n_sequences} 个序列")
        
        # 检查异常值是否在这个文件的范围内
        if cumulative_sequences <= max_idx < cumulative_sequences + n_sequences:
            found_file = file_name
            sequence_in_file = max_idx - cumulative_sequences
            
            print(f"\n🎯 找到了！异常值来自:")
            print(f"  文件: {file_name}")
            print(f"  文件中的序列索引: {sequence_in_file}")
            print(f"  对应原始数据行: {sequence_in_file + sequence_length - 1}")
            
            # 验证：检查这个文件的扭矩数据
            tau_data = data[:, 114:117]  # tau列
            print(f"\n文件 {file_name} 的扭矩统计:")
            print(f"  关节1范围: [{tau_data[:, 0].min():.6f}, {tau_data[:, 0].max():.6f}]")
            print(f"  关节2范围: [{tau_data[:, 1].min():.6f}, {tau_data[:, 1].max():.6f}]")
            print(f"  关节3范围: [{tau_data[:, 2].min():.6f}, {tau_data[:, 2].max():.6f}]")
            
            # 找到具体的异常行
            max_row_in_file = np.argmax(tau_data[:, 0])
            print(f"  异常值具体位置: 第 {max_row_in_file} 行")
            print(f"  异常值: {tau_data[max_row_in_file, 0]:.6f}")
            
            break
        
        cumulative_sequences += n_sequences
        
    except Exception as e:
        print(f"读取文件 {file_name} 时出错: {e}")

if found_file:
    print(f"\n🗑️  要删除的文件: {os.path.join(data_dir, found_file)}")
    print(f"\n删除命令:")
    print(f"rm '{os.path.join(data_dir, found_file)}'")
    
    # 询问是否立即删除
    choice = input(f"\n是否立即删除文件 {found_file}? (y/N): ").strip().lower()
    if choice == 'y':
        try:
            os.remove(os.path.join(data_dir, found_file))
            print(f"✅ 文件 {found_file} 已删除")
            print(f"请重新运行数据预处理来生成新的数据集")
        except Exception as e:
            print(f"❌ 删除失败: {e}")
    else:
        print("文件未删除")
else:
    print("❌ 未找到异常值来源文件")