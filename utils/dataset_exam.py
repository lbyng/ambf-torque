import pickle

with open('../data/features/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"总特征数: {len(dataset['feature_names'])}")
print("\n特征名称:")
for i, name in enumerate(dataset['feature_names']):
    print(f"  {i:2d}: {name}")
    
# 检查是否还有ctrl_cmd
ctrl_cmd_features = [name for name in dataset['feature_names'] if 'ctrl_cmd' in name]
print(f"\nctrl_cmd特征: {ctrl_cmd_features}")

pos_error_features = [name for name in dataset['feature_names'] if 'pos_error' in name]
print(f"pos_error特征: {pos_error_features}")