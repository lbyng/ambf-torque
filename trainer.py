import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import argparse

# ========== 简单的LSTM模型 ==========
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

# ========== 简单训练器 ==========
class SimpleTrainer:
    """简化的训练器"""
    
    def __init__(self, learning_rate=0.001, batch_size=64, num_epochs=50):
        torch.manual_seed(2025)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(2025)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
    
    def train(self, dataset, output_dir):
        """训练模型"""
        print("\n开始训练...")
        
        # 准备数据
        X_train = torch.FloatTensor(dataset['X_train'])
        y_train = torch.FloatTensor(dataset['y_train'])
        X_val = torch.FloatTensor(dataset['X_val'])
        y_val = torch.FloatTensor(dataset['y_val'])
        
        print(f"训练数据: {X_train.shape}")
        print(f"验证数据: {X_val.shape}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        generator = torch.Generator()
        generator.manual_seed(2025)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 创建模型
        input_dim = X_train.shape[-1]
        self.model = SimpleLSTM(input_dim=input_dim).to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            # 验证
            val_loss = self.validate_epoch(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(dataset, output_dir, epoch)
                print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.6f})")
            
            print()
        
        print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")
        return self.model
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 检查损失是否正常
            if torch.isnan(loss):
                print(f"警告: 第{batch_idx}批次出现NaN损失，跳过")
                continue
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            # 打印进度
            if batch_idx % 1000 == 0:
                print(f"    批次 {batch_idx}/{len(train_loader)}, 损失: {loss.item():.6f}")
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_model(self, dataset, output_dir, epoch):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f'simple_lstm_model_{timestamp}.pth')
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': dataset['X_train'].shape[-1],
            'sequence_length': dataset['X_train'].shape[1],
            'scaler_X': dataset['scaler_X'],
            'scaler_y': dataset['scaler_y'],
            'feature_names': dataset['feature_names'],
            'epoch': epoch,
            'config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            }
        }
        
        torch.save(save_dict, model_path)
        print(f"    模型已保存: {model_path}")

def load_dataset(dataset_path):
    """加载数据集"""
    print(f"加载数据集: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"数据集加载成功!")
    print(f"  训练样本: {dataset['X_train'].shape}")
    print(f"  验证样本: {dataset['X_val'].shape}")
    print(f"  特征数量: {len(dataset['feature_names'])}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description='简单LSTM扭矩模型训练')
    
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='预处理数据集路径')
    parser.add_argument('--output_dir', type=str, default='./simple_models',
                        help='模型输出目录')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    
    args = parser.parse_args()
    
    print("简单LSTM扭矩模型训练")
    print("=" * 40)
    print(f"数据集: {args.dataset_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"学习率: {args.learning_rate}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    
    try:
        # 加载数据
        dataset = load_dataset(args.dataset_path)
        
        # 创建训练器
        trainer = SimpleTrainer(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs
        )
        
        # 训练模型
        model = trainer.train(dataset, args.output_dir)
        
        print("\n" + "=" * 40)
        print("训练完成!")
        print("=" * 40)
        
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

'''
python3 trainer.py --dataset_path './data/features/dataset.pkl'
'''