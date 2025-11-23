import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pytorch

from pytorch import test_data

# 1. 重新实例化一个"空脑子"的模型
model_new = pytorch.MyFashionModel()

# 2. 加载保存好的参数 (注入记忆)
# map_location='cpu' 确保即使你在GPU上训练的，在没有GPU的电脑上也能加载
model_new.load_state_dict(torch.load("model_weights.pth", map_location='cpu'))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# 3. 把模型搬到现在的设备上
model_new.to(device)

print("模型加载成功！")

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


model_new.eval() # 【关键】切换到评估模式

# 1. 获取一张测试图片 (x) 和它的真实标签 (y)
x, y = test_data[1][0], test_data[1][1]

# 2. 处理数据维度 (最容易报错的一步！)
# 现在的 x 是 [1, 28, 28]，但模型期待的是 [Batch, Channel, Height, Width]
# 我们需要给它加一个"外包装"，变成 [1, 1, 28, 28]
x = x.unsqueeze(0) 

# 3. 搬运到设备
x = x.to(device)

# 4. 开始预测 (不需要算梯度，节省资源)
with torch.no_grad():
    pred = model_new(x)
    
    # pred 输出的是10个概率值，我们需要找到最大的那个的索引
    predicted_index = pred.argmax(1).item()
    
    predicted_label = classes[predicted_index]
    actual_label = classes[y]

print(f"模型预测是: {predicted_label}")
print(f"真实结果是: {actual_label}")