import os
import torch
import pandas as pd


# 1. 创建数据集并读取
root_path = r"E:\LearnSpace\DiveIntoDeepLearningImplementation\d2l-en\MyImplementation\Preliminaries"
os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
data_file = os.path.join(root_path, "data", "house_tiny.csv")

with open(data_file, "w") as f:
    f.write(
        """
NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000
        """
    )

data = pd.read_csv(data_file)
print(data)

# 2. 数据预处理
print("-" * 100)
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(value=pd.NA)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

inputs = inputs.fillna(inputs.mean())
print(f"inputs.mean() = {inputs.mean()}")
print(inputs)

# 3. 转换为张量
print("-" * 100)
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print(f"X = {X}, \ny = {y}")

## Exercise dataframe.loc()
print("-" * 100)
data = pd.read_csv(data_file)
inputs, targets = data.loc[:, ["NumRooms", "RoofType"]], data.loc[:, "Price"]
print(inputs)
print(targets)
