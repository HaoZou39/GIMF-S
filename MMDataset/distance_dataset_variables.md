# distance_dataset.npz 字段说明

本文档说明 `build_distance_dataset_v3.py` 生成的 NumPy 压缩文件 `distance_dataset.npz` 中保存的变量（keys）及其含义。

目前有向图距离语义并不正确，请使用无向图距离。具体请看示例

## 读取示例（Python）

```python
import numpy as np

data = np.load('./data/Gen_dataset/distance_dataset.npz', allow_pickle=True)
print(data.files)  # 查看 keys

# 请使用以下匹配后节点坐标和路网距离矩阵
node = data['matched_node_norm'] # 匹配且归一化后节点坐标
distance = data['undirected_dist_norm'] # 进行相应缩放后的无向图距离
```

