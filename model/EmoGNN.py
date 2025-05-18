import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx
import torch


device = torch.device("cuda:0")
# 读取 senticnet.xlsx 文件
data = pd.read_excel("D:/image2text_conversion - 副本/image2text_conversion/model/senticnet.xlsx")

# 初始化有向图
G = nx.DiGraph()

# 定义情感极性和情感标签映射规则
polarity_map = {
    "negative": -1,
    "positive": 1
}
# emotion_map = {"#sadness": 0, "#grief": 1, "#loathing": 2}  # 示例情感映射

# 构建节点及其特征
for _, row in data.iterrows():
    concept = row["CONCEPT"]
    features = {
        "introspection": float(row["INTROSPECTION"] or 0),
        "temper": float(row["TEMPER"] or 0),
        "attitude": float(row["ATTITUDE"] or 0),
        "sensitivity": float(row["SENSITIVITY"] or 0),
        "polarity_value": float(polarity_map.get(row["POLARITY VALUE"], 0)),
        "polarity_intensity": float(row["POLARITY INTENSITY"] or 0),
        # "primary_emotion": emotion_map.get(row["PRIMARY EMOTION"], -1),
        "semantics": row["SEMANTICS"]
    }
    G.add_node(concept, **features)

# 遍历数据行，添加语义关系边
for _, row in data.iterrows():
    concept = row["CONCEPT"]
    if pd.notna(row["SEMANTICS"]):  # 如果语义列不为空
        related_concepts = row["SEMANTICS"].split("\t")  # 分割语义词
        for related_concept in related_concepts:
            if related_concept in G:
                G.add_edge(concept, related_concept, relation="semantic")

# 给节点分配整数索引
node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}

# 构建边索引
gnn_data = from_networkx(G)
gnn_data.edge_index = torch.tensor([
    [node_to_idx[edge[0]], node_to_idx[edge[1]]]  # 使用整数索引代替字符串
    for edge in G.edges
], dtype=torch.long).t().contiguous()

gnn_data.edge_index = gnn_data.edge_index.to(device)

# 构建节点特征张量
gnn_data.x = torch.tensor([
    [
        G.nodes[node].get("introspection", 0),
        G.nodes[node].get("temper", 0),
        G.nodes[node].get("attitude", 0),
        G.nodes[node].get("sensitivity", 0),
        G.nodes[node].get("polarity_value", 0),
        G.nodes[node].get("polarity_intensity", 0)
    ]
    for node in G.nodes
], dtype=torch.float)

gnn_data.x = gnn_data.x.to(device)

# # 构建节点标签张量（情感分类任务）
# gnn_data.y = torch.tensor([
#     G.nodes[node].get("primary_emotion", -1)  # 默认为 -1 表示无标签
#     for node in G.nodes
# ], dtype=torch.long)

# gnn_data.y = gnn_data.y.to(device)  # 如果训练需要标签