
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, BertModel
from model.attention import DotProductAttention



# 在模块引入前，
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_geometric.nn import GCNConv



class EmotionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmotionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        device = next(self.parameters()).device  # 获取模型当前的设备（CPU 或 GPU）
        data.x = data.x.to(device)  # 将节点特征移动到模型设备
        data.edge_index = data.edge_index.to(device)  # 将边索引移动到模型设备

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()  # 图卷积 + 激活
        x = self.conv2(x, edge_index)        # 图卷积
        self.project_to_768 = nn.Linear(128, 768).to(device)
        x = self.project_to_768(x).to(device)
        return x


device = torch.device("cuda:0")
# Construct the joint Caption & Sentiment Classifier Model
class CustomModel(nn.Module):
    def __init__(self, cfg, catr, gnn_data):
        super().__init__()
        self.cfg = cfg
        self.catr = catr
        self.bert = RobertaModel.from_pretrained(self.cfg.model) # bert model
        # self.bert = BertModel.from_pretrained(self.cfg.model) # bert model
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 3) # 3 classes
        self.att = DotProductAttention(self.cfg.dropout)
        self.mul_att = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True, dropout=self.cfg.dropout)
        self.project = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self._init_weights(self.out)
        self._init_weights(self.project)

        # GNN 初始化
        self.gnn_data = gnn_data  # 知识图谱数据
        self.gnn = EmotionGNN(input_dim=6, hidden_dim=32, output_dim=128).to(device)  # 输出维度与 RoBERTa 嵌入一致



    def _init_weights(self, module):
        if isinstance(module, nn.AdaptiveAvgPool1d):
            torch.nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self,samples,caption,c_mask,input_ids,attention_mask,ce_loss,input_ids_tt,attention_mask_tt,input_ids_at,attention_mask_at,is_training=True):
        sample_k = self.cfg.sample_k
        cap_len = self.cfg.max_caption_len
        pad_token_id = self.cfg.pad_token_id
        end_token_id = self.cfg.end_token_id
        bs = caption.shape[0]
        device = caption.device
        # Calculate the origin length of the input text-pair
        origin_len = attention_mask.sum(dim=-1)
        # sep token embedding
        end_token_embedding = self.bert.embeddings.word_embeddings(torch.tensor(end_token_id, device=device))
        # Image-to-Text Conversion Module
        # Generate implicit tokens by catr and top-k sampling
        _, caption_out, _, cap_mask, finished = self.catr(samples,caption,c_mask,cap_len,sample_k,end_token_id,pad_token_id)
        sorted_out_id = torch.argsort(caption_out, dim=-1, descending=True)  # bs*tgt_len*vocab_size
        sample_out_id = sorted_out_id[:, :, :sample_k]
        sample_out = torch.zeros((bs, cap_len, sample_k),dtype=torch.float,device=device)
        for i in range(bs):
            for j in range(cap_len):
                sample_out[i][j] = caption_out[i, j, sample_out_id[i][j]]
        sample_prob = F.softmax(sample_out, dim=-1)
        sample_prob = sample_prob.unsqueeze(3)
        sample_caption_embedding = self.bert.embeddings.word_embeddings(sample_out_id)
        sample_caption_embedding = (sample_prob*sample_caption_embedding).sum(dim=2) # element-wise product
        # Calculate input embeddings
        for i in range(bs):
            if finished[i]:
                cap_mask[i, finished[i]] = True
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        inputs_embeds_tt = self.bert.embeddings.word_embeddings(input_ids_tt)
        inputs_embeds_at = self.bert.embeddings.word_embeddings(input_ids_at)
        target_len = attention_mask_at.sum(dim=-1)
        cap_mask = ~cap_mask  # cap_mask(valid: False; else:True) | attention_mask(valid:1; else:0)
        cap_mask = cap_mask.int()
        caption_len = cap_mask.sum(dim=-1)

        # GNN 输出情感增强特征
        gnn_embeddings = self.gnn(self.gnn_data)  # GNN 的输出
        gnn_context_embeddings = torch.mean(gnn_embeddings, dim=0)  # 平均池化作为情感上下文



        # Aspect-oriented Filtration Module
        att_output_at, _ = self.att(inputs_embeds_at + gnn_context_embeddings, inputs_embeds_tt, inputs_embeds_tt, attention_mask_tt.sum(dim=-1))
        inputs_embeds_at = (inputs_embeds_at + att_output_at) / 2 # average
        att_output_ai, _ = self.att(inputs_embeds_at, sample_caption_embedding + gnn_context_embeddings, sample_caption_embedding, caption_len)
        # concatenation
        for i in range(bs):
            attention_mask[i, origin_len[i]: origin_len[i]+target_len[i]] = 1
            inputs_embeds[i, origin_len[i]: origin_len[i]+target_len[i]] = att_output_ai[i, :target_len[i]]
            inputs_embeds[i, origin_len[i]+target_len[i]] = end_token_embedding
        # Prediction Module
        outputs = self.bert(inputs_embeds=inputs_embeds,attention_mask=attention_mask)
        outputs = outputs.pooler_output
        outputs = self.out(self.dropout(outputs))
        return outputs
