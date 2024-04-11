import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, softmax, attention_mask):
    qkt = torch.matmul(q,k.transpose(-2, -1))
    dk = torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))
    scaled_qkt = qkt / dk
    softmax = softmax(scaled_qkt.masked_fill(attention_mask == 0, -1e9))
    
    z = torch.matmul(softmax, v)
    return z


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = nn.ModuleList([nn.Linear(d_model, self.depth) for _ in range(num_of_heads)])
        self.wk = nn.ModuleList([nn.Linear(d_model, self.depth) for _ in range(num_of_heads)])
        self.wv = nn.ModuleList([nn.Linear(d_model, self.depth) for _ in range(num_of_heads)])
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q, K, V, F.softmax, attention_mask))
            
        multi_head = torch.cat(multi_attn, dim=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention


class Transformer(nn.Module):
    def __init__(self, num_blocks):
        super(Transformer, self).__init__()
        # Set your constant parameters here
        UNITS, MLP_RATIO, MLP_DROPOUT_RATIO =  768, 4, 0.1
        self.mhas = nn.ModuleList([MultiHeadAttention(UNITS, 8) for _ in range(num_blocks)])
        self.mlps = nn.ModuleList([nn.ModuleList([
                    nn.Linear(UNITS, UNITS * MLP_RATIO), 
                    nn.GELU(), 
                    nn.Dropout(p=MLP_DROPOUT_RATIO),
                    nn.Linear(UNITS * MLP_RATIO, UNITS),
                ]) for _ in range(num_blocks)])

    def forward(self, x, attention_mask):
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x, attention_mask)
            x = mlp(x)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, num_blocks, device):
        super().__init__()
        self.num_blocks = num_blocks
        self.ln_1s = [torch.nn.LayerNorm(UNITS) for _ in range(num_blocks)]
        self.mhas = [MultiHeadAttention(UNITS, 8) for _ in range(num_blocks)]
        self.ln_2s = [torch.nn.LayerNorm(UNITS) for _ in range(num_blocks)]
        self.mlps = [torch.nn.Sequential(
            torch.nn.Linear(UNITS * MLP_RATIO, UNITS),
            torch.nn.GELU(),
            torch.nn.Dropout(MLP_DROPOUT_RATIO),
            torch.nn.Linear(UNITS, UNITS)
        ) for _ in range(num_blocks)]

    def forward(self, x, attention_mask):
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x = ln_1(x + mha(x, attention_mask))
            x = ln_2(x + mlp(x))
        return x

class LandmarkEmbedding(torch.nn.Module):
    def __init__(self, units, name):
        super().__init__()
        self.units = units
        self.empty_embedding = torch.nn.Parameter(torch.zeros(units), requires_grad=False)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(units, units),
            torch.nn.GELU(),
            torch.nn.Linear(units, units)
        )

    def forward(self, x):
        return torch.where(
            torch.sum(x, dim=2, keepdim=True) == 0,
            self.empty_embedding,
            self.dense(x)
        )

class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_embedding = torch.nn.Embedding(INPUT_SIZE+1, UNITS)
        self.lips_embedding = LandmarkEmbedding(LIPS_UNITS, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'left_hand')
        self.pose_embedding = LandmarkEmbedding(POSE_UNITS, 'pose')
        self.landmark_weights = torch.nn.Parameter(torch.zeros(3), requires_grad=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(UNITS, UNITS),
            torch.nn.GELU(),
            torch.nn.Linear(UNITS, UNITS)
        )

    def forward(self, lips0, left_hand0, pose0, non_empty_frame_idxs, training=False):
        lips_embedding = self.lips_embedding(lips0)
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        pose_embedding = self.pose_embedding(pose0)
        x = torch.stack((
            lips_embedding, left_hand_embedding, pose_embedding,
        ), dim=3)
        x = x * torch.nn.functional.softmax(self.landmark_weights, dim=0)
        x = torch.sum(x, dim=3)

        x = self.fc(x)

        max_frame_idxs = torch.clamp(
            torch.max(non_empty_frame_idxs, dim=1, keepdim=True)[0],
            1,
            float('inf')
        )

        # TODO: Normalization, blahblahblah

        return x