import torch
import torch.nn as nn


class GNNCrossAttentionFusionNet(nn.Module):
    def __init__(self, gnn_backbone, esm_dim=1280, fusion_dim=256, heads=4):
        super(GNNCrossAttentionFusionNet, self).__init__()
        self.gnn = gnn_backbone  # ✅ 注意：这里传的是实例化后的对象

        assert hasattr(self.gnn, 'output_dim'), "GNN backbone must define self.output_dim"

        self.prot_proj = nn.Linear(esm_dim, fusion_dim)
        self.drug_proj = nn.Linear(self.gnn.output_dim, fusion_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=heads, batch_first=True)

        self.output = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        # Get graph embedding from GNN
        drug_embed = self.gnn.get_graph_embedding(data)       # [B, gnn_dim]
        drug_proj = self.drug_proj(drug_embed).unsqueeze(1)   # [B, 1, fusion_dim]

        prot_input = data.xt.view(-1, 1280)                   # [B, 1280]
        prot_proj = self.prot_proj(prot_input).unsqueeze(1)   # [B, 1, fusion_dim]

        # Protein queries drug
        attn_out, _ = self.cross_attn(query=prot_proj, key=drug_proj, value=drug_proj)  # [B, 1, fusion_dim]
        fusion = attn_out.squeeze(1)  # [B, fusion_dim]

        return self.output(fusion)  # [B, 1]
