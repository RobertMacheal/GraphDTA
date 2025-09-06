import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GNNTransformerFusionNet(nn.Module):
    def __init__(self, gnn_backbone, esm_dim=1280, fusion_dim=256, heads=4, layers=2):
        super(GNNTransformerFusionNet, self).__init__()
        self.gnn = gnn_backbone  # ✅ 注意：不加括号！

        assert hasattr(self.gnn, 'output_dim'), "GNN backbone must define self.output_dim"

        self.proj_drug = nn.Linear(self.gnn.output_dim, fusion_dim)
        self.proj_protein = nn.Linear(esm_dim, fusion_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=fusion_dim, nhead=heads, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=layers)

        self.output = nn.Linear(fusion_dim, 1)

    def forward(self, data):
        # Drug graph embedding
        drug_embed = self.gnn.get_graph_embedding(data)       # [B, gnn_dim]
        drug_proj = self.proj_drug(drug_embed).unsqueeze(1)   # [B, 1, fusion_dim]

        # Protein embedding (from ESM)
        prot_input = data.xt.view(-1, 1280)                    # [B, 1280]
        prot_proj = self.proj_protein(prot_input).unsqueeze(1)  # [B, 1, fusion_dim]

        # Combine and apply Transformer
        fusion_input = torch.cat([drug_proj, prot_proj], dim=1)   # [B, 2, fusion_dim]
        fusion_output = self.transformer(fusion_input)            # [B, 2, fusion_dim]

        fused = fusion_output.mean(dim=1)                         # [B, fusion_dim]
        return self.output(fused)                                 # [B, 1]
