import sys
import os
from pathlib import Path

# Resolve project root dynamically (this file is at reve_pipeline/runner/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from reve_pipeline.common.reve_embed import load_reve
from reve_pipeline.runner.end2end_phase2_c2cattn_loso import End2EndPhase2Model
import json

def count_parameters():
    # Load frozen REVE model
    reve, _ = load_reve("brain-bzh/reve-base", "brain-bzh/reve-positions", "cpu")
    frozen_params = sum(p.numel() for p in reve.parameters())

    # Get details from run_meta.json
    n_ch = 19
    n_classes = 2
    c2c_d = 64
    c2c_dropout = 0.1
    emb_dim = 19456

    model = End2EndPhase2Model(
        n_channels=n_ch,
        n_classes=n_classes,
        c2c_d=c2c_d,
        c2c_dropout=c2c_dropout,
        emb_dim=emb_dim
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 50)
    print(f"REVE-Base (Frozen) Parameters : {frozen_params:,}")
    print(f"C2C + Head (Trainable) Params : {trainable_params:,}")
    print(f"TOTAL Parameters              : {frozen_params + trainable_params:,}")
    print("-" * 50)
    
    # Detailed trainable params
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,}")

if __name__ == "__main__":
    count_parameters()
