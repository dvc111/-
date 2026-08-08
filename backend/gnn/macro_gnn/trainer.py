"""宏观模型训练：RGCNNodeClassifier 二分类（节点是不是答案）。"""
import torch; import torch.nn.functional as F
def train_macro_model(model, train_data, epochs=20, lr=0.001, save_path="macro_model.pth"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        total_loss = 0.0; total = 0; model.train()
        for gd, lbl in train_data:
            x, ei, et = gd.node_features, gd.edge_index, gd.edge_type
            logits = model(x, ei, et)
            loss = F.cross_entropy(logits, lbl.to(x.device))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * lbl.size(0); total += lbl.size(0)
        if ep == 1 or ep % 10 == 0: print(f"[{ep}/{epochs}] loss={total_loss/total:.4f}")
    ckpt = {"_model_config": {"in_dim": model.in_dim, "hidden_dim": model.hidden_dim, "num_relations": model.num_relations, "num_layers": 2, "num_classes": 2, "dropout": 0.2}}
    ckpt.update(model.state_dict()); torch.save(ckpt, save_path)
    return model