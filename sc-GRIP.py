import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import scanpy as sc
import os

class GAEModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAEModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.decoder = torch.nn.Bilinear(out_channels, out_channels, 1)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return torch.sigmoid(self.decoder(src, dst)).squeeze()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)

def load_cell_graphs(edge_file, adata_file=None, expression_file=None, selected_cells=None):
    if adata_file:
        adata = sc.read_h5ad(adata_file)
        expr_df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
    else:
        expr_df = pd.read_csv(expression_file, index_col=0)
    edge_df = pd.read_csv(edge_file, index_col=0)

    tf_list = edge_df.index.to_list()
    target_gene_list = edge_df.columns.to_list()
    all_interacting_genes = tf_list + target_gene_list
    common_genes = list(set(expr_df.columns) & set(all_interacting_genes))
    common_target_genes = list(set(expr_df.columns) & set(target_gene_list))
    common_tf_genes = list(set(expr_df.columns) & set(tf_list))
    expr_df = expr_df[common_genes]
    edge_df = edge_df.loc[common_tf_genes, common_target_genes]

    if selected_cells is not None:
        expr_df = expr_df.loc[selected_cells]

    adj = torch.tensor(edge_df.values, dtype=torch.float)
    edge_index = (adj > 0).nonzero(as_tuple=False).T

    data_list = []
    for i in range(expr_df.shape[0]):
        expr = torch.tensor(expr_df.iloc[i].values.reshape(-1, 1), dtype=torch.float)
        data = Data(x=expr, edge_index=edge_index)
        data_list.append(data)

    return edge_df, expr_df, data_list, edge_index

def train(model, dataset,cor_df,num_epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    correlation_dict = {
        (row['TF'], row['target']): row['Correlation']
        for _, row in cor_df.iterrows()}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for data in loader:
            optimizer.zero_grad()
            preds = model(data.x, data.edge_index)
            tf_expr = data.x[data.edge_index[0]]
            tgt_expr = data.x[data.edge_index[1]]
            labels = []

            for src, tgt in zip(edge_index[0], edge_index[1]):
                corr = correlation_dict.get((tf_expr, tgt_expr), correlation_dict.get((tgt_expr, tf_expr), 0))
                labels.append(float(corr > 0.5))  # threshold can be changed

            labels = torch.tensor(labels, dtype=torch.float32, device=preds.device)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataset):.4f}")

def get_consensus_edge_labels(model, dataset,cor_df):
    model.eval()
    all_preds = []

    correlation_dict = {
        (row['TF'], row['target']): row['Correlation']
        for _, row in cor_df.iterrows()}

    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=1):
            pred = model(data.x, data.edge_index)
            all_preds.append(pred)
    stacked = torch.stack(all_preds)
    avg_preds = stacked.median(dim=0).values

    return avg_preds

if __name__ == "__main__":
    species = "human"
    celltype = "muscle"

    correlation_df = pd.read_csv("sc-grip/"+species+"_"+celltype+"_correlations.csv")
    expression_file = "data/"+species+"_"+celltype+"_gex_common.csv"
    edge_file = "data/"+species+"_"+celltype+"_tf_interaction_common.csv"
    print(expression_file,edge_file)
    n_runs = 5
    num_epochs = 100

    edge_df, expr_df, dataset, edge_index = load_cell_graphs(expression_file=expression_file, edge_file=edge_file)
    print(f"Loaded {len(dataset)} graphs (1 per cell)")

    in_channels = dataset[0].x.shape[1]

    edge_map = (edge_df.values > 0).nonzero()
    tf_list = edge_df.index.to_list()
    tgt_list = edge_df.columns.to_list()
    edge_names = [f"{tf_list[i]}->{tgt_list[j]}" for i, j in zip(*edge_map)]

    all_edge_probs = []

    for run in range(n_runs):
        print(f"\n=== Run {run+1}/{n_runs} ===")
        model = GAEModel(in_channels=in_channels, out_channels=64)
        train(model, dataset, cor_df=correlation_df,num_epochs=num_epochs)
        edge_probs = get_consensus_edge_labels(model, dataset,cor_df = correlation_df)
        all_edge_probs.append(edge_probs.cpu().numpy())

    all_edge_probs = np.stack(all_edge_probs)
    mean_probs = all_edge_probs.mean(axis=0)
    std_probs = all_edge_probs.std(axis=0)

    output_df = pd.DataFrame({
        'edge': edge_names,
        'activation_score_mean': mean_probs,
        'activation_score_std': std_probs,
    })

    os.makedirs("predictions", exist_ok=True)
    output_df.to_csv("predictions/"+species+"_"+celltype+"_corr.csv", index=False)
