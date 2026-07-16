from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, GENConv, MLP, BatchNorm

import pickle

from framework import utils


class ProcessorBlock(nn.Module):
    """ ProcessorBlock class, implements a single message-passing layer that updates node features.
        Three message-passing blocks are implemented:
            - GAT: Graph Attention Network v2 (Brody et al., 2022)
            - GINE: modified GIN (Xu et al., 2019) that uses edge features in the message-passing (Hu et al., 2020)
            - GEN: GENeralized Graph Convolution (Li et al., 2020)

        args:
        mp_type: str, the type of message-passing block to use, one of ['GAT', 'GINE', 'GEN']
        node_in: int, the dimension of the input node features
        hidden_channel: int, the width of the GAT output (heads are averaged, not concatenated)
        node_out: int, the dimension of the output node features
        edge_in: int, the dimension of the edge features
        num_heads: int, the number of attention heads, only used by GAT
    """

    def __init__(self, mp_type, node_in, hidden_channel, node_out, edge_in, num_heads):
        super().__init__()
        self.mp_type = mp_type

        # initialize the message-passing conv layer
        if mp_type == 'GAT':
            # concat=False averages the heads, so the output width stays hidden_channel
            # regardless of num_heads, which keeps the downstream dimensions fixed
            self.conv = GATv2Conv(node_in, hidden_channel, heads=num_heads, edge_dim=edge_in, add_self_loops=True, concat=False)
        elif mp_type == 'GINE':
            # for GINE we need to define the update function of the node features, we use a 2-layer MLP
            gin_nn = MLP(channel_list=[node_in, node_out * 2, node_out], act='relu', norm='LayerNorm')
            self.conv = GINEConv(gin_nn, edge_dim=edge_in)
        elif mp_type == 'GEN':
            self.conv = GENConv(node_in, node_out, edge_dim=edge_in)
        else:
            raise ValueError(f"Unknown mp_type: {mp_type}")

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class GNN_LSTM(nn.Module):
    """ Overall virtual sensing model. A stack of message-passing blocks computes a graph embedding
        at every time step, and an LSTM processes the resulting sequence of embeddings temporally to
        reconstruct the response at the unsensed nodes.

        args:
        mp_type: str, the type of message-passing block to use, one of ['GAT', 'GINE', 'GEN']
        num_heads: int, the number of attention heads, only used when mp_type is 'GAT'
        dropout: float, the dropout probability
        in_dim: int, the dimension of the input node features (node attributes + the masked target channel)
        edge_in: int, the dimension of the edge features
        hidden_gnn: int, the hidden channel width of each GNN layer
        n_layers_gnn: int, the number of stacked GNN layers (depth of the processor)
        hidden_lstm: int, the hidden size of the LSTM
        n_layers_lstm: int, the number of stacked LSTM layers
        out_dim: int, the dimension of the predicted node output per time step
        time_chunk: int, the number of time steps message-passed as a single batched graph.
                    Trades memory for speed: lower it if a forward pass runs out of memory
    """

    def __init__(self, mp_type, num_heads, dropout, in_dim, edge_in, hidden_gnn, n_layers_gnn, hidden_lstm, n_layers_lstm, out_dim, time_chunk=100):
        super().__init__()

        self.dropout = dropout
        self.time_chunk = time_chunk
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # initialize the message-passing blocks, the first one takes the raw node features
        for i in range(n_layers_gnn):
            gnn_in = in_dim if i == 0 else hidden_gnn
            self.layers.append(ProcessorBlock(mp_type=mp_type, node_in=gnn_in, hidden_channel=hidden_gnn, node_out=hidden_gnn, edge_in=edge_in, num_heads=num_heads))
            self.norms.append(BatchNorm(hidden_gnn))

        # initialize the temporal model that process the per-timestep graph embeddings
        self.lstm = nn.LSTM(
            input_size=hidden_gnn,
            hidden_size=hidden_lstm,
            num_layers=n_layers_lstm,
            dropout=dropout if n_layers_lstm > 1 else 0.0,
            batch_first=True,
        )

        # map the LSTM hidden state back to the physical output space
        self.readout = nn.Linear(hidden_lstm, out_dim)

    def forward(self, data):
        """ Run message-passing on each time chunk and predict the output sequence with the LSTM.

            args:
            data: torch_geometric Data/Batch, with x (node features), y (masked target, zero at the unsensed nodes), edge_index and edge_attr
        """
        x0 = data.x
        y = data.y
        edge_index = data.edge_index
        edge_attribute = data.edge_attr

        N = x0.size(0)          # number of nodes in the (batched) graph
        T = y.size(1)           # number of timesteps in the target sequence
        device = x0.device

        z_list = []

        # process the sequence in chunks of `time_chunk` steps to lower memory usage: each timestep
        # becomes an independent copy of the graph, all stacked into one big batched graph so that
        # message-passing runs in parallel across time
        for t0 in range(0, T, self.time_chunk):
            t1 = min(t0 + self.time_chunk, T)
            Tc = t1 - t0

            yc = y[:, t0:t1]

            # repeat the node features for every timestep in the chunk
            x_chunked = x0.unsqueeze(0).expand(Tc, N, x0.size(1)).reshape(Tc * N, x0.size(1))

            # flatten the time window into a single graph batch
            y_chunked = yc.t().reshape(Tc * N, 1)

            # append the masked target as an extra input channel so the GNN sees the sensed response at each step
            x_in = torch.cat([x_chunked, y_chunked], dim=-1)

            # build the batched edge index for the current chunk, shifting each graph copy's node ids
            # by t*N so that the Tc copies stay disconnected.
            key = f"_ei_bd_{Tc}"
            ei_bd = getattr(data, key, None)
            if ei_bd is None or ei_bd.device != device:
                E = edge_index.size(1)
                offset = (torch.arange(Tc, device=device) * N).repeat_interleave(E)
                ei_bd = edge_index.repeat(1, Tc) + offset.unsqueeze(0)
                setattr(data, key, ei_bd)

            edge_attribute_chunk = edge_attribute.repeat(Tc, 1)

            # update the node features through the message-passing
            z = x_in
            for i, layer in enumerate(self.layers):
                z = layer(z, ei_bd, edge_attribute_chunk)
                z = self.norms[i](z)
                z = F.relu(z)
                z = F.dropout(z, p=self.dropout, training=self.training)

            z = z.reshape(Tc, N, z.size(-1)).transpose(0, 1).contiguous()
            z_list.append(z)

        # concatenate the chunks along time, then run the temporal model and the readout
        z_seq = torch.cat(z_list, dim=1)
        z_seq, _ = self.lstm(z_seq)
        y_pred = self.readout(z_seq)
        return y_pred


def train(model, loader, optimizer, criterion, device, epoch=0, grad_clip=1.0):
    """ Train the network for one epoch on a purely data-driven loss.
        Each batch is cropped to a random temporal window to reduce memory usage.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    win_len = 1000  # length of the random window cropped from each sequence

    for batch_idx, data in enumerate(loader):
        # send data to device
        data = data.to(device)

        # crop a random window of the sequence, seeded so that the crop is reproducible per epoch
        T = data.y.shape[1]
        if T > win_len:
            g = torch.Generator(device=device).manual_seed(epoch * 42 + batch_idx)
            start = torch.randint(0, T - win_len + 1, (1,), generator=g, device=device).item()
            end = start + win_len
        else:
            start = 0
            end = T

        y_full = data.y
        target_full = data.target

        data.y = y_full[:, start:end]
        data.target = target_full[:, start:end]

        # reset the gradients back to zero
        optimizer.zero_grad(set_to_none=True)

        # run the forward pass and compute the batch training loss
        y_pred = model(data)
        pred_target = y_pred[:, :, 0]  # channel 0 is the predicted target signal
        loss = criterion(pred_target, data.target)

        # backpropagate, clip the gradients and update the parameters
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_phys(model, loader, optimizer, criterion, device, scaler_path, channel, epoch=0,
               lambda_data=1, lambda_phy=1e-10, grad_clip=1.0, dt=None, highpass_hz=2):
    """ Train the network for one epoch on a combined data and physics-informed loss.
        The physics term is the residual of the equation of motion (M a + C v + K x = F).
    """
    if dt is None:
        raise ValueError("train_phys requires an explicit dt")

    # load the target scaler, needed to bring the prediction back to physical units
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    y_scaler = scalers["y"]

    target_mean = torch.tensor(y_scaler.mean_, dtype=torch.float64, device=device)
    target_scale = torch.tensor(y_scaler.scale_, dtype=torch.float64, device=device)

    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_phy_loss = 0.0
    n_batches = 0
    win_len = 1000  # length of the random window cropped from each sequence

    for batch_idx, data in enumerate(loader):
        # send data to device
        data = data.to(device)

        # crop a random window of the sequence, seeded so that the crop is reproducible per epoch
        T = data.y.shape[1]
        if T > win_len:
            g = torch.Generator(device=device).manual_seed(epoch * 42 + batch_idx)
            start = torch.randint(0, T - win_len + 1, (1,), generator=g, device=device).item()
            end = start + win_len
        else:
            start, end = 0, T

        data.y = data.y[:, start:end]
        data.target = data.target[:, start:end]

        # assemble the system matrices of the batch, the forcing is cropped to the same window
        M = torch.stack([torch.tensor(m, dtype=torch.float64) for m in data.M]).to(device)
        C = torch.stack([torch.tensor(c, dtype=torch.float64) for c in data.C]).to(device)
        K = torch.stack([torch.tensor(k, dtype=torch.float64) for k in data.K]).to(device)
        forcing = torch.stack([torch.tensor(f, dtype=torch.float64) for f in data.F])[:, :, start:end].to(device)

        # reset the gradients back to zero
        optimizer.zero_grad(set_to_none=True)

        # run the forward pass and compute the data loss
        y_pred = model(data).squeeze(-1)
        y_true = data.target
        loss_data = criterion(y_pred, y_true)

        # unflatten the stacked node predictions back into [batch, n_nodes, time]
        n_nodes = target_mean.shape[0]
        batch_size = y_pred.shape[0] // n_nodes
        y_phys = y_pred.view(batch_size, n_nodes, -1)

        # undo the target normalization so the response is in physical units before integrating
        y_phys_unscaled = y_phys * target_scale.view(1, n_nodes, 1) + target_mean.view(1, n_nodes, 1)

        # integration/differentiation to obtain the complete a v x
        a_pred, v_pred, x_pred = utils.integrate(y_phys_unscaled, dt, channel, highpass_hz=highpass_hz)
        loss_phy = utils.eom_loss(M, C, K, forcing, a_pred, v_pred, x_pred)

        loss = lambda_data * loss_data + lambda_phy * loss_phy

        # backpropagate, clip the gradients and update the parameters
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_data_loss += lambda_data * loss_data.item()
        total_phy_loss += lambda_phy * loss_phy.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "total": total_loss / n,
        "data": total_data_loss / n,
        "physics": total_phy_loss / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """ Evaluate the data loss of the model on a held-out set. """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for data in loader:
        data = data.to(device)
        y_pred = model(data)
        loss = criterion(y_pred[:, :, 0], data.target)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def test(model, loader, device):
    """ Generate the full-field predictions for every sample in the loader. """
    model.eval()
    all_preds = []
    for batch in loader:
        batch = batch.to(device)
        y_pred = model(batch)
        all_preds.append(y_pred.squeeze(-1).detach().cpu())
    return torch.cat(all_preds, dim=0)
