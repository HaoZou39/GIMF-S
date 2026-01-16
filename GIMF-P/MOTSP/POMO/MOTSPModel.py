import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ================================================================
# Edge-aware auxiliary head
# ================================================================

class EdgeDetourHead(nn.Module):
    """Predict pairwise *detour* between node pairs.

    This head is designed as a lightweight, dense edge predictor operating on
    *node embeddings* that already fuse satellite/basemap context via the
    encoder fusion layers.

    The recommended training target is **log-detour**:
        log( d_road / (d_euclid + eps) )

    so that the label is scale-invariant and numerically stable.

    Output:
        pred: (B, N, N)  (larger => more detour / harder to travel)

    Notes:
        - For correct detour calculation, use precomputed euclid_distance_matrix 
          that uses the same normalization as road distance_matrix.
        - We symmetrize predictions by default (most road distance matrices are symmetric).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 256, use_euclid: bool = True):
        super().__init__()
        self.use_euclid = use_euclid
        in_dim = embedding_dim * 3 + (1 if use_euclid else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def compute_euclid_from_coords(problems: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute Euclidean distances from normalized coordinates (FALLBACK).
        
        WARNING: This uses normalized coordinates which may have different scale
        than the road distance matrix. Use precomputed euclid_distance_matrix when available.

        Args:
            problems: (B, N, 2*num_objectives)
        Returns:
            (B, N, N)
        """
        coords = problems[..., 0:2]
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        return torch.sqrt((diff ** 2).sum(dim=-1) + eps)

    def forward(self, node_embed: torch.Tensor, problems: torch.Tensor, 
                euclid_distance_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_embed: (B, N, E) - node embeddings
            problems: (B, N, 2*num_objectives) - node coordinates
            euclid_distance_matrix: (B, N, N) - precomputed Euclidean distance matrix (recommended)
                                    If None, will compute from coordinates (may have scale mismatch!)
        
        Returns:
            pred: (B, N, N) - predicted log-detour
        """
        B, N, E = node_embed.shape

        hi = node_embed[:, :, None, :].expand(B, N, N, E)
        hj = node_embed[:, None, :, :].expand(B, N, N, E)
        habs = (hi - hj).abs()

        feats = [hi, hj, habs]
        if self.use_euclid:
            # Use precomputed Euclidean distance if available (correct normalization)
            if euclid_distance_matrix is not None:
                d_e = euclid_distance_matrix.unsqueeze(-1)  # (B, N, N, 1)
            else:
                # Fallback: compute from coordinates (may have scale mismatch!)
                d_e = self.compute_euclid_from_coords(problems).unsqueeze(-1)  # (B, N, N, 1)
            feats.append(d_e)

        x = torch.cat(feats, dim=-1)
        pred = self.mlp(x).squeeze(-1)  # (B, N, N)

        # Encourage symmetry
        pred = 0.5 * (pred + pred.transpose(1, 2))

        # Diagonal to 0
        diag = torch.arange(N, device=pred.device)
        pred[:, diag, diag] = 0.0

        return pred


# ================================================================
# Main Model
# ================================================================

class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)

        # Edge-aware options (all optional / backward-compatible)
        self.use_edge_head = bool(model_params.get('use_edge_head', False) or model_params.get('use_edge_bias', False))
        self.use_edge_bias = bool(model_params.get('use_edge_bias', False))

        self.edge_head: Optional[EdgeDetourHead] = None
        if self.use_edge_head:
            self.edge_head = EdgeDetourHead(
                embedding_dim=int(model_params['embedding_dim']),
                hidden_dim=int(model_params.get('edge_head_hidden_dim', 256)),
                use_euclid=bool(model_params.get('edge_head_use_euclid', True)),
            )

        # Cached tensors for a loaded batch
        self.encoded_nodes: Optional[torch.Tensor] = None   # (B, N+patches, E)
        self.edge_pred: Optional[torch.Tensor] = None       # (B, N, N)
        self.edge_bias: Optional[torch.Tensor] = None       # (B, N, N)

    def pre_forward(self, reset_state):
        """Encode a new batch and (optionally) pre-compute edge bias."""
        self.encoded_nodes = self.encoder(reset_state.problems, reset_state.xy_img)
        self.decoder.set_kv(self.encoded_nodes)

        self.edge_pred = None
        self.edge_bias = None

        if self.use_edge_head and self.edge_head is not None:
            node_size = reset_state.problems.size(1)
            node_tokens = self.encoded_nodes[:, :node_size, :]
            # Pass precomputed Euclidean distance matrix if available (for correct detour calculation)
            euclid_dist = getattr(reset_state, 'euclid_distance_matrix', None)
            self.edge_pred = self.edge_head(node_tokens, reset_state.problems, euclid_dist)

            # Detour is theoretically >= 1, hence log-detour >= 0.
            # Clamping helps stabilize logit bias injection and removes rare numerical artifacts.
            if bool(self.model_params.get('edge_pred_nonnegative', True)):
                self.edge_pred = torch.clamp(self.edge_pred, min=0.0)

            if self.use_edge_bias:
                alpha = float(self.model_params.get('edge_bias_alpha', 1.0))
                bias = -alpha * self.edge_pred  # larger detour => lower probability

                clip = self.model_params.get('edge_bias_clip', None)
                if clip is not None:
                    clip = float(clip)
                    bias = torch.clamp(bias, -clip, clip)

                self.edge_bias = bias

        self.decoder.set_edge_bias(self.edge_bias)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size, device=state.BATCH_IDX.device)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size), device=state.BATCH_IDX.device)

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q1(encoded_first_node)
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, current_node=state.current_node)

            if self.training or self.model_params.get('eval_type', 'softmax') == 'softmax':
                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, pomo_size)

                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob

    # ------------------------
    # Edge auxiliary losses
    # ------------------------

    def get_edge_prediction(self) -> Optional[torch.Tensor]:
        return self.edge_pred

    def compute_edge_supervised_loss(
        self,
        problems: torch.Tensor,
        distance_matrix: torch.Tensor,
        euclid_distance_matrix: Optional[torch.Tensor] = None,
        unreachable_threshold: Optional[float] = None,
        eps: float = 1e-6,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """Supervised regression loss for edge predictions.

        Target is log-detour = log(d_road / (d_euclid+eps)).

        Args:
            problems: (B, N, 2*num_objectives)
            distance_matrix: (B, N, N) - road distance matrix
            euclid_distance_matrix: (B, N, N) - precomputed Euclidean distance matrix (recommended)
                                    If None, will compute from coordinates (may have scale mismatch!)
            unreachable_threshold: if provided, edges with d_road >= threshold are ignored
        """
        if self.edge_pred is None:
            return torch.zeros((), device=problems.device)

        d_road = distance_matrix
        
        # Use precomputed Euclidean distance if available (correct normalization)
        if euclid_distance_matrix is not None:
            d_euclid = euclid_distance_matrix
        else:
            # Fallback: compute from coordinates (may have scale mismatch!)
            d_euclid = EdgeDetourHead.compute_euclid_from_coords(problems, eps=eps)
        
        ratio = d_road / (d_euclid + eps)
        ratio = torch.clamp(ratio, min=1.0)
        target = torch.log(ratio + eps)

        # mask invalid
        B, N, _ = target.shape
        mask = torch.ones((B, N, N), dtype=torch.bool, device=target.device)
        diag = torch.arange(N, device=target.device)
        mask[:, diag, diag] = False
        if unreachable_threshold is not None:
            mask &= (d_road < float(unreachable_threshold))

        pred = self.edge_pred
        if pred.shape != target.shape:
            pred = pred[:, :N, :N]

        if mask.sum() == 0:
            return torch.zeros((), device=problems.device)

        loss = F.smooth_l1_loss(pred[mask], target[mask], reduction=reduction)
        return loss

    def compute_edge_hard_ranking_loss(
        self,
        problems: torch.Tensor,
        distance_matrix: torch.Tensor,
        euclid_distance_matrix: Optional[torch.Tensor] = None,
        euclid_topk: int = 5,
        margin: float = 0.5,
        unreachable_threshold: Optional[float] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Hard-negative ranking loss within Euclidean-near neighbors.

        For each node i, among Euclidean top-k nearest candidates, pick:
            pos = road-nearest, neg = road-farthest
        Enforce:
            pred(i,pos) + margin <= pred(i,neg)

        pred is log-detour proxy: larger => worse.
        """
        if self.edge_pred is None:
            return torch.zeros((), device=problems.device)

        # Use precomputed Euclidean distance if available (correct normalization)
        if euclid_distance_matrix is not None:
            d_e = euclid_distance_matrix
        else:
            # Fallback: compute from coordinates (may have scale mismatch!)
            d_e = EdgeDetourHead.compute_euclid_from_coords(problems, eps=eps)  # (B,N,N)
        B, N, _ = d_e.shape
        k = int(min(max(1, euclid_topk), max(1, N - 1)))

        # Exclude self
        d_e = d_e + torch.eye(N, device=d_e.device).unsqueeze(0) * 1e6

        topk_idx = torch.topk(d_e, k=k, largest=False, dim=-1).indices  # (B,N,k)
        d_road_cand = torch.gather(distance_matrix, 2, topk_idx)  # (B,N,k)

        if unreachable_threshold is not None:
            d_road_cand = d_road_cand.clone()
            d_road_cand[d_road_cand >= float(unreachable_threshold)] = 1e6

        pos_in_k = d_road_cand.argmin(dim=-1, keepdim=True)  # (B,N,1)
        neg_in_k = d_road_cand.argmax(dim=-1, keepdim=True)

        pos_j = torch.gather(topk_idx, 2, pos_in_k).squeeze(-1)  # (B,N)
        neg_j = torch.gather(topk_idx, 2, neg_in_k).squeeze(-1)

        pred = self.edge_pred
        pred_pos = torch.gather(pred, 2, pos_j.unsqueeze(-1)).squeeze(-1)  # (B,N)
        pred_neg = torch.gather(pred, 2, neg_j.unsqueeze(-1)).squeeze(-1)

        loss = F.relu(float(margin) + pred_pos - pred_neg)
        return loss.mean()


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        fusion_layer_num = self.model_params['fusion_layer_num']
        num_objectives = self.model_params['num_objectives']

        coord_dim = 2 * num_objectives
        self.embedding = nn.Linear(coord_dim, embedding_dim)
        self.embedding_patch = PatchEmbedding(**model_params)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num - fusion_layer_num)])
        self.layers_img = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num - fusion_layer_num)])
        self.fusion_layers = nn.ModuleList([EncoderFusionLayer(**model_params) for _ in range(fusion_layer_num)])
        self.fcp = nn.Parameter(torch.randn(1, self.model_params['bn_num'], embedding_dim))
        self.fcp_img = nn.Parameter(torch.randn(1, self.model_params['bn_img_num'], embedding_dim))

    def forward(self, data, img):
        embedded_input = self.embedding(data)
        embedded_patch = self.embedding_patch(img)

        out = embedded_input
        out_img = embedded_patch
        for i in range(self.model_params['encoder_layer_num'] - self.model_params['fusion_layer_num']):
            out = self.layers[i](out)
            out_img = self.layers_img[i](out_img)

        fcp = self.fcp.repeat(data.shape[0], 1, 1)
        fcp_img = self.fcp_img.repeat(img.shape[0], 1, 1)

        for layer in self.fusion_layers:
            out, out_img, fcp, fcp_img = layer(out, out_img, fcp, fcp_img)

        return torch.cat((out, out_img), dim=1)


class PatchEmbedding(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.img_size = self.model_params['img_size']
        self.patch_size = self.model_params['patch_size']
        self.in_channels = self.model_params['in_channels']
        self.embed_dim = self.model_params['embedding_dim']

        self.patches = self.img_size // self.patch_size
        self.proj = nn.Linear(self.patch_size * self.patch_size * self.in_channels, self.embed_dim)

        self.position_proj = nn.Sequential(
            nn.Linear(2, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.in_channels, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(
            batch_size, -1, self.patch_size * self.patch_size * self.in_channels
        )

        embedded_patches = self.proj(patches)

        # positional embedding on the same device
        grid_x, grid_y = torch.meshgrid(
            torch.arange(self.patches, device=device),
            torch.arange(self.patches, device=device),
            indexing='ij'
        )
        xy = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        xy = xy / (self.patches - 1)
        embedded_patches = embedded_patches + self.position_proj(xy)

        return embedded_patches


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3


class EncoderFusionLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        self.Wq_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine_img = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1_img = Add_And_Normalization_Module(**model_params)
        self.feedForward_img = Feed_Forward_Module(**model_params)
        self.addAndNormalization2_img = Add_And_Normalization_Module(**model_params)

    def forward(self, input, input_img, fcp, fcp_img):
        input1 = torch.cat((input, fcp), dim=1)
        input1_img = torch.cat((input_img, fcp_img), dim=1)

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(torch.cat((input1, fcp_img), dim=1)), head_num=head_num)
        v = reshape_by_heads(self.Wv(torch.cat((input1, fcp_img), dim=1)), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        q_img = reshape_by_heads(self.Wq_img(input1_img), head_num=head_num)
        k_img = reshape_by_heads(self.Wk_img(torch.cat((input1_img, fcp), dim=1)), head_num=head_num)
        v_img = reshape_by_heads(self.Wv_img(torch.cat((input1_img, fcp), dim=1)), head_num=head_num)

        out_concat_img = multi_head_attention(q_img, k_img, v_img)
        multi_head_out_img = self.multi_head_combine_img(out_concat_img)

        out1_img = self.addAndNormalization1_img(input1_img, multi_head_out_img)
        out2_img = self.feedForward_img(out1_img)
        out3_img = self.addAndNormalization2_img(out1_img, out2_img)

        return (
            out3[:, :-self.model_params['bn_num']],
            out3_img[:, :-self.model_params['bn_img_num']],
            out3[:, -self.model_params['bn_num']:],
            out3_img[:, -self.model_params['bn_img_num']:],
        )


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        num_objectives = self.model_params['num_objectives']

        hyper_input_dim = num_objectives
        hyper_hidden_embd_dim = self.model_params['hyper_hidden_dim']
        self.embd_dim = max(2, num_objectives)
        self.hyper_output_dim = 5 * self.embd_dim

        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)

        self.hyper_Wq_first = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wq_last = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_multi_head_combine = nn.Linear(self.embd_dim, head_num * qkv_dim * embedding_dim, bias=False)

        self.Wq_last_para = None
        self.multi_head_combine_para = None

        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None

        # Edge bias (B,N,N) injected into logits
        self.edge_bias: Optional[torch.Tensor] = None

    def assign(self, pref):
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)

        self.Wq_first_para = self.hyper_Wq_first(mid_embd[:self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[self.embd_dim:2 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.Wk_para = self.hyper_Wk(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.Wv_para = self.hyper_Wv(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.multi_head_combine_para = self.hyper_multi_head_combine(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim)

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']

        num_patches = (self.model_params['img_size'] // self.model_params['patch_size']) * (
            self.model_params['img_size'] // self.model_params['patch_size']
        )
        node_size = encoded_nodes.shape[1] - num_patches

        self.k = reshape_by_heads(F.linear(encoded_nodes, self.Wk_para), head_num=head_num)
        self.v = reshape_by_heads(F.linear(encoded_nodes, self.Wv_para), head_num=head_num)

        self.single_head_key = encoded_nodes[:, :node_size].transpose(1, 2)

    def set_q1(self, encoded_q1):
        head_num = self.model_params['head_num']
        self.q_first = reshape_by_heads(F.linear(encoded_q1, self.Wq_first_para), head_num=head_num)

    def set_edge_bias(self, edge_bias: Optional[torch.Tensor]):
        self.edge_bias = edge_bias

    def forward(self, encoded_last_node, ninf_mask, current_node: Optional[torch.Tensor] = None):
        # encoded_last_node: (B,P,E)
        # ninf_mask: (B,P,N)

        head_num = self.model_params['head_num']

        q_last = reshape_by_heads(F.linear(encoded_last_node, self.Wq_last_para), head_num=head_num)
        q = self.q_first + q_last

        # Allow attending to patch tokens as well.
        num_patches = (self.model_params['img_size'] // self.model_params['patch_size']) * (
            self.model_params['img_size'] // self.model_params['patch_size']
        )
        zeros = torch.zeros(
            ninf_mask.shape[0], ninf_mask.shape[1], num_patches,
            device=ninf_mask.device, dtype=ninf_mask.dtype
        )
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=torch.cat((ninf_mask, zeros), dim=-1))
        mh_atten_out = F.linear(out_concat, self.multi_head_combine_para)

        score = torch.matmul(mh_atten_out, self.single_head_key)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        # Inject edge bias as explicit logit correction (before masking)
        if self.edge_bias is not None and current_node is not None:
            B, P, N = ninf_mask.shape
            bias_row = self.edge_bias.gather(1, current_node.unsqueeze(-1).expand(B, P, N))
            score_clipped = score_clipped + bias_row

        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float, device=score.device))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)

    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        return normalized.transpose(1, 2)


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
