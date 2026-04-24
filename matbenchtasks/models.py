from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import ModelConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HybridTRIADS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cfg: ModelConfig,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_steps = cfg.max_steps
        self.n_props = 22
        self.stat_dim = 6
        self.magpie_dim = self.n_props * self.stat_dim
        self.mat2vec_dim = 200
        self.extra_dim = max(input_dim - self.magpie_dim - self.mat2vec_dim, 0)

        self.tok_proj = nn.Sequential(nn.Linear(self.stat_dim, cfg.d_attn), nn.LayerNorm(cfg.d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(nn.Linear(self.mat2vec_dim, cfg.d_attn), nn.LayerNorm(cfg.d_attn), nn.GELU())

        self.sa1 = nn.MultiheadAttention(cfg.d_attn, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.sa1_n = nn.LayerNorm(cfg.d_attn)
        self.sa1_ff = nn.Sequential(
            nn.Linear(cfg.d_attn, cfg.d_attn * 2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_attn * 2, cfg.d_attn)
        )
        self.sa1_fn = nn.LayerNorm(cfg.d_attn)
        self.sa2 = nn.MultiheadAttention(cfg.d_attn, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.sa2_n = nn.LayerNorm(cfg.d_attn)
        self.sa2_ff = nn.Sequential(
            nn.Linear(cfg.d_attn, cfg.d_attn * 2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_attn * 2, cfg.d_attn)
        )
        self.sa2_fn = nn.LayerNorm(cfg.d_attn)
        self.ca = nn.MultiheadAttention(cfg.d_attn, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(cfg.d_attn)

        pool_in = cfg.d_attn + self.extra_dim
        self.pool = nn.Sequential(nn.Linear(pool_in, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden), nn.GELU())
        self.z_up = nn.Sequential(
            nn.Linear(cfg.d_hidden * 3, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
        )
        self.y_up = nn.Sequential(
            nn.Linear(cfg.d_hidden * 2, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
        )
        self.head = nn.Linear(cfg.d_hidden, output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        magpie = x[:, : self.magpie_dim]
        extra = x[:, self.magpie_dim : self.magpie_dim + self.extra_dim] if self.extra_dim else None
        mat2vec = x[:, -self.mat2vec_dim :]

        tokens = self.tok_proj(magpie.view(batch, self.n_props, self.stat_dim))
        ctx = self.m2v_proj(mat2vec).unsqueeze(1)

        tokens = self.sa1_n(tokens + self.sa1(tokens, tokens, tokens, need_weights=False)[0])
        tokens = self.sa1_fn(tokens + self.sa1_ff(tokens))
        tokens = self.sa2_n(tokens + self.sa2(tokens, tokens, tokens, need_weights=False)[0])
        tokens = self.sa2_fn(tokens + self.sa2_ff(tokens))
        tokens = self.ca_n(tokens + self.ca(tokens, ctx, ctx, need_weights=False)[0])
        pooled = tokens.mean(dim=1)
        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=-1)
        return self.pool(pooled)

    def forward(self, x: torch.Tensor, deep_supervision: bool = False):
        batch = x.size(0)
        xp = self._encode(x)
        z = torch.zeros(batch, self.cfg.d_hidden, device=x.device, dtype=x.dtype)
        y = torch.zeros(batch, self.cfg.d_hidden, device=x.device, dtype=x.dtype)
        preds = []
        for _ in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], dim=-1))
            y = y + self.y_up(torch.cat([y, z], dim=-1))
            preds.append(self.head(y).squeeze(-1))
        return preds if deep_supervision else preds[-1]


def scatter_sum(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), dtype=src.dtype, device=src.device)
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
    return out


class GraphMPLayer(nn.Module):
    def __init__(self, d_model: int, n_angle: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.bond_msg = nn.Sequential(nn.Linear(d_model * 2 + n_angle, d_model), nn.SiLU())
        self.bond_gate = nn.Sequential(nn.Linear(d_model * 2 + n_angle, d_model), nn.Sigmoid())
        self.bond_up = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.SiLU(), nn.Dropout(dropout))
        self.atom_msg = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.SiLU())
        self.atom_gate = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.Sigmoid())
        self.atom_up = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.SiLU(), nn.Dropout(dropout))

    def forward(
        self,
        atoms: torch.Tensor,
        bonds: torch.Tensor,
        edge_index: torch.Tensor,
        triplets: torch.Tensor,
        angle_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if triplets.numel() > 0:
            b_ij = bonds[triplets[0]]
            b_kj = bonds[triplets[1]]
            inp = torch.cat([b_ij, b_kj, angle_feat], dim=-1)
            msg = self.bond_msg(inp) * self.bond_gate(inp)
            agg = torch.zeros(bonds.shape, dtype=msg.dtype, device=bonds.device)
            agg.scatter_add_(0, triplets[0].unsqueeze(-1).expand_as(msg), msg)
            bonds = bonds + self.bond_up(torch.cat([bonds, agg], dim=-1))
        inp = torch.cat([atoms[edge_index[0]], atoms[edge_index[1]], bonds], dim=-1)
        msg = self.atom_msg(inp) * self.atom_gate(inp)
        agg = scatter_sum(msg, edge_index[1], atoms.size(0))
        atoms = atoms + self.atom_up(torch.cat([atoms, agg], dim=-1))
        return atoms, bonds


class GraphTRIADS(nn.Module):
    def __init__(self, comp_dim: int, global_dim: int, cfg: ModelConfig, output_dim: int = 1) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_graph
        self.n_props = 22
        self.stat_dim = 6
        self.magpie_dim = self.n_props * self.stat_dim
        self.mat2vec_dim = 200
        self.extra_dim = max(comp_dim - self.magpie_dim - self.mat2vec_dim, 1)
        self.output_dim = output_dim

        self.atom_embed = nn.Embedding(103, d)
        self.atom_feat_proj = nn.Linear(18, d)
        self.rbf_enc = nn.Linear(40, d)
        self.vec_enc = nn.Linear(3, d)
        self.phys_enc = nn.Linear(8, d)

        self.magpie_proj = nn.Linear(self.stat_dim, d)
        self.extra_proj = nn.Linear(self.extra_dim, d)
        self.m2v_proj = nn.Linear(self.mat2vec_dim, d)
        self.ctx_proj = nn.Linear(11 + global_dim, d)
        self.type_embed = nn.Embedding(2, d)

        self.warmup = GraphMPLayer(d, 8, cfg.dropout)
        self.trm_gnn = GraphMPLayer(d, 8, cfg.dropout)
        self.sa = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.sa_n = nn.LayerNorm(d)
        self.sa_ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, d))
        self.sa_fn = nn.LayerNorm(d)
        self.ca = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d)

        self.z_proj = nn.Linear(d * 3, d)
        self.z_up = nn.Sequential(nn.Linear(d * 2, d), nn.SiLU(), nn.Linear(d, d))
        self.z_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.y_up = nn.Sequential(nn.Linear(d * 2, d), nn.SiLU(), nn.Linear(d, d))
        self.y_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, output_dim))
        self._gate_sparsity = torch.tensor(0.0)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _composition_tokens(self, comp: torch.Tensor, global_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = comp.size(0)
        magpie = comp[:, : self.magpie_dim].view(batch, self.n_props, self.stat_dim)
        extra = comp[:, self.magpie_dim : self.magpie_dim + self.extra_dim]
        if extra.size(1) != self.extra_dim:
            extra = F.pad(extra, (0, self.extra_dim - extra.size(1)))
        m2v = comp[:, -self.mat2vec_dim :]
        mag_tok = self.magpie_proj(magpie)
        ext_tok = self.extra_proj(extra).unsqueeze(1)
        m2v_tok = self.m2v_proj(m2v).unsqueeze(1)
        comp_tok = torch.cat([mag_tok, ext_tok, m2v_tok], dim=1) + self.type_embed.weight[0]
        struct_meta = comp[:, self.magpie_dim + self.extra_dim : self.magpie_dim + self.extra_dim + 11]
        if struct_meta.size(1) != 11:
            struct_meta = F.pad(struct_meta, (0, 11 - struct_meta.size(1)))
        ctx = self.ctx_proj(torch.cat([struct_meta, global_phys], dim=-1))
        return comp_tok, ctx

    def forward(self, comp: torch.Tensor, global_phys: torch.Tensor, graph: Dict, deep_supervision: bool = False):
        batch = graph["n_crystals"]
        device = comp.device
        edge_index = graph["ei"]

        atoms = self.atom_embed(graph["atom_z"].clamp(0, 102)) + self.atom_feat_proj(graph["atom_feat"])
        bonds = self.rbf_enc(graph["rbf"]) * torch.tanh(self.vec_enc(graph["vec"])) + self.phys_enc(graph["phys"])
        atoms, bonds = self.warmup(atoms, bonds, edge_index, graph["triplets"], graph["angle_feat"])

        comp_tok, ctx = self._composition_tokens(comp, global_phys)
        z = torch.zeros(batch, self.cfg.d_graph, device=device, dtype=comp.dtype)
        y = torch.zeros(batch, self.cfg.d_graph, device=device, dtype=comp.dtype)
        preds: List[torch.Tensor] = []
        self._gate_sparsity = torch.tensor(0.0, device=device)

        for cyc in range(self.cfg.max_cycles):
            atoms, bonds = self.trm_gnn(atoms, bonds, edge_index, graph["triplets"], graph["angle_feat"])
            atom_tok, atom_mask = _pad_atoms(atoms, graph["n_atoms"], batch)
            atom_tok = atom_tok + self.type_embed.weight[1]
            full_tok = torch.cat([comp_tok, atom_tok], dim=1)
            full_mask = torch.cat(
                [
                    torch.zeros(batch, comp_tok.size(1), dtype=torch.bool, device=device),
                    atom_mask,
                ],
                dim=1,
            )
            sa_out = self.sa(full_tok, full_tok, full_tok, key_padding_mask=full_mask, need_weights=False)[0]
            full_tok = self.sa_n(full_tok + sa_out)
            full_tok = self.sa_fn(full_tok + self.sa_ff(full_tok))
            comp_tok = full_tok[:, : comp_tok.size(1)]
            atom_tok = full_tok[:, comp_tok.size(1) :]
            ca_out = self.ca(comp_tok, atom_tok, atom_tok, key_padding_mask=atom_mask, need_weights=False)[0]
            comp_tok = self.ca_n(comp_tok + ca_out)
            atoms = _unpad_atoms(atom_tok, graph["n_atoms"])

            xp = comp_tok.mean(dim=1)
            z_inp = self.z_proj(torch.cat([xp, ctx, y], dim=-1))
            z_cand = self.z_up(torch.cat([z_inp, z], dim=-1))
            z_gate = self.z_gate(torch.cat([z_inp, z], dim=-1))
            z = z + z_gate * z_cand
            y_cand = self.y_up(torch.cat([y, z], dim=-1))
            y_gate = self.y_gate(torch.cat([y, z], dim=-1))
            y = y + y_gate * y_cand
            self._gate_sparsity = self._gate_sparsity + (z_gate.mean() + y_gate.mean()) * 0.5
            preds.append(self.head(y).squeeze(-1))
            if cyc >= self.cfg.min_cycles - 1 and y_gate.max().item() < 0.05:
                break
        self._gate_sparsity = self._gate_sparsity / max(len(preds), 1)
        return preds if deep_supervision else preds[-1]


def _pad_atoms(atoms: torch.Tensor, n_atoms: List[int], batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_atoms = max(n_atoms)
    d = atoms.size(-1)
    out = atoms.new_zeros(batch, max_atoms, d)
    mask = torch.ones(batch, max_atoms, dtype=torch.bool, device=atoms.device)
    offset = 0
    for i, count in enumerate(n_atoms):
        out[i, :count] = atoms[offset : offset + count]
        mask[i, :count] = False
        offset += count
    return out, mask


def _unpad_atoms(atom_tok: torch.Tensor, n_atoms: List[int]) -> torch.Tensor:
    return torch.cat([atom_tok[i, :count] for i, count in enumerate(n_atoms)], dim=0)


def build_model(kind: str, input_dim: int, global_dim: int, cfg: ModelConfig, output_dim: int) -> nn.Module:
    if kind == "hybrid":
        return HybridTRIADS(input_dim=input_dim, cfg=cfg, output_dim=output_dim)
    if kind == "graph":
        return GraphTRIADS(comp_dim=input_dim, global_dim=global_dim, cfg=cfg, output_dim=output_dim)
    raise ValueError(f"Unknown model kind: {kind}")
