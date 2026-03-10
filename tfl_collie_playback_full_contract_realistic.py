#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn

DRIVE = "/content/drive/MyDrive"
NG_DIR = f"{DRIVE}/NG_SOM_ARTIFACTS"
PARQUET_PATH = os.path.join(NG_DIR, "tfl_ng_dataset.parquet")

MAX_HOLD = 28
MIN_HOLD_BARS = 5
TRANSACTION_COST = 0.05
CENT = 0.01
MAX_REWARD_CLIP = 10.0

LATENT_DIM = 32
N_FORMAT_CLUSTERS = 400
N_POS_CLUSTERS = 256
N_REGIME_CLUSTERS = 144
FMT_EMBED_DIM = 8
POS_EMBED_DIM = 8
REG_EMBED_DIM = 8
N_SCALARS_B = 17
STATE_DIM_B = LATENT_DIM + FMT_EMBED_DIM + POS_EMBED_DIM + REG_EMBED_DIM + N_SCALARS_B
HIDDEN_DIM = 256
N_ACTIONS_B = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REGIME_TOP_PCT = 0.40
REGIME_BOTTOM_PCT = 0.40
REGIME_MIN_SAMPLES = 30

DEFAULT_LOOKBACK = 120
DEFAULT_LOOKAHEAD = 20

CHECKPOINT_PATTERNS = [
    os.path.join(NG_DIR, "*v19*best*.pt"),
    os.path.join(NG_DIR, "*v19*best*.pth"),
    os.path.join(NG_DIR, "*qnet*best*.pt"),
    os.path.join(NG_DIR, "*qnet*best*.pth"),
    os.path.join(NG_DIR, "*checkpoint*.pt"),
    os.path.join(NG_DIR, "*checkpoint*.pth"),
    os.path.join(NG_DIR, "*.pt"),
    os.path.join(NG_DIR, "*.pth"),
]


class SharedTrunkB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fmt_embed = nn.Embedding(N_FORMAT_CLUSTERS + 1, FMT_EMBED_DIM, padding_idx=N_FORMAT_CLUSTERS)
        self.pos_embed = nn.Embedding(N_POS_CLUSTERS + 1, POS_EMBED_DIM, padding_idx=N_POS_CLUSTERS)
        self.reg_embed = nn.Embedding(N_REGIME_CLUSTERS + 1, REG_EMBED_DIM, padding_idx=N_REGIME_CLUSTERS)
        self.net = nn.Sequential(
            nn.LayerNorm(STATE_DIM_B),
            nn.Linear(STATE_DIM_B, HIDDEN_DIM), nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU(),
        )

    def forward(self, z, fmt_id, pos_id, reg_id, scalars):
        fi = fmt_id.clone()
        pi = pos_id.clone()
        ri = reg_id.clone()
        fi[fi < 0] = N_FORMAT_CLUSTERS
        pi[pi < 0] = N_POS_CLUSTERS
        ri[ri < 0] = N_REGIME_CLUSTERS
        x = torch.cat([z, self.fmt_embed(fi), self.pos_embed(pi), self.reg_embed(ri), scalars], dim=-1)
        return self.net(x)


class QNetB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = SharedTrunkB()
        self.adv_head = nn.Sequential(nn.Linear(HIDDEN_DIM, 96), nn.GELU(), nn.Linear(96, N_ACTIONS_B))
        self.val_head = nn.Sequential(nn.Linear(HIDDEN_DIM, 96), nn.GELU(), nn.Linear(96, 1))
        self.surv_head = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.GELU(), nn.Linear(64, 1))
        self.bars_head = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, z, fmt_id, pos_id, reg_id, scalars, return_aux=False):
        h = self.trunk(z, fmt_id, pos_id, reg_id, scalars)
        adv = self.adv_head(h)
        val = self.val_head(h)
        q = val + adv - adv.mean(dim=-1, keepdim=True)
        if not return_aux:
            return q
        surv = torch.sigmoid(self.surv_head(h))
        bars = torch.sigmoid(self.bars_head(h))
        return q, surv, bars


@dataclass
class TradeDecision:
    bar_idx: int
    timestamp: pd.Timestamp
    action: str
    q_hold: float
    q_exit: float
    survival: float
    bars_left_score: float
    unrealized_pnl_cents: float
    bars_held: int
    reason: str


@dataclass
class PlaybackTrade:
    contract: str
    trigger_bar: int
    entry_bar: int
    entry_time: pd.Timestamp
    entry_direction: int
    entry_price: float
    exit_bar: int
    exit_time: pd.Timestamp
    exit_price: float
    realized_pnl_cents: float
    exit_reason: str
    decisions: List[TradeDecision]
    blocked_regime: bool
    fmt_id: int
    pos_id: int
    reg_id: int
    trigger_label: str


def detect_checkpoint_path() -> Optional[str]:
    for pattern in CHECKPOINT_PATTERNS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_qnet(checkpoint_path: Optional[str]) -> Tuple[Optional[QNetB], str]:
    if checkpoint_path is None:
        return None, "No checkpoint found. Using heuristic demo mode for Agent B."
    model = QNetB().to(DEVICE)
    payload = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(payload, dict):
        candidate_keys = ["qnet", "qnet_state_dict", "model", "model_state_dict", "state_dict", "online_qnet", "agent_b", "agent_b_qnet"]
        state_dict = None
        for key in candidate_keys:
            if key in payload and isinstance(payload[key], dict):
                state_dict = payload[key]
                break
        if state_dict is None and all(isinstance(k, str) for k in payload.keys()):
            state_dict = payload
        if state_dict is None:
            raise ValueError(f"Could not infer state_dict from checkpoint: {checkpoint_path}")
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, f"Loaded Agent B checkpoint: {os.path.basename(checkpoint_path)}"


def compute_regime_filter(df: pd.DataFrame) -> Tuple[set, set]:
    reg_rewards: Dict[int, List[float]] = {}
    grouped = dict(tuple(df.sort_values(["contract", "bar_idx"]).groupby("contract", sort=False)))
    for _, grp in grouped.items():
        grp = grp.reset_index(drop=True)
        mask = grp["confirmed_entry"].fillna(False).values.astype(bool)
        rows = np.where(mask)[0]
        closes = grp["close"].astype(float).values
        regs = grp["reg_id"].fillna(-1).astype(int).values
        dirs = grp["entry_direction"].fillna(0).astype(int).values
        n = len(grp)
        for t in rows:
            if t + 1 + MAX_HOLD >= n:
                continue
            direction = int(np.sign(dirs[t]))
            reg = int(regs[t])
            if direction == 0 or reg < 0:
                continue
            pnl = ((float(closes[t + 1 + MAX_HOLD]) - float(closes[t + 1])) * direction / CENT) - TRANSACTION_COST
            pnl = float(np.clip(pnl, -MAX_REWARD_CLIP, MAX_REWARD_CLIP))
            reg_rewards.setdefault(reg, []).append(max(pnl, 0.0))
    scores = [(reg, float(np.mean(vals)), len(vals)) for reg, vals in reg_rewards.items() if len(vals) >= REGIME_MIN_SAMPLES]
    if not scores:
        return set(), set()
    scores.sort(key=lambda x: x[1], reverse=True)
    regs_sorted = [reg for reg, _, _ in scores]
    top_k = int(max(1, round(len(regs_sorted) * REGIME_TOP_PCT)))
    bot_k = int(max(1, round(len(regs_sorted) * REGIME_BOTTOM_PCT)))
    return set(regs_sorted[:top_k]), set(regs_sorted[-bot_k:])


def make_state_B(d: pd.DataFrame, i: int, position: int, bars_held: int, entry_px: float,
                 trade_dir: int, max_unrealized: float = 0.0, consec_adv: int = 0,
                 forced_exit_bar: Optional[int] = None, block_regs: Optional[set] = None):
    row = d.iloc[i]
    c = float(row["close"])
    atr = float(row["atr"]) if pd.notna(row["atr"]) and float(row["atr"]) > 1e-8 else max(c * 0.005, 1e-6)
    ts = pd.Timestamp(row["time"])
    unreal = float(np.clip((c - entry_px) * position / CENT, -MAX_REWARD_CLIP, MAX_REWARD_CLIP))
    peak = float(max_unrealized)
    dd = float(np.clip((peak - unreal) / (atr / CENT + 1e-6), 0.0, 20.0))
    pk_n = float(np.clip(peak / (atr / CENT + 1e-6), -20.0, 20.0))
    pr = float(np.clip(unreal / (abs(peak) + 0.1), -2.0, 2.0))
    brng_n = float(np.clip((float(row["high"]) - float(row["low"])) / (atr + 1e-6), 0.0, 4.0))
    forced_eta = 0.0 if forced_exit_bar is None else float(np.clip((forced_exit_bar - i) / max(MAX_HOLD, 1), 0.0, 1.0))
    reg_id = int(row["reg_id"]) if pd.notna(row["reg_id"]) else -1
    blocked_flag = 1.0 if block_regs and reg_id in block_regs else 0.0
    sc = np.array([
        float(position),
        float(trade_dir),
        float(bars_held) / MAX_HOLD,
        float(max(MAX_HOLD - bars_held, 0)) / MAX_HOLD,
        float(unreal),
        float(atr / max(c, 1e-6) * 100.0),
        float(math.sin(2 * math.pi * (ts.hour + ts.minute / 60.0) / 24.0)),
        float(math.cos(2 * math.pi * (ts.hour + ts.minute / 60.0) / 24.0)),
        float((float(row["ema6"]) - float(row["sma24"])) / (atr + 1e-6)),
        pk_n,
        dd,
        float(min(consec_adv, 8)) / 8.0,
        float((float(row["ema6"]) - float(row["sma12"])) / (atr + 1e-6)),
        pr,
        brng_n,
        forced_eta,
        blocked_flag,
    ], dtype=np.float32)
    sc[~np.isfinite(sc)] = 0.0
    z = np.array([row.get(f"z{dim:02d}", np.nan) for dim in range(LATENT_DIM)], dtype=np.float32)
    z[~np.isfinite(z)] = 0.0
    fmt_id = int(row["fmt_id"]) if pd.notna(row["fmt_id"]) else -1
    pos_id = int(row["pos_id"]) if pd.notna(row["pos_id"]) else -1
    return z, fmt_id, pos_id, reg_id, sc


def find_forced_exit_bar(contract_df: pd.DataFrame, entry_bar: int, entry_direction: int) -> int:
    end_cap = min(len(contract_df) - 1, entry_bar + MAX_HOLD)
    for i in range(entry_bar, end_cap + 1):
        fast_signal = int(contract_df.iloc[i]["fast_signal"])
        if fast_signal == -entry_direction:
            return i
    return end_cap


def heuristic_agent_b_action(scalars: np.ndarray, bars_held: int):
    unreal = float(scalars[4]); d24 = abs(float(scalars[8])); dd = float(scalars[10]); consec_adv = float(scalars[11])
    d12 = abs(float(scalars[12])); ratio = float(scalars[13]); eta = float(scalars[15]); blocked = float(scalars[16])
    continue_score = 0.60*d24 + 0.35*d12 + 0.15*ratio + 0.12*np.clip(unreal/2.0, -2.0, 2.0) - 0.65*dd - 0.30*consec_adv - 0.60*blocked - 0.40*(1.0-eta)
    if bars_held < MIN_HOLD_BARS:
        continue_score += 0.25
    q_hold = float(continue_score)
    q_exit = float(-continue_score + 0.10 * dd + 0.10 * (1.0 - eta))
    action = 0 if q_hold >= q_exit else 1
    survival = float(1.0 / (1.0 + math.exp(-continue_score)))
    bars_left = float(np.clip(eta * (0.65 + 0.35 * survival), 0.0, 1.0))
    return action, q_hold, q_exit, survival, bars_left


def infer_agent_b(model, z, fmt_id, pos_id, reg_id, scalars, bars_held):
    if model is None:
        return heuristic_agent_b_action(scalars, bars_held)
    with torch.no_grad():
        z_t = torch.tensor(z[None, :], dtype=torch.float32, device=DEVICE)
        fi_t = torch.tensor([fmt_id], dtype=torch.long, device=DEVICE)
        pi_t = torch.tensor([pos_id], dtype=torch.long, device=DEVICE)
        ri_t = torch.tensor([reg_id], dtype=torch.long, device=DEVICE)
        sc_t = torch.tensor(scalars[None, :], dtype=torch.float32, device=DEVICE)
        q, surv, bars = model(z_t, fi_t, pi_t, ri_t, sc_t, return_aux=True)
        q_np = q.squeeze(0).detach().cpu().numpy()
        return int(np.argmax(q_np)), float(q_np[0]), float(q_np[1]), float(surv.item()), float(bars.item())


def explain_decision(action: int, q_hold: float, q_exit: float, sc: np.ndarray) -> str:
    bits = []
    dd = float(sc[10]); d24 = float(sc[8]); d12 = float(sc[12]); eta = float(sc[15]); blocked = float(sc[16]); unreal = float(sc[4]); consec_adv = float(sc[11])
    if action == 0:
        bits.append("HOLD is preferred")
        if abs(d24) > 0.35: bits.append("trend still has structure")
        if abs(d12) > 0.20: bits.append("fast spread supports continuation")
        if dd < 0.75: bits.append("drawdown is controlled")
        if eta > 0.30: bits.append("deadline is not close")
        if unreal > 0: bits.append("trade is positive")
    else:
        bits.append("EXIT is preferred")
        if dd >= 0.75: bits.append("drawdown has grown")
        if eta <= 0.30: bits.append("deadline is close")
        if blocked >= 0.5: bits.append("regime is blocked")
        if consec_adv >= 0.50: bits.append("adverse bars are accumulating")
        if q_exit > q_hold: bits.append("continuation value weakened")
    return "; ".join(bits) + "."


def make_trigger_label(trigger_bar: int, ts: pd.Timestamp, direction: int, reg_id: int) -> str:
    side = "LONG" if direction > 0 else "SHORT"
    return f"bar={trigger_bar} | {ts} | {side} | reg={reg_id}"


def simulate_trade(contract_df: pd.DataFrame, trigger_local_idx: int, model, block_regs: set) -> PlaybackTrade:
    contract_df = contract_df.sort_values("bar_idx").reset_index(drop=True)
    trig = int(trigger_local_idx)
    trigger_row = contract_df.iloc[trig]
    entry_direction = int(np.sign(trigger_row["entry_direction"]))
    reg_id = int(trigger_row["reg_id"]) if pd.notna(trigger_row["reg_id"]) else -1
    blocked_regime = reg_id in block_regs if reg_id >= 0 else False
    if trig + 1 >= len(contract_df):
        raise ValueError("Trigger bar is too close to the end of the contract.")
    entry_bar = trig + 1
    entry_row = contract_df.iloc[entry_bar]
    entry_price = float(entry_row["close"])
    entry_time = pd.Timestamp(entry_row["time"])
    forced_exit_bar = find_forced_exit_bar(contract_df, entry_bar, entry_direction)
    decisions = []
    position = 1 if entry_direction > 0 else -1
    bars_held = 0
    max_unrealized = 0.0
    consec_adv = 0
    prev_close = entry_price
    exit_bar = forced_exit_bar
    exit_reason = "Forced exit"
    exit_price = float(contract_df.iloc[exit_bar]["close"])
    for i in range(entry_bar, min(forced_exit_bar, len(contract_df) - 1) + 1):
        row = contract_df.iloc[i]
        c = float(row["close"])
        unreal = (c - entry_price) * position / CENT
        max_unrealized = max(max_unrealized, unreal)
        if i > entry_bar:
            bar_mtm = (c - prev_close) * position / CENT
            consec_adv = consec_adv + 1 if bar_mtm < 0 else 0
        prev_close = c
        z, fmt_id, pos_id, reg_id_i, sc = make_state_B(
            contract_df, i, position, bars_held, entry_price, position,
            max_unrealized=max_unrealized, consec_adv=consec_adv,
            forced_exit_bar=forced_exit_bar, block_regs=block_regs
        )
        action, q_hold, q_exit, survival, bars_left = infer_agent_b(model, z, fmt_id, pos_id, reg_id_i, sc, bars_held)
        if bars_held < MIN_HOLD_BARS:
            action = 0
        if i >= forced_exit_bar:
            action = 1
            exit_reason = "Forced exit: opposite fast signal or max-hold cap"
        reason = explain_decision(action, q_hold, q_exit, sc)
        decisions.append(TradeDecision(
            bar_idx=i,
            timestamp=pd.Timestamp(row["time"]),
            action="HOLD" if action == 0 else "EXIT",
            q_hold=q_hold,
            q_exit=q_exit,
            survival=survival,
            bars_left_score=bars_left,
            unrealized_pnl_cents=float(unreal),
            bars_held=bars_held,
            reason=reason,
        ))
        if action == 1:
            exit_bar = i
            exit_price = c
            if i < forced_exit_bar:
                exit_reason = "Agent B voluntary exit"
            break
        bars_held += 1
    realized_pnl = (exit_price - entry_price) * position / CENT - TRANSACTION_COST
    return PlaybackTrade(
        contract=str(trigger_row["contract"]),
        trigger_bar=int(trigger_row["bar_idx"]),
        entry_bar=entry_bar,
        entry_time=entry_time,
        entry_direction=entry_direction,
        entry_price=entry_price,
        exit_bar=exit_bar,
        exit_time=pd.Timestamp(contract_df.iloc[exit_bar]["time"]),
        exit_price=exit_price,
        realized_pnl_cents=float(realized_pnl),
        exit_reason=exit_reason,
        decisions=decisions,
        blocked_regime=blocked_regime,
        fmt_id=int(trigger_row["fmt_id"]) if pd.notna(trigger_row["fmt_id"]) else -1,
        pos_id=int(trigger_row["pos_id"]) if pd.notna(trigger_row["pos_id"]) else -1,
        reg_id=reg_id,
        trigger_label=make_trigger_label(int(trigger_row["bar_idx"]), pd.Timestamp(trigger_row["time"]), entry_direction, reg_id),
    )


def build_trade_index(df: pd.DataFrame, block_regs: set) -> pd.DataFrame:
    rows = []
    grouped = dict(tuple(df.sort_values(["contract", "bar_idx"]).groupby("contract", sort=False)))
    for contract, grp in grouped.items():
        grp = grp.reset_index(drop=True)
        entries = grp[grp["confirmed_entry"].fillna(False)].copy()
        for _, row in entries.iterrows():
            trig = int(row["bar_idx"])
            local_idx = int(np.where(grp["bar_idx"].values == trig)[0][0])
            if local_idx + 1 >= len(grp):
                continue
            reg = int(row["reg_id"]) if pd.notna(row["reg_id"]) else -1
            label = make_trigger_label(trig, pd.Timestamp(row["time"]), int(row["entry_direction"]), reg)
            rows.append({
                "contract": str(contract),
                "trigger_bar": trig,
                "local_idx": local_idx,
                "entry_direction": int(row["entry_direction"]),
                "time": pd.Timestamp(row["time"]),
                "reg_id": reg,
                "label": label,
            })
    return pd.DataFrame(rows).sort_values(["contract", "trigger_bar"]).reset_index(drop=True)


def progressive_series(values: pd.Series, visible_end: int, start_idx: int = 0) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    vals = values.tolist()
    for idx in range(start_idx, len(vals)):
        val = vals[idx]
        if idx <= visible_end and pd.notna(val):
            out.append(float(val))
        else:
            out.append(None)
    return out


def window_bounds(n_rows: int, current_bar_idx: int, lookback: int, lookahead: int) -> Tuple[int, int]:
    start = max(0, current_bar_idx - max(lookback, 1))
    end = min(n_rows - 1, current_bar_idx + max(lookahead, 0))
    return start, end


def make_rolling_figure(contract_df: pd.DataFrame, trade: PlaybackTrade, current_bar_idx: int, shown_decisions: List[TradeDecision], lookback: int, lookahead: int, historical_trades: Optional[List[PlaybackTrade]] = None) -> go.Figure:
    fig = go.Figure()
    start_idx, end_idx = window_bounds(len(contract_df), current_bar_idx, lookback, lookahead)
    window_df = contract_df.iloc[start_idx:end_idx + 1].copy()
    vis_end = current_bar_idx
    visible_df = contract_df.iloc[start_idx:vis_end + 1].copy()

    fig.add_trace(go.Candlestick(
        x=visible_df["time"],
        open=visible_df["open"],
        high=visible_df["high"],
        low=visible_df["low"],
        close=visible_df["close"],
        name="Price",
        increasing_line_width=1,
        decreasing_line_width=1,
    ))

    fig.add_trace(go.Scatter(
        x=window_df["time"],
        y=progressive_series(contract_df["ema6"], vis_end, start_idx),
        mode="lines",
        name="EMA6",
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=window_df["time"],
        y=progressive_series(contract_df["sma12"], vis_end, start_idx),
        mode="lines",
        name="SMA12",
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=window_df["time"],
        y=progressive_series(contract_df["sma24"], vis_end, start_idx),
        mode="lines",
        name="SMA24",
        connectgaps=False,
    ))

    entry_ts = pd.Timestamp(contract_df.iloc[trade.entry_bar]["time"])
    exit_ts = pd.Timestamp(contract_df.iloc[trade.exit_bar]["time"])
    current_ts = pd.Timestamp(contract_df.iloc[current_bar_idx]["time"])

    if start_idx <= trade.entry_bar <= end_idx:
        fig.add_vline(x=entry_ts, line_width=1, line_dash="dot", line_color="green")
        fig.add_annotation(x=entry_ts, y=1, yref="paper", text="Agent A entry", showarrow=False, yshift=12)
    if start_idx <= trade.exit_bar <= end_idx:
        fig.add_vline(x=exit_ts, line_width=1, line_dash="dot", line_color="red")
        fig.add_annotation(x=exit_ts, y=1, yref="paper", text="Exit zone", showarrow=False, yshift=12)
    fig.add_vline(x=current_ts, line_width=1, line_dash="dash", line_color="black")
    fig.add_annotation(x=current_ts, y=1, yref="paper", text="Now", showarrow=False, yshift=12)

    if start_idx <= trade.entry_bar <= vis_end:
        fig.add_trace(go.Scatter(
            x=[entry_ts],
            y=[trade.entry_price],
            mode="markers+text",
            name="Agent A",
            text=["A"],
            textposition="top center",
            marker=dict(symbol="triangle-up" if trade.entry_direction > 0 else "triangle-down", size=14),
        ))

        historical_trades = historical_trades or []

    past_a_x, past_a_y, past_a_text = [], [], []
    past_b_x, past_b_y, past_b_text = [], [], []
    for past_trade in historical_trades:
        if start_idx <= past_trade.entry_bar <= end_idx:
            past_a_x.append(pd.Timestamp(contract_df.iloc[past_trade.entry_bar]["time"]))
            past_a_y.append(float(past_trade.entry_price))
            past_a_text.append("A")
        for dec in past_trade.decisions:
            if start_idx <= dec.bar_idx <= end_idx:
                row = contract_df.iloc[dec.bar_idx]
                past_b_x.append(pd.Timestamp(row["time"]))
                past_b_y.append(float(row["close"]))
                past_b_text.append("B EXIT" if dec.action == "EXIT" else "B HOLD")

    if past_a_x:
        fig.add_trace(go.Scatter(
            x=past_a_x,
            y=past_a_y,
            mode="markers+text",
            name="Past Agent A",
            text=past_a_text,
            textposition="top center",
            marker=dict(symbol="diamond", size=10),
        ))

    if past_b_x:
        fig.add_trace(go.Scatter(
            x=past_b_x,
            y=past_b_y,
            mode="markers+text",
            name="Past Agent B",
            text=past_b_text,
            textposition="bottom center",
            marker=dict(size=7, symbol="x"),
        ))

    if shown_decisions:
        xs, ys, texts = [], [], []
        for d in shown_decisions:
            if start_idx <= d.bar_idx <= vis_end:
                row = contract_df.iloc[d.bar_idx]
                xs.append(pd.Timestamp(row["time"]))
                ys.append(float(row["close"]))
                texts.append("B EXIT" if d.action == "EXIT" else "B HOLD")
        if xs:
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name="Agent B decisions",
                text=texts,
                textposition="bottom center",
                marker=dict(size=9, symbol="circle"),
            ))

    fig.update_layout(
        title=f"{trade.contract} — rolling market playback",
        xaxis_title="Time",
        yaxis_title="Price",
        height=760,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=0),
        uirevision="rolling-playback",
    )
    fig.update_xaxes(range=[window_df["time"].iloc[0], window_df["time"].iloc[-1]])
    return fig


class PlaybackApp:
    def __init__(self) -> None:
        if not os.path.exists(PARQUET_PATH):
            raise FileNotFoundError(f"Parquet not found: {PARQUET_PATH}")
        self.df = pd.read_parquet(PARQUET_PATH).sort_values(["contract", "bar_idx"]).reset_index(drop=True)
        self.df["time"] = pd.to_datetime(self.df["time"])
        self.allow_regs, self.block_regs = compute_regime_filter(self.df)
        self.trade_index = build_trade_index(self.df, self.block_regs)
        self.contracts = sorted(self.df["contract"].astype(str).unique().tolist())
        self.by_contract = {
            contract: grp.sort_values("bar_idx").reset_index(drop=True)
            for contract, grp in self.df.groupby("contract", sort=False)
        }
        checkpoint_path = detect_checkpoint_path()
        self.model, self.model_message = load_qnet(checkpoint_path)
        self.trade_cache: Dict[Tuple[str, str], PlaybackTrade] = {}

    def contract_trade_choices(self, contract: str) -> List[str]:
        sub = self.trade_index[self.trade_index["contract"] == contract].copy()
        return sub["label"].tolist() or ["No trades found"]

    def parse_trade_choice(self, contract: str, choice: str) -> int:
        sub = self.trade_index[self.trade_index["contract"] == contract]
        if len(sub) == 0:
            raise ValueError(f"No trades found for contract {contract}")
        if not choice or choice not in set(sub["label"].tolist()):
            return int(sub.iloc[0]["local_idx"])
        row = sub[sub["label"] == choice]
        return int(row.iloc[0]["local_idx"])

    def simulate(self, contract: str, trade_choice: str):
        cache_key = (contract, trade_choice)
        contract_df = self.by_contract[contract]
        if cache_key not in self.trade_cache:
            trigger_local_idx = self.parse_trade_choice(contract, trade_choice)
            self.trade_cache[cache_key] = simulate_trade(contract_df, trigger_local_idx, self.model, self.block_regs)
        return contract_df, self.trade_cache[cache_key]

    def all_trade_choices(self, contract: str) -> List[str]:
        return self.contract_trade_choices(contract)

    def render_state(self, contract: str, trade_choice: str, step_idx: int, lookback: int, lookahead: int, historical_labels: Optional[List[str]] = None):
        contract_df, trade = self.simulate(contract, trade_choice)
        n_steps = len(trade.decisions)
        step_idx = int(np.clip(step_idx, 0, max(n_steps - 1, 0)))
        current = trade.decisions[step_idx]
        shown_decisions = trade.decisions[:step_idx + 1]
        historical_labels = historical_labels or []
        historical_trades = []
        for label in historical_labels:
            if label != trade_choice:
                _, hist_trade = self.simulate(contract, label)
                historical_trades.append(hist_trade)
        fig = make_rolling_figure(contract_df, trade, current.bar_idx, shown_decisions, int(lookback), int(lookahead), historical_trades=historical_trades)

        agent_a_html = f"""
        <div style='padding:12px;border:1px solid #ddd;border-radius:12px;min-height:220px;'>
          <h3>Agent A</h3>
          <p><b>Trigger:</b> {trade.trigger_label}</p>
          <p><b>Direction:</b> {'LONG' if trade.entry_direction > 0 else 'SHORT'}</p>
          <p><b>Entry time:</b> {trade.entry_time}</p>
          <p><b>Entry price:</b> {trade.entry_price:.5f}</p>
          <p><b>Blocked regime:</b> {trade.blocked_regime}</p>
          <p><b>fmt / pos / reg:</b> {trade.fmt_id} / {trade.pos_id} / {trade.reg_id}</p>
        </div>
        """

        agent_b_html = f"""
        <div style='padding:12px;border:1px solid #ddd;border-radius:12px;min-height:220px;'>
          <h3>Agent B</h3>
          <p><b>Trigger:</b> {trade.trigger_label}</p>
          <p><b>Playback candle:</b> {step_idx + 1} / {n_steps}</p>
          <p><b>Action now:</b> {current.action}</p>
          <p><b>Bar:</b> {current.bar_idx}</p>
          <p><b>Time:</b> {current.timestamp}</p>
          <p><b>Q(HOLD):</b> {current.q_hold:.4f}</p>
          <p><b>Q(EXIT):</b> {current.q_exit:.4f}</p>
          <p><b>Survival:</b> {current.survival:.3f}</p>
          <p><b>Bars-left:</b> {current.bars_left_score:.3f}</p>
          <p><b>Unrealized pnl:</b> {current.unrealized_pnl_cents:.3f} cents</p>
          <p><b>Reason:</b> {current.reason}</p>
        </div>
        """

        summary_html = f"""
        <div style='padding:12px;border:1px solid #ddd;border-radius:12px;min-height:220px;'>
          <h3>Playback Summary</h3>
          <p><b>Contract:</b> {trade.contract}</p>
          <p><b>Trigger:</b> {trade.trigger_label}</p>
          <p><b>Visible bar time:</b> {current.timestamp}</p>
          <p><b>Exit reason:</b> {trade.exit_reason}</p>
          <p><b>Final realized pnl:</b> {trade.realized_pnl_cents:.3f} cents</p>
          <p><b>Model:</b> {self.model_message}</p>
        </div>
        """

        max_unreal = max([d.unrealized_pnl_cents for d in shown_decisions] or [0.0])
        consec_adv = 0
        for d in reversed(shown_decisions):
            if d.unrealized_pnl_cents < 0:
                consec_adv += 1
            else:
                break

        z, _, _, _, sc = make_state_B(
            contract_df, current.bar_idx, trade.entry_direction, current.bars_held, trade.entry_price, trade.entry_direction,
            max_unrealized=max_unreal, consec_adv=consec_adv, forced_exit_bar=trade.exit_bar, block_regs=self.block_regs
        )

        scalar_names = [
            "position", "trade_dir", "bars_held_frac", "bars_remaining_frac", "unrealized_pnl",
            "atr_pct", "tod_sin", "tod_cos", "d24_atr", "peak_unreal_norm", "drawdown_norm",
            "consec_adv_norm", "d12_atr", "unreal_peak_ratio", "bar_rng_norm", "forced_exit_eta",
            "blocked_regime_flag"
        ]
        state_df = pd.DataFrame({"feature": scalar_names, "value": [float(v) for v in sc]})
        z_df = pd.DataFrame({"latent_dim": [f"z{d:02d}" for d in range(LATENT_DIM)], "value": [float(v) for v in z]})
        ledger_df = pd.DataFrame([{
            "step": ix + 1,
            "bar_idx": d.bar_idx,
            "time": str(d.timestamp),
            "action": d.action,
            "q_hold": round(d.q_hold, 6),
            "q_exit": round(d.q_exit, 6),
            "unrealized_pnl_cents": round(d.unrealized_pnl_cents, 6),
            "reason": d.reason,
        } for ix, d in enumerate(shown_decisions)])

        status = f"**{trade.trigger_label}** — candle {step_idx + 1}/{n_steps}. Window rolls forward without compressing candles."
        return fig, agent_a_html, agent_b_html, summary_html, state_df, z_df, ledger_df, gr.update(value=step_idx, maximum=max(n_steps - 1, 0)), status

    def autoplay_single(self, contract: str, trade_choice: str, start_step: int, speed_seconds: float, lookback: int, lookahead: int):
        _, trade = self.simulate(contract, trade_choice)
        total_steps = len(trade.decisions)
        delay = max(0.1, float(speed_seconds))
        for step in range(int(start_step), total_steps):
            yield self.render_state(contract, trade_choice, step, lookback, lookahead)
            time.sleep(delay)

    def autoplay_all(self, contract: str, speed_seconds: float, lookback: int, lookahead: int):
        delay = max(0.1, float(speed_seconds))
        choices = self.all_trade_choices(contract)
        if not choices or choices == ["No trades found"]:
            raise ValueError(f"No trades found for contract {contract}")
        completed_labels: List[str] = []
        for choice in choices:
            _, trade = self.simulate(contract, choice)
            total_steps = len(trade.decisions)
            for step in range(total_steps):
                result = self.render_state(contract, choice, step, lookback, lookahead, historical_labels=completed_labels)
                fig, a, b, s, state_df, z_df, ledger_df, slider_update, status = result
                yield fig, a, b, s, state_df, z_df, ledger_df, slider_update, status, gr.update(value=choice)
                time.sleep(delay)
            completed_labels.append(choice)


def build_demo(app: PlaybackApp):
    with gr.Blocks(title="TFL-Collie Rolling Playback", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# TFL-Collie Rolling Playback\n"
            "This version keeps candle width natural by using a rolling window. You can play one trigger or all triggers in sequence."
        )
        status_md = gr.Markdown(f"**Loaded.** {app.model_message}")

        with gr.Row():
            contract_dd = gr.Dropdown(choices=app.contracts, value=app.contracts[0], label="Contract")
            first_choices = app.contract_trade_choices(app.contracts[0])
            trade_dd = gr.Dropdown(choices=first_choices, value=first_choices[0], label="Single trigger")
            step_slider = gr.Slider(minimum=0, maximum=0, value=0, step=1, label="Playback step")
            speed_slider = gr.Slider(minimum=0.2, maximum=3.0, value=1.0, step=0.1, label="Seconds per candle")

        with gr.Row():
            lookback_slider = gr.Slider(minimum=40, maximum=300, value=DEFAULT_LOOKBACK, step=10, label="Bars on screen behind current bar")
            lookahead_slider = gr.Slider(minimum=0, maximum=80, value=DEFAULT_LOOKAHEAD, step=5, label="Bars on screen ahead of current bar")

        with gr.Row():
            render_btn = gr.Button("Render current step", variant="secondary")
            play_one_btn = gr.Button("Play selected trigger", variant="primary")
            play_all_btn = gr.Button("Play all triggers in sequence", variant="primary")

        plot = gr.Plot(label="Rolling market playback")

        with gr.Row():
            agent_a_box = gr.HTML()
            agent_b_box = gr.HTML()
            summary_box = gr.HTML()

        with gr.Row():
            state_tbl = gr.Dataframe(label="Current Agent B scalar state", interactive=False)
            z_tbl = gr.Dataframe(label="Current VAE latent vector z", interactive=False)

        ledger_tbl = gr.Dataframe(label="Decision ledger so far", interactive=False)

        def update_trade_choices(contract):
            choices = app.contract_trade_choices(contract)
            return gr.update(choices=choices, value=choices[0] if choices else None), gr.update(value=0)

        contract_dd.change(update_trade_choices, inputs=contract_dd, outputs=[trade_dd, step_slider])

        render_outputs = [plot, agent_a_box, agent_b_box, summary_box, state_tbl, z_tbl, ledger_tbl, step_slider, status_md]

        trade_dd.change(app.render_state, inputs=[contract_dd, trade_dd, step_slider, lookback_slider, lookahead_slider], outputs=render_outputs)
        render_btn.click(app.render_state, inputs=[contract_dd, trade_dd, step_slider, lookback_slider, lookahead_slider], outputs=render_outputs)
        step_slider.change(app.render_state, inputs=[contract_dd, trade_dd, step_slider, lookback_slider, lookahead_slider], outputs=render_outputs)
        play_one_btn.click(app.autoplay_single, inputs=[contract_dd, trade_dd, step_slider, speed_slider, lookback_slider, lookahead_slider], outputs=render_outputs)
        play_all_btn.click(
            app.autoplay_all,
            inputs=[contract_dd, speed_slider, lookback_slider, lookahead_slider],
            outputs=[plot, agent_a_box, agent_b_box, summary_box, state_tbl, z_tbl, ledger_tbl, step_slider, status_md, trade_dd],
        )

        gr.Markdown(
            "What changed:\n"
            "- the screen rolls with the market instead of compressing candles\n"
            "- you can keep a selected single trigger\n"
            "- or play all triggers in sequence for the chosen contract\n"
            "- moving averages still reveal only when the corresponding bars become visible"
        )
    return demo


app = PlaybackApp()
demo = build_demo(app)
demo.queue()
demo.launch(share=True, debug=True)
