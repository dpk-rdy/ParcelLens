"""Chart generation. Every figure is written as both SVG and PNG."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1A1D1A"
MUTED = "#6E736C"
RULE = "#C9CDC6"
PRIMARY = "#1F4E79"
LEAK = "#8E2436"
SAVE = "#3F6E4E"
NEUTRAL = "#9AA39C"
PAPER = "none"

SERIES = [PRIMARY, LEAK, SAVE, "#8C6D3F", "#5C7C99", NEUTRAL]


def _style(ax, xlabel=None, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    return ax


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(f"{outdir}/{name}.{ext}", format=ext, dpi=160,
                    transparent=True, bbox_inches="tight")
    plt.close(fig)
    with open(f"{outdir}/{name}.svg") as fh:
        return fh.read()


def recovery_by_exception(summary: pd.DataFrame, outdir) -> str:
    d = summary.sort_values("recovery_usd")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = [SAVE if t == "service_failure_refund" else LEAK
              for t in d["exception_type"]]
    ax.barh(d["exception_type"].str.replace("_", " "), d["recovery_usd"],
            color=colors, height=0.62)
    for y, (v, n) in enumerate(zip(d["recovery_usd"], d["lines"])):
        ax.text(v + d["recovery_usd"].max() * 0.012, y, f"${v:,.0f} on {n:,} lines",
                va="center", fontsize=8.5, color=MUTED)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.55)
    ax.set_xlim(0, d["recovery_usd"].max() * 1.34)
    ax.xaxis.set_major_formatter(lambda v, p: f"${v/1000:,.0f}k")
    _style(ax, xlabel="Recoverable dollars")
    return _save(fig, outdir, "recovery_by_exception")


def monthly_trend(ship: pd.DataFrame, audited: pd.DataFrame, outdir) -> str:
    spend = ship.groupby("month")["expected_total"].sum()
    rec = (audited.assign(m=audited["month"])
           .groupby("m")[["recovery_usd", "gsr_refund_usd"]].sum().sum(axis=1))
    months = list(range(1, 13))
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 4.8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.fill_between(months, spend.reindex(months).values / 1000,
                     color=PRIMARY, alpha=0.16)
    ax1.plot(months, spend.reindex(months).values / 1000, color=PRIMARY, lw=1.8)
    ax1.yaxis.set_major_formatter(lambda v, p: f"${v:,.0f}k")
    _style(ax1, ylabel="Transportation spend")

    leak_rate = (rec.reindex(months).values / spend.reindex(months).values) * 100
    ax2.bar(months, leak_rate, color=LEAK, width=0.55)
    ax2.yaxis.set_major_formatter(lambda v, p: f"{v:.1f}%")
    _style(ax2, ylabel="Billing leakage")
    ax2.set_xticks(months)
    ax2.set_xticklabels(labels)
    return _save(fig, outdir, "monthly_trend")


def carrier_costs(sc: pd.DataFrame, outdir) -> str:
    d = sc.sort_values("cost_per_package", ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.3))

    base = d["base_spend"] / d["shipments"]
    acc = d["accessorial_spend"] / d["shipments"]
    fuel = d["fuel_spend"] / d["shipments"]
    y = np.arange(len(d))
    ax1.barh(y, base, color=PRIMARY, height=0.6, label="Transportation")
    ax1.barh(y, acc, left=base, color="#8C6D3F", height=0.6, label="Accessorials")
    ax1.barh(y, fuel, left=base + acc, color=NEUTRAL, height=0.6, label="Fuel")
    ax1.set_yticks(y)
    ax1.set_yticklabels(d["carrier"])
    ax1.xaxis.set_major_formatter(lambda v, p: f"${v:,.0f}")
    _style(ax1, xlabel="Cost per package")
    ax1.grid(axis="y", visible=False)
    ax1.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.55)
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, -0.20))

    ax2.scatter(d["accessorial_share"] * 100, d["on_time"] * 100,
                s=d["spend"] / d["spend"].max() * 380 + 45,
                color=PRIMARY, alpha=0.62, edgecolor="none")
    for k, (_, r) in enumerate(d.iterrows()):
        ax2.annotate(r["carrier"], (r["accessorial_share"] * 100, r["on_time"] * 100),
                     textcoords="offset points", xytext=(0, 15 if k % 2 == 0 else -22),
                     ha="center", fontsize=8, color=MUTED)
    ax2.margins(x=0.18, y=0.22)
    ax2.xaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    ax2.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    _style(ax2, xlabel="Accessorials as share of spend", ylabel="On-time delivery")
    return _save(fig, outdir, "carrier_costs")


def rateshop_savings(uncon: pd.DataFrame, con: pd.DataFrame, outdir) -> str:
    a = uncon.groupby("client")["savings"].sum()
    b = con.groupby("client")["savings"].sum()
    d = pd.DataFrame({"unconstrained": a, "constrained": b}).sort_values("unconstrained")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.barh(y + 0.19, d["unconstrained"] / 1000, height=0.36,
            color=NEUTRAL, label="Least cost, no capacity limit")
    ax.barh(y - 0.19, d["constrained"] / 1000, height=0.36,
            color=SAVE, label="Executable plan within capacity")
    ax.set_yticks(y)
    ax.set_yticklabels(d.index)
    ax.xaxis.set_major_formatter(lambda v, p: f"${v:,.0f}k")
    _style(ax, xlabel="Annual savings at constant delivery promise")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.55)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return _save(fig, outdir, "rateshop_savings")


def dim_exposure(pack_detail: pd.DataFrame, pack_client: pd.DataFrame, outdir) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9))
    ax1.hist(pack_detail["cube_utilization"] * 100, bins=28,
             color=PRIMARY, alpha=0.82, edgecolor="none")
    ax1.axvline(pack_detail["cube_utilization"].mean() * 100, color=LEAK,
                lw=1.4, ls="--")
    ax1.annotate(f"mean {pack_detail['cube_utilization'].mean()*100:.0f}%",
                 (pack_detail["cube_utilization"].mean() * 100, ax1.get_ylim()[1] * 0.92),
                 textcoords="offset points", xytext=(7, 0), fontsize=8.5, color=LEAK)
    ax1.xaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    ax1.yaxis.set_major_formatter(lambda v, p: f"{v/1000:,.0f}k")
    _style(ax1, xlabel="Box cube filled by the item", ylabel="Shipments")

    d = pack_client.sort_values("dim_billed_rate")
    y = np.arange(len(d))
    ax2.barh(y, d["dim_billed_rate"] * 100, color="#8C6D3F", height=0.6)
    for i, (r, s) in enumerate(zip(d["dim_billed_rate"], d["packaging_savings"])):
        ax2.text(r * 100 + 1.4, i, f"${s/1000:,.0f}k", va="center",
                 fontsize=8.5, color=MUTED)
    ax2.set_yticks(y)
    ax2.set_yticklabels(d["client"], fontsize=8.5)
    ax2.set_xlim(0, min(105, d["dim_billed_rate"].max() * 100 * 1.32))
    ax2.xaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    _style(ax2, xlabel="Shipments billed on dimensional weight")
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.55)
    return _save(fig, outdir, "dim_exposure")


def gri_levers(levers: pd.DataFrame, summary: dict, outdir) -> str:
    d = levers.sort_values("impact_usd")
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.barh(d["lever"], d["impact_usd"] / 1000, color=LEAK, height=0.58)
    for i, (v, sh) in enumerate(zip(d["impact_usd"], d["share_of_increase"])):
        ax.text(v / 1000 * 1.02, i, f"${v/1000:,.0f}k, {sh:.0%} of the increase",
                va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, d["impact_usd"].max() / 1000 * 1.38)
    ax.xaxis.set_major_formatter(lambda v, p: f"${v:,.0f}k")
    _style(ax, xlabel="Added annual cost on unchanged volume")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.55)
    return _save(fig, outdir, "gri_levers")
