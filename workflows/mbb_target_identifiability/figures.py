"""Present observed, reconstructed and evaluator-only surfaces separately."""

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree

from workflows.reconstruction_s0010.dense import IDENTITIES, nearest


COLORS = ("#1c77ad", "#9c49a7", "#e58f1e", "#18836c")
METHODS = ("B0", "B1", "B2", "B3")
METHOD_LABELS = {"B0": "Oracle raw", "B1": "U-Net raw", "B2": "Oracle fused",
                 "B3": "U-Net fused", "A0": "Oracle sparse", "A1": "U-Net sparse"}


def pct(stats):
    return 100 * stats["covered"] / stats["total"] if stats["total"] else np.nan


def finish(figure, path):
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def coverage_plot(summary, out):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    x = np.arange(4)
    labels = ["Global", "5 mm", "10 mm", "20 mm"]
    for i, method in enumerate(METHODS):
        row = summary["pooled"][method]
        keys = [row["global"], *[row["local"][str(r)]["all"]["coverage_eligible_visible"] for r in (5,10,20)]]
        axes[0].bar(x + (i-1.5)*.18, [pct(v) for v in keys], width=.18,
                    color=COLORS[i], label=METHOD_LABELS[method])
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0,110),
                title="Reconstructed eligible Oracle-visible samples", ylabel="Covered within 1.5 mm (%)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for i, name in enumerate(("RIGHT_L5_TP", "RIGHT_L5_SAP")):
        row = summary["pooled"]["B0"]["local"]
        evidence = [row[str(r)][name]["full_ct_boundary_observed_all_visible"] for r in (5,10,20)]
        position = np.arange(3) + (i-.5)*.32
        axes[1].bar(position, [pct(v) for v in evidence], color=COLORS[i+2], width=.32,
                    label=name.replace("_"," "))
        for p, item in zip(position,evidence):
            val=pct(item)
            axes[1].text(p,(val if np.isfinite(val) else 0)+2,
                         f"{item['covered']}/{item['total']}",ha="center",fontsize=8)
    axes[1].set(xticks=np.arange(3),xticklabels=labels[1:],ylim=(0,110),
                title="CT boundary near an actual observed sample")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Local surface evidence around the provisional RIGHT L5 proxy · four sessions")
    finish(figure,out/"coverage.png")


def set_axes(ax, center, radius):
    ax.set_box_aspect((1,1,1))
    ax.set_xlim(*(center[0] + np.array([-radius,radius])))
    ax.set_ylim(*(center[1] + np.array([-radius,radius])))
    ax.set_zlim(*(center[2] + np.array([-radius,radius])))
    ax.view_init(22,-59)
    ax.set_xlabel("X", labelpad=1)
    ax.set_ylabel("Y", labelpad=1)
    ax.set_zlabel("Z", labelpad=1)
    ax.tick_params(labelsize=7,pad=1)


def scatter_cloud(ax, points, codes):
    for i,color in enumerate(COLORS):
        selected=points[codes==i]
        if len(selected):
            ax.scatter(*selected.T,s=8,alpha=.75,depthshade=False,color=color)


def geometry_plot(data, methods, seed, out, title):
    figure, axes = plt.subplots(2,2,figsize=(11,9),subplot_kw={"projection":"3d"},layout="constrained")
    figure.get_layout_engine().set(h_pad=.32, w_pad=.27)
    points = data["midpoint"]
    tree = cKDTree(points)
    center = points.mean(0)
    for m,ax in zip(methods,axes.flat):
        xyz, code = data[m+"_xyz"], data[m+"_code"]
        local = tree.query(xyz)[0] <= 20
        scatter_cloud(ax,xyz[local],code[local])
        set_axes(ax,center,22)
        ax.set_title(f"{METHOD_LABELS[m]} · {int(local.sum())} local points")
    figure.legend(handles=[Line2D([0],[0],color=color,label=k.replace("_"," "),marker="o",ls="")
                           for k,color in zip(IDENTITIES,COLORS)],loc="outside lower center",ncols=4,frameon=False)
    figure.suptitle(f"{title} · seed {seed} · 20 mm\nRobot base coordinates (mm)")
    finish(figure,out)


def reference_plot(data, seed, out):
    figure, axes = plt.subplots(1,2,figsize=(11,5.8),subplot_kw={"projection":"3d"},layout="constrained")
    figure.get_layout_engine().set(w_pad=.27)
    tree = cKDTree(data["midpoint"])
    center = data["midpoint"].mean(0)
    visible, codes = data["visible_xyz"], data["visible_code"]
    in_local = tree.query(visible)[0] <= 20
    for i, identity in enumerate(IDENTITIES):
        pts=data["surface_"+identity]
        pts=pts[tree.query(pts)[0] <= 20]
        got=nearest(pts,visible[in_local & (codes==i)]) <= 1.5
        if np.any(~got):
            axes[0].scatter(*pts[~got].T,s=6,alpha=.28,color="#77848f",depthshade=False)
        if np.any(got):
            axes[0].scatter(*pts[got].T,s=10,alpha=.8,color=COLORS[i],depthshade=False)
    axes[0].scatter(*data["sap_core"].T,s=24,color="#b33240",marker="x",label="Strict SAP core")
    axes[0].scatter(*data["tp_core"].T,s=24,color="#354890",marker="x",label="Strict TP core")
    axes[0].legend(frameon=False,fontsize=8)
    scatter_cloud(axes[1],visible[in_local],codes[in_local])
    for ax,title in zip(axes,("Evaluator CT reference · grey outside measured support",
                              "Observed Oracle interface points")):
        set_axes(ax,center,22)
        ax.set_title(title)
    figure.suptitle(f"Observed and unobserved parts of the reference · seed {seed}\nRobot base coordinates (mm)")
    finish(figure,out/"reference_visibility.png")


def local_errors(outputs, out):
    figure,ax=plt.subplots(figsize=(8.4,4.4),layout="constrained")
    for method,color in zip(METHODS,COLORS):
        pieces=[]
        for d in outputs:
            tree=cKDTree(d["midpoint"])
            pts, code=d[method+"_xyz"], d[method+"_code"]
            local=tree.query(pts)[0] <= 10
            for i,k in enumerate(IDENTITIES):
                ref=d["surface_"+k]
                ref=ref[tree.query(ref)[0] <= 10]
                pieces.extend(nearest(pts[local & (code==i)],ref))
        errors=np.asarray(pieces)
        finite=np.sort(errors[np.isfinite(errors)])
        if len(finite):
            ax.step(finite,np.arange(1,len(finite)+1)/len(errors),color=color,
                    label=f"{METHOD_LABELS[method]} ({len(errors)} points)")
    ax.axvline(1.5,lw=1,ls=":",color=".4",label="1.5 mm")
    ax.set(xlabel="Distance to nearest same-class CT boundary voxel center (mm)",
           ylabel="Fraction of output points",ylim=(0,1.02),title="Local geometric error · radius 10 mm")
    ax.legend(frameon=False)
    finish(figure,out/"local_errors.png")


def missing_view_plot(summary,out):
    fig, axes=plt.subplots(2,2,figsize=(11,7),layout="constrained",squeeze=False)
    for session, ax in zip(summary["sessions"],axes.flat):
        for i,m in enumerate(METHODS):
            available=session["methods"][m]["local"]["20"]["leave_out"]
            base=available["baseline"]["observed_feature"]["closest_visible_pair_mm"]
            entries=[r["observed_feature"]["closest_visible_pair_mm"] for k,r in available.items()
                     if k.startswith("leave_frame")]
            values=[v for v in entries if v is not None]
            if values:
                ax.scatter(np.full(len(values),i),values,s=12,color=COLORS[i],alpha=.4)
            if base is not None:
                ax.scatter(i,base,marker="_",s=160,color="black")
            for family,marker in (("longitudinal","^"),("transverse","s")):
                candidate=available.get("leave_family_"+family, {}).get("observed_feature",{})
                value=candidate.get("closest_visible_pair_mm")
                if value is not None:
                    ax.scatter(i,value,marker=marker,s=44,facecolors="none",edgecolors=COLORS[i])
                elif candidate:
                    ax.text(i,.07 if family=="longitudinal" else .13,"L absent" if family=="longitudinal" else "T absent",
                            ha="center",transform=ax.get_xaxis_transform(),fontsize=6)
        ax.set(xticks=np.arange(4),xticklabels=[METHOD_LABELS[k] for k in METHODS],
               ylabel="Closest sampled SAP–TP distance (mm)",title=f'Seed {session["seed"]}')
    fig.suptitle("Sensitivity to missing views at 20 mm · dots omit one frame · line all views · triangle omit longitudinal · square omit transverse")
    finish(fig,out/"missing_view_sensitivity.png")


def annotation_plot(d, out):
    variants=[k for k in d if k.startswith("sap_") and k!="sap_original"]
    if not variants:
        return
    fig, axes=plt.subplots(1,1+min(2,len(variants)),figsize=(13,5),subplot_kw={"projection":"3d"},layout="constrained")
    fig.get_layout_engine().set(w_pad=.26)
    named=[("sap_original","reference")]
    for k in tuple(v for v in variants if "_2_voxel" in v) or tuple(variants[:2]):
        named.append((k,k[4:]))
    axes=np.atleast_1d(axes)
    center=d["reference"].mean(0)
    radius=16
    for (sap,key), ax in zip(named,axes):
        for name,color in (("tp_original",COLORS[2]), (sap,COLORS[3]),
                           ("protected_local_first_bone","#626b73")):
            points=d[name]
            local=np.all(np.abs(points-center) <= radius, axis=1)
            ax.scatter(*points[local].T,s=4,color=color,alpha=.35)
        ax.scatter(*d[key].T,s=28,color="#b33240",marker="x")
        set_axes(ax,center,radius)
        ax.set_title(key.replace("_"," ").title())
    legend=(("TP label",COLORS[2],"o"), ("SAP label",COLORS[3],"o"),
            ("Protected first-hit voxel","#626b73","o"),
            ("Provisional midpoint","#b33240","x"))
    fig.legend(handles=[Line2D([0],[0],color=color,label=label,marker=marker,ls="")
                        for label,color,marker in legend],loc="outside lower center",ncols=4,frameon=False)
    fig.suptitle("Evaluator-only semantic annotation sensitivity · CT RAS coordinates (mm)")
    finish(fig,out/"annotation_sensitivity.png")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    args=p.parse_args()
    out=args.results
    summary=json.loads((out/"summary.json").read_text())
    data=[]
    for session in summary["sessions"]:
        with np.load(out/f"plot_{session['seed']}.npz") as f:
            data.append(dict(f))
    index=next(i for i,x in enumerate(summary["sessions"]) if x["seed"]=="2026121900")
    coverage_plot(summary,out)
    geometry_plot(data[index],METHODS,"2026121900",out/"dense_local_comparison.png",
                  "Local reconstruction from the same viewpoint")
    geometry_plot(data[index],("A0","A1","B0","B1"),"2026121900",out/"sparse_dense_local.png",
                  "Sparse and dense local reconstruction")
    reference_plot(data[index],"2026121900",out)
    local_errors(data,out)
    missing_view_plot(summary,out)
    with np.load(out/"annotation_sensitivity_geometry.npz") as f:
        annotation_plot(dict(f),out)


if __name__=="__main__":
    main()
