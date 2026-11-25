"""Render diagnostics and an actual 102-waypoint development replay."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import yaml

from perception_metrics import pick_components
from replay import CONVERSION, context, infer, models_for, replay
from run_simple_landmark_mapping import load_evaluator, sha256, write_json
from two_stage_navigation.simple_landmark_mapping import project_image


def factorial(protocol, previous, historical, evaluator, output, device):
    folder=previous/"review-52711211/E3-dataset"
    audit=json.loads((folder/"factorial.json").read_text())
    archive=np.load(folder/"factorial.npz")
    original=protocol["models"]["sonoanatomy"]
    models=models_for(original)
    base=context(historical/"session_02",protocol,evaluator,"historical")
    nearby=context(historical/"session_03",protocol,evaluator,"historical")
    base_seq=next(i for i in base["valid"] if base["frames"][i]["waypoint_index"]==5)
    near_seq=next(i for i in nearby["valid"] if nearby["frames"][i]["waypoint_index"]==5)
    probs={l:list(infer(m,base["images"][base["valid"]],device,batch_size=1)) for l,(m,r) in models.items()}
    replacement_index=base["valid"].index(base_seq)
    report={}
    records={}
    for variant in ("A0","A1","B1","C1","D0"):
        actual=audit[variant]
        source=base if variant in ("A0","A1","D0") else nearby
        source_seq=base_seq if source is base else near_seq
        image=archive[f"{variant}_image"]
        labels=archive[f"{variant}_labels"]
        changed=deepcopy(base["frames"][base_seq])
        for key in ("actor_base_to_ee_position_m","actor_base_to_ee_rotation_columns",
                    "actor_base_to_ee_quaternion_wxyz","executed_relative_pose_index","level_hypotheses"):
            changed[key]=source["frames"][source_seq][key]
        changed["frame_sha256"]=actual["image_sha256"]
        ctx=dict(base,frames=list(base["frames"]),coords=dict(base["coords"]),visible=dict(base["visible"]))
        ctx["frames"][base_seq]=changed
        ctx["coords"][base_seq]=project_image(changed,image.shape,protocol["geometry"]["ultrasound_pixel_resolution_mm"])
        for c,k in ((1,"TP"),(2,"SAP")):
            name=f"RIGHT_L5_{k}"
            ctx["visible"].pop((name,base_seq),None)
            if (labels==c).any():
                ctx["visible"][name,base_seq]=cKDTree(np.argwhere(labels==c))
        perturbed_prob={l:list(old) for l,old in probs.items()}
        for level,(model,rules) in models.items():
            perturbed_prob[level][replacement_index]=next(infer(model,image[None],device,batch_size=1))
        update, overlay=replay(ctx,protocol,models,tp_all=False,probabilities=perturbed_prob)
        metric={}
        probabilities=perturbed_prob["L5"][replacement_index]
        prediction=overlay["L5"][base_seq]
        for c,k in ((1,"TP"),(2,"SAP")):
            identity=f"RIGHT_L5_{k}"
            target=labels==c
            rule=models["L5"][1][k]
            reps=pick_components(probabilities[c],probabilities.argmax(0)==c,
                                 rule["threshold"],rule["minimum_pixels"])
            for row in reps:
                row["coordinate_mm"]=ctx["coords"][base_seq][tuple(row["pixel"])].tolist()
            metric[k]={"oracle_pixels":int(target.sum()),"predicted_pixels":int((prediction==c).sum()),
                       "p_max_on_visible_pixels":float(np.max(probabilities[c][target])) if target.any() else None,
                       "p_max_full_image":float(probabilities[c].max()),"representatives":reps,
                       "counterfactual_map_outcome":update["score"]["landmarks"][identity]["outcome"]}
        records[variant]=(image,labels,prediction, probabilities)
        report[variant]={"render_control":actual,"perception":metric}
    identical=np.array_equal(records["A0"][3],records["D0"][3])
    write_json(output/"factorial_perception.json",{
        "cases":report,"identical_render_control_identical_probability":identical,
        "scope":"One-frame substitution into previously acquired historical sweep, using matched recorded poses"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(13,3.5),layout="constrained")
    upper=np.quantile([r[0] for r in records.values()],.995)
    for ax,(kind,(image,truth,prediction,_)) in zip(axes,records.items()):
        ax.imshow(image,cmap="gray",vmin=0,vmax=upper)
        for data,color in ((truth,"cyan"),(prediction,"orange")):
            if (data==1).any():
                ax.contour(data==1,levels=[.5],colors=[color],linewidths=.9)
        ax.set_title(f"{kind} | visible {int((truth==1).sum())} px")
        ax.axis("off")
    fig.suptitle("L5 TP at fixed rendering controls | cyan oracle · orange original model")
    fig.savefig(output/"speckle_controls.png",dpi=150)
    plt.close(fig)


def additional_figures(selected, eval_dir, data_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    receipt=json.loads((data_dir/"receipt.json").read_text())
    splits=receipt["splits"]
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout="constrained")
    for ax,level in zip(axes,("L4","L5")):
        x=np.arange(len(splits))
        for i,(key,label) in enumerate((("tp","TP"),("sap","SAP"),("other","Other bone"))):
            ax.bar(x+i*.24,[s["class_distribution"].get(f"{level}_{key}_frames",0) for s in splits.values()],
                   width=.24,label=label)
        ax.set_xticks(x+.24,list(splits),rotation=15)
        ax.set_title(level)
        ax.set_ylabel("Frames with a visible class")
        ax.legend(fontsize=8)
    fig.savefig(output/"dataset_distribution.png",dpi=150)
    plt.close(fig)

    combined=json.loads((eval_dir/"combined_validation.json").read_text())
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout="constrained")
    for ax,(level,sem) in zip(axes.flat,(("L4","TP"),("L4","SAP"),("L5","TP"),("L5","SAP"))):
        ax.plot([0,1],[0,1],"k:",alpha=.5)
        for arm in ("original","selected"):
            reliability=combined[arm][level]["curve"][sem]["reliability_correct_counts"]
            points=[(int(bin_id)/10+.05,g/count,count) for bin_id,(count,g) in reliability.items()]
            if points:
                ax.scatter([d[0] for d in points],[d[1] for d in points],
                           s=[4+2*d[2] for d in points],label=arm,alpha=.65)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="Component confidence",ylabel="Within 3 px", title=level+" "+sem)
        ax.legend(fontsize=8)
    fig.savefig(output/"combined_reliability.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
    classes=(("L4","TP"),("L4","SAP"),("L5","TP"),("L5","SAP"))
    for j,mode in enumerate(("original","selected")):
        for i,(level,sem) in enumerate(classes):
            d=combined[mode][level]["fixed_evaluation"][sem]["closest_representative_px"]
            if d["count"]:
                ax.bar(i+(j-.5)*.3,d["median"],width=.29, alpha=.75,label=mode if i==0 else None)
                ax.plot(i+(j-.5)*.3,d["p95"],"_",color="black")
    ax.set_xticks(range(4),[f"{l} {s}" for l,s in classes])
    ax.axhline(3,color="#a23d37",ls=":",label="3 px criterion")
    ax.set_ylabel("Closest accepted representative (px)\nBar median · tick p95")
    ax.legend(fontsize=8)
    fig.savefig(output/"combined_interface_localization.png",dpi=150)
    plt.close(fig)


def draw_map(session, protocol, evaluator, reference_spec, output, device, independent=False):
    import render_simple_sap_demo as demo
    from render_simple_landmark_mapping import actor_figure

    ctx=context(session,protocol,evaluator,"medium")
    acts={"original":models_for(protocol["models"]["sonoanatomy"]),"selected":models_for(reference_spec)}
    demo.TITLE=("FRESH INDEPENDENT ACQUISITION" if independent else "RECORDED DEVELOPMENT SWEEP - PERCEPTION REPLAY")
    paired_masks={}
    for name,model in acts.items():
        result,masks=replay(ctx,protocol,model,tp_all=True,device=device)
        paired_masks[name]=masks
        actor=dict(ctx["actor"],observations=result["observations"],
                   ambiguous_observations=result["ambiguous_observations"],landmarks=result["mapping"])
        actor["level_hypotheses_by_frame"]={str(k):ctx["frames"][k]["level_hypotheses"] for k in ctx["valid"]}
        report={"score":result["score"]}
        actor_figure(output/(name+"_map.png"),str(session.name),ctx["images"],ctx["frames"],
                     ctx["valid"],masks,actor,result["mapping"],title=demo.TITLE)
        if name=="selected":
            video=output/"landmark_sweep.mp4"
            receipt=demo.render_video(session,video,actor,ctx["frames"],ctx["images"],ctx["valid"],masks,report,protocol)
            if receipt["robot_camera_frames"]!=len(ctx["valid"]):
                raise ValueError("video did not verify every robot camera join")
            write_json(output/"video_verification.json",dict(receipt,sha256=sha256(video),
                                                            exact_case_source=str(session),scope=demo.TITLE))
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(4,3,figsize=(8,9),layout="constrained")
    for axis_row,(lvl,code) in zip(axes,(("L4",1),("L4",2),("L5",1),("L5",2))):
        visible=[i for i in ctx["valid"] if (ctx["labels"][lvl][i]==code).any()]
        if not visible:
            for axis in axis_row:
                axis.axis("off")
            continue
        index=sorted(visible,key=lambda i:int((ctx["labels"][lvl][i]==code).sum()))[len(visible)//2]
        im=ctx["images"][index]
        choices=(("Oracle",ctx["labels"][lvl][index]),("Original",paired_masks["original"][lvl][index]),
                 ("Selected",paired_masks["selected"][lvl][index]))
        for ax,(title,mask) in zip(axis_row,choices):
            ax.imshow(im,cmap="gray",vmin=0,vmax=np.quantile(im,.99))
            if (mask==code).any():
                ax.contour(mask==code,levels=[.5],colors=["cyan" if title=="Oracle" else "orange"],linewidths=.8)
            ax.set_title(f"{title} | {lvl} {('TP','SAP')[code-1]} WP {ctx['frames'][index]['waypoint_index']}",fontsize=8)
            ax.axis("off")
    fig.suptitle("Same executed pose and ultrasound on every row")
    fig.savefig(output/"paired_segmentation.png",dpi=145)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","profile","previous","historical","development","data","evaluation","output"):
        parser.add_argument("--"+key,required=True,type=Path)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    protocol=yaml.safe_load(args.protocol.read_text())
    campaign=yaml.safe_load(args.profile.read_text())
    evaluator=load_evaluator(protocol,CONVERSION)
    selected=json.loads((args.evaluation/"selected_before_combined.json").read_text())
    line=(args.evaluation/"selected_before_combined.sha256").read_text()
    assert sha256(args.evaluation/"selected_before_combined.json")==line.split()[0]
    factorial(protocol,args.previous,args.historical,evaluator,args.output,args.device)
    additional_figures(selected,args.evaluation,args.data,args.output)
    draw_map(args.development/str(campaign["development_seeds"][2]),protocol,evaluator,
             selected["models"],args.output,args.device)


if __name__=="__main__":
    main()
