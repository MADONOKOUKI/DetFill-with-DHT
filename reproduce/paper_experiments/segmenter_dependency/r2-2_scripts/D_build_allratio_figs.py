#!/home/madorin/anaconda3/envs/py37/bin/python
"""Build/inspect per-id all-ratio figure cells for R2-2 supp_fig_allratio.
Model: DanbooRegion-trained DetFill (raw eval, reaper2). Sketch index matched to existing figs (sk=0).
Modes: --dry (report) | --write --id <ID> [--sk N].  Emits <ID>_<seg>_{seg,hint,a000..a100}.png + <ID>_GT.png."""
import cv2, numpy as np, glob, os, argparse, sys
P   = "/home/madorin/gitlab/labrepo/main/revision_materials"
RAW = P + "/rebuttal/R2/R2-2_segmentation_dependency/output/D_retrain_eval_raw/reaper2/danbooregion_model/dataset_name"
FIGS= P + "/paper_src/figs/revision_figs/revise/R2/R2-2_segdep"
SEGMAP = {"felz":"/home/madorin/datasets/labrepo/main_exp_felzenszwalb_fixdot/illust/segmentation_regions/felzenszwalb",
          "slic":P+"/rebuttal/R2/R2-2_segmentation_dependency/retrain_data/slic",
          "danboo":P+"/rebuttal/R2/R2-2_segmentation_dependency/retrain_data/danbooregion"}
EVALSEG = {"felz":"on_felz","danboo":"on_danbooregion","slic":"on_slic"}
RATIOS  = [("a000","0.0"),("a001","0.01"),("a003","0.03"),("a005","0.05"),("a010","0.1"),("a025","0.25"),("a050","0.5"),("a100","1.0")]
CELL=256
def res_p(seg,ratio,i,sk): return "%s/%s/sample_to_eval/illust/scribble/%d/%s/200/%s.image.png"%(RAW,EVALSEG[seg],sk,ratio,i)
def hint_p(seg,i,sk):      return "%s/%s/sample_to_eval/illust/scribble/%d/1.0/condition/%s.image_hint.png"%(RAW,EVALSEG[seg],sk,i)
def hint_pr(seg,ratio,i,sk): return "%s/%s/sample_to_eval/illust/scribble/%d/%s/condition/%s.image_hint.png"%(RAW,EVALSEG[seg],sk,ratio,i)
def gt_p(i,sk):            return "%s/%s/sample_to_eval/illust/scribble/%d/1.0/ground_truth/%s.image_col.png"%(RAW,EVALSEG["danboo"],sk,i)
def seg_glob(seg,i):
    for pat in ("%s/*/%s.image*region64*.png","%s/*/%s*region64*.png","%s/*/%s.image.png"):
        h=glob.glob(pat%(SEGMAP[seg],i))
        if h: return h[0]
    return None
def lr(p,smooth):
    a=cv2.imread(p)
    if a is None: raise IOError("read fail "+p)
    return cv2.resize(a,(CELL,CELL),interpolation=cv2.INTER_AREA if smooth else cv2.INTER_NEAREST)
def match_sketch():
    ex=os.path.join(FIGS,"1019016_danboo_a050.png")
    if not os.path.exists(ex): return None
    e=cv2.imread(ex); H,W=e.shape[:2]; best=(None,1e9)
    for sk in (0,1,2):
        p=res_p("danboo","0.5","1019016",sk)
        if not os.path.exists(p): continue
        r=cv2.resize(cv2.imread(p),(W,H),interpolation=cv2.INTER_AREA)
        mae=float(np.abs(e.astype(int)-r.astype(int)).mean())
        if mae<best[1]: best=(sk,mae)
    return best
def build(i,sk):
    n=0
    cv2.imwrite(os.path.join(FIGS,"%s_GT.png"%i), lr(gt_p(i,sk),True)); n+=1
    for seg in ("felz","danboo","slic"):
        cv2.imwrite(os.path.join(FIGS,"%s_%s_seg.png"%(i,seg)),  lr(seg_glob(seg,i),False)); n+=1
        cv2.imwrite(os.path.join(FIGS,"%s_%s_hint.png"%(i,seg)), lr(hint_p(seg,i,sk),False)); n+=1
        for tag,r in RATIOS:
            cv2.imwrite(os.path.join(FIGS,"%s_%s_hint_%s.png"%(i,seg,tag)), lr(hint_pr(seg,r,i,sk),False)); n+=1
            cv2.imwrite(os.path.join(FIGS,"%s_%s_%s.png"%(i,seg,tag)), lr(res_p(seg,r,i,sk),True)); n+=1
    return n
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dry",action="store_true"); ap.add_argument("--write",action="store_true"); ap.add_argument("--id"); ap.add_argument("--sk",type=int,default=-1); a=ap.parse_args()
    ms=match_sketch(); sk=(a.sk if a.sk>=0 else (ms[0] if ms else 0))
    print("[sketch-match] best=%s  use_sk=%s"%(ms,sk))
    if a.write:
        assert a.id, "--id required"
        n=build(a.id,sk); print("[write] id=%s sk=%s wrote=%d files -> %s"%(a.id,sk,n,FIGS)); sys.exit(0)
    cands=sorted({os.path.basename(x)[7:-4] for x in glob.glob(FIGS+"/candidates/segdep_*.png")})
    for seg in ("felz","danboo","slic"): print("   segmap",seg,"->",seg_glob(seg,"1019016"))
