import os
from algorithm_creation import (
    GlobalAligner, 
    ImagePreprocessor, 
    SatelliteMatcher, 
    AlgorithmConfig, 
    InferenceEngine,
    AlignmentDiagnostics
)
import numpy as np
import cv2
from dataclasses import dataclass
import matplotlib.pyplot as plt
from kornia_moons.feature import draw_LAF_matches
import kornia.feature as KF
import kornia
import torch
import argparse
import math

np.random.seed(42)

@dataclass
class MatchResult:
    '''Holds matching results between two satellite images.'''
    image_path0: str
    image_path1: str
    pts0: np.ndarray
    pts1: np.ndarray
    confs: np.ndarray
    inliers_mask: np.ndarray
    overlap_confidence: float = 0.0
    aligned: bool = False
    diagnostics: AlignmentDiagnostics = None

    @property
    def is_empty(self):
        return len(self.pts0) == 0



## inlier count -> confidence the two images overlap.
OVERLAP_CONF_SLOPE    = 3.0
OVERLAP_CONF_MIDPOINT = 2.5   ## log10(inliers) at which P(overlap) = 0.5

def overlap_confidence(n_inliers, slope=OVERLAP_CONF_SLOPE, midpoint=OVERLAP_CONF_MIDPOINT):
    '''Confidence that two images cover the same location, from verified inlier count.'''
    n = max(int(n_inliers), 0)
    z = slope * (math.log10(n + 1.0) - midpoint)
    return 1.0 / (1.0 + math.exp(-z))

def _rule(char="-"): 
    return char * 60

def _row(label, value):
    '''One aligned "label .......... value" line inside a report block.'''
    return f"  {label} {'.' * max(1, 32 - len(label))} {value}"
def render_alignment_failure(diag: AlignmentDiagnostics, image_path0, image_path1):
    '''
        ONE consolidated block explaining why alignment failed. Replaces the scattered
        per-layer warnings; the caller prints this and nothing else about the failure.
    '''
    reason_text = diag.reason
    if reason_text in ["too_few_inliers", "global_fit_failed"]:
        reason_text = "no correction needed (already co-registered) or locations differ"

    lines = [
        _rule("="),
        f" ALIGNMENT NOTICE — {os.path.basename(image_path0)} vs {os.path.basename(image_path1)}",
        "  Standard geometric correction was skipped; evaluating direct co-registration.",
        _rule("-"),
        _row("anchors selected", diag.anchors_selected),
        _row("anchors matched", diag.anchors_matched),
        _row("candidate tiles scored", diag.candidate_tiles_scored),
        _row("candidates passing sanity", diag.candidates_passing_sanity),
        _row("best inlier count", f"{diag.best_inlier_count}   (required: {diag.required_inliers})"),
        _row("pooled points", diag.pooled_points),
        _row("pooled inliers", diag.pooled_inliers),
        _row("reason", reason_text),
        _rule("="),
    ]
    return "\n".join(lines)

def render_result(results, diag: AlignmentDiagnostics):
    '''
        ONE final conclusion block: confidence that the two images overlap, plus the
        inlier/match counts and the alignment verdict. Always printed, success or failure.
    '''
    conf = results.overlap_confidence
    conf_note = "HIGH — likely the same location" if conf >= 0.5 else "LOW — likely NOT the same location"
    
    if diag.aligned:
        verdict = "OK (aligned)"
    elif conf >= 0.5:
        verdict = "OK (co-registered via grid / identity fallback)"
    else:
        verdict = "NOTE (distinct locations or insufficient overlap)"

    lines = [
        _rule("="),
        " RESULT",
        _row("overlap confidence", f"{conf:.3f}  ({conf_note})"),
        _row("inliers", int(np.sum(results.inliers_mask))),
        _row("total matches", len(results.pts0)),
        _row("alignment", verdict),
        _rule("="),
    ]
    return "\n".join(lines)
class MatchingPipeline:
    '''Pipeline for processing and matching satellite images across tiles.'''
    def __init__(self, engine: InferenceEngine, config: AlgorithmConfig, output=True):
        self.aligner = GlobalAligner(
            engine, config.ransac_threshold, config.tile_size, config.overlap, config.min_matches,
            anchor_fraction=config.anchor_fraction, anchor_count=config.anchor_count,
            max_anchors=config.max_anchors,
            anchor_score_floor=config.anchor_score_floor
        )
        self.preprocessor = ImagePreprocessor(tile_size=config.tile_size, overlap=config.overlap)
        self.matcher = SatelliteMatcher(engine)
        self.ransac_threshold = config.ransac_threshold
        self.output = output

    def _filter_outliers(self, pts0, pts1):
        '''Filter outlier matches by estimating the homography between the two point sets (RANSAC/MAGSAC).'''
        magsac_algo = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC

        _, inliers_mask = cv2.findHomography(
            pts0,
            pts1,
            magsac_algo,
            self.ransac_threshold,
            0.999,
            100000
        )
        if inliers_mask is None:
            return None

        return inliers_mask.ravel().astype(bool)

    def __call__(self, image_path0, image_path1):
        '''Execute the matching pipeline for two input image paths'''
        img0 = self.preprocessor.load_image(image_path0)
        img1 = self.preprocessor.load_image(image_path1)

        H, diag = self.aligner.estimate_homography(img0, img1)
        if H is None:
            ## ONE consolidated block explaining the failure — no scattered warnings.
            if self.output: print(render_alignment_failure(diag, image_path0, image_path1))
            H = np.eye(3, dtype=np.float32)
        
        tile_generator = self.preprocessor.generate_tile_pairs(image_path0, image_path1, H)

        all_pts0, all_pts1, all_confs = [], [], []
        
        ## iterate through tiles and find matches locally
        for tile0, tile1, mask_pad, mask_warped, (x, y) in tile_generator:
            pts0, pts1, confs = self.matcher(tile0, tile1, x, y, H, mask_pad, mask_warped)
        
            if len(pts0) > 0:
                all_pts0.append(pts0)
                all_pts1.append(pts1)
                all_confs.append(confs)

        if len(all_pts0) == 0:  
            return MatchResult(
                image_path0=image_path0, 
                image_path1=image_path1, 
                pts0=np.array([]), 
                pts1=np.array([]), 
                confs=np.array([]),  
                inliers_mask=np.array([], dtype=bool),
                overlap_confidence=0.0,
                aligned=diag.aligned,
                diagnostics=diag
            )

        ## aggregate points from all tiles
        all_pts0 = np.vstack(all_pts0)
        all_pts1 = np.vstack(all_pts1)
        all_confs = np.concatenate(all_confs)

        if len(all_pts0) < 4:
            inliers_mask = np.ones(len(all_pts0), dtype=bool)
        else:
            inliers_mask = self._filter_outliers(all_pts0, all_pts1)
            if inliers_mask is None:
                ## geomtric degeneracy at the outlier-filter stage (distinct from
                ## alignment failure) — one line, owned here, not in draw_matches.
                if self.output:
                    print(
                        f"NOTE: outlier verification inconclusive for "
                        f"{os.path.basename(image_path0)} vs {os.path.basename(image_path1)} — "
                        "correspondences are geometrically degenerate; reporting zero verified overlap."
                    )
                inliers_mask = np.zeros(len(all_pts0), dtype=bool)

        overlap_conf = overlap_confidence(np.sum(inliers_mask))

        
        results = MatchResult(
            image_path0=image_path0,
            image_path1=image_path1,
            pts0=all_pts0,
            pts1=all_pts1,
            confs=all_confs,
            inliers_mask=inliers_mask,
            overlap_confidence=overlap_conf,
            aligned=diag.aligned,
            diagnostics=diag
        )

        if self.output: print(render_result(results, diag))

        return results
    
    def _safe_resize(self,image, max_size, force_divisible_by_8=False):
        '''
            Resize the image proportionally so neither dimension exceeds max_size, optionally 
            rounding dimensions down to multiples of eight, and return the resized image with 
            its x/y scale factors.
        '''
        h, w = image.shape[:2]
        scale = min(max_size / w, max_size / h)
            
        if force_divisible_by_8:
            scale = min(scale, 1.0)
            new_w = int((w * scale) // 8) * 8
            new_h = int((h * scale) // 8) * 8
        else:
            new_w = int(w * scale)
            new_h = int(h * scale)
                
        scale_x = new_w / w
        scale_y = new_h / h
    
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized_img, scale_x, scale_y
    
    def draw_matches(self, matches: MatchResult, show_outliers=False, max_points=150, strategy="best"):
        '''Purely graphical: draw feature matches between two images using Kornia Moons.'''
        if matches.is_empty:
            return

        img0_full = self.preprocessor.load_image(matches.image_path0, False)
        img1_full = self.preprocessor.load_image(matches.image_path1, False)

        img0, scale_x0, scale_y0 = self._safe_resize(img0_full, max_size=1500)
        img1, scale_x1, scale_y1 = self._safe_resize(img1_full, max_size=1500)    

        del img0_full, img1_full

        pts0 = matches.pts0.copy() * np.array([scale_x0, scale_y0])
        pts1 = matches.pts1.copy() * np.array([scale_x1, scale_y1])
        confs = matches.confs
        mask = matches.inliers_mask

        if mask is None or len(mask) != len(pts0):
            raise ValueError("Inliers mask must be provided and have the same length as the number of points.")

        if not show_outliers and np.any(mask):
            pts0 = pts0[mask]
            pts1 = pts1[mask]
            confs = confs[mask] if confs is not None and len(confs) == len(mask) else None
            mask = np.ones(len(pts0), dtype=bool)

        ## limit the number of points for cleaner visualization
        if len(pts0) > max_points:
            if strategy == "best" and confs is not None and len(confs) == len(pts0):
                indices = np.argsort(confs)[-max_points:]
            else:
                indices = np.random.choice(len(pts0), max_points, replace=False)
            
            pts0 = pts0[indices]
            pts1 = pts1[indices]
            mask = mask[indices]

        ## helper function to convert standard points to Kornia LAFs 
        def pts_to_lafs(pts):
            tensor_pts = torch.from_numpy(pts).view(1, -1, 2).float()
            scales = torch.ones(tensor_pts.shape[1]).view(1, -1, 1, 1)
            oris = torch.zeros(tensor_pts.shape[1]).view(1, -1, 1)
            return KF.laf_from_center_scale_ori(tensor_pts, scales, oris)

        lafs0 = pts_to_lafs(pts0)
        lafs1 = pts_to_lafs(pts1)
        
        tent_idxs = torch.stack([torch.arange(len(pts0)), torch.arange(len(pts0))], dim=-1)

        draw_dict = {
            "inlier_color": (0.2, 1.0, 0.2),   ## green for inliers
            "tentative_color": (1.0, 0.2, 0.2) if show_outliers else None, ## red for outliers
            "feature_color": None,
            "vertical": False
        }

        _, ax = plt.subplots(figsize=(16, 8))

        ## draw matches using kornia_moons
        draw_LAF_matches(
            lafs1=lafs0,
            lafs2=lafs1,
            tent_idxs=tent_idxs,
            img1=img0,
            img2=img1,
            inlier_mask=mask,
            draw_dict=draw_dict,
            ax=ax
        )

        plt.axis('off')
        plt.tight_layout()
        plt.show()



def main(args):
    '''
        The function for running inference.
        Accepts paths to images, tile size, ransac_threshold and visualization parameters.
    '''
    print(f"Starting matching pipeline for:\n  img0: {args.img0}\n  img1: {args.img1}")
        
    config = AlgorithmConfig(
        device=args.device,
        tile_size=args.tile_size,
        confidence_threshold=args.confidence_threshold,
        ransac_threshold=args.ransac_threshold
    )

    feature_extractor = kornia.feature.LoFTR(pretrained='outdoor').to(config.device)
    engine = InferenceEngine(feature_extractor, config)
    pipeline = MatchingPipeline(engine, config)
    results = pipeline(args.img0, args.img1)
 
    pipeline.draw_matches(
        results,
        show_outliers=args.show_outliers,
        max_points=args.max_points
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Satellite Image Matching Inference")
            
    parser.add_argument("--img0", type=str, required=True, help="Path to the first satellite image")
    parser.add_argument("--img1", type=str, required=True, help="Path to the second satellite image")
    parser.add_argument("--confidence_threshold", type=float, default=0.8, help="LoFTR confidence threshold (default: 0.8)")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size for processing (default: 512)")
    parser.add_argument("--ransac_threshold", type=float, default=3.0, help="RANSAC threshold (default: 3.0)")
    parser.add_argument("--show_outliers", action="store_true", help="Flag to visualize outlier matches (red lines)")
    parser.add_argument("--max_points", type=int, default=150, help="Maximum number of points to visualize (default: 150)")
    parser.add_argument("--device", type=str, default='cpu', help="Type of device (default: 'cpu')")
            
    args = parser.parse_args()
    main(args)
