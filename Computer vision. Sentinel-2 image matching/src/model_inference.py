from algorithm_creation import ImagePreprocessor, SatelliteMatcher
import numpy as np
import cv2
from dataclasses import dataclass
import matplotlib.pyplot as plt
from kornia_moons.feature import draw_LAF_matches
import kornia.feature as KF
import torch

@dataclass
class MatchResult:
    """Holds matching results between two satellite images."""
    image_path0: str
    image_path1: str
    pts0: np.ndarray
    pts1: np.ndarray
    confs: np.ndarray
    inliers_mask: np.ndarray

    @property
    def is_empty(self):
        return len(self.pts0) == 0


class MatchingPipeline:
    """Pipeline for processing and matching satellite images across tiles."""
    
    def __init__(self, tile_size=512, ransac_threshold=3.0):
        self.preprocessor = ImagePreprocessor(tile_size)
        self.matcher = SatelliteMatcher()
        self.ransac_threshold = ransac_threshold

    def _filter_outliers(self, pts0, pts1):
        """Filter outlier matches using fundamental matrix estimation (RANSAC/MAGSAC)."""
        magsac_algo = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC

        _, inliers_mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            magsac_algo,
            self.ransac_threshold,
            0.999,
            100000
        )

        if inliers_mask is None:
            return np.array([], dtype=bool)

        return inliers_mask.ravel().astype(bool)

    def __call__(self, image_path0, image_path1):
        """Execute the matching pipeline for two input image paths"""
        tiles_gen0 = self.preprocessor.extract_tiles(image_path0)
        tiles_gen1 = self.preprocessor.extract_tiles(image_path1)

        all_pts0, all_pts1, all_confs = [], [], []
        
        ## iterate through tiles and find matches locally
        for (tile0, coord0), (tile1, coord1) in zip(tiles_gen0, tiles_gen1):
            pts0, pts1, confs = self.matcher(tile0, tile1, coord0, coord1)

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
                inliers_mask=np.array([], dtype=bool)
            )

        ## aggregate points from all tiles
        all_pts0 = np.vstack(all_pts0)
        all_pts1 = np.vstack(all_pts1)
        all_confs = np.concatenate(all_confs)

        if len(all_pts0) < 8:
            inliers_mask = np.ones(len(all_pts0), dtype=bool)
        else:
            inliers_mask = self._filter_outliers(all_pts0, all_pts1)

        return MatchResult(
            image_path0=image_path0,
            image_path1=image_path1,
            pts0=all_pts0,
            pts1=all_pts1,
            confs=all_confs,
            inliers_mask=inliers_mask
        )
    
    def draw_matches(self, matches: MatchResult, show_outliers=False, max_points=150, strategy="best"):
        """Visualize feature matches between two images."""
        if matches.is_empty:
            print("Object doesn't have correct matches.")
            return

        img0 = cv2.cvtColor(cv2.imread(matches.image_path0), cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(cv2.imread(matches.image_path1), cv2.COLOR_BGR2RGB)

        pts0 = matches.pts0
        pts1 = matches.pts1
        confs = matches.confs
        mask = matches.inliers_mask

        if mask is None or len(mask) != len(pts0):
            mask = np.ones(len(pts0), dtype=bool)

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

        lafs1 = pts_to_lafs(pts0)
        lafs2 = pts_to_lafs(pts1)
        
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
            lafs1=lafs1,
            lafs2=lafs2,
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