import cv2
import os
import math
import kornia 
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import argparse
from dataclasses import dataclass


MIN_TILE_SIZE = 512
MIN_OVERLAP = 64

## Geometric sanity thresholds for an ACCEPTED co-registration homography.
## Two crops of the same area are related by a near-similarity transform, so any
## matrix that mirrors, collapses, or wildly rescales the image is a degenerate fit.
MIN_AREA_SCALE = 0.25    ## lower bound on det(A) -> linear scale >= 0.5x
MAX_AREA_SCALE = 4.0     ## upper bound on det(A) -> linear scale <= 2.0x
MAX_CONDITION = 10.0     ## max singular-value ratio of the 2x2 linear part
MAX_PERSPECTIVE = 1e-2   ## max |H[2,0]|, |H[2,1]| (units: 1/pixel)


@dataclass
class AlignmentDiagnostics:
    '''
        Structured "why" behind a homography estimate, so the caller can report a
        single consolidated message.
    '''
    aligned: bool = False
    reason: str = "not attempted"       
    anchors_selected: int = 0
    anchors_matched: int = 0
    candidate_tiles_scored: int = 0      ## candidate tiles per anchor (img1 grid size)
    candidates_passing_sanity: int = 0   ## candidates that survived _is_sane_homography
    best_inlier_count: int = 0           ## best per-anchor inlier count seen
    required_inliers: int = 0            ## min_matches threshold
    pooled_points: int = 0               ## correspondences pooled across anchors
    pooled_inliers: int = 0              ## inliers of the final global fit


@dataclass
class AlgorithmConfig:
    '''Centralized configuration for the matching algorithms and data processing'''
    device: str = 'cpu'
    tile_size: int = MIN_TILE_SIZE
    overlap: int = MIN_OVERLAP
    confidence_threshold: float = 0.8
    ransac_threshold: float = 3.0
    min_matches: int = 15

    ## multi-anchor H0 estimation 
    anchor_fraction: float = 0.3     
    anchor_count: int = None        
    max_anchors: int = 8            
    anchor_score_floor: float = 0.5  

    def __post_init__(self):
        if self.tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        elif self.tile_size < MIN_TILE_SIZE:
            raise ValueError(f"tile_size must be at least {MIN_TILE_SIZE} pixels to ensure sufficient feature coverage")
        elif self.tile_size % 8 != 0:
            raise ValueError("tile_size must be a multiple of 8 for compatibility with the neural network architecture")
        elif self.overlap < 0:
            raise ValueError("overlap must be a non-negative integer")
        elif self.overlap >= self.tile_size:
            raise ValueError("overlap must be strictly less than tile_size to ensure proper tiling")
        elif self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be in the range [0.0, 1.0]")
        elif self.ransac_threshold <= 0.0:
            raise ValueError("ransac_threshold must be a positive float")
        elif self.anchor_fraction <= 0.0 or self.anchor_fraction > 1.0:
            raise ValueError("anchor_fraction must be in the range (0.0, 1.0]")
        elif self.anchor_count is not None and self.anchor_count < 1:
            raise ValueError("anchor_count must be a positive integer when provided")
        elif self.max_anchors < 1:
            raise ValueError("max_anchors must be a positive integer")
        elif self.anchor_score_floor < 0.0 or self.anchor_score_floor > 1.0:
            raise ValueError("anchor_score_floor must be in the range [0.0, 1.0]")
                

class InferenceEngine:
    '''Encapsulates the specific Kornia model interface and device management'''
    def __init__(self, feature_extractor, config: AlgorithmConfig):
        self.matcher = feature_extractor
        self.device = config.device
        self.confidence_threshold = config.confidence_threshold

    @torch.no_grad()
    def extract(self, tensor0, tensor1):
        '''
            Runs the feature extractor on a pair of tensors and returns keypoints and confidences.
            The tensors are expected to be in the shape (C, H, W) or (B, C, H, W). If they are 3D, they will 
            be unsqueezed to 4D for batch processing.
        '''
        if tensor0.dim() == 3:
            tensor0 = tensor0.unsqueeze(0)
            tensor1 = tensor1.unsqueeze(0)
        ## pack tensors into the specific dictionary format required by the model (e.g., LoFTR)
        input_dict = {
            "image0": tensor0.to(self.device),
            "image1": tensor1.to(self.device),
        }
        
        output_dict = self.matcher(input_dict)

        ## filter out weak matches based on the confidence threshold from the config
        mask = output_dict['confidence'] > self.confidence_threshold
        
        mkpts0 = output_dict['keypoints0'][mask].cpu().numpy()
        mkpts1 = output_dict['keypoints1'][mask].cpu().numpy()
        confidences = output_dict['confidence'][mask].cpu().numpy()

        return mkpts0, mkpts1, confidences



def _get_safe_coords(tile_size, stride, length):
    '''
        Helper function to generate coordinates without going out of bounds.
        Implements the edge-shift logic.
    '''

    if length <= tile_size:
        return [0]

    ## generate coordinates ONLY where the tile perfectly fits
    coords = list(range(0, length - tile_size + 1, stride))

    ## check if the very end of the image is left uncovered
    if not coords or coords[-1] + tile_size < length:
        ## force add one last tile aligned strictly to the boundary
        coords.append(length - tile_size)

    return coords

class GlobalAligner:
    '''
        Module for coarse image alignment using a neural feature extractor.
        Estimates a global homography matrix robust to seasonal and lighting changes.
        
        Architectural Note (Dependency Injection):
        This class requires an instantiated feature extractor model. 
        It expects the model to take an input dict {"image0": tensor, "image1": tensor} 
        and return a dict with keys 'keypoints0', 'keypoints1', and 'confidence'.
    '''
    def __init__(self, engine: InferenceEngine, ransac_threshold: float, tile_size: int, overlap: int, min_matches:int,
                 anchor_fraction: float = 0.3, anchor_count=None, max_anchors: int = 8,
                 anchor_score_floor: float = 0.5):
        self.ransac_threshold = ransac_threshold
        self.engine = engine
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_matches = min_matches
        self.anchor_fraction = anchor_fraction
        self.anchor_count = anchor_count
        self.max_anchors = max_anchors
        self.anchor_score_floor = anchor_score_floor
        self.mertens = cv2.createMergeMertens()

    def _calculate_anchor_score(self, tile):
        '''
            Calculates a score for a given tile based on its detail richness.
            Uses Mertens exposure fusion to enhance the tile and then applies a Laplacian filter.
            The variance of the Laplacian is returned as the score, which reflects the amount of detail.
        '''
        dark = cv2.convertScaleAbs(tile, alpha=0.5, beta=0)
        bright = cv2.convertScaleAbs(tile, alpha=2.0, beta=0)
        
        hdr = self.mertens.process([dark, tile, bright])
        hdr_8bit = (np.clip(hdr, 0, 1) * 255).astype(np.uint8)
        
        if len(hdr_8bit.shape) == 3:
            hdr_8bit = cv2.cvtColor(hdr_8bit, cv2.COLOR_BGR2GRAY)

        '''
            Why I use the Laplacian filter here:
            The Laplacian is a second-order derivative filter that highlights strong edges in all directions. 
            In this case, we are not afraid of image noise. We don't need perfect or exact contours. 
            We just want to measure the general amount of details to find the most informative tile.
        '''
        return cv2.Laplacian(hdr_8bit, cv2.CV_64F).var()
    
    def _tile_positions(self, img):
        '''All candidate (x, y) tile positions in img '''
        h, w = img.shape[:2]
        y_coords = _get_safe_coords(self.tile_size, self.tile_size - self.overlap, h)
        x_coords = _get_safe_coords(self.tile_size, self.tile_size - self.overlap, w)

        return [(x, y) for y in y_coords for x in x_coords]

    def _anchor_count(self, n_tiles):
        '''
            k = min(max_anchors, anchor_count or ceil(anchor_fraction * n_tiles)), floored at 2.
            The floor keeps us from silently degrading back to a single anchor on tiny images.
        '''
        if self.anchor_count is not None:
            k = self.anchor_count
        else:
            k = math.ceil(self.anchor_fraction * n_tiles)

        return min(self.max_anchors, max(2, k), n_tiles)

    def _select_anchor_tiles(self, img, k):
        '''
            Greedy farthest-point anchor selection.

            1. score every candidate tile (Mertens fusion + Laplacian variance = detail proxy);
            2. drop featureless tiles below anchor_score_floor * best_score (clouds, water, fields);
            3. first anchor = the most detailed eligible tile;
            4. each next anchor = the tile farthest from the NEAREST anchor already chosen
               (max-min separation; squared distance avoids a sqrt per comparison).
        '''
        positions = self._tile_positions(img)
        tiles = {p: img[p[1]:p[1] + self.tile_size, p[0]:p[0] + self.tile_size] for p in positions}
        scores = {p: self._calculate_anchor_score(t) for p, t in tiles.items()}

        max_score = max(scores.values())
        eligible = [p for p in positions if scores[p] >= self.anchor_score_floor * max_score]
        if len(eligible) < k:
            ## relax the floor rather than fail: spread without full detail still beats nothing
            eligible = positions

        def center(p):
            return (p[0] + self.tile_size / 2.0, p[1] + self.tile_size / 2.0)

        best = [max(eligible, key=lambda p: scores[p])]

        while len(best) < k and len(best) < len(eligible):
            def min_dist2(p):
                cx, cy = center(p) 
                ## square root is a monotonic function, dropping the sqrt gives the exact same winning tile 
                ## while running much faster and avoiding unnecessary calculations
                return min((cx - center(a)[0]) ** 2 + (cy - center(a)[1]) ** 2 for a in best)
            
            remaining = [p for p in eligible if p not in best]
            best.append(max(remaining, key=min_dist2))

        return [(p, tiles[p]) for p in best]

    def _generate_spiral_coords(self, h, w): 
        '''
            Creates a list of (x, y) coordinates to search in the second image.
            It starts searching from the center of the image and goes out in a spiral.
            Returns a list of coordinates sorted from center to edge.
        '''
        stride = int(self.tile_size * 0.75) ## overlap by 25% so we do not miss any important details.

        y_coords = _get_safe_coords(self.tile_size, stride, h)
        x_coords = _get_safe_coords(self.tile_size, stride, w)

        all_coords = [(x, y) for y in y_coords for x in x_coords]
        center_x = w / 2
        center_y = h / 2

        '''
            Sort all tiles by their squared distance to the center. 
            We don't use square root because squared distance is faster to calculate 
            and gives the exact same sorting order (from center to edges)
        '''

        all_coords.sort(key=lambda pt: (pt[0] - center_x)**2 + (pt[1] - center_y)**2)

        return all_coords
        
    def _to_tensor(self, img):
        '''
            Convert a numpy tail in a tensor for models like like Kornia LoFTR.
            If the image has colors (3 channels), it changes it to gray first.
            Models like LoFTR wait for the shape (B, 1, H, W)
        '''
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = kornia.image.image_to_tensor(img, keepdim=False).float() / 255.0

        return tensor

    def _compare_tiles(self, anchor_tile, candidate_tile):
        '''
            Direct (rotation-free) matching between the anchor tile and a candidate tile.
            Converts both numpy tiles to model input tensors and runs the injected
            matcher (e.g. LoFTR) through the InferenceEngine.
            Returns keypoints in each tile's LOCAL coordinate space + confidences.
        '''
        tensor0 = self._to_tensor(anchor_tile)
        tensor1 = self._to_tensor(candidate_tile)

        return self.engine.extract(tensor0, tensor1)

    def _is_sane_homography(self, H):
        '''
            Geometric sanity check on an estimated homography.
            Rejects matrices that flip, collapse, or wildly distort the image.
            Returns True if the homography is plausible, False otherwise.
        '''
        if H is None:
            return False

        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            return False

        A = H[:2, :2]                          ## linear (affine) part
        det = float(np.linalg.det(A))          ## signed area scale

        ## area scale must be plausible and must not flip orientation (no mirroring)
        if not (MIN_AREA_SCALE <= det <= MAX_AREA_SCALE):
            return False

        ## the linear part must not be degenerate (collapsed or heavily sheared)
        singular_values = np.linalg.svd(A, compute_uv=False)
        if singular_values[-1] <= 0.0 or (singular_values[0] / singular_values[-1]) > MAX_CONDITION:
            return False

        ## perspective terms must be negligible (units: 1/pixel) — a strong perspective
        ## component is a red flag for a degenerate fit on planar satellite imagery
        if abs(H[2, 0]) > MAX_PERSPECTIVE or abs(H[2, 1]) > MAX_PERSPECTIVE:
            return False

        return True

    def _anchor_correspondences(self, anchor_coords, anchor_tile, img1, stats=None):
        '''
            Spiral-search ONE anchor tile across img1 and return that anchor's VERIFIED
            correspondences lifted into global image coordinates: (pts0_global, pts1_global).
            Returns None if no valid match is found.
        '''
        ax, ay = anchor_coords
        best = None   ## (inlier_count, pts0_global, pts1_global)

        candidates_scored = 0
        candidates_passing_sanity = 0
        best_seen_inliers = 0

        for bx, by in self._generate_spiral_coords(img1.shape[0], img1.shape[1]):
            candidate_tile = img1[by:by + self.tile_size, bx:bx + self.tile_size]
            if candidate_tile.shape[0] < self.tile_size or candidate_tile.shape[1] < self.tile_size:
                continue

            candidates_scored += 1
            mkpts0, mkpts1, _ = self._compare_tiles(anchor_tile, candidate_tile)

            ## a homography needs at least 4 correspondences — skip degenerate tiles early
            if len(mkpts0) < 4:
                continue

            ## lift local tile keypoints into the global coordinate space of each image
            global_mkpts0 = mkpts0 + np.array([ax, ay], dtype=np.float32)
            global_mkpts1 = mkpts1 + np.array([bx, by], dtype=np.float32)

            ## run findHomography here as a temporary filter just to clean bad matches and allow early exit
            H, mask = cv2.findHomography(global_mkpts0, global_mkpts1, cv2.RANSAC, self.ransac_threshold)
            if H is None or mask is None:
                continue

            ## reject bad matches (flipped, squished, or distorted)
            if not self._is_sane_homography(H):
                continue

            candidates_passing_sanity += 1
            inliers = int(mask.sum())
            best_seen_inliers = max(best_seen_inliers, inliers)
            keep = mask.ravel().astype(bool)
            if best is None or inliers > best[0]:
                best = (inliers, global_mkpts0[keep], global_mkpts1[keep])

            ## this anchor already carries enough evidence — move on to the next one
            if inliers >= self.min_matches:
                break

        if stats is not None:
            stats["candidates_scored"] = candidates_scored
            stats["candidates_passing_sanity"] = candidates_passing_sanity
            stats["best_inliers"] = best_seen_inliers

        if best is None:
            return None
        return best[1], best[2]

    def estimate_homography(self, img0, img1):
        '''
            Multi-anchor global homography estimation.

            Instead of trusting a single anchor tile, select k high-detail, spatially-spread
            anchors in img0, spiral-search each of them in img1, POOL all of their verified
            correspondences, and fit ONE global homography with a single RANSAC pass. Points 
            spread all over the entire image make the alignment much more accurate and stable
            than points crowded together in just one small tile.

            Returns (H, diagnostics): H is the 3x3 homography mapping img0 -> img1 (or None
            when no reliable alignment exists), and diagnostics is an AlignmentDiagnostics
            carrying the counts behind the outcome so the caller can report ONE message.
        '''
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        if (h0 < self.tile_size) or (w0 < self.tile_size) or (h1 < self.tile_size) or (w1 < self.tile_size):
            raise ValueError(
                f"Both images must be at least {self.tile_size}x{self.tile_size} pixels. "
                f"Current sizes: img0={w0}x{h0}, img1={w1}x{h1}"
            )

        positions = self._tile_positions(img0)
        k = self._anchor_count(len(positions))
        anchors = self._select_anchor_tiles(img0, k)

        diag = AlignmentDiagnostics(
            anchors_selected=len(anchors),
            required_inliers=self.min_matches,
            candidate_tiles_scored=len(self._generate_spiral_coords(h1, w1)),
        )
        if len(anchors) < k:
            diag.reason = "few_anchors"

        pool0, pool1 = [], []
        for coords, tile in anchors:
            stats = {}
            corr = self._anchor_correspondences(coords, tile, img1, stats=stats)
            diag.candidates_passing_sanity += stats.get("candidates_passing_sanity", 0)
            diag.best_inlier_count = max(diag.best_inlier_count, stats.get("best_inliers", 0))
            if corr is None:
                continue
            diag.anchors_matched += 1
            pool0.append(corr[0])
            pool1.append(corr[1])

        if not pool0:
            diag.aligned = False
            diag.reason = "no_anchor_matched"
            
            return None, diag

        pts0 = np.vstack(pool0)
        pts1 = np.vstack(pool1)
        diag.pooled_points = len(pts0)

        H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, self.ransac_threshold)
        if H is None or mask is None:
            diag.aligned = False
            diag.reason = "global_fit_failed"
            return None, diag

        diag.pooled_inliers = int(mask.sum())

        ## ACCEPTANCE GATE: one global fit, well-supported (>= min_matches pooled inliers)
        ## and geometrically sane. Returning None tells the caller "no reliable alignment".
        if not self._is_sane_homography(H):
            diag.aligned = False
            diag.reason = "implausible_homography"

            return None, diag
        if diag.pooled_inliers < self.min_matches:
            diag.aligned = False
            diag.reason = "too_few_inliers"
            
            return None, diag

        diag.aligned = True
        diag.reason = "aligned"
        return H, diag


class ImagePreprocessor:
    '''
        Handles image loading, padding, and tiling for satellite image.

        ImagePreprocessor is an internal component and may only be constructed from a pre-validated 
        AlgorithmConfig. Input validation is owned exclusively by AlgorithmConfig.__post_init__.
    '''
    @staticmethod
    def load_image(image_path, as_gray=True):
        ''' Loads a satellite image from disk and converts it to grayscale if requested '''
        with rasterio.open(image_path) as src:
            img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
            if as_gray:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return img


    def __init__(self, tile_size, overlap):
        self.tile_size = tile_size
        self.stride = self.tile_size - overlap

    def _pad_tile(self, tile, tile_h, tile_w):
        '''Fallback padding ONLY used if the ENTIRE image is smaller than tile_size'''
        pad_h = self.tile_size - tile_h 
        pad_w = self.tile_size - tile_w 
    
        tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        mask_pad = np.ones((tile_h, tile_w), dtype=np.float32)
        mask_pad = cv2.copyMakeBorder(mask_pad, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        return tile, mask_pad

    def _extract_tiles(self, image_path):
        '''Generator that yields image tiles as tensors and their global coordinates.'''
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file '{image_path}' does not exist.")

        img = self.load_image(image_path, as_gray=True)
        h, w = img.shape[:2]

        x_coords = _get_safe_coords(self.tile_size, self.stride, w)
        y_coords = _get_safe_coords(self.tile_size, self.stride, h)

        for y in y_coords:
            mask_pad = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
            for x in x_coords:
                tile = img[y:y+self.tile_size, x:x+self.tile_size]
                tile_h, tile_w = tile.shape
                if tile_h < self.tile_size or tile_w < self.tile_size:
                    tile, mask_pad = self._pad_tile(tile, tile_h, tile_w)

                tile_tensor = kornia.image.image_to_tensor(tile, keepdim=False).float() / 255.0

                yield tile_tensor, (x, y), mask_pad

    def _extract_corresponding_tile(self, img_reader, x, y, H):
        '''Dynamically extracts and warps the corresponding tile from the second image using the global homography matrix H'''
        ## define the 4 corners of the ideal square tile on the first image (clockwise)
        pts0 = np.float32([
            [x, y], 
            [x + self.tile_size, y],
            [x + self.tile_size, y + self.tile_size],
            [x, y + self.tile_size]
        ]).reshape(-1, 1, 2)

        ## project these corners onto the second image to find where this land is located
        pts1 = cv2.perspectiveTransform(pts0, H)

        ## calculate the bounding box for extraction
        min_x = max(0, int(np.floor(pts1[:, 0, 0].min())))
        min_y = max(0, int(np.floor(pts1[:, 0, 1].min())))
        max_x = min(img_reader.width, int(np.ceil(pts1[:, 0, 0].max())))
        max_y = min(img_reader.height, int(np.ceil(pts1[:, 0, 1].max())))

        w_box = max_x - min_x
        h_box = max_y - min_y

        if w_box <= 0 or h_box <= 0:
            return None, None ## the projected tile is completely outside the bounds of the second image

        ## read the raw, unaligned crop directly from the disk using Rasterio Window
        window = Window(min_x, min_y, w_box, h_box)
        raw_crop_3d = img_reader.read([1, 2, 3], window=window)
        raw_crop_hwc = np.moveaxis(raw_crop_3d, 0, -1)
        raw_crop = cv2.cvtColor(raw_crop_hwc, cv2.COLOR_RGB2GRAY)

        ## create matrices to link local coordinate systems with the global ones
        ### shifts local tile coordinates to global image 0 coordinates (+x, +y)
        T_offset0 = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ], dtype=np.float32)

        ### shifts global image 1 coordinates back to the local raw_crop coordinates (-min_x, -min_y)
        T_offset1 = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ], dtype=np.float32)

        ## build the final transformation pipeline.
        ### local_pixel -> add (x,y) -> apply H -> subtract (min_x, min_y) -> raw_crop pixel
        local_H = T_offset1 @ H @ T_offset0

        ## use the inverse of local_H to map pixels from the second image back to the first image's perspective
        aligned_tile = cv2.warpPerspective(
            raw_crop,
            np.linalg.inv(local_H),
            (self.tile_size, self.tile_size)
        )

        ## create a mask to identify valid pixels in the warped tile. This is important for later stages of matching, as some pixels may be invalid due to the warping process.
        raw_mask = np.ones_like(raw_crop, dtype=np.float32)
        mask_warped = cv2.warpPerspective(
            raw_mask, 
            np.linalg.inv(local_H), 
            (self.tile_size, self.tile_size), 
            flags=cv2.INTER_LINEAR
        )

        tile_tensor = kornia.image.image_to_tensor(aligned_tile, keepdim=False).float() / 255.0
        return tile_tensor, mask_warped

    def generate_tile_pairs(self, path0, path1, H):
        '''
            High-level wrapper that yields perfectly aligned pairs of tiles.
            Hides file I/O and spatial alignment logic from the main pipeline.
        '''
    
        with rasterio.open(path1) as img1_reader:
            for tile0, (x, y), mask_pad in self._extract_tiles(path0):
                tile1, mask_warped = self._extract_corresponding_tile(img1_reader, x, y, H)
                
                ## skip tiles that are entirely outside the bounds of the second image
                if tile1 is None or mask_warped is None:
                    continue
                    
                yield tile0, tile1, mask_pad, mask_warped, (x, y)



class SatelliteMatcher:
    '''
        Runs inference on perfectly aligned tile pairs and 
        projects local keypoint coordinates back to the global image space.
        
        To use a different model, you can pass it via the `feature_extractor` parameter. Ensure your custom model implements
        the same input/output dictionary interface as Kornia's LoFTR.
    '''
    def __init__(self, engine):
        self.engine = engine
        self.engine.matcher.eval()

    @torch.no_grad() ## disable grandient calculation to save memory and speed up inference 
    def __call__(self, tile_tensor0 :torch.Tensor, tile_tensor1: torch.Tensor, x, y, H, mask_pad, mask_warped):
        '''
            Executes the matching algorithm on a pair of tensors and calculates 
            global coordinates for the keypoints using homography H.
        '''
        mkpts0_local, mkpts1_local, confidences = self.engine.extract(tile_tensor0, tile_tensor1)

        if len(mkpts0_local) == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

        valid_points = []

        ## filter out keypoints that fall outside the valid region of the warped tile
        if mask_warped is not None or mask_pad is not None:
            for pt1, pt2, conf in zip(mkpts0_local, mkpts1_local, confidences):
                x0 = int(np.clip(pt1[0], 0, tile_tensor0.shape[-1] - 1))
                y0 = int(np.clip(pt1[1], 0, tile_tensor0.shape[-2] - 1))
                x1 = int(np.clip(pt2[0], 0, tile_tensor1.shape[-1] - 1))
                y1 = int(np.clip(pt2[1], 0, tile_tensor1.shape[-2] - 1))

                if  mask_pad[y0, x0] > 0.99 and mask_warped[y1, x1] > 0.99 :  ## ensure both points are in valid regions
                    valid_points.append((pt1, pt2, conf))
        else:
            valid_points = list(zip(mkpts0_local, mkpts1_local, confidences))

        if len(valid_points) == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

        pts0_local = np.array([pt[0] for pt in valid_points], dtype=np.float32)
        pts1_local = np.array([pt[1] for pt in valid_points], dtype=np.float32)
        valid_confidences = np.array([pt[2] for pt in valid_points], dtype=np.float32)

        offset = np.array([x, y], dtype=np.float32)
        mkpts0_global = pts0_local + offset
        mkpts1_aligned_global = pts1_local + offset

        pts_to_transform = mkpts1_aligned_global.reshape(-1, 1, 2)
        mkpts1_global = cv2.perspectiveTransform(pts_to_transform, H).reshape(-1, 2)

        return mkpts0_global, mkpts1_global, valid_confidences



def main(args):
    ''' Standalone test runner for algorithm_creation.py classes'''
    print(f"Testing base classes for:\n  img0: {args.img0}\n  img1: {args.img1}")

    config = AlgorithmConfig(
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap
    )

    feature_extractor = kornia.feature.LoFTR(pretrained='outdoor').to(config.device)
    engine = InferenceEngine(feature_extractor, config)
    
    aligner = GlobalAligner(
        engine, config.ransac_threshold, config.tile_size, config.overlap, config.min_matches,
        anchor_fraction=config.anchor_fraction, anchor_count=config.anchor_count,
        max_anchors=config.max_anchors, anchor_score_floor=config.anchor_score_floor
    )
    preprocessor = ImagePreprocessor(config.tile_size, config.overlap)
    matcher = SatelliteMatcher(engine)
        
    img0 = preprocessor.load_image(args.img0)
    img1 = preprocessor.load_image(args.img1)

    H, diag = aligner.estimate_homography(img0, img1)

    if H is None:
        print(f"Global homography estimation failed: {diag.reason}")

        H = np.eye(3, dtype=np.float32)
    else:
        print("Global Homography Matrix:\n", H)

    tile_generator = preprocessor.generate_tile_pairs(args.img0, args.img1, H)

    total_matches = 0
    for tile0, tile1, mask_pad, mask_warped, (x, y) in tile_generator:
        pts0, _, _ = matcher(tile0, tile1, x, y, H, mask_pad, mask_warped)

        if len(pts0) > 0:
            total_matches += len(pts0)
            
    print(f"Test passed! Found {total_matches} total matches across tiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Satellite Image Matching classes")
    parser.add_argument("--img0", type=str, required=True, help="Path to the first satellite image")
    parser.add_argument("--img1", type=str, required=True, help="Path to the second satellite image")
    parser.add_argument("--tile_size", type=int, default=MIN_TILE_SIZE, help=f"Tile size for processing (default: {MIN_TILE_SIZE})")
    parser.add_argument("--overlap", type=int, default=MIN_OVERLAP, help=f"Overlap between tiles (default: {MIN_OVERLAP})")
    parser.add_argument("--device", type=str, default="cpu", help="Type of device (default: 'cpu')")
    args = parser.parse_args()

    main(args)