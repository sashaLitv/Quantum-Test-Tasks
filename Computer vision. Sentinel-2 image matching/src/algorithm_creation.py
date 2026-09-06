import cv2
import os
import kornia 
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import argparse
from dataclasses import dataclass

@dataclass
class AlgorithmConfig:
    '''Centralized configuration for the matching algorithms and data processing.'''
    device: str = 'cpu'
    tile_size: int = 512
    overlap: int = 128
    confidence_threshold: float = 0.8
    ransac_threshold: float = 3.0
    global_max_size: int = 840

class InferenceEngine:
    '''Encapsulates the specific Kornia model interface and device management'''
    def __init__(self, feature_extractor, config: AlgorithmConfig):
        self.matcher = feature_extractor
        self.device = config.device
        self.confidence_threshold = config.confidence_threshold

    @torch.no_grad()
    def extract(self, tensor0, tensor1):
        '''
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

class GlobalAligner:
    '''
        Module for coarse image alignment using a neural feature extractor.
        Estimates a global homography matrix robust to seasonal and lighting changes.
        
        Architectural Note (Dependency Injection):
        This class requires an instantiated feature extractor model. 
        It expects the model to take an input dict {"image0": tensor, "image1": tensor} 
        and return a dict with keys 'keypoints0', 'keypoints1', and 'confidence'.
    '''
    def __init__(self, engine: InferenceEngine, global_max_size: int, ransac_threshold: float):
        self.max_size = global_max_size
        self.ransac_threshold = ransac_threshold
        self.engine = engine

    @staticmethod
    def safe_resize(image, max_size, force_divisible_by_8=False):
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

    def _prepare_image_tensor(self, image):
        '''Prepares the OpenCV image for the neural network (resizes and converts to Tensor)'''
        small_img, scale_x, scale_y = self.safe_resize(
            image, self.max_size, force_divisible_by_8=True
        )
        
        tensor = kornia.image.image_to_tensor(small_img, keepdim=False).float() / 255.0
        
        return tensor, scale_x, scale_y
    
    @torch.no_grad()
    def estimate_homography(self, img0, img1):
        '''
            Estimates the 3x3 homography matrix that maps img0 to img1.
            Returns an identity matrix (np.eye) if matching fails.
        '''
        ## downsample images to prevent memory issues and speed up execution
        tensor0, scale_x0, scale_y0 = self._prepare_image_tensor(img0)
        tensor1, scale_x1, scale_y1 = self._prepare_image_tensor(img1)
       
        pts1_small, pts2_small, _ = self.engine.extract(tensor0, tensor1)

        ## mathematical constraint: at least 4 points are required to compute a 3x3 perspective matrix
        if len(pts1_small) < 4:
            return np.eye(3, dtype=np.float32)

        ## map coordinates from the downsampled image back to the original full-resolution image space
        pts1 = pts1_small / np.array([scale_x0, scale_y0], dtype=np.float32)
        pts1 = pts1.reshape(-1, 1, 2).astype(np.float32)

        pts2 = pts2_small / np.array([scale_x1, scale_y1], dtype=np.float32)
        pts2 = pts2.reshape(-1, 1, 2).astype(np.float32)

        H, inliers_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_threshold)
        if H is None or inliers_mask is None or np.sum(inliers_mask) < 15:
            return np.eye(3, dtype=np.float32)

        return H


class ImagePreprocessor:
    '''
        Handles image loading, padding, and tiling for satellite image.
        
        Architectural Note: this data-processing logic would ideally be 
        extracted into a separate module (e.g., `data_processor.py`). 
        However, it is kept in `algorithm_creation.py` to strictly comply 
        with the required file structure of the technical assignment.
    '''
    def __init__(self, tile_size=256, overlap=128):
        self.tile_size = tile_size
        self.stride = self.tile_size - overlap

        if self.stride <= 0:
            raise ValueError("Overlap must be strictly less than tile_size")
        
    def _get_coords(self, length):
        '''Calculates the starting coordinates for image tiling'''
        if length <= self.tile_size:
            return [0]

        ## generate coordinates ONLY where the tile perfectly fits
        coords = list(range(0, length - self.tile_size + 1, self.stride))

        ## check if the very end of the image is left uncovered
        if not coords or coords[-1] + self.tile_size < length:
            ## force add one last tile aligned strictly to the boundary
            coords.append(length - self.tile_size)

        return coords

    def _pad_tile(self, tile, tile_h, tile_w):
        '''Fallback padding ONLY used if the ENTIRE image is smaller than tile_size'''
        pad_h = self.tile_size - tile_h 
        pad_w = self.tile_size - tile_w 
    
        return cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    def _extract_tiles(self, image_path):
        '''Generator that yields image tiles as tensors and their global coordinates.'''
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file '{image_path}' does not exist.")
        
        with rasterio.open(image_path) as img:
            w, h = img.width, img.height

            x_coords = self._get_coords(w)
            y_coords = self._get_coords(h)

            for y in y_coords:
                for x in x_coords:
                    window = Window(x, y, self.tile_size, self.tile_size)

                    tile = img.read(1, window=window)
                    tile_h, tile_w = tile.shape
                    if tile_h < self.tile_size or tile_w < self.tile_size:
                        tile = self._pad_tile(tile, tile_h, tile_w)

                    tile_tensor = kornia.image.image_to_tensor(tile, keepdim=False).float() / 255.0

                    yield tile_tensor, (x, y)

    def _extract_corresponding_tile(self, img_reader, x, y, H):
        '''
            Dynamically extracts and warps the corresponding tile from the second image using the global homography matrix H.
        '''
        ## define the 4 corners of the ideal square tile on the first image (clockwise)
        pts1 = np.float32([
            [x, y], 
            [x + self.tile_size, y],
            [x + self.tile_size, y + self.tile_size],
            [x, y + self.tile_size]
        ]).reshape(-1, 1, 2)

        ## project these corners onto the second image to find where this land is located
        pts2 = cv2.perspectiveTransform(pts1, H)

        ## calculate the bounding box for extraction
        min_x = max(0, int(np.floor(pts2[:, 0, 0].min())))
        min_y = max(0, int(np.floor(pts2[:, 0, 1].min())))
        max_x = min(img_reader.width, int(np.ceil(pts2[:, 0, 0].max())))
        max_y = min(img_reader.height, int(np.ceil(pts2[:, 0, 1].max())))

        w_box = max_x - min_x
        h_box = max_y - min_y

        if w_box <= 0 or h_box <= 0:
            return torch.zeros((1, 1, self.tile_size, self.tile_size), dtype=torch.float32)

        ## read the raw, unaligned crop directly from the disk using Rasterio Window
        window = Window(min_x, min_y, w_box, h_box)
        raw_crop = img_reader.read(1, window=window)

        ## сreate matrices to link local coordinate systems with the global ones
        ### shifts local tile coordinates to global image 1 coordinates (+x, +y)
        T_offset1 = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ], dtype=np.float32)

        ### shifts global image 2 coordinates back to the local raw_crop coordinates (-min_x, -min_y)
        T_offset2 = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ], dtype=np.float32)

        ## build the final transformation pipeline.
        ### local_pixel -> add (x,y) -> apply H -> subtract (min_x, min_y) -> raw_crop pixel
        local_H = T_offset2 @ H @ T_offset1

        ## use the inverse of local_H to map pixels from the second image back to the first image's perspective
        aligned_tile = cv2.warpPerspective(
            raw_crop,
            np.linalg.inv(local_H),
            (self.tile_size, self.tile_size)
        )

        tile_tensor = kornia.image.image_to_tensor(aligned_tile, keepdim=False).float() / 255.0
        return tile_tensor

    def generate_tile_pairs(self, path1, path2, H):
        '''
            High-level wrapper that yields perfectly aligned pairs of tiles.
            Hides file I/O and spatial alignment logic from the main pipeline.
        '''
    
        with rasterio.open(path2) as img2_reader:
            for tile1, (x, y) in self._extract_tiles(path1):
                tile2 = self._extract_corresponding_tile(img2_reader, x, y, H)
                
                ## skip tiles that are entirely outside the bounds of the second image
                if tile2.sum() == 0:
                    continue
                    
                yield tile1, tile2, (x, y)
    
class SatelliteMatcher:
    '''
        Runs inference on perfectly aligned tile pairs and 
        projects local keypoint coordinates back to the global image space.
        
        To use a different model (e.g. LightGlue, SuperPoint), you can pass it 
        via the `feature_extractor` parameter. Ensure your custom model implements
        the same input/output dictionary interface as Kornia's LoFTR.
    '''
    def __init__(self, engine):
        self.engine = engine
        self.engine.matcher.eval()

    @torch.no_grad() ## disable grandient calculation to save memory and speed up inference 
    def __call__(self, tile_tensor0 :torch.Tensor, tile_tensor1: torch.Tensor, x, y, H):
        '''
            Executes the matching algorithm on a pair of tensors and calculates 
            global coordinates for the keypoints using homography H.
        '''
        mkpts0_local, mkpts1_local, confidences = self.engine.extract(tile_tensor0, tile_tensor1)

        if len(mkpts0_local) == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

        ## image1 points: simply add the local bounding box offset
        mkpts0_global = mkpts0_local + np.array([x, y], dtype=np.float32)
        ## image 2 points:
        ### since tile2 was pre-warped to match tile1's perspective, its local points share the same local coordinate grid as tile1
        ### step A: shift these local points to image 1's GLOBAL coordinate space
        mkpts1_aligned_global = mkpts1_local + np.array([x, y])
        
        ### step B: use matrix H to find where these points live on the original image2
        pts_to_transform = mkpts1_aligned_global.reshape(-1, 1, 2).astype(np.float32)
        mkpts1_global = cv2.perspectiveTransform(pts_to_transform, H).reshape(-1, 2)

        return mkpts0_global, mkpts1_global, confidences


def main(args):
    '''
        Standalone test runner for algorithm_creation.py classes.
    '''
    print(f"Testing base classes for:\n  img1: {args.img1}\n  img2: {args.img2}")

    config = AlgorithmConfig(
        device=args.device,
        tile_size=args.tile_size
    )

    feature_extractor = kornia.feature.LoFTR(pretrained='outdoor').to(config.device)
    engine = InferenceEngine(feature_extractor, config)
    
    aligner = GlobalAligner(engine, config.global_max_size, config.ransac_threshold)
    preprocessor = ImagePreprocessor(config.tile_size, config.overlap)
    matcher = SatelliteMatcher(engine)
        
    img1_cv = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
    img2_cv = cv2.imread(args.img2, cv2.IMREAD_GRAYSCALE)

    if img1_cv is None or img2_cv is None:
        raise ValueError("Error: Could not read one or both images with OpenCV")

    H = aligner.estimate_homography(img1_cv, img2_cv)
    print("Global Homography Matrix:\n", H)

    tile_generator = preprocessor.generate_tile_pairs(args.img1, args.img2, H)

    total_matches = 0
    for tile0, tile1, (x, y) in tile_generator:
        pts0, _, _ = matcher(tile0, tile1, x, y, H)

        if len(pts0) > 0:
            total_matches += len(pts0)
            
    print(f"Test passed! Found {total_matches} total matches across tiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Satellite Image Matching classes")
    parser.add_argument("--img1", type=str, required=True, help="Path to the first satellite image")
    parser.add_argument("--img2", type=str, required=True, help="Path to the second satellite image")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size for processing (default: 256)")
    parser.add_argument("--device", type=str, default='cpu', help="Type of device (default: 'cpu')")
    args = parser.parse_args()

    main(args)

   