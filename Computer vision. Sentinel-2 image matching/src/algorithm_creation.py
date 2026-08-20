import cv2
import os
import kornia 
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import argparse
    


class ImagePreprocessor:
    '''
        Handles image loading, padding, and tiling for satellite image.
        
        Architectural Note: this data-processing logic would ideally be 
        extracted into a separate module (e.g., `data_processor.py`). 
        However, it is kept in `algorithm_creation.py` to strictly comply 
        with the required file structure of the technical assignment.
    '''
    def __init__(self, tile_size=256):
        self.tile_size = tile_size

    def _pad_tile(self, tile, tile_h, tile_w):
        '''Pads smaller tiles at the edges of the image to match tile_size.'''
        pad_h = self.tile_size - tile_h 
        pad_w = self.tile_size - tile_w 
    
        padded_tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
        return padded_tile

    def extract_tiles(self, image_path):
        '''Generator that yields image tiles as tensors and their global coordinates.'''
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file '{image_path}' does not exist.")
        
        with rasterio.open(image_path) as img:
            w, h = img.width, img.height

            for y in range(0, h, self.tile_size):
                for x in range(0, w, self.tile_size):
                    window = Window(x, y, self.tile_size, self.tile_size)

                    tile = img.read(1, window=window)
                    tile_h, tile_w = tile.shape
                    if tile_h < self.tile_size or tile_w < self.tile_size:
                        tile = self._pad_tile(tile, tile_h, tile_w)

                    tile_tensor = kornia.image.image_to_tensor(tile, keepdim=False).float() / 255.0

                    yield tile_tensor, (x, y)
    

class SatelliteMatcher:
    '''Handles feature matching between satellite image tiles using the SOTA LoFTR model.'''
    def __init__(self, device='cpu'):
        self.device = device
        self.matcher = kornia.feature.LoFTR(pretrained='outdoor').to(self.device)

    def __call__(self, tile_tensor0 :torch.Tensor, tile_tensor1: torch.Tensor, coord0, coord1):
        '''
            Executes the matching algorithm on a pair of tensors 
            and calculates global coordinates for the keypoints.
        '''
        input_dict = {
            "image0": tile_tensor0.to(self.device),
            "image1": tile_tensor1.to(self.device),
        }

        output_dict = self.matcher(input_dict)
        mkpts0_local = output_dict['keypoints0'].cpu().numpy()
        mkpts1_local = output_dict['keypoints1'].cpu().numpy()
        confidences = output_dict['confidence'].cpu().numpy()

        offset0 = np.asarray(coord0, dtype=np.float32)
        offset1 = np.asarray(coord1, dtype=np.float32)

        mkpts0_global = mkpts0_local + offset0
        mkpts1_global = mkpts1_local + offset1

        return mkpts0_global, mkpts1_global, confidences


def main(args):
    '''
        The function for running inference.
        Accepts paths to images, tile size.
    '''
    print(f"Initializing matching process for:\n  img1: {args.img1}\n  img2: {args.img2}")
    
    preprocessor = ImagePreprocessor(tile_size=args.tile_size)
    matcher = SatelliteMatcher(device='cpu')
        
    tiles_gen0 = preprocessor.extract_tiles(args.img1)
    tiles_gen1 = preprocessor.extract_tiles(args.img2)
    total_matches = 0
        
    for (tile0, coord0), (tile1, coord1) in zip(tiles_gen0, tiles_gen1):
        pts0, _, _ = matcher(tile0, tile1, coord0, coord1)
            
        if len(pts0) > 0:
            print(f"Tile at {coord0}: found {len(pts0)} tentative matches.")
            total_matches += len(pts0)
                
    print("-" * 40)
    print(f"Total tentative matches found across all tiles: {total_matches}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Satellite Image Matching classes")
    parser.add_argument("--img1", type=str, required=True, help="Path to the first satellite image")
    parser.add_argument("--img2", type=str, required=True, help="Path to the second satellite image")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size for processing (default: 256)")
    args = parser.parse_args()

    main(args)

   