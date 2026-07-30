import cv2
import os
import kornia 
import rasterio
from rasterio.windows import Window
import numpy as np


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
        '''Generator that yields image tiles as TensorFlow tensors and their global coordinates.'''
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
    def __init__(self):
        self.matcher = kornia.feature.LoFTR(pretrained='outdoor')

    def __call__(self, tile_tensor0, tile_tensor1, coord0, coord1):
        '''
            Executes the matching algorithm on a pair of tensors 
            and calculates global coordinates for the keypoints.
        '''
        input_dict = {
            "image0": tile_tensor0,
            "image1": tile_tensor1
        }

        output_dict = self.matcher(input_dict)
        mkpts0_local = output_dict['keypoints0'].numpy()
        mkpts1_local = output_dict['keypoints1'].numpy()
        confidences = output_dict['confidence'].numpy()

        offset0 = np.array([coord0[0], coord0[1]])
        offset1 = np.array([coord1[0], coord1[1]])

        mkpts0_global = mkpts0_local + offset0
        mkpts1_global = mkpts1_local + offset1

        return mkpts0_global, mkpts1_global, confidences
    