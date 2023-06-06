import rasterio
from PIL import Image
import numpy as np

def merge_rgb_and_convert_to_jpeg(tif_file, jpeg_file):
    # Open the TIFF file
    with rasterio.open(tif_file) as dataset:
        # Read the RGB bands (assuming they are in bands 1, 2, and 3)
        red = dataset.read(1)
        green = dataset.read(2)
        blue = dataset.read(3)
        
        # Stack the bands to create a RGB image
        rgb = [red, green, blue]
        
        # Normalize pixel values to the 0-255 range
        rgb = [(band * 255 / band.max()).astype('uint8') for band in rgb]
        
        # Create a PIL Image object from the RGB data
        image = Image.fromarray(np.dstack(rgb), 'RGB')
        
        # Save the image as a JPEG file
        image.save(jpeg_file, 'JPEG')

# Example usage:
tif_file = '/home/cos-bot/Downloads/LUON_maskRCNN/merged.tif'
jpeg_file = '/home/cos-bot/Downloads/LUON_maskRCNN/mask.jpg'

merge_rgb_and_convert_to_jpeg(tif_file, jpeg_file)
