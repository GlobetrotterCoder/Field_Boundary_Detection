import rasterio
from rasterio.merge import merge
import numpy as np

def merge_tiff_files(input_files, output_file):
    # Open all input TIFF files
    src_files = [rasterio.open(file) for file in input_files]

    # Merge the individual datasets
    merged_dataset, merged_transform = merge(src_files)

    # Normalize the pixel values to the range 0-255
    merged_dataset = merged_dataset.astype('float32')
    merged_dataset = (merged_dataset - np.min(merged_dataset)) / (np.max(merged_dataset) - np.min(merged_dataset))
    merged_dataset = (merged_dataset * 255).astype('uint8')

    # Update the metadata for the merged dataset
    merged_meta = src_files[0].meta.copy()
    merged_meta.update({
        'driver': 'GTiff',
        'height': merged_dataset.shape[1],
        'width': merged_dataset.shape[2],
        'transform': merged_transform
    })

    # Write the merged dataset to the output file
    with rasterio.open(output_file, 'w', **merged_meta) as dst:
        dst.write(merged_dataset)

    # Close all the opened datasets
    for src in src_files:
        src.close()

# Example usagemask
input_files = ['/home/cos-bot/Downloads/LUON_maskRCNN/mask_rwanda/B01.tif', '/home/cos-bot/Downloads/LUON_maskRCNN/mask_rwanda/B02.tif', '/home/cos-bot/Downloads/LUON_maskRCNN/mask_rwanda/B03.tif']
output_file = 'mask.tif'
merge_tiff_files(input_files, output_file)
