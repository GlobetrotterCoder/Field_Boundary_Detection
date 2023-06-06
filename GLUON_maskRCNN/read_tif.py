import rasterio
src = rasterio.open('/home/cos-bot/Downloads/LUON_maskRCNN/20220216_043712_70_2262_3B_AnalyticMS_SR_8b_clip_merged.tif')
# src1 = rasterio.open('/home/cos-bot/Downloads/LUON_maskRCNN/merged.tif')
# src2 = rasterio.open('/home/cos-bot/Downloads/LUON_maskRCNN/raster_labels.tif')


print(src.name)

array = src2.read(1)
array.shape

from matplotlib import pyplot
pyplot.imshow(array)

pyplot.show()

