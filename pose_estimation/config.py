class Config:
    input_shape = (256, 256) 
    output_shape = (input_shape[0]//8, input_shape[1]//8)
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

cfg = Config()

