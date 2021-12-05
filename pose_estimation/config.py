class Config:
    input_shape = (256, 256) 
    output_shape = (input_shape[0]//8, input_shape[1]//8)
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)
    joint_num = 21
    skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20),
    (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
    depth_dim = 32
    bbox_real = (2000, 2000)

cfg = Config()

