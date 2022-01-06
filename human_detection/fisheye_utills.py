# dependancies are as follows
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision import transforms as transforms
import time

class FisheyeUtills:
    def __init__(self, img = None,
                       sensor_height = 24, 
                       fov = 180, 
                       cam_height = 2300,
                       bbox_scale = 2.0 ): 
        self.sensor_height = sensor_height  # image sensor height in mm, default 
        self.fov = fov  # fov of fisheye image in degree
        self.cam_height = cam_height  # camera height in mm
        self.bbox_scale = bbox_scale  # scale factor from bbox to tangent image
        self.patch_height = 256
        self.patch_width = 256
        
        if img is not None:
            self.imgs = self.get_image_batch(img)  # self.imgs.shape [N, C, H, W]
            self.height = self.imgs.shape[2]
            self.width = self.imgs.shape[3]

            # sensor pitch i.e. height of single sensor element
            self.d = self.sensor_height/self.height  

            # principal point
            self.u0 = (self.height-1)/2
            self.v0 = (self.width-1)/2

            # focal length
            self.f = self.fov2f(self.u0*self.d, self.fov*np.pi/180, k=-0.5)

    def get_image_batch(self, img):
        '''
        convert rgb image batch to torch.Tensor
        imgs.shape : [N, C, H, W]
        '''
        transform = transforms.Compose([transforms.ToTensor()])
        transform_img = transform(img).cuda() if torch.cuda.is_available() else transform(img)
        imgs = torch.unsqueeze(transform_img, dim=0)
        return imgs
    
    def fov2f(self, R, fov, k=-0.5):
        '''
        calculate focal length from fov and distance to image edge
        based on PTGui 11 fisheye projection with fisheye factor k (https://wiki.panotools.org/Fisheye_Projection)
        R : distance from principal point to image edge in mm  
        fov : fov angle in radians
        k = -1.0 : orthographic
        k = -0.5 : equisolid
        k =  0.0 : equidistant
        k =  0.5 : stereographic
        k =  1.0 : rectilinear (perspective)
        '''
        # if not torch.jit.isinstance(fov, torch.Tensor):
        if not isinstance(fov, torch.Tensor):
            fov = torch.tensor(fov)
        
        if k >= -1 and k < 0:
            f = k * R / torch.sin(k * fov/2)
        elif k == 0:
            f = 2 * R / fov
        elif k > 0 and k <= 1:
            f = k * R / torch.tan(k * fov/2)
        else:
            raise ValueError(f'{k} is not valid value for fisheye projection')
        
        return f
    
    def sphere2cartesian(self, lon, lat, depth=None):
        '''
        convert spherical(geographic) coordinates to cartesian
        '''
        if not isinstance(lon, torch.Tensor):
            lon = torch.tensor(lon)
        if not isinstance(lat, torch.Tensor):
            lat = torch.tensor(lat)

        X = torch.cos(lat) * torch.cos(lon)
        Y = torch.cos(lat) * torch.sin(lon)
        Z = torch.sin(lat)
        
        if depth is not None:
            X = X * depth
            Y = Y * depth
            Z = Z * depth
        
        return torch.stack((X, Y, Z), dim=0)
    
    def cartesian2sphere(self, X, Y, Z):
        '''
        convert cartesian coordinates to spherical(geographic) coordinates
        '''
        lon = torch.atan2(Y, X)
        lat = torch.atan2(Z, torch.sqrt(X.square()+Y.square()))
        return torch.stack((lon, lat), dim=0)
        
    def fisheye2sphere(self, u, v, k=-0.5, cart=False):
        '''
        convert fisheye image pixel coordinates to spherical(geographic) coordinates of the virtual sphere
        based on PTGui 11 fisheye projection with fisheye factor k (https://wiki.panotools.org/Fisheye_Projection)
        k = -1.0 : orthographic
        k = -0.5 : equisolid
        k =  0.0 : equidistant
        k =  0.5 : stereographic
        k =  1.0 : rectilinear (perspective)
        '''
        # if not torch.jit.isinstance(u, torch.Tensor):
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u)
        # if not torch.jit.isinstance(v, torch.Tensor):
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        assert u.shape==v.shape
            
        # U, V denote displacement from principal point in mm
        U = (u - self.u0) * self.d
        V = (v - self.v0) * self.d
        
        # lon, lat denote lognitude and latitue in virtual sphere
        lon = torch.atan2(V,U) 
        
        # PTGui 11 fisheye reverse projection
        if k >= -1 and k < 0:
            lat = np.pi/2 - torch.asin(k*torch.sqrt(U.square()+V.square())/self.f) / k
        elif k == 0:
            lat = np.pi/2 - torch.sqrt(U.square()+V.square()) / self.f
        elif k > 0 and k <= 1:
            lat = np.pi/2 - torch.atan2(k*torch.sqrt(U.square()+V.square()),self.f) / k
        else:
            raise ValueError(f'{k} is not valid value for fisheye projection')
        
        if cart:  # output cartesian coordinates
            sphere = self.sphere2cartesian(lon, lat)
        else:
            sphere = torch.stack((lon, lat), dim=0)
        return sphere
    
    def sphere2fisheye(self, lon, lat, k=-0.5):
        '''
        convert spherical(geographic) coordinates of the virtual sphere to fisheye image pixel coordinates
        based on PTGui 11 fisheye projection with fisheye factor k (https://wiki.panotools.org/Fisheye_Projection)
        k = -1.0 : orthographic
        k = -0.5 : equisolid
        k =  0.0 : equidistant
        k =  0.5 : stereographic
        k =  1.0 : rectilinear (perspective)
        '''
        # PTGui 11 fisheye projection
        if k >= -1 and k < 0:
            R = self.f * torch.sin(k * (np.pi/2-lat)) / k
        elif k == 0:
            R = self.f * (np.pi/2-lat)
        elif k > 0 and k <= 1:
            R = self.f * torch.tan(k * (np.pi/2-lat)) / k
        else:
            raise ValueError(f'{k} is not valid value for fisheye projection')
        
        # U, V denote displacement from principal point in mm
        U = R * torch.cos(lon)
        V = R * torch.sin(lon)
        
        # u, v denote pixel coordinates in fisheye image
        u = U / self.d + self.u0
        v = V / self.d + self.v0
        return torch.stack((u, v), dim=0)
    
    def rotate_spherical(self, lon, lat, rot_axis, rot_angle, cart=False):
        '''
        rotate spherical(geographic) coordinates following Rodrigues rotation formula
        '''
        # Rodrigues rotation formula
        kx, ky, kz = rot_axis  # rotation axis   
        K = torch.tensor([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]])
        R = torch.eye(3) + torch.sin(rot_angle)*K + (1-torch.cos(rot_angle))*torch.mm(K,K)

        XYZ = self.sphere2cartesian(lon, lat)
        flat_XYZ = XYZ.view((3,-1))
        XYZ = torch.matmul(R, flat_XYZ).view(XYZ.shape)
        
        if not cart:
            XYZ = self.cartesian2sphere(XYZ[0],XYZ[1],XYZ[2])
        return XYZ

    def move2pole(self, lon, lat, t_lon, t_lat, cart=False):
        '''
        rotate spherical coordinates for target point to be moved to pole
        '''
        # if not torch.jit.isinstance(t_lon, torch.Tensor):
        if not isinstance(t_lon, torch.Tensor):
            t_lon = torch.tensor(t_lon)
        # if not torch.jit.isinstance(t_lat, torch.Tensor):
        if not isinstance(t_lat, torch.Tensor):
            t_lat = torch.tensor(t_lat)
        
        rot_axis = (torch.cos(t_lon+np.pi/2), torch.sin(t_lon+np.pi/2), 0)
        rot_angle = t_lat-np.pi/2
        return self.rotate_spherical(lon, lat, rot_axis, rot_angle, cart=cart)
        
    
    def uvwha2corners(self, u, v, w, h, a):
        '''
        calculate corner coordinates from uvwha bbox information
        ''' 
        center = torch.tensor([[u],[v]])
        corners = torch.tensor([[-h/2, -w/2],
                                [-h/2, w/2],
                                [h/2, -w/2],
                                [h/2, w/2]]).t()
        
        # rotate corners by given angle
        cos_a = torch.cos(a * np.pi/180)
        sin_a = torch.sin(a * np.pi/180)
        R = torch.tensor([[cos_a, -sin_a],
                          [sin_a, cos_a]])
        
        corners = torch.mm(R,corners) + center
        return corners
    
    def corners2fov(self, center, corners):
        '''
        calculate fov of tangent patch and longitude rotation angle
        '''
        center_and_corners = torch.cat((center,corners),dim=1)
        lon, lat = self.fisheye2sphere(center_and_corners[0], center_and_corners[1])
        
        # calculate maximum fov needed       
        fov = 2 * (np.pi/2 - torch.min(self.move2pole(lon[1:], lat[1:], lon[0], lat[0])[1]))
        
        # calculate longitude rotation angle to make image upright
        # lon_rot = np.pi - lon[0]
        return fov
    
    def patch_of_sphere(self, height, width, fov, center):
        '''
        make patch sized 'height x width' that each pixel of which contains
        spherical(geographic) coordinates of virtual sphere
        '''
        # sensor pitch
        d = self.sensor_height/height
        
        # principal point
        u0 = (height-1)/2
        v0 = (width-1)/2
        
        # calculate focal length correspond to patch width and height
        f = self.fov2f(d * np.sqrt(np.square(u0)+np.square(v0)), fov.numpy(), k=1)
        
        # u, v denote pixel coordinates in tangent image
        u, v = torch.meshgrid(torch.arange(height), torch.arange(width))#, indexing='ij')
        
        # U, V denote displacement from principal point in mm
        U = d * (u - u0)
        V = d * (v - v0)
        
        # calculate spherical coordinate of center point
        lon_c, lat_c = self.fisheye2sphere(center[0], center[1])
        
        # calculate longitude rotation angle to make image upright
        lon_rot = np.pi - lon_c
        
        # spherical coordinate of virtual sphere whose pole is set as bbox center
        # need to invert(-) longitude axis to match the direction on image with the direction on virtual sphere
        rotated_lon = - torch.atan2(V, U) - lon_rot
        rotated_lat = np.pi/2 - torch.atan2(torch.sqrt(U.square()+V.square()),f)
        
        # spherical coordinate of virtual sphere
        lon, lat = self.move2pole(rotated_lon, rotated_lat, lon_c + np.pi, lat_c)
        
        return torch.stack((lon, lat),dim=0)
        
    def visualize_patches(self, patches, k_values, detectnet): 
        ncols = int(np.round(np.sqrt(len(patches))))
        nrows = len(patches)//ncols + 1

        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
        for row in range(nrows):
            for col in range(ncols):
                idx = row * ncols + col
                if idx < len(patches):
                    axes[row, col].imshow(patches[idx].permute(1,2,0))
                    axes[row, col].set_xticklabels([])
                    axes[row, col].set_yticklabels([])
                    if detectnet:
                        axes[row, col].set_title(f'k = {int(k_values[idx]):d} mm')
                else:
                    axes[row, col].remove()
        plt.tight_layout()
    
    def get_tangent_patch(self, uvwha,
                          visualize=False, 
                          detectnet=False, 
                          height=None, 
                          width=None, 
                          scale=None):
        '''
        compute tangent image patch form uvwha bboxes
        uvwha.shape     : [N, 5(uvwha)] 
        return          : patches, lons, lats
        patch.shape     : [N, C, H, W]
        sphericals.shape: [N, 2(lon, lat), H, W]
        k_values.shape  : [N]
        '''
        if height is None:
            height = self.patch_height
        if width is None:
            width = self.patch_width
        if scale is None:
            scale = self.bbox_scale
        
        patches = []
        sphericals = []
        k_values = []
        import time
        for u, v, w, h, a in uvwha:

            # scale up  width and height
            w = w*scale
            h = h*scale
            
            # center and corner coordinates on fisheye image
            center = torch.tensor([[u],[v]])


            corners = self.uvwha2corners(u, v, w, h, a)
            
            # fov and longitude rotation
            fov = self.corners2fov(center, corners)

            # spherical(geographic) coordinates of virtual sphere

            lon, lat = self.patch_of_sphere(height, width, fov, center)
            if torch.cuda.is_available():
                lon = lon.cuda()
                lat = lat.cuda()

            if detectnet:
                sphericals.append(torch.stack((lon, lat), dim=0))
                k = self.cam_height * torch.tan(np.pi/2 - lat[int(height/2),int(width/2)])
                k_values.append(k)

            # fisheye image pixel coordinate
            grid_u, grid_v = self.sphere2fisheye(lon, lat)

            # scale each u, v axis of grid to [-1, 1]
            scaled_u = (grid_u - self.u0) / self.u0
            scaled_v = (grid_v - self.v0) / self.v0
            grid = torch.stack((scaled_u, scaled_v), dim=-1).unsqueeze(0)

            # get tangent patch
            patch = F.grid_sample(self.imgs, grid, mode='bilinear', align_corners=True).squeeze(0)



            patches.append(patch)
        if visualize:
            self.visualize_patches(patches, k_values, detectnet=detectnet)
        
        if detectnet:
            return patches, sphericals, k_values
        
        return patches