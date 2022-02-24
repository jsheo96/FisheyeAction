# dependancies are as follows
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision import transforms as transforms

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
        fov : fov angle in radians [B]
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
        
        return torch.stack((X, Y, Z), dim=-1)
    
    def cartesian2sphere(self, X, Y, Z):
        '''
        convert cartesian coordinates to spherical(geographic) coordinates
        '''
        lon = torch.atan2(Y, X)
        lat = torch.atan2(Z, torch.sqrt(X.square()+Y.square()))
        return torch.stack((lon, lat), dim=-1)
        
    def fisheye2sphere(self, uv, k=-0.5, cart=False):
        '''
        input: uv.shape: [N, 2]
        output: sphere.shape: [N, 2]

        convert fisheye image pixel coordinates to spherical(geographic) coordinates of the virtual sphere
        based on PTGui 11 fisheye projection with fisheye factor k (https://wiki.panotools.org/Fisheye_Projection)
        k = -1.0 : orthographic
        k = -0.5 : equisolid
        k =  0.0 : equidistant
        k =  0.5 : stereographic
        k =  1.0 : rectilinear (perspective)
        '''
        if not isinstance(uv, torch.Tensor):
            uv = torch.tensor(uv)
            
        # U, V denote displacement from principal point in mm
        uv0 = torch.tensor((self.u0, self.v0)).unsqueeze(0)
        
        UV = (uv - uv0) * self.d
        V = UV[:, 1]
        U = UV[:, 0]
        
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
            cartesian = self.sphere2cartesian(lon, lat)  # [N, 3]
            return cartesian
        else:
            sphere = torch.stack((lon, lat), dim=-1)  # [N, 2]
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
        return torch.stack((u, v), dim=-1)
    
    def rotate_spherical(self, lon, lat, rot_axis, rot_angle, cart=False):
        '''
        lon, lat: [B, N]
        rot_axis: [B, 3]
        rot_angle: [B]
        rotate spherical(geographic) coordinates following Rodrigues rotation formula
        '''
        # Rodrigues rotation formula
        kx, ky, kz = rot_axis.permute((1,0))  # rotation axis [B]
        zeros_like_k = torch.zeros_like(kx)
        K = torch.stack((torch.stack((zeros_like_k, kz, -ky), dim=1),  # [B, 3, 3]
                         torch.stack((-kz, zeros_like_k, kx), dim=1),
                         torch.stack((ky, -kx, zeros_like_k), dim=1)), dim=1)
        eye = torch.eye(3).unsqueeze(0).repeat(kx.shape[0], 1, 1)  # [B, 3, 3]

        rot_angle = rot_angle.view((rot_angle.shape[0], 1, 1))
        R = eye + torch.sin(rot_angle)*K + (1-torch.cos(rot_angle))*torch.bmm(K,K)

        XYZ = self.sphere2cartesian(torch.flatten(lon), torch.flatten(lat))
        XYZ = XYZ.view((kx.shape[0], -1, 3))  # [B, N, 3(x,y,z)]
        XYZ = torch.bmm(XYZ, R)  # [B, N, 3]
        
        if not cart:
            XYZ_flat = XYZ.view((-1, 3))
            lonlat = self.cartesian2sphere(XYZ_flat[:, 0], XYZ_flat[:, 1], XYZ_flat[:, 2])  # [B*N, 2(lon, lat)]
            lonlat = lonlat.view((kx.shape[0], -1, 2))  # [B, N, 2]
            return lonlat
        else:
            return XYZ

    def move2pole(self, lon, lat, t_lon, t_lat, cart=False):
        '''
        lon, lat: [B, N]
        t_lon, t_lat: [B]
        rotate spherical coordinates for target point to be moved to pole
        '''
        assert t_lon.shape == t_lat.shape
        # if not torch.jit.isinstance(t_lon, torch.Tensor):
        if not isinstance(t_lon, torch.Tensor):
            t_lon = torch.tensor(t_lon)
        # if not torch.jit.isinstance(t_lat, torch.Tensor):
        if not isinstance(t_lat, torch.Tensor):
            t_lat = torch.tensor(t_lat)
        
        rot_axis = torch.stack((torch.cos(t_lon+np.pi/2),  # [B, 3]
                                torch.sin(t_lon+np.pi/2), 
                                torch.zeros_like(t_lon)), dim=-1)
        rot_angle = t_lat-np.pi/2  # [B]
        return self.rotate_spherical(lon, lat, rot_axis, rot_angle, cart=cart)
        
    
    def uvwha2corners(self, uvwha):
        '''
        calculate corner coordinates from uvwha bbox information
        uvwha.shape : [B, 5]
        ''' 
        assert uvwha.dim()==2, 'variable uvwha must contain batches'
        hw = torch.flip(uvwha[:,2:4], dims=[1])  # [B, 2]
        hw_sign = hw.clone().detach()
        hw_sign[:,0] *= -1
        corners = torch.stack((-hw/2, hw_sign/2, -hw_sign/2, hw/2), dim=1)  # [B, 4, 2]

        # rotate corners by given angle
        cos_a = torch.cos(uvwha[:,4] * np.pi/180)
        sin_a = torch.sin(uvwha[:,4] * np.pi/180)  # [B]
        # rotation matric [B, 2, 2]
        R = torch.stack((torch.stack((cos_a, sin_a), dim=1),
                         torch.stack((-sin_a, cos_a), dim=1)), dim=1)
        
        center = uvwha[:,:2].unsqueeze(1)  # [B, 1, 2]
        corners = torch.bmm(corners, R) + center
        return corners
    
    def corners2fov(self, center, corners):
        '''
        calculate fov of tangent patch and longitude rotation angle
        '''
        center_and_corners = torch.cat((center,corners),dim=1)  # [B, 5(center_and_corners), 2]
        lonlat = self.fisheye2sphere(center_and_corners.view((-1, 2))).view((-1, 5, 2))
        lon = lonlat[:, :, 0]  # [B, 5]
        lat = lonlat[:, :, 1]  # [B, 5]
        # calculate maximum fov needed       
        moved_corners = self.move2pole(lon[:, 1:], lat[:, 1:], lon[:, 0], lat[:, 0])  # [B, 4, 2(lon, lat)]
        fov = 2 * (-torch.min(moved_corners[:,:,1], dim=1).values + np.pi/2)
        
        # calculate longitude rotation angle to make image upright
        # lon_rot = np.pi - lon[0]
        return fov
    
    def patch_of_sphere(self, height, width, fov, center):
        '''
        fov.shape: [B]
        center.shape: [B, 2]
        make patch sized 'height x width' that each pixel of which contains
        spherical(geographic) coordinates of virtual sphere
        '''
        # sensor pitch
        d = self.sensor_height/height
        
        # principal point
        u0 = (height-1)/2
        v0 = (width-1)/2
        
        # calculate focal length correspond to patch width and height
        f = self.fov2f(d * np.sqrt(np.square(u0)+np.square(v0)), fov, k=1)  # [B]
        
        # u, v denote pixel coordinates in tangent image
        u, v = torch.meshgrid(torch.arange(height), torch.arange(width))#, indexing='ij')
        u = u.unsqueeze(0).repeat(f.shape[0], 1, 1)  # [B, height, width]
        v = v.unsqueeze(0).repeat(f.shape[0], 1, 1)  # [B, height, width]
        
        # U, V denote displacement from principal point in mm
        U = d * (u - u0)  # [B, height, width]
        V = d * (v - v0)  # [B, height, width]
        
        # calculate spherical coordinate of center point
        lonlat_c = self.fisheye2sphere(center)  # [B, 2]
        
        # calculate longitude rotation angle to make image upright
        lon_rot = np.pi - lonlat_c[:, 0]  # [B]
        
        # spherical coordinate of virtual sphere whose pole is set as bbox center
        # need to invert(-) longitude axis to match the direction on image with the direction on virtual sphere
        rotated_lon = - torch.atan2(V, U) - lon_rot.reshape((lon_rot.shape[0], 1, 1))  # [B, height, width]
        rotated_lat = np.pi/2 - torch.atan2(torch.sqrt(U.square()+V.square()),f.reshape((f.shape[0], 1, 1)))  # [B, height, width]
        
        # spherical coordinate of virtual sphere
        lonlat = self.move2pole(rotated_lon.view((rotated_lon.shape[0], -1)), rotated_lat.view((rotated_lat.shape[0], -1)),
                                lonlat_c[:, 0] + np.pi, lonlat_c[:, 1])
        lonlat = lonlat.view((f.shape[0], height, width, 2))
        
        return lonlat
        
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
                          scale=None,
                          isbatch=True):
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

        if isinstance(uvwha, torch.Tensor) and isbatch:
            # scale up  width and height
            uvwha[:,2] *= scale
            uvwha[:,3] *= scale
            
            # center and corner coordinates on fisheye image
            center = uvwha[:,:2]
            corners = self.uvwha2corners(uvwha)

            # fov and longitude rotation
            fov = self.corners2fov(center.unsqueeze(1), corners)

            # spherical(geographic) coordinates of virtual sphere
            lonlat = self.patch_of_sphere(height, width, fov, center)  # [B, height, width, 2]
            
            if torch.cuda.is_available():
                lonlat = lonlat.cuda()

            # fisheye image pixel coordinate
            grid = self.sphere2fisheye(lonlat[:,:,:,0], lonlat[:,:,:,1])  # [B, height, width, 2]

            # scale each u, v axis of grid to [-1, 1]
            grid[:,:,:,0] = (grid[:,:,:,0] - self.u0) / self.u0
            grid[:,:,:,1] = (grid[:,:,:,1] - self.v0) / self.v0

            # get tangent patch
            patches = F.grid_sample(self.imgs.repeat(grid.shape[0], 1, 1, 1), grid, mode='bilinear', align_corners=True)

            if visualize:
                self.visualize_patches(patches, k_values, detectnet=detectnet)
            
            if detectnet:
                sphericals = lonlat
                k_values = self.cam_height * torch.tan(np.pi/2 - lonlat[:, int(height/2), int(width/2), 1])
                return patches, sphericals, k_values
            else:
                return patches
        else:
            patches = []
            sphericals = []
            k_values = []
            for uvwha_1 in uvwha:

                # scale up  width and height
                uvwha_1[2:4] *= scale
                
                # center and corner coordinates on fisheye image
                center = uvwha_1[:2].unsqueeze(0)
                corners = self.uvwha2corners(uvwha_1.unsqueeze(0))
                
                # fov and longitude rotation
                fov = self.corners2fov(center.unsqueeze(1), corners)

                # spherical(geographic) coordinates of virtual sphere

                lonlat = self.patch_of_sphere(height, width, fov, center)
                if torch.cuda.is_available():
                    lonlat = lonlat.cuda()

                # fisheye image pixel coordinate
                grid = self.sphere2fisheye(lonlat[:,:,:,0], lonlat[:,:,:,1])  # [B, height, width, 2]

                # scale each u, v axis of grid to [-1, 1]
                grid[:,:,:,0] = (grid[:,:,:,0] - self.u0) / self.u0
                grid[:,:,:,1] = (grid[:,:,:,1] - self.v0) / self.v0

                # get tangent patch
                patch = F.grid_sample(self.imgs.repeat(grid.shape[0], 1, 1, 1), grid, mode='bilinear', align_corners=True).squeeze(0)
                patches.append(patch)

            if visualize:
                self.visualize_patches(patches, k_values, detectnet=detectnet)
            if detectnet:
                return patches, sphericals, k_values
            else:
                return patches
