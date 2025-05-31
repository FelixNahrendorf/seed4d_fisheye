#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
# Modified for standalone use without CARLA dependency
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Dict, Union
import os


class BaseProjection:
    """Base projection class for fisheye camera models."""
    
    def __init__(self):
        pass
    
    @classmethod
    def from_fov(cls, width: int, height: int, fov: float, k0: float = 0.0, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0):
        """Constructor from fov."""
        raise NotImplementedError
    
    def from_3d_to_2d(self, points3d: np.ndarray) -> np.ndarray:
        """The camera projection from 3D to 2D image."""
        raise NotImplementedError
    
    def from_2d_to_3d(self, pixels_coords: np.ndarray) -> np.ndarray:
        """The inverse projection from 2d image to 3d space."""
        raise NotImplementedError


class EquidistantProjection(BaseProjection):
    """Equidistant fisheye projection model."""
    
    def __init__(self, fx: float, fy: float, cx: float, cy: float, k0: float = 0.0, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0):
        super().__init__()
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
    
    @classmethod
    def from_fov(cls, width: int, height: int, fov: float, k0: float = 0.0, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0):
        """Create equidistant projection from field of view."""
        cx = width / 2.0
        cy = height / 2.0
        # For equidistant projection: r = f * theta
        # At the edge: r = min(width, height) / 2, theta = fov/2
        r_max = min(width, height) / 2.0
        theta_max = np.deg2rad(fov / 2.0)
        fx = fy = r_max / theta_max
        return cls(fx, fy, cx, cy, k0, k1, k2, k3, k4)
    
    def from_2d_to_3d(self, pixels_coords: np.ndarray) -> np.ndarray:
        """Convert 2D pixels to 3D unit rays."""
        # Normalize coordinates
        x_norm = (pixels_coords[..., 0] - self.cx) / self.fx
        y_norm = (pixels_coords[..., 1] - self.cy) / self.fy
        
        # Calculate theta (angle from optical axis)
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = r  # For equidistant projection
        
        # Convert to 3D coordinates
        phi = np.arctan2(y_norm, x_norm)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        return np.stack([x, y, z], axis=-1)
    
    def from_3d_to_2d(self, points3d: np.ndarray) -> np.ndarray:
        """Convert 3D points to 2D pixels."""
        # Normalize to unit sphere
        norm = np.linalg.norm(points3d, axis=-1, keepdims=True)
        points3d_norm = points3d / (norm + 1e-8)
        
        x, y, z = points3d_norm[..., 0], points3d_norm[..., 1], points3d_norm[..., 2]
        
        # Calculate theta and phi
        theta = np.arccos(np.clip(z, -1, 1))
        phi = np.arctan2(y, x)
        
        # For equidistant projection: r = f * theta
        r = theta
        
        # Convert to image coordinates
        x_norm = r * np.cos(phi)
        y_norm = r * np.sin(phi)
        
        u = self.fx * x_norm + self.cx
        v = self.fy * y_norm + self.cy
        
        return np.stack([u, v], axis=-1)


class FisheyeGenerator:
    """Standalone fisheye image generator from 5 pinhole images."""
    
    def __init__(self, fisheye_width: int = 640, fisheye_height: int = 640, fov: int = 180, 
                 k0: float = 0.0, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0):
        """
        Initialize fisheye generator.
        
        Args:
            fisheye_width: Output fisheye image width
            fisheye_height: Output fisheye image height
            fov: Field of view in degrees
            k0-k4: Distortion coefficients (currently not used in equidistant model)
        """
        self.fisheye_width = fisheye_width
        self.fisheye_height = fisheye_height
        self.fov = fov
        
        # Initialize projection model
        self.projection_model = EquidistantProjection.from_fov(
            width=fisheye_width, 
            height=fisheye_height, 
            fov=fov, 
            k0=k0, k1=k1, k2=k2, k3=k3, k4=k4
        )
        
        # Assume pinhole cameras have FOV=90 degrees
        # Calculate pinhole dimensions based on fisheye focal length
        self.pinhole_width = int(2.0 * self.projection_model.fx)
        self.pinhole_height = int(2.0 * self.projection_model.fy)
        
        # Create pinhole intrinsic matrix (FOV=90 degrees)
        self.pinhole_intrinsic = np.eye(3)
        self.pinhole_intrinsic[0, 2] = self.pinhole_width / 2.0
        self.pinhole_intrinsic[1, 2] = self.pinhole_height / 2.0
        self.pinhole_intrinsic[0, 0] = self.pinhole_intrinsic[1, 1] = self.pinhole_width / 2.0  # tan(45°) = 1
        
        # Compute mapping table
        self.maptable = self._compute_mapping()
    
    def _compute_mapping(self) -> np.ndarray:
        """Compute mapping for inverse warping between 5 pinhole to fisheye."""
        
        # Get fisheye image coordinates
        y, x = np.meshgrid(range(self.fisheye_height), range(self.fisheye_width), indexing='ij')
        fisheye_image_coords = np.concatenate((x[..., None], y[..., None]), axis=-1)
        shape = fisheye_image_coords.shape
        fisheye_image_coords = fisheye_image_coords.reshape(-1, 2)
        
        maptable = np.zeros_like(fisheye_image_coords).T
        
        # Convert fisheye pixels to 3D rays
        fisheye_rays = self.projection_model.from_2d_to_3d(fisheye_image_coords)
        fisheye_rays = fisheye_rays.T
        
        # Process each camera direction
        cameras = {
            'front': {'mask': np.ones(shape[0] * shape[1], dtype=bool), 'box_offset': [2 * self.pinhole_width, 0]},
            'left': {'mask': fisheye_image_coords[:, 0] <= self.fisheye_width / 2.0, 'box_offset': [0, 0]},
            'right': {'mask': fisheye_image_coords[:, 0] > self.fisheye_width / 2.0, 'box_offset': [4 * self.pinhole_width, 0]},
            'top': {'mask': fisheye_image_coords[:, 1] <= self.fisheye_height / 2.0, 'box_offset': [self.pinhole_width, 0]},
            'bottom': {'mask': fisheye_image_coords[:, 1] > self.fisheye_height / 2.0, 'box_offset': [3 * self.pinhole_width, 0]}
        }
        
        for direction, config in cameras.items():
            mask, coords = self._get_coordinates_for_pinhole_image(
                fisheye_rays, config['mask'], direction, config['box_offset']
            )
            maptable[:, mask] = coords
        
        return maptable.T.reshape(shape).astype(np.float32)
    
    def _get_coordinates_for_pinhole_image(self, fisheye_rays: np.ndarray, camera_mask: np.ndarray, 
                                         camera_direction: str, box_offset: list, margin: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinates for the box image for given camera direction."""
        
        # Define camera transformations
        transforms = {
            'front': np.eye(3),
            'left': R.from_euler('xyz', [0.0, 90, 0.0], degrees=True).as_matrix(),
            'right': R.from_euler('xyz', [0.0, -90, 0.0], degrees=True).as_matrix(),
            'top': R.from_euler('xyz', [-90, 0.0, 0.0], degrees=True).as_matrix(),
            'bottom': R.from_euler('xyz', [90, 0.0, 0.0], degrees=True).as_matrix()
        }
        
        cam_transform = transforms[camera_direction]
        box_idx = np.array(box_offset)[:, None]
        
        # Apply camera mask
        masked_rays = fisheye_rays[:, camera_mask].copy()
        
        # Transform rays and project to pinhole camera
        transform = self.pinhole_intrinsic @ cam_transform
        cam_img_coords = transform @ masked_rays
        cam_img_coords = cam_img_coords[:2, :] / (cam_img_coords[2, :] + 1e-8)
        
        # Filter coordinates within image bounds (with margin)
        mask_bounds = ((cam_img_coords[0] >= -margin) & 
                      (cam_img_coords[0] < self.pinhole_width + margin) & 
                      (cam_img_coords[1] >= -margin) & 
                      (cam_img_coords[1] < self.pinhole_height + margin))
        
        cam_img_coords = cam_img_coords[:, mask_bounds]
        
        # Clamp coordinates to image bounds
        cam_img_coords[0] = np.clip(cam_img_coords[0], 0, self.pinhole_width - 1)
        cam_img_coords[1] = np.clip(cam_img_coords[1], 0, self.pinhole_height - 1)
        
        # Add box offset for cube mapping
        cam_img_coords += box_idx
        
        # Update camera mask
        camera_mask[camera_mask] = mask_bounds
        
        return camera_mask, cam_img_coords
    
    def generate_fisheye(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate fisheye image from 5 pinhole images.
        
        Args:
            images: Dictionary with keys ['front', 'left', 'right', 'top', 'bottom']
                   Each value should be a numpy array of shape (H, W, 3)
        
        Returns:
            fisheye_image: Generated fisheye image as numpy array
        """
        required_keys = ['front', 'left', 'right', 'top', 'bottom']
        if not all(key in images for key in required_keys):
            raise ValueError(f"Images dictionary must contain keys: {required_keys}")
        
        # Resize images to expected pinhole dimensions if needed
        resized_images = {}
        for key, img in images.items():
            if img.shape[:2] != (self.pinhole_height, self.pinhole_width):
                resized_images[key] = cv2.resize(img, (self.pinhole_width, self.pinhole_height))
            else:
                resized_images[key] = img
        
        # Create the 5-image panorama (left, top, front, bottom, right)
        five_pinhole_image = np.hstack([
            resized_images['left'],
            resized_images['top'], 
            resized_images['front'],
            resized_images['bottom'],
            resized_images['right']
        ]).astype(np.float32)
        
        # Apply remapping to create fisheye image
        fisheye_image = cv2.remap(
            five_pinhole_image, 
            self.maptable[..., 0], 
            self.maptable[..., 1], 
            cv2.INTER_LINEAR
        )
        
        return fisheye_image.astype(np.uint8)
    
    def generate_fisheye_from_files(self, image_paths: Dict[str, str]) -> np.ndarray:
        """
        Generate fisheye image from 5 pinhole image files.
        
        Args:
            image_paths: Dictionary with keys ['front', 'left', 'right', 'top', 'bottom']
                        Each value should be a file path to an image
        
        Returns:
            fisheye_image: Generated fisheye image as numpy array
        """
        images = {}
        for key, path in image_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            images[key] = cv2.imread(path)
            if images[key] is None:
                raise ValueError(f"Could not load image: {path}")
            # Convert BGR to RGB
            images[key] = cv2.cvtColor(images[key], cv2.COLOR_BGR2RGB)
        
        return self.generate_fisheye(images)


# Example usage and utility functions
def demo_with_sample_images():
    """Demo function showing how to use the FisheyeGenerator."""
    
    # Create generator
    generator = FisheyeGenerator(fisheye_width=640, fisheye_height=640, fov=180)
    
    # Create sample images (you would replace this with actual image loading)
    sample_images = {}
    colors = {
        'front': [255, 0, 0],    # Red
        'left': [0, 255, 0],     # Green  
        'right': [0, 0, 255],    # Blue
        'top': [255, 255, 0],    # Yellow
        'bottom': [255, 0, 255]  # Magenta
    }
    
    for direction, color in colors.items():
        # Create a solid color image with text
        img = np.full((generator.pinhole_height, generator.pinhole_width, 3), color, dtype=np.uint8)
        # Add text to identify the direction (requires PIL or cv2 with text support)
        cv2.putText(img, direction.upper(), (50, generator.pinhole_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        sample_images[direction] = img
    
    # Generate fisheye image
    fisheye_img = generator.generate_fisheye(sample_images)
    
    # Save result
    cv2.imwrite('fisheye_output.jpg', cv2.cvtColor(fisheye_img, cv2.COLOR_RGB2BGR))
    print("Fisheye image saved as 'fisheye_output.jpg'")
    
    return fisheye_img


def main():
    '''Main function to run the fisheye generator with actual image files.'''

    generator = FisheyeGenerator(fisheye_width=640, fisheye_height=640, fov=180)

    image_paths = {
        'front': '/seed4d/data/Town01/ClearNoon/vehicle.audi.tt/spawn_point_1/step_0/ego_vehicle/nuscenes/sensors/0_rgb.png',
        'left': '/seed4d/data/Town01/ClearNoon/vehicle.audi.tt/spawn_point_1/step_0/ego_vehicle/nuscenes/sensors/2_rgb.png', 
        'right': '/seed4d/data/Town01/ClearNoon/vehicle.audi.tt/spawn_point_1/step_0/ego_vehicle/nuscenes/sensors/1_rgb.png',
        'top': '/seed4d/data/Town01/ClearNoon/vehicle.audi.tt/spawn_point_1/step_0/ego_vehicle/nuscenes/sensors/3_rgb.png',
        'bottom': '/seed4d/data/Town01/ClearNoon/vehicle.audi.tt/spawn_point_1/step_0/ego_vehicle/nuscenes/sensors/4_rgb.png'
    }
    fisheye_img = generator.generate_fisheye_from_files(image_paths)
    cv2.imwrite(os.path.join("/seed4d/fisheye_computation/images", 'fisheye_stitch_output.png'), cv2.cvtColor(fisheye_img, cv2.COLOR_RGB2BGR))
    

if __name__ == '__main__':
    # Run demo
    #demo_with_sample_images()

    # Run main
    main()