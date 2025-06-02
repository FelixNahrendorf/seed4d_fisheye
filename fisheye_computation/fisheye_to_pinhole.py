#!/usr/bin/env python

# Fisheye to Pinhole Image Converter
# Converts a single fisheye image to 3 optimally positioned pinhole images
# to maximize information coverage and minimize loss

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Dict, List
import os
import json
import argparse


class EquidistantProjection:
    """Equidistant fisheye projection model for inverse mapping."""
    
    def __init__(self, fx: float, fy: float, cx: float, cy: float, 
                 k0: float = 0.0, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0):
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
    def from_fov(cls, width: int, height: int, fov: float, 
                 k0: float = 0.0, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0):
        """Create equidistant projection from field of view."""
        cx = width / 2.0
        cy = height / 2.0
        # For equidistant projection: r = f * theta
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


class FisheyeToPinholeConverter:
    """Converts fisheye images to optimally positioned pinhole images."""
    
    def __init__(self, fisheye_width: int = 640, fisheye_height: int = 640, fisheye_fov: int = 180,
                 pinhole_width: int = 640, pinhole_height: int = 640, pinhole_fov: int = 90):
        """
        Initialize the converter.
        
        Args:
            fisheye_width: Input fisheye image width
            fisheye_height: Input fisheye image height  
            fisheye_fov: Fisheye field of view in degrees
            pinhole_width: Output pinhole image width
            pinhole_height: Output pinhole image height
            pinhole_fov: Pinhole field of view in degrees
        """
        self.fisheye_width = fisheye_width
        self.fisheye_height = fisheye_height
        self.fisheye_fov = fisheye_fov
        self.pinhole_width = pinhole_width
        self.pinhole_height = pinhole_height
        self.pinhole_fov = pinhole_fov
        
        # Initialize fisheye projection model
        self.fisheye_projection = EquidistantProjection.from_fov(
            width=fisheye_width,
            height=fisheye_height,
            fov=fisheye_fov
        )
        
        # Create pinhole camera intrinsic matrix
        self.pinhole_intrinsic = self._create_pinhole_intrinsic()
        
        # Define optimal camera orientations for maximum coverage
        # These orientations are designed to minimize overlap while maximizing coverage
        self.camera_orientations = self._get_optimal_orientations()
        
        # Pre-compute mapping tables for each camera
        self.mapping_tables = self._compute_all_mappings()
    
    def _create_pinhole_intrinsic(self) -> np.ndarray:
        """Create pinhole camera intrinsic matrix."""
        cx = self.pinhole_width / 2.0
        cy = self.pinhole_height / 2.0
        
        # Calculate focal length from FOV
        f = (self.pinhole_width / 2.0) / np.tan(np.deg2rad(self.pinhole_fov / 2.0))
        
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        return K
    
    def _get_optimal_orientations(self) -> Dict[str, np.ndarray]:
        """
        Get optimal camera orientations for 3 pinhole cameras.
        
        The orientations are chosen to maximize coverage while minimizing overlap:
        - Front: Looking straight ahead (0°)
        - Left-Front: 45° to the left and slightly up
        - Right-Front: 45° to the right and slightly up
        
        This configuration captures most of the front hemisphere with good overlap
        for stereo vision while minimizing information loss.
        """
        orientations = {
            # Front camera - straight ahead
            'front': np.eye(3),
            
            # Left-front camera - 45° left, 15° up for better coverage
            'left_front': R.from_euler('xyz', [15, 45, 0], degrees=True).as_matrix(),
            
            # Right-front camera - 45° right, 15° up for better coverage  
            'right_front': R.from_euler('xyz', [15, -45, 0], degrees=True).as_matrix()
        }
        
        return orientations
    
    def _compute_mapping_for_camera(self, camera_name: str, orientation: np.ndarray) -> np.ndarray:
        """Compute mapping table for a specific camera orientation."""
        
        # Create pinhole image coordinate grid
        y, x = np.meshgrid(range(self.pinhole_height), range(self.pinhole_width), indexing='ij')
        pinhole_coords = np.stack([x, y], axis=-1).astype(np.float32)
        
        # Convert pinhole coordinates to 3D rays
        ones = np.ones((self.pinhole_height, self.pinhole_width, 1))
        pinhole_coords_homo = np.concatenate([pinhole_coords, ones], axis=-1)
        
        # Apply inverse intrinsic matrix
        K_inv = np.linalg.inv(self.pinhole_intrinsic)
        rays_camera = np.tensordot(pinhole_coords_homo, K_inv.T, axes=([2], [0]))
        
        # Normalize rays to unit length
        rays_norm = np.linalg.norm(rays_camera, axis=-1, keepdims=True)
        rays_camera = rays_camera / (rays_norm + 1e-8)
        
        # Apply camera orientation (inverse transform to world coordinates)
        orientation_inv = orientation.T
        rays_world = np.tensordot(rays_camera, orientation_inv.T, axes=([2], [0]))
        
        # Project 3D rays to fisheye coordinates using the fisheye projection model
        fisheye_coords = self._project_rays_to_fisheye(rays_world)
        
        return fisheye_coords
    
    def _project_rays_to_fisheye(self, rays: np.ndarray) -> np.ndarray:
        """Project 3D rays to fisheye image coordinates."""
        # Normalize rays
        rays_norm = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays = rays / (rays_norm + 1e-8)
        
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        
        # Calculate theta and phi
        theta = np.arccos(np.clip(z, -1, 1))
        phi = np.arctan2(y, x)
        
        # For equidistant projection: r = f * theta
        r = theta
        
        # Convert to normalized coordinates  
        x_norm = r * np.cos(phi)
        y_norm = r * np.sin(phi)
        
        # Convert to image coordinates
        u = self.fisheye_projection.fx * x_norm + self.fisheye_projection.cx
        v = self.fisheye_projection.fy * y_norm + self.fisheye_projection.cy
        
        # Create mapping coordinates
        coords = np.stack([u, v], axis=-1).astype(np.float32)
        
        # Mask out coordinates outside fisheye image bounds
        mask = ((u >= 0) & (u < self.fisheye_width) & 
                (v >= 0) & (v < self.fisheye_height))
        
        # Set invalid coordinates to -1 (will be handled by cv2.remap)
        coords[~mask] = -1
        
        return coords
    
    def _compute_all_mappings(self) -> Dict[str, np.ndarray]:
        """Compute mapping tables for all camera orientations."""
        mappings = {}
        
        print("Computing mapping tables...")
        for camera_name, orientation in self.camera_orientations.items():
            print(f"  - Computing mapping for {camera_name} camera...")
            mappings[camera_name] = self._compute_mapping_for_camera(camera_name, orientation)
        
        print("Mapping computation complete.")
        return mappings
    
    def convert_fisheye_to_pinhole(self, fisheye_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert fisheye image to multiple pinhole images.
        
        Args:
            fisheye_image: Input fisheye image as numpy array
            
        Returns:
            Dictionary of pinhole images with camera names as keys
        """
        if fisheye_image.shape[:2] != (self.fisheye_height, self.fisheye_width):
            print(f"Resizing fisheye image from {fisheye_image.shape[:2]} to ({self.fisheye_height}, {self.fisheye_width})")
            fisheye_image = cv2.resize(fisheye_image, (self.fisheye_width, self.fisheye_height))
        
        pinhole_images = {}
        
        print("Converting fisheye to pinhole images...")
        for camera_name, mapping in self.mapping_tables.items():
            print(f"  - Generating {camera_name} pinhole image...")
            
            # Apply remapping to extract pinhole view
            pinhole_img = cv2.remap(
                fisheye_image,
                mapping[..., 0],
                mapping[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            pinhole_images[camera_name] = pinhole_img
        
        return pinhole_images
    
    def convert_fisheye_from_file(self, fisheye_path: str) -> Dict[str, np.ndarray]:
        """
        Convert fisheye image from file to pinhole images.
        
        Args:
            fisheye_path: Path to fisheye image file
            
        Returns:
            Dictionary of pinhole images
        """
        if not os.path.exists(fisheye_path):
            raise FileNotFoundError(f"Fisheye image not found: {fisheye_path}")
        
        fisheye_img = cv2.imread(fisheye_path)
        if fisheye_img is None:
            raise ValueError(f"Could not load fisheye image: {fisheye_path}")
        
        # Convert BGR to RGB for processing
        fisheye_img = cv2.cvtColor(fisheye_img, cv2.COLOR_BGR2RGB)
        
        return self.convert_fisheye_to_pinhole(fisheye_img)
    
    def save_pinhole_images(self, pinhole_images: Dict[str, np.ndarray], output_dir: str = "pinhole_output") -> None:
        """
        Save pinhole images to files.
        
        Args:
            pinhole_images: Dictionary of pinhole images
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving pinhole images to {output_dir}...")
        for camera_name, img in pinhole_images.items():
            # Convert RGB back to BGR for saving
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_dir, f"{camera_name}_pinhole.jpg")
            cv2.imwrite(output_path, img_bgr)
            print(f"  - Saved {camera_name} pinhole image: {output_path}")
    
    def save_configuration(self, output_path: str = "fisheye_to_pinhole_config.json") -> None:
        """Save converter configuration to JSON file."""
        
        config_data = {
            "fisheye_camera": {
                "width": self.fisheye_width,
                "height": self.fisheye_height,
                "fov_degrees": self.fisheye_fov,
                "projection_model": "equidistant",
                "fx": float(self.fisheye_projection.fx),
                "fy": float(self.fisheye_projection.fy),
                "cx": float(self.fisheye_projection.cx),
                "cy": float(self.fisheye_projection.cy)
            },
            "pinhole_cameras": {
                "width": self.pinhole_width,
                "height": self.pinhole_height,
                "fov_degrees": self.pinhole_fov,
                "fx": float(self.pinhole_intrinsic[0, 0]),
                "fy": float(self.pinhole_intrinsic[1, 1]),
                "cx": float(self.pinhole_intrinsic[0, 2]),
                "cy": float(self.pinhole_intrinsic[1, 2]),
                "intrinsic_matrix": self.pinhole_intrinsic.tolist()
            },
            "camera_orientations": {
                name: orientation.tolist() 
                for name, orientation in self.camera_orientations.items()
            },
            "metadata": {
                "description": "Configuration for fisheye to pinhole conversion",
                "coordinate_system": "opencv",
                "optimal_coverage": "3 cameras positioned for maximum information retention",
                "camera_positions": {
                    "front": "Straight ahead (0° yaw, 0° pitch)",
                    "left_front": "45° left, 15° up", 
                    "right_front": "45° right, 15° up"
                }
            }
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"Configuration saved to: {output_path}")
    
    def analyze_coverage(self, pinhole_images: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze the coverage of each pinhole image.
        
        Args:
            pinhole_images: Dictionary of pinhole images
            
        Returns:
            Dictionary with coverage statistics
        """
        coverage_stats = {}
        
        for camera_name, img in pinhole_images.items():
            # Calculate non-black pixel ratio as coverage metric
            if len(img.shape) == 3:
                # Color image - check if any channel is non-zero
                non_black = np.any(img > 5, axis=2)  # Small threshold to account for interpolation artifacts
            else:
                # Grayscale image
                non_black = img > 5
            
            coverage_ratio = np.sum(non_black) / (img.shape[0] * img.shape[1])
            coverage_stats[camera_name] = coverage_ratio
        
        # Calculate total unique coverage (approximate)
        total_coverage = min(1.0, sum(coverage_stats.values()) * 0.85)  # Adjust for overlap
        coverage_stats['total_estimated'] = total_coverage
        
        return coverage_stats


def create_demo_fisheye():
    """Create a demo fisheye image for testing."""
    print("Creating demo fisheye image...")
    
    # Create a synthetic fisheye image with different colored sectors
    height, width = 640, 640
    center_x, center_y = width // 2, height // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center and angle
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    angle = np.arctan2(y - center_y, x - center_x)
    
    # Create fisheye image with radial pattern
    fisheye_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Only fill within fisheye circle
    mask = dist <= min(width, height) // 2
    
    # Create radial color pattern
    angle_normalized = (angle + np.pi) / (2 * np.pi)  # Normalize to 0-1
    dist_normalized = dist / (min(width, height) // 2)  # Normalize to 0-1
    
    # Color pattern based on angle and distance
    fisheye_img[mask, 0] = (255 * (1 - dist_normalized) * np.sin(4 * angle + np.pi/4))[mask].clip(0, 255)  # Red
    fisheye_img[mask, 1] = (255 * (1 - dist_normalized) * np.sin(4 * angle + np.pi/2))[mask].clip(0, 255)  # Green  
    fisheye_img[mask, 2] = (255 * (1 - dist_normalized) * np.sin(4 * angle + 3*np.pi/4))[mask].clip(0, 255)  # Blue
    
    # Add some text markers
    cv2.putText(fisheye_img, 'FRONT', (center_x-30, center_y-100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(fisheye_img, 'LEFT', (center_x-150, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(fisheye_img, 'RIGHT', (center_x+100, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(fisheye_img, 'TOP', (center_x-20, center_y-200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save demo image
    cv2.imwrite('demo_fisheye.jpg', cv2.cvtColor(fisheye_img, cv2.COLOR_RGB2BGR))
    print("Demo fisheye image saved as 'demo_fisheye.jpg'")
    
    return fisheye_img


def main():
    """Main function to demonstrate fisheye to pinhole conversion."""
    parser = argparse.ArgumentParser(description='Convert fisheye image to pinhole images')
    parser.add_argument('--input', '-i', type=str, help='Input fisheye image path')
    parser.add_argument('--output', '-o', type=str, default='pinhole_output', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Run with demo fisheye image')
    parser.add_argument('--fisheye-fov', type=int, default=180, help='Fisheye FOV in degrees')
    parser.add_argument('--pinhole-fov', type=int, default=90, help='Pinhole FOV in degrees')
    parser.add_argument('--pinhole-size', type=int, nargs=2, default=[640, 640], help='Pinhole image size (width height)')
    
    args = parser.parse_args()
    
    # Create converter
    converter = FisheyeToPinholeConverter(
        fisheye_width=640,
        fisheye_height=640,
        fisheye_fov=args.fisheye_fov,
        pinhole_width=args.pinhole_size[0],
        pinhole_height=args.pinhole_size[1], 
        pinhole_fov=args.pinhole_fov
    )
    
    # Get fisheye image
    if args.demo or not args.input:
        print("Running demo mode...")
        fisheye_img = create_demo_fisheye()
        fisheye_img = cv2.cvtColor(fisheye_img, cv2.COLOR_BGR2RGB)  # Convert for processing
    else:
        print(f"Loading fisheye image from: {args.input}")
        pinhole_images = converter.convert_fisheye_from_file(args.input)
    
    # Convert fisheye to pinhole images
    if args.demo or not args.input:
        pinhole_images = converter.convert_fisheye_to_pinhole(fisheye_img)
    
    # Analyze coverage
    coverage_stats = converter.analyze_coverage(pinhole_images)
    print("\nCoverage Analysis:")
    for camera_name, coverage in coverage_stats.items():
        print(f"  {camera_name}: {coverage:.1%}")
    
    # Save results
    converter.save_pinhole_images(pinhole_images, args.output)
    converter.save_configuration(os.path.join(args.output, 'config.json'))
    
    print(f"\nConversion complete! Check '{args.output}' directory for results.")
    print(f"Total estimated coverage: {coverage_stats.get('total_estimated', 0):.1%}")


if __name__ == '__main__':
    main()
