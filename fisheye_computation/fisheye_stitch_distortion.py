import numpy as np
import cv2
import os
from typing import Tuple
from scipy.spatial.transform import Rotation as R
from camera_models.base_projection import BaseProjection
from camera_models.equidistant_projection import EquidistantProjection

class OfflineFisheyeCamera:
    def __init__(self, camera_model: BaseProjection, width: int = 640, height: int = 640, fov: int = 180,
                 k0: float = 0.0, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0):
        """Initialize Offline Fisheye Camera."""
        self.image = None
        self.frame = 0

        # Initialize fisheye projection model
        self.projection_model = camera_model.from_fov(width=width, height=height, fov=fov, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)

        # Intrinsic for pinhole images (assuming same size and FOV = 90)
        pinhole_width = int(2.0 * self.projection_model.fx)
        pinhole_height = int(2.0 * self.projection_model.fy)

        calibration = np.identity(3)
        calibration[0, 2] = float(pinhole_width) / 2.0
        calibration[1, 2] = float(pinhole_height) / 2.0
        calibration[0, 0] = calibration[1, 1] = float(pinhole_width) / (2.0 * np.tan(np.deg2rad(90 / 2.0)))

        self.pinhole_intrinsic_matrix = calibration

        # Compute the mapping table
        self.maptable = self.compute_mapping(width, height, self.projection_model, self.pinhole_intrinsic_matrix)

    def compute_mapping(self, fisheye_width: int, fisheye_height: int, projection_model: BaseProjection,
                        pinhole_intrisic_matrix: np.ndarray) -> np.ndarray:
        """Compute mapping for inverse warping between 5 pinhole to fisheye."""
        y, x = np.meshgrid(range(fisheye_height), range(fisheye_width), indexing='ij')
        fisheye_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
        shape = (fisheye_height, fisheye_width)

        rays = projection_model.from_2d_to_3d(fisheye_coords).T

        maptable = np.zeros((2, rays.shape[1]), dtype=np.float32)

        pinhole_width = int(2.0 * projection_model.fx)
        pinhole_height = int(2.0 * projection_model.fy)

        # Front
        mask = np.ones(rays.shape[1], dtype=bool)
        mask, coords = self.get_coords(rays, pinhole_width, pinhole_height, pinhole_intrisic_matrix, mask, "front")
        maptable[:, mask] = coords

        # Left
        mask = fisheye_coords[:, 0] <= fisheye_width / 2.0
        mask, coords = self.get_coords(rays, pinhole_width, pinhole_height, pinhole_intrisic_matrix, mask, "left")
        maptable[:, mask] = coords

        # Right
        mask = fisheye_coords[:, 0] > fisheye_width / 2.0
        mask, coords = self.get_coords(rays, pinhole_width, pinhole_height, pinhole_intrisic_matrix, mask, "right")
        maptable[:, mask] = coords

        # Top
        mask = fisheye_coords[:, 1] <= fisheye_height / 2.0
        mask, coords = self.get_coords(rays, pinhole_width, pinhole_height, pinhole_intrisic_matrix, mask, "top")
        maptable[:, mask] = coords

        # Bottom
        mask = fisheye_coords[:, 1] > fisheye_height / 2.0
        mask, coords = self.get_coords(rays, pinhole_width, pinhole_height, pinhole_intrisic_matrix, mask, "bottom")
        maptable[:, mask] = coords

        return maptable.T.reshape(fisheye_height, fisheye_width, 2)

    def get_coords(self, rays: np.ndarray, width: int, height: int, intrinsic: np.ndarray,
                   mask: np.ndarray, direction: str, margin: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:

        if direction == "front":
            R_mat = np.eye(3)
            box_offset = np.array([2 * width, 0.0])[:, None]
        elif direction == "left":
            R_mat = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
            box_offset = np.array([0, 0])[:, None]
        elif direction == "right":
            R_mat = R.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix()
            box_offset = np.array([4 * width, 0])[:, None]
        elif direction == "top":
            R_mat = R.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
            box_offset = np.array([width, 0])[:, None]
        elif direction == "bottom":
            R_mat = R.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()
            box_offset = np.array([3 * width, 0])[:, None]

        rays_local = rays[:, mask]
        proj = intrinsic @ R_mat
        img_coords = proj @ rays_local
        img_coords = img_coords[:2, :] / img_coords[2:3, :]

        valid = (
            (img_coords[0] >= -margin) & (img_coords[0] < width + margin) &
            (img_coords[1] >= -margin) & (img_coords[1] < height + margin)
        )

        img_coords = img_coords[:, valid]
        img_coords[0] = np.clip(img_coords[0], 0, width - 1)
        img_coords[1] = np.clip(img_coords[1], 0, height - 1)

        img_coords += box_offset
        final_mask = mask.copy()
        final_mask[mask] = valid

        return final_mask, img_coords

    def create_fisheye_image(self, left_img: np.ndarray, top_img: np.ndarray, front_img: np.ndarray,
                             bottom_img: np.ndarray, right_img: np.ndarray, save_path: str = "fisheye.png") -> None:
        """Create and save fisheye image from 5 input images."""
        five_pinhole_image = np.hstack([left_img, top_img, front_img, bottom_img, right_img]).astype(np.float32)

        remapped = cv2.remap(
            five_pinhole_image,
            self.maptable[..., 0],
            self.maptable[..., 1],
            interpolation=cv2.INTER_NEAREST
        )

        self.image = remapped.astype(np.uint8)
        self.frame += 1
        cv2.imwrite(save_path, self.image)

def load_pinhole_images(distinct_pinhole_imagedir_path:str):
    """Load five pinhole images from given paths."""
    left_img = cv2.imread(os.path.join(distinct_pinhole_imagedir_path, "2_rgb.png"))
    top_img = cv2.imread(os.path.join(distinct_pinhole_imagedir_path, "3_rgb.png"))
    front_img = cv2.imread(os.path.join(distinct_pinhole_imagedir_path, "0_rgb.png"))
    bottom_img = cv2.imread(os.path.join(distinct_pinhole_imagedir_path, "4_rgb.png"))
    right_img = cv2.imread(os.path.join(distinct_pinhole_imagedir_path, "1_rgb.png"))
    
    return left_img, top_img, front_img, bottom_img, right_img


def main():
    """Main function."""
    distinct_pinhole_imagedir_path = f"/seed4d/data/Town01/ClearNoon/vehicle.audi.tt/spawn_point_1/step_0/ego_vehicle/nuscenes/sensors"
    # Initialize fisheye processor
    camera = OfflineFisheyeCamera(camera_model=EquidistantProjection, width=640, height=640, fov=180)
    # Create fisheye image
    camera.create_fisheye_image(*load_pinhole_images(distinct_pinhole_imagedir_path), save_path='images/fisheye_image.png')
    print("Fisheye image created and saved as 'images/fisheye_image.png'.")

if __name__ == "__main__":
    main()