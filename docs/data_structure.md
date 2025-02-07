# Data Structure

The data is seperated into static and dynamic data. Static data consist of only a single timepoint but high-quality images. The dynamic data part consists of multiple timepoints but with a lower resolution. 

Intrinsics and extrinsics information are 

Example folder tree of the dynamic data:

```
dynamic 
│
└───Town01/Weather/Vehicle
│   │
│   └───spawnpoint4
│       │ 
│       └───step_0
│       │    │ 
│       │    └───371
│       │    │  │ 
│       │    │  └───nuscenes
│       │    │  │   │ - sensor_info.json
│       │    │  │   └───sensors
│       │    │  │   │   │ - depth.png
│       │    │  │   │   │ - instance_segmentation.png
│       │    │  │   │   │ - mask.png
│       │    │  │   │   │ - obj.png
│       │    │  │   │   │ - optical_flow.png
│       │    │  │   │   │ - rgb.png
│       │    │  │   └───transforms
│       │    │  │   │   │ - transforms_background.json
│       │    │  │   │   │ - transforms_ego.json
│       │    │  │   │   │ - transforms_normalized.json
│       │    │  │   │   │ - transforms_obj.json
│       │    │  │   │   │ - transforms.json
│       │    │  └───nuscenes_lidar
│       │    │  │   │ - sensor_info.json
│       │    │  │   └───sensors
│       │    │  │   │   │ - lidar.ply
│       │    │  │   └───transforms
│       │    │  │   │   │ - lidar_transforms.json
│       │    │  └───sphere
│       │    │      │ - sensor_info.json
│       │    │      └───sensors
│       │    │      │   │ - depth.png
│       │    │      │   │ - instance_segmentation.png
│       │    │      │   │ - mask.png
│       │    │      │   │ - obj.png
│       │    │      │   │ - optical_flow.png
│       │    │      │   │ - rgb.png
│       │    │      └───transforms
│       │    │          │ - transforms_background.json
│       │    │          │ - transforms_ego.json
│       │    │          │ - transforms_obj.json
│       │    │          │ - transforms.json
│       │    │ 
│       │    └───371
│       │    └───372
│       │    ...
│       │ 
│       └───step_1
│       └───step_2
│       ...
│   
└───Town02/Weather/Vehicle
└───Town03/Weather/Vehicle
...
```
