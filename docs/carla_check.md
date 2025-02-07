# Carla sanity check

Carla is great but depending on ones system setup problems can occur. This sanity-check helps to identify problems and to check if Carla is working correctly.

## Step 1 - Clone the repository

Clone the seed4d repository to your local machine:

```
git clone https://github.com/continental/seed4d.git
```

## Step 2 - Build the SEED4D docker image and run container(s)

Enter the `seed4d` directory and build the container:

```
docker build -t seed4d .
```

Choose the GPU device(s) you want to use and run the container. Change `X` to the GPU device number(s) you want to use. We found that we needed to mount the `libnvidia-gpucomp.so` and `icd.d` directories to get Carla to work. Then you for example need to add: `-v /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.550.90.07:/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.550.90.07 \`, see for [further information](https://github.com/carla-simulator/carla/issues/8079#issuecomment-2312693140). The `sleep infinity` command is used to keep the container running. The last `-v` flag is used to mount the SEED4D directory to the container. This is where the data will be saved, change `SEED4D/` to the path of the repository. If you have the capacity, the fastest way is generating all the data in parallel on 8 different GPUs.

```
docker run --name carla \
--gpus '"device=X"' \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d \
-v SEED4D/:/seed4d \
seed4d \
sleep infinity
```

# Step 3 - Run the sanity check


Enter the docker container:
```
docker exec -it carla /bin/bash
```

Check if vulkan is working correctly. Important: `vulkaninfo --summary` should not yield any errors on the host machine, if it does your problem lies elsewhere. Here we verify that vulkan can properly be used from within the container:
```
vulkaninfo --summary
```

Check if Carla is working correctly. Make the carla binary executable and run it. CarlaUE4 should now be running. You can verify this by checking watch -n 0.01 nvidia-smi, the gpu should be taxed now.
```
chmod +x "/home/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"

/home/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -carla-server -RenderOffScreen
```

The terminal output should looke like this:  `4.26.2-0+++UE4+Release-4.26 522 0` and via  `nvidia-smi ` one should see that a part of the gpu is blocked. Great everything seems to work!
The container can be stopped.

