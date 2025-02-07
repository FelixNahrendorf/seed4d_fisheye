# Generate SEED4D datasets

## Step 1 - Clone the repository

Navigate to the folder where you want your data to be generated and clone the seed4d repository there:

```
git clone https://github.com/continental/seed4d.git
```

## Step 2 - Build the SEED4D docker image and run container(s)

Enter the `seed4d` directory and build the container:

```
docker build -t seed4d .
```

Choose the GPU device(s) you want to use and run the container. Change `X` to the GPU device number(s) you want to use. We found that we needed to mount the `libnvidia-gpucomp.so` and `icd.d` directories to get Carla to work. Then you for example need to add: `-v /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.550.90.07:/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.550.90.07 \`, see for [further information](https://github.com/carla-simulator/carla/issues/8079#issuecomment-2312693140). The `sleep infinity` command is used to keep the container running. The last -v flag is used to mount the SEED4D directory to the container. This is where the data will be saved, change `SEED4D/` to the path of the repository.

```
docker run --name seed4d_gpuX \
--gpus '"device=X"' \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d \
-v SEED4D/:/seed4d \
seed4d \
sleep infinity
```

## Step 3 - Generate all config files

Enter the container and navigate to the seed4d directory containing the repository:

```
docker exec -it seed4d_gpuX /bin/bash

cd /seed4d/
```

Run the following command to generate the required config files:

```
python3.8 config/cloner.py --yaml_file config/static.yaml --output_dir /seed4d/config/static --type static

python3.8 config/cloner.py --yaml_file config/static.yaml --output_dir /seed4d/config/dynamic --type dynamic
```
This function simpliy clones the configuration files for the different towns. The `--type` flag can be set to `static`, `dynamic` or `custom` depending on the type of data you want to generate. The `--yaml_file` flag should point to the yaml file containing the configuration that should be cloned. The `--output_dir` flag should point to the directory where the config files will be saved. The number of .yaml files required for the static data is 2002 and 498 for the dynamic data.

If you are unable to generate files within the created docker container try running the following command:

```sh
chmod -R 777 /seed4d
```

## Step 4 - Generate the data

Now we are ready to generate the data! We recommend generating the data in a [tmux session](https://linuxize.com/post/getting-started-with-tmux/), this will be done by default when running the following commands to generate the data. Change `Y` to the town you want to generate for example `02`.

```
tmux 

python3.8 main.py --config_dir /seed4d/config/static/TownX  --carla_executable /home/carla/CarlaUE4.sh --data_dir /seed4d/data/static --normalize_coords False --combine_transforms False --map False

python3.8 main.py --config_dir /seed4d/config/dynamic/TownX  --carla_executable /home/carla/CarlaUE4.sh --data_dir  /seed4d/data/dynamic
```

## Step 5 - Check the generated files

Once the data is generated one can run (either with the `static` or `dynamic` category): 

```
python3.8 utils/check_dataset.py --data_dir /seed4d/data/static --category static
```

When running this for the dynamic dataset potentially detach the session (e.g., via using tmux) since it can take a lot of time.

## Step 6 - Generate missing files

If you want to generate the missing files, run the following (example) command (similar to above) for the missing or incomplete .yaml files:

```
python3.8 main.py --config /seed4d/config/static/Town07/static_Town07_Spawnpoint101.yaml  --carla_executable /home/carla/CarlaUE4.sh --data_dir /seed4d/data/static --normalize_coords False --combine_transforms False --map False
```

In the example spawnpoint 1 from Town07 is missing. Here a single yaml config file is provided together with the `--output_dir` flag pointing to the parent directory where the missing files should be saved. The `--config` flag should point to the yaml file containing the configuration that should be created. The utils folder contains a file with which the data can be split into train and test.