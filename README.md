<h1 align="center">
Differentiable Physics-based System Identification for 

Robotic Manipulation of Elastoplastic Materials
</h1>
<h2 align="center">
Code: <a href="https://github.com/IanYangChina/SI4RP-data"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="20" height="20"></a>
Video: <a href="https://www.youtube.com/watch?v=2-9JWRsQhTU"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/2560px-YouTube_full-color_icon_%282017%29.svg.png" width="25" height="20"></a>
Paper: <a href="https://arxiv.org/abs/2411.00554"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/PDF_icon.svg/1200px-PDF_icon.svg.png" width="25" height="20"></a>
</h2>

### [Project page](https://ianyangchina.github.io/SI4RP-data/)

### Instructions
- On an Ubuntu (>18.04) terminal and run:
  - `sudo apt-get update`
  - `sudo apt-get install build-essential`
  - `sudo apt-get install libx11-6`
- Download the code and data. They should be in three folders: 
  - Clone this repository `git clone https://github.com/IanYangChina/SI4RP-data.git`
  - Install the [simulator](https://github.com/IanYangChina/SI4RP-data/tree/main/simulator) `cd SI4RP-data/simulator`
- You are ready to run the scripts.
  - `cd SI4RP-data`
  - **Read the structure of the folder below**
  - `conda activate DPS`
  - Understand what a script does in detail: `python scripts/ANY_SCRIPT_FOLDER/ANY_SCRIPT.py -h`

### Structure of the repository
- `data/SI4RP-data/data` contains the data for the SI4RP dataset
  - `data-motion-*` contains the point cloud data collected by different motions
  - `trajectories` contains the trajectories of the motions from the MOVEIT! planner that was used on the real robot, as well as the constructed trajectories used in the simulation
- `data/SI4RP-data/figures` contains the visualisation figures from the experiment results
- `data/SI4RP-data/gradient-analysis` contains the gradient analysis results
- `data/SI4RP-data/loss-lanscape-analysis` contains the loss landscape analysis results
- `data/SI4RP-data/optimisation-results` contains the system identification results
- `data/SI4RP-data/scripts` contains the various scripts
  - `data_generation` contains the scripts for generating heightmaps and meshes from point clouds
  - `experiments` contains the scripts for computing the loss landscapes, gradients and running the system identification experiments
  - `plotting` contains the scripts for plotting the results
  - `testing` contains the scripts for various testing purposes
