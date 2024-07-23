## Physics Parameter Identification for Robotic Elastoplastic Object Manipulation via Differentiable Physics
### Experiment scripts and data
#### **It is highly suggestted that you download the codes from Code Ocean and run them at your local Ubuntu machine.

### Instructions for reviewers
- Launch an Ubuntu 20.04 machine and download the code and data
  - `sudo apt-get update`
  - `sudo apt-get install build-essential`
  - `sudo apt-get install libx11-6`
- Launch a terminal
- Install the simulation package, which should be found in `~/capsule/code`
  - `cd deformable-object-manipulation-1.0`
  - create your conda env `conda env create -f environment.yml`
  - `conda activate DPS`
  - `pip install .`
- (Optional) If you would like to verify some of the plotting scripts
  - Make sure you launch an Ubuntu
  - `cd ~/capsule/code` 
  - `git clone git@github.com:IanYangChina/DRL_Implementation.git`
  - `cd DRL_Implementation`
  - `pip install .`
- `cd ~/capsule/data/SI4RP-data`
  - **Read the structure of the folder below**
  - `cd scripts`
  - `conda activate DPS`
  - Understand what a script does in detail: `python ANY_SCRIPT_FOLDER/ANY_SCRIPT.py -h`

#### Structure of the data folder
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