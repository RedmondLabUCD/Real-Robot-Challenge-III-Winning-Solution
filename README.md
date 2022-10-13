# RRC 2022 solution --- team excludedrice
This repository includes the source python scripts of our solutions to RRC 2022.

## Contents
- rrc2022: This folder contains the scripts for deploying the trained models on the real robot cluster.
- trained_models: This folder contains our final submitted models to the RRC2022.
- dataset_aug.py: The dataset augmentation script.
- dataset_filter.py: The main scripts for filtering the mixed dataset.
- main_bc_train_tune.py: The main scripts for training the BC models.
- models.py: The neural networks.
- trainer.py: The implementations of Behaviour Cloning(BC)
- utils.py: Tools.
- requirements.txt: The required python libraries.


## Installation
**1.** Our implementations are tested only on [Anadonda](https://www.anaconda.com/products/distribution). We recommend you install Anaconda if you would reproduce our work, although other development environments should also work.

**2.** Activate your vitural environment(or others) and run:

    pip install -r requirements.txt
        
where we installed the official RRC 2022 library through the local installation method; please see the [official document](https://webdav.tuebingen.mpg.de/real-robot-challenge/2022/docs/simulation_phase/index.html#get-the-software) for more details. 


## Reproducing our experiments
In our experiment, we mainly has three part. 

1) Filter the mixed dataset. **Note: Start from step 2 if you are training from the expert dataset**
2) Augment the dataset by geometry. 
3) Traing the Behaviour Cloning(BC) model. 

Because the RRC dataset is large, our approach for processing the mixed datasets requires a machine with 16GB RAM, whereas for the expert datasets, we require 32GB. To save the RAM, we keep saving the file on SSD from RAM and load it in the next iteration; hence you should have at least 50GB of free space left on your SSD.

### Filter the mixed dataset
Only the mixed datasets need to be filtered, and this step will not work for the expert dataset. Some errors will be raised if you are trying to filter the expert datasets. Assuming you filter the lift mixed dataset, activate your virtual environment and run the following command.
    
    python dataset_filter.py --task="real_lift_mix"
    
**!!!Note!!!: The abbreviations of four tasks are "real_lift_mix", "real_lift_exp", "real_push_mix", "real_push_mix".**

This process would take 1-2 hours. After this process, you will see a new "save" folder that includes all training and filtering history files. Now you can delete everything except!!! **turn_final_positive.npy** if you want to save SSD space.
    
### Augment the dataset by geometry
Once the dataset is filtered, or if you are training from the expert dataset. Run the following command(assuming you are processing the lift mixed dataset):
    
    python dataset_aug.py --task="real_lift_mix"
    
This pricess would take 40-60 minutes. After the process, you will see a new **xxx_aug.npy** file.

### Train BC models
Note: The normalization only works for the lift task and does not work for the push. Assuming you are training the lift mixed task, run:
    
    python main_bc_train_tune.py --exp-name="real_push_mix_test1" --task="real_lift_mix" --norm=1
    
Once this process finished, you will see trained models in **"./save/real_lift_mix/models/xxx_tune"**, make you enter the correct name.

## Deploy the model on the real robot
The deployment scripts is in under "rrc2022" folder. Steps:
1) Upload the trained BC model and the normalization parameters on  RRC robot cluster(more details see the [website](https://webdav.tuebingen.mpg.de/real-robot-challenge/2022/docs/robot_phase/submission_system.html)). You can find the model in **"./save/real_lift_mix/models/xxx_tune/ckpt_50.pth"** and you can find the normalization parameters in **"./save/real_lift_mix/datasets/train_aug_norm_params.npy"**.
2) Change the file's name in the deployment script to make them direct to the models you submitted; you can do this by easily editing the files located in rrc2022 on GitHub.
3) Submit jobs to the system.
