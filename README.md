# RRC 2022 solution --- team excludedrice
This repository includes the source python scripts of our solutions to RRC 2022.


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
Only the mixed datasets need to be filtered, so this step won't work for the expert dataset. If you are trying the filter the expert datasets, some error will be raised. Assuming you are filtering the lift mixed dataset, activate your vertiral environment and run the following command. This process would take 1-2 hours. After this process, you will see a new "save" folder which include all history files during training and filtereing. Now you can delete everying except "turn_final_positive.npy" if you want to save SSD space.
    
    python dataset_filter.py --task="real_lift_mix"
    
### Augment the dataset by geometry
Once the dataset is filtered
