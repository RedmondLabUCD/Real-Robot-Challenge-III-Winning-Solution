RRC 2022 solution --- team excludedrice
========================
This repository includes the source python scripts of our solutions to RRC 2022.


Installation
----------------
**1.** Our implementations are tested only on [Anadonda](https://www.anaconda.com/products/distribution). We recommend you install Anaconda if you would reproduce our work, although other development environments should also work.

**2.** Activate your vitural environment(or others) and run:

    singularity run /path/to/user_image.sif mpirun -np 8 python3 train.py --exp-dir='reproduce' --n-epochs=300 2>&1 | tee reproduce.log
        
We installed the official RRC 2022 library through local installation method, for more details please see official document. 


If you would like to deploy this repo by Apptainer/Singularity image, please download from:
For evaluation purpose of originizer, you can find the latest image from cloud.
