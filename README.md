# Finetune Action Recognition Models with GluonCv Torch
This repo contains the modified gluoncv torch codes for finetuning the torch models in gluoncv. 

- The code is adopted from the https://github.com/dmlc/gluon-cv/blob/master/scripts/action-recognition/train_ddp_pytorch.py

- The main aim of this repo is to provide a single-gpu based tutorial  to this tutorial page: https://cv.gluon.ai/build/examples_torch_action_recognition/finetune_custom.html 

- Configure your own paths in the config file

- Gluon-CV provides a large number of SoTA models where the yaml finds can be found here: https://github.com/dmlc/gluon-cv/tree/master/scripts/action-recognition/configuration
- According to the model you want to use modify the yaml file. A modified example can be found under the config folder
- Dont change the frame lengths randomly, especially if you are using slow-fast networks
- I advise you to use Gluon-CV for fast prototyping https://github.com/dmlc/gluon-cv https://cv.gluon.ai/index.html
- More Gluon-CV torch tutorials : https://github.com/dmlc/gluon-cv/tree/master/scripts/action-recognition
