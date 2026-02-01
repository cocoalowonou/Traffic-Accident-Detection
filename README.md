# DoTA_TempoLearn
This is the source code for Spatio-Temporal Learning for Traffic Accident Detection trained on [Detection of Traffic Anomaly (DoTA)](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly) Dataset. The model supports 2 staged-pipeline: anomaly localization and classification of anomaly type. 

# Model Interface
Our model includes following stages
1. Video feature extraction layer ( Pretrained ResNet-101 model is used and fine-tune the last layer of the model)
2. Temporal feature learning layer
3. Temporal proposal generation layer
4. Segment classification layer. 

#### Input to the model
To start the training, the model simply accepts only the path to raw video frames. No pre_processing stages are necessary. Configurations such as kernel scales, IOU thresholds and learning rate can be configured in ```opt.py``` file before training is started.

#### Output
The output from the model is the proposals in the form of ```(actioness_score, prop_center, prop_len, anomaly_class)``` pairs. 

# Requirements
- [PyTorch]
- [Tensorflow]
- [SKlearn]

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   
In a nutshell here's how to train this model, so **for example** assume you want to train the model, so you should do the following:
- In `opt.py`  file, configure the path to raw video frames, path to meta_data json file and folder name to save the model checkpoints. Then, one can simply start run the python script.

#### 1. Training and Validation
For training and validation, first, set the dataset paths in ```opt.py``` file to configure 
```
python train.py
```

We also provide DistributedDataParallel training if there're more than one gpu. For example, training with 2 gpus: 
```
python train.py -g 2
```

#### 2. Testing

For testing the model, one can simply run the python script of test file. The testing results of generated proposals and classification of segments will be saved under the same directory `./results`. The test.py script proceed the evaluation together at the end of testing.
```
python test.py
```


#### 3. Inference

For inference for one video, one can simply run the python script of inference file. The testing results of generated proposals and classification of segments will be saved under the same directory `./results`.
```
python inference.py -v trimmed.mp4
```


# In Details
```
├──  opt.py   - Here's the default configuration of hyper-parameters for training and testing the model.
│  
│
│
├──  datasets  - The dataset file that handles to generating ground-truth anchors and segment labels .
│
│
│
├── train.py    - The train.py file includes the complete training script of the model.
│   
│
│ 					
├── test.py     -  This is the implementation of testing the model and generating the testing results.
│   
│
│
├──  models    - This folder contains the implementation of TempoLearn model.
│    └── model_main.py          - This is the complete TempoLearn model implementation from proposal generation to classification.  
│    └── model_tempolearn.py    - This is the implementation of temporal learning layer of the model.
│    └── model_transformer.py   - This is the implementation of transformer model for classification task of the model.
│
│
├──  eval      - This folder contains python scripts for evaluating the model.
│    └── eval_proposal.py       - Evaluation script for proposal generation in AUC and AR_AN score
│    └── eval_detection.py      - Evaluation script for recognition in mAP score.
│
│
└── utils.py    - Implementation of utilities functions.
```



