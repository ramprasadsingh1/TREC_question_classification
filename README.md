# TREC Question Classification

## Running the code
In order to run the code, download the folder, which contains all the required files while ensuring the folder structure and file locations are not changed.  

The **data** folder contains the following:  
* **config.yml** (config file)
* **glove.small.txt** (word embedding)
* **test.label** (test dataset)
* **train.label** (train dataset)

> ![image](https://user-images.githubusercontent.com/29594609/222275748-2f1fdb41-28b7-40cb-94f9-f6a40a41e239.png)

The **src** folder contains the following:
*  **labels** (predicted labels are saved in this folder after testing)
*  **models** (models are saved in this folder after training)
*  **question_classifier.py** (source code)

> ![image](https://user-images.githubusercontent.com/29594609/222276857-cccf2df5-80cd-4c5f-ba61-71c1a61d0072.png)
>>![image](https://user-images.githubusercontent.com/29594609/222277671-00aceba7-ff96-4ed4-be85-be3ad13a5ed0.png)
>>![image](https://user-images.githubusercontent.com/29594609/222277460-4098a6e0-5ed5-4292-ae0b-fb929ad8a336.png)


### Training and Testing
To **train** a model use the following command:  
> & python3 question_classifier.py --train --config [configuration_file_path]

*Example:*  
![image](https://user-images.githubusercontent.com/29594609/222270542-52b4a1de-5d9f-4a12-ad5a-f5876a566177.png)

To **test** a model by issuing the following command:  
> & python3 question_classifier.py --test --config [configuration_file_path]

*Example:*  
![image](https://user-images.githubusercontent.com/29594609/222270610-cd456a72-bb73-443d-a31d-973774886319.png)

#### In an event the above commands dont work, alternatively try this:

**Train:**  
> & python3 question_classifier.py --trainmodel --config [configuration_file_path]

**Test:**  
> & python3 question_classifier.py --testmodel --config [configuration_file_path]

## Changing the settings
The settings for model training and testing can be set by editing **"config.yml"**. This file contains the following fields that will only accept the below mentioned values:  

* path_train: **data/train.label** (relative path for train dataset)  
* path_test: **data/test.label** (relative path for test dataset)  
* path_embedding: **data/glove.small.txt** (relative path for word embedding)  
* classification: **coarse** or **fine** (choose between Coarse or Fine classification)   
* model: **bow** or **bilstm** (choose between BoW or BiLSTM model)  
* embedding: **random** or **glove** (choose between Random or Glove embedding)  
* setting: **freeze** or **finetune** (choose between Freeze and Fine Tuning the settings)  

*Example:*  
![image](https://user-images.githubusercontent.com/29594609/222266197-dba1cb24-240a-4c09-a924-5b289251c0e8.png)

**The possible settings are as follows: (6 settings each for Coarse and Fine classification)**  

Coarse Classification/ Fine Classification:  
* BoW + Random + Fine Tune  
* BoW + Glove + Freeze  
* Bow + Glove + Fine Tune  
* BiLSTM + Random + Fine Tune  
* BiLSTM + Glove + Freeze  
* BiLSTM + Glove + Fine Tune  
