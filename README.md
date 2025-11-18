# Machine Learning

--- 

learning about how Ml Work .

## Setup 
```shell
conda create -n ml_env
conda activate ml_env
conda deactivate

conda install --file requirements.txt
```
---

## what do you mean by this `train_test_split(x, y, test_size=0.2)`

This function splits your dataset into two parts:

- Training set (80%) → used to train the model

- Testing set (20%) → used to evaluate the model on unseen data <br>

Meaning of each part

- x → Input features (data you give to the model)

- y → Labels/targets (answers the model needs to learn)

- test_size=0.2 → Use 20% of the data for testing, 80% for training

---