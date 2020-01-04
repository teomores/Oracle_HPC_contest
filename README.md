# Oracle Entity Resolution Contest at Polimi
This is our team repository to solve the Entity Resolution Contest hosted by Polimi and Oracle Labs; you can find in depth details [here](https://www.kaggle.com/c/oracle-polimi-contest-2019/overview).

## Getting Started
In this paragraph, we'll see all the steps needed to get the project up and running on your local machine.
### Prerequisites
1. [Python 3.7.4](https://www.python.org/downloads/release/python-374/)
2. Download the data from the [Kaggle Competition page](https://www.kaggle.com/c/oracle-polimi-contest-2019/data)

## How to run
First of all, clone the repo or download and unzip it. Then install the requirements:
```console
foo@bar:~$ <PATH-TO-THE-REPO>pip install requirements.txt 
```
After that, you need to create these folders:
.
|
|_ dataset
|        |_ original
|                  |_ train.csv
|                  |_ test.csv
|                  |_ evaluation_script.py

After that you can choose to run the single modules or, in alternative, we set up a demo that runs all the files necessary to reproduce our best results in the `run_all_model.py` script, so you can simply run:
```console
foo@bar:~$ <PATH-TO-THE-REPO>python run_all_model.py
```
And you will find our 0.55027 score submission. 
