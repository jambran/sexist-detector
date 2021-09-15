# sexist-detector
Project bulit on the Kaggle dataset to predict whether a 
workplace comment is misogynistic.

## The Dataset
Users may download the dataset from Kaggle 
[here](https://www.kaggle.com/dgrosz/sexist-workplace-statements).

## Installation
1. Create and source a new virtual environment.
   ``` 
   python3 -m venv venv_sexist_detector
   source venv_sexist_detector/bin/activate
   ```
   
2. Install project dependencies
   ```
   pip install -r requirements.txt
   ```
3. Run inference by calling `python run_inference.py`

## Training the Model
To retrain the model after making architecture or hyperparameter 
changes, users may run `python train.py`. 

