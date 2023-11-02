# Machine Learning Coursework 1 - Decision Trees

### Installation and Usage

Once you have cloned the repo first make sure the correct virtual environment is setup on the DOC lab machines. You can do this by entering into the terminal:

- `source /vol/lab/intro2ml/venv/bin/activate` 

You can then train and test a decision tree using 10-fold cross validation using the following commands:

- `cd CW1`
- `python3 src/main.py {PATH_TO_DATAFILE}`

Where {PATH_TO_DATAFILE} is replaced by the actual path of the data you want to train a decision tree on.

For example, if we want to train, test and evaluate on the clean dataset we would run the following (assuming we are currrently in the mlcw1 directory and the venv is setup correctly):

- `cd CW1`
- `python3 src/main.py wifi_db/clean_dataset.txt`

Running this code will run a 10-fold cross validation on the dataset and will print out multiple different evaluation metrics onto the screen.



