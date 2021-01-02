# Disaster Response Pipeline Project

### Description:
This project builds on a data set containing real messages that were sent during disaster events. It contains code for building a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project Components
This project has three main components.

1. ETL Pipeline
The ETL script does the following:
- Load messages and categories datasets
- Merge and clean the data
- Save the cleaned data in a SQLite database

2. Classifier training
- Train a classifier on a test set
- Hyperparameter optimization using GridSearchCV
- Save trained model as pickle file

3. Web App
-Create Flask web app
- Show plotly visuals with information on test set
- Provide opportunity for a user to classify new messages
