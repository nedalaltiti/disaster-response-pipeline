# Disaster Response Pipeline Project
### Description:
developing a model to classify messages sent during disasters. There are 36 predefined categories such as 'Child alone', 'Aid related', or 'Medical help'. This classification is crucial for directing these messages to the appropriate disaster relief agencies.

The initial dataset contains pre-labelled tweet and messages from real-life disasters. The aim of this project is to build a Natural Language Processing tool that categorize messages.

The Project is divided into the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure.
2. Machine Learning Pipeline to train a model which is able to classify text messages in 36 categories.
3. Web Application using Flask to show model results and predictions in real time.

### Data:

The data in this project comes from Figure Eight - Multilingual Disaster Response Messages. This dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters.

The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.

Data includes 2 csv files:

1. disaster_messages.csv: Messages data.
2. disaster_categories.csv: Disaster categories of messages.

The end result will be a web application that takes a message as input and provides the corresponding classification as output:

![image](https://github.com/nedalaltiti/disaster-response-pipeline/assets/106015333/9b1ddc86-ebba-49c7-b7c9-31740e4e6b8e)

### Folder Structure:

- app
    - templates
        - master.html 
        - go.html
    - run.py 

- data
    - disaster_categories.csv 
    - ETL Pipeline Preparation.ipynb
    - disaster_messages.csv 
    - process_data.py
    - DisasterResponse.db
   
- models
    - ML Pipeline Preparation.ipynb
    - train_classifier.py
    - classifier.pkl 

- README.md

### Installation:

This project requires Python 3.x and the following Python libraries:

* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly
  
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
