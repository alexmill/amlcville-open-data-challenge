# CHARLOTTESVILLE OPEN DATA CHALLENGE
## Best Predictive Model Competition
=======================================
Team Name: Love Thy K-Nearest Neighbor
Team members: Alex P. Miller (alexmill@wharton.upenn.edu)
=======================================


Final submission location:
=======================================
Predictions for test period are in folder: Submission/my_predictions/
Predictions for each variable are in their own file:
    - Submission/my_predictions/
        - clients.csv
        - sessions.csv
        - usage.csv



Code Requirements:
=======================================
My models are built in pure Python 3 and should work out of the box
on the Anaconda Continuum Python 3 distribution. However, the only 
non-standard package dependencies are: sklearn, pandas, numpy. Any
machine with Python 3 and those packages installed should be able to
run my code.



Code Structure:
=======================================
I built and trained my models using the code in NotebookWalkthrough.ipynb,
with the addition of several custom functions and classes that are 
contained in the file 'alexs_models/imports.py'. Afer the models
were trained, I saved them in pickle format in the alexs_models folder.



To run the predictive models:
=======================================
Navigate to this directory in a bash shell. Run the following command,
where `VARIABLE` is one of: clients, session, or usage (corresponding
to the three models required in the data challenge).

`$ python predict.py -v VARIABLE -x test`

This will print out the model predictions for the variable provided.
The `-x test` argument tells the model to make predictions for the
test period (Dec 21-27). Predictions for the training data can be made
by changing this argument to `-x 10,20`, where the 10 and 20 arguments
tell the model to make predictions for this range of rows in the 
training data (i.e., between rows 10 and 20).




To train the predictive models:
=======================================
The easiest way to see how the data are cleaned and the models are 
trained is to work through the NotebookWalkthrough.ipynb Jupyter 
notebook included in this submission directory. 

The notebook has clear heading titles about what each block is doing.
It approximately follows the following structure:

    - import statements
    - Data Cleaning
        - Formatting dependent variables
        - Formatting predictors
    - Brief mathematical description of models
    - Num Clients Model
        - Model definition
        - Data formatting
        - Cross-validation
        - Final model training
    - Num Sessions Model
        - Model definition
        - Data formatting
        - Cross-validation
        - Final model training
    - Num Sessions Model
        - Model definition
        - Data formatting
        - Final model training
    




Primary data sources:
=======================================
All data has been stored offline for purposes of this challenge. If
this model were to be put into production, it would require changes
to fetch data from official APIs. Because I accessed/saved all pages
manually, I did not violate any terms of service to obtain the data.


Weather data, obtained from Weather Underground:
    - Link to raw data:
        - https://www.wunderground.com/history/airport/KCHO/2017/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2017&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=
    - Link to Terms of Service:
        - https://www.wunderground.com/weather/api/d/terms.html
    - Offline file stored at:
        - Submission/my_data/weather_data.csv
        
UVA Basketball Game Data, obtained from ESPN
    - Link to Terms of Use:
        - https://disneytermsofuse.com/
    - Links to raw data:
        - 2016-2017 Season: http://www.espn.com/mens-college-basketball/team/schedule/_/id/258/year/2017
        - 2017-2018 Season: http://www.espn.com/mens-college-basketball/team/schedule/_/id/258/virginia-cavaliers
    - Offline file stored at:
        - Submission/my_data/basketball_data.csv

Downtown Charlottesville Local Event Data:
    - Sample link to January 2017 Calendar:
        - http://www.downtowncharlottesville.net/events/2017/01/
    - Offline file stored at:
        - Submission/my_data/events.csv






