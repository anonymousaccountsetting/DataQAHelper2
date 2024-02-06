from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent

load_dotenv(find_dotenv())
chat = ChatOpenAI(model_name="gpt-4", temperature=0)

### Below are regression model experiments
# df=pd.read_csv("./regression_data/Real estate.csv")
# agent=create_pandas_dataframe_agent(chat,df,verbose=True)
# agent.run("This is a dataset about how the transaction date, the house age, the distance to the nearest MRT station, and the number of convenience stores affect the house price of one unit area."
#           "Please use the transaction date, the house age, the distance to the nearest MRT station, and the number of convenience stores as independent variables and the house price of one unit area as the dependent variable and use a linear regression model to answer the following questions"
#           "1.Is the relationship between the dependent variable and the independent variable strong in this model?"
#           "2.What impact will different factors have on real estate prices?"
#           "3.Which factor has the greatest impact on real estate prices?")


# df = pd.read_csv("./regression_data/insurance.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how age, BMI, and the number of children affect the insurance charge"
#   "Please use the age, BMI, and the number of children as independent variables and the insurance charge as the dependent variable and use a linear regression model to answer the following questions"
#   "What impact will different factors have on insurance?"
#   "Does having children have a significant impact on insurance prices?"
#   "How will age affect insurance prices?"
#   "Which factor has the greatest impact on insurance prices?"))

# df = pd.read_csv("./regression_data/Life Expectancy Data.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the Year, the Alcohol, the percentage expenditure, the BMI, the HIVAIDS, the Population, and the GDP affect the Life expectancy."
#   "Please use the Year, the Alcohol, the percentage expenditure, the BMI, the HIVAIDS, the Population, and the GDP as independent variables and the Life expectancy as the dependent variable and use a linear regression model to answer the following questions"
#   "What factors have a significant impact on life expectancy?"
#   "What factors are positively correlated with life expectancy?"
#   "What is the most important factor for life expectancy?"))

# df = pd.read_csv("./regression_data/car data.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the present price, the Kms driven, and the year of car production affect the selling price."
#   "Please use the present price, the Kms driven, and the year of car production as independent variables and the selling price as the dependent variable and use a linear regression model to answer the following questions"
#   "Is the relationship between the dependent variable and the independent variable strong in this model?"
#   "What impact will different factors have on the selling price?"
#   "Which factor has the greatest impact on the selling price?"))

# df = pd.read_csv("./regression_data/cancer_reg.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the mean per capita (100,000) cancer diagnoses, the median income per county, the population of a county, the percent of the populace in poverty, and the median age of a county residents affect the mean per capita (100,000) cancer mortalities"
#   "Please use the mean per capita (100,000) cancer diagnoses, the median income per county, the population of a county, the percent of the populace in poverty, and the median age of a county residents as independent variables and the mean per capita (100,000) cancer mortalities as the dependent variable and use a linear regression model to answer the following questions"
#   "What impact will different factors have on the death rate?"
#   "Does average income have a significant impact on the mortality rate?"
#   "What is the biggest factor affecting the mortality rate?"))

# df = pd.read_csv("./regression_data/Airline Passenger Satisfaction/train.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the Flight Distance, the Inflight wifi service, Age, Departure Delay in Minutes, and Arrival Delay in Minutes affect the Inflight service."
#   "Please use the Flight Distance, the Inflight wifi service, Age, Departure Delay in Minutes, and Arrival Delay in Minutes as independent variables and the Inflight service as the dependent variable and use a linear regression model to answer the following questions"
#   "What impact will different factors have on the satisfaction of Inflight services?"
#   "How will Departure Delay affect satisfaction with Inflight services?"
#   "How will Arrival Delay affect satisfaction with Inflight services?"
#   "Which factor has the greatest impact on the satisfaction of inflight services?"))


# df = pd.read_csv("./regression_data/auto-mpg.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the cylinders, the displacement, the horsepower, the weight, the acceleration, and the model year affect the mpg"
#   "Please use the cylinders, the displacement, the horsepower, the weight, the acceleration, and the model year as independent variables and the mpg as the dependent variable and use a linear regression model to answer the following questions"
#   "What impact will different factors have on miles per gallon?"
#   "Does the year have a significant impact on the miles per gallon?"
#   "Which factor has the greatest impact on the miles per gallon?"))

# df = pd.read_csv("./regression_data/bike+sharing+dataset/day.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the holiday, the weekday, the temperature, the humidity, the windspeed, and the season affect the count of bicycles rented out"
#   "Please use the holiday, the weekday, the temperature, the humidity, the windspeed, and the season as independent variables and the count of bicycles rented out as the dependent variable and use a linear regression model to answer the following questions"
#   "What impact will different factors have on the number of bicycles rented out?"
#   "What is the impact of weekdays on the number of bicycles rented?"
#   "Which factor has the greatest impact on the number of bicycles rented?"))

# df = pd.read_csv("./regression_data/concrete+compressive+strength/Concrete_Data.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how Cement, Blast Furnace Slag, the Fly As, Water, Superplasticize, Coarse Aggregate, Fine Aggregate, and Age (day) affect the Concrete compressive strength"
#   "Please use the Cement, Blast Furnace Slag, the Fly As, Water, Superplasticize, Coarse Aggregate, Fine Aggregate, and Age (day) as independent variables and the Concrete compressive strength as the dependent variable and use a linear regression model to answer the following questions"
#   "What are the effects of different components on the compressive strength of concrete?"
#   "Is the effect of age on the compressive strength of concrete significant?"
#   "Which factor has the greatest impact on the compressive strength of concrete?"))

# df = pd.read_csv("./regression_data/Credit Risk Dataset/credit_risk_dataset.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the person’s income, employment length, percent income, and credit history length affect the loan amount"
#   "Please use the person’s income, employment length, percent income, and credit history length as independent variables and the loan amount as the dependent variable and use a linear regression model to answer the following questions"
#   "What impact will various factors have on the loan amount?"
#   "Is the impact of employment duration on loan amount significant?"
#   "Which factor has the greatest impact on the loan amount?"))

# df = pd.read_csv("./regression_data/energy+efficiency/ENB2012_data.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the X1, the X2, the X3, the X4, the X5, the X6, the X7, and the X8 affect the Y1, and below are the data analysis results by fitting the dataset to the Linear Model. (X1 Relative Compactness X2 Surface Area X3 Wall Area X4 Roof Area X5 Overall Height X6 Orientation X7 Glazing Area X8 Glazing Area Distribution Y1 Energy Heating Load) "
#   "Please use the X1, the X2, the X3, the X4, the X5, the X6, the X7, and the X8 as independent variables and the Y1 as the dependent variable and use a linear regression model to answer the following questions"
#   "Is the relationship between X and y strong?"
#   "What impact will different factors have on y?"
#   "Which factor has the greatest impact on y?"))

# df = pd.read_csv("./regression_data/energy+efficiency/ENB2012_data.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the X1, the X2, the X3, the X4, the X5, the X6, the X7, and the X8 affect the Y2, and below are the data analysis results by fitting the dataset to the Linear Model. (X1 Relative Compactness X2 Surface Area X3 Wall Area X4 Roof Area X5 Overall Height X6 Orientation X7 Glazing Area X8 Glazing Area Distribution Y2 Energy Cooling Load) "
#   "Please use the X1, the X2, the X3, the X4, the X5, the X6, the X7, and the X8 as independent variables and the Y2 as the dependent variable and use a linear regression model to answer the following questions"
#   "Is the relationship between X and Y2 strong?"
#   "What impact will different factors have on Y2?"
#   "Which factor has the greatest impact on Y2?"))

# df = pd.read_csv("./regression_data/forest+fires/forestfires.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the X, the Y, the FFMC, the DMC, the DC, the ISI, the temp, the RH, the wind, and the rain affect the area. (X - x-axis spatial coordinate within the Montesinho park map: 1 to 9 Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9 FFMC - FFMC index from the FWI system: 18.7 to 96.20 DMC - DMC index from the FWI system: 1.1 to 291.3  DC - DC index from the FWI system: 7.9 to 860.6  ISI - ISI index from the FWI system: 0.0 to 56.10 temp - temperature in Celsius degrees: 2.2 to 33.30 RH - relative humidity in %: 15.0 to 100 wind - wind speed in km/h: 0.40 to 9.40  rain - outside rain in mm/m2 : 0.0 to 6.4  area - the burned area of the forest (in ha): 0.00 to 1090.84)"
#   "Please use the X, the Y, the FFMC, the DMC, the DC, the ISI, the temp, the RH, the wind, and the rain as independent variables and the area as the dependent variable and use a linear regression model to answer the following questions"
#   "Is the relationship between the independent variable and the dependent variable strong enough?"
#   "What is the impact of different independent variables on the area of fire damage?"
#   "Which independent variable has the greatest impact on the area of fire damage?"))

# df = pd.read_csv("./regression_data/online+news+popularity/OnlineNewsPopularity.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the Number of words in the title (n_tokens_title), the number of words in the content (n_tokens_content), the Rate of unique words in the content (n_unique_tokens), the Rate of non-stop words in the content (n_non_stop_unique_tokens), the number of links (num_hrefs), the number of links to other articles (num_self_hrefs), the number of images (num_imgs), the number of videos (num_videos), the Average length of the words in the content (average_token_length), and the number of keywords in the metadata (num_keywords) affect the shares"
#   "Please use the Number of words in the title (n_tokens_title), the number of words in the content (n_tokens_content), the Rate of unique words in the content (n_unique_tokens), the Rate of non-stop words in the content (n_non_stop_unique_tokens), the number of links (num_hrefs), the number of links to other articles (num_self_hrefs), the number of images (num_imgs), the number of videos (num_videos), the Average length of the words in the content (average_token_length), and the number of keywords in the metadata (num_keywords) as independent variables and the shares as the dependent variable and use a linear regression model to answer the following questions"
#   "Is the relationship between the dependent variable and the independent variable strong?"
#   "How do different factors affect the number of shares?"
#   "Which factor has the greatest impact on the number of shares?"))

# df = pd.read_csv("./regression_data/CarPrice_Assignment.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the symboling, the wheelbase, the car length, the car width, the car height, the curb weight, the engine size, the bore ratio, the stroke, the compression ratio, the horsepower, the peak rpm, the city mpg, and the highway mpg affect the car price"
#   "Please use the the symboling, the wheelbase, the car length, the car width, the car height, the curb weight, the engine size, the bore ratio, the stroke, the compression ratio, the horsepower, the peak rpm, the city mpg, and the highway mpg as independent variables and the car price as the dependent variable and use a linear regression model to answer the following questions"
#   "What are the effects of different factors on car prices?"
#   "What factors have a significant impact on the price of cars?"
#   "Which factor has the greatest impact on car prices?"))

# df = pd.read_csv("./regression_data/CAhousing/housing.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the longitude, latitude, median age of a house within a block, the total rooms, the total bedrooms, the total number of people residing within a block (population), the total number of households, and the median income for households within a block of houses affect the house value."
#   "Please use the longitude, latitude, median age of a house within a block, the total rooms, the total bedrooms, the total number of people residing within a block (population), the total number of households, and the median income for households within a block of houses as independent variables and the house value as the dependent variable and use a linear regression model to answer the following questions"
#   "What impact will different factors have on California housing prices?"
#   "Does the number of bedrooms have a significant impact on housing prices?"
#   "Which factor has the greatest impact on housing prices?"))

# df = pd.read_csv("./regression_data/Cereals/cereal.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the calories, protein, fat, sodium, fibre, complex carbohydrates, sugars, potassium, vitamins, the level of the display shelf, the weight in ounces of one serving, and the number of cups in one serving affect the rating of cereals"
#   "Please use the calories, protein, fat, sodium, fibre, complex carbohydrates, sugars, potassium, vitamins, the level of the display shelf, the weight in ounces of one serving, and the number of cups in one serving as independent variables and the the rating of cereals as the dependent variable and use a linear regression model to answer the following questions"
#   "What is the impact of different formulas on ratings?"
#   "Which components have a significant impact on ratings?"
#   "Which component has the greatest impact on ratings?"))

# colname=['Sex','Length','Diameter','Height','Whole weight', 'Shucked weight','Viscera weight','Shell weight','Rings']
# df = pd.read_csv('./regression_data/abalone/abalone.data',header=None, names=colname)
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, and Shell weight affect the number of rings of an abalone"
#   "Please use the Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, and Shell weight as independent variables and the the number of rings of an abalone as the dependent variable and use a linear regression model to answer the following questions"
#   "What are the effects of multiple different factors on the age of abalone?"
#   "What factors have a significant impact on the age of abalone?"
#   "Which factor has the greatest impact on abalone age?"))

# df = pd.read_csv('./regression_data/combined+cycle+power+plant/Folds5x2_pp.csv')
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the Temperature (AT), Exhaust Vacuum (V), Ambient Pressure (AP), and Relative Humidity (RH) affect the net hourly electrical energy output (PE)"
#   "Please use the Temperature (AT), Exhaust Vacuum (V), Ambient Pressure (AP), and Relative Humidity (RH)  as independent variables and the net hourly electrical energy output (PE) as the dependent variable and use a linear regression model to answer the following questions"
#   "What are the impacts of different environmental factors on hourly electrical energy output?"
#   "What factors have a significant impact on the hourly electrical energy output?"
#   "Which factor has the greatest impact on hourly electrical energy output?"))

### Below are regression model experiments
# df = pd.read_csv("./classifier_data/Iris.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the Sepal Length, Sepal Width, Petal Length, and Petal Width affect the iris Species."
#   "Please use the Sepal Length, Sepal Width, Petal Length, and Petal Width as independent variables and the Species as the dependent variable and use Linear Discriminant Analysis model to answer the following questions"
#   "1. How accurate is the classifier?"
#   "2. How will four data on Petal and Sepal affect different classification results?"
#   "3. On average, which of the four data on Petal and Sepal has the greatest impact on classification results?"
#   ))

# df = pd.read_csv("./classifier_data/glass.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the RI, Na, Mg, Al, Si, K, Ca, Ba, and Fe affect the glass Type."
#   "Please use the RI, Na, Mg, Al, Si, K, Ca, Ba, and Fe as independent variables and the Type as the dependent variable and use Linear Discriminant Analysis to answer the following questions"
#   "1. How accurate is the classifier?"
#   "2. What is the impact of different components on the classification of different glass types?"
#   "3. On average, which component has the greatest impact on classification results?"
#   ))


df = pd.read_csv("./classifier_data/diabetesWithhead.csv")
(create_pandas_dataframe_agent(chat, df, verbose=True).run
 ("This is a dataset about how pregnancy, glucose level, blood pressure, the triceps skin fold thickness, insulin level, pedigree, BMI, and age affect the diagnosis of diabetes"
  "Please use the pregnancy, glucose level, blood pressure, the triceps skin fold thickness, insulin level, pedigree, BMI, and age  as independent variables and the diabetes as the dependent variable and use a Logistic Regression model to answer the following questions"
  "1. Is the accuracy of the classifier trustworthy?"
  "2. What impact will different factors have on the classification of diabetes?"
  "3. What is the most influential factor for diabetes?"
  ))

# df = pd.read_csv("./classifier_data/classwinequalityred.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the citric acid, the chlorides, the free sulfur dioxide, the total sulfur dioxide, the sulphates, and the alcohol affect the quality"
#   "Please use the citric acid, the chlorides, the free sulfur dioxide, the total sulfur dioxide, the sulphates, and the alcohol as independent variables and the quality as the dependent variable and use Linear Discriminant Analysis to answer the following questions"
#   "1.How accurate is the classifier?"
#   "2.What are the effects of different factors on the quality of wine?"
#   "3.On average, which factor has the greatest impact on the quality of wine?"
#   ))

# df = pd.read_csv("./classifier_data/adult/adult.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the age, the educational-num (educational level), the fnlwgt(weight assigned by the Census Bureau), the capital-gain, the capital-loss, and the hours-per-week affect the income"
#   "Please use the age, the educational-num (educational level), the fnlwgt(weight assigned by the Census Bureau), the capital-gain, the capital-loss, and the hours-per-week as independent variables and the income as the dependent variable and use Linear Discriminant Analysis to answer the following questions"
#   "1. How accurate is it?"
#   "2. What are the effects of different factors on whether the income is greater than 50000?"
#   "3. What is the most important factor for whether the income is greater than 50000?"
#   ))

# df = pd.read_csv("./classifier_data/Hotel booking demand/hotel_bookings.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the lead time, the arrival date week, the arrival date day of the month, the adults number, the children number, the babies number, if it is repeated guest, the agent, the average daily rate(adr), if required car parking spaces, and the total of special requests affect if the hotel booking is cancelled"
#   "Please use the lead time, the arrival date week, the arrival date day of the month, the adults number, the children number, the babies number, if it is repeated guest, the agent, the average daily rate(adr), if required car parking spaces, and the total of special requests  as independent variables and the cancelled booking as the dependent variable and use a Logistic Regression model to answer the following questions"
#   "1. How accurate is the model?"
#   "2. What impact will different factors have on whether a reservation is cancelled?"
#   "3. Which factor has the greatest impact?"
#   ))

# df = pd.read_csv("./classifier_data/Credit Card Fraud Detection/creditcard.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the transaction amount, the V1, the V2, the V3, the V4, the V5, the V6, the V7, the V8, the V9, the V10, the V11, the V12, the V13, the V14, the V15, the V16, the V17, the V18, the V19, the V20, the V21, the V22, the V23, the V24, the V25, the V26, the V27, and the V28 affect if there are frauds or not"
#   "Please use the V1, the V2, the V3, the V4, the V5, the V6, the V7, the V8, the V9, the V10, the V11, the V12, the V13, the V14, the V15, the V16, the V17, the V18, the V19, the V20, the V21, the V22, the V23, the V24, the V25, the V26, the V27, and the V28 as independent variables and the frauds as the dependent variable and use a Logistic Regression model to answer the following questions"
#   "1.How accurate is the model in predicting the existence of credit card fraud?"
#   "2.How do different factors affect whether fraud occurs?"
#   "3.Which factor has the greatest impact on determining whether fraud has occurred?"
#   ))

# df = pd.read_csv("./classifier_data/superconductivty+data/train.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the mean atomic mass, the mean field, the Geometric Mean of atomic masses, the entropy atomic mass, the range atomic mass, the std atomic mass, the Geometric mean field, the entropy field, the range field, the std field, the mean atomic radius, the Geometric mean atomic radius, the entropy atomic radius, the range atomic radius, and the critical temp affect the number of elements of superconductors"
#   "Please use the mean atomic mass, the mean field, the Geometric Mean of atomic masses, the entropy atomic mass, the range atomic mass, the std atomic mass, the Geometric mean field, the entropy field, the range field, the std field, the mean atomic radius, the Geometric mean atomic radius, the entropy atomic radius, the range atomic radius, and the critical temp as independent variables and the number of elements of superconductors as the dependent variable and use Linear Discriminant Analysis to answer the following questions"
#   "1. What is the impact of different eigenvalues on determining the number of elements in superconductors?"
#   "2. On average, which feature value has the greatest impact on judgment?"
#   "3. From the perspective of accuracy, are these impacts credible?"
#   ))

# df = pd.read_csv("./classifier_data/titanic/train.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the Ticket class, the Age, the number of spouses aboard the Titanic, the number of children aboard the Titanic, and the Fare affect if Survived or not"
#   "Please use the Ticket class, the Age, the number of spouses aboard the Titanic, the number of children aboard the Titanic, and the Fare as independent variables and the Survived as the dependent variable and use a Logistic Regression model to answer the following questions"
#   "1. What impact did different conditions have on whether passengers survived?"
#   "2. Which factor has the greatest impact on survival?"
#   "3. How accurate is this model?"
#   ))

# df = pd.read_csv("./classifier_data/GiveMeSomeCredit/cs-training.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the age, the Revolving Utilization Of Unsecured Lines, the Debt Ratio, the Monthly Income, the Number Of Open Credit Lines And Loans, and the Number of Real Estate Loans Or Lines affect the Number Of Times 30-59 Days Past Due Not Worse (In the leftmost column from 0 to 98)"
#   "Please use the age, the Revolving Utilization Of Unsecured Lines, the Debt Ratio, the Monthly Income, the Number Of Open Credit Lines And Loans, and the Number of Real Estate Loans Or Lines as independent variables and the Number Of Times 30-59 Days Past Due Not Worse as the dependent variable and use Linear Discriminant Analysis to answer the following questions"
#   "1. What is the impact of different information about borrowers on the number of times they are overdue for more than 90 days?"
#   "2. Which information has the greatest impact on overdue times?"
#   "3. Is the accuracy of this classifier reliable?"
#   ))

# df = pd.read_csv("./classifier_data/Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn2.csv")
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how if the customer is a Senior Citizen, the number of months the customer has stayed with the company(tenure), and the Monthly Charges affect the if the Customer left within the last month"
#   "Please use the the customer is a Senior Citizen, the number of months the customer has stayed with the company(tenure), and the Monthly Charges as independent variables and the if the Customer left within the last month as the dependent variable and use a Logistic Regression model to answer the following questions"
#   "1. What is the impact of different factors on the classification of whether customers leave or not?"
#   "2. Which factor has the greatest impact on it?"
#   "3. Is the accuracy of this model trustworthy?"
#   ))

# df = pd.read_csv("./classifier_data/bank+marketing/bank.csv", delimiter=';')
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how the age, balance, marital stage, education level, and the previous effect if the client subscribed to a term deposit (y)"
#   "Please use the age, balance, marital stage, education level, and the previous as independent variables and the y as the dependent variable and use a Logistic Regression model to answer the following questions"
#   "1. Is the accuracy of the model trustworthy?"
#   "2. What are the effects of different factors on whether a product is purchased?"
#   "3. Which factor is most important for whether a product is ultimately purchased?"
#   ))

# df = pd.read_csv("./classifier_data/student+performance/student-mat.csv", delimiter=';')
# (create_pandas_dataframe_agent(chat, df, verbose=True).run
#  ("This is a dataset about how sex, age, the home to school travel time, study time, number of past class failures, if there is extra educational support (schoolsup), if there is family educational support (famsup), the quality of family relationships (famrel), the free time after school (freetime), the if going out with friends (goout), the workday alcohol consumption (Dalc), the weekend alcohol consumption (Walc), health status, and the number of absences (absences) affect the Grade(G3 in the leftmost column from 0 to 20)"
#   "Please use the sex, age, the home to school travel time, study time, number of past class failures, if there is extra educational support (schoolsup), if there is family educational support (famsup), the quality of family relationships (famrel), the free time after school (freetime), the if going out with friends (goout), the workday alcohol consumption (Dalc), the weekend alcohol consumption (Walc), health status, and the number of absences as independent variables and the Grade as the dependent variable and use Linear Discriminant Analysis to answer the following questions"
#   "1. Is the accuracy of the model trustworthy?"
#   "2. What is the impact of different conditions on whether students receive additional learning outside of school?"
#   "3. Which condition has the greatest impact?"
#   ))
#

