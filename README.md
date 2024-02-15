# DataQAHelper:computer:
###### *Developed by - Ruilin Wang *


# Overview:
This document provides an overview of how you can use this tool in your data exploration workflow.

Note: This document assumes that you have already cloned the package, if you have not done so already, please check out this [page](https://github.com/tangjikededela/DataQAHelper2).

----

# About the application:
DataQAHelper is a Python-based prototype that integrates a wide array of commonly used data science algorithms along with a comprehensive question bank commonly used to interpret analysis results. Once operational, it allows users to select the dataset they wish to analyze, pose questions, and either choose the model they wish to use or let the prototype automatically select the most appropriate model. Subsequently, the prototype will perform model fitting to complete the data analysis and answer the user's questions based on the analysis results. This question-and-answer process can be repeated until the user is satisfied.

____
## System Requirements 
* Python version  - '3.10'
____

## Packages Requirement

### The following packages are required to run the prototype:
```
beautifulsoup4==4.12.3
catboost==1.2.2
Jinja2==3.1.2
langchain==0.1.7
lightgbm==4.3.0
matplotlib==3.7.2
numpy==1.23.5
openai==0.27.8
pandas==1.5.3
pip==23.2.1
pycaret==3.0.4
python-dotenv==1.0.1
python_docx==0.8.11
Requests==2.31.0
scikit_learn==1.2.2
seaborn==0.13.2
statsmodels==0.14.0
xgboost==2.0.3
```
____
# Guidelines
## Requirements Installation
Before everything starts, please make sure that the Python version is 3.10.6, and Microsoft Visual C++ 14.0 or greater is required. Get it with: ["Microsoft C++ Build Tools"](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Then, please use the following commands to install necessary packages:
```
pip install -r requirements.txt
```
After completing the installation of the package, enter the following command on the terminal to run the prototype:
```
Python prototype. py
```

## Prototype workflow
**Step 1:**  
The user selects the dataset they wish to analyze.

**Step 2:**  
The user selects the independent and dependent variables.

**Step 3:**  
The user selects the type of machine learning (regression fitting or classification) / The user can let the prototype automatically select the most suitable model based on the criteria provided by the user.

**Step 4:**  
Select a required model and set hyperparameters / Check the results of the automatically selected model.

**Step 5:**  
The user needs to provide:  
1. A question they want to learn about from the dataset.  
2. Their OpenAI key.  
3. The model number used (optional).  
4. Background knowledge about the dataset (optional).  
5. A target text snippet to ensure the style and terminology of the output answer remain consistent (optional).

**Step 6:**  
Review the answer from the prototype.

**Step 7:**  
Pose another question.

**Step 8:**  
Review the answer from the prototype and choose to:  
1. Repeat Step 7.  
2. Summarize all Q&As and generate a text report.
