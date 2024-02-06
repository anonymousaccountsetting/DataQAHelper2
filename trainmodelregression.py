import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from jinja2 import Environment, FileSystemLoader
import openai
import json
import requests
# Loading the folder that contains the txt templates

file_loader = FileSystemLoader('./apptemplates')

# Creating a Jinja Environment

env = Environment(loader=file_loader)

# Loading the Jinja templates from the folder
# For the regression
linearSummary = env.get_template('linearSummary.txt')
linearSummary2 = env.get_template('linearSummary2.txt')
linearSummary3 = env.get_template('linearSummary3.txt')
linearQuestion = env.get_template('linearQuestionset.txt')

# For ChatGPT
databackground = env.get_template('databackground.txt')
questionrequest=env.get_template('question_request.txt')


class LinearRegressionModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression for a Good Model")
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.iv_columns = []

    def create_widgets(self):
        self.step1_widgets() # For choosing dataset
        self.step2_1_widgets() # For choosing X
        self.step2_2_widgets() # For choosing y and model
        self.step3_widgets() # For showing R2
        self.step4_widgets() # For showing mse...
        self.step6_widgets() # For asking questions to ChatGPT, inputting key, and GPT model
        self.step7_widgets() # For showing the answer
        self.step8_widgets() # For asking another question
        self.step9_widgets() # For showing the new answer

    def step1_widgets(self):
        self.step1_frame = tk.Frame(self.root)
        self.step1_frame.pack(fill=tk.BOTH, expand=True)

        self.open_button = tk.Button(self.step1_frame, text="Select CSV File", command=self.load_csv)
        self.open_button.pack(pady=10)

        self.next_button = tk.Button(self.step1_frame, text="Next", command=self.go_to_step2_1)
        self.next_button.pack(pady=10)

    def step2_1_widgets(self):
        self.step2_1_frame = tk.Frame(self.root)

        self.var_listbox = tk.Listbox(self.step2_1_frame, selectmode=tk.MULTIPLE)
        self.var_listbox.pack(fill=tk.BOTH, expand=True)

        self.var_label = tk.Label(self.step2_1_frame, text="Select independent variables (X)")
        self.var_label.pack(pady=10)

        self.back_button = tk.Button(self.step2_1_frame, text="Back", command=self.go_to_step1)
        self.back_button.pack(side=tk.LEFT, pady=10)

        self.next_button = tk.Button(self.step2_1_frame, text="Next", command=self.go_to_step2_2)
        self.next_button.pack(side=tk.LEFT, pady=10)

    def step2_2_widgets(self):
        self.step2_2_frame = tk.Frame(self.root)

        self.dep_var_label = tk.Label(self.step2_2_frame, text="Select dependent variable (y)")
        self.dep_var_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.dep_var_listbox = tk.Listbox(self.step2_2_frame, width=50, height=10)
        self.dep_var_listbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.model_var = tk.IntVar()

        # Use Gradient Boosting Regressor
        self.gradient_boosting_button = tk.Radiobutton(self.step2_2_frame, text="Use Gradient Boosting",
                                                       variable=self.model_var, value=0)
        self.gradient_boosting_button.grid(row=2, column=0, sticky=tk.E + tk.W, padx=10)

        # Use Random Forest Regressor
        self.random_forest_button = tk.Radiobutton(self.step2_2_frame, text="Use Random Forest Regression",
                                                   variable=self.model_var, value=1)
        self.random_forest_button.grid(row=3, column=0, sticky=tk.E + tk.W, padx=10)

        # Use CatBoost Regressor
        self.catboost_button = tk.Radiobutton(self.step2_2_frame, text="Use CatBoost Regression",
                                              variable=self.model_var, value=2)
        self.catboost_button.grid(row=4, column=0, sticky=tk.E + tk.W, padx=10)

        # Use Light Gradient Boosting Machine
        self.light_gbm_button = tk.Radiobutton(self.step2_2_frame, text="Use Light Gradient Boosting Machine",
                                               variable=self.model_var, value=3)
        self.light_gbm_button.grid(row=7, column=0, sticky=tk.E + tk.W, padx=10)

        # Use AdaBoost Regressor
        self.adaboost_button = tk.Radiobutton(self.step2_2_frame, text="Use AdaBoost Regression",
                                              variable=self.model_var, value=4)
        self.adaboost_button.grid(row=8, column=0, sticky=tk.E + tk.W, padx=10)

        # Use Extreme Gradient Boosting
        self.xgboost_button = tk.Radiobutton(self.step2_2_frame, text="Use Extreme Gradient Boosting",
                                             variable=self.model_var, value=5)
        self.xgboost_button.grid(row=9, column=0, sticky=tk.E + tk.W, padx=10)

        # Use Decision Tree Regressor
        self.decision_tree_button = tk.Radiobutton(self.step2_2_frame, text="Use Decision Tree Regression",
                                                   variable=self.model_var, value=6)
        self.decision_tree_button.grid(row=10, column=0, sticky=tk.E + tk.W, padx=10)

        self.back_button = tk.Button(self.step2_2_frame, text="Back", command=self.go_to_step2_1)
        self.back_button.grid(row=11, sticky=tk.E + tk.W, padx=10)

        self.next_button = tk.Button(self.step2_2_frame, text="Next", command=self.go_to_step3)
        self.next_button.grid(row=11, sticky=tk.E+tk.W,padx=10)

    def step3_widgets(self):
        self.step3_frame = tk.Frame(self.root)

        self.selected_x_label = tk.Label(self.step3_frame, text="")
        self.selected_x_label.pack(pady=5)

        self.selected_y_label = tk.Label(self.step3_frame, text="")
        self.selected_y_label.pack(pady=5)

        self.r_squared_label = tk.Label(self.step3_frame, text="")
        self.r_squared_label.pack(pady=10)

        self.back_button = tk.Button(self.step3_frame, text="Back", command=self.go_to_step2_2)
        self.back_button.pack(side=tk.LEFT, pady=10)

        self.next_button = tk.Button(self.step3_frame, text="Next", command=self.go_to_step4)
        self.next_button.pack(side=tk.LEFT, pady=10)

    def step4_widgets(self):
        self.step4_frame = tk.Frame(self.root)

        self.error_metrics_label = tk.Label(self.step4_frame, text="Error Metrics")
        self.error_metrics_label.pack(pady=10)

        self.mse_label = tk.Label(self.step4_frame, text="Mean Squared Error (MSE):")
        self.mse_label.pack(pady=5)

        self.rmse_label = tk.Label(self.step4_frame, text="Root Mean Squared Error (RMSE):")
        self.rmse_label.pack(pady=5)

        self.mae_label = tk.Label(self.step4_frame, text="Mean Absolute Error (MAE):")
        self.mae_label.pack(pady=5)

        self.tip = tk.Label(self.step4_frame, text="Lower values for MSE, RMSE, and MAE are generally better.")
        self.tip.pack(pady=5)

        self.back_button = tk.Button(self.step4_frame, text="Back", command=self.go_to_step3)
        self.back_button.pack(side=tk.LEFT, pady=10)

        self.next_button = tk.Button(self.step4_frame, text="Next", command=self.go_to_step6)
        self.next_button.pack(side=tk.LEFT, pady=10)

    def step6_widgets(self):
        self.step6_frame = tk.Frame(self.root)

        self.user_text_label = tk.Label(self.step6_frame, text="Enter question(s) for Chat-GPT:")
        self.user_text_label.pack(pady=10)

        self.user_text = tk.StringVar()
        self.user_text_entry = tk.Entry(self.step6_frame, textvariable=self.user_text)
        self.user_text_entry.pack(pady=10)

        self.api_key_label = tk.Label(self.step6_frame, text="Enter your API key:")
        self.api_key_label.pack(pady=10)

        self.api_key = tk.StringVar()
        self.api_key_entry = tk.Entry(self.step6_frame, textvariable=self.api_key, show='*')
        self.api_key_entry.pack(pady=10)

        self.gpt_model_label = tk.Label(self.step6_frame, text="Enter GPT model name (default: gpt-3.5-turbo):")
        self.gpt_model_label.pack(pady=10)

        self.gpt_model_name = tk.StringVar()
        self.gpt_model_entry = tk.Entry(self.step6_frame, textvariable=self.gpt_model_name)
        self.gpt_model_entry.pack(pady=10)

        self.back_button = tk.Button(self.step6_frame, text="Back", command=self.go_to_step4)
        self.back_button.pack(side=tk.LEFT, pady=10)

        self.next_button = tk.Button(self.step6_frame, text="Next", command=self.go_to_step7)
        self.next_button.pack(side=tk.LEFT, pady=10)

    def step7_widgets(self):

        self.step7_frame = tk.Frame(self.root)

        self.selected_x_label_step7 = tk.Label(self.step7_frame, text="")
        self.selected_x_label_step7.pack(pady=5)

        self.selected_y_label_step7 = tk.Label(self.step7_frame, text="")
        self.selected_y_label_step7.pack(pady=5)

        self.r_squared_label_step7 = tk.Label(self.step7_frame, text="")
        self.r_squared_label_step7.pack(pady=10)

        self.slope_and_pvalues_text = tk.Text(self.step7_frame, width=40, height=10)
        self.slope_and_pvalues_text.pack(pady=10)

        self.user_text_label = tk.Label(self.step7_frame, text="")
        self.user_text_label.pack(pady=5)

        self.answerbyGPT=tk.Label(self.step7_frame, text="")
        self.answerbyGPT.pack(pady=5)

        self.back_button = tk.Button(self.step7_frame, text="Back", command=self.go_to_step6)
        self.back_button.pack(side=tk.LEFT, pady=10)

        self.ask_another_button = tk.Button(self.step7_frame, text="Ask another question", command=self.go_to_step8)
        self.ask_another_button.pack(side=tk.LEFT, pady=10)

        self.do_it_again_button = tk.Button(self.step7_frame, text="Do it again", command=self.go_to_step1)
        self.do_it_again_button.pack(side=tk.LEFT, pady=10)

    def step8_widgets(self):
        self.step8_frame = tk.Frame(self.root)

        self.ask_another_question_label = tk.Label(self.step8_frame, text="Enter another question for Chat-GPT:")
        self.ask_another_question_label.pack(pady=10)

        self.another_question_text = tk.StringVar()
        self.another_question_entry = tk.Entry(self.step8_frame, textvariable=self.another_question_text)
        self.another_question_entry.pack(pady=10)

        self.back_button = tk.Button(self.step8_frame, text="Back", command=self.go_to_step7)
        self.back_button.pack(side=tk.LEFT, pady=10)

        self.next_button = tk.Button(self.step8_frame, text="Next", command=self.go_to_step9)
        self.next_button.pack(side=tk.LEFT, pady=10)

    # Add step9_widgets function
    def step9_widgets(self):
        self.step9_frame = tk.Frame(self.root)

        self.new_answer_label = tk.Label(self.step9_frame, text="")
        self.new_answer_label.pack(pady=5)

        self.ask_another_button = tk.Button(self.step9_frame, text="Ask another question", command=self.go_to_step8)
        self.ask_another_button.pack(side=tk.LEFT, pady=10)

        self.do_it_again_button = tk.Button(self.step9_frame, text="Do it again", command=self.go_to_step1)
        self.do_it_again_button.pack(side=tk.LEFT, pady=10)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.var_listbox.delete(0, tk.END)
            self.dep_var_listbox.delete(0, tk.END)
            for column in self.df.columns:
                self.var_listbox.insert(tk.END, column)
                self.dep_var_listbox.insert(tk.END, column)

    def preselect_iv_variables(self):
        if self.iv_columns:
            self.var_listbox.selection_clear(0, tk.END)
            for col in self.iv_columns:
                index = self.var_listbox.get(0, tk.END).index(col)
                self.var_listbox.selection_set(index)

    def update_iv_columns(self):
        iv_indices = list(self.var_listbox.curselection())
        self.iv_columns = [self.var_listbox.get(i) for i in iv_indices]

    def go_to_step1(self):
        self.hide_all_frames()
        self.step1_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step2_1(self):
        self.hide_all_frames()
        self.step2_1_frame.pack(fill=tk.BOTH, expand=True)
        self.preselect_iv_variables()

    def go_to_step2_2(self):
        iv_indices = list(self.var_listbox.curselection())
        self.iv_columns = [self.var_listbox.get(i) for i in iv_indices]
        self.update_iv_columns()
        self.hide_all_frames()
        self.step2_2_frame.pack(fill=tk.BOTH, expand=True)
        self.preselect_iv_variables()

    def go_to_step3(self):
        dv_index = self.dep_var_listbox.curselection()[0]
        self.dv_column = self.dep_var_listbox.get(dv_index)

        self.selected_x_label.config(text=f"Independent variables (X): {', '.join(self.iv_columns)}")
        self.selected_y_label.config(text=f"Dependent variable (y): {self.dv_column}")

        self.perform_regression()

        self.hide_all_frames()
        self.step3_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step4(self):

        self.mse_label.config(text=f"Mean Squared Error (MSE): {self.mean_mse}")
        self.rmse_label.config(text=f"Root Mean Squared Error (RMSE): {self.mean_rmse}")
        self.mae_label.config(text=f"Mean Absolute Error (MAE): {self.mean_mae}")

        self.hide_all_frames()
        self.step4_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step6(self):
        self.hide_all_frames()
        self.step6_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step7(self):

        #key = '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'
        user_api_key = self.api_key.get()
        if not user_api_key:
            self.answerbyGPT.config(text="Invalid API key. Please enter a valid API key.")
            return
        key=user_api_key

        gpt_model_name_input = self.gpt_model_name.get()
        gpt_model_name = gpt_model_name_input if gpt_model_name_input else "gpt-3.5-turbo"

        if self.model_var.get() == 0:
            modelname="Gradient Boosting"
        elif self.model_var.get() == 1:
            modelname="Random Forest Regression"
        elif self.model_var.get() == 2:
            modelname="CatBoost Regression"
        elif self.model_var.get() == 3:
            modelname="Light Gradient Boosting Machine"
        elif self.model_var.get() == 4:
            modelname="AdaBoost Regression"
        elif self.model_var.get() == 5:
            modelname = "Extreme Gradient Boosting"
        elif self.model_var.get() == 6:
            modelname="Decision Tree Regression"

        self.user_text_label.config(text=f"User input: {self.user_text.get()}")
        self.r_squared_label_step7.config(text=f"R-squared: {self.r_squared:.4f}")
        self.selected_y_label_step7.config(text=f"Dependent variable (y): {self.dv_column}")
        self.selected_x_label_step7.config(text=f"Independent variables (X): {', '.join(self.iv_columns)}")

        modelinformation=f"The R-squared of the model is: {self.r_squared:.4f}\n"+\
                         f"The Mean Squared Error (MSE) of the model is: {str(self.mse_scores)}\n"+\
                         f"The Root Mean Squared Error (RMSE) of the model is: {str(self.rmse_scores)}\n"+\
                         f"The Mean Absolute Error (MAE) of the model is: {str(self.mae_scores)}\n"


        url, background, chatmodel, headers, messages=self.set_chatGPT(self.iv_columns, self.dv_column, modelname, modelinformation,key)
        payload, messages=self.set_payload(self.user_text.get(),gpt_model_name,messages)
        output, messages=self.send_response_receive_output(url, headers, payload, messages)

        self.messages = messages
        self.url = url
        self.headers = headers
        self.gpt_model_name=gpt_model_name

        self.answerbyGPT.config(text=f"Answer from Chat-GPT: {output}")

        self.hide_all_frames()
        self.step7_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step8(self):
        self.hide_all_frames()
        self.step8_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step9(self):
        question = self.another_question_text.get()
        payload, messages = self.set_payload(question, self.gpt_model_name, self.messages)
        output, messages = self.send_response_receive_output(self.url, self.headers, payload, messages)
        self.messages = messages
        print(self.messages)
        self.new_answer_label.config(text=f"New answer from Chat-GPT: {output}")
        self.hide_all_frames()
        self.step9_frame.pack(fill=tk.BOTH, expand=True)

    def set_chatGPT(self, Xcol, ycol, modelname, modelinformation, key, chatmodel="gpt-3.5-turbo", url="https://api.openai.com/v1/chat/completions"):
        openai.api_key = key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        background = databackground.render(xcol=Xcol, ycol=ycol, modelname=modelname, modelinformation=modelinformation)
        messages = [{"role": "system", "content": background}, ]
        return (url, background, chatmodel, headers, messages)

    def set_payload(self, message,GPTmodelname="gpt-3.5-turbo", messages=[]):
        messages.append({"role": "user", "content": message}, )
        payload = {
            "model": GPTmodelname,
            "messages": messages,
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        return (payload, messages)

    def send_response_receive_output(self, URL, headers, payload, messages):
        response = requests.post(URL, headers=headers, json=payload, stream=False)
        print(json.loads(response.content))
        output = json.loads(response.content)["choices"][0]['message']['content']
        messages.append({"role": "assistant", "content": output})
        return (output, messages)

    def hide_all_frames(self):
        for frame in [self.step1_frame, self.step2_1_frame, self.step2_2_frame, self.step3_frame, self.step4_frame,self.step6_frame, self.step7_frame]:
            frame.pack_forget()

    def perform_regression(self):
        iv_indices = [self.var_listbox.get(0, tk.END).index(col) for col in self.iv_columns]
        dv_index = self.var_listbox.get(0, tk.END).index(self.dv_column)
        X = self.df.iloc[:, iv_indices]
        y = self.df.iloc[:, dv_index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_var.get() == 0:
            model = GradientBoostingRegressor()
        elif self.model_var.get() == 1:
            model = RandomForestRegressor()
        elif self.model_var.get() == 2:
            model = CatBoostRegressor(verbose=0)
        elif self.model_var.get() == 3:
            model = LGBMRegressor()
        elif self.model_var.get() == 4:
            model = AdaBoostRegressor()
        elif self.model_var.get() == 5:
            model = XGBRegressor()
        elif self.model_var.get() == 6:
            model = DecisionTreeRegressor()


        # Fit the model
        model.fit(X_train, y_train)

        # Calculate R-squared
        self.r_squared = r2_score(y_test, model.predict(X_test))

        k = 5
        # Calculate negative MSE scores
        neg_mse_scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')

        # Convert negative MSE scores to positive MSE scores
        self.mse_scores = -neg_mse_scores
        self.mean_mse = np.mean(self.mse_scores)
        self.std_mse = np.std(self.mse_scores)
        # Calculate RMSE scores
        self.rmse_scores = np.sqrt(self.mse_scores)
        self.mean_rmse = np.mean(self.rmse_scores)
        self.std_rmse = np.std(self.rmse_scores)
        # Calculate MAE scores
        self.mae_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')
        self.mean_mae = np.mean(self.mae_scores)
        self.std_mae = np.std(self.mae_scores)
        question=linearQuestion.render(section=1,indeNum=len(self.iv_columns),xcol=self.iv_columns,ycol=self.dv_column)
        intro=linearSummary2.render(r2=self.r_squared)
        self.r_squared_label.config(text=question+f"\n"+ intro)

    def on_close(self):
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = LinearRegressionModelApp(root)
    root.mainloop()
