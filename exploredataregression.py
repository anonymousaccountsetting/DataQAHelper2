import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from jinja2 import Environment, FileSystemLoader
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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


class LinearRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression for Exploring data")
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.p_values = None
        self.iv_columns = []

    def create_widgets(self):
        self.step1_widgets() # For choosing dataset
        self.step2_1_widgets() # For choosing X
        self.step2_2_widgets() # For choosing y and model
        self.step3_widgets() # For showing R2
        self.step4_widgets() # For showing slopes and P-values
        self.step5_widgets() # For importance score
        self.step6_widgets() # For asking questions to ChatGPT, inputting key, and GPT model.
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

        # Use Statsmodels Linear Regression
        self.statsmodels_button = tk.Radiobutton(self.step2_2_frame, text="Use Statsmodels Linear Regression",
                                                 variable=self.model_var, value=0)
        self.statsmodels_button.grid(row=2, column=0, sticky=tk.E+tk.W, padx=10)

        # Use Scikit-learn Linear Regression
        self.sklearn_button = tk.Radiobutton(self.step2_2_frame, text="Use Scikit-learn Linear Regression",
                                             variable=self.model_var, value=1)
        self.sklearn_button.grid(row=3, column=0, sticky=tk.E+tk.W, padx=10)

        # Add Ridge Regression option
        self.ridge_checkbutton = tk.Radiobutton(self.step2_2_frame, text="Use Ridge Regression", variable=self.model_var,
                                                value=2)
        self.ridge_checkbutton.grid(row=4, column=0, sticky=tk.E+tk.W, padx=10)

        self.ridge_alpha_label = tk.Label(self.step2_2_frame, text="Enter Alpha value (default 1.0):")
        self.ridge_alpha_label.grid(row=5, column=0, sticky=tk.E+tk.W, padx=10)
        self.ridge_alpha_entry = tk.Entry(self.step2_2_frame)
        self.ridge_alpha_entry.grid(row=6, column=0, sticky=tk.E+tk.W, padx=10, pady=5)

        # Add Lasso Regression option
        self.lasso_checkbutton = tk.Radiobutton(self.step2_2_frame, text="Use Lasso Regression", variable=self.model_var,
                                                value=3)
        self.lasso_checkbutton.grid(row=2, column=1, sticky=tk.E+tk.W, padx=10)

        self.lasso_alpha_label = tk.Label(self.step2_2_frame, text="Enter Alpha value (default 1.0):")
        self.lasso_alpha_label.grid(row=3, column=1, sticky=tk.E+tk.W, padx=10)
        self.lasso_alpha_entry = tk.Entry(self.step2_2_frame)
        self.lasso_alpha_entry.grid(row=4, column=1, sticky=tk.E+tk.W, padx=10, pady=5)

        # Add Bayesian Ridge Regression option
        self.bayesian_ridge_checkbutton = tk.Radiobutton(self.step2_2_frame, text="Use Bayesian Ridge Regression",
                                                         variable=self.model_var, value=4)
        self.bayesian_ridge_checkbutton.grid(row=5, column=1, sticky=tk.E+tk.W, padx=10)

        self.bayesian_ridge_label = tk.Label(self.step2_2_frame, text="Enter two Alpha values, and Lambda values:")
        self.bayesian_ridge_label.grid(row=6, column=1, sticky=tk.E+tk.W, padx=10)
        self.alpha_1_entry = tk.Entry(self.step2_2_frame, width=8)
        self.alpha_1_entry.grid(row=7, column=1, sticky=tk.E+tk.W, padx=10)
        self.alpha_2_entry = tk.Entry(self.step2_2_frame, width=8)
        self.alpha_2_entry.grid(row=7, column=1, padx=(100, 0), sticky=tk.E+tk.W)
        self.lambda_1_entry = tk.Entry(self.step2_2_frame, width=8)
        self.lambda_1_entry.grid(row=7, column=1, padx=(190, 0), sticky=tk.E+tk.W)
        self.lambda_2_entry = tk.Entry(self.step2_2_frame, width=8)
        self.lambda_2_entry.grid(row=7, column=1, padx=(280, 0), sticky=tk.E+tk.W)

        # Create a new frame for the 'Next' and 'Back' buttons
        self.buttons_frame = tk.Frame(self.step2_2_frame)
        self.buttons_frame.grid(row=8, column=0, columnspan=2, pady=10)

        self.back_button = tk.Button(self.buttons_frame, text="Back", command=self.go_to_step2_1)
        self.back_button.grid(row=0, column=0, sticky=tk.E+tk.W,padx=10)

        self.next_button = tk.Button(self.buttons_frame, text="Next", command=self.go_to_step3)
        self.next_button.grid(row=0, column=1, sticky=tk.E+tk.W,padx=10)

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

        self.slope_text = tk.Text(self.step4_frame, width=100, height=40)
        self.slope_text.pack(pady=10)

        self.back_button = tk.Button(self.step4_frame, text="Back", command=self.go_to_step3)
        self.back_button.pack(side=tk.LEFT, pady=10)

        self.next_button = tk.Button(self.step4_frame, text="Next", command=self.go_to_step5)
        self.next_button.pack(side=tk.LEFT, pady=10)



    def step5_widgets(self):

        self.step5_frame = tk.Frame(self.root)

        self.fig_canvas = None

        # Add a label to display the most important X
        self.most_important_x_label = tk.Label(self.step5_frame, text="")
        self.most_important_x_label.pack(pady=5)

        self.question_answer= tk.Text(self.step5_frame, width=100, height=10)
        self.question_answer.pack(pady=10)

        self.back_button = tk.Button(self.step5_frame, text="Back", command=self.go_to_step4)
        self.back_button.pack(side=tk.LEFT, pady=10)

        self.next_button = tk.Button(self.step5_frame, text="Next", command=self.go_to_step6)
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

        self.back_button = tk.Button(self.step6_frame, text="Back", command=self.go_to_step5)
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
        self.slope_text.delete(1.0, tk.END)
        coef_text = f"Intercept: {self.coefficients[0]}\n"

        answers=[]
        questions=[]
        for index, row in self.coef_pval_df.iterrows():
            question=linearQuestion.render(section=2,xcol=row['Xcol'],ycol=self.dv_column)
            answer=linearSummary.render(coeff=row['Coefficients'],p=row['P-values'],xcol=row['Xcol'],ycol=self.dv_column)
            questions.append(question)
            answers.append(answer)

        for i, (col, coef) in enumerate(zip(self.iv_columns, self.coefficients[1:])):
            coef_text += f"{col}: {questions[i]}\n{answers[i]}\n\n"

        self.slope_text.insert(tk.END, coef_text)
        self.hide_all_frames()
        self.step4_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step5(self):
        self.plot_importance_score_map()
        self.hide_all_frames()
        self.step5_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step6(self):
        self.hide_all_frames()
        self.step6_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step7(self):
        user_api_key = self.api_key.get()
        if not user_api_key:
            self.answerbyGPT.config(text="Invalid API key. Please enter a valid API key.")
            return
        key=user_api_key

        gpt_model_name_input = self.gpt_model_name.get()
        gpt_model_name = gpt_model_name_input if gpt_model_name_input else "gpt-3.5-turbo"

        if self.model_var.get() == 0:
            modelname="Statsmodels Linear Regression"
        elif self.model_var.get() == 1:
            modelname="Scikit-learn Linear Regression"
        elif self.model_var.get() == 2:
            modelname="Ridge Regression"
        elif self.model_var.get() == 3:
            modelname="Lasso Regression"
        elif self.model_var.get() == 4:
            modelname="Bayesian Ridge Regression"

        self.user_text_label.config(text=f"User input: {self.user_text.get()}")
        self.r_squared_label_step7.config(text=f"R-squared: {self.r_squared:.4f}")
        self.selected_y_label_step7.config(text=f"Dependent variable (y): {self.dv_column}")
        self.selected_x_label_step7.config(text=f"Independent variables (X): {', '.join(self.iv_columns)}")

        modelinformation=f"The R-squared of the model is: {self.r_squared:.4f}\n"+str(self.coef_pval_df)

        coef_text = "Coefficients:\n\n"
        coef_text += f"Intercept: {self.coefficients[0]}\n"
        for col, coef, pval in zip(self.iv_columns, self.coefficients[1:], self.p_values[1:]):
            coef_text += f"{col}: {coef} (P-value: {pval})\n"
        self.slope_and_pvalues_text.delete(1.0, tk.END)
        self.slope_and_pvalues_text.insert(tk.END, coef_text)

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
        for frame in [self.step1_frame, self.step2_1_frame, self.step2_2_frame, self.step3_frame, self.step4_frame,
                      self.step5_frame, self.step6_frame, self.step7_frame]:
            frame.pack_forget()

    def perform_regression(self):
        iv_indices = [self.var_listbox.get(0, tk.END).index(col) for col in self.iv_columns]
        dv_index = self.var_listbox.get(0, tk.END).index(self.dv_column)
        X = self.df.iloc[:, iv_indices]
        y = self.df.iloc[:, dv_index]
        if self.model_var.get() == 0:
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            self.coefficients = model.params  # Save slope
            self.p_values = model.pvalues  # Save p-values
            self.r_squared = model.rsquared
        elif self.model_var.get()==1:
            model = LinearRegression().fit(X, y)
            self.coefficients = np.append(model.intercept_, model.coef_)
            self.p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
            self.r_squared = model.score(X, y)
        elif self.model_var.get()==2:
            self.p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            alpha = float(self.ridge_alpha_entry.get() or 1.0)
            model = Ridge(alpha=alpha).fit(X_train, y_train)
            self.coefficients = np.append(model.intercept_, model.coef_)
            self.r_squared = r2_score(y_test, model.predict(X_test))
        elif self.model_var.get()==3:
            self.p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            alpha = float(self.lasso_alpha_entry.get() or 1.0)
            model = Lasso(alpha=alpha).fit(X_train, y_train)
            self.coefficients = np.append(model.intercept_, model.coef_)
            self.r_squared = r2_score(y_test, model.predict(X_test))
        elif self.model_var.get()==4:
            self.p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            alpha_1 = float(self.alpha_1_entry.get() or 1e-06)
            alpha_2 = float(self.alpha_2_entry.get() or 1e-06)
            lambda_1 = float(self.lambda_1_entry.get() or 1e-06)
            lambda_2 = float(self.lambda_2_entry.get() or 1e-06)
            model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2).fit(X_train, y_train)
            # Get the coefficients for each predictor
            self.coefficients = np.append(model.intercept_, model.coef_)
            # Calculate R-squared
            y_pred = model.predict(X_test)
            self.r_squared = r2_score(y_test, y_pred)

        # Create a dataframe with coefficients and p-values (if available) for independent variables only
        data_dict = {'Coefficients': self.coefficients[1:]}
        data_dict['P-values'] = self.p_values[1:]
        self.coef_pval_df = pd.DataFrame(data_dict, index=self.iv_columns)
        self.coef_pval_df.index.name = "Xcol"
        self.coef_pval_df = self.coef_pval_df.reset_index()
        question=linearQuestion.render(section=1,indeNum=len(self.coef_pval_df['Xcol']),xcol=self.coef_pval_df['Xcol'],ycol=self.dv_column)
        intro=linearSummary2.render(r2=self.r_squared)
        self.r_squared_label.config(text=question+f"\n"+ intro)

    def plot_importance_score_map(self):
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()

        fig = plt.figure(figsize=(3, 2))

        plt.barh(range(len(self.coefficients) - 1), self.coefficients[1:])
        plt.yticks(range(len(self.coefficients) - 1), self.iv_columns)
        plt.xlabel("Importance score")
        plt.tight_layout()

        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.step5_frame)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Update the most_important_x
        question=linearQuestion.render(section=3,ycol=self.dv_column)
        most_important_x = self.iv_columns[np.argmax(np.abs(self.coefficients[1:]))]

        # Display X variables with positive and negative slopes
        positive_slopes = [x for i, x in enumerate(self.iv_columns) if self.coefficients[i + 1] > 0]
        negative_slopes = [x for i, x in enumerate(self.iv_columns) if self.coefficients[i + 1] < 0]
        # Display X variables with P-values greater than and less than 0.05
        high_p_values = [x for i, x in enumerate(self.iv_columns) if self.p_values[i+1] > 0.05]
        low_p_values = [x for i, x in enumerate(self.iv_columns) if self.p_values[i+1] <= 0.05]

        answer=linearSummary3.render(ss=low_p_values,lenss=len(low_p_values),nss=high_p_values,lennss=len(high_p_values),pf=positive_slopes,lenpf=len(positive_slopes),nf=negative_slopes,lennf=len(negative_slopes),ycol=self.dv_column,imp=most_important_x)

        self.question_answer.delete(1.0, tk.END)
        self.question_answer.insert(tk.END, question+"\n"+answer)

    def on_close(self):
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    # alphas = np.logspace(-4, 4, 100)
    # print(alphas)
    root = tk.Tk()
    root.geometry("800x600")
    app = LinearRegressionApp(root)
    root.mainloop()

