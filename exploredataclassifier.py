import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from jinja2 import Environment, FileSystemLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve
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
# For the classifier
classifiersummary1=env.get_template('classifiersummary1.txt')
classifiersummary2=env.get_template('classifiersummary2.txt')
classifiersummary3=env.get_template('classifiersummary3.txt')
classifierquestion=env.get_template('classifierQuestionset.txt')

# For ChatGPT
databackground = env.get_template('databackground.txt')
questionrequest=env.get_template('question_request.txt')


class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classifier for Exploring data")
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.p_values = None
        self.iv_columns = []

    def create_widgets(self):
        self.step1_widgets() # For choosing dataset
        self.step2_1_widgets() # For choosing X
        self.step2_2_widgets() # For choosing y and model
        self.step3_widgets() # For showing Accuracy
        self.step4_widgets() # For showing coefficient
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
        self.statsmodels_button = tk.Radiobutton(self.step2_2_frame, text="Use Logistic Regression",
                                                 variable=self.model_var, value=0)
        self.statsmodels_button.grid(row=2, column=0, sticky=tk.E+tk.W, padx=10)

        # Use Scikit-learn Linear Regression
        self.sklearn_button = tk.Radiobutton(self.step2_2_frame, text="Use Linear Discriminant Analysis",
                                             variable=self.model_var, value=1)
        self.sklearn_button.grid(row=3, column=0, sticky=tk.E+tk.W, padx=10)

        # Add Ridge Regression option
        self.ridge_checkbutton = tk.Radiobutton(self.step2_2_frame, text="Use SVM - Linear Kernel", variable=self.model_var,
                                                value=2)
        self.ridge_checkbutton.grid(row=4, column=0, sticky=tk.E+tk.W, padx=10)

        # Add Lasso Regression option
        self.lasso_checkbutton = tk.Radiobutton(self.step2_2_frame, text="Use Ridge Classifier", variable=self.model_var,
                                                value=3)
        self.lasso_checkbutton.grid(row=5, column=0, sticky=tk.E+tk.W, padx=10)

        # Create a new frame for the 'Next' and 'Back' buttons
        self.buttons_frame = tk.Frame(self.step2_2_frame)
        self.buttons_frame.grid(row=7, column=0, columnspan=2, pady=10)

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

        self.accuracy_label = tk.Label(self.step3_frame, text="")
        self.accuracy_label.pack(pady=10)

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

        self.accuracy_label_step7 = tk.Label(self.step7_frame, text="")
        self.accuracy_label_step7.pack(pady=10)

        self.coefficients_text = tk.Text(self.step7_frame, width=40, height=10)
        self.coefficients_text.pack(pady=10)

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

        if self.model_var.get() == 0:
            self.modelname="Logistic Regression"
        elif self.model_var.get() == 1:
            self.modelname="Linear Discriminant Analysis"
        elif self.model_var.get() == 2:
            self.modelname="SVM-Linear Kernel"
        elif self.model_var.get() == 3:
            self.modelname="Ridge Classifier"

        self.perform_regression()

        self.hide_all_frames()
        self.step3_frame.pack(fill=tk.BOTH, expand=True)

    def go_to_step4(self):
        self.slope_text.delete(1.0, tk.END)
        text=''
        for target_names, row in self.coeff_df.iterrows():
            question=classifierquestion.render(section=2, classes=target_names,ycol=self.dv_column)
            text=text+question
            positivecol = []
            negativecol = []
            notchangecol = []
            for self.iv_columns, coeff in row.items():
                if coeff > 0:
                    positivecol.append(self.iv_columns)
                elif coeff < 0:
                    negativecol.append(self.iv_columns)
                else:
                    notchangecol.append(self.iv_columns)
            answer=classifiersummary2.render(classes=target_names,pc=positivecol,nc=negativecol,ncc=notchangecol)
            text=text+'\n'+answer+'\n\n'
        print(text)
        self.slope_text.insert(tk.END, text)
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

        self.user_text_label.config(text=f"User input: {self.user_text.get()}")
        self.accuracy_label_step7.config(text=f"R-squared: {self.accuracy:.4f}")
        self.selected_y_label_step7.config(text=f"Dependent variable (y): {self.dv_column}")
        self.selected_x_label_step7.config(text=f"Independent variables (X): {', '.join(self.iv_columns)}")

        modelinformation=f"The accuracy of the model is: {self.accuracy:.4f}\n"+str(self.coefficients)

        coef_text = "Coefficients:\n\n"
        for i in range(len(self.iv_columns)):
            coef_text += f"{self.iv_columns[i]}: {self.coefficients[i]}\n"
        self.coefficients_text.delete(1.0, tk.END)
        self.coefficients_text.insert(tk.END, coef_text)

        url, background, chatmodel, headers, messages=self.set_chatGPT(self.iv_columns, self.dv_column, self.modelname, modelinformation,key)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if self.model_var.get() == 0:
            model= LogisticRegression(max_iter=1000)

        elif self.model_var.get()==1:
            model=LinearDiscriminantAnalysis()

        elif self.model_var.get()==2:
            model=SVC(kernel='linear')

        elif self.model_var.get()==3:
            model=RidgeClassifier()

        model.fit(X_train, y_train)

        self.coefficients=model.coef_
        target_names = model.classes_
        # Create a dataframe using the extracted coefficients, X columns, and target names
        self.coeff_df = pd.DataFrame(model.coef_, columns=self.iv_columns, index=target_names)
        # for row in self.coeff_df:
        #     print(self.coeff_df[row])
        # print(self.coeff_df)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        # Calculate and output the accuracy
        self.accuracy = accuracy_score(y_test, y_pred)

        question=classifierquestion.render(section=1,modelname=self.modelname)
        intro=classifiersummary1.render(r2=self.accuracy,modelName=self.modelname)
        self.accuracy_label.config(text=question+f"\n"+ intro)

    def plot_importance_score_map(self):
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()

        sns.pairplot(self.df, hue=self.dv_column,height=2, aspect=3/2)
        # Save the pairplot to a file
        plt.savefig("./pic/pairplot.png")
        # Clear the current plot
        plt.clf()
        # Read the saved pairplot file
        img = plt.imread("./pic/pairplot.png")
        # Create a new Figure instance and plot the image on it
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.step5_frame)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        text=""
        # Update the most_important_x
        feature_importance = np.mean(np.abs(self.coefficients), axis=0)
        print(feature_importance)
        question=classifierquestion.render(section=3,ycol=self.dv_column)
        most_important_x = self.coeff_df.columns[np.argmax(np.abs(feature_importance))]
        answer=classifiersummary3.render(imp=most_important_x)
        for target_name, row in self.coeff_df.iterrows():
            most_important_feature = row.abs().idxmax()
            text=text+f"\nFor the {target_name},the most important feature is {most_important_feature}."
        text=text+'\n'+answer
        print(text)
        self.question_label=tk.Label(self.step5_frame, text=question)
        self.question_label.pack(pady=5)
        self.answer_label = tk.Label(self.step5_frame, text=text)
        self.answer_label.pack(pady=5)

    def on_close(self):
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":

    root = tk.Tk()
    root.geometry("800x600")
    app = ClassifierApp(root)
    root.mainloop()

