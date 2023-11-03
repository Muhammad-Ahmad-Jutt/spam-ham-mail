import time
import tkinter as tk
from tkinter import Radiobutton, IntVar, messagebox
import tkinter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
def show_loading():
    loading_window = tk.Toplevel(root)
    loading_window.title("Loading")
    load_l = tk.Label(loading_window, text="Please Wait while the algoritm do its work")
    load_l.pack()
    root.update()
    return loading_window
def check_email_file():
    email = entry.get()
    value = selected_algorithm_var.get()
    print(email, value)
    if email == '':
        messagebox.showwarning(Warning, "Please Enter mail")
    elif value == -1:
        messagebox.showwarning(Warning, "Please select an Algorithm to perform")
    else:
        loading_window = show_loading()
        mail_data = pd.read_csv("mail_data.csv")
        mail_data = mail_data.drop("Unnamed: 0", axis=1)
        mail_data = mail_data.drop("label_num", axis=1)
        mail_data['label'] = mail_data['label'].map({'ham': 0, 'spam': 1})
        mail_data = mail_data.rename(columns={'text': 'mail', 'label': 'catalog'})
        y = mail_data['catalog']
        x = mail_data['mail']
        print(x.shape, y.shape)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
        tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
        x_train_features = tfidf_vectorizer.fit_transform(x_train)
        x_test_features = tfidf_vectorizer.transform(x_test)
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')
        b_email = tfidf_vectorizer.transform([email])

        start_time = time.time()
        
        print(start_time)
        match value:
            case 0:
                clf = RandomForestClassifier()
                clf.fit(x_train_features, y_train)
                prediction_on_training = clf.predict(x_train_features)
                accuracy_on_train_data = accuracy_score(y_train, prediction_on_training)
                print(accuracy_on_train_data)
                prediction_on_test = clf.predict(x_test_features)
                accuracy_on_test_data = accuracy_score(y_test, prediction_on_test)
                print(accuracy_on_test_data)
                clf.score(x_train_features, y_train)
                clf.score(x_test_features, y_test)
                answer = clf.predict(b_email)
                
            case 1:
                model = SVC()
                model.fit(x_train_features, y_train)
                prediction_on_training = model.predict(x_train_features)
                accuracy_on_train_data = accuracy_score(y_train, prediction_on_training)
                prediction_on_test = model.predict(x_test_features)
                accuracy_on_test_data = accuracy_score(y_test, prediction_on_test)
                answer = model.predict(b_email)

            case 2:
                model = DecisionTreeClassifier()
                model.fit(x_train_features, y_train)
                prediction_on_training = model.predict(x_train_features)
                accuracy_on_train_data = accuracy_score(y_train, prediction_on_training)
                prediction_on_test = model.predict(x_test_features)
                accuracy_on_test_data = accuracy_score(y_test, prediction_on_test)
                answer = model.predict(b_email)
            case 3:
                model = MultinomialNB()
                model.fit(x_train_features, y_train)
                prediction_on_training = model.predict(x_train_features)
                accuracy_on_train_data = accuracy_score(y_train, prediction_on_training)
                prediction_on_test = model.predict(x_test_features)
                accuracy_on_test_data = accuracy_score(y_test, prediction_on_test)
                answer = model.predict(b_email)
            case 4:
                model = GaussianNB()
                model.fit(x_train_features.toarray(), y_train)
                prediction_on_training = model.predict(x_train_features.toarray())
                accuracy_on_train_data = accuracy_score(y_train, prediction_on_training)
                prediction_on_test = model.predict(x_test_features.toarray())
                accuracy_on_test_data = accuracy_score(y_test, prediction_on_test)
                answer = model.predict(b_email)
 
        loading_window.destroy()
        elapsed_time = time.time() - start_time
        if answer == 0:
            messagebox.showinfo("info_message", "The mail is not spam")
        elif answer == 1:
            messagebox.showinfo("info_message", "This mail is spam")
        info_message = f"Time taken for Algorithm {elapsed_time} seconds\n"
        info_message += f"Accuracy on Training Data: {accuracy_on_train_data}\n"
        info_message += f"Accuracy on Test Data: {accuracy_on_test_data}"
        messagebox.showinfo("Results", info_message)



algorithms = {
    "RandomForestClassifier()": 0,
    "SVM()": 1,
    "DecisionTreeClassifier()": 2,
    "MultinomialNB()": 3,
    "GaussianNB()": 4
}

root = tkinter.Tk()
root.title("Email Checker")

# Create and configure the widgets
label_email = tk.Label(root, text="Enter your email to check")
entry = tk.Entry(root, width=40)
label_algorithm = tk.Label(root, text='Choose the machine learning algorithm(s)')
submit_button = tk.Button(root, text="Submit", command=check_email_file)
exit_btn = tk.Button(root, text="Exit", bg='red', command= lambda : root.destroy())
# Create Tkinter IntVars to store the selected algorithms
selected_algorithm_var = IntVar(value=-1)

# Create checkboxes for each algorithm
for algorithm, value in algorithms.items():
    radio = Radiobutton(root, text=algorithm, variable=selected_algorithm_var, value=value )
    radio.grid(row=3, column=value)

label_email.grid(row=0, column=0, columnspan=5, pady=10)
entry.grid(row=1, column=0, columnspan=5, padx=10, pady=10)
label_algorithm.grid(row=2, column=0, columnspan=5, pady=10)
submit_button.grid(row=4, column=0, columnspan=5, pady=10)
exit_btn.grid(row=4, column=1, columnspan=5, pady= 10)
root.mainloop()
