import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import numpy as np
from datetime import datetime

# dessa klasser är skrivna för att användas i en notebook, därav print statements och sådant, hårdkodade grejer, 
# allt är byggt från början i en notebook sedan konverterad till en klass i denna fil. för att sedan kunna importeras i en annan notebook.
# generellt, imo, ful kod men slipper iallafall kod i rapporten.
# desto längre ner man kommer i dokumentet desto stökigare..

# eda
class EDA:
    def __init__(self, df):
        self.df = df
        
    def count_cardio_cases(self):
        positive = self.df[self.df["cardio"] == 1]
        negative = self.df[self.df["cardio"] == 0]
        print("Positive: ", len(positive))
        print("Negative: ", len(negative))
    
    def plot_cholesterol_distribution(self):
        normal = self.df[self.df["cholesterol"] == 1]
        above_normal = self.df[self.df["cholesterol"] == 2]
        well_above_normal = self.df[self.df["cholesterol"] == 3]
    
        cholesterol_data = [len(normal), len(above_normal), len(well_above_normal)]
        cholesterol_labels = ["Normal", "Över normal", "Långt över normal"]
    
        plt.pie(cholesterol_data, labels=cholesterol_labels, autopct="%.1f%%")
        plt.title("Cholesterol Levels")
        plt.show()
    
    def plot_age_distribution(self):
        ages = self.df["age"].values/365.25
        sns.histplot(ages, bins=20, legend=False)
        plt.title("Åldersfördelning")
        plt.xlabel("Ålder")
        plt.ylabel("Count")
        plt.show()
    
    def plot_smoking_distribution(self):
        print("Andel rökare", len(self.df[self.df["smoke"] == 1]))
        print("Andel icke rökare", len(self.df[self.df["smoke"] == 0]))
        labels = ["icke rökare", "rökare"]
    
        self.df["smoke"].value_counts().plot(kind="pie", labels=labels, autopct="%.1f%%", ylabel="", legend=True)
        plt.show()
    
    def plot_weight_distribution(self):
        sns.histplot(self.df["weight"], bins=50)
        plt.xlabel("Weight (KG)")
        plt.show()
    
    def plot_height_distribution(self):
        sns.histplot(self.df["height"], bins=50)
        plt.xlabel("height (cm)")
        plt.xlim((100,210))
        plt.show()
    
    def plot_gender_distribution(self):
        positive = self.df[self.df["cardio"] == 1]
        women = len(positive[positive["gender"]== 1])
        men = len(positive[positive["gender"]== 2])
    
        data = [women, men]
        labels = ["Kvinnor", "Män"]
    
        plt.pie(data, labels=labels, autopct="%.1f%%")
        plt.title("Könsfördelning bland hjärt-kärlsjukdomar")
        plt.show()

# feature engineering
class FEng:
    def __init__(self, df):
        self.df = df
    
    def bmi_calc(self, weight, height):
        return (weight / ((height/100)**2))
    
    def bmi_class_assign(self, bmi):
        if bmi < 24.9:
            return "normal range"
        elif bmi < 29.9:
            return "over-weight"
        elif bmi < 34.9:
            return "obese (class I)"
        elif bmi < 39.9:
            return "obese (class II)"
        else:
            return "obese (class III)"
        
    def bp_class_assign(self, systolic, diastolic):
        if systolic >= 180 or diastolic >= 120:
            return "hypertension crisis"
        elif systolic >= 140 or diastolic >= 90:
            return "stage 2 hypertension"
        elif (130 <= systolic < 140) or (80 <= diastolic < 90):
            return "stage 1 hypertension"
        elif 120 <= systolic < 130 and diastolic < 80:
            return "elevated"
        elif systolic < 120 and diastolic < 80:
            return "healthy"
    
    def get_bmi_features(self):
        self.df["bmi"] = self.bmi_calc(self.df["weight"], self.df["height"])

        self.df = self.df[(self.df["bmi"] >= 15) & (self.df["bmi"] <= 60)].copy() # copy undviker pd varningar
        self.df["bmi class"] = self.df["bmi"].apply(self.bmi_class_assign)
        return self.df
    
    def get_bp_features(self):
        self.df = self.df[(self.df["ap_hi"] >= 60) & (self.df["ap_hi"] <= 200)].copy()
        self.df = self.df[(self.df["ap_lo"] >= 30) & (self.df["ap_lo"] <= 200)].copy()
        self.df["blood pressure class"] = self.df.apply(
            lambda row: self.bp_class_assign(row["ap_hi"], row["ap_lo"]), axis=1
        )
        return self.df
    
    def engineer_features(self): # applies both eng features and returns df
        self.get_bmi_features()
        self.get_bp_features()
        return self.df

# viz
class Viz:
    def __init__(self, df):
        self.df = df
    
    def plot_positive_vs_negative_bp(self): # plots positive vs negative blood pressure classes
        positive = self.df[self.df["cardio"] == 1] 
        negative = self.df[self.df["cardio"] == 0]
        
        positive_bp = positive["blood pressure class"].value_counts()
        negative_bp = negative["blood pressure class"].value_counts()
        
        plt.figure(figsize=(12, 6))
        bp_classes = [
            "healthy",
            "elevated",
            "stage 1 hypertension",
            "stage 2 hypertension",
            "hypertension crisis"
        ]

        x = range(len(bp_classes))
        width = 0.40
        pos_values = [positive_bp.get(cls, 0) for cls in bp_classes]
        neg_values = [negative_bp.get(cls, 0) for cls in bp_classes]
        plt.bar([i - width/2 for i in x], pos_values, width, label='Heart Disease Positive', color='#1f77b4')
        plt.bar([i + width/2 for i in x], neg_values, width, label='Heart Disease Negative', color='#ff7f0e')

        plt.xlabel('Blood Pressure Class')
        plt.ylabel('Count')
        plt.title('Blood pressure class Distribution: Positive vs Negative for heart Disease')
        plt.xticks(x, bp_classes)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_positive_vs_negative_bmi(self): # plot positive vs negative bmi
        positive = self.df[self.df["cardio"] == 1] 
        negative = self.df[self.df["cardio"] == 0]

        positive_bmi = positive["bmi class"].value_counts()
        negative_bmi = negative["bmi class"].value_counts()

        plt.figure(figsize=(12, 6))
        
        bmi_classes = [
            "normal range",
            "over-weight",
            "obese (class I)",
            "obese (class II)",
            "obese (class III)"
        ]
        
        x = range(len(bmi_classes))
        width = 0.40
        pos_values = [positive_bmi.get(cls, 0) for cls in bmi_classes]
        neg_values = [negative_bmi.get(cls, 0) for cls in bmi_classes]
        plt.bar([i - width/2 for i in x], pos_values, width, label='Heart Disease Positive', color='#1f77b4')
        plt.bar([i + width/2 for i in x], neg_values, width, label='Heart Disease Negative', color='#ff7f0e')
        
        plt.xlabel('BMI Class')
        plt.ylabel('Count')
        plt.title('BMI Distribution: Heart Disease Positive vs Negative')
        plt.xticks(x, bmi_classes)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def bp_data(self, df=None):
        if df is None:
            df = self.df
        healthy = df[df["blood pressure class"] == "healthy"]
        elevated = df[df["blood pressure class"] == "elevated"]
        stage1 = df[df["blood pressure class"] == "stage 1 hypertension"]
        stage2 = df[df["blood pressure class"] == "stage 2 hypertension"]
        crisis = df[df["blood pressure class"] == "hypertension crisis"]
        return [len(healthy), len(elevated), len(stage1), len(stage2), len(crisis)]

    def plot_bmi_vs_bp(self):
        colors = sns.color_palette("Set2", 5)  
        
        normal_range = self.df[self.df["bmi class"] == "normal range"]
        over_weight = self.df[self.df["bmi class"] == "over-weight"]
        obese1 = self.df[self.df["bmi class"] == "obese (class I)"]
        obese2 = self.df[self.df["bmi class"] == "obese (class II)"]
        obese3 = self.df[self.df["bmi class"] == "obese (class III)"]

        bp_labels = ["healthy", "elevated", "stage 1 hypertension", "stage 2 hypertension", "hypertension crisis"]

        fig, axs = plt.subplots(3, 2, figsize=(15, 15), dpi=120)
        fig.suptitle("BMI vs Blood pressure", fontsize=25)
        
        def plot(ax, df, title):
            ax.pie(self.bp_data(df), labels=bp_labels, autopct="%.1f%%", colors=colors)
            ax.set_title(title)
            ax.set()

        plot(axs[0, 1], normal_range, "BMI: Normal Range")
        plot(axs[1, 0], over_weight, "BMI: Over Weight")
        plot(axs[1, 1], obese1, "BMI: Obese (Class I)")
        plot(axs[2, 0], obese2, "BMI: Obese (Class II)")
        plot(axs[2, 1], obese3, "BMI: Obese (Class III)")

        patches = [mpatches.Patch(color=colors[i], label=bp_labels[i]) for i in range(len(bp_labels))]
        axs[0, 0].legend(handles=patches, loc='center', fontsize='x-large',
                        title='Blood Pressure Classes', title_fontsize='xx-large',
                        bbox_to_anchor=(0, 0, 1, 1), bbox_transform=axs[0, 0].transAxes)
        axs[0, 0].axis('off')
        plt.show()

    def subplot_negative_vs_positive(self):
        positive = self.df[self.df["cardio"] == 1] 
        negative = self.df[self.df["cardio"] == 0]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=120)
        fig.suptitle('Heart Disease Positive vs Negative')
        axes = axes.flatten()
        colors = sns.color_palette("Set2", 5)

        for ax in axes:
            for bar, color in zip(ax.patches, colors):
                bar.set_color(color)

        def add_p_labels(ax): # adds percentage labels to the top of the bars
            total = sum([p.get_height() for p in ax.patches])
            for p in ax.patches:
                height = p.get_height()
                percentage = (height / total) * 100
                ax.text(p.get_x() + p.get_width() / 2., height + 0.01, f'{percentage:.1f}%', ha='center', va='bottom')

        def cholesterol_data():
            cholesterol_positive = positive["cholesterol"].value_counts().sort_index()
            cholesterol_negative = negative["cholesterol"].value_counts().sort_index()
            cholesterol_data = pd.DataFrame({"Positive": cholesterol_positive, "Negative": cholesterol_negative})
            return cholesterol_data

        cholesterol_data_df = cholesterol_data()
        cholesterol_data_df.plot(kind='bar', ax=axes[0])
        axes[0].set_xticklabels(["normal", "above normal","well above normal"], rotation=0)
        axes[0].set_title("Cholesterol")
        axes[0].set_xlabel("Cholesterol Class")
        axes[0].set_ylabel("Count")
        axes[0].set_ylim(0,cholesterol_data_df.values.max()*1.1)
        add_p_labels(ax = axes[0])

        def glucose_data():
            glucose_positive = positive["gluc"].value_counts().sort_index()
            glucose_negative = negative["gluc"].value_counts().sort_index()
            glucose_data = pd.DataFrame({"Positive": glucose_positive, "Negative": glucose_negative})
            return glucose_data

        glucose_data_df = glucose_data()
        glucose_data_df.plot(kind='bar', ax=axes[1])
        axes[1].set_xticklabels(["normal", "above normal","well above normal"], rotation=0)
        axes[1].set_title("Glucose")
        axes[1].set_xlabel("Glucose Class")
        axes[1].set_ylabel("Count")
        axes[1].set_ylim(0,glucose_data_df.values.max()*1.1)
        add_p_labels(ax = axes[1])

        def smoking_data():
            smoking_positive = positive["smoke"].value_counts().sort_index()
            smoking_negative = negative["smoke"].value_counts().sort_index()
            smoking_data = pd.DataFrame({"Positive": smoking_positive, "Negative": smoking_negative})
            return smoking_data

        smoking_data_df = smoking_data()
        smoking_data_df.plot(kind='bar', ax=axes[2])
        axes[2].set_xticklabels(["Non Smoker", "Smoker"], rotation=0)
        axes[2].set_title("cigarette use")
        axes[2].set_xlabel("Smoking Class")
        axes[2].set_ylabel("Count")
        axes[2].set_ylim(0,smoking_data_df.values.max()*1.1)
        add_p_labels(ax = axes[2])

        def alcohol_data():
            alcohol_positive = positive["alco"].value_counts().sort_index()
            alcohol_negative = negative["alco"].value_counts().sort_index()
            alchohol_data = pd.DataFrame({"Positive": alcohol_positive, "Negative": alcohol_negative})
            return alchohol_data

        alcohol_data_df = alcohol_data()
        alcohol_data_df.plot(kind="bar", ax=axes[3])
        axes[3].set_xticklabels(["Non Drinker", "Drinker"], rotation=0)
        axes[3].set_title("Alcohol use")
        axes[3].set_xlabel("Alcohol Class")
        axes[3].set_ylabel("Count")
        axes[3].set_ylim(0,alcohol_data_df.values.max()*1.1)
        add_p_labels(ax = axes[3])

        def physical_activity_data():
            physical_positive = positive["active"].value_counts().sort_index()
            physical_negative = negative["active"].value_counts().sort_index()
            physical_activity_data = pd.DataFrame({"Positive": physical_positive, "Negative": physical_negative})
            return physical_activity_data

        physical_activity_df = physical_activity_data()
        physical_activity_df.plot(kind="bar", ax=axes[4])
        axes[4].set_xticklabels(["No", "Yes"], rotation=0)
        axes[4].set_title("Physical Activity")
        axes[4].set_xlabel("Physical Activity")
        axes[4].set_ylabel("Count")
        axes[4].set_ylim(0,physical_activity_df.values.max()*1.1)
        add_p_labels(ax = axes[4])

        def age_data():
            positive_age = positive["age"]/365.25
            negative_age = negative["age"]/365.25
            age_data = pd.DataFrame({"Positive": positive_age, "Negative": negative_age})
            return age_data

        age_data_df = age_data()
        age_data_df.plot(kind="hist", ax=axes[5], alpha=0.5, bins=30)
        axes[5].set_title("Age Distribution")
        axes[5].set_xlabel("Age")
        axes[5].set_ylabel("Frequency")
        axes[5].set_xlim(35, 70)
        axes[5].legend(["Positive", "Negative"])

        plt.tight_layout()
        plt.show()

    def plot_heatmap(self):
        bmi_classes_to_numeric = {"normal range": 0, "over-weight": 1, "obese (class I)": 2, "obese (class II)": 3, "obese (class III)": 4}
        self.df["bmi class"] = self.df["bmi class"].map(bmi_classes_to_numeric)
        blood_pressure_classes_to_numeric = {"healthy": 0, "elevated": 1, "stage 1 hypertension": 2, "stage 2 hypertension": 3, "hypertension crisis": 4}
        self.df["blood pressure class"] = self.df["blood pressure class"].map(blood_pressure_classes_to_numeric)
        corr = self.df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", square=True, linewidths=0.5)
        plt.title('Correlation heatmap', fontsize=16)
        plt.tight_layout()
        plt.show()




class MachineLearning:
    def __init__(self, df):
        self.results = {}
        self.scaler = StandardScaler()
        self.normalisation = MinMaxScaler()
        self.df = df
        

    def read_and_split_data(self):
        df = self.df
        X, y = df.drop(columns=["cardio"]), df["cardio"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
        
        self.X_train, self.X_test, self.X_val = X_train, X_test, X_val
        self.y_train, self.y_test, self.y_val = y_train, y_test, y_val
        
        return (X_train, X_test, X_val, y_train, y_test, y_val)
    
    
    def scale_and_normalize_data(self):
        self.scaled_X_train = self.scaler.fit_transform(self.X_train)
        self.scaled_X_test = self.scaler.transform(self.X_test)
        self.scaled_X_val = self.scaler.transform(self.X_val)

        self.norm_X_train = self.normalisation.fit_transform(self.scaled_X_train)
        self.norm_X_test = self.normalisation.transform(self.scaled_X_test)
        self.norm_X_val = self.normalisation.transform(self.scaled_X_val)
        
        return (self.norm_X_train, self.norm_X_test, self.norm_X_val)
    
    
    def grid_search_cv(self, X_train, y_train, X_val, y_val, model, param_grid, model_name):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        return {
            "model": best_model,
            "params": best_params,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    
    def run_grid_search_for_all_models(self):
        rf_param_grid = {
            "n_estimators": [100, 150, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4, 8],
        }

        lr_param_grid = {
            "penalty": ["l1", "l2", "elasticnet"],
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["saga"],
        }

        knn_param_grid = {
            "n_neighbors": [11, 15, 20, 25, 30, 50],
            "weights": ["uniform", "distance"],
        }
        

        rf = RandomForestClassifier()
        lr = LogisticRegression()
        knn = KNeighborsClassifier()
        

        self.results["Random forest"] = self.grid_search_cv(
            self.norm_X_train,
            self.y_train,
            self.norm_X_val,
            self.y_val,
            rf,
            rf_param_grid,
            "Random Forest"
        )
        
        self.results["Logistic regression"] = self.grid_search_cv(
            self.norm_X_train,
            self.y_train,
            self.norm_X_val,
            self.y_val,
            lr,
            lr_param_grid,
            "Logistic Regression"
        )

        self.results["KNN"] = self.grid_search_cv(
            self.norm_X_train, 
            self.y_train, 
            self.norm_X_val, 
            self.y_val, 
            knn, 
            knn_param_grid, 
            "KNN"
        )

        results_df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f"CV_results_{timestamp}.csv")
        
        return self.results, results_df
    
    def create_voting_classifier(self):
        X_train, y_train = self.norm_X_train, self.y_train
        X_test, y_test = self.norm_X_test, self.y_test
        
        vote_clf = VotingClassifier(
            estimators=[
                ("rf", self.results["Random forest"]["model"]),
                ("lr", self.results["Logistic regression"]["model"]),
                ("knn", self.results["KNN"]["model"])
            ],
            voting="hard"
        )


        vote_clf.fit(X_train, y_train)
        y_pred = vote_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return vote_clf, accuracy
    
    
    def evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        X_train_full = np.concatenate((X_train, X_val), axis=0)
        y_train_full = np.concatenate((y_train, y_val), axis=0)
        
        model.fit(X_train_full, y_train_full)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        
    def run_full_pipeline(self):
        self.read_and_split_data()
        self.scale_and_normalize_data()
        self.run_grid_search_for_all_models()
        vote_clf, vote_accuracy = self.create_voting_classifier()
        
        print("Evaluating best Random Forest model on test set:")
        evaluation_results = self.evaluate_model(
            self.results["Random forest"]["model"], 
            self.norm_X_train,
            self.norm_X_val, 
            self.norm_X_test, 
            self.y_train,
            self.y_val, 
            self.y_test
        )
        
        return self.results, vote_clf, evaluation_results


        # börjar få ont om tid här..
    def classification_report(self): # snuskig snabb lösning för notebook..
        model = self.results["Random forest"]["model"]
        X_train = self.norm_X_train
        X_val = self.norm_X_val
        X_test = self.norm_X_test
        y_train = self.y_train
        y_val = self.y_val
        y_test = self.y_test
        
        X_train_full = np.concatenate((X_train, X_val), axis=0)
        y_train_full = np.concatenate((y_train, y_val), axis=0)
        
        model.fit(X_train_full, y_train_full)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))
        
        
    def confusion_matrix(self):         # och här..
        model = self.results["Random forest"]["model"]
        X_train = self.norm_X_train
        X_val = self.norm_X_val
        X_test = self.norm_X_test
        y_train = self.y_train
        y_val = self.y_val
        y_test = self.y_test
        
        X_train_full = np.concatenate((X_train, X_val), axis=0)
        y_train_full = np.concatenate((y_train, y_val), axis=0)
        
        model.fit(X_train_full, y_train_full)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()
    
    
    
# fixat denna print lite med GPT så att den inte går sönder utan ha kört modeller först.
    def __str__(self): 
        if not hasattr(self, 'results') or not self.results:
            return "No models have been trained yet."
        
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        model_name, model_info = best_model
        
        return f"Best model after Gridsearch Results:\n" \
               f"Best model: {model_name}\n" \
               f"Accuracy: {model_info['accuracy']:.4f}\n" \
               f"Precision: {model_info['precision']:.4f}\n" \
               f"Recall: {model_info['recall']:.4f}\n" \
               f"F1 Score: {model_info['f1']:.4f}\n" \
               f"Best parameters: {model_info['params']}"

