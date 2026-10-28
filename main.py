# Import Dependencies
import os

import yaml
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt
import gradio as gr


class DiseasePrediction:
    # Initialize and Load the Config File
    def __init__(self, model_name=None):
        # Load Config File
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")

        # Verbose
        self.verbose = self.config['verbose']
        # Load Training Data
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        # Load Test Data
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        # Feature Correlation in Training Data
        self._feature_correlation(data_frame=self.train_df, show_fig=False)
        # Model Definition
        self.model_name = model_name or 'random_forest'
        # Model Save Path
        self.model_save_path = self.config['model_save_path']
        os.makedirs(self.model_save_path, exist_ok=True)
        self.symptom_names = list(self.train_features.columns)

    # Function to Load Train Dataset
    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        df_train = df_train.drop(columns=['Unnamed: 133'], errors='ignore')
        feature_columns = [col for col in df_train.columns if col != 'prognosis']
        train_features = df_train[feature_columns]
        train_labels = df_train['prognosis']

        # Check for data sanity
        assert (len(train_features.iloc[0]) == 132)
        assert (len(train_labels) == train_features.shape[0])

        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
            print("Training Labels: ", train_labels.shape)
        return train_features, train_labels, df_train

    # Function to Load Test Dataset
    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        df_test = df_test.drop(columns=['Unnamed: 133'], errors='ignore')
        feature_columns = [col for col in df_test.columns if col != 'prognosis']
        test_features = df_test[feature_columns]
        test_labels = df_test['prognosis']

        # Check for data sanity
        assert (len(test_features.iloc[0]) == 132)
        assert (len(test_labels) == test_features.shape[0])

        if self.verbose:
            print("Length of Test Data: ", df_test.shape)
            print("Test Features: ", test_features.shape)
            print("Test Labels: ", test_labels.shape)
        return test_features, test_labels, df_test

    # Features Correlation
    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Get Feature Correlation
        corr = data_frame.drop(columns=['prognosis'], errors='ignore').select_dtypes(include=['number']).corr()
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig('feature_correlation.png')

    # Dataset Train Validation Split
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])

        if self.verbose:
            print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(len(X_train), len(y_train)))
            print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(len(X_val), len(y_val)))
        return X_train, y_train, X_val, y_val

    # Model Selection
    def select_model(self):
        if self.model_name == 'mnb':
            self.clf = MultinomialNB()
        elif self.model_name == 'decision_tree':
            self.clf = DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            self.clf = RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'gradient_boost':
            self.clf = GradientBoostingClassifier(n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                                                  criterion=self.config['model']['gradient_boost']['criterion'])
        return self.clf

    # ML Model
    def train_model(self):
        # Get the Data
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        # Training the Model
        classifier = classifier.fit(X_train, y_train)
        # Trained Model Evaluation on Validation Dataset
        confidence = classifier.score(X_val, y_val)
        # Validation Data Prediction
        y_pred = classifier.predict(X_val)
        # Model Validation Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Model Confusion Matrix
        conf_mat = confusion_matrix(y_val, y_pred)
        # Model Classification Report
        clf_report = classification_report(y_val, y_pred)
        # Model Cross Validation Score
        score = cross_val_score(classifier, X_val, y_val, cv=3)

        if self.verbose:
            print('\nTraining Accuracy: ', confidence)
            print('\nValidation Prediction: ', y_pred)
            print('\nValidation Accuracy: ', accuracy)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        # Save Trained Model
        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))

    def ensure_model_available(self):
        model_path = os.path.join(self.model_save_path, f"{self.model_name}.joblib")
        if os.path.exists(model_path):
            try:
                load(model_path)
                return
            except Exception:
                pass
        self.train_model()

    # Function to Make Predictions on Test Data
    def make_prediction(self, saved_model_name=None, test_data=None):
        model_path = os.path.join(self.model_save_path, f"{saved_model_name}.joblib")
        try:
            # Load Trained Model
            clf = load(model_path)
        except Exception as e:
            print("Model not found... Retraining model.")
            self.train_model()
            clf = load(model_path)

        if test_data is not None:
            result = clf.predict(test_data)
            return result
        else:
            result = clf.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, result)
        clf_report = classification_report(self.test_labels, result)
        return accuracy, clf_report

    def predict_from_symptoms(self, selected_symptoms):
        selected_symptoms = selected_symptoms or []
        if len(selected_symptoms) < 2:
            return "Please select at least 2 symptoms for a more reliable prediction."

        self.ensure_model_available()
        model_path = os.path.join(self.model_save_path, f"{self.model_name}.joblib")
        clf = load(model_path)

        feature_row = {feature: 1 if feature in selected_symptoms else 0 for feature in self.symptom_names}
        input_df = pd.DataFrame([feature_row], columns=self.symptom_names)
        prediction = clf.predict(input_df)[0]
        return str(prediction)

    def build_gradio_ui(self):
        symptom_choices = sorted(self.symptom_names)

        def predict(symptoms):
            return self.predict_from_symptoms(symptoms)

        with gr.Blocks(title="Disease Prediction") as demo:
            gr.Markdown("# Disease Prediction from Symptoms")
            gr.Markdown("### Select at least 2 related symptoms and click Predict Disease.")
            with gr.Row():
                with gr.Column(scale=2):
                    symptom_input = gr.CheckboxGroup(
                        choices=symptom_choices,
                        label="Symptoms",
                        info="Select all symptoms that apply to the patient. Predictions update only when you click the button.",
                    )
                with gr.Column(scale=1):
                    output = gr.Textbox(label="Predicted Disease", lines=3, placeholder="Your prediction will appear here.")
                    predict_btn = gr.Button("Predict Disease", variant="primary")
                    clear_btn = gr.ClearButton([symptom_input, output], value="Clear")
            predict_btn.click(fn=predict, inputs=symptom_input, outputs=output)

        return demo


if __name__ == "__main__":
    current_model_name = 'random_forest'
    dp = DiseasePrediction(model_name=current_model_name)
    dp.ensure_model_available()
    app = dp.build_gradio_ui()
    app.launch(server_name='0.0.0.0', server_port=7860, share=False, theme=gr.themes.Soft())
