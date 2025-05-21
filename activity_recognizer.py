import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

current_directory = os.path.dirname(__file__)
images_directory = os.path.join(current_directory, "img")

samples_size_per_exercise = 50 #should match about how long 1 execution of a exercise is
exercise_name_list = ["jumpingjack", "running", "lifting", "rowing"]
seconds_required_for_exercise = 10


class ExerciseClassifier:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.model = RandomForestClassifier(n_estimators=12, random_state=1567892)

    def train(self, data_frame):
        labels = self.label_encoder.fit_transform(data_frame["label"])
        feature_matrix = data_frame.drop(columns=["label"])
        scaled_features = self.scaler.fit_transform(feature_matrix)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, labels, test_size=0.2, random_state=1234542
        )
        self.model.fit(X_train, y_train)
        accuracy = (self.model.predict(X_test) == y_test).mean().round(3)
        print("Model trained. Accuracy:", accuracy)

    def predict(self, sensor_window):
        feature_row = extract_features(sensor_window)
        vector = pd.DataFrame([feature_row])
        scaled_vector = self.scaler.transform(vector)
        probabilities = self.model.predict_proba(scaled_vector)[0]
        best_index = probabilities.argmax()
        label = self.label_encoder.inverse_transform([best_index])[0]
        probability = probabilities[best_index]
        return label, probability


def extract_features(sensor_window):
    feature_dict = {}
    for axis in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        values = sensor_window[axis].values
        feature_dict[f"{axis}_mean"] = values.mean()
        feature_dict[f"{axis}_std"] = values.std()
        fft_values = np.fft.rfft(values)
        magnitudes = np.abs(fft_values)
        feature_dict[f"{axis}_energy"] = magnitudes[1:].mean()
    return feature_dict


def build_dataset(data_directory):
    rows = []
    for csv_path in glob.glob(os.path.join(data_directory, "*.csv")):
        data_frame = pd.read_csv(csv_path).dropna()
        parts = os.path.basename(csv_path).split("-")
        label = parts[1] if len(parts) > 1 else "unknown"
        for start in range(0, len(data_frame) - samples_size_per_exercise + 1, samples_size_per_exercise):
            window = data_frame.iloc[start : start + samples_size_per_exercise]
            feature_row = extract_features(window)
            feature_row["label"] = label
            rows.append(feature_row)
    return pd.DataFrame(rows)
