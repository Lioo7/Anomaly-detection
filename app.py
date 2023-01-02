import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st

# file path - this for linux windows you will need "//"
f_path = "/app/conn_attack.csv"
df = pd.read_csv(f_path, names=[
                 "record ID", "duration_", "src_bytes", "dst_bytes"], header=None)

# caluculates and stores the median and the std of each feature
median_duration = df['duration_'].median()
std_duration = df['duration_'].std()
median_src_b = df['src_bytes'].median()
std_src_b = df['src_bytes'].std()
median_dst_b = df['dst_bytes'].mean()
std_dst_b = df['dst_bytes'].std()
total_smaples = df.count()

# calculates the suspected share of outliers in the dataset
num_of_suspected_outliers = df[df['src_bytes']
                               > (median_src_b + std_dst_b)].count()
estimte_contamination = num_of_suspected_outliers / total_smaples
estimte_contamination = estimte_contamination[0]
print("estimte_contamination:", estimte_contamination)
n_estimators = 50
data = df[["duration_", "src_bytes", "dst_bytes"]]

# trains the model
model = IsolationForest(
    contamination=estimte_contamination, n_estimators=n_estimators)
model.fit(data)

# predicts
df["is_anomaly?_"] = pd.Series(model.predict(data))
# maps the predictions from 1->0 and from -1->1
df["is_anomaly?_"] = df["is_anomaly?_"].map({1: 0, -1: 1})

# # counts the number of anomalies that we got
# anomaly = df["is_anomaly?_"].value_counts()[1]
# st.text(f'anomaly: {anomaly}')

# loads the anomaly_labels file
f_path = "/app/conn_attack_anomaly_labels.csv"
df_labels = pd.read_csv(f_path, names=['record ID', 'anomaly'], header=None)
y_test, y_pred = df_labels['anomaly'], df["is_anomaly?_"]

# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# st.text(f'TN: {tn}')
# st.text(f'FP: {fp}')
# st.text(f'FN: {fn}')
# st.text(f'TP: {tp}')

# Display model accuracy
accuracy = accuracy_score(df_labels['anomaly'], df['is_anomaly?_'])
st.text(f'Model Accuracy: {accuracy}')

# Get input from user
record_ID = st.number_input("record ID", value=0)
duration_ = st.number_input("Duration", value=0)
src_bytes = st.number_input("Source bytes", value=0)
dst_bytes = st.number_input("Destination bytes", value=0)

if st.button("Predict"):
    # Predict result
    sample_data = [duration_, src_bytes, dst_bytes]
    columns = ['duration_', 'src_bytes', 'dst_bytes']
    df_data = pd.DataFrame([sample_data], columns=columns)

    prediction = model.predict(df_data)[0]
    prediction_label = None
    if prediction == -1:
        prediction_label = 'Anomaly'
    else:
        prediction_label = 'Benign'

    # Display prediction
    if prediction_label == 'Anomaly':
        st.markdown(
            f'Prediction Result: <font color="red"> {prediction_label}</font>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'Prediction Result: <font color="green"> {prediction_label}</font>', unsafe_allow_html=True)
