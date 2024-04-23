import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import joblib

st.title('Flight Price Predictions')

df = pd.read_csv("Clean_Dataset.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)

st.subheader("Data")
st.dataframe(df.sample(7), use_container_width=True)

st.subheader("Visualizations")
st.markdown("___")


def plot_bar(df):
    fig1, _ = plt.subplots(figsize=(5, 5))
    df.plot.bar()
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xticks(rotation=45)
    buf = BytesIO()
    fig1.savefig(buf, format="png")
    st.image(buf)


def plot_line(df):
    fig1, _ = plt.subplots(figsize=(5, 5))
    df.plot()
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xticks(rotation=45)
    buf = BytesIO()
    fig1.savefig(buf, format="png")
    st.image(buf)


st.write('Prices of Destination cities mapped to corr mean prices')
plot_bar(df.groupby("destination_city")['price'].mean())

st.markdown("___")
st.write('Prices of Source cities mapped to corr mean prices')
plot_bar(df.groupby("source_city")['price'].mean())

st.markdown("___")
st.write('Prices of Airlines mapped to corr mean prices')
plot_bar(df.groupby("airline")['price'].mean())

st.markdown("___")
st.write('Number of Flights of each Airline')
plot_bar(df.airline.value_counts())

st.markdown("___")
st.write('Effect of Stops on Price')
plot_bar(df.groupby("stops")['price'].mean())

st.markdown("___")
st.write('Effect of Days left on Price')
plot_line(df.groupby("days_left")['price'].mean())

st.markdown("___")
st.write('Effect of Departure Time on Price')
plot_line(df.groupby("departure_time")['price'].mean())

st.markdown("___")
st.write('Effect of Departure Time on Price')
plot_line(df.groupby("arrival_time")['price'].mean())

st.markdown("___")
st.write('Effect of Class on Price of diff. Airlines')
fig1, _ = plt.subplots(figsize=(5, 5))
sns.barplot(x="class", y="price", data=df, estimator=np.median, hue="airline")
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xticks(rotation=45)
buf = BytesIO()
fig1.savefig(buf, format="png")
st.image(buf)

df_main = pd.read_csv(r"final_ds.csv", index_col=False)
df_main.drop('Unnamed: 0', axis=1, inplace=True)

st.markdown("___")
st.write('Effect of Flight Number on Prices')
fig1, _ = plt.subplots(figsize=(5, 5))
sns.scatterplot(x="flight_number", y="price", data=df_main)
plt.xticks(range(0, 10001, 1000))
plt.yticks(range(0, 120001, 10000))
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xticks(rotation=45)
buf = BytesIO()
fig1.savefig(buf, format="png")
st.image(buf)

st.markdown("___")
st.subheader("Final Data After Feature Encoding")
st.dataframe(df_main.sample(7), use_container_width=True)

st.markdown("___")
st.subheader("Co-relation Heatmap")
fig = plt.figure(figsize=(15, 12))
sns.heatmap(df_main.corr(), annot=True)
st.pyplot(fig)

st.markdown("___")
st.header("Testing")

t1 = st.text_input('Airline')
t2 = st.text_input('Source City')
t3 = st.text_input('Departure Time')
t4 = st.text_input('Stops')
t5 = st.text_input('Arrival Time')
t6 = st.text_input('Destination City')
t7 = st.text_input('Class')
t8 = st.text_input('Duration')
t9 = st.text_input('Days Left')
t10 = st.text_input('Flight Number')

pipeline = joblib.load(r'model_main.pkl')
label_encoders = np.load("label_encoders.npy", allow_pickle=True)

if st.button('Predict'):
    yp = pd.DataFrame({"airline": [t1], "source_city": [t2], "departure_time": [t3], "stops": [t4],
                       "arrival_time": [t5], "destination_city": [t6], "class": [t7], "duration": [t8],
                       "days_left": [t9], "flight_number": [t10]})

    # Convert 'stops' values to integers
    yp['stops'] = yp['stops'].replace({'one': 1, 'zero': 0, 'two_or_more': 2})

    # Transform categorical features using label encoders
    categorical_cols = ['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class']
    for column, label_encoder in zip(categorical_cols, label_encoders):
        yp[column] = label_encoder.transform(yp[column])

    # Debugging: Print input data before prediction
    st.write("Input data for prediction:")
    st.write(yp)

    # Make prediction
    try:
        prediction = pipeline.predict(yp)
        st.write("The predicted price is ", int(prediction[0]))
    except Exception as e:
        st.error(f"Prediction error: {e}")
