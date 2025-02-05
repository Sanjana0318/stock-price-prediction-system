import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import io
import base64
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from werkzeug.utils import secure_filename

# Set Matplotlib backend to 'Agg' to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("StockPredictionApp").getOrCreate()

# Folder to save uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded files (allow multiple files for stock data)
    stock_files = request.files.getlist('stock_data')  # Use getlist() to get all files
    tweets_file = request.files['tweets_data']

    # Save tweets data to the uploads directory
    tweets_filename = secure_filename(tweets_file.filename)
    tweets_filepath = os.path.join(app.config['UPLOAD_FOLDER'], tweets_filename)
    tweets_file.save(tweets_filepath)

    # Load tweets data using Pandas
    tweets_data = pd.read_csv(tweets_filepath)

    # Convert 'Date' in tweets_data to datetime format and remove timezone info
    tweets_data['Date'] = pd.to_datetime(tweets_data['Date']).dt.tz_localize(None)

    # Initialize sentiment analyzer and calculate sentiment scores for tweets
    analyzer = SentimentIntensityAnalyzer()
    tweets_data['compound'] = tweets_data['Tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # Summarize sentiments
    sentiment_summary = {
        'positive': np.mean(tweets_data['compound'] > 0) * 100,
        'negative': np.mean(tweets_data['compound'] < 0) * 100,
        'neutral': np.mean(tweets_data['compound'] == 0) * 100,
    }

    # Store prediction results and graphs for each stock file
    graphs = []

    # Process each stock CSV file
    for stock_file in stock_files:
        stock_filename = secure_filename(stock_file.filename)
        stock_filepath = os.path.join(app.config['UPLOAD_FOLDER'], stock_filename)
        stock_file.save(stock_filepath)

        # Load stock data using PySpark
        stock_data_spark = spark.read.csv(stock_filepath, header=True, inferSchema=True)

        # Convert the 'Date' column in stock data to datetime format in Spark
        stock_data_spark = stock_data_spark.withColumn('Date', to_date(col('Date'), 'yyyy-MM-dd'))

        # Convert stock data from Spark DataFrame to Pandas DataFrame for further processing
        stock_data = stock_data_spark.toPandas()

        # Convert 'Date' columns in stock_data to datetime64[ns]
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d')

        # Merge stock data with sentiment data based on the 'Date' column
        daily_sentiment = tweets_data.groupby('Date')['compound'].mean().reset_index()
        stock_data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')

        # Fill NaN values for compound sentiment scores with 0
        stock_data['compound'] = stock_data['compound'].fillna(0)

        # Prepare features for prediction
        stock_data['Day'] = stock_data['Date'].dt.dayofyear
        X = stock_data[['Day', 'compound']]
        y = stock_data['Close']

        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        stock_data['Predicted'] = model.predict(X)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y, stock_data['Predicted']))

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Prices', color='blue')
        plt.plot(stock_data['Date'], stock_data['Predicted'], label='Predicted Prices', color='orange')
        plt.title(f'Stock Price Prediction for {stock_filename}')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.xticks(rotation=45)

        # Save plot to a PNG image in memory
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Append RMSE and plot for each stock file
        graphs.append({
            'stock_filename': stock_filename,
            'rmse': rmse,
            'plot_url': plot_url
        })

    # Render result.html with multiple graphs
    return render_template('result.html', sentiment_summary=sentiment_summary, graphs=graphs)

if __name__ == '__main__':
    app.run(debug=True)
