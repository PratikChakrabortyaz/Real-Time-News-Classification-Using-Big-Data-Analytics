# Real-Time News Classification Using Big Data Analytics

## Project Overview
This project implements a real-time news classification system using big data analytics. By leveraging tools such as **Apache Spark**, **Confluent Kafka**, and **NewsAPI**, the system can stream and classify news articles into categories such as entertainment, politics, business, sports, climate, and science. This project combines traditional machine learning models and advanced NLP techniques, including **TF-IDF** and **BERT embeddings**, with PySpark to handle large data volumes efficiently.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Setup and Installation on Google Colab](#setup-and-installation-on-google-colab)
- [Usage](#usage)
- [License](#license)

## Technologies Used
The project incorporates a range of libraries and tools for data processing, streaming, machine learning, and deep learning:
- **PySpark**: Distributed data processing with Apache Spark.
- **Confluent Kafka**: High-throughput streaming data platform for real-time analytics.
- **NewsAPI**: Fetches real-time news articles from various sources.
- **scikit-learn**: For traditional machine learning algorithms.
- **pandas** and **numpy**: For data manipulation and numerical operations.
- **matplotlib**: For data visualization.
- **PyTorch**: Deep learning framework.
- **transformers**: For pre-trained NLP models like BERT.
- **spaCy** and **NLTK**: For text processing tasks.

## Methodology
This project is divided into several stages:

### 1. Initial Setup and Installations
- **PySpark and Java Setup**: Installs OpenJDK and sets up environment variables for Apache Spark. `FindSpark` is used to easily locate and initialize the Spark package in Colab.
- **Library Installations**: Installs additional libraries like `pyspark`, `confluent_kafka`, and `newsapi-python`.

### 2. Streaming Data with Kafka and NewsAPI
- **NewsAPI Client Setup**: Initializes the NewsAPI client to fetch real-time news articles, using functions to extract and format relevant fields (e.g., title, description, content).
- **Kafka Producer Configuration**: Configures a Confluent Kafka producer to send fetched news articles to a Kafka topic.
- **Periodic Data Ingestion**: Sets up an infinite loop to fetch and send news articles to Kafka at regular intervals, enabling continuous real-time data ingestion.

### 3. Kafka Consumer and Article Classification
- **Kafka Consumer Setup**: Configures a consumer to read messages from the Kafka topic, decoding and storing them for classification.
- **Zero-Shot Classification with Transformers**: Uses BERT-based zero-shot classification to categorize articles into predefined labels (entertainment, politics, business, sports, climate, science).
- **Data Storage**: Classified data is saved to a CSV file for further analysis and visualization.

### 4. Text Processing Pipeline
The pipeline includes:
1. **Text Cleaning**: Removes unwanted characters, standardizes text, and reduces noise.
2. **Tokenization**: Breaks down text into tokens using NLTK.
3. **Stopword Removal**: Filters out non-informative words.
4. **Lemmatization and Stemming**: Reduces words to their base forms.
5. **Vectorization**: Converts text to numerical form using TF-IDF for ML models and BERT embeddings for deep learning.

### 5. Machine Learning Pipeline with PySpark
The final stage involves building and training machine learning models using both TF-IDF and BERT embeddings in PySpark:
- **Model Initialization**: Sets up Spark ML pipeline models including Naive Bayes, Random Forest, SVM, and Logistic Regression.
- **Train-Test Split**: Splits data for training and testing.
- **Model Evaluation**: Evaluates each model’s performance on real-time news data.

## Setup and Installation on Google Colab
To run this project on Google Colab, follow these steps:

1. **Open the Notebook**:
   - Upload the `real_time_news_classification.ipynb` notebook to Google Colab.

2. **Set Up the Dataset**:
   - If using a NewsAPI key, provide it directly in the notebook.
   - If using Kafka, configure connection details and topic names in the notebook.

3. **Install Dependencies**:
   - Run the following code to install all dependencies listed in `requirements.txt`:
     ```python
     !pip install -r requirements.txt
     ```

4. **Run Each Section in the Notebook**:
   - **NewsAPI Client Setup**: Initialize the client and fetch real-time news articles.
   - **Kafka Producer and Consumer**: Stream data to and from Kafka topics.
   - **Text Processing Pipeline**: Process text with tokenization, stopword removal, and lemmatization.
   - **Vectorization and Classification**: Use TF-IDF and BERT embeddings with machine learning models in PySpark.

5. **View Results**:
   - The notebook will display the classified news articles along with model performance metrics.

## Usage
- **Streaming Real-Time News**: Streams news articles in real-time, classifying them based on predefined labels.
- **Scalable ML Pipelines**: Uses PySpark to process large datasets and train models efficiently, making it ideal for big data applications.
- **Model Comparison**: Compares traditional ML and deep learning approaches to classify articles with TF-IDF and BERT embeddings.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
