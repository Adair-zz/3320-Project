# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
stop=set(stopwords.words('english'))


# Method
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

# Clean method for our Training data
def clean(df):
    # Removing the (@user)
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['OriginalTweet'], '@[\w]*')

    # Removing Special Characters, Punctuation & Numbers
    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z]", " ")

    # Removing Short Words
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))

    # Stemming
    tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = " ".join(tokenized_tweet[i])
        df['clean_tweet'] = tokenized_tweet

    return df

# reference: https://keras.io/examples/nlp/active_learning_review_classification/
# Helper function for merging new history objects with older ones
def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["binary_accuracy"]
    val_accuracy = val_accuracy + history.history["val_binary_accuracy"]
    return losses, val_losses, accuracy, val_accuracy


# Plotter function
def plot_history(losses, val_losses, accuracies, val_accuracies):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig.savefig('Loss Graph.png', dpi=1000)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(["train_accuracy", "val_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    fig.savefig('Accuracy Graph.png', dpi=1000)


# build biLSTM model
def create_model():
    model = keras.models.Sequential(
        [
            layers.Input(shape=(150,)),
            layers.Embedding(input_dim=3000, output_dim=128),
            layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
            layers.GlobalMaxPool1D(),
            layers.Dense(20, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.summary()
    return model

def train_full_model(full_train_dataset, val_dataset, test_dataset):
    model = create_model()
    model.compile(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.FalseNegatives(),
            keras.metrics.FalsePositives(),
        ],
    )

    # We will save the best model at every epoch and load the best one for evaluation on the test set
    history = model.fit(
        full_train_dataset.batch(256),
        epochs=20,
        validation_data=val_dataset,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=4, verbose=1),
            keras.callbacks.ModelCheckpoint(
                "FullModelCheckpoint.h5", verbose=1, save_best_only=True
            ),
        ],
    )

    # Plot history
    plot_history(
        history.history["loss"],
        history.history["val_loss"],
        history.history["binary_accuracy"],
        history.history["val_binary_accuracy"],
    )

    # Loading the best checkpoint
    model = keras.models.load_model("FullModelCheckpoint.h5")

    print("-" * 100)
    print(
        "Test set evaluation: ",
        model.evaluate(test_dataset, verbose=0, return_dict=True),
    )
    print("-" * 100)
    return model

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

def vectorize_text(text, label):
    text = vectorizer(text)
    return text, label


# data cleaning For tweets
def clean1(df):
    for i in range(len(df)):
        df['OriginalTweet'][i] = df['OriginalTweet'][i].strip('\n')

        # Removing the (@user)
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['OriginalTweet'], '@[\w]*')

    # Removing Special Characters, Punctuation & Numbers
    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z]", " ")

    # Removing Short Words
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))

    # Stemming
    tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = " ".join(tokenized_tweet[i])
        df['clean_tweet'] = tokenized_tweet

    return df








if __name__ == "__main__":
    # Import Training dataset
    try:
        df = pd.read_csv('Coronavirus Tweets.csv', encoding = "ISO-8859-1")
    except:
        print('No this file!')

    """# Preprocessing"""

    df = df[['OriginalTweet', 'Sentiment']]
    df['Sentiment'].value_counts()
    df['Sentiment'] = df['Sentiment'].map({"Extremely Positive": 1, "Extremely Negative": 0, "Negative": 0, "Positive": 1, "Neutral": -1})
    df = df[[df['Sentiment'][i]!=-1 for i in range(len(df))]]
    df['Sentiment'].value_counts()
    df = df.reset_index(drop=True)

    """## Data Cleaning"""
    df = clean(df)



    """# Training Model"""

    df.columns = ['OriginalTweet', 'labels', 'reviews']
    reviews = np.array(df['reviews'][:30000])
    labels = np.array(df['labels'][:30000])

    val_split = 250
    test_split = 250
    train_split = 750

    # Separating the negative and positive samples for manual stratification
    x_positives, y_positives = reviews[labels == 1], labels[labels == 1]
    x_negatives, y_negatives = reviews[labels == 0], labels[labels == 0]

    # Creating training, validation and testing splits
    x_val, y_val = (
        tf.concat((x_positives[:val_split], x_negatives[:val_split]), 0),
        tf.concat((y_positives[:val_split], y_negatives[:val_split]), 0),
    )
    x_test, y_test = (
        tf.concat(
            (
                x_positives[val_split : val_split + test_split],
                x_negatives[val_split : val_split + test_split],
            ),
            0,
        ),
        tf.concat(
            (
                y_positives[val_split : val_split + test_split],
                y_negatives[val_split : val_split + test_split],
            ),
            0,
        ),
    )
    x_train, y_train = (
        tf.concat(
            (
                x_positives[val_split + test_split : val_split + test_split + train_split],
                x_negatives[val_split + test_split : val_split + test_split + train_split],
            ),
            0,
        ),
        tf.concat(
            (
                y_positives[val_split + test_split : val_split + test_split + train_split],
                y_negatives[val_split + test_split : val_split + test_split + train_split],
            ),
            0,
        ),
    )

    # Remaining pool of samples are stored separately. These are only labeled as and when required
    x_pool_positives, y_pool_positives = (
        x_positives[val_split + test_split + train_split :],
        y_positives[val_split + test_split + train_split :],
    )
    x_pool_negatives, y_pool_negatives = (
        x_negatives[val_split + test_split + train_split :],
        y_negatives[val_split + test_split + train_split :],
    )

    # Creating TF Datasets for faster prefetching and parallelization
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    pool_negatives = tf.data.Dataset.from_tensor_slices(
        (x_pool_negatives, y_pool_negatives)
    )
    pool_positives = tf.data.Dataset.from_tensor_slices(
        (x_pool_positives, y_pool_positives)
    )

    print(f"Initial training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")
    print(f"Unlabeled negative pool: {len(pool_negatives)}")
    print(f"Unlabeled positive pool: {len(pool_positives)}")



    vectorizer = layers.TextVectorization(
        3000, standardize=custom_standardization, output_sequence_length=150
    )
    # Adapting the dataset
    vectorizer.adapt(
        train_dataset.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE).batch(256)
    )



    train_dataset = train_dataset.map(
        vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    pool_negatives = pool_negatives.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
    pool_positives = pool_positives.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(256).map(
        vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.batch(256).map(
        vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
    )



    # Sampling the full train dataset to train on
    full_train_dataset = (
        train_dataset
        .cache()
        .shuffle(200)
    )

    # Training the full model
    full_dataset_model = train_full_model(full_train_dataset, val_dataset, test_dataset)





    '''# Omicron Dataset Part'''

    data = pd.read_csv('omicron.csv', index_col=0)
    data = data.dropna()
    data = data.reset_index(drop=True)
    data_new = pd.DataFrame()
    data_new['OriginalTweet'] = data['Text']
    data_new['Sentiment'] = 0

    data_new = clean1(data_new)

    data_new.head()

    data_new.columns = ['OriginalTweet', 'labels', 'reviews']
    reviews = np.array(data_new['reviews'])
    labels = np.array(data_new['labels'])

    x_positives, y_positives = reviews[labels == 1], labels[labels == 1]
    x_negatives, y_negatives = reviews[labels == 0], labels[labels == 0]

    x_test, y_test = (
        tf.concat(
            (
                x_positives,
                x_negatives,
            ),
            0,
        ),
        tf.concat(
            (
                y_positives,
                y_negatives,
            ),
            0,
        ),
    )

    raw = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    raw = raw.batch(256).map(
        vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
    )


    result = full_dataset_model.predict(raw)

    data['result'] = result
    data['clean_tweet'] = data_new['reviews']
    # save sentiment result to a csv file
    data.to_csv('result.csv')