from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.sql import text
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer


connection_string = "postgresql://postgres:password@localhost/social"
db_local = create_engine(connection_string)


def flatten_tweets(tweets):
    tweets = pd.json_normalize(tweets.tweet)
    tweets["retweeted_status.entities.user_mentions"] = tweets[
        "retweeted_status.entities.user_mentions"
    ].apply(lambda d: d if isinstance(d, list) else [])
    return tweets


def process_df_columns(tweets):

    tweets_final = pd.DataFrame(
        columns=[
            "created_at",
            "id",
            "in_reply_to_status_id",
            "in_reply_to_screen_name",
            "retweeted_status.id",
            "retweeted_status.user.screen_name",
            "user_mentions_screen_name",
            "full_text",
            "user.screen_name",
        ]
    )

    tweets_flattened = flatten_tweets(tweets)

    equal_columns = ["created_at", "id", "full_text", "user.screen_name"]
    tweets_final[equal_columns] = tweets_flattened[equal_columns]

    # EXTRACTING USER MENTIONS
    tweets_final["user_mentions_screen_name"] = tweets_flattened[
        "retweeted_status.entities.user_mentions"
    ].apply(lambda x: x[0]["screen_name"] if len(x) > 0 else np.nan)

    # EXTRACTING RETWEETED ORIGINAL USER
    tweets_final["retweeted_status.user.screen_name"] = tweets_flattened[
        "retweeted_status.user.screen_name"
    ].apply(lambda x: x if x is not np.nan else np.nan)

    # EXTRACTING IN REPLY TO
    tweets_final["in_reply_to_screen_name"] = tweets_flattened[
        "in_reply_to_screen_name"
    ]

    tweets_final = tweets_final.where((pd.notnull(tweets_final)), None)

    tweets_final = add_nlp_scores_to_df(tweets_final)

    return tweets_final


def save_tweets_to_file(tweets):
    tweets.to_pickle("./tweets.pkl")


def save_formatted_df_to_file(tweets):
    tweets.to_pickle("tweets_final_df_formatted.pkl")


def get_tweets_from_db():
    tweets = pd.read_sql("SELECT * FROM tweets limit 1000000", db_local)
    save_tweets_to_file(tweets)
    tweets_final = process_df_columns(tweets)
    save_formatted_df_to_file(tweets_final)
    return tweets_final


def get_tweets_from_file():
    tweets_final = pd.read_pickle("tweets_final_df_formatted.pkl")
    return tweets_final


def add_nlp_scores_to_df(tweets_df):
    sid = SentimentIntensityAnalyzer()
    tweets_df["nlp_score"] = tweets_df["full_text"].apply(
        lambda x: sid.polarity_scores(x)["compound"]
    )
    return tweets_df
