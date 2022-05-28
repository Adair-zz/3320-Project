# -*- coding: utf-8 -*-

import snscrape.modules.twitter as sntwitter
import pandas as pd

# scrape tweet about Omicron using #omicron keyword
def scrape():
    tweets_list2 = []
    for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper('#omicron since:2022-02-01 until:2022-04-01 lang:en').get_items()):
        tweets_list2.append([tweet.date, tweet.content])

    df = pd.DataFrame(tweets_list2, columns=['Datetime', 'Text'])

    # save tweets to a csv file
    df.to_csv('omicron.csv')


if __name__ == "__main__":
    scrape()