import tweepy
import twitter_reader_model as s

# tokens = save_data
api_key = 'CcVrw8qYhrQ5GD79zqyjcoXwx'
api_secret = '7OdUZq1sX0aZOSzaiPKQEyag24fkcBmrXQuwjI4HCtnN66PHR6'
access_token = '1489327655618899970-87WkoIiGQqwPl0x0t5aefzZCJf8TXp'
access_secret = 'EUALKF9JCsRrerzmFym2adGKUGfnqwxmaWTL9hhSuX13F'

#tweepy intro
bearer = 'AAAAAAAAAAAAAAAAAAAAAGtaYwEAAAAAg3BE2cRxvQTNiuzbLgPrRiOKhuU%3DbuGfLA5XaGtMBjhAC9R3s8fiHYRpcgViSltp3S4SuvV30Zn8Nn'

class SentimentStream(tweepy.Stream):
    pos_tweets = []
    neg_tweets = []
    total_tweets = []
    not_confident_tweets = []
    limit = 5
    PositiveSentiment = open("C:/Users/parte/pythonapps/twitter_sentiment/dataset/PositiveTweets.txt", "a")
    NegativeSentiment = open("C:/Users/parte/pythonapps/twitter_sentiment/dataset/NegativeTweets.txt", "a")
    NotConfident = open("C:/Users/parte/pythonapps/twitter_sentiment/dataset/NotConfidentTweets.txt", "a")

    def get_text(self, status):
        if not status.truncated:
            return status.text
        else:
            return status.extended_tweet['full_text']
    
    def on_status(self, status):
        if not status.retweeted and ('RT @' not in status.text):
            text= self.get_text(status)
            classification, confidence = s.sentiment(text)
            self.total_tweets.append(text)
            if classification == 'pos' and confidence >= .66:    #little bit more confident in the positive tweets
                print(classification, confidence, text, end="\n\n")
                self.pos_tweets.append([classification, confidence, text])
                self.PositiveSentiment.write(text)
                self.PositiveSentiment.write("\n")

            elif classification == 'neg' and confidence >= 1.0:
                self.neg_tweets.append([classification, confidence, text])
                print(classification, confidence, text, end="\n\n")
                self.NegativeSentiment.write(text)
                self.NegativeSentiment.write("\n")
            else:
                print('Not confident enough', classification, confidence, text)
                self.not_confident_tweets.append([classification, confidence, text])
                self.NotConfident.write(text)
                self.NotConfident.write("\n")

            if len(self.total_tweets) >= self.limit and self.limit > 0:
                self.PositiveSentiment.close()
                self.NegativeSentiment.close()
                self.NotConfident.close()
                self.disconnect() 
   
stream = SentimentStream(api_key, api_secret, access_token, access_secret)

keywords = ['destiny2, @A_dmg04, bungie, DestinyTheGame, BungieHelp']
languages = ["en"]

stream.filter(languages=languages, track=keywords)

#To track adds per run still - reused code directly
print()
print('Positive tweets:', len(stream.pos_tweets))
print('Negative tweets:', len(stream.neg_tweets))
print('Not Confident tweets: ', len(stream.not_confident_tweets))
print('Total Tweets: ', len(stream.total_tweets))