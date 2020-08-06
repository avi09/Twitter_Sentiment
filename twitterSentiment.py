from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt

def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    # Create a streaming context with batch interval of 10 sec
    ssc = StreamingContext(sc, 10)   
    ssc.checkpoint("checkpoint")
    sc.setLogLevel("ERROR")

    pwords = load_wordlist("./Dataset/positive.txt")
    nwords = load_wordlist("./Dataset/negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)
    
    
def updateFunction(newValues, runningCount):
    if runningCount is None:
       runningCount = 0
    return sum(newValues, runningCount) 



def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    cn1 = []
    cn2 = []
    time = []

    for x in counts:
        y1 = x[0]
        cn1.append(y1[1])
        y2 = x[1]
        cn2.append(y2[1])

    for i in range(len(counts)):
        time.append(i)

    posLine = plt.plot(time, cn1,'bo-', label='Positive')
    negLine = plt.plot(time, cn2,'go-', label='Negative')
    plt.axis([0, len(counts), 0, max(max(cn1), max(cn2))+50])
    plt.xlabel('Time step')
    plt.ylabel('Word count')
    plt.legend(loc = 'upper left')
    plt.show()
    plt.savefig("plot.png", format="png")




def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    words = {}
    f = open(filename, 'rU')
    text = f.read()
    text = text.split('\n')
    for line in text:
        words[line] = 1
    f.close()
    return words


def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1])

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE

    words = tweets.flatMap(lambda line:line.split(" "))
    
    wx = words.map(lambda x: ('Positive', 1) if x in pwords else ('Positive', 0))
    wy = words.map(lambda x: ('Negative', 1) if x in nwords else ('Negative', 0))
    
    total = wx.union(wy)
    sentimentCounts = total.reduceByKey(lambda x,y:x+y)
    
    runningSentimentCounts = sentimentCounts.updateStateByKey(updateFunction)
    runningSentimentCounts.pprint()    
    
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    sentimentCounts.foreachRDD(lambda t, rdd: counts.append(rdd.collect()))

    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()
