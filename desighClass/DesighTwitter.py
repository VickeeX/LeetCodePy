# -*- coding: utf-8 -*-

"""
    File name    :    DesighTwitter
    Date         :    06/05/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

from collections import defaultdict, deque


class Twitter:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.time = 0  # to sort tweets
        self.tweets = defaultdict(deque)  # to store 10 tweets at most for each user
        self.following = defaultdict(set)  # to record followees

    def postTweet(self, userId: int, tweetId: int) -> None:
        """
        Compose a new tweet.
        """
        self.time += 1
        tws = self.tweets[userId]
        tws.appendleft([self.time, tweetId])  # latest sort
        if len(tws) > 10:  # store at most 10 tweets
            tws.pop()

    def getNewsFeed(self, userId: int) -> list:
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        """
        tws = list(self.tweets[userId])
        for u in self.following[userId]:
            tws.extend(self.tweets[u])
        tws.sort(key=lambda p: p[0], reverse=True)
        return [tw for _, tw in tws[:10]]

    def follow(self, followerId: int, followeeId: int) -> None:
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        """
        if followerId != followeeId:  # avoid repeat of slef's tw
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        """
        fl = self.following[followerId]
        if followeeId in fl:
            fl.remove(followeeId)

            # Your Twitter object will be instantiated and called as such:
            # obj = Twitter()
            # obj.postTweet(userId,tweetId)
            # param_2 = obj.getNewsFeed(userId)
            # obj.follow(followerId,followeeId)
            # obj.unfollow(followerId,followeeId)
