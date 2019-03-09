# -*- coding: utf-8 -*-

"""
    File name    :    WordDictionary
    Date         :    08/03/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

from collections import defaultdict

class WordDictionary:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.words = defaultdict(list)

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        if word:
            self.words[len(word)].append(word)

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        if not word:
            return False

        for w in self.words[len(word)]:
            match = True
            for i,c in enumerate(word):
                if c!=w[i] and c!='.' and w[i]!='.':
                    match = False
                    break
            if match:
                return True
        return False



        # Your WordDictionary object will be instantiated and called as such:
        # obj = WordDictionary()
        # obj.addWord(word)
        # param_2 = obj.search(word)
