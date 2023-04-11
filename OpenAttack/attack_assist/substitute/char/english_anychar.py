from .base import CharSubstitute
from ....tags import *
import random


A = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
"0","1","2","3","4","5","6","7","8","9",
"/","?", ":", "@", "&", "=", "+", "$","-", "_", ".", "!", "~", "*", "'", "(", ")", "#"]


class AnycharSubstitute(CharSubstitute):

    TAGS = { TAG_English }

    def __init__(self):
        """
        Returns the chars that is visually similar to the input.

        DCES substitute used in :py:class:`.VIPERAttacker`.

        :Data Requirements: :py:data:`.AttackAssist.SIM`
        :Language: english

        """
        self.A = A

    def substitute(self, char: str):
        """
        :param word: the raw char, threshold: return top k chars.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        randInt = random.randint(0,len-1)
        print("randWord: "+str(A[randInt]))
        if char not in self.h:
            return [(char, 1)]

        
        return [(A[randInt], 1)]
