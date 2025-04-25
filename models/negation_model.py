"""
Authors: Adan Baca and Aidan Linder

Implements another simple model that checks for the
presence of negation words and ranks documents
by the document with the larger amount of 
negation words if there is a negation word
in the query and vice versa otherwise.

"""

class negationModel:
    def __init__(self):
        self.negation_words = ['no', 'not', 'never', 'none', 'nobody', 'nothing',
        'neither', 'nor', 'cannot', "can't", "don't",
        "doesn't", "didn't", "won't", "wouldn't",
        "shouldn't", "couldn't", "isn't", "aren't",
        "wasn't", "weren't", "haven't", "hasn't",
        "hadn't", "mustn't", "needn't", "shan't",
        'without', 'barely', 'hardly', 'scarcely', 'rarely',
        'fail', 'fails', 'failed', 'lacks', 'lacked',
        'absent', 'except']
        
    
    def rank_documents(self, query, documents):
        negation_counts = {0:0, 1:0}
        
        for i in range(len(documents)):
            for token in documents[i].split():
                if token.lower() in self.negation_words:
                    negation_counts[i] += 1
                    
        for token in query.split():
            if token.lower() in self.negation_words:
                if negation_counts[0] > negation_counts[1]:
                    return (0,1)
                else:
                    return (1,0)
                
        if negation_counts[0] < negation_counts[1]:
            return (0,1)
        else:
            return (1,0)