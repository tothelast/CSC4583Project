"""
Authors: Adan Baca and Aidan Linder
TF-IDF model for NevIR 

Ranks documents based on TF-IDF Scores. When running this on the
train set, we get an accuracy of 0.00 as opposed to 0.02 that was
mentioned in the paper, and we are pretty sure thisis because in 
their implemenation, they stem and preprocess words, whereas we don't.
"""


import collections
import math


class tfidfModel:

    def __init__(self, f):
        # Use lnc to weight terms in the documents:
        #   l: logarithmic tf
        #   n: no df
        #   c: cosine normalization

        # Store the vecorized representation for each document
        #   and whatever information you need to vectorize queries in _run_query(...)

        docs = []
        for line in f:
                docs.append(line.strip('\n'))
            
        self.num_docs = len(docs)
        
        self.doc_vecs = []
        self.df = collections.defaultdict(int)
        
        for doc in docs:
            tokens = doc.lower().split()
            tf = {}
            for token in tokens[1:]:
                if token == '-':
                    continue
                if token not in tf:
                    tf[token] = 0
                tf[token] += 1
            
            vec = {}
            for t,f in tf.items():
                if f > 0:
                    vec[t] = 1 + math.log10(f)
                else:
                    vec[t] = 0
            
            for term in set(tokens):
                self.df[term] += 1
                
            norm = self.cosine_norm(vec)
            
            if norm > 0:
                for term in vec:
                    vec[term] /= norm
            
            self.doc_vecs.append(vec)
        
        self.idf = {}
        for t, df in self.df.items():
            if df > 0:
                self.idf[t] = math.log10(self.num_docs/df)
            else:
                self.idf[t] = 0
                
        
        self.inverted_index = collections.defaultdict(list)
        doc_id = 0
        for vec in self.doc_vecs:
            for term, weight in vec.items():
                self.inverted_index[term].append((doc_id, weight))
            doc_id += 1
                
                    
    def cosine_norm(self,vec):
        norm = 0
        for weight in vec.values():
            norm += weight**2
        norm = math.sqrt(norm)
        return norm
                
    def run_query(self, query):
        terms = query.lower().split()
        return self._run_query(terms)

    def _run_query(self, terms):
        # Use ltn to weight terms in the query:
        #   l: logarithmic tf
        #   t: idf
        #   n: no normalization

        # Return the top-10 document for the query 'terms'
        result = []
        
        query_tf = {}
        for term in terms:
            if term not in query_tf:
                query_tf[term] = 0
            query_tf[term]+=1
        
        query_vec = {}
        for t,f in query_tf.items():
            if f > 0:
                if t in self.idf:
                    query_vec[t] = (1+math.log10(f)) * self.idf[t]
                else:
                    query_vec[t] = 0
                    
        
        scores = {}
        for term, query_weight in query_vec.items():
            if term in self.inverted_index:
                for doc_id, doc_weight in self.inverted_index[term]:
                    if doc_id in scores:
                        scores[doc_id] += (query_weight * doc_weight)
                    else:
                        scores[doc_id] = query_weight * doc_weight
                        
                        
        for doc_id in range(self.num_docs):
            if doc_id not in scores:
                scores[doc_id] = 0
                
        
        sorted_docs = sorted(scores.items())
        
            
        return (sorted_docs[0][0], sorted_docs[1][0])
    
    def rank_documents(self, query, documents):
        self.__init__(documents)
        
        rank = self.run_query(query)
        return rank