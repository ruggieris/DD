# -*- coding: utf-8 -*-
"""
dd Discrimination Discovery
version 1.1 May 2022

@author: Salvatore Ruggieri
"""

import numpy as np
import pandas as pd
import pyroaring
import csv
import fim
import sys
import urllib
import gzip
import codecs
import heapq
import time

def argmax(values, f):
    """ Argmax function 
    
    Parameters:
    values (iterable): collection of values
    f (value->number): functional 
    
    Returns:
    bestp: index of best value
    bestf: best value, i.e., max{f(v) | v in values}
    """
    bestf = None
    bestp = None
    for i, v in enumerate(values):
        fv = f(v)
        if bestf is None or (fv is not None and bestf < fv):
            bestf, bestp = fv, i
    return bestp, bestf

class Pair:
    """ pairs of objects """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __lt__(self, other):
        """ lexicographic ordering """
        return self.a < other.a or (self.a == other.a and self.b < other.b)
    
    def __le__(self, other):
        """ lexicographic ordering """
        return self.a < other.a or (self.a == other.a and self.b <= other.b)
    
    def __eq__(self, other):
        """ lexicographic ordering """
        return self.a == other.a and self.b == other.b
    
def getReader(filename, encoding='utf8'):
    """ Return a reader from a file, url, or gzipped file/url """
    if filename=='':
        return sys.stdin
    try:
        if filename.endswith('.gz'):
            file = gzip.open(filename)
        else:
            file = open(filename, encoding=encoding)
    except:
        file = urllib.request.urlopen(filename)
        if filename.endswith('.gz'):
            file = gzip.GzipFile(fileobj=file)
        reader = codecs.getreader(encoding)
        file = reader(file)
    return file  

def getCSVattributes(freader, sep=','):
    """ Return the list of attributes in the header of a CSV or ARFF file reader """
    result = []
    line = freader.readline()
    if line.startswith('@relation'):
        # read header from ARFF
        for line in freader:
            if line.startswith('@data'):
                break
            else:
                if line.startswith('@attribute'):
                    result.append(line.split(' ')[1])
    else:
        # read header from CSV
        result = line.strip().split(sep)
    return result

def get_att(itemDesc):
    """ Extract attribute name from attribute=value string """
    pos = itemDesc.find('=')
    if pos>=0:
        return itemDesc[:pos]
    else:
        return ''
 
def CSV2tranDB(filename, sep=',', na_values={'?'}, domains=dict()):
    """ Read a CSV or ARFF file and encode it as a transaction database of attribute=value items
    
    Parameters:
    filename (string): filename, url, or gzipped file/url
    sep (string): column separator
    na_values (set: string coding missing values
    domains (dictionary): dictionary mapping binary variable names to values to be encoded. If a variable is not mapped, all values are encoded
    
    Returns:
    list: list of transactions
    dict: coding of items to numbers
    dict: decoding of numbers to items
    """
    with getReader(filename) as inputf:
        # header 
        attributes = getCSVattributes(inputf, sep=sep)
        # reader for the rest of the file
        csvreader = csv.reader(inputf, delimiter=sep)
        nitems = int(0)
        codes = {}
        tDB = []
        # scan rows in CSV
        for values in csvreader:
            if len(values)==0:
                continue
            # create transaction
            transaction = []
            for att, item in zip(attributes, values):
               if item in na_values:
                   continue
               if att in domains and item not in domains(att):
                   continue                
               attitem = att + "=" + item
               code = codes.get(attitem) # code of attitem
               if code is None:
                   codes[attitem] = code = nitems
                   nitems += 1
               transaction.append(code)
            # append transaction
            tDB.append(np.array(transaction))
    # decode list
    decodes = {code:attitem for attitem, code in codes.items()}
    return (tDB, codes, decodes)

def PD2tranDB(df, na_values={'NaN'}, domains=dict()):
    """ Read a dataframe and encode it as a transaction database of attribute=value items
    
    Parameters:
    df (pd.DataFrame): dataframe
    na_values (set: string coding missing values
    domains (dictionary): dictionary mapping binary variable names to values to be encoded. If a variable is not mapped, all values are encoded
    
    Returns:
    list: list of transactions
    dict: coding of items to numbers
    dict: decoding of numbers to items
    """
    nitems = int(0)
    codes = {}
    tDB = []
    for _, row in df.iterrows():
        transaction = []
        for att, item in row.iteritems():
            if item in na_values:
                continue
            if att in domains and item not in domains(att):
                continue                
            attitem = att + "=" + str(item)
            code = codes.get(attitem) # code of attitem
            if code is None:
               codes[attitem] = code = nitems
               nitems += 1
            transaction.append(code)
        # append transaction
        tDB.append(np.array(transaction))
    # decode list
    decodes = {code:attitem for attitem, code in codes.items()}
    return (tDB, codes, decodes)

from sklearn.preprocessing import LabelEncoder

def encode(df, na_values={'NaN'}, domains=dict()):
    """ Encode a dataframe of categories
    
    Parameters:
    df (pd.DataFrame): dataframe
    na_values (set: string coding missing values
    domains (dictionary): dictionary mapping binary variable names to values to be encoded. If a variable is not mapped, all values are encoded
    
    Returns:
    dict: coding of items to numbers
    dict: decoding of numbers to items
    """
    res = pd.DataFrame()
    encoders = dict()
    for col in df.columns:
        col_encoder = LabelEncoder()
        res[col] = col_encoder.fit_transform(df[col])
        res.loc[df[col].isnull(), col] = np.nan
        res[col] = res[col].astype('category')
        encoders[col] = col_encoder
    return (res, encoders)

class tDBIndex:
    """ A transaction database index storing covers of item in bitmaps """
    def __init__(self, tDB):
        """ Create database index from a list of transactions """
        items =  { item for t in tDB for item in t }
        covers = { item:pyroaring.BitMap() for item in items }
        for tid, t in enumerate(tDB):
            for item in t:
                covers[item].add(tid)
        self.covers = { s:pyroaring.FrozenBitMap(b) for s,b in covers.items() }
        self.ncolumns = len(items)
        self.nrows = len(tDB)
           
    def cover(self, itemset, base=None):
        """ Return cover of an itemset (list of items) """
        nitems = len(itemset)
        if nitems==0:
            return pyroaring.BitMap(np.arange(self.nrows)) if base is None else base
        if base is None:
            return pyroaring.BitMap.intersection(*[self.covers[item] for item in itemset])
        return pyroaring.BitMap.intersection(base, *[self.covers[item] for item in itemset])
    
    def cover_none(self):
        """ Return empty cover """
        return pyroaring.BitMap([])

    def cover_all(self):
        """ Return cover all set """
        return pyroaring.BitMap(np.arange(self.nrows))

    def supp(self, itemset, base=None):
        """ Return support of an itemset (list of items) """
        return len(self.cover(itemset, base))

class ContingencyTable:
    """ A contingency table. 
    
     contingency table for inference (protected=true.pos, n1=a,n2=c)
     ===========
     true.pos      a             
     true.neg      c             
     ===========   n()
    
     confusion matrix (protected=true.pos)
     =========== pred.pos === pred.neg === 
     true.pos        a            b       n1()
     true.neg        c            d       n2()
     ===========    m1()  ===    m2()  ==  n()

     contingency table for independence
     =========== positives === negatives === 
     protected       a            b       n1()
     unprotected     c            d       n2()
     ===========    m1()  ===    m2()  ==  n()
    
     contingency table for separation
          protected                                   unprotected
     ========= pred.pos  ==  pred.neg  ===     ====  pred.pos  ==  pred.neg  === 
     true.pos    TPp          FNp      Pp()           TPu           FNu      Pu()
     true.neg    FPp          TNp      Np()           FPu           TNu      Nu()
     ==========   a     =====  b  ===  n1()   ====     c    ====     d  ===  n2()
    """
    def __init__(self, a, n1, c, n2, TPp=None, Pp=None, TPu=None, Pu=None, 
                 avgdecNeg=0.5, avgtruNeg=0.5, ctx=None, ctx_n=None, protected=None):
        """ Init the contingency table. """
        # store both contingency table for independence
        self.a = int(a)
        self.b = int(n1-a)
        self.c = int(c)
        self.d = int(n2-c)
        # and for separation
        self.TPp = self.a if TPp is None else int(TPp)
        self.FPp = self.a - self.TPp
        self.FNp = int(0) if Pp is None else int(Pp)-self.TPp
        self.TNp = self.b - self.FNp

        self.TPu = self.a if TPu is None else int(TPu)
        self.FPu = self.c - self.TPu
        self.FNu = int(0) if Pu is None else int(Pu)-self.TPu
        self.TNu = self.d - self.FNu

        self.avgdecNeg = None if avgdecNeg is None else float(avgdecNeg)
        self.avgtruNeg = None if avgtruNeg is None else float(avgtruNeg)
        self.ctx = ctx
        self.ctx_n = int(0) if ctx_n is None else int(ctx_n) 
        self.protected = protected
        
    def __lt__(self, other):
        return self.ctx < other.ctx
    
    def __eq__(self, other):
        return self.ctx == other.ctx
    
    def __hash__(self):
        return hash(self.ctx)

    def n1(self):
        return self.a+self.b
    
    def n2(self):
        return self.c+self.d
    
    def n(self):
        return self.a+self.b+self.c+self.d
    
    def m1(self):
        return self.a+self.c
    
    def m2(self):
        return self.b+self.d
    
    def p1(self):
        n1 = self.n1()
        if n1 > 0:
            return self.a / n1
        return self.avgdecNeg
        
    def p2(self):
        n2 = self.n2()
        if n2 > 0:
            return self.c / n2
        return self.avgdecNeg
        
    def p(self):
        return self.m1() / self.n()
        
    def rd(self):
        return self.p1() - self.p2()
        
    def ed(self):
        return self.p1() - self.p()
        
    def rr(self):
        p2 = self.p2();
        if p2==0:
            p2 = 1/(self.n2()+1)
        return self.p1()/p2
        
    def er(self):
        p = self.p();
        if p==0:
            return 0
        return self.p1()/p
        
    def rc(self):
        p2 = self.p2();
        if p2==1:
            p2 = self.n2/(self.n2()+1)
        return (1-self.p1()) / (1-p2)
        
    def ec(self):
        p = self.p();
        if p==1:
            n = self.n()
            p = n/(n+1)
        return (1-self.p1()) / (1-p)
        
    def orisk(self):
        p1 = self.p1();
        if p1==1:
            p1 = self.a/(self.a+1)
        p2 = self.p2()
        if p2==0:
            p2 = 1/(self.n2()+1)
        return p1/(1-p1)*(1-p2)/p2
    
    def Pp(self):
        return self.TPp+self.FNp
    
    def Np(self):
        return self.FPp+self.TNp

    def Pu(self):
        return self.TPu+self.FNu
    
    def Nu(self):
        return self.FPu+self.TNu
    
    def accu(self):
        return (self.TPu + self.TNu)/self.n2()
    
    def accp(self):
        return (self.TPp + self.TNp)/self.n1()
    
    def acc_diff(self):
        ''' Accuracy equality '''
        return self.accu() - self.accp()
   
    def tpru(self):
        return self.TPu/self.Pu()
    
    def tprp(self):
        return self.TPp/self.Pp()
    
    def tpr_diff(self):
        ''' Equal opportunity '''
        return self.tprp() - self.tpru()
    
    def tnru(self):
        return self.TNu/self.Nu()
    
    def tnrp(self):
        return self.TNp/self.Np()
    
    def tnr_diff(self):
        ''' Equal opportunity '''
        return self.tnru()-self.tnrp()
    
    def eq_odds(self):
        ''' Equalized odds '''
        return max(self.fpr_diff(), self.tpr_diff())
    
class DD:
    """ Discrimination discovery class. """
    def __init__(self, df, unprotectedItem, denydecItem=None, denytruItem=None, na_values={'NaN', '?'}, domains=dict()):
        """ Init with given parameters
        
        Parameters:
        df (pd.DataFrame): dataframe
        unprotectedItem (string): item of unprotected group in the form "att_name=value", e.g., "sex=male". All other values of att_name will be considered as protected grounds
        denydecItem (string): item of negative decision in the form "att_name=value", e.g., "grant=deny". The att_name is assumed to be binary
        denytruItem (string): optional item of correct negative decision in the form "att_name=value", e.g., "will_repay=no". The att_name is assumed to be binary
        na_values (set: string coding missing values
        domains (dictionary): dictionary mapping binary variable names to values to be encoded. If a variable is not mapped, all values are encoded
        """
        if isinstance(df, pd.DataFrame):
            self.tDB, self.codes, self.decodes = PD2tranDB(df, na_values=na_values, domains=domains)
        else:
            self.tDB, self.codes, self.decodes = CSV2tranDB(df, na_values=na_values, domains=domains)
        self.decodes[-1] = unprotectedItem.replace('=', '!=')
        #print(self.decodes)
        self.itDB = tDBIndex(self.tDB)

        self.unprotectedItem = unprotectedItem
        self.sensitiveAtt = get_att(unprotectedItem)
        self.unprotected = self.codes[unprotectedItem]
        self.protected = [self.codes[v] for v in self.codes if get_att(v)==self.sensitiveAtt and self.codes[v]!=self.unprotected]
        self.unprCover = self.itDB.covers[self.unprotected]

        self.denydecItem = denydecItem
        if denydecItem is None: 
            self.decisionAtt = self.negDec = self.posDec = self.grantdecItem = None
            self.negdecCover = self.avgdecNeg = None
            denytruItem = None # force true item to None
        else:
            self.decisionAtt = get_att(denydecItem)
            self.negDec = self.codes[denydecItem]
            posDecs = [self.codes[v] for v in self.codes if get_att(v)==self.decisionAtt and self.codes[v]!=self.negDec]
            assert len(posDecs) == 1, "binary decisions only!"
            self.posDec = posDecs[0]
            self.grantdecItem = self.decodes[self.posDec]
            self.negdecCover = self.itDB.covers[self.negDec]
            self.avgdecNeg = len(self.negdecCover) / self.itDB.nrows
            
        self.denytruItem = denytruItem
        if denytruItem is None:
            self.truAtt = self.negtruDec = self.posTru = self.granttruItem = None 
            self.negtruCover = self.avgtruNeg = self.unpr_negtru = None
        else:
            self.truAtt = get_att(denytruItem)
            self.negtruDec = self.codes[denytruItem]
            posTrus = [self.codes[v] for v in self.codes if get_att(v)==self.truAtt and self.codes[v]!=self.negtruDec]
            assert len(posTrus) == 1, "binary decisions only!"
            self.posTru = posTrus[0]
            self.granttruItem = self.decodes[self.posTru]
            self.negtruCover = self.itDB.covers[self.negtruDec] if self.negtruDec is not None else self.itDB.cover([])
            self.avgtruNeg = len(self.negtruCover) / self.itDB.nrows
            self.unpr_negtru = self.unprCover & self.negtruCover
       
    def ctg(self, itemset, protected, cover=None):
        if len(itemset)>0 and isinstance(itemset[0],str):
            itemset = [self.codes[i] for i in itemset]
        ctx = self.itDB.cover(itemset) if cover is None else cover
        if self.denydecItem is None:
            c = ctx.intersection_cardinality(self.unprCover)
            if protected==-1: # any protected
                prCover = self.itDB.cover_none()
                for pr in self.protected:
                    prCover |= self.itDB.covers[pr]
            else:
                prCover = self.itDB.covers[protected]
            a = ctx.intersection_cardinality(prCover)
            ctg = ContingencyTable(a=a, n1=a, c=c, n2=c, TPp=None, Pp=None, 
                    TPu=None, Pu=None, avgdecNeg=self.avgdecNeg, avgtruNeg=self.avgtruNeg,
                    ctx=itemset, ctx_n=len(ctx), protected=protected)
        else:
            n2 = ctx.intersection_cardinality(self.unprCover)
            ctx_negdec = ctx & self.negdecCover
            c = ctx_negdec.intersection_cardinality(self.unprCover)
            if self.negtruDec is None:
                TPu = Pu = None
            else:
                TPu = ctx_negdec.intersection_cardinality(self.unpr_negtru)
                Pu = ctx.intersection_cardinality(self.unpr_negtru)
            if protected==-1: # any protected
                prCover = self.itDB.cover_none()
                for pr in self.protected:
                    prCover |= self.itDB.covers[pr]
            else:
                prCover = self.itDB.covers[protected]
            n1 = ctx.intersection_cardinality(prCover)
            a = ctx_negdec.intersection_cardinality(prCover)
            if self.negtruDec is None:
                TPp = Pp = None
            else:
                pr_negtru = prCover & self.negtruCover
                TPp = ctx_negdec.intersection_cardinality(pr_negtru)
                Pp = ctx.intersection_cardinality(pr_negtru)
            ctg = ContingencyTable(a=a, n1=n1, c=c, n2=n2, TPp=TPp, Pp=Pp, 
                   TPu=TPu, Pu=Pu, avgdecNeg=self.avgdecNeg, avgtruNeg=self.avgtruNeg,
                   ctx=itemset, ctx_n=len(ctx), protected=protected)
        return ctg

    def ctg_global(self, itemset=[]):
        """ Return contingency table(s) for the whole dataset """
        return [self.ctg(itemset, protected) for protected in self.protected]

    def ctg_rel(self, ctg, base=None):
        """ Return contingency table by changing the context to a given bitmap """
        if base is None:
            return self.ctg(ctg.ctx, ctg.protected, cover=self.itDB.cover(ctg.ctx))            
        return self.ctg(ctg.ctx, ctg.protected, cover=self.itDB.cover(ctg.ctx) & base)

    def ctg_any(self, itemset=[], cover=None):
        """ Return contingency table(s) for a specified coverage and ANY protected """
        return self.ctg(itemset, -1, cover=cover)

    def extract(self, minSupp=20, testCond=lambda x: 0, topk=0, target='c'):
        """ Extract top-k contingency tables with minimum support and satisfying a test condition
        
        Parameters:
        minSupp (int): minimum support of contingency table context (negative = absolute, positive = percentage)
        testCond (functional): a function testing a contingency table. testCond(ct) will return None if ct is not to be considered, and a numeric value to be used in ordering contingency tables otherwise 
        topk (int): maximum number of contingency tables in output. The top-k will be outputed wrt the testCond() output
        target (string): type of frequent itemsets ('c' for closed, 's' frequent, 'm' maximal) 
        
        Returns:
        list: list of pairs (testCond(ct), ct) where ct is a topk contingency table
        """
        exclude = {self.codes[v] for v in self.codes if get_att(v) in {self.sensitiveAtt, self.decisionAtt}}
        tDBprojected = [list(set(t)-exclude) for t in self.tDB]
        fisets = fim.fpgrowth(tDBprojected, supp=minSupp, zmin=0, target=target)
        q = []
        if self.denydecItem is None:
            for fi in fisets:
                ctx = self.itDB.cover(fi[0])
                c = ctx.intersection_cardinality(self.unprCover)
                for protected in self.protected:
                    prCover = self.itDB.covers[protected]
                    a = ctx.intersection_cardinality(prCover)
                    ctg = ContingencyTable(a=a, n1=a, c=c, n2=c, TPp=None, Pp=None, 
                            TPu=None, Pu=None, avgdecNeg=self.avgdecNeg, avgtruNeg=self.avgtruNeg,
                            ctx=fi[0], ctx_n=int(fi[1]), protected=protected)
                    v = testCond(ctg)
                    if v is not None and v != False:
                        if len(q) < topk:
                            heapq.heappush(q, (v, ctg))
                        else:
                            heapq.heappushpop(q, (v, ctg))
        else:
            for fi in fisets:
                ctx = self.itDB.cover(fi[0])
                n2 = ctx.intersection_cardinality(self.unprCover)
                ctx_negdec = ctx & self.negdecCover
                c = ctx_negdec.intersection_cardinality(self.unprCover)
                if self.negtruDec is None:
                    TPu = Pu = None
                else:
                    TPu = ctx_negdec.intersection_cardinality(self.unpr_negtru)
                    Pu = ctx.intersection_cardinality(self.unpr_negtru)
                for protected in self.protected:
                    prCover = self.itDB.covers[protected]
                    n1 = ctx.intersection_cardinality(prCover)
                    a = ctx_negdec.intersection_cardinality(prCover)
                    if self.negtruDec is None:
                        TPp = Pp = None
                    else:
                        pr_negtru = prCover & self.negtruCover
                        TPp = ctx_negdec.intersection_cardinality(pr_negtru)
                        Pp = ctx.intersection_cardinality(pr_negtru)
                    ctg = ContingencyTable(a=a, n1=n1, c=c, n2=n2, TPp=TPp, Pp=Pp, 
                           TPu=TPu, Pu=Pu, avgdecNeg=self.avgdecNeg, avgtruNeg=self.avgtruNeg,
                           ctx=fi[0], ctx_n=int(fi[1]), protected=protected)
                    v = testCond(ctg)
                    if v is not None and v != False:
                        if len(q) < topk:
                            heapq.heappush(q, (v, ctg))
                        else:
                            heapq.heappushpop(q, (v, ctg))
        if len(self.protected)>1:
            ms = -minSupp if minSupp<0 else int(minSupp*self.itDB.nrows)
            q = [ctg for ctg in q if ctg[1].n() >= ms] 
        return sorted(q, reverse=True)
    
    def print(self, ctg):
        """ Pretty print of a contingency table ctg """
        protectedDesc = ' ' if self.unprotected is None else self.decodes[ctg.protected]
        n = ctg.n()
        if ctg.ctx != [-1]:
            print('-----\nContext =', ' AND '.join([self.decodes[it] for it in ctg.ctx]) if ctg.ctx !=[] else 'ALL')
        else:
            print('-----\nContext = <extensional>')
        print('Size = {}  Perc = {:.2f}%'.format(ctg.ctx_n, 100.0*ctg.ctx_n/self.itDB.nrows))
        if self.denydecItem is None:
            xlen = max(len(protectedDesc), len(self.unprotectedItem))
            spec = ('{:'+str(xlen)+'}|{:'+str(len(str(n)))+'}')
            print(spec.format('', ''))
            print(spec.format(protectedDesc, ctg.a))
            print(spec.format(self.unprotectedItem, ctg.c))
            print(spec.format('', ctg.m1()))
        elif self.negtruDec is None:
            xlen = max(len(protectedDesc), len(self.unprotectedItem))
            spec = ('{:'+str(xlen)+'}|{:'+str(len(self.denydecItem))+'}|{:'+str(len(self.grantdecItem))+'}|{:'+str(len(str(n)))+'}')
            print(spec.format('', self.denydecItem, self.grantdecItem, ''))
            print(spec.format(protectedDesc, ctg.a, ctg.b, ctg.n1()))
            print(spec.format(self.unprotectedItem, ctg.c, ctg.d, ctg.n2()))
            print(spec.format('', ctg.m1(), ctg.m2(), n))
        else:
            xlen = max(len(self.denytruItem), len(self.granttruItem))
            spec = '{:'+str(xlen+3+len(self.denydecItem)+len(self.grantdecItem)+len(str(ctg.n1())))+'}      '+'{}'
            print(spec.format(protectedDesc, self.unprotectedItem))
            spec = ('{:'+str(xlen)+'}|{:'+str(len(self.denydecItem))+'}|{:'+str(len(self.grantdecItem))+'}|{:'+str(len(str(n)))+'}')
            spec = spec + '     ' + spec
            print(spec.format('', self.denydecItem, self.grantdecItem, '', '', self.denydecItem, self.grantdecItem, ''))
            print(spec.format(self.denytruItem, ctg.TPp, ctg.FNp, ctg.Pp(), self.denytruItem, ctg.TPu, ctg.FNu, ctg.Pu()))
            print(spec.format(self.granttruItem, ctg.FPp, ctg.TNp, ctg.Np(), self.granttruItem, ctg.FPu, ctg.TNu, ctg.Nu()))
            print(spec.format('', ctg.a, ctg.b, ctg.n1(), '', ctg.d, ctg.c, ctg.n2()))
            
    def cover_n(self, patterns, f, k=None):
        """ Naive max cover 
        
        Parameters:
        patterns (iterable): collection of patterns
        k (int): max number of patterns in cover
        f (pattern, bitmap->number): pattern importance relative to uncovered subset
        
        Returns:
        list: greedy cover of argmax_{x subseteq aset, |x| <= k} f(x)
        """
        db = self.itDB
        start_time = time.perf_counter()
        active = db.cover_all() 
        pset = list(patterns)
        covers = []
        residuals = []
        times = []
        nuncovered = db.nrows - len(self.unprCover)
        if k is None or k >= len(pset):
            k = len(pset)
        while k > 0 and nuncovered > 0:
            bestp, bestv = argmax(pset, lambda patt: f(self.ctg_rel(patt, base=active)))
            if bestv is None:
                break
            bestb = self.ctg_rel(pset[bestp], base=active)
            new_cov = db.cover(bestb.ctx+(bestb.protected,), base=active)
            new_supp = len(new_cov)
            if new_supp==0:
                break
            tm = time.perf_counter() - start_time
            covers.append(bestb)
            residuals.append(new_supp)
            times.append(tm)
            active -= new_cov
            nuncovered -= new_supp
            #print(bestv, new_supp, nuncovered)
            k -= 1
            del pset[bestp]
        all_covered = db.cover_none()
        for c in covers:
            all_covered |= db.cover(c.ctx)
        ctg = self.ctg_any([-1], cover=all_covered)
        ctg_rem = self.ctg_any([-1], cover=db.cover_all()-all_covered)
        return (covers, residuals, times, active-self.unprCover, ctg, ctg_rem)

""" Sample usage """
if __name__ == '__main__':
        
    def check_acc(ctg):
        n = ctg.n()
        if n == 0:
            return None # this may occur for relative contingency table
        acc = max(ctg.a,ctg.c)/n
        return ( int(acc*n/10), acc) if acc>0.9 and n>10 else None

    start_time=time.perf_counter() 
    
    #disc = DD("../data/credit.csv", 'age=from_41d4_le_52d6', 'class=bad')
    disc = DD("../data/credit.csv", 'class=good')
    print('== Global ==')
    for ctg in disc.ctg_global():
       disc.print(ctg)
       #print("RD = {:f}".format(ctg.rd()))    
       #print("ACC = {:.2f}".format(check_acc(ctg)))
    ctgs = disc.extract(testCond=check_acc, minSupp=-20, topk=2000)
    print('== Top ==')
    for v, ctg in ctgs[:2]:
        disc.print(ctg)
        #print("RD = {:f}".format(v))
        #print("ACC = {:.2f}".format(check_acc(ctg)))
    print('== Rel top1 ==')    
    disc.print(disc.ctg_rel(ctgs[0][1], disc.itDB.cover_all()))
    covers, residuals, times, covered, uncovered = disc.cover_n([ctg for _, ctg in ctgs], check_acc, 100)
    print('== Cover ==')    
    for ctg, res in zip(covers, residuals):
        print(res)
        disc.print(ctg)
        #print("ACC = {:.2f}".format(check_acc(ctg)))
        #print("RD = {:f}".format(ctg.rd()))    

    elapsed_time=time.perf_counter()-start_time 
    print('Elapsed time (s): {:.2f}'.format(elapsed_time) )
    print('Contingency tables: {}'.format(len(ctgs)))
