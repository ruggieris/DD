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
import queue as Q

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
        nitems = 0
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
            tDB.append(transaction)
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
    nitems = 0
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
        tDB.append(np.array(transaction, dtype=np.int))
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

""" A transaction database index storing covers of item in bitmaps """
class tDBIndex:
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

    def supp(self, itemset, base=None):
        """ Return support of an itemset (list of items) """
        return len(self.cover(itemset, base))

class ContingencyTable:
    """ A contingency table. 
     contingency table for independence
     =========== dec.deny === dec.grant === 
     protected     a            b     n1()
     unprotected   c            d     n2()
     ===========   m1() ===   m2()  == n()
    
     contingency table for separation
          protected                               unprotected
     ========== tru.deny == true.grant ===   === tru.deny == true.grant === 
     dec.deny    TPp         FPp       a          TPu          FPu       c
     dec.grant   FNp         TNp       b          FNu          TNu       d
     ==========  Pp()  ===== Np() ===  n1()  ===  Pu()  ====   Nu() ===  n2()
    """
    def __init__(self, a, n1, c, n2, TPp=None, Pp=None, TPu=None, Pu=None, avgdecNeg=0.5, avgtruNeg=0.5, ctx=None, protected=None):
        """ Init the contingency table """
        self.a = int(a)
        self.b = int(n1-a)
        self.c = int(c)
        self.d = int(n2-c)
        self.avgdecNeg = float(avgdecNeg)
        self.avgtruNeg = float(avgtruNeg)
        self.ctx = ctx
        self.ctx_n = int(0)
        self.protected = protected
        self.TPp = self.a if TPp is None else int(TPp)
        self.FPp = self.a - self.TPp
        self.FNp = int(0) if Pp is None else int(Pp) - self.TPp
        self.TNp = self.b - self.FNp
        self.TPu = self.c if TPu is None else int(TPu)
        self.FPu = self.c - self.TPu
        self.FNu = int(0) if Pu is None else int(Pu) - self.TPu
        self.TNu = self.d - self.FNu    
        
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
    
    def fpr_diff(self):
        ''' Predictive equality '''
        return self.FPu/self.Nu() - self.FPp/self.Np()
    
    def tpru(self):
        return self.TPu/self.Pu()
    
    def tprp(self):
        return self.TPp/self.Pp()
    
    def tpr_diff(self):
        ''' Equal opportunity '''
        return self.tpru() - self.tprp()
    
    def eq_odds(self):
        ''' Equalized odds '''
        return max(self.fpr_diff(), self.tpr_diff())
    
  
        
def check_rd(ctg, minSupp = 20, threshold=0.1):
    """ Sample on risk difference on a contingency table """
    # at least 20 protected with negative decision and 20 unprotected in total
    if ctg.a < 20 or ctg.n2() < 20:
        return None
    v = ctg.rd()
    # risk difference greater than 0.1
    return v if v > 0.1 else None

class DD:
    """ Discrimination discovery class. """
    def __init__(self, df, unprotectedItem, denydecItem, denytruItem=None, na_values={'NaN', '?'}, domains=dict()):
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
        #print(self.decodes)
        self.unprotectedItem = unprotectedItem
        self.sensitiveAtt = get_att(unprotectedItem)
        self.denydecItem = denydecItem
        self.decisionAtt = get_att(denydecItem)
        self.unprotected = self.codes[unprotectedItem]
        self.protected = [self.codes[v] for v in self.codes if get_att(v)==self.sensitiveAtt and self.codes[v]!=self.unprotected]
        self.negDec = self.codes[denydecItem]
        posDecs = [self.codes[v] for v in self.codes if get_att(v)==self.decisionAtt and self.codes[v]!=self.negDec]
        assert len(posDecs) == 1, "binary decisions only!"
        self.posDec = posDecs[0]
        self.grantdecItem = self.decodes[self.posDec]
        self.denytruItem = denytruItem
        if denytruItem is not None:
            self.truAtt = get_att(denytruItem)
            self.negtruDec = self.codes[denytruItem]
            posTrus = [self.codes[v] for v in self.codes if get_att(v)==self.truAtt and self.codes[v]!=self.negtruDec]
            assert len(posTrus) == 1, "binary decisions only!"
            self.posTru = posTrus[0]
            self.granttruItem = self.decodes[self.posTru]
        else:
            self.truAtt = self.negtruDec = self.posTru = self.granttruItem = None
        self.itDB = tDBIndex(self.tDB)
        self.unprCover = self.itDB.covers[self.unprotected]
        self.negdecCover = self.itDB.covers[self.negDec]
        self.avgdecNeg = len(self.negdecCover) / self.itDB.nrows
        self.negtruCover = self.itDB.covers[self.negtruDec] if self.negtruDec is not None else self.itDB.cover([])
        self.avgtruNeg = len(self.negtruCover) / self.itDB.nrows
        
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
        q = Q.PriorityQueue()
        unpr_negtru = self.unprCover & self.negtruCover
        for fi in fisets:
            ctx = self.itDB.cover(fi[0])
            n2 = ctx.intersection_cardinality(self.unprCover)
            ctx_negdec = ctx & self.negdecCover
            c = ctx_negdec.intersection_cardinality(self.unprCover)
            if self.negtruDec is None:
                Pu = TPu = None
            else:
                TPu = ctx_negdec.intersection_cardinality(unpr_negtru)
                Pu = ctx.intersection_cardinality(unpr_negtru)
            for protected in self.protected:
                prCover = self.itDB.covers[protected]
                n1 = ctx.intersection_cardinality(prCover)
                a = ctx_negdec.intersection_cardinality(prCover)
                if self.negtruDec is None:
                    Pp = TPp = None
                else:                    
                    pr_negtru = prCover & self.negtruCover
                    TPp = ctx_negdec.intersection_cardinality(pr_negtru)
                    Pp = ctx.intersection_cardinality(pr_negtru)
                ctg = ContingencyTable(a, n1, c, n2, TPp, Pp, TPu, Pu, avgdecNeg=self.avgdecNeg, avgtruNeg=self.avgtruNeg)
                v = testCond(ctg)
                if v is not None and v != False:
                    ctg.ctx, ctg.ctx_n, ctg.protected = fi[0], int(fi[1]), protected # set only if test pass
                    q.put( (v, ctg) )
                    if q.qsize() > topk:
                        q.get()
        return sorted([ x for x in q.queue ], reverse=True)

    def ctg_global(self):
        """ Return contingency table(s) for the whole dataset """
        unpr_negtru = self.unprCover & self.negtruCover
        ctx = self.itDB.cover([])
        n2 = ctx.intersection_cardinality(self.unprCover)
        ctx_negdec = ctx & self.negdecCover
        c = ctx_negdec.intersection_cardinality(self.unprCover)
        if self.negtruDec is None:
            Pu = TPu = None
        else:
            TPu = ctx_negdec.intersection_cardinality(unpr_negtru)
            Pu = ctx.intersection_cardinality(unpr_negtru)
        res = []
        for protected in self.protected:
            prCover = self.itDB.covers[protected]
            n1 = ctx.intersection_cardinality(prCover)
            a = ctx_negdec.intersection_cardinality(prCover)
            if self.negtruDec is None:
                Pp = TPp = None
            else:                    
                pr_negtru = prCover & self.negtruCover
                TPp = ctx_negdec.intersection_cardinality(pr_negtru)
                Pp = ctx.intersection_cardinality(pr_negtru)
            ctg = ContingencyTable(a, n1, c, n2, TPp, Pp, TPu, Pu, avgdecNeg=self.avgdecNeg, avgtruNeg=self.avgtruNeg)
            ctg.ctx, ctg.ctx_n, ctg.protected = [], int(self.itDB.nrows), protected 
            res.append(ctg)
        return res

    def print(self, ctg):
        """ Pretty print of a contingency table ctg """
        protectedDesc = self.decodes[ctg.protected]
        n = ctg.n()
        print('-----\nContext =', ' AND '.join([self.decodes[it] for it in ctg.ctx]) if ctg.ctx !=[] else 'ALL')
        print('Size = {}  Perc = {:.2f}%'.format(ctg.ctx_n, 100.0*ctg.ctx_n/self.itDB.nrows))
        if self.negtruDec is None:
            xlen = max(len(protectedDesc), len(self.unprotectedItem))
            spec = ('{:'+str(xlen)+'}|{:'+str(len(self.denydecItem))+'}|{:'+str(len(self.grantdecItem))+'}|{:'+str(len(str(n)))+'}')
            print(spec.format('', self.denydecItem, self.grantdecItem, ''))
            print(spec.format(protectedDesc, ctg.a, ctg.b, ctg.n1()))
            print(spec.format(self.unprotectedItem, ctg.c, ctg.d, ctg.n2()))
            print(spec.format('', ctg.m1(), ctg.m2(), n))
        else:
            xlen = max(len(self.denydecItem), len(self.grantdecItem))
            spec = '{:'+str(xlen+3+len(self.denytruItem)+len(self.granttruItem)+len(str(ctg.n1())))+'}      '+'{}'
            print(spec.format(protectedDesc, self.unprotectedItem))
            spec = ('{:'+str(xlen)+'}|{:'+str(len(self.denytruItem))+'}|{:'+str(len(self.granttruItem))+'}|{:'+str(len(str(n)))+'}')
            spec = spec + '     ' + spec
            print(spec.format('', self.denytruItem, self.granttruItem, '', '', self.denytruItem, self.granttruItem, ''))
            print(spec.format(self.denydecItem, ctg.TPp, ctg.FPp, ctg.a, self.denydecItem, ctg.TPu, ctg.FPu, ctg.c))
            print(spec.format(self.grantdecItem, ctg.FNp, ctg.TNp, ctg.b, self.grantdecItem, ctg.FNu, ctg.TNu, ctg.d))
            print(spec.format('', ctg.Pp(), ctg.Np(), ctg.n1(), '', ctg.Pu(), ctg.Nu(), ctg.n2()))
            
""" Sample usage """
if __name__ == '__main__':
    import time
    
    start_time=time.perf_counter() 
    
    disc = DD("data/credit.csv", 'foreign_worker=no', 'class=bad', 'class=bad')
    #disc = DD("adult.csv", 'sex=Male', 'class=-50K')
    for ctg in disc.ctg_global():
        disc.print(ctg)
        print(check_rd(ctg))
    ctgs = disc.extract(testCond=check_rd, minSupp=-20, topk=100)
    for v, ctg in ctgs[:2]:
        disc.print(ctg)
        print(v)
    print('==================')

    elapsed_time=time.perf_counter()-start_time 
    print('Elapsed time (s): {:.2f}'.format(elapsed_time) )
    print('Contingency tables: {}'.format(len(ctgs)))
