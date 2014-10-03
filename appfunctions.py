import MySQLdb as mdb
#import pandas as pd
import numpy as np
import nltk
from collections import Counter
from operator import itemgetter
import string
import sys
from sets import Set
import time as tm
import datetime as dt
#import matplotlib.pyplot as plt
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})


def collect(l, index):
    return map(itemgetter(index), l)
   
def getdata_and_nouns_idf(items):
    conn = mdb.connect('localhost', 'root', '', 'Amazon') #host, user, password, #database
    cur = conn.cursor()
    
    #limit = "10000"
    
    # Select all review text
    sql = "SELECT " + items + " FROM Reviews_nounlists WHERE reviewtime <> -1" # LIMIT " + limit
    cur.execute(sql)
    rows = cur.fetchall()
    
    sql = "SELECT Noun, IDF FROM NounlistIDF"
    cur.execute(sql)
    nouns_idf = cur.fetchall()
    
    cur.execute('SELECT MAX(reviewtime) AS time_max FROM Reviews_nounlists') # LIMIT ' + limit)
    time_max = cur.fetchall()
    
    cur.execute('SELECT MIN(reviewtime) AS time_min FROM Reviews_nounlists WHERE reviewtime <> -1') # LIMIT ' + limit)
    time_min = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return rows, nouns_idf, time_max, time_min

def wordfreq3(keyword):
    #--------------------- Setup ----------------------
    start = tm.time()
    
    keyword = keyword.lower()

    rows, nouns_idf, time_max, time_min = getdata_and_nouns_idf("title, reviewnouns, score, productID, reviewtime, reviewlength")
    
    time_max = (time_max[0])[0]
    time_min = (time_min[0])[0]

    nouns_idf = dict(nouns_idf)
    
    alltext = ''
    allpostext = ''
    allnegtext = ''

    n_posrev = 0
    n_negrev = 0
    
    n_posrev_all = 0
    n_negrev_all = 0
    
    titlelist = []
    nwords = []
    nkeywords = []
    bookscore = []
    nrev = []
    bookID = []
    
    reviewnouns = []
    
    # For time plot
    nbins = 10
    bsize = (time_max - time_min)/nbins
    time_arr = np.arange(nbins)*bsize + bsize/2 + time_min
    
    nrev_wk_time = np.zeros(nbins)
    #nrev_wk_time[:,0] = time
    
    word_occ = {}
    
    time1 = tm.time()
    for row in rows:
        reviewnouns = row[1]
    
        score = row[2]
        if score in [1,2,3]: 
            n_negrev_all += 1
        if score in [4,5]: 
            n_posrev_all += 1
        
        #lowers = row[1].lower()    
        #if keyword in lowers:
            #if not keyword in nounlist:
            #    print 'Keyword',keyword,'in reviewtext but not in nounlist'
            #remove the punctuation using the character deletion step of translate
        #    no_punctuation = lowers.translate(string.maketrans(string.punctuation, ' '*len(string.punctuation)))
        #    tokens = nltk.word_tokenize(no_punctuation)
        #    if keyword in tokens or keyword_pl in tokens:
        if keyword in reviewnouns:
            bin = (row[4]-time_min)/bsize
            #print bin
            nrev_wk_time[bin] += 1
            nt = float(row[5])
            #alltext += tokens
            #nt = float(len(tokens))
            #nk = float(tokens.count(keyword)) + float(tokens.count(keyword_pl))
            nk = float(reviewnouns.count(keyword))  
            # Positive and negative review text
            if score in [1,2,3]: 
                allnegtext += reviewnouns
                n_negrev += 1
            if score in [4,5]: 
                allpostext += reviewnouns
                n_posrev += 1
                    
            # Save book info according to number of keywords
            if row[0] not in titlelist:
                titlelist.append(row[0])
                nwords.append(nt)
                nkeywords.append(nk)
                bookscore.append(row[2])
                nrev.append(1)
                bookID.append(row[3])
            else:
                ind = titlelist.index(row[0])
                nwords[ind] += nt
                nkeywords[ind] += nk
                bookscore[ind] += row[2]
                nrev[ind] += 1
    #print "First loop:",tm.time()-time1,"seconds"
    #time1 = tm.time()
    
    if len(titlelist) == 0: keywordinreviews = False
    else: keywordinreviews = True
    
    nreviews = sum([n for n in nrev])
    
    revstats = (n_posrev,n_negrev,float(n_posrev)/float(n_posrev_all),float(n_negrev)/float(n_negrev_all),nreviews)
    
    #----------- Rank words by relative frequency --------- 
    
    #print type(allpostext)
    allpostext = nltk.word_tokenize(allpostext)                
    fdistpos = nltk.FreqDist(allpostext)
    
    allnegtext = nltk.word_tokenize(allnegtext)
    fdistneg = nltk.FreqDist(allnegtext)
    
    alltext = nltk.word_tokenize(alltext)
    fdist = nltk.FreqDist(alltext)
    
    for w in fdistpos.keys():
        print w
        #if w == 'etc': print 'w == etc'
    sys.exit(1)
    
    tfidf_pos = [(w, float(fdistpos[w])/float(fdistpos.N())*nouns_idf[w]) for w in fdistpos.keys() if len(w) > 1 and fdistpos[w] > 1]
    tfidf_neg = [(w, float(fdistneg[w])/float(fdistneg.N())*nouns_idf[w]) for w in fdistneg.keys() if len(w) > 1 and fdistneg[w] > 1]
    tfidf_all = [(w, float(fdist[w])/float(fdist.N())*nouns_idf[w]) for w in fdist.keys() if len(w) > 1 and fdist[w] > 1]
    
    #print tfidf_pos
    
    wordcomp = []

    #print "Stuff:",tm.time()-time1,"seconds"
    
    for i in range(len(tfidf_all)):
        w = (tfidf_all[i])[0]
        if w in collect(tfidf_pos,0): posratio = ((tfidf_pos[indp])[1])
        else: posratio = 0.
        if w in collect(wordfreqneg,0): negratio = ((tfidf_neg[indn])[1])
        else: negratio = 0.
        wordcomp.append((w,posratio,negratio,posratio-negratio))
    
    #time1 = tm.time()
    #for i in range(len(tfidf_all)):
    #    w = (tfidf_all[i])[0]
    #    f = (tfidf_all[i])[1]
    #    if w in collect(tfidf_pos,0):
    #        indp = collect(tfidf_pos,0).index(w)
    #        posratio = ((tfidf_pos[indp])[1])/f
    #    else: posratio = 0.
    #    if w in collect(wordfreqneg,0):
    #        indn = collect(wordfreqneg,0).index(w)
    #        negratio = ((wordfreqneg[indn])[1])/f
    #    else: negratio = 0.
    #    wordcomp.append((w,posratio,negratio,posratio-negratio))
    #    #print w, posratio, posratio-negratio
    #print "2nd loop:",tm.time()-time1,"seconds"     
    #top5_pos = sorted(wordcomp,key=itemgetter(3),reverse=True)[:5]
    #top5_neg = sorted(wordcomp,key=itemgetter(3))[:5]
    
    #----------- Find Top 5 books by number of keywords in review ---------
    
    n_rel_occ = np.array(nkeywords)/np.array(nwords)
    nbooks = len(titlelist)
    bookscore = np.array(bookscore)/np.array(nrev)
    sort_index = np.argsort(nkeywords)
    
    #time1 = tm.time()
    num = 5
    if nbooks < 5: num = nbooks 
    top5books = []
    for i in range(num):
        ind = sort_index[-(i+1)]
        top5books.append((titlelist[ind],int(nkeywords[ind]),bookscore[ind],nrev[ind],bookID[ind])) 
    #print "3rd loop:",tm.time()-time1,"seconds" 
    
    #----------- Find Top 5 pos and neg words ---------
    
    top5_pos_wds = []
    top5_neg_wds = []
    
    n = 0
    for word in sorted(tfidf_pos,key=lambda x: x[1], reverse=True):
        # word, posratio, negratio, posratio-negratio, scorediffpos, n_in_posreviews, n_in_negreviews
        # word, tfidf, scorediffpos, n_in_posreviews, n_in_negreviews
        if word[0] != keyword and word[0] != keyword+'s' and word[0] != keyword+'es':
            top5_pos_wds.append((word[0],word[1],0.5,5,1))
            n += 1
            if n == 10: break
    n = 0
    for word in sorted(tfidf_neg,key=lambda x: x[1], reverse = True):
        if word[0] != keyword and word[0] != keyword+'s' and word[0] != keyword+'es':
            top5_neg_wds.append((word[0],word[1],0.,0.,0.))
            #print word[0]
            n += 1
            if n == 10: break

    #--------------- Calculate keyword with time ------------------
    
    dtime = [dt.datetime.fromtimestamp(t).strftime("%Y/%m") for t in time_arr]
    
    keywd_w_time = [] 
    for i in range(nbins):
        keywd_w_time.append([dtime[i],nrev_wk_time[i]])

    return keywordinreviews, top5books, top5_pos_wds, top5_neg_wds, revstats, keywd_w_time
    
keywordinreviews, top5books, top5_pos_wds, top5_neg_wds, revstats, keywd_w_time = wordfreq3("dragon")
print "done!"
    
def extrawordstats(word, keyword):

    keyword = keyword.lower()
    
    rows, nouns_idf, time_max, time_min = getdata_and_nouns_idf("title, reviewnouns, score, productID, reviewtime, reviewlength")
    #print "Read in SQL in",tm.time()-time1,"seconds"
    
    #time_max = (time_max[0])[0]
    #time_min = (time_min[0])[0]

    #nouns_idf = dict(nouns_idf)
    
    scorew = 0.
    nw = 0.
    scorewout = 0.
    nwout = 0.
    npos_all = 0.
    npos_w = 0.
    nneg_all = 0.
    nneg_w = 0.
    
    books = {}
    
    for row in rows:
        lowers = row[1].lower()    
        if keyword in lowers:
        
            score = row[2]
            if score in [1,2,3]: nneg_all += 1.
            else: npos_all += 1.
            
            if word in lowers:
                scorew += score
                nw += 1.
                if score in [1,2,3]: nneg_w += 1.
                else: npos_w += 1.
                if row[0] in books:
                    books[row[0]] = books[row[0]] + 1
                else:
                    books[row[0]] = 1
            else:
                scorewout += score
                nwout += 1.
            
            
    
    scorew = scorew / nw
    scorewout = scorewout / nwout
    
    result = [scorew, scorewout, npos_all, npos_w, nneg_all, nneg_w]
    
    return result, books
    