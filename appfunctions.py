import MySQLdb as mdb
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from operator import itemgetter
import string
import sys
from sets import Set
import time as tme
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def collect(l, index):
    return map(itemgetter(index), l)
   
def getdata(items):
    conn = mdb.connect('localhost', 'root', '', 'Amazon') #host, user, password, #database
    cur = conn.cursor()
    
    # Select all review text
    sql = "SELECT " + items + " FROM Reviews LIMIT 1000"
    cur.execute(sql)
    rows = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return rows

def getdata_and_nouns(items):
    conn = mdb.connect('localhost', 'root', '', 'Amazon') #host, user, password, #database
    cur = conn.cursor()
    
    limit = "10000"
    
    # Select all review text
    sql = "SELECT " + items + " FROM Reviews WHERE reviewtime <> -1 LIMIT " + limit
    cur.execute(sql)
    rows = cur.fetchall()
    
    sql = "SELECT * FROM Nounlist"
    cur.execute(sql)
    nounlist = cur.fetchall()
    
    cur.execute('SELECT MAX(reviewtime) AS time_max FROM Reviews LIMIT ' + limit)
    time_max = cur.fetchall()
    
    cur.execute('SELECT MIN(reviewtime) AS time_min FROM Reviews WHERE reviewtime <> -1 LIMIT ' + limit)
    time_min = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return rows, nounlist, time_max, time_min
    
def getdata_and_nouns_keyword(items,keyword):

    conn = mdb.connect('localhost', 'root', '', 'Amazon') #host, user, password, #database
    cur = conn.cursor()
    
    #limit = "10000"
    
    # Select all review text
    sql = "SELECT " + items + " FROM " + keyword + " WHERE reviewtime <> -1 LIMIT 100"
    cur.execute(sql)
    rows = cur.fetchall()
    
    sql = "SELECT * FROM Nounlist"
    cur.execute(sql)
    nounlist = cur.fetchall()
    
    cur.execute('SELECT MAX(reviewtime) AS time_max FROM ' + keyword + ' LIMIT 100')
    time_max = cur.fetchall()
    
    cur.execute('SELECT MIN(reviewtime) AS time_min FROM ' + keyword + ' WHERE reviewtime <> -1 LIMIT 100')
    time_min = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return rows, nounlist, time_max, time_min

    
def wordfreq2(keyword):
    #--------------------- Setup ----------------------
    start = tme.time()
    
    keyword = keyword.lower()
    keyword_pl = keyword + 's'
    
    if keyword in ['dragon']:
        rows, nounlist, time_max, time_min = getdata_and_nouns_keyword("title, reviewtext, score, productID, reviewtime",keyword)
    else:
        rows, nounlist, time_max, time_min = getdata_and_nouns("title, reviewtext, score, productID, reviewtime")
    
    time_max = (time_max[0])[0]
    time_min = (time_min[0])[0]

    nounlist = [w[0] for w in nounlist if w not in [keyword,keyword_pl]]
    nounlist = Set(nounlist)
    
    alltext = []
    allpostext = []
    allnegtext = []

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
    time = np.arange(nbins)*bsize + bsize/2 + time_min
    
    nrev_wk_time = np.zeros(nbins)
    #nrev_wk_time[:,0] = time
    
    for row in rows:
    
        score = row[2]
        if score in [1,2,3]: 
            n_negrev_all += 1
        if score in [4,5]: 
            n_posrev_all += 1
        
        lowers = row[1].lower()    
        if keyword in lowers:
            #if not keyword in nounlist:
            #    print 'Keyword',keyword,'in reviewtext but not in nounlist'
            #remove the punctuation using the character deletion step of translate
            no_punctuation = lowers.translate(string.maketrans(string.punctuation, ' '*len(string.punctuation)))
            tokens = nltk.word_tokenize(no_punctuation)
            if keyword in tokens or keyword_pl in tokens:
                bin = (row[4]-time_min)/bsize
                #print bin
                nrev_wk_time[bin] += 1
                
                alltext += tokens
                nt = float(len(tokens))
                nk = float(tokens.count(keyword)) + float(tokens.count(keyword_pl))
                
                # Positive and negative review text
                if score in [1,2,3]: 
                    allnegtext += tokens
                    n_negrev += 1
                if score in [4,5]: 
                    allpostext += tokens
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
    
    if len(titlelist) == 0: keywordinreviews = False
    else: keywordinreviews = True
    
    nreviews = sum([n for n in nrev])
    
    revstats = (n_posrev,n_negrev,float(n_posrev)/float(n_posrev_all),float(n_negrev)/float(n_negrev_all),nreviews)
    #print 'revstats = ',revstats
    
    #----------- Rank words by relative frequency ---------                
    filtered = [w for w in allpostext if w in nounlist]
    fdistpos = nltk.FreqDist(filtered)
    
    filtered = [w for w in allnegtext if w in nounlist]
    fdistneg = nltk.FreqDist(filtered)
    
    filtered = [w for w in alltext if w in nounlist]
    fdist = nltk.FreqDist(filtered)
    
    wordfreqpos = [(w, float(fdistpos[w])/float(fdistpos.N())) for w in fdistpos.keys() if len(w) > 1 and fdistpos[w] > 1]
    wordfreqneg = [(w, float(fdistneg[w])/float(fdistneg.N())) for w in fdistneg.keys() if len(w) > 1 and fdistneg[w] > 1]
    wordfreq = [(w, float(fdist[w])/float(fdist.N())) for w in fdist.keys() if len(w) > 1 and fdist[w] > 1]
    
    wordcomp = []

    for i in range(len(wordfreq)):
        w = (wordfreq[i])[0]
        f = (wordfreq[i])[1]
        if w in collect(wordfreqpos,0):
            indp = collect(wordfreqpos,0).index(w)
            posratio = ((wordfreqpos[indp])[1])/f
        else: posratio = 0.
        if w in collect(wordfreqneg,0):
            indn = collect(wordfreqneg,0).index(w)
            negratio = ((wordfreqneg[indn])[1])/f
        else: negratio = 0.
        wordcomp.append((w,posratio,negratio,posratio-negratio))
        #print w, posratio, posratio-negratio
        
    #top5_pos = sorted(wordcomp,key=itemgetter(3),reverse=True)[:5]
    #top5_neg = sorted(wordcomp,key=itemgetter(3))[:5]
    
    #----------- Find Top 5 books by number of keywords in review ---------
    n_rel_occ = np.array(nkeywords)/np.array(nwords)
    nbooks = len(titlelist)
    bookscore = np.array(bookscore)/np.array(nrev)
    sort_index = np.argsort(nkeywords)
    
    num = 5
    if nbooks < 5: num = nbooks 
    top5books = []
    for i in range(num):
        ind = sort_index[-(i+1)]
        top5books.append((titlelist[ind],int(nkeywords[ind]),bookscore[ind],nrev[ind],bookID[ind])) 
        
    #---------------- Calculate score with and without topwords -----------------
    # Later, do for all nouns occurring in reviews, not just topwords
    
    scorew = np.zeros(len(wordcomp))
    scorewout = np.zeros(len(wordcomp))
    nw = np.zeros(len(wordcomp))
    nwout = np.zeros(len(wordcomp))
            
    for row in rows:
        if keyword in row[1].lower():
            # Save time by not doing word_tokenize for every review
            lowers = row[1].lower()
            no_punctuation = lowers.translate(None, string.punctuation)
            tokens = nltk.word_tokenize(no_punctuation)
            if keyword in tokens or keyword_pl in tokens:
                for i in range(len(wordcomp)):
                    topw = (wordcomp[i])[0]
                    if topw in tokens or topw+'s' in tokens:
                        scorew[i] += float(row[2])
                        nw[i] += 1.
                    else:
                        scorewout[i] += float(row[2])
                        nwout[i] += 1.
    
    scorediffpos = scorew/nw - scorewout/nwout
    #print 'scorew:',scorew,'nw:',nw,'scorewout:',scorewout,'nwout:',nwout
    
    allwords = []
    for i in range(len(wordcomp)):
        wd = wordcomp[i]
        allwords.append((wd[3],scorediffpos[i],wd[0],wd[1],wd[2]))

    #allwords = sorted(allwords,reverse=True)[:10]
    
    #------------ Find number of each word in reviews --------------
    pos_wds = []
    neg_wds = []
    for wd in sorted(allwords,reverse=True)[:10]:
        pos_wds.append(wd[2])
    for wd in sorted(allwords)[:10]:
        neg_wds.append(wd[2])
    
    pos_pos = np.zeros(10)
    pos_neg = np.zeros(10)
    neg_pos = np.zeros(10)
    neg_neg = np.zeros(10)
    
    for row in rows:
        score = row[2]
        lowers = row[1].lower()
        if keyword in lowers:
            for i in range(10):
                if pos_wds[i] in lowers:
                    if score in [1,2,3]: pos_neg[i] += 1
                    if score in [4,5]: pos_pos[i] += 1
                if neg_wds[i] in lowers:
                    if score in [1,2,3]: neg_neg[i] += 1
                    if score in [4,5]: neg_pos[i] += 1
    
    top5_pos_wds = []
    top5_neg_wds = []
    
    n = 0
    for wd in sorted(allwords,reverse=True)[:10]:
        # word, posratio, negratio, posratio-negratio, scorediffpos, n_in_posreviews, n_in_negreviews
        top5_pos_wds.append((wd[2],wd[3],wd[4],wd[0],wd[1],pos_pos,pos_pos[n],pos_neg[n]))
        n += 1
    n = 0
    for wd in sorted(allwords)[:10]:
        top5_neg_wds.append((wd[2],wd[3],wd[4],wd[0],wd[1],neg_pos[n],neg_neg[n]))
        n +=1
        
    # Save results to SQL database
    #saveresults(keyword, top5, top5_pos_wds, top5_neg, revstats)
    
    
    # Make plot
    dtime = [dt.datetime.fromtimestamp(t).strftime("%Y/%m") for t in time]
    df = pd.DataFrame(nrev_wk_time.T, index=dtime, columns=[keyword])
    
    print df
    
    #fig1 = plt.figure()
    #with plt.style.context('fivethirtyeight'):
    #    df.plot()
    #    plt.xlabel('Date')
    #    plt.ylabel('Occurrences of word in reviews')
    #    plt.legend(loc=2)
        #ax = df.plot()
        #ax.set_xlabel('Date')
        #ax.set_ylabel('Occurrences of word in reviews')
        ##ax.legend()
        #gca().tight_layout()
    #plt.savefig('app/static/images/keyword_trend.png')

    #plt.show()
    
    #fig1 = plt.figure(facecolor="yellow")
    #ax1 = plt.axes(frameon=False)
    #ax1.set_frame_on(False)
    #ax1.get_xaxis().tick_bottom()
    #ax1.axes.get_yaxis().set_visible(False)
    #xmin, xmax = ax1.get_xaxis().get_view_interval()
    #ymin, ymax = ax1.get_yaxis().get_view_interval()
    #ax1.plot(time,nrev_wk_time,label=keyword)
    ##ax.xaxis.set_major_formatter(ticker.FuncFormatter(time))
    #ax1.set_xlabel('Date')
    #ax1.legend()
    #fig1.autofmt_xdate()
    #fig1.savefig('app/static/images/keyword_trend.png')
    

    keywd_w_time = [] #[['Date',str(keyword)]]
    for i in range(nbins):
        keywd_w_time.append([dtime[i],nrev_wk_time[i]])    
    
    print keywd_w_time
        
    return keywordinreviews, top5books, top5_pos_wds, top5_neg_wds, revstats, keywd_w_time #, df

#keywordinreviews, top5books, top5_pos_wds, top5_neg, revstats = wordfreq2("dragon")

#print "done!"
'''
def getdata(items,keyword):
    conn = mdb.connect('localhost', 'root', '', 'Amazon') #host, user, password, #database
    cur = conn.cursor()
    
    # Select all review text
    sql = "SELECT " + items + " FROM Reviews 
    WHERE " + keyword + " IN reviewtext"
    cur.execute(sql)
    rows = cur.fetchall()
    
    # OR " + keyword + "IN title
    
    cur.close()
    conn.close()
    
    return rows
'''
'''
def topbooks(keyword):

    rows = getdata("title, reviewtext, score")

    #----------- Perform Analysis -----------

    # Make keyword lower case for matching
    keyword = keyword.lower()
    keyword_pl = keyword + 's'
    
    titlelist = []
    nwords = []
    nkeywords = []
    bookscore = []
    nrev = []
    
    alltext = ''
    #allpostext = ''
    #allnegtext = ''
    
    for row in rows:
        if keyword in row[1].lower():
            # Save time by not doing word_tokenize for every review
            text = " ".join(row[1].lower().split('.'))
            tokens = nltk.word_tokenize(text)
            if keyword in tokens or keyword_pl in tokens:
                alltext += text + ' '
                # Positive and negative review text
                #if row[2] in [1,2]: allpostext += text + ''
                #if row[2] in [4,5]: allnegtext += text + ''
                # Number of words occurring
                nt = float(len(tokens))
                nk = float(tokens.count(keyword)) + float(tokens.count(keyword_pl))
                # Save book info according to number of keywords
                if row[0] not in titlelist:
                    titlelist.append(row[0])
                    nwords.append(nt)
                    nkeywords.append(nk)
                    bookscore.append(row[2])
                    nrev.append(1)
                else:
                    ind = titlelist.index(row[0])
                    nwords[ind] += nt
                    nkeywords[ind] += nk
                    bookscore[ind] += row[2]
                    nrev[ind] += 1
    
    alltext = nltk.word_tokenize(alltext)
    #allpostext = nltk.word_tokenize(allpostext)
    #allnegtext = nltk.word_tokenize(allnegtext)
    #pos = []
    #neg = []
    #for i in range(len(allpostext)):
    #    pos.append('positive')
    #for i in range(len(allnegtext)):
    #    neg.append('negative')
    #pospairs = zip(pos, allpostext)
    #negpairs = zip(neg, allnegtext)
    
    #print 'negpairs = ',negpairs
    #posneg_word = pospairs + negpairs
    
    #cfd = nltk.ConditionalFreqDist(posneg_word)
    #print cfd
    #print cfd.conditions()
    #cfd.tabulate(conditions=['positive','negative'],samples=range(10),cumulative=False)
    #wlist = [w for w in alltext if len(w) > 11]
    #print wlist
    #cfd.tabulate(conditions=['positive','negative'],samples=wlist)
    #for x in (y for y in items if y > 10):

    #----------- Find Top 5 books by number of keywords in review ---------
    n_rel_occ = np.array(nkeywords)/np.array(nwords)
    nbooks = len(titlelist)
    bookscore = np.array(bookscore)/np.array(nrev)
    sort_index = np.argsort(n_rel_occ)
    
    num = 5
    if nbooks < 5: num = nbooks 
    top5 = []
    for i in range(num):
        ind = sort_index[-(i+1)]
        top5.append((titlelist[ind],int(nkeywords[ind]),bookscore[ind],nrev[ind])) 
    
    #----------- Find Top 5 words associated with keyword ---------
    
    topnum = 200
    
    #text = nltk.word_tokenize(alltext.lower())
    #top100w = Counter(text).most_common(100)
    #postags = nltk.pos_tag(top100w[])
    #topwords = [word in top100w ]
    topwcount = Counter(alltext).most_common(topnum)
    topw = [w[0] for w in topwcount]
    postags = nltk.pos_tag(topw)
    
    topwords = []
    n = 0    
    for i in range(topnum):
        if (postags[i])[1] in ['NN','NNS']:
            #print postags[i]
            if (postags[i])[0] not in commonwords+[keyword_pl]:
                topwords.append(topwcount[i])
                n += 1
        if n == 5: break
    

    
    #text = nltk.word_tokenize(alltext.lower())
    #bigwords = [ textitem[0] for textitem in nltk.pos_tag(text) if textitem[1] in ['NN'] ]
    #topwords = Counter(bigwords).most_common(5)
    
    
    #---------------- Calculate score with and without topwords -----------------
    # Later, do for all nouns occurring in reviews, not just topwords
    
    scorew = np.zeros(5)
    scorewout = np.zeros(5)
    nw = 0.
    nwout = 0.
    
    for row in rows:
        if keyword in row[1].lower():
            # Save time by not doing word_tokenize for every review
            text = " ".join(row[1].lower().split('.'))
            tokens = nltk.word_tokenize(text)
            if keyword in tokens or keyword_pl in tokens:
                for i in range(5):
                    topw = (topwords[i])[0]
                    if topw in tokens or topw+'s' in tokens:
                        scorew[i] += row[2]
                        nw += 1
                    else:
                        scorewout[i] += row[2]
                        nwout += 1
     
    scorediff = scorew/nw - scorewout/nwout
    #print 'score with topword:',scorew
    #print 'score without topword:', scorewout
    return topwords, scorediff
'''
'''  
def wordfreq(keyword):
    keyword = keyword.lower()
    keyword_pl = keyword + 's'
    
    rows = getdata("title, reviewtext, score")
    
    commonwords = ['still','relates','hardy','given','beyond','makes','bookmy','order','reader','example','point','first','second','third','fourth','whose','highly','besides','though','never','far','others','shelly','literary','(','think','am','getten','communicate','book','story','series','reading','read','up','way','novel','do','books','people','something','make','year','years']
    names = nltk.corpus.names
    namelist = [name.lower() for name in names.words()]
    avoidlist = commonwords + [keyword,keyword_pl] + stopwords.words('english') + namelist
    
    alltext = []
    allpostext = []
    allnegtext = []

    n_posrev = 0
    n_negrev = 0
    
    n_posrev_all = 0
    n_negrev_all = 0
    
    titlelist = []
    nwords = []
    nkeywords = []
    bookscore = []
    nrev = []
    
    for row in rows:
        if row[2] in [1,2]: 
            n_negrev_all += 1
        if row[2] in [4,5]: 
            n_posrev_all += 1
        if keyword in row[1].lower():
            lowers = row[1].lower()
            #remove the punctuation using the character deletion step of translate
            no_punctuation = lowers.translate(None, string.punctuation)
            tokens = nltk.word_tokenize(no_punctuation)
            if keyword in tokens or keyword_pl in tokens:
                alltext += tokens
                nt = float(len(tokens))
                nk = float(tokens.count(keyword)) + float(tokens.count(keyword_pl))
                
                # Positive and negative review text
                if row[2] in [1,2]: 
                    allnegtext += tokens
                    n_negrev += 1
                if row[2] in [4,5]: 
                    allpostext += tokens
                    n_posrev += 1
                    
                # Save book info according to number of keywords
                if row[0] not in titlelist:
                    titlelist.append(row[0])
                    nwords.append(nt)
                    nkeywords.append(nk)
                    bookscore.append(row[2])
                    nrev.append(1)
                else:
                    ind = titlelist.index(row[0])
                    nwords[ind] += nt
                    nkeywords[ind] += nk
                    bookscore[ind] += row[2]
                    nrev[ind] += 1
    
    revstats = (n_posrev,n_negrev,float(n_posrev)/float(n_posrev_all),float(n_negrev)/float(n_negrev_all))
    print 'revstats = ',revstats
    
    #----------- Rank words by relative frequency ---------                
    filtered = [w for w in allpostext if not w in avoidlist and len(w) > 4 and w[-2:] != 'ly']
    fdistpos = nltk.FreqDist(filtered)
    
    filtered = [w for w in allnegtext if not w in avoidlist and len(w) > 4 and w[-2:] != 'ly']
    fdistneg = nltk.FreqDist(filtered)
    
    filtered = [w for w in alltext if not w in avoidlist and len(w) > 4 and w[-2:] != 'ly']
    fdist = nltk.FreqDist(filtered)
    
    wordfreqpos = [(w, float(fdistpos[w])/float(fdistpos.N())) for w in fdistpos.keys() if len(w) > 1 and fdistpos[w] > 1]
    wordfreqneg = [(w, float(fdistneg[w])/float(fdistneg.N())) for w in fdistneg.keys() if len(w) > 1 and fdistneg[w] > 1]
    wordfreq = [(w, float(fdist[w])/float(fdist.N())) for w in fdist.keys() if len(w) > 1 and fdist[w] > 1]
    
    wordcomp = []

    for i in range(len(wordfreq)):
        w = (wordfreq[i])[0]
        f = (wordfreq[i])[1]
        if w in collect(wordfreqpos,0):
            indp = collect(wordfreqpos,0).index(w)
            posratio = ((wordfreqpos[indp])[1])/f
        else: posratio = 0.
        if w in collect(wordfreqneg,0):
            indn = collect(wordfreqneg,0).index(w)
            negratio = ((wordfreqneg[indn])[1])/f
        else: negratio = 0.
        wordcomp.append((w,posratio,negratio,posratio-negratio))
        #print w, posratio, posratio-negratio
        
    top5_pos = sorted(wordcomp,key=itemgetter(3),reverse=True)[:5]
    top5_neg = sorted(wordcomp,key=itemgetter(3))[:5]
    
    #----------- Find Top 5 books by number of keywords in review ---------
    n_rel_occ = np.array(nkeywords)/np.array(nwords)
    nbooks = len(titlelist)
    bookscore = np.array(bookscore)/np.array(nrev)
    sort_index = np.argsort(n_rel_occ)
    
    num = 5
    if nbooks < 5: num = nbooks 
    top5 = []
    for i in range(num):
        ind = sort_index[-(i+1)]
        top5.append((titlelist[ind],int(nkeywords[ind]),bookscore[ind],nrev[ind])) 
        
    #---------------- Calculate score with and without topwords -----------------
    # Later, do for all nouns occurring in reviews, not just topwords
    
    scorew = np.zeros(5)
    scorewout = np.zeros(5)
    nw = 0.
    nwout = 0.
            
    for row in rows:
        if keyword in row[1].lower():
            # Save time by not doing word_tokenize for every review
            lowers = row[1].lower()
            no_punctuation = lowers.translate(None, string.punctuation)
            tokens = nltk.word_tokenize(no_punctuation)
            if keyword in tokens or keyword_pl in tokens:
                for i in range(5):
                    topw = (top5_pos[i])[0]
                    if topw in tokens or topw+'s' in tokens:
                        scorew[i] += row[2]
                        nw += 1
                    else:
                        scorewout[i] += row[2]
                        nwout += 1
     
    scorediffpos = scorew/nw - scorewout/nwout
    top5_pos_wds = []
    
    for i in range(5):
        top5_pos_wds.append(top5_pos[i]+(scorediffpos[i],))
    
    return top5, top5_pos_wds, top5_neg, revstats
'''

#def writedict():



    
def topbook(keyword):
    conn = mdb.connect('localhost', 'root', '', 'Amazon') #host, user, password, #database
    cur = conn.cursor()
    
    # Select all review text
    cur.execute("SELECT title, reviewtext FROM Reviews")
    rows = cur.fetchall()
    
    cur.close()
    conn.close()
    
    #----------- Perform Analysis -----------

    max_occ = 0
    title_max = 'No books found :('
    
    for row in rows:
        if keyword in row[1]:
            tokens = nltk.word_tokenize(row[1])
            if tokens.count(keyword) > max_occ:
                max_occ = tokens.count(keyword)
                title_max = row[0]
    
    return title_max

#top5, top5_pos_wds, top5_neg, revstats = wordfreq2("dragon")
#for wd in allwords:
#    print wd[2],wd[0],wd[1]
    
#print 'done!'
