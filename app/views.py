from flask import Flask, render_template, request, jsonify
from app import app
import MySQLdb as mdb
import appfunctions as appfunc

#db = mdb.connect(user="root", host="localhost", db="world_innodb", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
    
	return render_template("index.html")

@app.route('/search.html')
def search():
    keyword = request.args.get("query",None)
    #topwords, scorediff = appfunc.topbooks(keyword)
    if " " in keyword:
        return render_template('search3.html', keyword=keyword)
    else:
        keywordinreviews, top5books, top5_pos_wds, top5_neg_wds, revstats, keywd_w_time = appfunc.wordfreq3(keyword)
    
    if keywordinreviews:
        return render_template('search.html', keyword=keyword, top5books=top5books, 
                            top5_pos_wds=top5_pos_wds,top5_neg_wds=top5_neg_wds,revstats=revstats,keywd_w_time=keywd_w_time)
    else: 
        return render_template('search2.html', keyword=keyword)

@app.route('/author.html')
def author():
    return render_template('author.html')
    
@app.route('/slides')
def slides():
    return render_template('slidestack.html')

@app.route("/sentences", methods=['GET'])
def get_sentences():
    wordname = request.args.get('wordname', type=str)
    kwordname = request.args.get('kwordname', type=str)
    nreviews = request.args.get('nreviews', type=int)
    
    wordresults, books = appfunc.extrawordstats(wordname, kwordname)
    scorediff = wordresults[0]-wordresults[1]
    
    data = {}
    data[0] = '<b> Occurrence of "' + wordname + '" in reviews:</b> <br>'
    data[0] += '<li type="disc"> Occurs in ' + str(wordresults[3]) + ' out of ' + str(wordresults[2]) + ' positive reviews containing <b>' + kwordname + '.</b></li>'
    data[0] += '<li type="disc"> Occurs in ' + str(wordresults[5]) + ' out of ' + str(wordresults[4]) + ' negative reviews containing <b>' + kwordname + '.</b></li><br>'
    if scorediff > 0:
        data[0] += 'Adding <b> ' + wordname +' </b> increases the average star rating from <b>' + str("%.1f" % wordresults[1]) + '</b> to <b>' + str("%.1f" % wordresults[0]) + '</b><br>'
    else:
        data[0] += 'Adding <b> ' + wordname +' </b> decreases the average star rating from <b>' + str("%.1f" % wordresults[1]) + '</b> to <b>' + str("%.1f" % wordresults[0]) + '</b><br>'
    data[0] += '<br><b> Top books containing both "' + kwordname + '" and "' + wordname + '": </b><br>'
    
    for item in sorted(books.items(), key=lambda x: x[1]):
        if item[1] == 1: data[0] += '<li type="disc">' + item[0] + ' (' + str(item[1]) + ' occurrence of "' + wordname + '"). </li><br>'
        else: data[0] += '<li type="disc">' +  item[0] + ' (' + str(item[1]) + ' occurrences of "' + wordname + '"). </li><br>'
 
    return jsonify(data)

'''
@app.route("/db_json")
def cities_json():
    with db:
        cur = db.cursor()
        cur.execute("SELECT Name, CountryCode, Population FROM City ORDER BY Population DESC;")
        query_results = cur.fetchall()
    cities = []
    for result in query_results:
        cities.append(dict(name=result[0], country=result[1], population=result[2]))
    return jsonify(dict(cities=cities))
    
@app.route('/db')
def cities_page():
	with db: 
		cur = db.cursor()
		cur.execute("SELECT Name FROM City LIMIT 15;")
		query_results = cur.fetchall()
	cities = ""
	for result in query_results:
		cities += result[0]
		cities += "<br>"
	return cities

@app.route("/db_fancy")
def cities_page_fancy():
	with db:
		cur = db.cursor()
		cur.execute("SELECT Name, CountryCode, Population FROM City ORDER BY Population LIMIT 15;")

		query_results = cur.fetchall()
	cities = []
	for result in query_results:
		cities.append(dict(name=result[0], country=result[1], population=result[2]))
	return render_template('cities.html', cities=cities)

'''