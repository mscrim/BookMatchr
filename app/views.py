from flask import Flask, render_template, request, jsonify
from app import app
import pymysql as mdb
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
        keywordinreviews, top5books, top5_pos_wds, top5_neg, revstats, keywd_w_time = appfunc.wordfreq2(keyword)
    
    if keywordinreviews:
        return render_template('search.html', keyword=keyword, top5books=top5books, 
                            top5_pos_wds=top5_pos_wds,top5_neg=top5_neg,revstats=revstats,keywd_w_time=keywd_w_time)
    else: 
        return render_template('search2.html', keyword=keyword)

@app.route('/author.html')
def author():
    return render_template('author.html')

@app.route("/sentences", methods=['GET'])
def get_sentences():
    wordname = request.args.get('wordname', type=str)
    kwordname = request.args.get('kwordname', type=str)
    scorediff = request.args.get('scorediff', type=float)
    nreviews = request.args.get('nreviews', type=int)
    
    data = {}
    data[0] = 'Occurrence of <b>' + wordname + '</b> in reviews:'
    data[0] += '<br> <b>"' + wordname + '"</b> occurs in ?? out of ' + str(nreviews) + ' reviews contraining <b>"' + kwordname + '"</b><br>'
    if scorediff > 0:
        data[0] += '<br> Adding <b> ' + wordname +' </b> will increase the average star rating by <b>' + str("%.1f" % scorediff) + '</b> <br>'
    else:
        data[0] += '<br> Adding <b> ' + wordname +' </b> will decrease the average star rating by <b>' + str("%.1f" % scorediff) + '</b> <br>'
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