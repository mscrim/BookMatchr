#!/opt/local/bin/python
from flask import Flask
from app import app
#app.run(debug = True)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)