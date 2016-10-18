import flask
from flask import Flask, jsonify
from flask import request 
from flask import render_template
from page_model import PageModel
app = Flask(__name__)

page_model = PageModel("profile1.jpg", "profile2.jpg", "profile3.jpg")

@app.route("/get_imgs")
def get_imgs():
    return jsonify(page_model.get_imgs_list()) 

@app.route("/get_response", methods = ['POST'])
def get_response():
    if request.method == 'POST':
        data = request.get_data()
        print data
    return data

@app.route("/")
def index():
    return render_template('test.html')
 
if __name__ == "__main__":
    app.run()