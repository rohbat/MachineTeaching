from flask import Flask, jsonify, request, render_template
from database import db
from page_model import PageModel

app = Flask(__name__)
app.config["DEBUG"] = True

SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
    username="cs101teaching",
    password="gogo_teaching",
    hostname="cs101teaching.mysql.pythonanywhere-services.com",
    databasename="cs101teaching$teaching",
)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_POOL_RECYCLE"] = 299
db.init_app(app)

page_model = PageModel("profile1.jpg", "profile2.jpg", "profile3.jpg")

@app.route("/get_imgs")
def get_imgs():
    return jsonify(page_model.get_imgs_list()) 

@app.route("/get_response", methods = ['POST'])
def get_response():
    if request.method == 'POST':
        data = request.get_data()
        print data
    if data == "0":
        page_model.main_img, page_model.compare_img_1 = page_model.compare_img_1, page_model.main_img
    elif data == "1":
        page_model.main_img, page_model.compare_img_2 = page_model.compare_img_2, page_model.main_img 
    return jsonify(page_model.get_imgs_list())

@app.route("/")
def index():
    return render_template('test.html')
 
if __name__ == "__main__":
    app.run()