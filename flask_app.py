from flask import Flask, jsonify, request, render_template
from sqlalchemy.orm.session import sessionmaker, make_transient
from sqlalchemy import create_engine
from database import db
from page_model import PageModel
import glob
import os 
import random

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
db.app = app

engine = create_engine(SQLALCHEMY_DATABASE_URI)

Session = sessionmaker(bind=engine, expire_on_commit=False)
session = Session()

image_list = glob.glob("./static/chinese/ims/*/*")
image_list.sort()
print len(image_list)
print os.getcwd()
print image_list

page_model = PageModel()

def update_page_with_random():
    page_ims = random.sample(range(len(image_list)), 3)

    page_model.main_img = page_ims[0]
    page_model.compare_img_1 = page_ims[1]
    page_model.compare_img_2 = page_ims[2]

    page_model.main_path = image_list[page_ims[0]]
    page_model.compare_1_path = image_list[page_ims[1]]
    page_model.compare_2_path = image_list[page_ims[2]]

random.seed()
update_page_with_random()

@app.route("/get_imgs")
def get_imgs():
    return jsonify(page_model.get_imgs_list()) 

@app.route("/get_response", methods = ['POST'])
def get_response():
    if request.method == 'POST':
        data = request.get_data()
        if data == "0":
            page_model.set_chosen(page_model.compare_img_1)
        elif data == "1":
            page_model.set_chosen(page_model.compare_img_2)
    make_transient(page_model)
    page_model.id = None
    session.add(page_model)
    session.commit()

    update_page_with_random()

    return jsonify(page_model.get_imgs_list())

@app.route("/")
def index():
    return render_template('test.html')
 
if __name__ == "__main__":
    app.run()