from flask import Flask, jsonify, request, render_template
#from database import db
from sqlalchemy import Column, Integer, String
from flask_sqlalchemy import SQLAlchemy
#from page_model import PageModel

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

#db.init_app(app)
#db.app = app
db = SQLAlchemy(app)

class PageModel(db.Model):
    __tablename__ = 'triplets'
    id = Column(Integer, primary_key=True)
    main_img = Column(String(100))
    compare_img_1 = Column(String(100))
    compare_img_2 = Column(String(100))
    chosen = Column(String(100))

    def __init__(self, main_img, compare_img_1, compare_img_2):
        self.main_img = main_img
        self.compare_img_1 = compare_img_1
        self.compare_img_2 = compare_img_2
        self.chosen = ''

    def set_chosen(self, img):
        self.chosen = img

page_model = PageModel("profile1.jpg", "profile2.jpg", "profile3.jpg")
main_img = page_model.main_img
compare_img_1 = page_model.compare_img_1
compare_img_2 = page_model.compare_img_2

def get_imgs_list(page_model):
    return [main_img, compare_img_1, compare_img_2]

#print 'adding initial pagemodel to session'
db.session.add(page_model)
#print 'commiting initial session'
db.session.commit()
#print 'commited initial session'

@app.route("/get_imgs")
def get_imgs():
    return jsonify(get_imgs_list(page_model)) 

@app.route("/get_response", methods = ['POST'])
def get_response():
    if request.method == 'POST':
        data = request.get_data()
        print data
    if data == "0":
        page_model.main_img, page_model.compare_img_1 = page_model.compare_img_1, page_model.main_img
    elif data == "1":
        page_model.main_img, page_model.compare_img_2 = page_model.compare_img_2, page_model.main_img 
    
    #print 'adding pagemodel to session on click'
    db.session.add(page_model)
    #print 'commiting pagemodel to session on click'
    db.session.commit()
    #print 'session commited on click'
    return jsonify(get_imgs_list(page_model))

@app.route("/")
def index():
    return render_template('test.html')
 
if __name__ == "__main__":
    app.run()