from flask import Flask, jsonify, request, render_template
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy import create_engine
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
db.app = app

engine = create_engine(SQLALCHEMY_DATABASE_URI)

Session = sessionmaker(bind=engine, expire_on_commit=False)
session = Session()

page_model = PageModel("profile1.jpg", "profile2.jpg", "profile3.jpg")

#print 'adding initial pagemodel to session'
session.add(page_model)
#print 'commiting initial session'
session.commit()
#print 'commited initial session'

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
    
    #print 'adding pagemodel to session on click'
    session.add(page_model)
    #print 'commiting pagemodel to session on click'
    session.commit()
    #print 'session commited on click'
    return jsonify(page_model.get_imgs_list())

@app.route("/")
def index():
    return render_template('test.html')
 
if __name__ == "__main__":
    app.run()