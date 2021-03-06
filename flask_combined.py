from flask import Flask, jsonify, request, render_template, redirect, url_for, session
from sqlalchemy.orm.session import sessionmaker, make_transient
from sqlalchemy import create_engine
from database import db
from page_model import PageModel
import glob
import os 
import random
import numpy as np
from cython_tste.tste_next_point import *



# Make Flask app

app = Flask(__name__)
app.config["DEBUG"] = True



# Set up database

SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
    username="cs101teaching",
    password="gogo_teaching",
    hostname="cs101teaching.mysql.pythonanywhere-services.com",
    databasename="cs101teaching$teaching",
)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_POOL_RECYCLE"] = 299
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

db.init_app(app)
db.app = app

engine = create_engine(SQLALCHEMY_DATABASE_URI)

Session_sql = sessionmaker(bind=engine, expire_on_commit=False)
session_sql = Session_sql()



# WHAT AM I DOING?

class_names = glob.glob("/home/cs101teaching/MachineTeaching/static/chinese/ims/*")

class_name_dict = {}
for class_name in class_names:
    class_name_dict[class_name] = glob.glob(class_name + "/*")

name_class = {}
for k, v in class_name_dict.iteritems():
    for elm in v:
        name_class[elm] = k

image_list = glob.glob("/home/cs101teaching/MachineTeaching/static/chinese/ims/*/*")
image_list.sort()

classes = np.zeros(len(image_list))
for i in range(len(image_list)):
    classes[i] = class_names.index(name_class[image_list[i]])

classes_dict = {}
for i in range(len(class_names)):
    classes_dict[i] = []
for i in range(len(classes)):
    classes_dict[classes[i]].append(i)

print classes, len(classes)
print classes_dict, len(classes_dict)



# Set values for tste

N = len(classes)
no_dims = 10
alpha = no_dims - 1
eta = 0.01



# WHAT AM I DOING?

image_list = [img.replace("/home/cs101teaching/MachineTeaching", "") for img in image_list]



# Initialize page model

page_model = PageModel()



# Make dictionaries
#
#  - user_x_dict stores the users' kernels
#  - user_nclicks_dict stores the number of jobs each user has done

user_x_dict = {}
user_nclicks_dict = {}



# Choose a random triplet and set the page's triplet accordingly

def update_page_with_random():
    page_ims = random.sample(range(len(image_list)), 3)

    page_model.main_img = page_ims[0]
    page_model.compare_img_1 = page_ims[1]
    page_model.compare_img_2 = page_ims[2]

    page_model.main_path = image_list[page_ims[0]]
    page_model.compare_1_path = image_list[page_ims[1]]
    page_model.compare_2_path = image_list[page_ims[2]]



# Set triplet

random.seed()
update_page_with_random()

@app.route("/")
def to_login():
    return redirect(url_for('login'))



# Return image list in JSON format 

@app.route("/get_imgs")
def get_imgs():
    return jsonify(page_model.get_imgs_list()) 



# THESE DO LOTS
#
# - Gets the response from the user
# - Updates the user's kernel

@app.route("/teaching/get_response", methods = ['POST'])
def get_response():
    if request.method == 'POST':
        data = request.get_data()
        if data == "0":
            page_model.set_chosen(page_model.compare_img_1)
        elif data == "1":
            page_model.set_chosen(page_model.compare_img_2)
        user_nclicks_dict[session['name']] += 1
        K = np.zeros((N, N))
        Q = np.zeros((N, N))
        G = np.zeros((N, no_dims))
        tste_grad(user_x_dict[session['name']], N, no_dims, page_model.get_index_list(), 0, no_dims-1, K, Q, G)
        user_x_dict[session['name']] = user_x_dict[session['name']] - (float(eta) / N) * G
    make_transient(page_model)
    page_model.id = None
    session_sql.add(page_model)
    session_sql.commit()

    update_page_with_random()
    return jsonify(page_model.get_imgs_list())

@app.route("/kernel/get_response", methods = ['POST'])
def get_response_kernel():
    if request.method == 'POST':
        data = request.get_data()
        if data == "0":
            page_model.set_chosen(page_model.compare_img_1)
        elif data == "1":
            page_model.set_chosen(page_model.compare_img_2)
        user_nclicks_dict[session['name']] += 1

    make_transient(page_model)
    page_model.id = None
    session_sql.add(page_model)
    session_sql.commit()
    print user_nclicks_dict

    update_page_with_random()
    return jsonify(page_model.get_imgs_list())



# Render main page

@app.route("/teaching/")
def index():
    return render_template('test.html')

@app.route("/kernel/")
def kernel_index():
    return render_template('kernel.html')



# Create new user
#
# - Store given user name
# - Create initial kernel for new user

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        np.save('nclicks.npy', user_nclicks_dict)
        if request.form['username'] != '':
            session['name'] = request.form['username']
            user_x_dict[session['name']] = np.random.rand(N, no_dims)
            user_nclicks_dict[session['name']] = 0
            return redirect(url_for('kernel_index'))
    return render_template('login.html', error=error)



# Run
 
if __name__ == "__main__":
    app.run()



