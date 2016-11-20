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
import time






# Choose a random triplet and set the page's triplet accordingly

def update_page_with_random():
    page_ims = random.sample(range(len(image_list)), 3)

    page_model.main_img = page_ims[0]
    page_model.compare_img_1 = page_ims[1]
    page_model.compare_img_2 = page_ims[2]

    page_model.main_path = image_list[page_ims[0]]
    page_model.compare_1_path = image_list[page_ims[1]]
    page_model.compare_2_path = image_list[page_ims[2]]






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



# Make image list

image_list = glob.glob("/home/cs101teaching/MachineTeaching/static/chinese/ims/*/*")
image_list.sort()

image_list = [img.replace("/home/cs101teaching/MachineTeaching", "") for img in image_list]



# Initialize page model
page_model = PageModel()



# Make dictionary to keep track of how many jobs each user has done
user_nclicks_dict = {}
user_id_dict = {}
user_time_dict = {}



# Set triplet

random.seed()
update_page_with_random()
counter = 0
max_clicks = 100



# Redirect user to login page

@app.route("/")
def to_login():
    return redirect(url_for('login_rand'))



# Return image list in JSON format 

@app.route("/get_imgs")
def get_imgs():
    return jsonify(page_model.get_imgs_list()) 

@app.route("/end/")
def logout():
    end_id = session['name']
    print 'end id'
    return render_template('end.html')


# THESE DO LOTS
#
# - Gets the response from the user
# - Updates the user's kernel

@app.route("/kernel/get_response", methods = ['POST'])
def get_response_kernel():
        
    if request.method == 'POST':
        data = request.get_data()
        if data == "0":
            page_model.set_chosen(page_model.compare_img_1)
        elif data == "1":
            page_model.set_chosen(page_model.compare_img_2)
        user_nclicks_dict[session['name']] += 1
        user_time_dict[session['name']][1] = time.time()

        if user_nclicks_dict[session['name']] == max_clicks: 
            end_id = session['name']
            user_id_dict[session['name']] = end_id
            # return redirect(url_for('logout'), code=307)
            return jsonify([url_for('logout'), 0])

    make_transient(page_model)
    page_model.id = None
    session_sql.add(page_model)
    session_sql.commit()

    update_page_with_random()
    return jsonify(page_model.get_imgs_list())



# Render main page

@app.route("/kernel/")
def kernel_index():
    return render_template('kernel.html')



# Create new user
#
# - Store given user name
# - Create initial kernel for new user

@app.route('/login', methods=['GET', 'POST'])
def login():
    global counter 
    error = None
    if request.method == 'POST':
        np.save('nclicks.npy', user_nclicks_dict)
        if request.form['username'] != '':
            session['name'] = request.form['username'] 
            user_nclicks_dict[session['name']] = 0
            return redirect(url_for('kernel_index'))
    return render_template('login.html', error=error)

@app.route('/login_rand', methods=['GET', 'POST'])
def login_rand(): 
    global counter 

    error = None
    if request.method == 'POST':
        np.save('nclicks.npy', user_nclicks_dict)
        np.save('time.npy', user_time_dict)
        if request.form['cont'] == "Continue": 
            return redirect(url_for('kernel_index'))
    elif request.method == 'GET':
        print(dict(request.args))
        if 'assignmentId' in request.args:
            session['name'] = request.args['assignmentId']
            session['assignmentId'] = request.args['assignmentId']
            session['hitId'] = request.args['hitId']
            session['turkSubmitTo'] = request.args['turkSubmitTo']
            session['workerId'] = request.args['workerId']
            print(dict(request.args))
        else:
            session['name'] = counter
            counter += 1
        user_nclicks_dict[session['name']] = 0
        user_time_dict[session['name']] = [time.time(), 0]
    return render_template('login_rand.html', error=error)


# Run
if __name__ == "__main__":
    app.run()
