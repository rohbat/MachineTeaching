from flask import Flask, jsonify, request, render_template, redirect, url_for, session
# from sqlalchemy.orm.session import sessionmaker, make_transient
# from sqlalchemy import create_engine
# from database import db
from page_model_no_db import PageModel
import glob
import os 
import random
import numpy as np
from cython_tste.tste_next_point import *
import time
import hashlib
import sys

print 'Initializing site!'

def update_page_with_indices(main, comp1, comp2):
    # Switch order of images with probability .5
    if random.random() > .5:
        temp = comp1
        comp1 = comp2
        comp2 = temp

    page_model_dict[session['name']].main_img = main
    page_model_dict[session['name']].compare_img_1 = comp1
    page_model_dict[session['name']].compare_img_2 = comp2

    page_model_dict[session['name']].main_path = image_list[main]
    page_model_dict[session['name']].compare_1_path = image_list[comp1]
    page_model_dict[session['name']].compare_2_path = image_list[comp2]


# Choose a random triplet and set the page's triplet accordingly

def update_page(selection_method):
    train = range(N)
    X = user_x_dict[session['name']]
    lamb = 0

    if selection_method == 1:
        (main, comp1, comp2) = random_triplet(train, classes, classes_dict)
    elif selection_method == 2:
        ((main, comp1, comp2), p) = most_uncertain_triplet(train,X,N,no_dims,alpha,lamb,classes,classes_dict,eta,no_classes=3,sample_class = 0.135)
    elif selection_method == 3:
        ((main, comp1, comp2), p) = best_gradient_triplet(train,X,N,no_dims,alpha,lamb,classes,classes_dict,eta,no_classes=3,sample_class = 0.06)
    else:
        ((main, comp1, comp2), p) = best_gradient_triplet_rand_evaluation(train,X,N,no_dims,alpha,lamb,classes,classes_dict,eta,no_classes=3,sample_class = 0.025)

    update_page_with_indices(main, comp1, comp2)


def get_label_list():
    main_label = class_names[classes[page_model_dict[session['name']].main_img]]
    compare_img_1_label = class_names[classes[page_model_dict[session['name']].compare_img_1]]
    compare_img_2_label = class_names[classes[page_model_dict[session['name']].compare_img_2]]
    return (main_label, compare_img_1_label, compare_img_2_label)

def get_result_img(result):
    if result == True:
        return '/static/correct_check_mark.png'
    else:
        return '/static/incorrect_x_mark.png'

# Make Flask app

app = Flask(__name__)
#app.config["DEBUG"] = True




# # Set up database

# SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
#     username="cs101teaching2",
#     password="gogo_teaching",
#     hostname="cs101teaching2.mysql.pythonanywhere-services.com",
#     databasename="cs101teaching2$teaching",
# )
# app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
# app.config["SQLALCHEMY_POOL_RECYCLE"] = 200
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

# db.init_app(app)
# db.app = app

# engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_recycle=200)

# Session_sql = sessionmaker(bind=engine, expire_on_commit=False)
# session_sql = Session_sql()



# Set up classes and classes_dict
import os
path = os.getcwd()
class_names = glob.glob(path + "/MachineTeaching/static/chinese/ims/*")
class_names.sort()

print 'CLASS NAMES'
print class_names
class_name_dict = {}
for class_name in class_names:
    class_name_dict[class_name] = glob.glob(class_name + "/*")

name_class = {}
for k, v in class_name_dict.iteritems():
    for elm in v:
        name_class[elm] = k

image_list = glob.glob(path + "/MachineTeaching/static/chinese/ims/*/*")
image_list.sort()
N = len(image_list)

classes = np.zeros(len(image_list), dtype=int)
for i in range(len(image_list)):
    classes[i] = class_names.index(name_class[image_list[i]])


class_names = [c.replace(path + "/MachineTeaching/static/chinese/ims/", "") for c in class_names]

classes_dict = {}
for i in range(len(class_names)):
    classes_dict[i] = []

N_test = 30

for i in range(N):
    classes_dict[classes[i]].append(i)

for key in classes_dict.keys():
    classes_dict['not'+str(key)] = []
    for key1 in classes_dict.keys():
        if key != key1 and not 'not' in str(key1):
            classes_dict['not'+str(key)].extend(classes_dict[key1])

print classes, len(classes)
print classes_dict, len(classes_dict)



# Set values for tste

N = len(classes)
no_dims = 5
alpha = no_dims - 1.0
eta = 0.1



# Make image list

image_list = [img.replace(path + "/MachineTeaching", "") for img in image_list]



# Make dictionary to keep track of page models
page_model_dict = {}



# Make dictionary to keep track of how many jobs each user has done
user_nclicks_dict = {}
user_x_dict = {}
user_id_dict = {}
user_time_dict = {}
user_code_dict = {}
user_test_counter_dict = {}
user_selection_method_dict = {}
user_images_dict = {}
user_test_images_dict = {}
user_test_error_dict = {}
user_test_ans_dict = {}
user_test_time_dict = {}

# Set triplet

random.seed()
counter = 0
max_clicks = 2
max_test = 5


# Redirect user to login page

@app.route("/")
def to_login():
    return redirect(url_for('route'))

# Return image list in JSON format 

@app.route("/get_imgs", methods = ['GET', 'POST'])
def get_imgs():
    if user_nclicks_dict[session['name']] == max_clicks:
        user_test_images_dict[session['name']] = random.sample(set(range(N)) - user_images_dict[session['name']], N_test)
        user_test_counter_dict[session['name']] = 1
        user_test_error_dict[session['name']] = 0
        user_test_ans_dict[session['name']] = []
        user_test_time_dict[session['name']] = time.time()
        return jsonify([url_for('testing_index'), 0])

    update_page(user_selection_method_dict[session['name']])
    user_images_dict[session['name']].update(page_model_dict[session['name']].get_index_list())
    return jsonify(page_model_dict[session['name']].get_imgs_list() + [str(user_nclicks_dict[session['name']]), get_label_list()[0]]) 

@app.route("/end/")
def logout():
    if ('name' in session and session['name'] in user_nclicks_dict and 
        user_nclicks_dict[session['name']] == max_clicks and user_test_counter_dict[session['name']] == max_test):
            # end_id = session['name']
            print user_test_ans_dict 
            user_test_time_dict[session['name']] = time.time() - user_test_time_dict[session['name']] 
            user_test_error_dict[session['name']] = 1-float(np.sum(user_test_ans_dict[session['name']])) / float(max_test)
            end_id = hashlib.md5(str(session['name'])).hexdigest()
            print user_test_counter_dict
            print user_test_error_dict

            np.save('./testfiles/ans_dict.npy', user_test_ans_dict)
            np.save('./testfiles/error_dict.npy', user_test_error_dict)
            with open('./testfiles/' + str(session['name']) + "_test.txt", "w") as myfile:
                for item in user_test_ans_dict[session['name']]: 
                    myfile.write("%s," % item)
                myfile.write("\n")
                myfile.write(str(user_test_error_dict[session['name']]) + '\n')
                myfile.write(str(user_test_time_dict[session['name']]) + '\n')

            return render_template('end.html', end_id=end_id)
    else:
        return redirect(url_for('login'))



# THESE DO LOTS
#
# - Gets the response from the user
# - Updates the user's kernel

@app.route("/kernel/get_response", methods = ['POST'])
def get_response_kernel():
    if not 'name' in session or not session['name'] in user_nclicks_dict:
        return jsonify([url_for('login'), 0])
    if request.method == 'POST':
        data = request.get_data()
        (main_label, c1_label, c2_label) = get_label_list()
        if data == "0":
            page_model_dict[session['name']].set_chosen(page_model_dict[session['name']].compare_img_1)
            if (main_label == c1_label):
                result = True
            else:
                result = False
        elif data == "1":
            page_model_dict[session['name']].set_chosen(page_model_dict[session['name']].compare_img_2)
            if (main_label == c2_label):
                result = True
            else:
                result = False
        user_nclicks_dict[session['name']] += 1
        user_time_dict[session['name']][1] = time.time()
        K = np.zeros((N, N))
        Q = np.zeros((N, N))
        G = np.zeros((N, no_dims))
        tste_grad(user_x_dict[session['name']], N, no_dims, page_model_dict[session['name']].get_index_list(), 0, no_dims-1, K, Q, G)
        user_x_dict[session['name']] = user_x_dict[session['name']] - 0.4 * G
        #print user_x_dict[session['name']]

    # make_transient(page_model_dict[session['name']])
    # page_model_dict[session['name']].id = None
    # session_sql.add(page_model_dict[session['name']])
    # session_sql.commit()

    
    return jsonify([main_label, c1_label, c2_label, get_result_img(result)])


@app.route("/testing/get_response", methods = ['POST'])
def get_response_testing():
    if request.method == 'POST':
        # print 'clicked'
        data = request.get_data()
        # print 'test image list', user_test_images_dict[session['name']]
        # print 'test image index', user_test_counter_dict[session['name']]-1
        # print 'image index', user_test_images_dict[session['name']][user_test_counter_dict[session['name']]-1]
        truth = classes[user_test_images_dict[session['name']][user_test_counter_dict[session['name']]-1]]
        print 'data', data 
        print 'truth', truth 
        # print user_test_ans_dict
        # print user_test_counter_dict
        # print '\n'

        if str(truth) == str(data): 
            user_test_ans_dict[session['name']].append(1)
        else: 
            user_test_ans_dict[session['name']].append(0)
    if user_test_counter_dict[session['name']] == max_test: 
        return jsonify([url_for('logout'), 0])
    user_test_counter_dict[session['name']] += 1
    return jsonify([image_list[user_test_images_dict[session['name']][user_test_counter_dict[session['name']]-1]], str(user_test_counter_dict[session['name']]), 0]) 


# Test stuff
@app.route("/get_test_img", methods = ['GET', 'POST'])
def get_test_img():
    if user_test_counter_dict[session['name']] == max_test: 
        return jsonify([url_for('logout'), 0])
    # user_test_counter_dict[session['name']] += 1
    return jsonify([image_list[user_test_images_dict[session['name']][user_test_counter_dict[session['name']]-1]], str(user_test_counter_dict[session['name']]), 0]) 

# Render main page
@app.route("/teaching/")
def teaching_index():
    if not 'name' in session or not session['name'] in user_nclicks_dict:
        return redirect(url_for('login'))
    return render_template('teaching.html')

@app.route("/testing/")
def testing_index():
    if not 'name' in session or not session['name'] in user_test_counter_dict:
        return redirect(url_for('login'))
    print 'going to test'
    print user_test_ans_dict
    print user_test_counter_dict
    return render_template('test.html')


counter = 0
# Create new user
#
# - Store given user name
# - Create initial kernel for new user

urls = ['http://cs101teaching2.pythonanywhere.com/login', 
'http://machineteaching1.pythonanywhere.com/login', 
'http://machineteaching2.pythonanywhere.com/login', 
'http://machineteaching3.pythonanywhere.com/login', 
'http://machineteaching4.pythonanywhere.com/login', 
'http://machineteaching5.pythonanywhere.com/login']

@app.route('/login', methods=['GET', 'POST'])
def login():
    global counter
    error = None
    if request.method == 'GET':
        session['name'] = str(urls.index(request.url)) + "," + str(counter)
        print "Session name", session['name']
        counter += 1
    if request.method == 'POST' and request.form['cont'] == "Continue":
        np.save('nclicks_dict.npy', user_nclicks_dict)
        np.save('time_dict.npy', user_time_dict)
        np.save('code_dict.npy', user_code_dict)
        page_model_dict[session['name']] = PageModel()
        user_selection_method_dict[session['name']] = random.randint(1, 4)
        user_nclicks_dict[session['name']] = 0
        user_time_dict[session['name']] = [time.time(), 0]
        user_x_dict[session['name']] = np.load("MachineTeaching/static/X_initial.npy")
        user_images_dict[session['name']] = set([])

        print 'Selection Method: ' + str(user_selection_method_dict[session['name']])

        return redirect(url_for('teaching_index'))
    return render_template('login_rand.html', error=error)

routing_counter = 0
@app.route('/route', methods=['GET', 'POST'])
def route():
    global routing_counter
    routing_counter += 1
    return redirect(urls[routing_counter%len(urls)])


# Run
if __name__ == "__main__":
    app.run()
