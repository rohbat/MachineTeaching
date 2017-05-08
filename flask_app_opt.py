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

    page_model_dict_oct[session['name']].main_img = main
    page_model_dict_oct[session['name']].compare_img_1 = comp1
    page_model_dict_oct[session['name']].compare_img_2 = comp2

    page_model_dict_oct[session['name']].main_path = image_list[main]
    page_model_dict_oct[session['name']].compare_1_path = image_list[comp1]
    page_model_dict_oct[session['name']].compare_2_path = image_list[comp2]



# Choose a random triplet and set the page's triplet accordingly

def update_page(selection_method):
    train = range(N)
    X = user_x_dict_oct[session['name']]
    lamb = 0

    if selection_method == 1:
        (main, comp1, comp2) = random_triplet(train, classes, classes_dict_oct)
    elif selection_method == 2:
        ((main, comp1, comp2), p) = most_uncertain_triplet(train,X,N,no_dims,alpha,lamb,classes,classes_dict_oct,eta,no_classes=3,sample_class = 0.135)
    elif selection_method == 3:
        ((main, comp1, comp2), p) = best_gradient_triplet(train,X,N,no_dims,alpha,lamb,classes,classes_dict_oct,eta,no_classes=3,sample_class = 0.06)
    else:
        ((main, comp1, comp2), p) = best_gradient_triplet_rand_evaluation(train,X,N,no_dims,alpha,lamb,classes,classes_dict_oct,eta,no_classes=3,sample_class = 0.025)

    update_page_with_indices(main, comp1, comp2)


def update_test(): 
    (main, comp1, comp2) = user_test_images_dict_oct[session['name']][user_test_counter_dict_oct[session['name']]-1]
    # print user_test_counter_dict_oct[session['name']], (main, comp1, comp2)
    update_page_with_indices(main, comp1, comp2)



def get_label_list():
    print page_model_dict_oct[session['name']].get_imgs_list()
    main_label = class_names[classes[page_model_dict_oct[session['name']].main_img]]
    compare_img_1_label = class_names[classes[page_model_dict_oct[session['name']].compare_img_1]]
    compare_img_2_label = class_names[classes[page_model_dict_oct[session['name']].compare_img_2]]
    return (main_label, compare_img_1_label, compare_img_2_label)


def get_result_img(result):
    if result == True:
        return '/static/correct_check_mark.png'
    else:
        return '/static/incorrect_x_mark.png'

def get_result_text(result): 
    if result == True: 
        return 'CORRECT! '
    else: 
        return 'INCORRECT! '

def get_result_color(result): 
    if result == True: 
        return '#1E90FF'
    else: 
        return 'red'

def get_test_images(selection_method): 
    result = []
    train = range(N)

    for i in range(N_test): 
        result.append(random_triplet(train, classes, classes_dict_oct))
    # temp = np.random.choice(range(len(difficult_all)), diff_test)

    
    # result = result + [difficult_all[i] for i in temp]

    # print 'result', result
    return result


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



# Set up classes and classes_dict_oct
import os
path = os.getcwd()
class_names = glob.glob(path + "/MachineTeaching/static/oct/data_resized/*")
class_names.sort()

class_name_dict_oct = {}
for class_name in class_names:
    class_name_dict_oct[class_name] = glob.glob(class_name + "/*")

name_class = {}
for k, v in class_name_dict_oct.iteritems():
    for elm in v:
        name_class[elm] = k

image_list = glob.glob(path + "/MachineTeaching/static/oct/data_resized/*/*")
image_list.sort()
N = len(image_list)

classes = np.zeros(len(image_list), dtype=int)
for i in range(len(image_list)):
    classes[i] = class_names.index(name_class[image_list[i]])


class_names = [c.replace(path + "/MachineTeaching/static/oct/data_resized/", "") for c in class_names]
# print 'class names: ', class_names

classes_dict_oct = {}
for i in range(len(class_names)):
    classes_dict_oct[i] = []

N_test = 15

for i in range(N):
    classes_dict_oct[classes[i]].append(i)

for key in classes_dict_oct.keys():
    classes_dict_oct['not'+str(key)] = []
    for key1 in classes_dict_oct.keys():
        if key != key1 and not 'not' in str(key1):
            classes_dict_oct['not'+str(key)].extend(classes_dict_oct[key1])

# print classes, len(classes)
# print classes_dict_oct, len(classes_dict_oct)



# Set values for tste

N = len(classes)
no_dims = 5
alpha = no_dims - 1.0
eta = 0.1



# Make image list

image_list = [img.replace(path + "/MachineTeaching", "") for img in image_list]



# Make dictionary to keep track of page models
page_model_dict_oct = {}



# Make dictionary to keep track of how many jobs each user has done
user_nclicks_dict_oct = {}
user_x_dict_oct = {}
user_id_dict_oct = {}
user_time_dict_oct = {}
user_code_dict_oct = {}
user_test_counter_dict_oct = {}
user_selection_method_dict_oct = {}
user_images_dict_oct = {}
user_test_images_dict_oct = {}
user_test_error_dict_oct = {}
user_test_ans_dict_oct = {}
user_test_time_dict_oct = {}
user_train_ans_dict_oct = {}

# Set triplet

random.seed()
counter = 0
max_clicks = 15
diff_test = 5
max_test = N_test + diff_test

difficult_triplets_distance = [(147, 142, 451), (30, 59, 619), (261, 288, 37), \
(452, 288, 536), (488, 701, 43), (539, 701, 376)]
difficult_triplets_acc = [(95, 216, 360), (95, 216, 607), (120, 81, 95), \
(120, 81, 607), (127, 69, 95), (127, 69, 360)]
difficult_triplets_tste = [(648, 617, 81), (612, 497, 392), (644, 666, 346), \
(701, 599, 7), (520, 664, 150), (115, 84, 708), (99, 53, 284), (561, 610, 289), \
(18, 159, 302), (26, 98, 641)] 

# difficult_all = []

# for i in range(diff_test): 
#     difficult_all.append((np.random.choice(image_list), np.random.choice(image_list), np.random.choice(image_list)))

# difficult_all = difficult_triplets_distance + difficult_triplets_acc + difficult_triplets_tste
# print 'difficult_all', difficult_all

# Redirect user to login page

@app.route("/")
def to_login():
    return redirect(url_for('route'))

# Return image list in JSON format 

@app.route("/get_imgs", methods = ['GET', 'POST'])
def get_imgs():
    # print 'nclicks', user_nclicks_dict_oct
    if user_nclicks_dict_oct[session['name']] == max_clicks:
        # user_test_images_dict_oct[session['name']] = random.sample(set(range(N)) - user_images_dict_oct[session['name']], N_test)

        user_test_images_dict_oct[session['name']] = get_test_images('random')
        user_test_counter_dict_oct[session['name']] = 1
        user_test_error_dict_oct[session['name']] = 0
        user_test_ans_dict_oct[session['name']] = []
        user_test_time_dict_oct[session['name']] = time.time()
        return jsonify([url_for('testing_index'), 0])

    update_page(user_selection_method_dict_oct[session['name']])
    user_images_dict_oct[session['name']].update(page_model_dict_oct[session['name']].get_index_list())
    return jsonify(page_model_dict_oct[session['name']].get_imgs_list() + [str(user_nclicks_dict_oct[session['name']]), get_label_list()[0]]) 


# # Test stuff
# @app.route("/get_test_img", methods = ['GET', 'POST'])
# def get_test_img():
#     if user_test_counter_dict_oct[session['name']] == max_test: 
#         return jsonify([url_for('logout'), 0])
#     # user_test_counter_dict_oct[session['name']] += 1
#     return jsonify([image_list[user_test_images_dict_oct[session['name']]\
#         [user_test_counter_dict_oct[session['name']]-1]], \
#         str(user_test_counter_dict_oct[session['name']]), 0]) 


@app.route("/testing/get_imgs", methods = ['GET', 'POST'])
def testing_get_imgs():
    if user_test_counter_dict_oct[session['name']] == max_test: 
        return jsonify([url_for('logout'), 0])

   
    update_test() 
    return jsonify(page_model_dict_oct[session['name']].get_imgs_list() \
        + [str(user_test_counter_dict_oct[session['name']])]) 



# THESE DO LOTS
#
# - Gets the response from the user
# - Updates the user's kernel

@app.route("/kernel/get_response", methods = ['POST'])
def get_response_kernel():
    if not 'name' in session or not session['name'] in user_nclicks_dict_oct:
        return jsonify([url_for('login'), 0])
    if request.method == 'POST':
        data = request.get_data()
        (main_label, c1_label, c2_label) = get_label_list()
        if data == "0":
            page_model_dict_oct[session['name']].set_chosen(page_model_dict_oct[session['name']].compare_img_1)
            if (main_label == c1_label):
                result = True
            else:
                result = False
        elif data == "1":
            page_model_dict_oct[session['name']].set_chosen(page_model_dict_oct[session['name']].compare_img_2)
            if (main_label == c2_label):
                result = True
            else:
                result = False
        # print 'RESULT: ', result
        user_train_ans_dict_oct[session['name']].append(result)
        # print 'TRAIN ANS: ', user_train_ans_dict_oct[session['name']]
        user_nclicks_dict_oct[session['name']] += 1
        user_time_dict_oct[session['name']][1] = time.time()
        K = np.zeros((N, N))
        Q = np.zeros((N, N))
        G = np.zeros((N, no_dims))
        tste_grad(user_x_dict_oct[session['name']], N, no_dims, page_model_dict_oct[session['name']].get_index_list(), 0, no_dims-1.0, K, Q, G)
        user_x_dict_oct[session['name']] = user_x_dict_oct[session['name']] - 0.4 * G
        #print user_x_dict_oct[session['name']]

    # make_transient(page_model_dict_oct[session['name']])
    # page_model_dict_oct[session['name']].id = None
    # session_sql.add(page_model_dict_oct[session['name']])
    # session_sql.commit()

    if main_label == c1_label: 
        c1_border = '5px solid #1E90FF'
        c2_border = '0px solid #1E90FF'
    elif main_label == c2_label: 
        c1_border = '0px solid #1E90FF'
        c2_border = '5px solid #1E90FF'

    
    return jsonify([main_label, c1_label, c2_label, get_result_text(result), get_result_color(result), c1_border, c2_border])
    # return jsonify([main_label, c1_label, c2_label, get_result_img(result)])


@app.route("/testing/get_response", methods = ['POST'])
def get_response_testing():
    if request.method == 'POST':
        # print 'clicked'
        data = request.get_data()
        (main_label, c1_label, c2_label) = get_label_list()

        if data == '0': 
            if main_label == c1_label: 
                result = True
            else:
                result = False
        elif data == '1': 
            if main_label == c2_label: 
                result = True
            else: 
                result = False

        # print 'result: ', result

        user_test_ans_dict_oct[session['name']].append(result)

    if user_test_counter_dict_oct[session['name']] == max_test: 
        return jsonify([url_for('logout'), 0])

    user_test_counter_dict_oct[session['name']] += 1
    update_test() 
    
    return jsonify(page_model_dict_oct[session['name']].get_imgs_list()  +\
     [str(user_test_counter_dict_oct[session['name']])]) 



# Render main page
@app.route("/teaching/")
def teaching_index():
    if not 'name' in session or not session['name'] in user_nclicks_dict_oct:
        return redirect(url_for('login'))
    return render_template('teaching_triplet_chinese.html')

@app.route("/testing/")
def testing_index():
    if not 'name' in session or not session['name'] in user_test_counter_dict_oct:
        return redirect(url_for('login'))
    return render_template('test_triplet_chinese.html')


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
        np.save('nclicks_dict_oct.npy', user_nclicks_dict_oct)
        np.save('time_dict_oct.npy', user_time_dict_oct)
        np.save('code_dict_oct.npy', user_code_dict_oct)
        page_model_dict_oct[session['name']] = PageModel()
        user_selection_method_dict_oct[session['name']] = random.randint(1, 4)
        user_nclicks_dict_oct[session['name']] = 0
        user_time_dict_oct[session['name']] = [time.time(), 0]
        user_train_ans_dict_oct[session['name']] = []
        user_x_dict_oct[session['name']] = np.load("MachineTeaching/static/X_initial_opt.npy")
        #user_x_dict_oct[session['name']] = np.random.rand(N, no_dims)
        user_images_dict_oct[session['name']] = set([])

        # print 'Selection Method: ' + str(user_selection_method_dict_oct[session['name']])

        return redirect(url_for('teaching_index'))
    return render_template('login_instr_triplet.html', error=error)



@app.route("/end/")
def logout():
    if ('name' in session and session['name'] in user_nclicks_dict_oct and 
        user_nclicks_dict_oct[session['name']] == max_clicks and user_test_counter_dict_oct[session['name']] == max_test):
            
            user_test_time_dict_oct[session['name']] = time.time() - user_test_time_dict_oct[session['name']] 
            user_test_error_dict_oct[session['name']] = 1-float(np.sum(user_test_ans_dict_oct[session['name']])) / float(max_test)
            end_id = hashlib.md5(str(session['name'])).hexdigest()

            # print 'name: ', session['name'], '\n'
            # print 'method: ', user_selection_method_dict_oct[session['name']], '\n'
            # print 'error: ', user_test_error_dict_oct[session['name']], '\n'

            np.save('./testfiles_oct/user_x_dict_oct.npy', user_x_dict_oct)
            np.save('./testfiles_oct/ans_dict_oct.npy', user_test_ans_dict_oct)
            np.save('./testfiles_oct/test_images_dict_oct.npy', user_test_images_dict_oct)
            np.save('./testfiles_oct/error_dict_oct.npy', user_test_error_dict_oct)
            np.save('./testfiles_oct/method_dict_oct.npy', user_selection_method_dict_oct)
            np.save('./testfiles_oct/train_ans_dict_oct.npy', user_train_ans_dict_oct)
            with open('./testfiles_oct/' + str(session['name']) + "_test.txt", "w") as myfile:
                for item in user_test_ans_dict_oct[session['name']]: 
                    myfile.write("%s," % item)
                myfile.write("\n")
                myfile.write(str(user_test_error_dict_oct[session['name']]) + '\n')
                myfile.write(str(user_test_time_dict_oct[session['name']]) + '\n')
                myfile.write(str(user_selection_method_dict_oct[session['name']]) + '\n')

            return render_template('end.html', end_id=end_id)
    else:
        return redirect(url_for('login'))



routing_counter = 0
@app.route('/route', methods=['GET', 'POST'])
def route():
    global routing_counter
    routing_counter += 1
    return redirect(urls[routing_counter%len(urls)])


# Run
if __name__ == "__main__":
    app.run()
