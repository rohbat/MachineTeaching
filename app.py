import flask
from flask import Flask
from page_model import PageModel

app = Flask(__name__)

pm = PageModel()

@app.route("/get_response", methods=['POST'])
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

