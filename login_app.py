
from flask import Flask, render_template, redirect, url_for, request, session


app = Flask(__name__)
app.config["DEBUG"] = True

app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

# route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != '':
            session['cookie-parameter'] = request.form['username']
            return redirect(url_for('index'))
    return render_template('login.html', error=error)
    # return redirect(url_for('index'))


if __name__ == "__main__":
    app.run()
