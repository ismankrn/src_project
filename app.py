from flask import Flask, redirect, url_for, render_template, request, send_from_directory
import os
import pandas
from data_prep import *
from model_pred_sup import *
from model_pred_uns import *
import shutil

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/",methods=["GET", "POST"])
def home():

    if request.method=="POST":
        try:
            getfile = request.files['file_ord']
        except:
            return render_template("index.html", message="")

        if request.files['file_ord'].filename == '' :
            return render_template("index.html", message="Please upload the csv file")
        else:
            # save file
            file_ord = request.files["file_ord"]
            # destination = "/".join([target,file_ord.filename])
            file_ord.save("./uploads/{}".format(file_ord.filename))

            file_ord_prod = request.files["file_ord_prod"]
            # destination = "/".join([target,file_ord_prod.filename])
            file_ord_prod.save("./uploads/{}".format(file_ord_prod.filename))

            # data prep
            prep_data(file_ord.filename, file_ord_prod.filename)
            pred_sup(file_ord.filename)
            pred_uns(file_ord.filename)
            shutil.make_archive("outputs", 'zip', "./outputs")

            return render_template("index.html", message="Success")
    return render_template("index.html", message="")

@app.route('/download_file')
def download_file():
    filename = "outputs.zip"
    return send_from_directory(os.path.join(APP_ROOT), filename=filename, as_attachment=True)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(debug=True)
