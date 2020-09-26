from flask import Flask, redirect, url_for, render_template, request, send_from_directory
import os
import pandas
from data_prep import *
from model_pred_sup import *
from model_pred_uns import *
import csv
import shutil

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/",methods=["GET", "POST"])
def home():
    target = os.path.join(APP_ROOT, 'uploads/')
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
            destination = "/".join([target,file_ord.filename])
            file_ord.save(destination)

            file_ord_prod = request.files["file_ord_prod"]
            destination = "/".join([target,file_ord_prod.filename])
            file_ord_prod.save(destination)

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

# @app.route("/",methods=["GET", "POST"])
# def build():
#     target = os.path.join(APP_ROOT, 'uploads/')
#
#     if request.method=="POST":
#         gen=gen=request.form['generate']
#         if gen=="0" :
#             try:
#                 getfile = request.files['file']
#             except:
#                 return render_template("build.html",selected="0", message="",output="-")
#
#             if request.files['file'].filename == '' :
#                 return render_template("build.html",message="Please choose CSV file")
#             else:
#                 file = request.files["file"]
#                 destination = "/".join([target,file.filename])
#                 file.save(destination)
#                 maxModel = count_clmn(destination)
#                 return render_template("build.html",selected="1", pass_name=destination, maxmodel=maxModel, file_name=file.filename)
#         elif gen=="1":
#             # Generate model
#             destination = request.form['destination']
#             csv_destination = destination.replace(".csv","")
#             bestModel = request.form['bestModel']
#             try:
#                 qsar_mlr.qsar_web(csv_destination,1,bestModel)
#                 output = readoverview()
#                 for i in range(int(bestModel)):
#                     qsar_mlr.qsar_web(csv_destination,2,'model_'+str(i+1))
#                     csv_pred = "/".join([APP_ROOT,'model_'+str(i+1)+'_pred'])
#                     createPlot.create(csv_pred,i)
#                 plotsrc = "/".join([APP_ROOT,'plot_'])
#                 # return plotsrc
#                 return render_template("build.html",message="Success",selected="0", output=output, n_model=int(bestModel))
#             except:
#                 return render_template("build.html",message="error",selected="0")
#     return render_template("build.html",selected="0", message="",output="-")
#
# @app.route("/predict",methods=["GET", "POST"])
# def prediction():
#     target = os.path.join(APP_ROOT, 'uploads/')
#
#     if request.method=="POST":
#         try:
#             request.files['fileCsv']
#         except:
#             return render_template("predict.html",messaeg="-")
#         if request.files['fileCsv'].filename == '' :
#                 return render_template("predict.html",message="Please choose CSV file")
#         else:
#             fileCsv = request.files["fileCsv"]
#             fileModel = request.files["fileModel"]
#             csv_destination = "/".join([target,fileCsv.filename])
#             modelDestination = "/".join([target,fileModel.filename])
#             fileCsv.save(csv_destination)
#             fileModel.save(modelDestination)
#             csv_destination = csv_destination.replace(".csv","")
#             model_Destination = str(fileModel.filename).replace(".p","")
#             try:
#                 qsar_mlr.qsar_web(csv_destination,2,model_Destination)
#                 resultname = (fileModel.filename).replace(".p","_pred.csv")
#                 head = readresulthead(resultname)
#                 return render_template("predict.html",message="Success",name=resultname,head=head)
#             except:
#                 return render_template("predict.html",message="error")
#     return render_template("predict.html",messaeg="-")
#
# @app.route("/methods",methods=["GET", "POST"])
# def methods():
#     return render_template("methods.html",messaeg="-")
#
# @app.route('/build/<string:filename>')
# def download_filse(filename):
#     # return folder
#     # try:
#     #     response = send_from_directory(os.path.join(APP_ROOT),
#     #                                    filename=filename)
#     #     response.cache_control.max_age = 60  # e.g. 1 minute
#     #     return response
#
#     # except:
#     #     return str("asd")
#     return send_from_directory(os.path.join(APP_ROOT),
#                                filename=filename, as_attachment=True)
#
# def readoverview():
#     with open("output.txt", "r") as input:
#         overview = input.read().split("\n\n\n")
#     return overview
#
# def readresulthead(name):
#     df = pandas.read_csv(name)
#     head = df.head().round(3)
#     actual = head['actual'].tolist()
#     prediction = head['prediction'].tolist()
#     residual = head['residual'].tolist()
#     data = [actual,prediction,residual]
#     return(data)
#
# def count_clmn(filename):
#     with open(filename) as csvfile:
#         readCSV = csv.reader(csvfile, delimiter=',')
#         for row in readCSV:
#             return(len(row))

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
