# web
from flask import Flask, render_template, request, redirect, url_for
from my_wtforms import *

# ml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from ensembles import RandomForestMSE, GradientBoostingMSE

# global vars
X_train = None
y_train = None
X_val = None
y_val = None
model = None
conv = None
params = None
y_pred = None


app = Flask(__name__)
app.config['SECRET_KEY'] = 'tt'


@app.route("/", methods=['GET', 'POST'])
def upload_files():
    form = TrainValUpload()

    # if files are submitted
    if request.method == 'POST' and form.validate_on_submit():
        global X_train, y_train, X_val, y_val
        # load datasets
        train = pd.read_csv(form.train.data)
        val = None
        if form.val.data:
            val = pd.read_csv(form.val.data)
        
        # ensure it has only numerical values
        if not train.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all():
            return render_template("index.html", form=form, non_numeric=True)

        # make samples
        X_train = train.drop(columns=[form.target_col.data]).to_numpy()
        y_train = train[form.target_col.data].to_numpy()

        if val is not None:
            X_val = val.drop(columns=[form.target_col.data]).to_numpy()
            y_val = val[form.target_col.data].to_numpy()
        
        return render_template('choose-model.html')

    # otherwise load page
    return render_template("index.html", form=form, non_numeric=False)


@app.route("/set-rf-params", methods=['GET', 'POST'])
def set_rf_params():
    form = SetRFParams()

    if request.method == 'POST' and form.validate_on_submit():
        global model, conv, params

        # read parameters
        params = {
            'n_estimators': form.n_estimators.data,
            'max_depth': form.max_depth.data,
            'subspace_size': form.subspace_size.data
        }

        # fit model
        model = RandomForestMSE(**params)
        conv = model.fit(X_train, y_train, X_val, y_val)

        # make and save plot
        fig, ax = plt.subplots(1, figsize=(4,5))

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: '{:.0f}'.format(x/1000)))

        ax.plot(conv['train'], label='train', linewidth=3, alpha=0.75)
        if X_val is not None and y_val is not None:
            ax.plot(conv['val'], label='val', linewidth=3, alpha=0.75)
        ax.legend()

        ax.set_xlabel('# iterarions')
        ax.set_ylabel(r'RMSE, $10^3$')

        plt.savefig('src/static/plot.svg', bbox_inches='tight')

        return redirect(url_for("learning_results"))

    return render_template("set-rf-params.html", form=form)

@app.route("/set-gb-params", methods=['GET', 'POST'])
def set_gb_params():
    form = SetGBParams()

    if request.method == 'POST' and form.validate_on_submit():
        global model, conv, params

        # read parameters
        params = {
            'n_estimators': form.n_estimators.data,
            'max_depth': form.max_depth.data,
            'subspace_size': form.subspace_size.data,
            'learning_rate': form.learning_rate.data
        }

        # fit model
        model = GradientBoostingMSE(**params)
        conv = model.fit(X_train, y_train, X_val, y_val)

        # make and save plot
        fig, ax = plt.subplots(1, figsize=(4,5))

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: '{:.0f}'.format(x/1000)))

        ax.plot(conv['train'], label='train', linewidth=3, alpha=0.75)
        if X_val is not None and y_val is not None:
            ax.plot(conv['val'], label='val', linewidth=3, alpha=0.75)
        ax.legend()

        ax.set_xlabel('# iterarions')
        ax.set_ylabel(r'RMSE, $10^3$')

        plt.savefig('src/static/plot.svg', bbox_inches='tight')

        return redirect(url_for("learning_results"))

    return render_template("set-gb-params.html", form=form)


@app.route("/learning-results", methods=['GET', 'POST'])
def learning_results():
    return render_template('learning-results.html', params=params)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = Predict()

    # if files are submitted
    if request.method == 'POST' and form.validate_on_submit():
        global y_pred
        # load datasets
        X = pd.read_csv(form.X.data)
        
        # ensure it has only numerical values
        if not X.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all():
            return render_template("predict.html", form=form, non_numeric=True)

        # make sample
        X = X.to_numpy()
        
        # make prediction
        y_pred = model.predict(X)
        np.savetxt("src/static/predictions.csv", y_pred, delimiter=",")

        return render_template('predictions.html', data=zip(np.arange(y_pred.size), y_pred))

    # otherwise load page
    return render_template("predict.html", form=form, non_numeric=False)

if __name__ == "__main__":
    app.run(debug=True)
