from unicodedata import digit
from flask import Flask, render_template , request
from flask_wtf import Form
#from wtforms import TextField, IntegerField, TextAreaField, SubmitField, RadioField, SelectField
import pickle
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
from keras.models import load_model

import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()



def text_process(mess):

    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    no_stop_word=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(no_stop_word)




# loading our saved vectorizer

vec=pickle.load(open('vectorizer.sav','rb'))

# loading our saved deep learning model

deep_model=load_model('my_model.h5')

app=Flask(__name__)



@app.route('/')
def spam():
    return render_template('email_spam.html',pred_text='',email='')
 
@app.route('/spam_detector')

def spam_detector():
    return render_template('email_spam.html')

@app.route('/spam_detect' , methods=["POST","GET"])
def spam_detect():
    if request.method=="POST":
        msg=request.form["exampleFormControlTextarea1"]
        #print(msg)
        msg_processed=text_process(msg)
        msg_vec=vec.transform([msg_processed]).toarray()
        pred=deep_model.predict_classes(msg_vec)
        if pred==1:

            return render_template("email_spam.html" , pred_text="spam" , email=msg)
        else:

            return render_template("email_spam.html" , pred_text="ham" , email=msg)


@app.errorhandler(404)
def error(e):
    return render_template('404.html')


if __name__=="__main__":
    app.run(debug=True)
