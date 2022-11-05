from flask import Flask,render_template, url_for ,flash , redirect, Markup
from flask import request
from flask import send_from_directory
from requests_html import HTMLSession
from utils.fertilizer import fertilizer_dict
from utils.disease import disease_dic
import requests
import pandas as pd
import pickle
import tensorflow
import numpy as np
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from crop_predict import Crop_Predict
from market_stat import Market
from weather import Weather
import requests
import bs4
from keras.models import load_model
import os
from tensorflow.keras.utils import load_img, img_to_array


classifier = load_model('Trained_model.h5')
classifier.make_predict_function()





s = HTMLSession()
app=Flask(__name__,template_folder='template')
# app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

model = load_model("Trained.h5")


crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)



@app.route("/")
def home():
    return render_template("home.html")

@app.route("/Complaint")
def Complaint():
    return render_template("complaint.html")

@app.route("/Fertilizer_recommend")
def Fertilizer_recommend():
    return render_template("fert_form.html")

@app.route("/Crop_recommend")
def Crop_recommend():
    return render_template("crop_recommend.html")

@app.route("/Donate")
def Donate():
    return render_template("donate.html")

@app.route("/rain_output")
def rain_output():
    return render_template("rain_output.html")

@app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Crop_NPK.csv')

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,recommendation2=response2, recommendation3=response3,diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)



@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
    return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/'+final_prediction+'.jpg')
    

@app.route("/weather")
def weather():
    return render_template("weather.html")


@app.route("/crop-predict")
def crop_predict():
    return render_template("crop_prediction.html")



@app.route('/crop',methods=['GET','POST'])
def crop():

    model = Crop_Predict()
    if request.method == 'POST':
        crop_name = model.crop()

        return render_template('crop_prediction_result.html',crops=crop_name,crop_num = len(crop_name),display=True)
    return render_template("crop_prediction.html")

@app.route('/market',methods=['POST','GET'])
def market():

    model = Market()
    states,crops = model.State_Crop()
    if request.method == 'POST':
        state = request.form['state']
        crop = request.form['crop']
        lt = model.predict_data(state,crop)

        return render_template('market.html',result=lt,result_len =len(lt),display=True,states=states,crops=crops)

    return render_template('market.html',states=states,crops=crops) 

@app.route("/d",methods=['POST'])
def d():

    if request.method == "POST":
        x = request.form.to_dict()
        data = x['location']
        query = data
        res = requests.get('https://www.timeanddate.com/weather/india/'+query+'/ext')
        data = bs4.BeautifulSoup(res.text,'lxml')
        temp = data.find_all('table')
        for i in temp:
            res = i
        res = Markup(str(res))
        return render_template("d.html",res = res)
    
@app.route("/PesticideRecommendation")
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/dp", methods=['GET', 'POST'])
def dp():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename
        file_path = os.path.join(r'static\user uploaded', filename)
        file.save(file_path)
        try:
            test_image = load_img(file_path, target_size=(150, 150))
            test_image = img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)
            pred_arr = []
            for i in result:
                pred_arr.extend(i)
            
            pred = pred_arr.index(1.0)
            
        except:
            pred =  'x'
        
        if pred == 'x':
            return render_template('unaptfile.html')
        if pred == 0:
            pest_identified = 'aphids'
        elif pred == 1:
            pest_identified = 'armyworm'
        elif pred == 2:
            pest_identified = 'beetle'
        elif pred == 3:
            pest_identified = 'bollworm'
        elif pred == 4:
            pest_identified = 'earthworm'
        elif pred == 5:
            pest_identified = 'grasshopper'
        elif pred == 6:
            pest_identified = 'mites'
        elif pred == 7:
            pest_identified = 'mosquito'
        elif pred == 8:
            pest_identified = 'sawfly'
        elif pred == 9:
            pest_identified = 'stem borer'

        return render_template(pest_identified + ".html",pred=pest_identified)



@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename
        file_path = os.path.join(r'static\user uploaded', filename)
        file.save(file_path)
        try:
            test_image = load_img(file_path, target_size=(64, 64))
            test_image = img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = (classifier.predict(test_image) > 0.5).astype("int32")
            pred_arr = []
            for i in result:
                pred_arr.extend(i)
                
            index = pred_arr.index(1)
            pred = index
            
        except:
            pred =  'x'
        
        if pred == 'x':
            return render_template('unaptfile.html')
        if pred == 0:
            pest_identified = 'aphids'
        elif pred == 1:
            pest_identified = 'armyworm'
        elif pred == 2:
            pest_identified = 'beetle'
        elif pred == 3:
            pest_identified = 'bollworm'
        elif pred == 4:
            pest_identified = 'earthworm'
        elif pred == 5:
            pest_identified = 'grasshopper'
        elif pred == 6:
            pest_identified = 'mites'
        elif pred == 7:
            pest_identified = 'mosquito'
        elif pred == 8:
            pest_identified = 'sawfly'
        elif pred == 9:
            pest_identified = 'stem borer'

        return render_template(pest_identified + ".html",pred=pest_identified)

if __name__ == "__main__":
    app.run(debug=True)