from flask import Flask, render_template, request
import pickle
import numpy as np

mymodel = pickle.load(open('mymodel.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_price():
    bedrooms = request.form.get('bedrooms')
    bathrooms = request.form.get('bathrooms')
    sqft_living = request.form.get('sqft_living')
    sqft_lot = request.form.get('sqft_lot')
    floors = request.form.get('floors')
    condition = request.form.get('condition')
    sqft_above = request.form.get('sqft_above')
    sqft_basement = request.form.get('sqft_basement')

    # converting data
    bedroom = int(bedrooms)
    bathroom = int(bathrooms)
    living = int(sqft_living)
    lot = int(sqft_lot)
    floor = int(floors)
    cond = int(condition)
    above = int(sqft_above)
    base = int(sqft_basement)

    # prediction
    input_data = np.array([bedroom, bathroom, living, lot, floor, cond, above, base]).reshape(1, -1)
    results = mymodel.predict(input_data)
    output = "{0:.{1}f}".format(results[0], 2)
    return render_template('index.html', pred='price of property is {}'.format(output) )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)