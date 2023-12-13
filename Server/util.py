import json
import pickle

import numpy as np

__locations = None
__data_columns = None
__model = None


def get_location_names():
    return __locations

def get_estimated_price(location,sqft,bhk,bath):
    #model in which we will predict method
    try:
        loc_index=__data_columns.index(location)
    except:
        loc_index=-1

    #x which is input to model and it is two dimensions array so we will create np array with all zeroz like in notebook
    x=np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1

    return round(__model.predict([x])[0],2)
#first used routine
def load_saved_artifacts():
    print("Loading saved artifacts ... start")
    global __locations
    global __data_columns
    global __model
    # Read columns from json and return the list of locations
    with open('./artifacts/columns.json', 'r') as f:
        data = json.load(f)
        __data_columns = data.get('data_columns', [])

        # Starting from column 3 because the first 3 columns aren't location
        __locations = __data_columns[3:]

    with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)

    print("Loading saved artifacts ... done")
    print("__locations:", __locations)  # Add this line to print __locations


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000,3,3))
    print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
