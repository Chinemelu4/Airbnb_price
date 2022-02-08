
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor

st.write("""
# AIRBNB HOUSE PRICE PREDICT APP!
""")

model=pickle.load(open('model(xg).pkl', 'rb'))

scaler=pickle.load(open('Scal_func', 'rb'))


st.sidebar.header('User Input Parameters')

def user_input_features():
    Room_type= st.sidebar.selectbox('Room Type',('Private room','Entire home/apt','Shared room'))
    if Room_type=='Entire home/apt':
        Room_type=1
    else:
        Room_type=0 
        
    Region_hood= st.sidebar.selectbox('Region',('North Region','Central Region','East Region','West Region','North-East Region'))
    if Region_hood=='Central Region':
        Region_hood=1
    else:
        Region_hood=0

        
    Latitude = st.number_input('Latitude')
    Longitude = st.number_input('Longitude')
    Nights = st.number_input('Minimum Nights Stay')
    HL_count = st.number_input('Host Listing Count')
    Availability=st.number_input('No of Days Available')
    
    data = {'latitude':Latitude,
            'longitude': Longitude ,
            'minimum_nights':Nights,
            'calculated_host_listings_count': HL_count,
            'availability_365':Availability,
            'Entire home/apt':Room_type,
            'Central Region':Region_hood}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
input_df=scaler.transform(input_df)


if st.button('PREDICT'):
    y_out=model.predict(input_df)
    st.write(f' This room will cost you $',y_out[0])

