import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import pickle
from transform import process_df


# path_to_repo = Path(__file__).parent.resolve()
# path_to_data = path_to_repo / 'data' / 'transformed'
path_to_data = 'fraud_oracle.csv'

def display_bulldozer_img(index):
    st.subheader('Whether a Vechile Insurance Claim is Fraud ')
    empty1, col, empty2 = st.columns([0.25, 0.5, 0.25])
    #ind = index % len(st.session_state.imgs)
    img = 'vehicle.png'
    col.image(img, use_column_width = 'always')
    return


def display_bulldozer_price(index):
    # compute model prediction
    pred_price = st.session_state.model.predict([st.session_state.X.values[index]])[0]
    true_price = st.session_state.y.iloc[index]
    
    pred_price = 'Yes' if pred_price==1 else 'No'
    true_price = 'Yes' if pred_price==1 else 'No'
    
    col_pred, col_gold = st.columns(2)
    with col_pred:
        st.subheader('Estimated Result')
        st.write(pred_price)
    with col_gold:
        st.subheader('True Result')
        st.write(true_price)
    return


def display_bulldozer_features(index):
    st.subheader('Bulldozer features')
    feat0, val0, feat1, val1 = st.columns([3.5, 1.5, 3.5, 1.5])
    row = st.session_state.X.values[index]
    for i, feature in enumerate(st.session_state.X.columns):
        ind = i % 2
        if ind == 0:
            with feat0:
                st.info(feature)
            with val0:
                st.success(str(row[i]))
        elif ind == 1:
            with feat1:
                st.info(feature)
            with val1:
                st.success(str(row[i]))
    return


def init_session_state():
    # session state
    if 'loaded' not in st.session_state:
        # validation set given in notebook
        n_valid = 1000

        # import raw data
        df_raw = pd.read_csv(path_to_data)

        # preprocess data
        
        x_train,x_test,y_train,y_test = process_df(df_raw)
        X, y = x_test.iloc[:n_valid], y_test.iloc[:n_valid]

        # load regression model
        path_to_model = os.path.join('saves', 'lgb_model.sav')
        with open(path_to_model, 'rb') as file:
            model = pickle.load(file)

        # load bulldozer images
        # imgs = []
        # path_to_imgs = path_to_repo / '_docs' / 'img'
        # img_files = os.listdir(path_to_imgs)
        # for img_file in img_files:
        #     img = Image.open(os.path.join(path_to_imgs, img_file))
        #     img = np.array(img)
        #     imgs.append(img)

        # store in cache
        st.session_state.loaded = True
        st.session_state.n_valid = n_valid
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.model = model
        #st.session_state.imgs = imgs


def app():
    init_session_state()
    st.title('Choose data')
    options = ['-'] + list(range(1, st.session_state.n_valid + 1))
    index = st.selectbox(label = 'Choose a bulldozer index', options = options, index = 0)
    if index != '-':
        display_bulldozer_img(index)
        display_bulldozer_price(index)
        display_bulldozer_features(index)
    return


if __name__ == '__main__':
    app()