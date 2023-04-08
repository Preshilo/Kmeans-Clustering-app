import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os
import sys
import streamlit as st

from sklearn.decomposition import PCA
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

sys.path.append('.../py_script')
from Utilss import extract_time, num_to_word, user_input_features, cluster_mapper, cluster_predict

def func_page_1():
    st.title('Model fitting')
    #st.header()

    num_clus = st.session_state['num_clus']
    pca_features = st.session_state['pca-features']
    pca_df = st.session_state['pca_df']
    pca3 = st.session_state['pca3']
    df_col_name = st.session_state['df_col_name']


    @st.cache_data
    def KMclusterer():
        km = KMeans(n_clusters = num_clus, random_state = 23)
        km.fit(pca_features)
        pca_df['Cluster'] = km.labels_
        pca_df ['Cluster'] = pca_df['Cluster'].map(cluster_mapper(num_clus))
        fig = px.scatter_3d(pca_df, x='PCA_1', y ='PCA_2', z='PCA_3',  color = 'Cluster',)
        fig.update_layout(title={'text': '3D scatter plot of users cluster assignments', 'x': 0.45,
                        'xanchor': 'center','yanchor': 'top', 'font': {'size': 24} },
                        legend= {'itemsizing': 'constant' }, autosize=False, width=900, height=600)
        fig.update_traces(marker_size = 2)
        return km, fig
    km, fig3 = KMclusterer()
    st.plotly_chart(fig3)

    input_ = st.sidebar.text_input("Enter the user's power usage")
    data_input = input_.replace(" ", "").split(',')
    st.sidebar.markdown('***HINT:*** Seperate the values using comma(E.g: 1.0, 2.0, 3.0....')

    if len(data_input) < len(df_col_name):
        st.sidebar.error(f"You've entered incomplete amount of data....{len(data_input)} datapoints recieved...", icon = "🚨")
        data_input  = None
        if 'data_input' in st.session_state:
            cached_data_input = st.session_state['data_input']
            st.sidebar.info('Cached input returned')
        else:
            st.info('No cached data input available, enter new data')
            st.stop()
        #st.stop()
    elif len(data_input) > len(df_col_name):
        st.sidebar.error(f"You've entered excess amount of data....{len(data_input)} datapoints recieved...", icon = "🚨")
        data_input = None
        if 'data_input' in st.session_state:
            cached_data_input = st.session_state['data_input']
            st.info('The previous input returned')
        else:
            st.info('No cached data input available, enter new data')
            st.stop()
        #st.stop()
    else:
        st.sidebar.success('Data entry completed...')
        st.session_state['data_input'] = data_input
        new_data_input = st.session_state['data_input']

    if data_input != None:
        new_data = user_input_features(data_input, df_col_name)
        st.dataframe(new_data)
        #st.session_state['new_data'] = new_data
        new_transformed_data = pca3.transform(new_data)
        clus_pred = km.predict(new_transformed_data)[0]
        st.success(f'This user belongs to **{cluster_predict(num_clus, clus_pred)}**')
        dd = new_data.iloc[0,: ].to_list()
        t = extract_time(datetime(2012, 9, 1, 0, 30), datetime(2012, 9, 2, 0, 30), timedelta(minutes=30))
        df = pd.DataFrame({'Consumption':dd, 'Time' : t})
        fig4 = px.line(df, y = 'Consumption', x = 'Time', title = 'A line graph showing the users consumption')
        st.plotly_chart(fig4)

    elif cached_data_input:
        new_data = user_input_features(cached_data_input, df_col_name)
        #st.write("There's no cached new user data")
        #new_data = user_input_features(new_data_input)
        st.dataframe(new_data)
        new_transformed_data = pca3.transform(new_data)
        clus_pred = km.predict(new_transformed_data)[0]
        st.success(f'This user belongs to **{cluster_predict(num_clus, clus_pred)}**')
        dd = new_data.iloc[0,: ].to_list()
        t = extract_time(datetime(2012, 9, 1, 0, 30), datetime(2012, 9, 2, 0, 30), timedelta(minutes=30))
        df = pd.DataFrame({'Consumption':dd, 'Time' : t})
        fig4 = px.line(df, y = 'Consumption', x = 'Time', title = 'A line graph showing the users consumption')
        st.plotly_chart(fig4)

    else:
        st.stop()
