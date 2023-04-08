import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import plotly.express as px

from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import  PCA
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

sys.path.append('.../py_script')
from Utilss import extract_time, num_to_word, user_input_features, cluster_mapper
from PAGE1 import func_page_1

st.set_page_config(layout="wide")


def main():
    if "page_state" not in st.session_state:
        st.session_state['page_state'] = 'Main Page'
    # Writing page selection to session state
    st.sidebar.subheader('Page selection')
    if st.sidebar.button('Main Page'):
        st.session_state['page_state'] = 'Main Page'
    if st.sidebar.button('Next Page'):
        st.session_state['page_state'] = 'Page 1'
    pages_main = {'Main Page': main_page, 'Page 1': run_page_1}
    # Run selected page
    pages_main[st.session_state['page_state']]()


def main_page():
    st.title('KMEANS CLUSTERING OF LONDON ELECTRICITY USERS')
    st.header('')

    col1, col2 = st.columns([3, 2])
    with col1:
        exp = st.expander('**ABOUT THE DATASET**')
        exp.markdown("""The default datasets used contain the energy consumption readings
                for a sample of 5,567 London Households that took part in the UK
                Power Networks led Low Carbon London project between November 2011
                and February 2014. For this project,  the **hhblock_dataset**  was used.
                It contained the half-hourly smart meter measurement from households
                in 112 blocks with unique user identities.
                ***[Click here](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london)***
                to access the datasets…""")
    with col2:
        exp1 = st.expander('**ABOUT THE APP**')
        exp1.markdown(""" The application showcases the utilization of the KMEANS clustering algorithm
         on London Electricity users, while also enabling users to easily implement the same algorithm
         on a distinct dataset through the upload of a CSV file with a long-formatted structure. """ )

    st.header('Dataset')
    data_file = st.sidebar.file_uploader('Upload a csv format of the dataset')
    if data_file is not None:
        data = pd.read_csv(data_file)
    else:
        os.chdir('.../pickles')
        with open('data.pickle', 'rb') as gh:
             data = pickle.load(gh)
    os.chdir('.../py_script')
    st.dataframe(data.head(5))
    st.write('')

    num_of_cols = len(data.columns)-1

    user_id = st.sidebar.text_input('Enter the column name of User_ID column')
    st.sidebar.markdown("***NOTE:*** Enter *NIL* if there's no column for user identity")

    if user_id == "":
        st.info("You haven't entered a user_id columnn")
        if 'cached_user_id' in st.session_state:
            st.info('cached_id returned')
            cached_user_id = st.session_state['cached_user_id']
            st.sidebar.info(f'Cached columnn name user_id : {cached_user_id}')
            feature_data = data.loc[:, data.columns != cached_user_id]
            df_col_name = feature_data.columns
            x = np.asarray(num_features) #Seleting all features columns except the user_id columnn
            x = StandardScaler().fit_transform(x)
            pca = PCA().fit(x)
            st.session_state['pca'] = pca
            st.session_state['x'] = x
            st.session_state['df_col_name'] = df_col_name
        else:
            st.info("There's no cached **user_id**, a user_id column has to be entered")
            st.stop()
    else:
        if user_id == 'NIL':
            try:
                x = np.asarray(data) #Seleting all features columns except the user_id columnn
                x = StandardScaler().fit_transform(x)
                pca = PCA().fit(x)
                df_col_name = data.columns
                st.session_state['pca'] = pca
                st.session_state['x'] = x
                st.session_state['cached_user_id'] = user_id
                st.session_state['df_col_name'] = df_col_name
            except ValueError as e:
                st.error(f'🎈🎈{e}')
                st.stop()
        else:
            if user_id in data.columns:
                feature_data = data.loc[:, data.columns != user_id]
                x = np.asarray(feature_data) #Seleting all features columns except the user_id columnn
                x = StandardScaler().fit_transform(x)
                pca = PCA().fit(x)
                df_col_name = feature_data.columns
                st.session_state['pca'] = pca
                st.session_state['x'] = x
                st.session_state['cached_user_id'] = user_id
                st.session_state['df_col_name'] = df_col_name
            else:
                st.warning(f"There's no column in the dataframe that has the name \"{user_id}\" ", icon="⚠️")
                st.stop()


    x = st.session_state['x']
    pca = st.session_state['pca']

    df_pca_com = pd.DataFrame({'Number of components':range(1,15),
                'Explained variance ratio':pca.explained_variance_ratio_[0:14]})
    fig1 = px.line(df_pca_com, x = 'Number of components', y = 'Explained variance ratio', markers=True)
    fig1.update_xaxes(tickmode = 'array',tickvals = df_pca_com['Number of components'])
    st.subheader('Principal components anaysis of the dataset')

    st.plotly_chart(fig1)

    #PCA and selccting a number of components
    @st.cache_data
    def cached_pca_function(df):
        kl = KneeLocator(df['Number of components'],
            df['Explained variance ratio'], curve = "convex", direction = "decreasing")
        number_of_components = kl.elbow
        pca3 = PCA(n_components = number_of_components)
        pca_features = pca3.fit_transform(x)
        pca_df = pd.DataFrame(data = pca_features, columns = [f'PCA_{i}' for i in range(1, number_of_components+1)] )
        return number_of_components, pca3, pca_features, pca_df

    num_components, pca3, pca_features, pca_df = cached_pca_function(df_pca_com)

    st.info(f"""According to the principal component analysis, retaining the first
            **{num_to_word(num_components)}** components will preserve a significant
            amount of the information (explained variance) present in the initial dataset...""")

    st.session_state['pca3'] = pca3
    st.session_state['pca-features'] = pca_features
    st.session_state['pca_df'] = pca_df

    st.subheader('Cluster Analysis')

    @st.cache_data
    def cluster_analysis(k_range):
        cal = []
        for i in k_range:
            km = KMeans(n_clusters = i).fit(pca_features)
            cal_score = calinski_harabasz_score(pca_features, km.labels_)
            cal.append(cal_score)

        df_cal = pd.DataFrame({'Number of clusters' : k_range, 'Calinski Harabasz score':cal})
        fig = px.line(df_cal, x = 'Number of clusters', y = 'Calinski Harabasz score',
                    title = 'Calinski Harabasz Score Elbow for KMeans Clustering', markers=True)
        fig.update_xaxes(tickmode = 'array',tickvals = df_cal['Number of clusters'])
        kl = KneeLocator(df_cal['Number of clusters'], df_cal['Calinski Harabasz score'],
                    curve="concave", direction="increasing")
        number_of_clusters = kl.elbow
        return fig, number_of_clusters

    fig2, num_clus = cluster_analysis(range(2, 12))

    st.plotly_chart(fig2)

    st.info(f"""Based on the chart presented above, it can be observed that the
            elbow point occurs at ***k = {num_clus}***, indicating that having
            **{num_to_word(num_clus)}** clusters would be appropriate.""")
    st.session_state['num_clus'] = num_clus

def run_page_1():
    func_page_1()

if __name__ == '__main__':
    main()
