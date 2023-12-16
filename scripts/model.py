import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class DataModel:
    """
    a class with list of data modeling methods.
    """
    def __init__(self) -> None:
        pass

    def generateModel(self, x, y):
        """
        a function that generates a regression model
        """
        # Creating regression model and fitting it

        # creating an object of LinearRegression class
        model = LinearRegression()

        # fitting the training data
        model.fit(x,y)
        print("Model created Sucessfully.")
        
        return model


    def clusterGenerator(self, df, selected_features, num_clusters, clust_name):
        """
        a function that generates a K means cluster value to rows.
        """
        selected_metrics = df[selected_features]

        # Creating normalized dataframes

        temp_arr = selected_metrics.loc[:, selected_features].values
        temp_arr = StandardScaler().fit_transform(temp_arr) # normalizing the features

        # Turning the temporary array of normalized values into a dataframe.

        normal_df = pd.DataFrame(temp_arr,columns=selected_features)

        # creating the clusters

        kmeans = KMeans(num_clusters)
        kmeans.fit(temp_arr)

        # Generating cluster value to each row

        identified_clusters = kmeans.fit_predict(temp_arr)
        # Adding the generated array of cluster values to the dataframe as a column

        data_with_clusters = df.copy()
        data_with_clusters[clust_name] = identified_clusters 
        
        return normal_df, data_with_clusters, kmeans

    
    def calcScore(self, df, cent, features, title):
        df[title] = df.apply(lambda df : np.linalg.norm(df[features] - cent), axis = 1)
        df2 = df[title]
        df_norm = ((df2-df2.min())/(df2.max()-df2.min()))*100
        df[title] = df_norm
        return df 