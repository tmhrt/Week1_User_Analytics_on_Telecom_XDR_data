import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

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
    def normalize(self, x):
        scaled = StandardScaler().fit_transform(x)
        return Normalizer().fit_transform(scaled)
    
    def cluster(self, x, n, feature_names, cluster_names, elev, azim):
        model = KMeans(n_clusters = n, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
        y_clusters = model.fit_predict(x)

        colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "gray"]
        fig = plt.figure(figsize = (15,15))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n):
            ax.scatter(x[y_clusters == i,0],x[y_clusters == i,1],x[y_clusters == i,2], s = 40 , color = colors[i], label = f"cluster {cluster_names[i]}")
        
        ax.set_xlabel(f"{feature_names[0]}-->")
        ax.set_ylabel(f"{feature_names[1]}-->")
        ax.set_zlabel(f"{feature_names[2]}-->")
        ax.view_init(elev, azim)
        ax.legend()
        plt.show()

        return model.labels_

    def optimum_k(self,X):
        cost =[]
        for i in range(1, 11):
            KM = KMeans(n_clusters = i, max_iter = 500)
            KM.fit(X)
            # calculates squared error for the clustered points
            cost.append(KM.inertia_)     
        
        # plot the cost against K values
        style.use("fivethirtyeight")
        plt.plot(range(1, 11), cost, color ='g', linewidth ='3')
        plt.xlabel("Value of K")
        plt.ylabel("Squared Error (Cost)")
        plt.show() # clear the plot
        
        # the point of the elbow is the 
        # most optimal value for choosing k



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