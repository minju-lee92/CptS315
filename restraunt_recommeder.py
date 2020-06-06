import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import plotly
import plotly_express as px

#Location based recommendation filtering 

#read in files and data prep ------------------------------------------------------------------------------------
ratings = pd.read_csv('rating_final.csv')
print(ratings.head())
user = pd.read_csv('userprofile.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')
geoplace = pd.read_csv('geoplaces2.csv', encoding= 'latin-1')
#droping attributes that are not important
geoplace_cleaned = geoplace.drop(columns = ['fax', 'accessibility','the_geom_meter','address','alcohol',
                                            'smoking_area','dress_code','accessibility','Rambience','area','url','franchise','other_services'])
print(geoplace_cleaned.head())

# count how many ratings each restaurant has and get median ratings for each restaueant
rate_info = pd.DataFrame(ratings.groupby('placeID').agg({'rating':['count','median']}))
rate_info.columns = rate_info.columns.droplevel(0)
# merge rate info with geoplace data
rest_info = geoplace_cleaned.merge(rate_info, on ='placeID')
rest_info.rename(columns={'median':'median_rate','count':'Rcount'}, inplace = True)
print(rest_info)
#-----------------------------------------------------------------------------------------------------------------------
# Data exploration ------------------------------------------------------------------------------------------------------
# 1. Top 10 most rated restaurants
plt.show(cuisine['Rcuisine'].value_counts()[:10].plot(kind = 'bar', figsize=(8,6)))

# 2. distribution of the restaurant rating
plt.figure(figsize = (12,4))
ax = sns.countplot(rest_info['median_rate'])
plt.title('Distrubution of Restaurant Rating')
plt.show()

# 3. top restaurant based on it's count and median rate
top_restaurants = rest_info.sort_values(by=['Rcount','median_rate'], ascending = False)[:20]
print(top_restaurants.head())
fig, ax=plt.subplots(figsize=(12,10))
sns.barplot(x='median_rate', y = 'name', data = top_restaurants, ax= ax)
plt.show()

# 4. graph all the restaurants from given data by it's location
#map box access token
px.set_mapbox_access_token(open('.mapbox_token').read())
#configure_plotly_browser_state()
fig = px.scatter_mapbox(rest_info, lat="latitude", lon="longitude",color="median_rate", size='Rcount' ,
                   size_max=30, zoom=3, width=1200, height=800)
fig.show()

# 5. from above code we can see there are 3 different main locations we will only look at the San Luis Potosi area 
rest_info['city_upper']=rest_info['city'].str.upper()
SanLuisPotosi = rest_info[rest_info.city_upper =='SAN LUIS POTOSI']
print(SanLuisPotosi)
# all the restaurants in San Luis Potosi we can see it int the map
px.scatter_mapbox(SanLuisPotosi, lat="latitude", lon="longitude", color="median_rate", size='Rcount' ,
                   size_max=15, zoom=10, width=1200, height=800)

# 6. determine the optimal number of clusters(K) using elbow metho
coordinates = SanLuisPotosi[['latitude','longitude']] 
distortions=[]
K =range(1,25)
for k in K:
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model = kmeans_model.fit(coordinates)
    
    distortions.append(kmeans_model.inertia_)


fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(K, distortions, marker='o')
plt.xlabel('k')
plt.ylabel('Distortions')
plt.title('Elbow Method For Optimal k')
plt.savefig('elbow.png')
plt.show()
#---------------------------------------------------------------------------------------------------------#
# Recommendation algorithms 
#K-menas clustering-------------------------------------------------------------
kmeans = KMeans(n_clusters = 5, init='k-means++')
kmeans.fit(coordinates)

# cluster by location in SanLuisPotosi
SanLuisPotosi.loc[:,'cluster'] = kmeans.predict(SanLuisPotosi[['latitude','longitude']])
print(SanLuisPotosi.head())
# representation of cluster in a scatter mapbox
cluster_fig = px.scatter_mapbox(SanLuisPotosi, lat="latitude", lon="longitude", color="cluster", size='Rcount', 
                  hover_data= ['name', 'latitude', 'longitude'], zoom=10, width=1200, height=800)
cluster_fig.show()
#---------------------------------------------------------------------------------------------------------------
def recommend_restaurants_by_location(df, latitude, longitude):
    # Predict the cluster for longitude and latitude provided
    cluster = kmeans.predict(np.array([latitude,longitude]).reshape(1,-1))[0]   
    # Get the best restaurant in this cluster
    return cluster, df[df['cluster']==cluster].iloc[:][['placeID','name', 'latitude','longitude','cluster']]

# for user = U1008 (22.122989,-100.923811)
cluster, near_rest = recommend_restaurants_by_location(SanLuisPotosi,22.122989,-100.923811)
near_rest = near_rest.append({'name' : 'U1008' , 'latitude' : 22.122989,'longitude':-100.923811}, ignore_index=True)
print(near_rest)

# let's see it in the map
recommendation_map = px.scatter_mapbox(near_rest, lat = 'latitude', lon = 'longitude', color = 'cluster', text='name',hover_data=['name'],zoom=10, width=1200, height=800)
recommendation_map.show()
#####################################################################################################################
# join tables 
join_rate_info = ratings.merge(rate_info, on='placeID', how='right')
join_rate_info = join_rate_info.drop(columns=['food_rating','service_rating'])
joined_near_rest = near_rest.merge(join_rate_info, left_on='placeID', right_on='placeID', how='left')
print(joined_near_rest)

# Transform the values(rating) of the matrix dataframe into a scipy sparse matrix
restaurant_features = joined_near_rest.pivot(index = 'name', columns = 'userID', values = 'rating').fillna(0)
restaurant_features_matrix = csr_matrix(restaurant_features.values)

# k-nn for the item-based collaborative filtering
knn_recomm = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
knn_recomm.fit(restaurant_features_matrix)

# recommendation (Item-item collaborative filtering)
randomChoice = np.random.choice(restaurant_features.shape[0])
distances, indices = knn_recomm.kneighbors(restaurant_features.iloc[randomChoice].values.reshape(1, -1), n_neighbors = 5)
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for Restaurant {0} on priority basis:\n'.format(restaurant_features.index[randomChoice]))
    else:
        print('{0}: {1}'.format(i, restaurant_features.index[indices.flatten()[i]]))






