import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('D:/Movies/spotify dataset.csv')
print("Missing values summary:\n", data.isnull().sum())
for col in data.columns:
    if data[col].dtype == 'object':  
        data[col] = data[col].fillna(data[col].mode()[0])  
    elif data[col].dtype == 'float64':  
        data[col] = data[col].fillna(data[col].mean())  
        pass
categorical_features = [col for col in data.columns if data[col].dtype == 'object']
label_encoder = LabelEncoder()
for feature in categorical_features:
    data[feature] = label_encoder.fit_transform(data[feature])
numerical_features = [col for col in data.columns if col not in categorical_features and col != 'playlist_genre']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
data.hist(figsize=(10, 6))
sns.boxplot(x="playlist_genre", y="danceability", showmeans=True, data=data) 
corr = data.corr()
sns.heatmap(corr, annot=True)
plt.show()
kmeans = KMeans(n_clusters=5)
kmeans.fit(data.drop('playlist_genre', axis=1)) 
data['cluster'] = kmeans.labels_
sns.scatterplot(x="danceability", y="energy", hue="cluster", data=data)
plt.show()  
X = data.drop('playlist_genre', axis=1)
y = data['playlist_genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
