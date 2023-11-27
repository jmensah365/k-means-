'''
Implementation of k-means++ in Python for senior comps project
Jeremiah Mensah
'''
import pandas as pd
import numpy as np
import tkinter as tk
import math
from tkinter import filedialog
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA

def kmeans_plusplus(data, k, max_iters=300):
    n_samples, n_features = data.shape
    
    #Initialize the first centroid randomly
    centroids = [data[np.random.randint(0, n_samples)]]
    
    for _ in range(1, k):
        #Calculate the squared distance from each point to the nearest centroid
        min_distances = [] 
        
        for data_point in data:
            min_distance = float('inf') 
            for c in centroids:
                distance = np.sum((data_point - c) ** 2)
                min_distance = min(min_distance, distance)
            min_distances.append(min_distance)
                  
          
        
        #Choose the next centroid with probability proportional to distance squared
        distances = np.array(min_distances)
        probabilities = distances / distances.sum()
        next_centroid_index = np.random.choice(range(n_samples), p=probabilities)
        centroids.append(data[next_centroid_index])
    
    centroids = np.array(centroids)

    # Initialize cluster assignments and prev_centroids
    labels = np.zeros(n_samples)
    for _ in range(max_iters):
        #Assign each data point to the nearest centroid
        for i in range(n_samples):
            distances = []  
            for c in centroids:
                distance = 0
                for x_i, c_i in zip(data[i], c):
                    distance += math.pow(x_i - c_i, 2)
                distance = math.sqrt(distance)  
                distances.append(distance)  
            
            labels[i] = distances.index(min(distances))  # Assign the point to the nearest centroid
        
        #Update centroids
        new_centroids = np.empty((k, data.shape[1])) 
        for j in range(k):
            #calculates mean for each attribute in the selected clsuter
            new_centroids[j] = data[labels == j].mean(axis=0)
        
        #Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

#Loading dataset
def load_dataset():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path)  # Load your dataset here
            return data
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
    return None

#Computing within-clusters sum for elbow method 
def WCSS(k_range, data):
    wcss_values = []
    label_values = []
    for k_value in k_range:
        centroids, labels = kmeans_plusplus(data, k_value)
        wcss = 0
        #Calculating within-cluster sum (WCSS)
        for i in range(k_value):
            cluster_points = data[labels == i]
            cluster_center = centroids[i]
            distance = np.sum((cluster_center - cluster_points) ** 2)
            wcss += distance
        wcss_values.append(wcss)
        label_values.append(labels)
        print(f'k = {k_value}, WCSS = {wcss}')
    return wcss_values, label_values

#Saving clusters to csv
def save_clusters_to_csv(k, label_values, data):
    file_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Cluster', 'stX1PAREDU', 'stX1SES', 'stINCOMEPERHHMEM', 'stS1HRACTIVITY', 'X1SEX', 'X1RACE'])
                
                # Loop through data and labels to save clusters
                for i in range(k):
                    cluster_points = data[label_values[k - 1] == i]
                    cluster_column = [f'Cluster {i}'] * len(cluster_points)
                    cluster_data = cluster_points.to_numpy()
                    for row in cluster_data:
                        writer.writerow([cluster_column[0]] + list(row))
                    
            tk.messagebox.showinfo("Cluster Data Saved", f"Cluster assignments saved to {file_path}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error saving cluster data: {str(e)}")


#Performs clustering and updates the plot
def cluster_and_plot():
    k = int(k_entry.get())
    data = load_dataset()
    if data is not None:
        k_range = list(range(1, k + 1))
        selected_columns = ['stX1PAREDU','stX1SES','stINCOMEPERHHMEM','stS1HRACTIVITY']
        numeric_columns = data[selected_columns].values
         # Apply PCA to reduce the dimensionality of the data
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(numeric_columns)

        #calculating WCSS for elbow method
        wcss_values, label_values = WCSS(k_range, data_pca)
        labels = label_values[k - 1]
        centroids, _ = kmeans_plusplus(data_pca, k) 
        ax.clear()

        #Plotting cluster points and centroids
        for i in range(k):
            cluster_points = data_pca[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'Cluster {i}')
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', label='Centroids')
        ax.set_xlabel('Principle Component 1')
        ax.set_ylabel('Principle Component 2')
        ax.set_title('K-Means++ Clustering with PCA')
        ax.legend()
        canvas.draw()
        save_clusters_button.config(command=lambda: save_clusters_to_csv(k,label_values,data))

#main application window
root = tk.Tk()
root.title('K-Means++ Clustering GUI')

# Create a frame for the input widgets
input_frame = ttk.Frame(root)
input_frame.pack(pady=10)

# Label and entry for specifying k (number of clusters)
k_label = ttk.Label(input_frame, text='Number of Clusters (k):')
k_label.grid(row=0, column=0, padx=10, pady=5)
k_entry = ttk.Entry(input_frame)
k_entry.grid(row=0, column=1, padx=10, pady=5)

# Button to perform clustering and update the plot
cluster_button = ttk.Button(input_frame, text='Cluster Data', command=cluster_and_plot)
cluster_button.grid(row=0, column=2, padx=10, pady=5)

# Create a figure for the plot
fig = Figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# Create a canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

save_clusters_button = ttk.Button(input_frame, text = 'Save Clusters', state = tk.ACTIVE)
save_clusters_button.grid(row=0,column=3,padx=10,pady=5)

# Start the GUI main loop
root.mainloop()
