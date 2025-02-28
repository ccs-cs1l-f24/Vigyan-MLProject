import os

# source: https://medium.com/@shahsanap89/different-ways-to-create-a-folder-in-python-38857d776d65

num = 64

for i in range(num):
    folder_path = './Data/BayesianModels/'+str(i)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")