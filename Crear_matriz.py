import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

def Create_matrix(img, d):

	Matrix = np.zeros((256,256))

	for r in range(0, img.shape[0]- d):
		for c in range(img.shape[1] - d):
			a = img[r, c]
			b = img[r + 1, c]

			Matrix[a,b] += 1

	return Matrix


def Get_mean(Matrix):
	mu = 0
	S = np.sum(Matrix)
	for i in range(Matrix.shape[0]):
		for j in range(Matrix.shape[1]):
			mu += i*(Matrix[i,j]/ S)

	return mu

def Get_energy(Matrix):
	Energy = 0
	S = np.sum(Matrix)
	for i in range(Matrix.shape[0]):
		for j in range(Matrix.shape[1]):
			Energy += (Matrix[i,j]/ S)**2
	
	return Energy

def Get_entropy(Matrix):
	Entropy = 0
	s = np.sum(Matrix)
	for i in range(Matrix.shape[0]):
		for j in range(Matrix.shape[1]):
			if Matrix[i,j] == 0:
				continue
			Entropy += -(Matrix[i,j]/ s)*np.log(Matrix[i,j]/ s)

	return Entropy

def Get_contrast(Matrix):
	Contrast = 0
	s = np.sum(Matrix)
	for i in range(Matrix.shape[0]):
		for j in range(Matrix.shape[1]):
			Contrast += ((i-j)**2)*(Matrix[i,j] / s)
	return Contrast

def Get_homogeneity(Matrix):
	Homogeneity = 0
	s = np.sum(Matrix)

	for i in range(Matrix.shape[0]):
		for j in range(Matrix.shape[1]):
			if 1 + (i-j) == 0:
				continue
			Homogeneity += (1 / (1+(i-j)))*(2*(Matrix[i,j]/ s))

	return Homogeneity

def New_register(path,ID,df,cat):
	img 	 	= cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	Matrix  	= Create_matrix(img, d = 2)
	Mean 	 	= Get_mean(Matrix)
	Energy   	= Get_energy(Matrix)
	Entropy     = Get_entropy(Matrix)
	Contrast    = Get_contrast(Matrix)
	Homogeneity = Get_homogeneity(Matrix)
	new = pd.DataFrame({'Category': cat,'ID': [ID] ,'Mean': [Mean], 'Energy': [Energy],
	'Entropy': [Entropy], 'Contrast': [Contrast], 'Homogeneity': [Homogeneity]})
	
	df = pd.concat([df, new], ignore_index=True)
	return df


if __name__ == "__main__":

	df = pd.DataFrame(columns=['Category', 'ID', 'Mean', 'Energy', 'Entropy', 'Contrast', 'Homogeneity'])

	for i in range(1,101):
		num  = i
		path = "./Data/Nondefective/Non_defective_01_("+str(num)+").jpg"
		img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		df = New_register(path,i,df, 1)
	
	for i in range(1,101):
		num  = i
		path = "./Data/Grain_off/Grain_off_01_("+str(num)+").jpg"
		img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		df = New_register(path,i,df, 2)

	for i in range(1,101):
		num  = i
		path = "./Data/Grow_mark/Growth_mark01_("+str(num)+").jpg"
		img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		df = New_register(path,i,df,3)

	for i in range(1,101):
		num  = i
		path = "./Data/Pinhole/Pinhole_01_("+str(num)+").jpg"
		img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		df = New_register(path,i,df,4)

	df.to_csv('base_de_datos.csv', index=False)