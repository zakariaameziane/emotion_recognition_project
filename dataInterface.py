#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:38:22 2020

@author: zakariaameziane
"""


import numpy as np
import os
import cv2

OUTPUT_DIRECTORY = os.path.join("Emotion Dataset")
EMOTION_DIRECTORIES = ["/0_NEUTRAL","/1_ANGER","/5_HAPPY","/6_SADNESS"]


def storeLabledImagesInFile():
	"""Consolidates the images in the emotion directories and stores them in data.npy and lables.npy
	file. Does virtual sampling for classes which do not have sufficient samples"""
	data = []
	labels = []
	if os.path.exists(OUTPUT_DIRECTORY):
		images = []
		noOfImages = []
		for dir in EMOTION_DIRECTORIES:
			if os.path.exists(OUTPUT_DIRECTORY+dir):
				images.append(os.listdir(OUTPUT_DIRECTORY+dir))
				noOfImages.append(len(images[-1]))
		targetCount = max(noOfImages)
		for i in range(0,len(EMOTION_DIRECTORIES)):
			if os.path.exists(OUTPUT_DIRECTORY+EMOTION_DIRECTORIES[i]):
				mask = np.zeros((100,100))
				for j in range(0,targetCount):
					if(j!=0 and j%noOfImages[i] == 0):
						mask = np.random.randint(0,3,(100,100))
					face = cv2.imread(OUTPUT_DIRECTORY+EMOTION_DIRECTORIES[i]+"/"+images[i][j%noOfImages[i]])[:,:,1]
					face = face + mask
					face[np.where(face>=256)] = 255
					data.append(face)
					labels.append(i)
		np.save(OUTPUT_DIRECTORY+"/data",np.array(data))
		np.save(OUTPUT_DIRECTORY+"/labels",np.array(labels))
	else:
		print("Invalid path "+ OUTPUT_DIRECTORY)
		return False

def interface():
	"""User interface for this module"""
	print("====================Data Interface====================")
	while True:
		print("\nOptions:\n1. Store labled faces in output directory.\n2. Create data.npy and lables.npy files.\n3. Exit.")
		opt = input("Enter the option number : ")
		if opt == "1":
			storeLabledImagesInFile()
		elif opt == "2":
			storeLabledImagesInFile()
		elif opt == "3":
			break
		else:
			print("Invalid option")
	print("\n====================Main Interface====================")