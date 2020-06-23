#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:45:39 2020

@author: zakariaameziane
"""



import applicationInterface as app
import cnnInterface as cnn
import dataInterface as data

if __name__ == '__main__':
	"""Runs the entire application"""
	cnn.loadModel()
	print("====================Main Interface====================")
	while True:
		print("\nOptions:\n1. Run Applications.\n2. Process Data for training.\n3. Build and test network.\n4. Exit")
		opt = input("Enter the option number : ")
		if opt == "1":
			app.interface()
		elif opt == "2":
			data.interface()
		elif opt == "3":
			cnn.interface()
		elif opt == "4":
			break
		else:
			print("Invalid option")