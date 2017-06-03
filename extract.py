from __future__ import print_function

import numpy as np
import tweepy

from util import load_dataset, save_model, load_model

# Tweepy API Configuration

auth = tweepy.OAuthHandler('12lMzu1Pzhwm1S97OTktP8tBO', 'c5vq4nZBGr3LS4mlNahukeCAbGTnU81g6jgmanYwbIh2AwAjyy')
auth.set_access_token('45496778-hsBpmEg7WuNiCxXEsyupbnEhsf6caG3rEhETFDfcf', 'H94iFF4sb5kue7xezUKBjUsxAVt21YU4OOucOoXVL5CNM')

api = tweepy.API(auth)


# Replacing name

dataArray = load_dataset('test.gender.nolabel.data')

screen_name = []
name = ""


for line in dataArray:
	
	# if username exists
	try:
		name = api.get_user(line[0]).name
	
	# if username does not exist anymore
	except:
		name = line[0]


	## Printing result to file

	i = 0

	for element in line:

		if i == 1:
			
			try:
				if ',' in name :
					name = line[0]

				print(name, end=','),

			except:
				print(line[0], end=','),

		elif i == 9:
			print(element)

		else:
			print(element, end=','),

		i += 1