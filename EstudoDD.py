# encoding: utf-8
# coding=utf-8
import datetime
import matplotlib.pyplot as plt
from matplotlib import pylab
import networkx
import ConfigParser
import os
import unidecode
import unicodedata
from stop_words import get_stop_words
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import Stemmer
import csv
import random
import json
import cPickle as pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
np.seterr(divide='ignore', invalid= 'ignore') # ignora divisao por zero do numpy
from PyWANN import WiSARD

class Configs():
	
	def __init__(self, path):
		self.path                  = path

		self.file_path             = None
		self.json_path             = None
		self.test_set_path         = None
		self.load_from_json        = None
		self.wisard_name		   = None
		self.key_index             = None
		self.complaint_type_index  = None
		self.description_index     = None

		self.main_complaint_filter = None
		self.weight_chart          = None
		self.sample_size           = None
		self.number_of_lines       = None

		self.classificationList    = None
			   
		self.LoadConfigs()
	
	def ConfigSectionMap(self, section):
		Config = ConfigParser.ConfigParser()
		Config.read(self.path)
		dict1 = {}
		options = Config.options(section)
		for option in options:
			try:
				dict1[option] = Config.get(section, option)
				if dict1[option] == -1:
					DebugPrint("skip: %s" % option)
			except:
				print("exception on %s!" % option)
				dict1[option] = None
		return dict1

	def LoadConfigs(self):
		dict = self.ConfigSectionMap("SectionOne")

		self.file_path             = dict['file_path']
		self.json_path             = dict['json_path']
		self.test_set_path         = dict['test_set_path']

		self.load_from_json        = False if dict['load_from_json'] == "False" else True
		self.wisard_name           = dict['wisard_name']
		self.key_index             = int(dict['key_index'])
		self.complaint_type_index  = int(dict['complaint_type_index'])
		self.description_index     = int(dict['description_index'])

		self.main_complaint_filter = dict['main_complaint_filter']
		self.weight_chart          = dict['weight_chart'] == 'True' or dict['weight_chart'] == 'true'
		self.sample_size           = int(dict['sample_size'])
		self.number_of_lines       = int(dict['number_of_lines'])

		self.classificationList    = dict['classification_list'].split (",")

#"C:\\Projects\\Estudo_DD\\config.ini"
config_file_path = os.getcwd() + "\\config.ini" 
configs = Configs(config_file_path)

vectorizer = None

def __main__():
	
	register_list = None

	if configs.load_from_json:
		
		# load from jsonFile
		text_file = open (configs.json_path, "r")
		json_string = text_file.read()
		text_file.close()
		register_list = json.loads(json_string)
	else:
		#load from csv
		register_list = get_list_of_lists(configs.sample_size)
		# Cleaning description text of each complaint
		print "Cleaning reports descriptions text..."
		for register in register_list:
				register[configs.description_index] = clean_portuguese_text(register[configs.description_index])

		# Create Json file
		#text_file = open ("estudo_algoritimo2015_json.txt", "w")
		#json_string = text_file.write (json.dumps (register_list))
		#text_file.close()
	#print "Total de registros = " , len(register_list)

	# complaint_counter = 0
	# sub_complaint_counter = 0
	# for register in register_list:
	#     if register[configs.complaint_type_index].strip() == '1':
	#         complaint_counter += 1
	#     elif register[configs.complaint_type_index].strip() == '0':
	#         sub_complaint_counter += 1

	# print "Denuncias = " , complaint_counter
	# print "Sub denuncias = " , sub_complaint_counter
	
	# relation_dict = get_complaints_relation_dict(register_list)
		
	# print relation_dict
	# print len(relation_dict)


	# dicionario de stopwords
	sw = []
	for stword in list(get_stop_words('pt')):
		sw.append(stword.lower())

	wisard = wisard_training (register_list, sw)

	wisard_predict_all (wisard, sw)
	
	#with open("estudoDD_Tratado.csv", "wb") as f:
	#    writer = csv.writer(f)
	#    writer.writerows(register_list)
						
	#if(configs.weight_chart):
	#    plot_weight_graph(register_list)
	#else:
	#    plot_simple_graph(register_list)


def clean_text(text):
	try :
		unicode(text, "ascii" )
	except UnicodeError:
		text = unicode(text, "utf-8" ).encode("utf-8-sig")
	except TypeError:
		pass
	else :
		# value was valid ASCII data
		pass 
	text = text.replace(".", " ")
	text = text.replace("\n", " ")
	text = text.replace("\r", " ")
	text = text.replace("(", " ")
	text = text.replace(")", " ")	
	text = text.replace("?", " ")
	text = text.replace(",", " ")
	text = text.replace("%", " ")
	text = text.replace("R$", " ")
	text = text.replace("/", " ")
	text = text.replace("-", " ")
	text = text.replace(":", " ")
	text = text.replace("0", " ")
	text = text.replace("1", " ")
	text = text.replace("2", " ")
	text = text.replace("3", " ")
	text = text.replace("4", " ")
	text = text.replace("5", " ")
	text = text.replace("6", " ")
	text = text.replace("7", " ")
	text = text.replace("8", " ")
	text = text.replace("9", " ")
	text = text.replace("'", " ")
	text = text.replace('"', " ")
	text = text.replace(';', " ")
	'''
	st = RSLPStemmer()
	retorno = ''
	for token in text.split():
		 retorno +=  st.stem(token) + ' '
	print retorno
	'''
	
	return text

def clean_portuguese_text(text):
	text = clean_text(text)

	stop_words = get_stop_words('pt')
	stop_words = get_stop_words('portuguese')

	stop_words.append ("rua")
	stop_words.append ("estrada")
	stop_words.append ("citada")
	stop_words.append ("citado")
	stop_words.append ("endereço")
	stop_words.append ("endereco")
	stop_words.append ("caminho")
	stop_words.append ("período")
	stop_words.append ("periodo")
	stop_words.append ("próximo")
	stop_words.append ("proximo")
	stop_words.append ("próxima")
	stop_words.append ("proxima")
	stop_words.append ("mencionado")
	stop_words.append ("mencionada")
	stop_words.append ("altura")
	stop_words.append ("complementa")
	stop_words.append ("denuncia")
	stop_words.append ("denúncia")
	stop_words.append ("diariamente")
	stop_words.append ("avenida")
	stop_words.append ("município")
	stop_words.append ("municipio")

	words = text.split(' ')

	content = ""

	stemmer = Stemmer.Stemmer('portuguese')

	for word in words:
		word = word.lower()
		if word not in stop_words and word.strip():
			content = content + stemmer.stemWord(word.lower()).upper() + " ";

	# Removing last space
	content.rstrip()
	content.lstrip()
	
	return content
	
def plot_weight_graph(register_list):
	count_dict = get_main_complaints_count_dict(register_list)
	ns = []

	for c in count_dict.values():
		# Scale this up by 1000
		ns.append(c*100)
   
	tuples_list = get_complaints_relation_tuples_list(register_list)
	
	the_graph = networkx.Graph()
	
	for tuple_r in tuples_list:
		a = tuple_r[0]
		b = tuple_r[1]
		if (the_graph.has_edge(a,b)):
			the_graph[a][b]['weight'] += 2
		else:
			the_graph.add_edge(a,b,weight=2)    
		the_graph.add_edge(a,b)  
	
	weight_list = []
	for u,v in the_graph.edges():
		w = the_graph[u][v]['weight']
		weight_list.append(w)
	
	the_graph.add_nodes_from(count_dict.keys())
	the_graph.add_edges_from(tuples_list)
	
	# set the graphic inline in screen
	# set the figure size
	
	plt.figure(figsize=(20,20))

	# do some layouts
	pos1=networkx.fruchterman_reingold_layout(the_graph)

	# set the labels, draw the graph
	networkx.draw(the_graph,pos1,nodelist=count_dict.keys(),font_size=12,with_labels=True,node_color='#006FFF',edge_color='#42b7a0',alpha=0.35,node_size=ns,width=weight_list)
	pylab.show()
	
def plot_simple_graph(register_list):
	count_dict = get_main_complaints_count_dict(register_list)
	   
	tuples_list = get_complaints_relation_tuples_list(register_list)
	
	the_graph = networkx.Graph()
	
	for tuple_r in tuples_list:
		a = tuple_r[0]
		b = tuple_r[1]
		the_graph.add_edge(a,b)  
	
	the_graph.add_nodes_from(count_dict.keys())
	the_graph.add_edges_from(tuples_list)
	
	# set the figure size
	plt.figure(figsize=(20,20))

	# do some layouts
	pos1=networkx.fruchterman_reingold_layout(the_graph)

	# set the labels, draw the graph
	networkx.draw(the_graph,pos1,nodelist=count_dict.keys(),font_size=12,with_labels=True)
	pylab.show()

def get_complaints_relation_tuples_list(register_list):
	r_list = list()
	list_buffer=[]
	for register in register_list:
		if len(list_buffer) == 0 or list_buffer[-1][configs.description_index] == register[configs.description_index]:
			list_buffer.append(register)
		else :# else = the description of the last element at the buffer is different from the current one
			key = get_key_at_list (list_buffer)
			for buffer_register in list_buffer:# Loop over buffer relating registers
				if(buffer_register[configs.complaint_type_index] != '1'):# and buffer_register[configs.key_index] == 'TRAFICO DE DROGAS'):
					if(configs.main_complaint_filter == '' or buffer_register[configs.key_index] == configs.main_complaint_filter):
						r_tuple = (key, buffer_register[configs.key_index])
						r_list.append (r_tuple)
			# clear list buffer
			del list_buffer[:]
			list_buffer.append(register)
	return r_list

# {"Trafico de drogas": {"Porte de arma":3, "Agressao":4}}
def get_complaints_relation_dict(register_list):
	r_dict={}
	list_buffer=[]
	for register in register_list:
		if len(list_buffer) == 0 or list_buffer[-1][configs.description_index] == register[configs.description_index]:
			list_buffer.append(register)
		else :# else = the description of the last element at the buffer is different from the current one
			# Loop over buffer relating registers
			key = get_key_at_list (list_buffer)
			if key is not None:
				sub_list = get_sub_complaints(list_buffer)
				join_at_dict(r_dict, key, sub_list)
			# clear list buffer
			del list_buffer[:]
			list_buffer.append(register)
	return r_dict
	
def join_at_dict(main_dict, key, sub_list):
	sub_dict = {}
	
	if main_dict.__contains__(key):
		sub_dict = main_dict[key]
	else:
		main_dict[key] = sub_dict
		 
	for register in sub_list:
		if sub_dict.__contains__(register[configs.key_index]):
			sub_dict[register[configs.key_index]] += 1
		else:
			sub_dict[register[configs.key_index]] = 1
	
def get_sub_complaints(list_buffer):
	sub_list=[]
	for register in list_buffer:
		if register[configs.complaint_type_index] == '0':
			sub_list.append(register)
	return sub_list
	
def get_key_at_list(r_list):
	for register in r_list:
		if register[configs.complaint_type_index] == '1':
			return register[configs.key_index]
	return None

def get_complaints_count_dict(register_list):
	c_dict = dict()
	for register in register_list:
		if not (register[configs.key_index] in c_dict):
			c_dict[register[configs.key_index]] = 1
		else:
			c_dict[register[configs.key_index]] += 1
	return c_dict
	
def get_main_complaints_count_dict(register_list):
	c_dict = dict()
	for register in register_list:
		if register[configs.complaint_type_index] == '1': # and register[configs.key_index] == 'TRAFICO DE DROGAS':
			if(configs.main_complaint_filter == '' or register[configs.key_index] == configs.main_complaint_filter):
				if not (register[configs.key_index] in c_dict):
					c_dict[register[configs.key_index]] = 1
				else:
					c_dict[register[configs.key_index]] += 1
	return c_dict
	
def get_list_of_lists(max_size):
	register_list = []

	if not configs.sample_size == 0:
		sorted_lines = random.sample(range(1, configs.number_of_lines), configs.sample_size)

	with open(configs.file_path, 'r') as csvfile:
		reader = csv.reader (csvfile, delimiter=';')
		counter = 0
		for row in reader:
			# Jump if not in sorted line list
			if(configs.sample_size != 0 and counter not in sorted_lines):
				counter = counter + 1
				continue

			row[configs.key_index] = remove_accents(unicode(row[configs.key_index], 'utf8')).strip()
			row[configs.complaint_type_index] = row[configs.complaint_type_index].strip() 
			row[configs.description_index] = remove_accents(unicode(row[configs.description_index], 'utf8')).strip()
			counter = counter + 1
			register_list.append(row)
	return register_list

def remove_accents(input_str):
	nfkd_form = unicodedata.normalize('NFKD', input_str)
	only_ascii = nfkd_form.encode('ASCII', 'ignore')
	return only_ascii


# Wisard Region

def wisard_training (register_list, sw):
	
    # Get training set
	global vectorizer

	# Import wisard from file if the name is set
	if configs.wisard_name:
		with open(configs.wisard_name, 'rb') as input:
			wisard = pickle.load(input)
		with open('vectorizer_' + configs.wisard_name, 'rb') as input:
			vectorizer = pickle.load(input)

		#print vectorizer.vocabulary
		return wisard

	train_set    = []
	labels       = []
	count_dict   = {}

	# Set SKL vectorizer
	#vectorizer =  TfidfVectorizer(use_idf = True, lowercase=True, norm = 'l2', ngram_range=(1,1), min_df=0.0, max_df = 0.8)
	vectorizer =  CountVectorizer(lowercase=True, ngram_range=(1,1), min_df=0.0, max_df = 0.8, binary=True)

	# Building training set
	for register in register_list:
		if register[configs.complaint_type_index] == '1' and register[configs.key_index] in configs.classificationList:
			report_classification = register[configs.key_index]
			report_description    = register[configs.description_index]
			
			if(report_classification not in count_dict):
					count_dict[report_classification] = 0
	
			if(count_dict[report_classification] >= 10):
				continue
	
			count_dict[report_classification] = count_dict[report_classification] + 1
			labels.append (report_classification)
			train_set.append (register[configs.description_index])

	print "Vectorizing sets..."
	
	V = vectorizer.fit_transform(train_set).toarray()
	
	print "Training Wisard"	
	
	labels = np.array (labels) # labels
	wisard = WiSARD.WiSARD (3, True, ignore_zero_addr=True)#,True, 3) # Atribui a classe Wisard para w
	v_1    = np.array (V)     # conjunto de treinamento
	wisard.fit (v_1, labels)  # Associacao dos labels aos conjuntos de treinamento

	# Saving wisard to file
	file_name = 'wisard' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.pkl'

	with open (file_name, 'wb') as output:
		pickle.dump (wisard, output, pickle.HIGHEST_PROTOCOL)

	with open ('vectorizer_' + file_name, 'wb') as output:
		pickle.dump (vectorizer, output, pickle.HIGHEST_PROTOCOL)
	
	return wisard

def wisard_predict_all (wisard, sw):
	
	# Saving result to file
	file_name = 'result' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt'
	with open (file_name, 'wb') as output:
		with open(configs.test_set_path, 'r') as csvfile:
			reader = csv.reader (csvfile, delimiter=';')
			for row in reader:
				row[configs.key_index]            = remove_accents(unicode(row[configs.key_index], 'utf8')).strip()
				row[configs.complaint_type_index] = row[configs.complaint_type_index].strip() 
				row[configs.description_index]    = remove_accents(unicode(row[configs.description_index], 'utf8')).strip()
				classification = wisard_predict_single(wisard, row)
				output.write(row[configs.key_index] + "," + classification+ "\n")


def wisard_predict_single(wisard, row):
	global vectorizer
	
	#maus tratos  
	#test_report = remove_accents(u"NO ENDEREÇO CITADO, PRÓXIMO AO COLÉGIO MUNICIPAL CASTRO ALVES, LOCALIZA-SE UMA CASA COM CERCA, E PORTÃO DE MADEIRA, DE TIJOLO, ONDE RESIDE 'RENATA FARIA' (NÃO CARACTERIZADA), QUE PODE SER ENCONTRADA DIARIAMENTE, EM QUALQUER HORÁRIO, A QUAL É ALCOÓLATRA, POSSUI TRÊS FILHOS COM IDADES ENTRE 7 E 12 ANOS, OS QUAIS SÃO CONSTANTEMENTE AGREDIDOS VERBAL E FISICAMENTE, DEIXADOS SOZINHOS DENTRO DE CASA, ALÉM DE SEREM DOPADOS COM REMÉDIOS.")
	#barulho
	#test_report = remove_accents(u"NA RUA CITADA, ANA ALTURA DO Nº 115 (CENTO E QUINZE), ACESSO PELA RUA VISCONDE DE SANTA ISABEL, NESTE MOMENTO INDIVÍDUOS (NÃO IDENTIFICADOS) COLOCARAM MÚSICAS PARA SEREM TOCADAS EM UM VOLUME EXTREMAMENTE ALTO, IMPOSSIBILITANDO O DESCANSO DOS MORADORES.")
	
	test_report = row[configs.description_index]
	
	test_str = clean_portuguese_text(remove_accents(unicode(test_report, 'utf8')).strip())
	
	#vectorizer.min_df = 0.0
	#vectorizer.max_df = 1.0

	vetorized_set = vectorizer.transform ([test_str]).toarray()
	
	#print test_report

	#X_test = np.array ([vetorized_set]) # vetor de testes
	percentual = wisard.predict_proba (vetorized_set) # calcula a probabilidade
	percentual = percentual[0]
	result     = wisard.predict (vetorized_set) # Retorna o resultado
	return result[0]

	#build perc visualizer
	perc_visualizer = []
	for i in range(len(wisard.classes_)):
		perc_visualizer.append ([percentual[i]*100, wisard.classes_[i]])

	# Inplace sort
	perc_visualizer.sort(key=lambda tup: tup[0], reverse=True)

	for i in range(len(perc_visualizer)):
		item = perc_visualizer[i]
		item[0] = "{0:.2f}%".format(item[0])

	print "\n============================================="		
	print 'Tested Report :' , row[configs.key_index]
	print ""
	#print "Report recommendation:", wisard.classes_
	#print "Result:", result
	#print "Classification percentage : ", (percentual)*100
	for i in range(len(perc_visualizer)):
		print perc_visualizer[i][0], perc_visualizer[i][1] 

	print "============================================="

	return perc_visualizer[0][1]


	
__main__()



