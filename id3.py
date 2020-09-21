import numpy as np
import pandas as pd
import random
from pprint import pprint


def train_test_split(data,test_size):
	indices=data.index.tolist()
	test_indices=random.sample(population=indices,k=test_size)
	test_data=data.loc[test_indices]
	train_data=data.drop(test_indices)
	return train_data,test_data

def check_puruty(data):
	unique_label=np.unique(data[:,-1])
	if len(unique_label) == 1:
		return True
	else:
		return False

def classify(data):
	lablel_column=data[:,-1]
	unique_classes,counts_unique_classes=np.unique(lablel_column,return_counts=True)
	c=counts_unique_classes.argmax()
	return unique_classes[c] 

def get_potential_splits(data):
	potential_splits={}
	_,n_columns=data.shape
	for x in range(n_columns-1):
		potential_splits[x]=[]
		val=data[:,x]
		unique_values=np.unique(val)
		for y in range(len(unique_values)):
			if y!=0:
				current=unique_values[y]
				previous=unique_values[y-1]
				avg=(current+previous)/2
				potential_splits[x].append(avg)
	return potential_splits

def split_data(data,split_column,split_value):
	split_column_values=data[:,split_column]
	data_below=data[split_column_values<=split_value]
	data_above=data[split_column_values>split_value]
	return data_below,data_above

def calculate_entropy(data):
	lablel_column=data[:,-1]
	_,counts=np.unique(lablel_column,return_counts=True)
	probabilities=counts/counts.sum()
	entropy=sum(probabilities*-np.log2(probabilities))
	return entropy

def calculate_overall_entropy(data_below,data_above):
	p_data_below=len(data_below)/(len(data_below)+len(data_above))
	p_data_above=len(data_above)/(len(data_below)+len(data_above))
	overall_entropy=(p_data_above*calculate_entropy(data_above)+p_data_below*calculate_entropy(data_below))
	return overall_entropy

def determine_best_split(data,potential_splits):
	overall_entropy=9999
	for column_index in potential_splits:
		for value in potential_splits[column_index]:
			data_below,data_above=split_data(data,split_column=column_index,split_value=value)
			current_overall_entropy=calculate_overall_entropy(data_below,data_above)
			if current_overall_entropy<overall_entropy:
				overall_entropy=current_overall_entropy
				best_split_column=column_index
				best_split_value=value
	return best_split_column,best_split_value

def decision_tree_algorithm(df,counter=0,min_samples=2):
	
	if counter==0:
		global COLUMN_HEADERS
		COLUMN_HEADERS=df.columns
		data=df.values
	else:
		data=df
	
	if check_puruty(data) or len(data)<min_samples:
		classification=classify(data)
		return classification

	else:
		counter+=1

		potential_splits=get_potential_splits(data)
		split_column,split_value=determine_best_split(data,potential_splits)
		data_below,data_above=split_data(data,split_column,split_value)

		feature_name=COLUMN_HEADERS[split_column]
		question="{} <= {}".format(feature_name,split_value)
		sub_tree={question: []}

		yes_answer=decision_tree_algorithm(data_below,counter,min_samples)
		no_answer=decision_tree_algorithm(data_above,counter,min_samples)
		if yes_answer==no_answer:
			sub_tree=yes_answer
		else:
			sub_tree[question].append(yes_answer)
			sub_tree[question].append(no_answer)

		return sub_tree

def classify_example(example,tree):
	question=list(tree.keys())[0]
	if len(question.split())<=3:
		feature_name,comparision_operator,value=question.split()
	if example[feature_name]<=float(value):
		answer=tree[question][0]
	else:
		answer=tree[question][1]
	if not isinstance(answer,dict):
		return answer
	else:
		residual_tree=answer
		return classify_example(example,residual_tree)

def calculate_accuracy(df,tree):
	df["classification"]=df.apply(classify_example,axis=1,args=(tree,))
	df["classification_correct"]=df["classification"] == df["quality"]
	accuracy=df["classification_correct"].mean()
	return accuracy

	
df=pd.read_csv("wine_quality.csv")

train,test=train_test_split(df,40)
tree=decision_tree_algorithm(train)
# pprint(test.loc[0,test.columns],classify_example(test.iloc[0],tree))
pprint(calculate_accuracy(test,tree))
