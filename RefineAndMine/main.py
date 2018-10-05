
from util.util import writeCSVwithHeader,readCSVwithHeader,transform_to_the_wanted_structure
from bisect import bisect_left,bisect_right
from time import time
from random import random,randrange,randint
from operator import or_
from itertools import chain
import cProfile
import pstats
import signal
import sys
from copy import deepcopy
from intbitset import intbitset
from itertools import product
import argparse
from sys import stdout
import os
import pickle
from interruptingcow import timeout
def encode_sup_old(arr_pos):
    return sum(2**k for k in arr_pos)

def encode_sup(arr_pos):
    return intbitset(arr_pos)

def decode_sup(bitset_sup):
	bin_bitset_sup=bin(bitset_sup)[:1:-1]
	sup_ret=set()
	for i,v in enumerate(bin_bitset_sup):
		if v=='1':
			sup_ret|={i}
	return sup_ret


# def nb_bit_1(n):
# 	count=0
# 	while n:
# 		n &= (n-1)
# 		count+=1
# 	return count

def nb_bit_1_old(n):
	return bin(n).count('1')

def nb_bit_1(n):
	return len(n)

def transform(number):
	k=0
	n=number
	while n>10**3:
		k+=3
		n/=10**3
	return '$'+str(n)+' \\times '+ '10^{'+str(k)+'}$'

def transform_dataset(dataset,attributes,attr_label,wanted_label,verbose=True):
	new_dataset=[]
	statistics={}
	alpha_ratio_class=0.
	positive_extent=set()
	negative_extent=set()

	for k in range(len(dataset)):
		row=dataset[k]
		new_row={attr_name:row[attr_name] for attr_name in attributes}
		new_row['positive']=(row[attr_label]==wanted_label)
		if new_row['positive']:
			positive_extent|={k}
			alpha_ratio_class+=1
		else:
			negative_extent|={k}

		new_dataset.append(new_row)

	statistics['rows']=len(new_dataset)
	statistics['alpha']=alpha_ratio_class/len(dataset)
	nb_possible_intervals=1.
	for attr in attributes:
		statistics['|dom('+attr+')|']=float(len(set(x[attr] for x in dataset)))
		nb_possible_intervals*=(statistics['|dom('+attr+')|']*(statistics['|dom('+attr+')|']+1))/2.
	statistics['intervals']=nb_possible_intervals
	statistics['intervalsGood']=transform(nb_possible_intervals)

	if verbose:
		print '------------------------------------------------------------'
		for x in statistics:
			print x, ' ',statistics[x]
		print '------------------------------------------------------------'
		#raw_input('......')
	return new_dataset,positive_extent,negative_extent,alpha_ratio_class/len(dataset),statistics


#######################################################################################
def compute_index_per_attribute(dataset,attr_name,positive_extent=None,negative_extent=None):
	l_values_distinct=[];l_values_distinct_append=l_values_distinct.append
	l_indices=[];l_indices_append=l_indices.append
	l_ranges=[];l_ranges_append=l_ranges.append
	index_values_indices={}
	values_min_max_partitions=[];values_min_max_partitions_append=values_min_max_partitions.append

	


	for i in range(len(dataset)):
		row=dataset[i]
		value=row[attr_name]
		if value not in index_values_indices:
			index_values_indices[value]=set()
		index_values_indices[value]|={i}
	i=0

	for v in sorted(index_values_indices):
		s=index_values_indices[v]
		l_values_distinct_append(v)
		l_indices_append(s)
		l_ranges_append([i,i+len(s)-1])
		i=i+len(s)
		values_min_max_partitions_append([v,v])

	values_to_indices={v:k for k,v in enumerate(l_values_distinct)}
	


	if positive_extent is not None:
		#print l_values_distinct
		all_indices=sorted(values_to_indices.values())
		#print ' '
		#print all_indices
		l_values_distinct_to_consider_pos=sorted({dataset[i][attr_name] for i in positive_extent})

		l_values_distinct_to_consider_neg=sorted({dataset[i][attr_name] for i in negative_extent})
		l_indices_to_consider=[values_to_indices[x] for x in l_values_distinct_to_consider_pos]
		l_indices_to_consider_pos=[values_to_indices[x] for x in l_values_distinct_to_consider_pos]
		l_indices_to_consider_neg=[values_to_indices[x] for x in l_values_distinct_to_consider_neg]

		#print l_values_distinct
		#print [l_values_distinct[x] for x in l_indices_to_consider_pos]
		#print l_indices_to_consider
		#print [l_values_distinct[x] for x in l_indices_to_consider]
		#print l_indices_to_consider_neg

		l_indices_to_consider_set=set(l_indices_to_consider)
		l_indices_to_consider_neg_set=set(l_indices_to_consider_neg)
		pure_positives=[]
		iter_l_indices_to_consider = iter(range(len(l_indices_to_consider)))
		pure_positives.append([l_indices_to_consider[next(iter_l_indices_to_consider)]])

		
		#print l_indices_to_consider
		#print pure_positives[-1],l_values_distinct[pure_positives[-1][0]]
		#raw_input('...')
		for k in iter_l_indices_to_consider:
			#print k,l_indices_to_consider[k]
			#print l_indices_to_consider[k],l_values_distinct[l_indices_to_consider[k]]
			#print k,pure_positives,l_values_distinct[l_indices_to_consider[k]]
			if l_indices_to_consider[k] in l_indices_to_consider_neg_set :
				pure_positives.append([l_indices_to_consider[k]])
			elif l_indices_to_consider[k]-1 in l_indices_to_consider_neg_set :
				pure_positives.append([l_indices_to_consider[k]])
			else:
				pure_positives[-1].append(l_indices_to_consider[k])

			#print pure_positives[-1],l_values_distinct[pure_positives[-1][0]]
			#raw_input('...')
		#pure_positives.append([l_indices_to_consider[-1]])
		l_indices_to_consider= sorted(set([x[0] for x in pure_positives]))
		# print l_indices_to_consider_pos
		# print l_indices_to_consider_neg
		# print l_indices_to_consider
		#for k in l_indices_to_consider_neg:

		#print sorted(set(l_indices_to_consider)&set(l_indices_to_consider_neg)|set([x+1 for x in l_indices_to_consider_neg])) 
		#print pure_positives

		

		#raw_input('......')
		#print l_values_distinct_to_consider_pos
		#print ' '
		#print l_indices_to_consider
		#TODO
		
		regrouping_partitions=[]
		#print l_indices_to_consider#print l_values_distinct_to_consider_pos
		#print l_indices_to_consider_pos 
		#print [l_values_distinct[x] for x in l_indices_to_consider]
		#raw_input('....')
		

		interval=range(0,l_indices_to_consider[0])

		if len(interval):
			regrouping_partitions.append(interval)
		for k in range(len(l_indices_to_consider)):
				

			if (k==len(l_indices_to_consider)-1):
				interval=range(l_indices_to_consider[k],l_indices_to_consider_pos[-1]+1)
				if len(interval):
					regrouping_partitions.append(interval)
			else:
				interval=range(l_indices_to_consider[k],l_indices_to_consider[k]+1)
				regrouping_partitions.append(interval)
				interval=range(l_indices_to_consider[k]+1,l_indices_to_consider[k+1])
				if len(interval):
					regrouping_partitions.append(interval)
		interval=range(l_indices_to_consider_pos[-1]+1,all_indices[-1]+1)
		if len(interval):
			regrouping_partitions.append(interval)
		
		#print regrouping_partitions
		#print ' '
		regrouping_partitions_rev={k:i for i,bins in enumerate(regrouping_partitions) for k in bins}
		#print ' '
		#print regrouping_partitions_rev

		l_values_distinct_to_consider_pos=[l_values_distinct[bins[0]] for bins in sorted(regrouping_partitions)]

		#print ' '
		#print l_values_distinct_to_consider_pos 
		######################

		l_values_distinct_to_consider_pos=l_values_distinct_to_consider_pos
		values_to_indices_to_consider={ l_values_distinct[k]:i for i,bins in enumerate(regrouping_partitions) for k in bins}
		l_indices_to_consider=[set.union(*(l_indices[x] for x in bins)) for i,bins in enumerate(regrouping_partitions)]
		l_ranges_to_consider=[]
		i=0
		for k,s in enumerate(l_indices_to_consider):
			l_ranges_to_consider.append([i,i+len(s)-1])
			i=i+len(s)
			#l_ranges_to_consider=[[i,i+len(s)-1] for i,s in enumerate(l_indices_to_consider)]

		

		#########################################POSITIVE_CUT_POINTS####################################
		all_indices_of=sorted(set(values_to_indices_to_consider.values()))
		v_pos=sorted({dataset[i][attr_name] for i in positive_extent})
		v_pos_indices=[values_to_indices_to_consider[x] for x in v_pos]
		pos_indices_to_indices_to_add={}
		# if v_pos_indices[0]!=0:
		# 	pos_indices_to_indices_to_add[0]=0
		for i,c in enumerate(v_pos_indices):
			if i<len(v_pos_indices)-1:
				c_next=v_pos_indices[i+1]
				pos_indices_to_indices_to_add[c]=c if c_next==(c+1) else c+1
			else:
				c_next=all_indices_of[-1]
				pos_indices_to_indices_to_add[c]=c if c_next==c else c+1

		# print all_indices_of
		# print v_pos_indices
		# print pos_indices_to_indices_to_add
		divers={}
		divers['pos_indices_to_indices_to_add']=pos_indices_to_indices_to_add
		divers['pos_indices_to_indices_to_add_rev']={v:k for k,v in pos_indices_to_indices_to_add.iteritems()}
		divers['pos_indices']=v_pos_indices
		

		
		#########################################POSITIVE_CUT_POINTS####################################

		#print values_to_indices_to_consider
		#print len(l_ranges), len(l_values_distinct)
		#print len(l_indices_to_consider)

		#print len(l_ranges_to_consider), len(values_to_indices_to_consider)
		#print l_ranges_to_consider
		values_min_max_partitions_to_consider=[[l_values_distinct[bins[0]],l_values_distinct[bins[-1]]] for i,bins in enumerate(regrouping_partitions)]
		#print values_min_max_partitions
		#print ' '
		# print regrouping_partitions
		#print l_values_distinct_to_consider_pos
		# raw_input('.....')
		return  l_values_distinct_to_consider_pos,values_to_indices_to_consider,l_indices_to_consider,l_ranges_to_consider,values_min_max_partitions_to_consider,divers
		

	
	return l_values_distinct,values_to_indices, l_indices, l_ranges,values_min_max_partitions,{}

def compute_index_all_attributes(dataset,attributes,positive_extent=None,negative_extent=None):
	index_attr={};partitions_to_dataset={};partitions_to_dataset_bitset={}
	dataset_to_partitions=[dict(dataset[k]) for k in range(len(dataset))]
	
	for attr_name in attributes:
		l_values_distinct,values_to_indices, l_indices, l_ranges,values_min_max_partitions,divers=compute_index_per_attribute(dataset,attr_name,positive_extent=positive_extent,negative_extent=negative_extent)
		
		indices_base_partitions_flattened=[]
		for k,v in enumerate(l_indices):
			indices_base_partitions_flattened+=[k]*len(v)
		
		index_attr[attr_name]={
			'values':l_values_distinct,
			'values_to_indices':values_to_indices,
			'indices_base_partitions_flattened':indices_base_partitions_flattened,
			'ranges':l_ranges,
			'ranges_first_indices':[x[0] for x in l_ranges],
			'indices':l_indices,
			'indices_bitset':[encode_sup(l_indices[kk]) for kk in range(len(l_indices))]
		}

		for k,v in divers.iteritems():
			index_attr[attr_name][k]=v

		# if positive_extent is not None:
		# 	print divers
		# 	raw_input('....')
		


		indices_sup_inf={}
		indices_bitset_sup_inf={}

		lower_actu=set()
		upper_actu=set(range(len(dataset)))
		for kk,vv in enumerate(index_attr[attr_name]['indices']):
			#print kk
			lower_actu=lower_actu|vv
			
			indices_sup_inf[kk]={}
			indices_bitset_sup_inf[kk]={}

			indices_sup_inf[kk]['<']=set(lower_actu)
			indices_bitset_sup_inf[kk]['<']=encode_sup(indices_sup_inf[kk]['<'])
			indices_sup_inf[kk]['>']=set(upper_actu)
			indices_bitset_sup_inf[kk]['>']=encode_sup(indices_sup_inf[kk]['>'])
			
			indices_sup_inf[kk]['=']=set(vv)
			indices_bitset_sup_inf[kk]['=']=encode_sup(indices_sup_inf[kk]['='])

			indices_sup_inf[kk]['val_min']=values_min_max_partitions[kk][0]#l_values_distinct[kk]
			indices_sup_inf[kk]['val_max']=values_min_max_partitions[kk][1]#l_values_distinct[kk]

			upper_actu=upper_actu-lower_actu
		
		index_attr[attr_name]['indices_sup_inf']=indices_sup_inf
		index_attr[attr_name]['indices_bitset_sup_inf']=indices_bitset_sup_inf

		# partitions_to_dataset[attr_name]={kk:vv for kk,vv in enumerate(index_attr[attr_name]['indices'])}
		# partitions_to_dataset_bitset[attr_name]={kk:encode_sup(vv) for kk,vv in partitions_to_dataset[attr_name].iteritems()}

		partitions_to_dataset[attr_name]=index_attr[attr_name]['indices_sup_inf']
		partitions_to_dataset_bitset[attr_name]=index_attr[attr_name]['indices_bitset_sup_inf']


		values_to_indices=index_attr[attr_name]['values_to_indices']
		for i in range(len(dataset)):
			dataset_to_partitions[i][attr_name]=values_to_indices[dataset_to_partitions[i][attr_name]]

	return index_attr,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset
#######################################################################################


def discretize(dataset,attr_name,index_attr,nb=3,positive_extent=None,hardcore_optimize=True):
	ranges_first_indices=index_attr[attr_name]['ranges_first_indices']
	values_to_indices=index_attr[attr_name]['values_to_indices']
	indices_base_partitions_flattened=index_attr[attr_name]['indices_base_partitions_flattened']
	values=index_attr[attr_name]['values']
	




	#ranges_full=index_attr[attr_name]['ranges_full']
	nb=min(nb,len(ranges_first_indices))
	
	len_dataset=len(dataset)*1.
	cut_indices=[(k/float(nb))*len_dataset for k in range(nb)]

	#print indices_base_partitions_flattened
	###OLD######
	#cut_points_indices=sorted(set([bisect_left(ranges_first_indices,ic) for ic in cut_indices]))
	###OLD######
	####NEW#####
	cut_points_indices=[indices_base_partitions_flattened[int(k)] for k in cut_indices]
	cut_points_indices=sorted(set(cut_points_indices))
	#print cut_points_indices,values
	if len(cut_points_indices)<nb:
		cut_indices=[(k/float(nb))*len(values) for k in range(nb)]
		cut_points_indices=[values_to_indices[values[int(k)]] for k in cut_indices]
		cut_points_indices=sorted(set(cut_points_indices))



	if hardcore_optimize and positive_extent is not None:
		pos_indices_to_indices_to_add=index_attr[attr_name]['pos_indices_to_indices_to_add']
		pos_indices_to_indices_to_add_rev=index_attr[attr_name]['pos_indices_to_indices_to_add_rev']
		pos_indices= index_attr[attr_name]['pos_indices']
		# print values
		# print [values[k] for k in pos_indices]
		#cut_indices_x=[int((k/float(nb))*len(dataset)*1.) for k in range(nb)]
		#cut_indices_x=[indices_base_partitions_flattened[int(k)] for k in cut_indices_x]
		cut_indices_x=set(cut_points_indices) | {0,pos_indices[0],pos_indices[-1]}
		#print pos_indices_to_indices_to_add
		#pos_indices_to_indices_to_add_rev[0]=pos_indices_to_indices_to_add_rev.get(0,0)
		cut_indices_x_augmented=sorted(set(cut_indices_x)|set([pos_indices_to_indices_to_add[x] if x in pos_indices_to_indices_to_add else pos_indices_to_indices_to_add_rev.get(x,0) for x in cut_indices_x]))
		#print cut_points_indices
		
		# print cut_indices_x_augmented
		# print cut_points_indices
		# raw_input('......')
		#TODO!
		cut_points_indices=cut_indices_x_augmented
		#cut_points_indices=cut_indices_x_augmented
		#print cut_indices_x
		#print [indices_base_partitions_flattened[int(k)] for k in cut_indices_x]
		#raw_input('....')

	cut_points_values=[values[k] for k in cut_points_indices]
	#print cut_points_values
	partitions_indices={}
	partitions_values={}
	len_cut_points_indices=len(cut_points_indices)-1
	for k in range(len(cut_points_indices)):
		next_k=cut_points_indices[k+1] if k<len_cut_points_indices else len(values)
		partitions_indices[cut_points_indices[k]]=range(cut_points_indices[k],next_k)
		partitions_values[cut_points_indices[k]]=[values[x] for x in partitions_indices[cut_points_indices[k]]]
	



	return cut_points_indices,cut_points_values,partitions_indices,partitions_values
# def discretize(dataset,index_attr,attributes): #Standard Three By Attribute

def discretize_all_attributes(dataset,attributes,index_attr,nb=3,positive_extent=None):
	attributes_discretized={}
	for attr_name in attributes:
		values=index_attr[attr_name]['values']
		values_to_indices=index_attr[attr_name]['values_to_indices']
		cut_points_indices,cut_points_values,partitions_indices,partitions_values=discretize(dataset,attr_name,index_attr,nb,positive_extent=positive_extent)
		
		
		#TODO
		if positive_extent==None:
			remaining_cut_points_indices=sorted(set(values_to_indices.values()) - set(cut_points_indices))
		else:
			
			
			remaining_cut_points_indices=sorted(set(values_to_indices.values()) - set(cut_points_indices))


			
			# v_pos=sorted({dataset[i][attr_name] for i in positive_extent})
			# v_pos_indices=[values_to_indices[x] for x in v_pos]
			# remaining_cut_points_indices=sorted(set(v_pos_indices) - set(cut_points_indices))
		# print cut_points_indices,remaining_cut_points_indices
		# raw_input('........')
		# if positive_extent:
		# 	l_values_distinct_to_consider_pos=sorted({dataset[i][attr_name] for i in positive_extent})
		# 	indices_points_to_consider=


		partitions_dead=set(k for k in cut_points_indices if len(partitions_indices[k])==1)
		attributes_discretized[attr_name]={
			'cut_points_indices':cut_points_indices,
			'remaining_cut_points_indices':remaining_cut_points_indices,
			'cut_points_values':cut_points_values,
			'partitions_indices':partitions_indices,
			'partitions_values':partitions_values,
			'partitions_dead':partitions_dead
		}
		# print attr_name
		# print partitions_values
		# print index_attr[attr_name]['values']
		# print attr_name
		# print '\t\t','cut_points_indices : ',cut_points_indices
		# print '\t\t','remaining_cut_points_indices : ',remaining_cut_points_indices
		# print '\t\t','cut_points_values : ',cut_points_values
		# print '\t\t','partitions_indices : ',partitions_indices
		# print '\t\t','partitions_values : ',partitions_values
		# raw_input('....')
		# indices_base_partitions_flattened=index_attr[attr_name]['indices_base_partitions_flattened']
		# dictas={}
		# for k in indices_base_partitions_flattened:
		# 	dictas[k]=dictas.get(k,0.)+1
		# for k in dictas:
		# 	dictas[k]=dictas[k]/float(len(indices_base_partitions_flattened))
		# dictas_l=[]
		# for k in sorted(dictas):
		# 	if len(dictas_l)==0:
		# 		dictas_l.append(dictas[k])
		# 	else:
		# 		dictas_l.append(dictas_l[-1]+dictas[k])
		# print '\t\t',index_attr[attr_name]['indices_base_partitions_flattened']
		# print '\t\t',dictas_l
		# raw_input('..-------------..')
	#print attributes_discretized
	return  attributes_discretized
# 	return 



def image_partition_by_descritization_all_attributes(dataset,attributes,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes_discretized):
	dataset_to_partitions_new=[];dataset_to_partitions_new_append=dataset_to_partitions_new.append
	partitions_to_dataset_new={}
	partitions_to_dataset_bitset_new={}
	attributes_old_to_new_partitions={}
	for attr_name in attributes:
		partitions_indices=attributes_discretized[attr_name]['partitions_indices']
		partitions_attr_old_to_new=[]
		partitions_to_dataset_new[attr_name]={}
		partitions_to_dataset_bitset_new[attr_name]={}
		for k in sorted(partitions_indices):
			v=partitions_indices[k]
			partitions_attr_old_to_new+=[k]*len(v)

			partitions_to_dataset_new[attr_name][k]={}
			partitions_to_dataset_bitset_new[attr_name][k]={}

			min_v=min(v)
			max_v=max(v)

			partitions_to_dataset_new[attr_name][k]['=']=partitions_to_dataset[attr_name][min_v]['>']&partitions_to_dataset[attr_name][max_v]['<']#(set.union(*(partitions_to_dataset[attr_name][x]['='] for x in v)))
			partitions_to_dataset_bitset_new[attr_name][k]['=']=partitions_to_dataset_bitset[attr_name][min_v]['>']&partitions_to_dataset_bitset[attr_name][max_v]['<']#reduce(or_,(partitions_to_dataset_bitset[attr_name][x]['='] for x in v))

			partitions_to_dataset_new[attr_name][k]['>']=set(partitions_to_dataset[attr_name][min_v]['>'])
			partitions_to_dataset_bitset_new[attr_name][k]['>']=partitions_to_dataset_bitset[attr_name][min_v]['>']

			partitions_to_dataset_new[attr_name][k]['<']=set(partitions_to_dataset[attr_name][max_v]['<'])
			partitions_to_dataset_bitset_new[attr_name][k]['<']=partitions_to_dataset_bitset[attr_name][max_v]['<']

			partitions_to_dataset_new[attr_name][k]['val_min']=partitions_to_dataset[attr_name][min_v]['val_min']
			partitions_to_dataset_new[attr_name][k]['val_max']=partitions_to_dataset[attr_name][max_v]['val_max']

		attributes_old_to_new_partitions[attr_name]=partitions_attr_old_to_new
		
	

	for k in range(len(dataset_to_partitions)):
		row=dict(dataset_to_partitions[k])
		for attr_name in attributes:
			row[attr_name]=attributes_old_to_new_partitions[attr_name][row[attr_name]]
		dataset_to_partitions_new_append(row)
	return dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new


#################################################################################################################################################

def children_numeric(interval,actual_support,actual_support_bitset,index_values,index_values_bitset,refinement_index=0):
	if len(interval)>1:
		arr_left=interval[1:]
		arr_right=interval[:-1]
		possible_children=[[arr_left,arr_right],[0,1]]
		for k in range(refinement_index,2):
			if k==0:
				to_remove=interval[0]
			else:
				to_remove=interval[-1]
			sup_to_ret=actual_support-index_values[to_remove]['=']
			sup_to_ret_bitset=actual_support_bitset&~index_values_bitset[to_remove]['=']
			if len(sup_to_ret):
				yield possible_children[0][k],possible_children[1][k],sup_to_ret,sup_to_ret_bitset

# def enumerator_numeric(domain,pattern,refinement_index=0):
# 	yield pattern
# 	for child,refin_child in children_numeric(domain,pattern,refinement_index):
# 		for child_pattern in enumerator_numeric(domain,child,refin_child):
# 			yield child_pattern


def respect_order(intervals,intervals_closed,refinement_index):
	
	for k in range(0,refinement_index):
		if intervals[k][0]!=intervals_closed[k][0] or intervals[k][-1]!=intervals_closed[k][-1]:
			return False

	return True


def closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,support_indices):
	enum_attrs=list(enumerate(attributes))
	intervals_closed_values=[set() for k in range(len(enum_attrs))]
	enum_attrs_zippas=zip(enum_attrs,[(x,x.add) for x in intervals_closed_values])
	for i in support_indices:
		elem=dataset_to_partitions[i]
		for (k,a),(s,s_add) in enum_attrs_zippas:
			elem_a=elem[a]
			if elem_a not in s:
				s_add(elem_a)
	intervals_closed=[sorted(intervals_closed_values[k]) for k in range(len(intervals_closed_values))]
	return intervals_closed




def closed_numeric_by_desc_combined(dataset_to_partitions,partitions_to_dataset,attributes,support_indices,intervals,refinement_index):
	#return closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,support_indices)
	interval_closed=[];interval_closed_append=interval_closed.append
	#enum_intervals=enumerate(intervals)
	

	for i in range(refinement_index):
		interval=intervals[i]
		new_interval=interval[:]
		interval_closed_append(new_interval)
		interval_min=new_interval[0]
		interval_max=new_interval[-1]
		attr_concerned=attributes[i]
		partitions_attr_concerned=partitions_to_dataset[attr_concerned]
		if i<refinement_index:
			if len(support_indices&partitions_attr_concerned[interval_min]['='])==0:
				new_interval.pop(0)
				return interval_closed
			if len(support_indices&partitions_attr_concerned[interval_max]['='])==0:
				new_interval.pop()
				return interval_closed
		
	interval_closed=interval_closed+closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes[refinement_index:],support_indices)
	

		# else:
		# 	while len(support_indices&partitions_attr_concerned[interval_min]['='])==0:
		# 		new_interval.pop(0)
		# 		interval_min=new_interval[0]
		# 	while len(support_indices&partitions_attr_concerned[interval_max]['='])==0:
		# 		new_interval.pop()
		# 		interval_max=new_interval[-1]
		
	return interval_closed
	
	# enum_attrs=list(enumerate(attributes))
	# intervals_closed_values=[set() for k in range(len(enum_attrs))]
	# enum_attrs_zippas=zip(enum_attrs,[(x,x.add) for x in intervals_closed_values])
	# for i in support_indices:
	# 	elem=dataset_to_partitions[i]
	# 	for (k,a),(s,s_add) in enum_attrs_zippas:
	# 		elem_a=elem[a]
	# 		if elem_a not in s:
	# 			s_add(elem_a)
	# intervals_closed=[sorted(intervals_closed_values[k]) for k in range(len(intervals_closed_values))]
	# return intervals_closed


def compute_support_interval_OLD(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals):
	ret_sup=None
	ret_sup_bitset=0
	for ia,a in enumerate(attributes):
		interval=intervals[ia]
		if ret_sup is None:
			ret_sup=set.union(*(partitions_to_dataset[a][x] for x in interval))
			ret_sup_bitset=reduce(or_,(partitions_to_dataset_bitset[a][x] for x in interval))
		else:
			ret_sup&=set.union(*(partitions_to_dataset[a][x] for x in interval))
			ret_sup_bitset&=reduce(or_,(partitions_to_dataset_bitset[a][x] for x in interval))
			#reduce(set.union,(partitions_to_dataset[a][x] for x in interval))
	return ret_sup,ret_sup_bitset

def compute_support_interval(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,current_refinement_index=float('inf')):
	ret_sup=None
	ret_sup_bitset=0

	enum_attributes=enumerate(attributes)
	
	ia,a=next(enum_attributes)
	interval=intervals[ia]
	interval_min=interval[0]
	interval_max=interval[-1]
	#one_dead_cell=interval_min==interval_max
	partitions_to_dataset_a=partitions_to_dataset[a]
	
	if interval_min==interval_max:
		ret_sup=set()|partitions_to_dataset_a[interval_min]['=']
	else:
		ret_sup=(partitions_to_dataset_a[interval_min]['>']&partitions_to_dataset_a[interval_max]['<'])
	for ia,a in enum_attributes:
		if ia>current_refinement_index:
			break
		interval=intervals[ia]
		interval_min=interval[0]
		interval_max=interval[-1]
		partitions_to_dataset_a=partitions_to_dataset[a]
		if interval_min==interval_max:
			ret_sup&=partitions_to_dataset_a[interval_min]['=']
		else:
			ret_sup&=(partitions_to_dataset_a[interval_min]['>']&partitions_to_dataset_a[interval_max]['<'])

	return ret_sup,ret_sup_bitset



def enumerator_on_partitions(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,support_indices,support_indices_bitset,attrs_refinement_indexes,refinement_index,continue_dict={'continue':True}):
	
	intervals_closed=closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,support_indices)
	if respect_order(intervals,intervals_closed,refinement_index):
		continue_dict_copy=continue_dict.copy()
		yield intervals_closed,support_indices,support_indices_bitset,refinement_index,continue_dict_copy
		if continue_dict_copy['continue']:
			for k in range(refinement_index,len(attributes)):
				index_values=partitions_to_dataset[attributes[k]]
				index_values_bitset=partitions_to_dataset_bitset[attributes[k]]
				actual_support=support_indices
				actual_support_bitset=support_indices_bitset
				for child,refin_child,child_attr_support,child_attr_support_bitset in children_numeric(intervals_closed[k],actual_support,actual_support_bitset,index_values,index_values_bitset,attrs_refinement_indexes[k]):
					attrs_refinement_indexes_new=attrs_refinement_indexes[:]
					intervals_new=intervals_closed[:]
					attrs_refinement_indexes_new[k]=refin_child
					intervals_new[k]=child
					for child_intervals,child_support,child_support_bitset,child_refienemt_index,child_continue_dict in enumerator_on_partitions(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals_new,child_attr_support,child_attr_support_bitset,attrs_refinement_indexes_new,k,continue_dict_copy):
						yield child_intervals,child_support,child_support_bitset,child_refienemt_index,child_continue_dict
	
def enumerator_on_partitions_init(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,indices_to_start_with=None,info_to_push={}):
	refinement_index=0
	intervals=[sorted(partitions_to_dataset[attr_name]) for attr_name in attributes]
	attrs_refinement_indexes=[0 for k in range(len(attributes))]
	if indices_to_start_with is None:
		indices_to_start_with=set(range(len(dataset_to_partitions)))
	refinement_index=0
	indices_bitset_to_start_with=encode_sup(indices_to_start_with)
	info_to_push['continue']=True
	if len(indices_to_start_with):
		intervals=closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,indices_to_start_with)
		
		for pattern,sup,sup_bitset,current_refinement_index,continue_dict in enumerator_on_partitions(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,indices_to_start_with,indices_bitset_to_start_with,attrs_refinement_indexes,refinement_index,info_to_push):
			#yieldous=pattern_to_yield(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,pattern)
			yieldous=pattern
			yield yieldous,sup,sup_bitset,current_refinement_index,continue_dict




def children_numeric_particular(interval,actual_support,actual_support_bitset,actual_support_full,actual_support_full_bitset,index_values,index_values_bitset,refinement_index=0,threshold_sup=1,fixed_left=False,fixed_right=False):
	if len(interval)>1:
		arr_left=interval[1:]
		arr_right=interval[:-1]
		possible_children=[[arr_left,arr_right],[0,1]]
		
		for k in range(refinement_index,2):
			if fixed_left and k==0:
				continue
			if fixed_right and k==1:
				continue
			if k==0:
				to_remove=interval[0]
				to_keep_full=arr_left[0]
				to_keep_inequality='>'
			else:
				to_remove=interval[-1]
				to_keep_full=arr_right[-1]
				to_keep_inequality='<'

			sup_to_ret=actual_support-index_values[to_remove]['=']
			if len(sup_to_ret)>=threshold_sup:
				sup_to_ret_bitset=actual_support_bitset&~index_values_bitset[to_remove]['=']
				sup_full_to_ret=actual_support_full&index_values[to_keep_full][to_keep_inequality]  #actual_support_full-index_values[to_remove]['=']
				sup_full_to_ret_bitset=actual_support_full_bitset&index_values_bitset[to_keep_full][to_keep_inequality]
				yield possible_children[0][k],possible_children[1][k],sup_to_ret,sup_to_ret_bitset,sup_full_to_ret,sup_full_to_ret_bitset


def children_numeric_particular_bitset_old(interval,actual_support,actual_support_bitset,actual_support_full,actual_support_full_bitset,index_values,index_values_bitset,refinement_index=0,threshold_sup=1,fixed_left=False,fixed_right=False):
	ZERO=children_numeric_particular_bitset_old.ZERO
	if len(interval)>1:
		arr_left=interval[1:]
		arr_right=interval[:-1]
		possible_children=[[arr_left,arr_right],[0,1]]
		
		for k in range(refinement_index,2):
			# if fixed_left and k==0:
			# 	continue
			# if fixed_right and k==1:
			# 	continue
			if k==0:
				if fixed_left:
					continue
				to_remove=interval[0]
				to_keep_full=arr_left[0]
				to_keep_inequality='>'
			else:
				if fixed_right:
					continue
				to_remove=interval[-1]
				to_keep_full=arr_right[-1]
				to_keep_inequality='<'

			#sup_to_ret_bitset=actual_support_bitset&~index_values_bitset[to_remove]['=']
			sup_to_ret_bitset=actual_support_bitset-index_values_bitset[to_remove]['=']
			sup_to_ret=set()#actual_support-index_values[to_remove]['=']
			if sup_to_ret_bitset!=ZERO:
				if threshold_sup>1 and len(sup_to_ret_bitset)<threshold_sup:
					continue
				sup_full_to_ret=set()#actual_support_full&index_values[to_keep_full][to_keep_inequality] 
				sup_full_to_ret_bitset=actual_support_full_bitset&index_values_bitset[to_keep_full][to_keep_inequality]
				yield possible_children[0][k],possible_children[1][k],sup_to_ret,sup_to_ret_bitset,sup_full_to_ret,sup_full_to_ret_bitset
children_numeric_particular_bitset_old.ZERO=intbitset()

def children_numeric_particular_bitset(interval,actual_support,actual_support_bitset,actual_support_full,actual_support_full_bitset,index_values,index_values_bitset,refinement_index=0,threshold_sup=1,fixed_left=False,fixed_right=False):
	ZERO=children_numeric_particular_bitset.ZERO
	if len(interval)>1:
		arr_left=interval[1:]
		arr_right=interval[:-1]
		possible_children=[[arr_left,arr_right],[0,1]]
		
		for k in range(refinement_index,2):
			# if fixed_left and k==0:
			# 	continue
			# if fixed_right and k==1:
			# 	continue
			if k==0:
				if fixed_left:
					continue
				to_remove=interval[0]
				to_keep_full=arr_left[0]
				to_keep_inequality='>'
			else:
				if fixed_right:
					continue
				to_remove=interval[-1]
				to_keep_full=arr_right[-1]
				to_keep_inequality='<'

			#sup_to_ret_bitset=actual_support_bitset&~index_values_bitset[to_remove]['=']
			sup_to_ret_bitset=actual_support_bitset-index_values_bitset[to_remove]['=']
			sup_to_ret=set()#actual_support-index_values[to_remove]['=']
			if sup_to_ret_bitset!=ZERO:
				if threshold_sup>1 and len(sup_to_ret_bitset)<threshold_sup:
					continue
				sup_full_to_ret=set()#actual_support_full&index_values[to_keep_full][to_keep_inequality] 
				sup_full_to_ret_bitset=actual_support_full_bitset&index_values_bitset[possible_children[0][k][0]]['>']&index_values_bitset[possible_children[0][k][-1]]['<']
				yield possible_children[0][k],possible_children[1][k],sup_to_ret,sup_to_ret_bitset,sup_full_to_ret,sup_full_to_ret_bitset
children_numeric_particular_bitset.ZERO=intbitset()

def respect_order_particular(intervals,intervals_closed,refinement_index):
	
	for k in range(0,refinement_index):
		if intervals[k][0]!=intervals_closed[k][0] or intervals[k][-1]!=intervals_closed[k][-1]:
			return False,[],False
	new_borders_that_changed=[[None,None] for x in range(len(intervals))]
	bool_borders_changed=False
	for k in range(refinement_index,len(intervals)):
		if intervals[k][0]!=intervals_closed[k][0]:
			new_borders_that_changed[k][0]=intervals_closed[k][0]
			bool_borders_changed=True
		if intervals[k][-1]!=intervals_closed[k][-1]:
			new_borders_that_changed[k][-1]=intervals_closed[k][-1]
			bool_borders_changed=True
	return True,new_borders_that_changed,bool_borders_changed




def closed_numeric_by_desc(dataset_to_partitions,partitions_to_dataset,attributes,support_indices,intervals,refinement_index):
	#return closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,support_indices)
	interval_closed=[];interval_closed_append=interval_closed.append
	
	for i,interval in enumerate(intervals):
		new_interval=interval[:]
		interval_closed_append(new_interval)
		interval_min=new_interval[0]
		interval_max=new_interval[-1]
		attr_concerned=attributes[i]
		partitions_attr_concerned=partitions_to_dataset[attr_concerned]
		if i<refinement_index:
			if len(support_indices&partitions_attr_concerned[interval_min]['='])==0:
				new_interval.pop(0)
				return interval_closed
			if len(support_indices&partitions_attr_concerned[interval_max]['='])==0:
				new_interval.pop()
				return interval_closed
		else:
			while len(support_indices&partitions_attr_concerned[interval_min]['='])==0:
				new_interval.pop(0)
				interval_min=new_interval[0]
			while len(support_indices&partitions_attr_concerned[interval_max]['='])==0:
				new_interval.pop()
				interval_max=new_interval[-1]
		
	return interval_closed



def closed_numeric_by_desc_bitset(dataset_to_partitions,partitions_to_dataset_bitset,attributes,support_bitset,intervals,refinement_index):
	#return closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,support_indices)
	interval_closed=[];interval_closed_append=interval_closed.append
	ZERO=closed_numeric_by_desc_bitset.ZERO
	for i,interval in enumerate(intervals):
		new_interval=interval[:];new_interval_pop=new_interval.pop
		interval_closed_append(new_interval)
		interval_min=new_interval[0]
		interval_max=new_interval[-1]
		attr_concerned=attributes[i]
		partitions_attr_concerned=partitions_to_dataset_bitset[attr_concerned]
		if i<refinement_index:
			if support_bitset&partitions_attr_concerned[interval_min]['=']==ZERO:
				new_interval_pop(0)
				return interval_closed,False
			if support_bitset&partitions_attr_concerned[interval_max]['=']==ZERO:
				new_interval_pop()
				return interval_closed,False
		else:
			while support_bitset&partitions_attr_concerned[interval_min]['=']==ZERO:
				new_interval_pop(0)
				interval_min=new_interval[0]
			while support_bitset&partitions_attr_concerned[interval_max]['=']==ZERO:
				new_interval_pop()
				interval_max=new_interval[-1]
		
	return interval_closed,True
closed_numeric_by_desc_bitset.ZERO=intbitset()

def enumerator_on_partitions_particular(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,support_indices,support_indices_bitset,support_full_indices,support_full_indices_bitset,attrs_refinement_indexes,refinement_index,continue_dict={'continue':True},threshold_sup=1,fixed_left=False,fixed_right=False):
	
	int_left_old=intervals[0][0]
	int_right_old=intervals[0][-1]

	intervals_closed=closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,support_indices)
	#intervals_closed=closed_numeric_by_desc_combined(dataset_to_partitions,partitions_to_dataset,attributes,support_indices,intervals,refinement_index)
	int_left_new=intervals_closed[0][0]
	int_right_new=intervals_closed[0][-1]

	LAUNCH_ENUM=True

	if fixed_left:
		LAUNCH_ENUM=(int_left_new==int_left_old)
	elif fixed_right:
		LAUNCH_ENUM=(int_right_new==int_right_old)

	if LAUNCH_ENUM:


		#intervals_closed=closed_numeric_by_desc(dataset_to_partitions,partitions_to_dataset,attributes,support_indices,intervals,refinement_index)
		resp_ord,new_borders_that_changed,bool_borders_changed=respect_order_particular(intervals,intervals_closed,refinement_index)
		if resp_ord:
			
			if bool_borders_changed:
				for k in range(refinement_index,len(intervals)):
					new_borders_that_changed_k=new_borders_that_changed[k]
					new_borders_that_changed_k_left=new_borders_that_changed_k[0]
					new_borders_that_changed_k_right=new_borders_that_changed_k[1]
					if new_borders_that_changed_k_left is not None:
						support_full_indices&=partitions_to_dataset[attributes[k]][new_borders_that_changed_k_left]['>']
						support_full_indices_bitset&=partitions_to_dataset_bitset[attributes[k]][new_borders_that_changed_k_left]['>']
					if new_borders_that_changed_k_right is not None:
						support_full_indices&=partitions_to_dataset[attributes[k]][new_borders_that_changed_k_right]['<']
						support_full_indices_bitset&=partitions_to_dataset_bitset[attributes[k]][new_borders_that_changed_k_right]['<']

			continue_dict_copy=continue_dict.copy()
			yield intervals_closed,support_indices,support_indices_bitset,support_full_indices,support_full_indices_bitset,refinement_index,continue_dict_copy
			if continue_dict_copy['continue']:
				for k in range(refinement_index,len(attributes)):
					index_values=partitions_to_dataset[attributes[k]]
					index_values_bitset=partitions_to_dataset_bitset[attributes[k]]
					actual_support=support_indices
					actual_support_bitset=support_indices_bitset
					actual_support_full=support_full_indices
					actual_support_full_bitset=support_full_indices_bitset
					for child,refin_child,child_attr_support,child_attr_support_bitset,child_attr_support_full,child_attr_support_full_bitset in children_numeric_particular(intervals_closed[k],actual_support,actual_support_bitset,actual_support_full,actual_support_full_bitset,index_values,index_values_bitset,attrs_refinement_indexes[k],threshold_sup=threshold_sup,fixed_left=fixed_left&(k==0),fixed_right=fixed_right&(k==0)):
						attrs_refinement_indexes_new=attrs_refinement_indexes[:]
						intervals_new=intervals_closed[:]
						attrs_refinement_indexes_new[k]=refin_child
						intervals_new[k]=child
						for child_intervals,child_support,child_support_bitset,child_support_full,child_support_full_bitset,child_refinement_index,child_continue_dict in enumerator_on_partitions_particular(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals_new,child_attr_support,child_attr_support_bitset,child_attr_support_full,child_attr_support_full_bitset,attrs_refinement_indexes_new,k,continue_dict_copy,threshold_sup=threshold_sup,fixed_left=fixed_left,fixed_right=fixed_right):
							yield child_intervals,child_support,child_support_bitset,child_support_full,child_support_full_bitset,child_refinement_index,child_continue_dict



# def adjust(partitions_to_dataset_bitset,new_borders_that_changed,support_full_indices_bitset,intervals):
# 	for k in range(refinement_index,len(intervals)):
# 		new_borders_that_changed_k=new_borders_that_changed[k]
# 		new_borders_that_changed_k_left=new_borders_that_changed_k[0]
# 		new_borders_that_changed_k_right=new_borders_that_changed_k[1]
# 		if new_borders_that_changed_k_left is not None:
# 			#support_full_indices&=partitions_to_dataset[attributes[k]][new_borders_that_changed_k_left]['>']
# 			#support_full_indices=
# 			support_full_indices_bitset&=partitions_to_dataset_bitset[attributes[k]][new_borders_that_changed_k_left]['>']
# 		if new_borders_that_changed_k_right is not None:
# 			#support_full_indices&=partitions_to_dataset[attributes[k]][new_borders_that_changed_k_right]['<']
# 			support_full_indices_bitset&=partitions_to_dataset_bitset[attributes[k]][new_borders_that_changed_k_right]['<']
# 	return support_full_indices_bitset

def enumerator_on_partitions_particular_bitset(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,support_indices,support_indices_bitset,support_full_indices,support_full_indices_bitset,attrs_refinement_indexes,refinement_index,continue_dict={'continue':True},threshold_sup=1,fixed_left=False,fixed_right=False):
	
	int_left_old=intervals[0][0]
	int_right_old=intervals[0][-1]

	#intervals_closed=closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,support_indices)
	intervals_closed,_=closed_numeric_by_desc_bitset(dataset_to_partitions,partitions_to_dataset_bitset,attributes,support_indices_bitset,intervals,refinement_index)
	#intervals_closed=closed_numeric_by_desc_combined(dataset_to_partitions,partitions_to_dataset,attributes,support_indices,intervals,refinement_index)
	int_left_new=intervals_closed[0][0]
	int_right_new=intervals_closed[0][-1]

	LAUNCH_ENUM=True

	if fixed_left:
		LAUNCH_ENUM=(int_left_new==int_left_old)
	elif fixed_right:
		LAUNCH_ENUM=(int_right_new==int_right_old)

	if LAUNCH_ENUM:


		#intervals_closed=closed_numeric_by_desc(dataset_to_partitions,partitions_to_dataset,attributes,support_indices,intervals,refinement_index)
		resp_ord,new_borders_that_changed,bool_borders_changed=respect_order_particular(intervals,intervals_closed,refinement_index)
		if resp_ord:
			
			if bool_borders_changed:
				for k in range(refinement_index,len(intervals)):
					new_borders_that_changed_k=new_borders_that_changed[k]
					new_borders_that_changed_k_left=new_borders_that_changed_k[0]
					new_borders_that_changed_k_right=new_borders_that_changed_k[1]
					if new_borders_that_changed_k_left is not None:
						#support_full_indices&=partitions_to_dataset[attributes[k]][new_borders_that_changed_k_left]['>']
						#support_full_indices=
						support_full_indices_bitset&=partitions_to_dataset_bitset[attributes[k]][new_borders_that_changed_k_left]['>']
					if new_borders_that_changed_k_right is not None:
						#support_full_indices&=partitions_to_dataset[attributes[k]][new_borders_that_changed_k_right]['<']
						support_full_indices_bitset&=partitions_to_dataset_bitset[attributes[k]][new_borders_that_changed_k_right]['<']

			continue_dict_copy=continue_dict.copy()
			yield intervals_closed,support_indices,support_indices_bitset,support_full_indices,support_full_indices_bitset,refinement_index,continue_dict_copy
			if continue_dict_copy['continue']:
				for k in range(refinement_index,len(attributes)):
					index_values=partitions_to_dataset[attributes[k]]
					index_values_bitset=partitions_to_dataset_bitset[attributes[k]]
					actual_support=support_indices
					actual_support_bitset=support_indices_bitset
					actual_support_full=support_full_indices
					actual_support_full_bitset=support_full_indices_bitset
					for child,refin_child,child_attr_support,child_attr_support_bitset,child_attr_support_full,child_attr_support_full_bitset in children_numeric_particular_bitset(intervals_closed[k],actual_support,actual_support_bitset,actual_support_full,actual_support_full_bitset,index_values,index_values_bitset,attrs_refinement_indexes[k],threshold_sup=threshold_sup,fixed_left=fixed_left&(k==0),fixed_right=fixed_right&(k==0)):
						attrs_refinement_indexes_new=attrs_refinement_indexes[:]
						intervals_new=intervals_closed[:]
						attrs_refinement_indexes_new[k]=refin_child
						intervals_new[k]=child
						for child_intervals,child_support,child_support_bitset,child_support_full,child_support_full_bitset,child_refinement_index,child_continue_dict in enumerator_on_partitions_particular_bitset(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals_new,child_attr_support,child_attr_support_bitset,child_attr_support_full,child_attr_support_full_bitset,attrs_refinement_indexes_new,k,continue_dict_copy,threshold_sup=threshold_sup,fixed_left=fixed_left,fixed_right=fixed_right):
							yield child_intervals,child_support,child_support_bitset,child_support_full,child_support_full_bitset,child_refinement_index,child_continue_dict


def respect_order_particular_bitset(intervals,intervals_closed,refinement_index,partitions_to_dataset_bitset,attributes,support_full_indices_bitset):
	
	for k in range(0,refinement_index):
		if intervals[k][0]!=intervals_closed[k][0] or intervals[k][-1]!=intervals_closed[k][-1]:
			return False,0

	for k in range(refinement_index,len(intervals)):
		left_closed=intervals_closed[k][0]
		right_closed=intervals_closed[k][-1]
		if intervals[k][0]!=left_closed:
			support_full_indices_bitset&=partitions_to_dataset_bitset[attributes[k]][left_closed]['>']
		if intervals[k][-1]!=right_closed:
			support_full_indices_bitset&=partitions_to_dataset_bitset[attributes[k]][right_closed]['<']
	return True,support_full_indices_bitset


def closed_numeric_by_desc_bitset_last(dataset_to_partitions,partitions_to_dataset_bitset,attributes,support_bitset,support_full_indices_bitset,intervals,refinement_index,fixed_left=False,fixed_right=False):
	#return closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,support_indices)
	interval_closed=[];interval_closed_append=interval_closed.append
	ZERO=closed_numeric_by_desc_bitset.ZERO
	for i,interval in enumerate(intervals):
		new_interval=interval[:];new_interval_pop=new_interval.pop
		interval_closed_append(new_interval)
		interval_min=new_interval[0]
		interval_max=new_interval[-1]
		attr_concerned=attributes[i]
		partitions_attr_concerned=partitions_to_dataset_bitset[attr_concerned]
		if i<refinement_index:
			if support_bitset&partitions_attr_concerned[interval_min]['=']==ZERO:
				new_interval_pop(0)
				return interval_closed,support_full_indices_bitset,False
			if support_bitset&partitions_attr_concerned[interval_max]['=']==ZERO:
				new_interval_pop()
				return interval_closed,support_full_indices_bitset,False
		else:
			
			b=False
			while support_bitset&partitions_attr_concerned[interval_min]['=']==ZERO:
				new_interval_pop(0)
				interval_min=new_interval[0]
				b=True
			if b: support_full_indices_bitset&=partitions_to_dataset_bitset[attr_concerned][interval_min]['>']
			
			b=False
			while support_bitset&partitions_attr_concerned[interval_max]['=']==ZERO:
				new_interval_pop()
				interval_max=new_interval[-1]
				b=True
			if b: support_full_indices_bitset&=partitions_to_dataset_bitset[attr_concerned][interval_max]['<']
	
	return interval_closed,support_full_indices_bitset,True
closed_numeric_by_desc_bitset.ZERO=intbitset()


def enumerator_on_partitions_particular_bitset_iterativement(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,support_indices,support_indices_bitset,support_full_indices,support_full_indices_bitset,attr_refinement_index,refinement_index,continue_dict={'continue':True},threshold_sup=1,fixed_left=False,fixed_right=False):
	PILE=[]
	PILE_POP=PILE.pop;PILE_PUSH=PILE.append
	PILE_PUSH((intervals,support_indices,support_indices_bitset,support_full_indices,support_full_indices_bitset,attr_refinement_index,refinement_index,continue_dict))
	nb_attributes=len(attributes)

	while len(PILE):
		intervals,support_indices,support_indices_bitset,support_full_indices,support_full_indices_bitset,attr_refinement_index,refinement_index,continue_dict=PILE_POP()

		intervals_closed,support_full_indices_bitset,respect=closed_numeric_by_desc_bitset_last(dataset_to_partitions,partitions_to_dataset_bitset,attributes,support_indices_bitset,support_full_indices_bitset,intervals,refinement_index,fixed_left=fixed_left,fixed_right=fixed_right)
		if respect:
			continue_dict_copy=continue_dict.copy()
			yield intervals_closed,support_indices,support_indices_bitset,support_full_indices,support_full_indices_bitset,refinement_index,continue_dict_copy
			if continue_dict_copy['continue']:
				for k in range(refinement_index,nb_attributes):
					cur_attr_refinement_index=attr_refinement_index if refinement_index==k else 0
					for child,refin_child,child_attr_support,child_attr_support_bitset,child_attr_support_full,child_attr_support_full_bitset in children_numeric_particular_bitset(intervals_closed[k],support_indices,support_indices_bitset,support_full_indices,support_full_indices_bitset,partitions_to_dataset[attributes[k]],partitions_to_dataset_bitset[attributes[k]],cur_attr_refinement_index,threshold_sup=threshold_sup,fixed_left=fixed_left&(k==0),fixed_right=fixed_right&(k==0)):
						intervals_new=intervals_closed[:]
						intervals_new[k]=child



						PILE_PUSH((intervals_new,child_attr_support,child_attr_support_bitset,child_attr_support_full,child_attr_support_full_bitset,refin_child,k,continue_dict_copy))
			# else:
			# 	raw_input('......')
			# 	print 'hazeqsd'

def pattern_to_yield(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals):
	new_intervals=[]
	for ia,a in enumerate(attributes):
		p_d_a=sorted(partitions_to_dataset[a])
		interval_a=intervals[ia]
		new_intervals.append(p_d_a[bisect_left(p_d_a,interval_a[0]):bisect_right(p_d_a,interval_a[-1])])
	return new_intervals

def enumerator_on_partitions_particular_init(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,indices_to_start_with=None,indices_to_start_with_full=None,threshold_sup=1,fixed_left=False,fixed_right=False,info_to_push={}):
	refinement_index=0
	intervals=[sorted(partitions_to_dataset[attr_name]) for attr_name in attributes]
	
	


	attrs_refinement_indexes=[0 for k in range(len(attributes))]
	if indices_to_start_with is None:
		indices_to_start_with=set(range(len(dataset_to_partitions)))
	
	if indices_to_start_with_full is None:
		indices_to_start_with_full=set(indices_to_start_with)

	refinement_index=0
	indices_bitset_to_start_with=encode_sup(indices_to_start_with)
	info_to_push['continue']=True
	if len(indices_to_start_with)>=threshold_sup:
		
		intervals=closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,indices_to_start_with)
		
		LAUNCH_ENUM=True



		if fixed_left:
			LAUNCH_ENUM=(intervals[0][0]==info_to_push['equal_with'])
		elif fixed_right:
			LAUNCH_ENUM=(intervals[0][-1]==info_to_push['equal_with'])

		# if fixed_left:
		# 	LAUNCH_ENUM=(intervals[0][0]<=info_to_push['equal_with'])
		# elif fixed_right:
		# 	LAUNCH_ENUM=(intervals[0][-1]>=info_to_push['equal_with'])
		
		if LAUNCH_ENUM:
			indices_to_start_with_full,__=compute_support_interval(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals)
			indices_bitset_to_start_with_full=encode_sup(indices_to_start_with_full)
			for pattern,sup,sup_bitset,sup_full_indices,sup_full_indices_bitset,current_refinement_index,continue_dict in enumerator_on_partitions_particular(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,indices_to_start_with,indices_bitset_to_start_with,indices_to_start_with_full,indices_bitset_to_start_with_full,attrs_refinement_indexes,refinement_index,info_to_push,threshold_sup=threshold_sup,fixed_left=fixed_left,fixed_right=fixed_right):
				yieldous=pattern
				yield yieldous,sup,sup_bitset,sup_full_indices,sup_full_indices_bitset,current_refinement_index,continue_dict





def compute_closed_on_the_positive_full_extent(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,extent,full_positive_extent): #TODOTODO
	positive_extent=extent&full_positive_extent
	if len(extent)==0:
		return closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,positive_extent),positive_extent,encode_sup(positive_extent)

	intervals=closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,positive_extent)
	sup,__=compute_support_interval(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals)
	sup_bitset=encode_sup(sup)
	return intervals,sup,sup_bitset

def enumerator_on_partitions_particular_bitset_init(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,indices_to_start_with=None,indices_to_start_with_full=None,threshold_sup=1,fixed_left=False,fixed_right=False,info_to_push={}):
	if indices_to_start_with is None or len(indices_to_start_with)>=threshold_sup: 
		refinement_index=0
		intervals=[sorted(partitions_to_dataset[attr_name]) for attr_name in attributes]
		
		


		attrs_refinement_indexes=[0 for k in range(len(attributes))]
		if indices_to_start_with is None:
			indices_to_start_with=set(range(len(dataset_to_partitions)))
		
		if indices_to_start_with_full is None:
			indices_to_start_with_full=set(indices_to_start_with)

		refinement_index=0
		indices_bitset_to_start_with=encode_sup(indices_to_start_with)
		info_to_push['continue']=True
		
			
		intervals=closed_numeric(dataset_to_partitions,partitions_to_dataset,attributes,indices_to_start_with)
		
		LAUNCH_ENUM=True



		if fixed_left:
			
			LAUNCH_ENUM=(intervals[0][0]==info_to_push['equal_with'])
			
		elif fixed_right:
			
			LAUNCH_ENUM=(intervals[0][-1]==info_to_push['equal_with'])
			

		# if fixed_left:
		# 	LAUNCH_ENUM=(intervals[0][0]<=info_to_push['equal_with'])
		# elif fixed_right:
		# 	LAUNCH_ENUM=(intervals[0][-1]>=info_to_push['equal_with'])
		
		if LAUNCH_ENUM:
			indices_to_start_with_full,__=compute_support_interval(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals)
			indices_bitset_to_start_with_full=encode_sup(indices_to_start_with_full)
			#for pattern,sup,sup_bitset,sup_full_indices,sup_full_indices_bitset,current_refinement_index,continue_dict in enumerator_on_partitions_particular_bitset_iterativement(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,indices_to_start_with,indices_bitset_to_start_with,indices_to_start_with_full,indices_bitset_to_start_with_full,attrs_refinement_indexes,refinement_index,info_to_push,threshold_sup=threshold_sup,fixed_left=fixed_left,fixed_right=fixed_right):
			for pattern,sup,sup_bitset,sup_full_indices,sup_full_indices_bitset,current_refinement_index,continue_dict in enumerator_on_partitions_particular_bitset_iterativement(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,indices_to_start_with,indices_bitset_to_start_with,indices_to_start_with_full,indices_bitset_to_start_with_full,0,refinement_index,info_to_push,threshold_sup=threshold_sup,fixed_left=fixed_left,fixed_right=fixed_right):
				yieldous=pattern
				yield yieldous,sup,sup_bitset,sup_full_indices,sup_full_indices_bitset,current_refinement_index,continue_dict


#####################################################################################################################################################################

#TODO TODO
def from_closed_bitset_to_support_corresponding_to_the_closed_on_the_positive_full_extent(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals,support_indices_bitset,positive_extent):
	#print intervals
	pos_to_consider=support_indices_bitset&positive_extent
	if pos_to_consider==closed_numeric_by_desc_bitset.ZERO:
		return intbitset()
	
	intervals_closed,_=closed_numeric_by_desc_bitset(dataset_to_partitions,partitions_to_dataset_bitset,attributes,pos_to_consider,intervals,0)
	#print intervals_closed
	indices_to_start_with_full,__=compute_support_interval(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,intervals_closed)
	indices_bitset_to_start_with_full=encode_sup(indices_to_start_with_full)
	return indices_bitset_to_start_with_full

def informedness(tpr,fpr,alpha):
	return tpr-fpr

def wracc(tpr,fpr,alpha):
	return alpha*(1-alpha)*(tpr-fpr)

def linear_corr(tpr,fpr,alpha):
	s=alpha*tpr+(1-alpha)*fpr
	if s==1:
		return 0.
	aa=alpha*(1-alpha)
	ret=aa/((s)*(1-s))
	ret=(ret)**(0.5)

	return (tpr-fpr)*ret


def core(p,sup,partitions_to_dataset,attributes,attributes_stized=None):
	core_p=[[] for k in range(len(p))]
	if any(len(k)<=2 for k in p):
		return core_p,set()
	else:
		set_to_ret=sup
		for i,k in enumerate(p):
			attr_concerned=attributes[i]
			core_p[i]=k[1:-1]
			set_to_ret=set_to_ret-partitions_to_dataset[attr_concerned][k[0]]['=']
			set_to_ret=set_to_ret-partitions_to_dataset[attr_concerned][k[-1]]['=']
		return core_p,set_to_ret




def core_stringent(p,sup,sup_bitset,partitions_to_dataset,partition_to_dataset_bitset,attributes,attributes_discretized=None):
	if attributes_discretized is None:
		return core(p,sup,partitions_to_dataset,attributes)
	else:
		set_to_ret=sup
		bitset_to_ret=sup_bitset
		core_p=[[] for k in range(len(p))] #NOT CALCULATED YET
		
		for i,p_i in enumerate(p):
			attr_concerned=attributes[i]
			partitions_dead_i = attributes_discretized[attr_concerned]['partitions_dead']
			borders_cells_to_remove={p_i[0],p_i[-1]}
			borders_cells_to_remove=borders_cells_to_remove-partitions_dead_i
			for b in borders_cells_to_remove:
				set_to_ret=set_to_ret-partitions_to_dataset[attr_concerned][b]['=']
				bitset_to_ret=bitset_to_ret-partition_to_dataset_bitset[attr_concerned][b]['=']
				#bitset_to_ret&=~partition_to_dataset_bitset[attr_concerned][b]['=']
		return core_p,set_to_ret,bitset_to_ret





def core_stringent_exhausitve(p,sup,partitions_to_dataset,attributes,index_attr=None):
	set_to_ret=sup
	core_p=[[] for k in range(len(p))] #NOT CALCULATED YET
	for i,p_i in enumerate(p):
		attr_concerned=attributes[i]
		partitions_actu=sorted(partitions_to_dataset[attr_concerned])
		borders_cells_to_remove={p_i[0],p_i[-1]}
		borders_cells_to_remove_to_use=set()
		for b in borders_cells_to_remove:
			ind_b=bisect_left(partitions_actu,b)
			if b+1==len(index_attr[attr_concerned]['values']):
				continue
			
			if ind_b+1 >= len(partitions_actu):
				borders_cells_to_remove_to_use|={b}
			else:
				if partitions_actu[ind_b+1]-b>1:
					borders_cells_to_remove_to_use|={b}
		for b in borders_cells_to_remove_to_use:
			set_to_ret=set_to_ret-partitions_to_dataset[attr_concerned][b]['=']
	return core_p,set_to_ret



def update_discretization_and_partitions_without_zone(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,attributes_discretized,attr_to_cut,cut_point_indice):
	values=index_attr[attr_to_cut]['values']
	cut_points_indices=attributes_discretized[attr_to_cut]['cut_points_indices']
	remaining_cut_points_indices=attributes_discretized[attr_to_cut]['remaining_cut_points_indices'][:]
	
	partitions_indices=attributes_discretized[attr_to_cut]['partitions_indices']
	partitions_values=attributes_discretized[attr_to_cut]['partitions_values']
	partitions_dead=attributes_discretized[attr_to_cut]['partitions_dead']
	keys=sorted(partitions_indices)
	partition_in=keys[bisect_left(keys,cut_point_indice)-1]
		
	old_partition=partitions_indices[partition_in]
	old_partition_values=partitions_values[partition_in]
	ind_cut_point_indice_in_old_partition=bisect_left(old_partition,cut_point_indice)
	new_partitions_left=old_partition[0:ind_cut_point_indice_in_old_partition]
	new_partitions_right=old_partition[ind_cut_point_indice_in_old_partition:]
	new_partitions_values_left=old_partition_values[0:ind_cut_point_indice_in_old_partition]
	new_partitions_values_right=old_partition_values[ind_cut_point_indice_in_old_partition:]


	cut_points_indices=sorted(cut_points_indices+[cut_point_indice])
	len_cut_points_indices=len(cut_points_indices)-1
	remaining_cut_points_indices.remove(cut_point_indice)
	cut_points_values=[values[k] for k in cut_points_indices]

	partitions_indices_new={k:v for k,v in partitions_indices.iteritems() if k!=partition_in}
	partitions_indices_new[new_partitions_left[0]]=new_partitions_left
	partitions_indices_new[new_partitions_right[0]]=new_partitions_right
	
	if len(new_partitions_left)==1:
		partitions_dead|={new_partitions_left[0]}
	if len(new_partitions_right)==1:
		partitions_dead|={new_partitions_right[0]}

	partitions_values_new={k:v for k,v in partitions_values.iteritems() if k!=partition_in}
	partitions_values_new[new_partitions_left[0]]=new_partitions_values_left
	partitions_values_new[new_partitions_right[0]]=new_partitions_values_right

	attributes_discretized[attr_to_cut]={
		'cut_points_indices':cut_points_indices,
		'remaining_cut_points_indices':remaining_cut_points_indices,
		'cut_points_values':cut_points_values,
		'partitions_indices':partitions_indices_new,
		'partitions_values':partitions_values_new,
		'partitions_dead':partitions_dead
	}

	cur_attr_partitions=partitions_to_dataset[attr_to_cut]
	cur_attr_partitions_bitset=partitions_to_dataset_bitset[attr_to_cut]
	basline_partitions=index_attr[attr_to_cut]['indices']
	baseline_partitions_bitset=index_attr[attr_to_cut]['indices_bitset']

	basline_partitions_sup_inf=index_attr[attr_to_cut]['indices_sup_inf']
	baseline_partitions_sup_inf_bitset=index_attr[attr_to_cut]['indices_bitset_sup_inf']

	new_partitions_left_min=new_partitions_left[0]
	new_partitions_left_max=new_partitions_left[-1]

	new_partitions_right_min=new_partitions_right[0]
	new_partitions_right_max=new_partitions_right[-1]



	new_partitions_left_all_indices=basline_partitions_sup_inf[new_partitions_left_min]['>']&basline_partitions_sup_inf[new_partitions_left_max]['<']
	new_partitions_left_all_indices_bitset=baseline_partitions_sup_inf_bitset[new_partitions_left_min]['>']&baseline_partitions_sup_inf_bitset[new_partitions_left_max]['<']

	new_partitions_right_all_indices=cur_attr_partitions[partition_in]['=']-new_partitions_left_all_indices
	#new_partitions_right_all_indices_bitset=cur_attr_partitions_bitset[partition_in]['=']&~new_partitions_left_all_indices_bitset
	new_partitions_right_all_indices_bitset=cur_attr_partitions_bitset[partition_in]['=']-new_partitions_left_all_indices_bitset

	name_partition_left=new_partitions_left[0]
	name_partition_right=new_partitions_right[0]
	del cur_attr_partitions[partition_in]
	del cur_attr_partitions_bitset[partition_in]
	
	cur_attr_partitions[name_partition_left]={}
	cur_attr_partitions[name_partition_right]={}
	cur_attr_partitions_bitset[name_partition_left]={}
	cur_attr_partitions_bitset[name_partition_right]={}

	cur_attr_partitions[name_partition_left]['=']=set(new_partitions_left_all_indices)
	cur_attr_partitions[name_partition_left]['>']=set(basline_partitions_sup_inf[new_partitions_left_min]['>'])
	cur_attr_partitions[name_partition_left]['<']=set(basline_partitions_sup_inf[new_partitions_left_max]['<'])
	cur_attr_partitions[name_partition_left]['val_min']=basline_partitions_sup_inf[new_partitions_left_min]['val_min']
	cur_attr_partitions[name_partition_left]['val_max']=basline_partitions_sup_inf[new_partitions_left_max]['val_max']


	cur_attr_partitions_bitset[name_partition_left]['=']=new_partitions_left_all_indices_bitset
	cur_attr_partitions_bitset[name_partition_left]['>']=baseline_partitions_sup_inf_bitset[new_partitions_left_min]['>']
	cur_attr_partitions_bitset[name_partition_left]['<']=baseline_partitions_sup_inf_bitset[new_partitions_left_max]['<']
	


	cur_attr_partitions[name_partition_right]['=']=set(new_partitions_right_all_indices)
	cur_attr_partitions[name_partition_right]['>']=set(basline_partitions_sup_inf[new_partitions_right_min]['>'])
	cur_attr_partitions[name_partition_right]['<']=set(basline_partitions_sup_inf[new_partitions_right_max]['<'])
	cur_attr_partitions[name_partition_right]['val_min']=basline_partitions_sup_inf[new_partitions_right_min]['val_min']
	cur_attr_partitions[name_partition_right]['val_max']=basline_partitions_sup_inf[new_partitions_right_max]['val_max']


	cur_attr_partitions_bitset[name_partition_right]['=']=new_partitions_right_all_indices_bitset
	cur_attr_partitions_bitset[name_partition_right]['>']=baseline_partitions_sup_inf_bitset[new_partitions_right_min]['>']
	cur_attr_partitions_bitset[name_partition_right]['<']=baseline_partitions_sup_inf_bitset[new_partitions_right_max]['<']
	


	for i in new_partitions_right_all_indices:
		dataset_to_partitions[i][attr_to_cut]=name_partition_right

	return cur_attr_partitions,name_partition_left,name_partition_right

def update_discretization_and_partitions(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,attributes_discretized,attr_to_cut,cut_point_indice,hardcore_optimize=False):
	new_attributes_discretized={attr_name:dict(attributes_discretized) for attr_name in attributes}
	


	if hardcore_optimize:
		pos_indices_to_indices_to_add=index_attr[attr_to_cut]['pos_indices_to_indices_to_add']
		pos_indices_to_indices_to_add_rev=index_attr[attr_to_cut]['pos_indices_to_indices_to_add_rev']
		if cut_point_indice in pos_indices_to_indices_to_add:
			pos_cut_point=cut_point_indice
			neg_cut_point=pos_indices_to_indices_to_add[cut_point_indice]
		else:
			pos_cut_point=pos_indices_to_indices_to_add_rev[cut_point_indice]
			neg_cut_point=cut_point_indice
		print 'POS = ',pos_cut_point
		print 'NEG = ',neg_cut_point
		#raw_input('......')

		if pos_cut_point!=neg_cut_point:
			cur_attr_partitions,name_partition_left_POS,name_partition_right_POS=update_discretization_and_partitions_without_zone(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,attributes_discretized,attr_to_cut,pos_cut_point)
			cur_attr_partitions,name_partition_left_NEG,name_partition_right_NEG=update_discretization_and_partitions_without_zone(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,attributes_discretized,attr_to_cut,neg_cut_point)
			
			if name_partition_left_POS not in pos_indices_to_indices_to_add:
				name_partition_left_POS=name_partition_left_POS-1

			left_zone_to_mine_next=cur_attr_partitions[name_partition_left_POS]['<']|cur_attr_partitions[name_partition_left_POS]['=']
			right_zone_to_mine_next=cur_attr_partitions[name_partition_right_POS]['>']|cur_attr_partitions[name_partition_right_POS]['=']
			left_zone_to_mine_next_right_end=name_partition_left_POS
			right_zone_to_mine_next_left_end=name_partition_right_POS
			#####################################################################
			print index_attr[attr_to_cut]['pos_indices']
			print attributes_discretized[attr_to_cut]['cut_points_indices']
			print name_partition_right_NEG
			print name_partition_left_POS
			indice_next_name_parition_right_neg=attributes_discretized[attr_to_cut]['cut_points_indices'][bisect_left(attributes_discretized[attr_to_cut]['cut_points_indices'],name_partition_right_NEG)+1] if name_partition_right_NEG not in pos_indices_to_indices_to_add else name_partition_right_NEG
			
			indice_next_name_partition_left_pos=attributes_discretized[attr_to_cut]['cut_points_indices'][bisect_left(attributes_discretized[attr_to_cut]['cut_points_indices'],name_partition_left_POS)-1]
			if indice_next_name_partition_left_pos not in pos_indices_to_indices_to_add:
				indice_next_name_partition_left_pos=indice_next_name_partition_left_pos-1
			if indice_next_name_partition_left_pos<0:
				indice_next_name_partition_left_pos=0
			left_zone_to_mine_next_to_check_guarantees_in=cur_attr_partitions[indice_next_name_parition_right_neg]['<']#left_zone_to_mine_next|cur_attr_partitions[name_partition_right_NEG]['=']
			right_zone_to_mine_next_to_check_guarantees_in=cur_attr_partitions[indice_next_name_partition_left_pos]['>']#right_zone_to_mine_next|cur_attr_partitions[name_partition_left_NEG]['=']
			left_zone_to_mine_next_to_check_guarantees_in_right_end=indice_next_name_parition_right_neg
			right_zone_to_mine_next_to_check_guarantees_in_left_end=indice_next_name_partition_left_pos
			#####################################################################

		else:
			cur_attr_partitions,name_partition_left,name_partition_right=update_discretization_and_partitions_without_zone(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,attributes_discretized,attr_to_cut,pos_cut_point)
			left_zone_to_mine_next=cur_attr_partitions[name_partition_left]['<']|cur_attr_partitions[name_partition_left]['=']
			right_zone_to_mine_next=cur_attr_partitions[name_partition_right]['>']|cur_attr_partitions[name_partition_right]['=']
			left_zone_to_mine_next_right_end=name_partition_left
			right_zone_to_mine_next_left_end=name_partition_right
			#####################################################################
			left_zone_to_mine_next_to_check_guarantees_in=left_zone_to_mine_next|cur_attr_partitions[name_partition_right]['=']
			right_zone_to_mine_next_to_check_guarantees_in=right_zone_to_mine_next|cur_attr_partitions[name_partition_left]['=']
			left_zone_to_mine_next_to_check_guarantees_in_right_end=name_partition_right
			right_zone_to_mine_next_to_check_guarantees_in_left_end=name_partition_left
			#####################################################################

	else:
		cur_attr_partitions,name_partition_left,name_partition_right=update_discretization_and_partitions_without_zone(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,attributes_discretized,attr_to_cut,cut_point_indice)
		left_zone_to_mine_next=cur_attr_partitions[name_partition_left]['<']|cur_attr_partitions[name_partition_left]['=']
		right_zone_to_mine_next=cur_attr_partitions[name_partition_right]['>']|cur_attr_partitions[name_partition_right]['=']
		left_zone_to_mine_next_right_end=name_partition_left
		right_zone_to_mine_next_left_end=name_partition_right
		#####################################################################
		left_zone_to_mine_next_to_check_guarantees_in=left_zone_to_mine_next|cur_attr_partitions[name_partition_right]['=']
		right_zone_to_mine_next_to_check_guarantees_in=right_zone_to_mine_next|cur_attr_partitions[name_partition_left]['=']
		left_zone_to_mine_next_to_check_guarantees_in_right_end=name_partition_right
		right_zone_to_mine_next_to_check_guarantees_in_left_end=name_partition_left
		#####################################################################
	return left_zone_to_mine_next,right_zone_to_mine_next,left_zone_to_mine_next_right_end,right_zone_to_mine_next_left_end, \
		   left_zone_to_mine_next_to_check_guarantees_in,right_zone_to_mine_next_to_check_guarantees_in,left_zone_to_mine_next_to_check_guarantees_in_right_end,right_zone_to_mine_next_to_check_guarantees_in_left_end


def select_next_cut_point_random(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent,can_enhance=True):
	attributes_to_select_from=[]
	for a in attributes_discretized:
		if len(attributes_discretized[a]['remaining_cut_points_indices'])>0:
			attributes_to_select_from.append(a)
	if len(attributes_to_select_from)==0:
		return None,None,False
	
	ind_attr=randint(0,len(attributes_to_select_from)-1)
	attr_to_cut=attributes_to_select_from[ind_attr]
	remaining_cut_points_to_select_from=attributes_discretized[attr_to_cut]['remaining_cut_points_indices']
	ind_cut_point_ind=randint(0,len(remaining_cut_points_to_select_from)-1)
	cut_point_indice=remaining_cut_points_to_select_from[ind_cut_point_ind]
	can_enhance=True
	return attr_to_cut,cut_point_indice,can_enhance


def select_next_cut_point_maximize_fpr_in_core(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent,can_enhance=True):
	candidate_attributes_indices=[]
	for i_attra,attra in enumerate(attributes):
		attr_candidate=attra
		remaining_cut_points_indices=attributes_discretized[attr_candidate]['remaining_cut_points_indices']
		baseline_partitions=index_attr[attr_candidate]['indices']
		if len(remaining_cut_points_indices)==0:
			continue
		for v_attr in range(interval_guarantee[i_attra][0]+1,interval_guarantee[i_attra][1]):
			if v_attr in remaining_cut_points_indices:
				candidate_attributes_indices.append((attr_candidate,v_attr,len(baseline_partitions[v_attr]&negative_extent)))
		for v_attr in range(interval_guarantee[i_attra][-1]+1,remaining_cut_points_indices[-1]+1):
			if v_attr in remaining_cut_points_indices:
				candidate_attributes_indices.append((attr_candidate,v_attr,len(baseline_partitions[v_attr]&negative_extent)))
	
	if len(candidate_attributes_indices):
		candidate_attributes_indices=sorted(candidate_attributes_indices,key=lambda x:x[2],reverse=True)
		attr_to_cut,cut_point_indice,ss=candidate_attributes_indices[0]
		can_enhance=True
	else:
		attr_to_cut,cut_point_indice,___=select_next_cut_point_random(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent)
		can_enhance=False ## I can't break the guarantees of the best actual one thus the max guarantee has been found
	return attr_to_cut,cut_point_indice,can_enhance


def select_next_cut_point_median_value_in_the_border(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent,can_enhance=True):
	attributes_to_select_from=[]
	if can_enhance:
		for a in attributes_discretized:
			if len(attributes_discretized[a]['remaining_cut_points_indices'])>0:
				attributes_to_select_from.append(a)
		if len(attributes_to_select_from)==0:
			return None,None,False
		zones_attributes_to_consider=[]
		#print interval_guarantee
		for ia,a in enumerate(attributes):
			remaining_cut_points_indices=attributes_discretized[a]['remaining_cut_points_indices']
			actual_cut_points_indices=attributes_discretized[a]['cut_points_indices']
			indices_base_partitions_flattened=index_attr[a]['indices_base_partitions_flattened']
			ranges=index_attr[a]['ranges']
			
			# print interval_guarantee
			# print actual_cut_points_indices


			#border_left_a=interval_guarantee[ia][0]+1,actual_cut_points_indices[bisect_left(actual_cut_points_indices,interval_guarantee[ia][1]-1)]-1
			#print actual_cut_points_indices,interval_guarantee
			index_corresp_st=bisect_left(actual_cut_points_indices,interval_guarantee[ia][0]+1)
			if index_corresp_st==len(actual_cut_points_indices):
				end_border_right_a=indices_base_partitions_flattened[-1]
			else:
				end_border_right_a=actual_cut_points_indices[index_corresp_st]-1

			border_left_a=interval_guarantee[ia][0]+1,end_border_right_a


			#actual_cut_points_indices[bisect_left(actual_cut_points_indices,interval_guarantee[ia][1]-1)]-1

			#interval_guarantee[ia][1]-1
			
			index_corresp_st=bisect_left(actual_cut_points_indices,interval_guarantee[ia][-1]+1)
			if index_corresp_st==len(actual_cut_points_indices):
				end_border_right_a=indices_base_partitions_flattened[-1]
			else:
				end_border_right_a=actual_cut_points_indices[index_corresp_st]-1

			border_right_a=interval_guarantee[ia][-1]+1,end_border_right_a


			#print a,actual_cut_points_indices,border_left_a,border_right_a
			#raw_input('....')
			if border_left_a[0]<=border_left_a[1]:
				zones_attributes_to_consider.append((a,indices_base_partitions_flattened[ranges[border_left_a[0]][0]:ranges[border_left_a[1]][1]+1]))
			if border_right_a[0]<=border_right_a[1]:
				zones_attributes_to_consider.append((a,indices_base_partitions_flattened[ranges[border_right_a[0]][0]:ranges[border_right_a[1]][1]+1]))
		
		if len(zones_attributes_to_consider):
			attr_to_cut,zone_to_pick_from=zones_attributes_to_consider[randint(0,len(zones_attributes_to_consider)-1)]

			#cut_point_indice=zone_to_pick_from[len(zone_to_pick_from)/2]
			cut_point_indice=zone_to_pick_from[len(zone_to_pick_from)/2]
			#print attr_to_cut,cut_point_indice
			can_enhance=True
		else:
			attr_to_cut,cut_point_indice,___=select_next_cut_point_random(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent)
			can_enhance=False ## I can't break the guarantees of the best actual one thus the max guarantee has been found
		return attr_to_cut,cut_point_indice,can_enhance
	else:
		attr_to_cut,cut_point_indice,___=select_next_cut_point_random(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent)
	return attr_to_cut,cut_point_indice,can_enhance


def select_next_cut_point_median_value_in_the_border_MAX_FPR(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent,can_enhance=True):
	attributes_to_select_from=[]
	for a in attributes_discretized:
		if len(attributes_discretized[a]['remaining_cut_points_indices'])>0:
			attributes_to_select_from.append(a)
	if len(attributes_to_select_from)==0:
		return None,None,False
	zones_attributes_to_consider=[]
	#print interval_guarantee
	for ia,a in enumerate(attributes):
		remaining_cut_points_indices=attributes_discretized[a]['remaining_cut_points_indices']
		actual_cut_points_indices=attributes_discretized[a]['cut_points_indices']
		indices_base_partitions_flattened=index_attr[a]['indices_base_partitions_flattened']
		ranges=index_attr[a]['ranges']

		index_corresp_st=bisect_left(actual_cut_points_indices,interval_guarantee[ia][0]+1)
		if index_corresp_st==len(actual_cut_points_indices):
			end_border_right_a=indices_base_partitions_flattened[-1]
		else:
			end_border_right_a=actual_cut_points_indices[index_corresp_st]-1

		border_left_a=interval_guarantee[ia][0]+1,end_border_right_a

		index_corresp_st=bisect_left(actual_cut_points_indices,interval_guarantee[ia][-1]+1)
		if index_corresp_st==len(actual_cut_points_indices):
			end_border_right_a=indices_base_partitions_flattened[-1]
		else:
			end_border_right_a=actual_cut_points_indices[index_corresp_st]-1

		border_right_a=interval_guarantee[ia][-1]+1,end_border_right_a


		#print a,actual_cut_points_indices,border_left_a,border_right_a
		#raw_input('....')
		if border_left_a[0]<=border_left_a[1]:
			zones_attributes_to_consider.append((a,indices_base_partitions_flattened[ranges[border_left_a[0]][0]:ranges[border_left_a[1]][1]+1]))
		if border_right_a[0]<=border_right_a[1]:
			zones_attributes_to_consider.append((a,indices_base_partitions_flattened[ranges[border_right_a[0]][0]:ranges[border_right_a[1]][1]+1]))
	
	if len(zones_attributes_to_consider):
		candidate_attributes_indices=[]
		for attr,zone_to_pick_from in zones_attributes_to_consider:
			#print attr
			#baseline_partitions=index_attr[attr]['indices']
			baseline_partitions=index_attr[attr]['indices_sup_inf']
			for cc in set(zone_to_pick_from):
				#candidate_attributes_indices.append(attr,cc,len(partitions_to_dataset[attr][cc]['=']&negative_extent))
				if True:
					actual_cut_points_indices=attributes_discretized[attr]['cut_points_indices']
					pos_right_cc=bisect_left(actual_cut_points_indices,cc)
					pos_left_cc=bisect_left(actual_cut_points_indices,cc)-1
					#print attr,cc,actual_cut_points_indices,pos_left_cc,pos_right_cc,actual_cut_points_indices[pos_left_cc],actual_cut_points_indices[pos_right_cc]
					pos_right_cc=actual_cut_points_indices[pos_right_cc]
					pos_left_cc= actual_cut_points_indices[pos_left_cc]
					

					candidate_attributes_indices.append((attr,cc,   min(len(baseline_partitions[pos_right_cc]['<']&baseline_partitions[cc]['>']&negative_extent),len(baseline_partitions[pos_left_cc]['>']&baseline_partitions[cc]['<']&negative_extent)) ))
					#candidate_attributes_indices.append((attr,cc,   min(len(baseline_partitions[pos_right_cc]['<']&baseline_partitions[cc]['>']&positive_extent),len(baseline_partitions[pos_left_cc]['>']&baseline_partitions[cc]['<']&positive_extent)) ))
					#raw_input('...')
				#print actual_cut_points_indices
				#raw_input('....')
				else:
					candidate_attributes_indices.append((attr,cc,len(baseline_partitions[cc]['=']&negative_extent)))

		candidate_attributes_indices=sorted(candidate_attributes_indices,key=lambda x:x[2],reverse=True)
		
		#attr_to_cut,zone_to_pick_from=zones_attributes_to_consider[randint(0,len(zones_attributes_to_consider)-1)]
		#cut_point_indice=zone_to_pick_from[len(zone_to_pick_from)/2]
		attr_to_cut,cut_point_indice,ss=candidate_attributes_indices[0]
		#print ss
		#raw_input('.....')
		#print attr_to_cut,cut_point_indice
		can_enhance=True
	else:
		attr_to_cut,cut_point_indice,___=select_next_cut_point_random(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent)
		can_enhance=False ## I can't break the guarantees of the best actual one thus the max guarantee has been found
	return attr_to_cut,cut_point_indice,can_enhance


def exhaustive_mine(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,positive_extent,negative_extent,alpha_ratio_class,PROCEDURE_OF_SELECTION=None,DISCRIMINATION_MEASURE=informedness,threshold_sup=1,threshold_qual=0,keep_patterns_found=False,verbose=True,MAXIMUM_TIME=float('inf'),LIGHTER_PATTERN_SET=False,similarity_threshold=1.):
	
	ONLY_BITSET=True
	TIGHT_CORE=True
	TIMESERIE_QUALITY=[]#(numero,quality)
	TIMESERIE_GUARANTEE=[]#(numero,quality)
	#dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new=dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset
	nb_positives=float(len(positive_extent))
	nb_negatives=float(len(negative_extent))

	positive_extent_bitset=encode_sup(positive_extent)
	negative_extent_bitset=encode_sup(negative_extent)
	
	numero=0
	Guarantee=-1000
	best=([],set(),0)
	CLOSE_ON_POSITIVE=True
	starting_support= set(range(len(dataset_to_partitions))) & positive_extent if CLOSE_ON_POSITIVE else set(range(len(dataset_to_partitions)))
	START=time()
	reorganized_attributes=attributes
	crunchiness=0.
	COMPUTE_GUARANTEE=False


	#ALL_PATTERNS_FOUND={}
	ALL_PATTERNS_FOUND=[];APPEND_RESULTS=ALL_PATTERNS_FOUND.append
	HASHMAP_ALL_PATTERNS_FOUND={}
	if ONLY_BITSET==True:
		enum_function_to_use=enumerator_on_partitions_particular_bitset_init
	else:
		enum_function_to_use=enumerator_on_partitions_particular_init

	
	for p,supPOS,sup_bitset,sup,sup_full_bitset,current_refinement_index,cnt_dict in enum_function_to_use(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,starting_support,threshold_sup=threshold_sup):
		if numero%100000==0:
			print numero,best[2],time()-START 
			if time()-START> MAXIMUM_TIME:
				break
		numero+=1
		if ONLY_BITSET:
			tpr=nb_bit_1(sup_bitset)/nb_positives
			fpr=nb_bit_1(sup_full_bitset&negative_extent_bitset)/nb_negatives		
		else:
			tpr=len(supPOS)/nb_positives
			fpr=len(sup&negative_extent)/nb_negatives
		
		if fpr==0:			
			cnt_dict['continue']=True

		quality_pattern=DISCRIMINATION_MEASURE(tpr,fpr,alpha_ratio_class)
		if quality_pattern>best[2]:
			best=(pattern_value_to_print(p,partitions_to_dataset,attributes,reorganized_attributes),sup,quality_pattern)
		if keep_patterns_found:
			if quality_pattern>threshold_qual:
				if LIGHTER_PATTERN_SET:
				#ALL_PATTERNS_FOUND[sup_bitset]=(pattern_value_to_print(p,partitions_to_dataset,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern)
					tt=time()
					compressed_sup_full_bitset=sup_full_bitset.fastdump()
					START+=time()-tt
					#compressed_sup_full_bitset=sup_full_bitset.strbits()
					APPEND_RESULTS((compressed_sup_full_bitset,quality_pattern))
					#APPEND_RESULTS((sup_full_bitset,quality_pattern))
				else:
					APPEND_RESULTS((pattern_value_to_print(p,partitions_to_dataset,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern))
					HASHMAP_ALL_PATTERNS_FOUND[sup_bitset]=(pattern_value_to_print(p,partitions_to_dataset,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern)


	print 'Time spent : ',time()-START
		
	if verbose:
		print '------------Start------------------'
		print 'nb : ',numero
		print 'pattern : ', best[0]
		print 'quality : ' , best[2]
		print 'Guarantee : ',Guarantee
		print 'Time spent : ',time()-START
		print 'CRUNCHINESS : ',crunchiness/float(numero)
		
		print '------------Start------------------'
		TIMESERIE_QUALITY.append((time()-START,best[2],numero))
		TIMESERIE_GUARANTEE.append((time()-START,-1,numero))
		#print len(ALL_PATTERNS_FOUND),'sdkjhazerluaopzdsmlkjdzupore'
		yield TIMESERIE_QUALITY,TIMESERIE_GUARANTEE,ALL_PATTERNS_FOUND#ALL_PATTERNS_FOUND.values()
		#yield TIMESERIE_QUALITY,TIMESERIE_GUARANTEE,HASHMAP_ALL_PATTERNS_FOUND.values()
	else: 
		#print 'CRUNCHINESS : ',1-float(crunchiness)/numero
		#return 1-float(crunchiness)/numero
		print 'CRUNCHINESS : ',crunchiness/float(numero)
		yield crunchiness/float(numero)
		#return 1-float(crunchiness)/numero
	







def jaccard(s1,s2):
	return float(len(s1&s2))/len(s1|s2)


def get_top_k_div_from_a_pattern_set(patterns,threshold_sim=0.5,k=1): #patterns = [(p,sup,supbitset,qual)]
	
	
	returned_patterns=[]
	sorted_patterns=sorted(patterns,key=lambda x:x[3],reverse=True)
	tp=sorted_patterns[0]
	del sorted_patterns[0]
	returned_patterns.append(tp)
	to_remove=None
	while 1:
		if len(returned_patterns)==k:
			break

		found_yet=False
		
		for indice,(p,sup,supbitset,qual) in enumerate(sorted_patterns):
			if all(jaccard(sup,supc)<=threshold_sim for _,supc,_,_ in returned_patterns):
				found_yet=True
				t_p=(p,sup,supbitset,qual)
				returned_patterns.append(t_p)
				to_remove=indice
				break
		if to_remove is not None:
			del sorted_patterns[to_remove]

		if not found_yet:
			break

		
	return returned_patterns

def decode_a_string(s2_compressed):
	sup_ret=set()
	for i,v in enumerate(s2_compressed):
		if v=='1':
			sup_ret|={i}
	return intbitset(sup_ret) 

def jaccard_with_compressed(s1,s2_compressed):
	s2=intbitset()
	s2.fastload(s2_compressed)
	
	#s2=decode_a_string(s2_compressed)
	return float(len(s1&s2))/len(s1|s2)				

def get_top_k_div_from_a_pattern_set_new(patterns,threshold_sim=0.5,k=1): #patterns = [(supbitset,qual)]
	
	
	returned_patterns=[]
	sorted_patterns=sorted(patterns,key=lambda x:x[1],reverse=True)
	s2=intbitset()
	s2.fastload(sorted_patterns[0][0])
	#s2=decode_a_string(sorted_patterns[0][0])
	tp=(s2,sorted_patterns[0][1])
	returned_patterns.append(tp)
	
	while 1:
		if len(returned_patterns)==k:
			break



		found_yet=False
		
		for (sup,qual) in sorted_patterns:
			if all(jaccard_with_compressed(supc,sup)<=threshold_sim for supc,_ in returned_patterns):
				found_yet=True
				s2=intbitset()
				s2.fastload(sup)
				#s2=decode_a_string(sup)
				t_p=(s2,qual)
				returned_patterns.append(t_p)
				break
		
		if not found_yet:
			break

	return returned_patterns


def get_top_k_div_from_a_pattern_set_union(patterns,positive_extent,negative_extent,alpha_ratio_class,DISCRIMINATION_MEASURE=informedness,k=1): #patterns = [(p,sup,qual)]
	
	
	returned_patterns=[]
	sorted_patterns=sorted(patterns,key=lambda x:x[2],reverse=True)
	tp=sorted_patterns[0]
	returned_patterns.append(tp)
	actual_union=set(tp[1])
	actual_qual=tp[2]
	while 1:
		if len(returned_patterns)==k:
			break

		maximizing=0
		best_actu=None
		for (p,sup,qual) in sorted_patterns:
			
			test_union=actual_union|sup
			tpr_union=float(len(test_union&positive_extent))/len(positive_extent)
			fpr_union=float(len(test_union&negative_extent))/len(negative_extent)

			quality_union=DISCRIMINATION_MEASURE(tpr_union,fpr_union,alpha_ratio_class)

			if quality_union-actual_qual>maximizing:
				best_actu=(p,sup,qual,quality_union)
				maximizing=quality_union-actual_qual


		if best_actu is not None:
			returned_patterns.append((best_actu[0],best_actu[1],best_actu[2]))
			actual_qual=best_actu[3]

			actual_union|=best_actu[1]

		else:
			break
		
	
		
	return returned_patterns,actual_qual


TIMESERIE_QUALITY=[]
TIMESERIE_GUARANTEE=[]
TIMESERIE_GLOBALITY=[]
#THRESHOLD_QUALITY=0.12



def pattern_value_to_print(p,partitions_to_dataset,attributes,reorganized_attributes):
	index_reorganization=[attributes.index(a) for a in reorganized_attributes]
	new_p=[[]]*len(p)
	for indice_ii,ii in enumerate(index_reorganization):
		new_p[ii]=p[indice_ii]
	for ia,a in enumerate(attributes):
		partitions_a=partitions_to_dataset[a]
		new_p[ia]=[partitions_a[new_p[ia][0]]['val_min'],partitions_a[new_p[ia][-1]]['val_max']]

	return new_p

def pattern_reorganize(p,attributes,reorganized_attributes):
	index_reorganization=[attributes.index(a) for a in reorganized_attributes]
	new_p=[[]]*len(p)
	
	for indice_ii,ii in enumerate(index_reorganization):
		new_p[ii]=p[indice_ii]
	return new_p

def discretize_and_mine_STRINGENT_2(dataset,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr,positive_extent,negative_extent,alpha_ratio_class,PROCEDURE_OF_SELECTION=select_next_cut_point_random,DISCRIMINATION_MEASURE=informedness,threshold_sup=1,threshold_qual=0.,keep_patterns_found=False,verbose=True,MAXIMUM_TIME=float('inf'),CRISPINESS_COMPUTATION=False,CLEAN_PATTERNS_FOUND_LIST=False,LIGHTER_PATTERN_SET=False,USE_ALGO=False):
	#TODO : When adding a cut point if positive is on all side or in all another side then there is no reason to mine patterns - need to check (left or right)
	#TODO : When adding a cut point if negative is on all side or in all another there is no reason to check guarantees - need to check (left or right)
	
	CORE_CLOSE_ON_POSITIVE=True

	len_full_dataset=len(dataset_to_partitions)
	START=time()
	CLOSE_ON_POSITIVE=True
	ONLY_BITSET=True
	if ONLY_BITSET==True:
		enum_function_to_use=enumerator_on_partitions_particular_bitset_init
	else:
		enum_function_to_use=enumerator_on_partitions_particular_init

	ZERO=intbitset()
	#START=time()
	HASHMAP_GUARANTEES={};HASHMAP_GUARANTEES_POP=HASHMAP_GUARANTEES.pop;HASHMAP_GUARANTEES_get=HASHMAP_GUARANTEES.get
	HASHMAP_GUARANTEES_SUP={};HASHMAP_GUARANTEES_SUP_POP=HASHMAP_GUARANTEES_SUP.pop
	HASHMAP_GUARANTEES_INTERVAL={};HASHMAP_GUARANTEES_INTERVAL_POP=HASHMAP_GUARANTEES_INTERVAL.pop

	HASHMAP_CRISPENESS={}; HASHMAP_CRISPENESS_POP=HASHMAP_CRISPENESS.pop; HASHMAP_CRISPENESS_GET=HASHMAP_CRISPENESS.get

	HASHMAP_ALL_PATTERNS_FOUND={}
	#ALL_PATTERNS_FOUND={};
	ALL_PATTERNS_FOUND=[];APPEND_RESULTS=ALL_PATTERNS_FOUND.append
	attributes_discretized=discretize_all_attributes(dataset,attributes,index_attr,3,positive_extent=positive_extent)
	reorganized_attributes=attributes
	dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new=image_partition_by_descritization_all_attributes(dataset,attributes,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes_discretized)
	positive_extent_bitset=encode_sup(positive_extent)
	nb_positives=float(len(positive_extent))
	nb_negatives=float(len(negative_extent))
	positive_extent_bitset=encode_sup(positive_extent)
	negative_extent_bitset=encode_sup(negative_extent)

	ALL_REMAINING_POINTS=float(sum([len(attributes_discretized[a]['remaining_cut_points_indices']) for a in attributes_discretized ]))

	numero=0
	Guarantee=-1000
	best=([],set(),0)
	can_enhance=True

	FIRST_TIME=True
	
	starting_support= set(range(len(dataset_to_partitions_new))) & positive_extent if CLOSE_ON_POSITIVE else set(range(len(dataset_to_partitions_new)))
	#print 'support_size : ',len(starting_support)

	enum_full_init=enum_function_to_use(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,attributes,index_attr,starting_support,threshold_sup=threshold_sup)
	for p,supPOS,sup_bitset,sup,sup_full_bitset,current_refinement_index,cnt_dict in enum_full_init:
		
		#raw_input('.....')
		numero+=1
		if ONLY_BITSET:
			supNEG_bitset=sup_full_bitset&negative_extent_bitset
			tpr=nb_bit_1(sup_bitset)/nb_positives
			fpr=nb_bit_1(supNEG_bitset)/nb_negatives
		else:
			supNEG=sup&negative_extent
			tpr=len(supPOS)/nb_positives
			fpr=len(supNEG)/nb_negatives

		core_p,sup_core_p,bitset_core_p=core_stringent(p,sup,sup_full_bitset,partitions_to_dataset_new,partitions_to_dataset_bitset_new,attributes,attributes_discretized)
		if CORE_CLOSE_ON_POSITIVE:#FIX Error 
			#print 'A',len(sup_core_p)
			#core_p,sup_core_p,bitset_core_p=compute_closed_on_the_positive_full_extent(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,attributes,sup_core_p,positive_extent)
			#print p
			bitset_core_p_plus=from_closed_bitset_to_support_corresponding_to_the_closed_on_the_positive_full_extent(dataset_to_partitions,partitions_to_dataset_new,partitions_to_dataset_bitset_new,attributes,p,bitset_core_p,positive_extent_bitset)
			


		else:
			bitset_core_p_plus=bitset_core_p



		fpr_core=0.
		
		if ONLY_BITSET:
			if bitset_core_p!=ZERO:
				fpr_core=nb_bit_1(bitset_core_p&supNEG_bitset)/nb_negatives
		else:
			if len(sup_core_p):
				fpr_core=len(sup_core_p&supNEG)/nb_negatives

		local_Guarantee=DISCRIMINATION_MEASURE(tpr,fpr_core,alpha_ratio_class)

		if FIRST_TIME:
			HASHMAP_GUARANTEES[sup_bitset]=local_Guarantee#min(local_Guarantee,HASHMAP_GUARANTEES_get(sup_bitset,local_Guarantee))
			if CRISPINESS_COMPUTATION:
				before_writing=time()
				HASHMAP_CRISPENESS[sup_bitset.strbits()]= float(len(sup_full_bitset - bitset_core_p_plus)/(2.*float(len_full_dataset)))#,HASHMAP_CRISPENESS_GET(sup_bitset,1.))
				START+=time()-before_writing
			#HASHMAP_GUARANTEES_SUP[sup_bitset]=sup
			FIRST_TIME=False

		quality_pattern=DISCRIMINATION_MEASURE(tpr,fpr,alpha_ratio_class)
		if quality_pattern>best[2]:
			best=(pattern_value_to_print(p,partitions_to_dataset_new,attributes,reorganized_attributes),sup,quality_pattern)


		if keep_patterns_found:
			if quality_pattern>threshold_qual:
				#ALL_PATTERNS_FOUND[sup_bitset]=((pattern_value_to_print(p,partitions_to_dataset_new,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern))
				if LIGHTER_PATTERN_SET:
				#ALL_PATTERNS_FOUND[sup_bitset]=(pattern_value_to_print(p,partitions_to_dataset,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern)
					compressed_sup_full_bitset=sup_full_bitset.fastdump()
					APPEND_RESULTS((compressed_sup_full_bitset,quality_pattern))
				else:
					APPEND_RESULTS((pattern_value_to_print(p,partitions_to_dataset_new,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern))
					if USE_ALGO:
						HASHMAP_ALL_PATTERNS_FOUND[sup_bitset.strbits()]=(pattern_value_to_print(p,partitions_to_dataset_new,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern)
		
		if local_Guarantee<best[2]: 
			HASHMAP_GUARANTEES_POP(sup_bitset, None)
			#HASHMAP_GUARANTEES_SUP_POP(sup_bitset, None)
			HASHMAP_GUARANTEES_INTERVAL_POP(sup_bitset, None)
			if CRISPINESS_COMPUTATION:
				#HASHMAP_CRISPENESS_POP(sup_bitset, None)
				before_writing=time()
				HASHMAP_CRISPENESS[sup_bitset.strbits()]= float(len(sup_full_bitset - bitset_core_p_plus)/(2.*float(len_full_dataset)))#,HASHMAP_CRISPENESS_GET(sup_bitset,1.))
				START+=time()-before_writing
		else: 
			HASHMAP_GUARANTEES[sup_bitset]=local_Guarantee#min(local_Guarantee,HASHMAP_GUARANTEES_get(sup_bitset,local_Guarantee))
			#HASHMAP_GUARANTEES_SUP[sup_bitset]=supPOS
			HASHMAP_GUARANTEES_INTERVAL[sup_bitset]=pattern_reorganize(p,attributes,reorganized_attributes)
			if CRISPINESS_COMPUTATION:
				before_writing=time()
				HASHMAP_CRISPENESS[sup_bitset.strbits()]= float(len(sup_full_bitset - bitset_core_p_plus)/(2.*float(len_full_dataset)))#,HASHMAP_CRISPENESS_GET(sup_bitset,1.))
				START+=time()-before_writing

	if verbose:
		print '------------Start------------------'
		print 'nb : ',numero
		print 'pattern : ', best[0]
		print 'quality : ' , best[2]
		print 'Guarantee : ',max(HASHMAP_GUARANTEES.values()) if len(HASHMAP_GUARANTEES) else Guarantee
		print 'Length of HashMap : ',len(HASHMAP_GUARANTEES)
		print 'Time spent : ',time()-START
		print 'Length of HashMap : ',len(HASHMAP_GUARANTEES)
		if CRISPINESS_COMPUTATION:
			print 'CRISPINESS : ',max(HASHMAP_CRISPENESS.values())
		print '------------Start------------------'

	
	BIG_ITERATION=1
	sup_bitset_guarantee,val_guarantee=max(HASHMAP_GUARANTEES.items(),key=lambda x:x[1])
	
	if ONLY_BITSET:
		interval_guarantee=pattern_to_yield(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,attributes,HASHMAP_GUARANTEES_INTERVAL[sup_bitset_guarantee])
	else:
		sup_guarantee=HASHMAP_GUARANTEES_SUP[sup_bitset_guarantee]
		interval_guarantee=closed_numeric(dataset_to_partitions_new,partitions_to_dataset_new,attributes,sup_guarantee)
	
	reorganized_attributes=attributes
	attr_to_cut,cut_point_indice,can_enhance=PROCEDURE_OF_SELECTION(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent,can_enhance=can_enhance)
	
	while attr_to_cut is not None:
		TIMESERIE_QUALITY.append((time()-START,best[2],numero))
		GUARANTEE=max(HASHMAP_GUARANTEES.values())
		TIMESERIE_GUARANTEE.append((time()-START,GUARANTEE,numero))
		TIMESERIE_GLOBALITY.append((time()-START,max(HASHMAP_CRISPENESS.values()),numero)) if CRISPINESS_COMPUTATION else TIMESERIE_GLOBALITY.append((time()-START,1.,numero))

		if time()-START> MAXIMUM_TIME:
			break
		if verbose:
			print '----------------------------------------'
			print 'REMAINING_CUT_POINTS : ',
			for a in attributes_discretized:
				print a, len(attributes_discretized[a]['remaining_cut_points_indices']), ' ',
			print ''
			print 'nb : ',numero
			print 'pattern : ', best[0]
			print 'quality : ' , best[2]
			print 'Guarantee : ',max(HASHMAP_GUARANTEES.values())
			print 'BIG_ITERATION : ', BIG_ITERATION
			print 'Length of HashMap : ',len(HASHMAP_GUARANTEES)
			print 'Time spent : ',time()-START
			if CRISPINESS_COMPUTATION:
				print 'CRISPINESS : ',max(HASHMAP_CRISPENESS.values())
			print '----------------------------------------'
		else:
			before_writing=time()
			REMAINING_POINTS=sum([len(attributes_discretized[a]['remaining_cut_points_indices']) for a in attributes_discretized])
			if not CRISPINESS_COMPUTATION:
				stdout.write('%s\r' % ('Percentage Done: ' + ('%.2f' % ((1-REMAINING_POINTS/ALL_REMAINING_POINTS)*100)).zfill(5) + '%   ' + 'Time : ' +  ('%.2f' % ((time()-START))) + '  ' + 'Best Quality : ' + str(best[2]) + '  ' + 'Guarantee : ' + str(GUARANTEE)));stdout.flush();
			else:
				stdout.write('%s\r' % ('Percentage Done: ' + ('%.2f' % ((1-REMAINING_POINTS/ALL_REMAINING_POINTS)*100)).zfill(5) + '%   ' + 'Time : ' +  ('%.2f' % ((time()-START))) + '  ' + 'Best Quality : ' + str(best[2]) + '  ' + 'Guarantee : ' + str(GUARANTEE) + '  '+ 'Crispiness : '+ str('%.5f'%max(HASHMAP_CRISPENESS.values()))));stdout.flush();
			START+=time()-before_writing	
		before_yield=time()
		if USE_ALGO:
			accuracy_guarantee_to_ret=max(HASHMAP_GUARANTEES.values()) if len(HASHMAP_GUARANTEES) else Guarantee
			if CRISPINESS_COMPUTATION:
				crispiness_guarantee_to_ret=max(HASHMAP_CRISPENESS.values())
			else:
				crispiness_guarantee_to_ret=1.
			#yield ALL_PATTERNS_FOUND,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,accuracy_guarantee_to_ret,crispiness_guarantee_to_ret
			yield HASHMAP_ALL_PATTERNS_FOUND.values(),dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,accuracy_guarantee_to_ret,crispiness_guarantee_to_ret

		else:
			yield ALL_PATTERNS_FOUND,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new
		if CLEAN_PATTERNS_FOUND_LIST:
			ALL_PATTERNS_FOUND=[]
			APPEND_RESULTS=ALL_PATTERNS_FOUND.append
		after_yield=time()-before_yield
		START+=after_yield
		BIG_ITERATION+=1
		left_zone_to_mine_next,right_zone_to_mine_next,left_partition_name,right_partition_name, \
		left_zone_to_mine_next_to_check_guarantees_in,right_zone_to_mine_next_to_check_guarantees_in,left_zone_to_mine_next_to_check_guarantees_in_right_end,right_zone_to_mine_next_to_check_guarantees_in_left_end = update_discretization_and_partitions(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,index_attr,attributes_discretized,attr_to_cut,cut_point_indice)
		
		reorganized_attributes=[attr_to_cut]+[a for a in attributes if a != attr_to_cut]
		right_partition_name_is_neg=partitions_to_dataset_bitset_new[attr_to_cut][right_partition_name]['=']&positive_extent_bitset==ZERO
		left_partition_name_is_pos_is_neg=partitions_to_dataset_bitset_new[attr_to_cut][left_partition_name]['=']&positive_extent_bitset==ZERO
		
		############################Guarantees_Checking###################################################
		
		if can_enhance or CRISPINESS_COMPUTATION:
			if CRISPINESS_COMPUTATION and not can_enhance:
				before_writing_at_all=time()
			#enum_guarantees_left_zone=enum_function_to_use(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,index_attr,left_zone_to_mine_next_to_check_guarantees_in,info_to_push={'indice_to_check':-1,'equal_with':left_zone_to_mine_next_to_check_guarantees_in_right_end},threshold_sup=threshold_sup,fixed_left=False,fixed_right=True)
			#enum_guarantees_right_zone=enum_function_to_use(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,index_attr,right_zone_to_mine_next_to_check_guarantees_in,info_to_push={'indice_to_check':0,'equal_with':right_zone_to_mine_next_to_check_guarantees_in_left_end},threshold_sup=threshold_sup,fixed_left=True,fixed_right=False)
			ITERATORS_TO_CHAINS=[]
			if not right_partition_name_is_neg:
				left_zone_to_mine_next_to_check_guarantees_in=left_zone_to_mine_next_to_check_guarantees_in&positive_extent
				enum_guarantees_left_zone=enum_function_to_use(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,index_attr,left_zone_to_mine_next_to_check_guarantees_in,info_to_push={'indice_to_check':-1,'equal_with':right_partition_name},threshold_sup=threshold_sup,fixed_left=False,fixed_right=True)
				ITERATORS_TO_CHAINS.append(enum_guarantees_left_zone)
			if not left_partition_name_is_pos_is_neg:
				right_zone_to_mine_next_to_check_guarantees_in=right_zone_to_mine_next_to_check_guarantees_in&positive_extent
				enum_guarantees_right_zone=enum_function_to_use(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,index_attr,right_zone_to_mine_next_to_check_guarantees_in,info_to_push={'indice_to_check':0,'equal_with':left_partition_name},threshold_sup=threshold_sup,fixed_left=True,fixed_right=False)
				ITERATORS_TO_CHAINS.append(enum_guarantees_right_zone)
			
			if len(ITERATORS_TO_CHAINS):
				enum_left_right_guarantees_zone=chain(*ITERATORS_TO_CHAINS)
				for p,supPOS,sup_bitset,sup,sup_full_bitset,current_refinement_index,cnt_dict in enum_left_right_guarantees_zone:
					

					if ONLY_BITSET:
						supNEG_bitset=sup_full_bitset&negative_extent_bitset
						tpr=nb_bit_1(sup_bitset)/nb_positives
					else:
						supNEG=sup&negative_extent
						tpr=len(supPOS)/nb_positives

					core_p,sup_core_p,bitset_core_p=core_stringent(p,sup,sup_full_bitset,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,attributes_discretized)
					if CORE_CLOSE_ON_POSITIVE:#FIX Error 
						#core_p,sup_core_p,bitset_core_p=compute_closed_on_the_positive_full_extent(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,sup_core_p,positive_extent)
						bitset_core_p_plus=from_closed_bitset_to_support_corresponding_to_the_closed_on_the_positive_full_extent(dataset_to_partitions,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,p,bitset_core_p,positive_extent_bitset)
						#print 'hey'
						# print bitset_core_p_plus==bitset_core_p,bitset_core_p_plus<=bitset_core_p

						# if bitset_core_p != bitset_core_p_plus:
						# 	print bitset_core_p_plus,bitset_core_p,len(bitset_core_p_plus),len(bitset_core_p)

					else:
						bitset_core_p_plus=bitset_core_p


					tpr_core=0.
					fpr_core=0.
					if ONLY_BITSET:
						if bitset_core_p!=ZERO:
							fpr_core=nb_bit_1(bitset_core_p&supNEG_bitset)/nb_negatives
					else:
						if len(sup_core_p):
							fpr_core=len(sup_core_p&supNEG)/nb_negatives

					if fpr_core==0.:
						if not CRISPINESS_COMPUTATION:
							cnt_dict['continue']=False
						else:
							cnt_dict['continue']=True
							
						#cnt_dict['continue']=False


					local_Guarantee=DISCRIMINATION_MEASURE(tpr,fpr_core,alpha_ratio_class)
					if local_Guarantee<best[2]: 
						HASHMAP_GUARANTEES_POP(sup_bitset, None)
						#HASHMAP_GUARANTEES_SUP_POP(sup_bitset, None)
						HASHMAP_GUARANTEES_INTERVAL_POP(sup_bitset, None)
						if CRISPINESS_COMPUTATION:
							#HASHMAP_CRISPENESS_POP(sup_bitset, None)
							before_writing=time()
							HASHMAP_CRISPENESS[sup_bitset.strbits()]= float(len(sup_full_bitset - bitset_core_p_plus)/(2.*float(len_full_dataset)))#,HASHMAP_CRISPENESS_GET(sup_bitset,1.))
							if (can_enhance):
								START+=time()-before_writing
					else: 
						HASHMAP_GUARANTEES[sup_bitset]=local_Guarantee#min(local_Guarantee,HASHMAP_GUARANTEES_get(sup_bitset,local_Guarantee))
						#HASHMAP_GUARANTEES_SUP[sup_bitset]=supPOS
						HASHMAP_GUARANTEES_INTERVAL[sup_bitset]=pattern_reorganize(p,attributes,reorganized_attributes)
						if CRISPINESS_COMPUTATION:
							before_writing=time()
							HASHMAP_CRISPENESS[sup_bitset.strbits()]= float(len(sup_full_bitset - bitset_core_p_plus)/(2.*float(len_full_dataset)))#,HASHMAP_CRISPENESS_GET(sup_bitset,1.))
							if (can_enhance):
								START+=time()-before_writing
							
			if CRISPINESS_COMPUTATION and not can_enhance:
				START+=time()-before_writing_at_all
		############################Guarantees_Checking###################################################

		############################New_Patterns_Enumeration###################################################
		#if CLOSE_ON_POSITIVE:
		
		ITERATORS_TO_CHAINS=[]
		
		if not left_partition_name_is_pos_is_neg:
			left_zone_to_mine_next=left_zone_to_mine_next&positive_extent
			enum_left_zone=enum_function_to_use(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,index_attr,left_zone_to_mine_next,info_to_push={'indice_to_check':-1,'equal_with':left_partition_name},threshold_sup=threshold_sup,fixed_left=False,fixed_right=True)
			ITERATORS_TO_CHAINS.append(enum_left_zone)

		if not right_partition_name_is_neg:
			right_zone_to_mine_next=right_zone_to_mine_next&positive_extent
			enum_right_zone=enum_function_to_use(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,index_attr,right_zone_to_mine_next,info_to_push={'indice_to_check':0,'equal_with':right_partition_name},threshold_sup=threshold_sup,fixed_left=True,fixed_right=False)
			ITERATORS_TO_CHAINS.append(enum_right_zone)
		
		if len(ITERATORS_TO_CHAINS):	
			enum_left_right_zone=chain(*ITERATORS_TO_CHAINS)
			for p,supPOS,sup_bitset,sup,sup_full_bitset,current_refinement_index,cnt_dict in enum_left_right_zone:
				
				numero+=1
				if ONLY_BITSET:
					supNEG_bitset=sup_full_bitset&negative_extent_bitset
					tpr=nb_bit_1(sup_bitset)/nb_positives
					fpr=nb_bit_1(supNEG_bitset)/nb_negatives
					
					if fpr==0:
						if not CRISPINESS_COMPUTATION:
							cnt_dict['continue']=False
						else:
							cnt_dict['continue']=True
							

				else:
					supNEG=sup&negative_extent
					tpr=len(supPOS)/nb_positives
					fpr=len(supNEG)/nb_negatives

				quality_pattern=DISCRIMINATION_MEASURE(tpr,fpr,alpha_ratio_class)
				if quality_pattern>best[2]:
					best=(pattern_value_to_print(p,partitions_to_dataset_new,attributes,reorganized_attributes),sup,quality_pattern)

				if keep_patterns_found:
					if quality_pattern>threshold_qual:
						#ALL_PATTERNS_FOUND[sup_bitset]=((pattern_value_to_print(p,partitions_to_dataset_new,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern))
						

						if LIGHTER_PATTERN_SET:
						#ALL_PATTERNS_FOUND[sup_bitset]=(pattern_value_to_print(p,partitions_to_dataset,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern)
							compressed_sup_full_bitset=sup_full_bitset.fastdump()
							APPEND_RESULTS((compressed_sup_full_bitset,quality_pattern))
						else:
							APPEND_RESULTS((pattern_value_to_print(p,partitions_to_dataset_new,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern))
							if USE_ALGO:
								HASHMAP_ALL_PATTERNS_FOUND[sup_bitset.strbits()]=(pattern_value_to_print(p,partitions_to_dataset_new,attributes,reorganized_attributes),sup,sup_full_bitset,quality_pattern)

				if can_enhance or CRISPINESS_COMPUTATION:
					do_i_update=cnt_dict.get('continue_computation_guarantees',can_enhance)
					
					if do_i_update or CRISPINESS_COMPUTATION:
						if CRISPINESS_COMPUTATION and (not do_i_update):
							before_writing_at_all=time()
						core_p,sup_core_p,bitset_core_p=core_stringent(p,sup,sup_full_bitset,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,attributes_discretized)
						if CORE_CLOSE_ON_POSITIVE:#FIX Error 
							#core_p,sup_core_p,bitset_core_p=compute_closed_on_the_positive_full_extent(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,sup_core_p,positive_extent)
							bitset_core_p_plus=from_closed_bitset_to_support_corresponding_to_the_closed_on_the_positive_full_extent(dataset_to_partitions,partitions_to_dataset_new,partitions_to_dataset_bitset_new,reorganized_attributes,p,bitset_core_p,positive_extent_bitset)
						else:
							bitset_core_p_plus=bitset_core_p

						tpr_core=0.
						fpr_core=0.
						if ONLY_BITSET:
							if bitset_core_p!=ZERO:
								fpr_core=nb_bit_1(bitset_core_p&supNEG_bitset)/nb_negatives
						else:
							if len(sup_core_p):
								fpr_core=len(sup_core_p&supNEG)/nb_negatives
						local_Guarantee=DISCRIMINATION_MEASURE(tpr,fpr_core,alpha_ratio_class)
						Guarantee=max(Guarantee,local_Guarantee)
						if local_Guarantee<best[2]: 
							HASHMAP_GUARANTEES_POP(sup_bitset, None)
							#HASHMAP_GUARANTEES_SUP_POP(sup_bitset, None)
							HASHMAP_GUARANTEES_INTERVAL_POP(sup_bitset, None)
							if CRISPINESS_COMPUTATION:
								#HASHMAP_CRISPENESS_POP(sup_bitset, None)
								before_writing=time()
								HASHMAP_CRISPENESS[sup_bitset.strbits()]= float(len(sup_full_bitset - bitset_core_p_plus)/(2.*float(len_full_dataset)))#,HASHMAP_CRISPENESS_GET(sup_bitset,1.))
								if (do_i_update):
									START+=time()-before_writing
								
						else: 
							HASHMAP_GUARANTEES[sup_bitset]=local_Guarantee#min(local_Guarantee,HASHMAP_GUARANTEES_get(sup_bitset,local_Guarantee))
							#HASHMAP_GUARANTEES_SUP[sup_bitset]=supPOS
							HASHMAP_GUARANTEES_INTERVAL[sup_bitset]=pattern_reorganize(p,attributes,reorganized_attributes)
							if CRISPINESS_COMPUTATION:
								before_writing=time()
								HASHMAP_CRISPENESS[sup_bitset.strbits()]= float(len(sup_full_bitset - bitset_core_p_plus)/(2.*float(len_full_dataset)))#,HASHMAP_CRISPENESS_GET(sup_bitset,1.))
								if (do_i_update):
									START+=time()-before_writing
								
								
						
						if CRISPINESS_COMPUTATION and (not do_i_update):
							START+=time()-before_writing_at_all
					if fpr_core==0.:
						cnt_dict['continue_computation_guarantees']=False
			
		############################New_Patterns_Enumeration###################################################
		
		sup_bitset_guarantee,val_guarantee=max(HASHMAP_GUARANTEES.items(),key=lambda x:x[1])
		#interval_guarantee=pattern_to_yield(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,attributes,closed_numeric(dataset_to_partitions_new,partitions_to_dataset_new,attributes,sup_guarantee))
		
		if ONLY_BITSET:
			interval_guarantee=pattern_to_yield(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,attributes,HASHMAP_GUARANTEES_INTERVAL[sup_bitset_guarantee])
		else:
			sup_guarantee=HASHMAP_GUARANTEES_SUP[sup_bitset_guarantee]
			interval_guarantee=closed_numeric(dataset_to_partitions_new,partitions_to_dataset_new,attributes,sup_guarantee)
		
		attr_to_cut,cut_point_indice,can_enhance=PROCEDURE_OF_SELECTION(dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,attributes,attributes_discretized,index_attr,interval_guarantee,positive_extent,negative_extent,can_enhance=can_enhance)
		#print can_enhance

	#raw_input('*******')
	print ''
	print '------------FINAL------------------'
	print 'nb : ',numero
	print 'pattern : ', best[0]
	print 'quality : ' , best[2]
	print 'Guarantee : ',max(HASHMAP_GUARANTEES.values())
	print 'Length of HashMap : ',len(HASHMAP_GUARANTEES)#,HASHMAP_GUARANTEES.values()
	print 'Time spent : ',time()-START
	TIMESERIE_QUALITY.append((time()-START,best[2],numero))
	TIMESERIE_GUARANTEE.append((time()-START,max(HASHMAP_GUARANTEES.values()),numero))
	TIMESERIE_GLOBALITY.append((time()-START,max(HASHMAP_CRISPENESS.values()),numero)) if CRISPINESS_COMPUTATION else TIMESERIE_GLOBALITY.append((time()-START,1.,numero))
	if CRISPINESS_COMPUTATION:
		print 'CRISPINESS : ',max(HASHMAP_CRISPENESS.values())
	print '------------FINAL------------------'

	if USE_ALGO:
		accuracy_guarantee_to_ret=max(HASHMAP_GUARANTEES.values()) if len(HASHMAP_GUARANTEES) else Guarantee
		if CRISPINESS_COMPUTATION:
			crispiness_guarantee_to_ret=max(HASHMAP_CRISPENESS.values())
		else:
			crispiness_guarantee_to_ret=1.
		#yield ALL_PATTERNS_FOUND,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,accuracy_guarantee_to_ret,crispiness_guarantee_to_ret
		yield HASHMAP_ALL_PATTERNS_FOUND.values(),dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,accuracy_guarantee_to_ret,crispiness_guarantee_to_ret
		
	else:
		yield ALL_PATTERNS_FOUND,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new

	
	
	#return TIMESERIE_QUALITY,TIMESERIE_GUARANTEE


def signal_handler(signal, frame):
	from plotter.plotter import plot_timeseries
	
	#timeseries={'Quality':([x[0] for x in timeserie_qual],[x[1] for x in timeserie_qual]),'Guarantees':([x[0] for x in timeserie_guarantee],[x[1] for x in timeserie_guarantee])}
	timeserie_qual,timeserie_guarantee=TIMESERIE_QUALITY,TIMESERIE_GUARANTEE
	#timeseries={'Quality':([x[1] for x in timeserie_qual]),'Guarantees':([x[1] for x in timeserie_guarantee]),'Crispiness':TIMESERIE_GLOBALITY}
	timeseries={'Quality':([x[0] for x in timeserie_qual],[x[1] for x in timeserie_qual]),'Guarantees':([x[0] for x in timeserie_guarantee],[x[1] for x in timeserie_guarantee]),'Crispiness':TIMESERIE_GLOBALITY}
	plot_timeseries(timeseries)
	print('You pressed Ctrl+C!')
	sys.exit(0)


def mymax(l):

	l_iter=iter(l)
	try:
		max_to_ret=next(l_iter)
		for e in l_iter:
			if e>max_to_ret: max_to_ret=e
			if max_to_ret>=1.:
				return max_to_ret
		return max_to_ret
	except:
		return 0.

def mymin(l):

	l_iter=iter(l)
	try:
		min_to_ret=next(l_iter)
		for e in l_iter:
			if e<min_to_ret: min_to_ret=e
			if min_to_ret<=0.:
				return min_to_ret
		return min_to_ret
	except:
		return 0.

def similarity_between_patterns_set(list_of_extents_1,list_of_extents_2,dictionnary_infos={}):
	

	if len(list_of_extents_1)==0 or len(list_of_extents_2)==0:
		return 0.,0.,0.
	
	returned_1=0.
	for extent1 in list_of_extents_1:
		returned_1+=mymax(jaccard(extent1,extent2) for extent2 in list_of_extents_2)
	recall=returned_1/len(list_of_extents_1)
	returned_2=0.
	precision=1.
	
	# for extent2 in list_of_extents_2:
	# 	returned_2+=mymax(jaccard(extent1,extent2) for extent1 in list_of_extents_1)
	
	# precision=returned_2/len(list_of_extents_2)
	
	fscore=2./(1./precision+1./recall)
	return fscore,precision,recall




# def size_of_symmetric_difference(extent1,extent2,dataset):
# 	return (len(extent1 | extent2)-len(extent1 & extent2))/float(len(dataset))

def size_of_symmetric_difference(extent1,extent2,dataset):
	return (len(extent1 | extent2)-len(extent1 & extent2))/float(len(dataset))

def similarity_between_patterns_set_globaility(list_of_extents_1,list_of_extents_2,dataset,dictionnary_infos={}):
	globaility=0.
	for extent1 in list_of_extents_1:
		globaility=max(globaility,mymin(size_of_symmetric_difference(extent1,extent2,dataset) for extent2 in list_of_extents_2))
	return globaility



def similarity_between_patterns_set_new(list_of_extents_1,list_of_extents_2,dictionnary_infos={}):
	

	# if len(list_of_extents_1)==0 or len(list_of_extents_2)==0:
	# 	return 0.,0.,0.
	
	returned_1=0.
	for i,extent1 in enumerate(list_of_extents_1):
		if len(list_of_extents_2)>0:
			if dictionnary_infos[i]==1.:
				v=dictionnary_infos[i]
			else:
				v=max(dictionnary_infos[i],mymax(jaccard(extent1,extent2) for extent2 in list_of_extents_2))
		else:
			v=dictionnary_infos[i]
		dictionnary_infos[i]=v
		returned_1+=v
	recall=returned_1/len(list_of_extents_1)
	returned_2=0.
	precision=1.
	
	# for extent2 in list_of_extents_2:
	# 	returned_2+=mymax(jaccard(extent1,extent2) for extent1 in list_of_extents_1)
	
	# precision=returned_2/len(list_of_extents_2)
	
	fscore=2./(1./precision+1./recall)
	return fscore,precision,recall

def similarity_between_patterns_set_globaility_new(list_of_extents_1,list_of_extents_2,dataset,dictionnary_infos={}):
	globaility=0.
	for i,extent1 in enumerate(list_of_extents_1):
		if len(list_of_extents_2)>0:
			if dictionnary_infos[i]==0.:
				v=dictionnary_infos[i]
			else:
				v=min(dictionnary_infos[i],mymin(size_of_symmetric_difference(extent1,extent2,dataset) for extent2 in list_of_extents_2))
		else:
			v=dictionnary_infos[i]
		dictionnary_infos[i]=v
		globaility=max(globaility,v)
	return globaility


alphabet=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
nbHeader=[''.join(x) for x in product(alphabet,repeat=1)]+[''.join(x) for x in product(alphabet,repeat=2)]

DATASETDICTIONNARY={
	'HABERMAN':{
		'data_file':'.//datasets//haberman.csv',
		'attributes':['a','b','c'],
		'attr_label':'class',
		'wanted_label':'1'
	},
	'AUTOS':{
		'data_file':'.//datasets//autos.csv',
		'attributes':['compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price'],
		'attr_label':'symboling',
		'wanted_label':'0'
	},
	'BREAST':{
		'data_file':'.//datasets//breastCancerOriginal.csv',
		'attributes':['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses'],
		'attr_label':'Class',
		'wanted_label':'4'
	},
	
	'BEAST_TISSUE':{
		'data_file':'.//datasets//BreastTissue.csv',
		'attributes':['I0','PA500','HFS','DA','Area','A/DA','Max IP','DR','P'],
		'attr_label':'Class',
		'wanted_label':'car'
	},
	
	'CREDITA':{
		'data_file':'.//datasets//creditApproved.csv',
		'attributes':['A2','A3','A8','A11','A14','A15'],
		'attr_label':'class',
		'wanted_label':'+'
	},
	'GLASS':{
		'data_file':'.//datasets//glass.csv',
		'attributes':['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'],
		'attr_label':'class',
		'wanted_label':'1'
	},
	'SONAR':{
		'data_file':'.//datasets//sonar.csv',
		'attributes':nbHeader[:60],
		'attr_label':'class',
		'wanted_label':'R'
	},
	'DERMATOLOGY':{
		'data_file':'.//datasets//dermatology.csv',
		'attributes':['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','ac','ad','ae','af','ag','ah'],
		'attr_label':'class',
		'wanted_label':'1'
	},
	'ABALONE':{
		'data_file':'.//datasets//abalone.csv',
		'attributes':['a','b','c','d','e','f','g','h'],
		'attr_label':'class',
		'wanted_label':'M'
	},
	'BALANCE':{
		'data_file':'.//datasets//balance.csv',
		'attributes':['a','b','c','d'],
		'attr_label':'class',
		'wanted_label':'R'
	},


	'CMC':{
		'data_file':'.//datasets//CMC.csv',
		'attributes':['a','b','c','d','e','f','g','h'],
		'attr_label':'class',
		'wanted_label':'1'
	},

	'IRIS':{
		'data_file':'.//datasets//iris.csv',
		'attributes':['a','b','c','d'],
		'attr_label':'class',
		'wanted_label':'Iris-setosa'
	},
	'OLFACTION':{
		'data_file':'.//datasets//olfaction.csv',
		'attributes':['H%','C%','N%','O%','X%'],
		'attr_label':'musk',
		'wanted_label':'1'
	}
}

MEASURES_DICTIONNARY={
	'informedness':informedness,
	'wracc':wracc,
	'linearCorr':linear_corr
}

def stringifier(t):
	s=''
	for i in range(len(t)):
		s+=str(t[i]+1)+' '
	s=s[:-1]
	return s






def Refine_and_mine_to_use(
	file_path,
	attributes,
	attr_label,
	wanted_label,
	time_budget=7200,
	selected_quality_measure='informedness',
	threshold_quality=0.,
	threshold_sup=1.,
	threshold_sim=0.,
	top_k=10,
	specifity_guarantee_computation=False,
	delimiter='\t'):
	

	dataset_file_to_consider=file_path
	attributes=attributes
	attr_label=attr_label
	wanted_label=wanted_label
	
	MAXIMUM_TIME=time_budget
	THRESHOLD_QUAL=threshold_quality
	THRESHOLD_SUP=threshold_sup
	THRESHOLD_SIM=threshold_sim
	SELECTED_MEASURE=selected_quality_measure
	TOP_K=top_k
	CRISPINESS_COMPUTATION=specifity_guarantee_computation



	##################################################################################################
	SELECTED_MEASURE=MEASURES_DICTIONNARY[SELECTED_MEASURE]
	index_initialization=time()
	dataset,h=readCSVwithHeader(dataset_file_to_consider,numberHeader=attributes,delimiter=delimiter)
	dataset,positive_extent,negative_extent,alpha_ratio_class,statistics=transform_dataset(dataset,attributes,attr_label,wanted_label,verbose=False)
	index_attr,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset=compute_index_all_attributes(dataset,attributes,positive_extent=positive_extent,negative_extent=negative_extent)
	index_initialization=time()-index_initialization
	nb_pos=float(len(positive_extent))
	
	if THRESHOLD_SUP<1:
		THRESHOLD_SUP=THRESHOLD_SUP*nb_pos

	for yielded_pattern_found,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,acc_guarantee,acc_crispiness  in discretize_and_mine_STRINGENT_2(dataset,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
											positive_extent,negative_extent,alpha_ratio_class,
											DISCRIMINATION_MEASURE=SELECTED_MEASURE,PROCEDURE_OF_SELECTION=select_next_cut_point_median_value_in_the_border,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,verbose=False,MAXIMUM_TIME=MAXIMUM_TIME,CRISPINESS_COMPUTATION=CRISPINESS_COMPUTATION,USE_ALGO=True):
		continue
	

	#print 'REFINEANDMINE:',len(yielded_pattern_found)


	topk_patterns_refine_and_mine=[(a,c,c,d) for (a,b,c,d) in yielded_pattern_found]
	################################### >POST PROCESSING< ##################################
	topk_patterns_refine_and_mine=get_top_k_div_from_a_pattern_set(topk_patterns_refine_and_mine,threshold_sim=THRESHOLD_SIM,k=TOP_K)
	################################### >POST PROCESSING< ##################################
	HEADER=['id_pattern','attributes','pattern','support_size','support_size_ratio',selected_quality_measure,'tpr','fpr','guarantee_accuracy','guarantee_specificity']


	to_return=[]
	for id_pattern,(p,p_sup,p_bitset,qual) in enumerate(topk_patterns_refine_and_mine):
		#print attributes,p,len(set(p_bitset)),qual #p,p_sup,set(p_bitset)
		to_return.append({
			'id_pattern':id_pattern,
			'attributes':attributes,
			'pattern':p,
			'support_size':len(set(p_bitset)),
			'support_size_ratio':len(set(p_bitset))/float(len(dataset)),
			selected_quality_measure : qual,
			'tpr':len(set(p_bitset)&positive_extent)/float(len(positive_extent)),

			'fpr':len(set(p_bitset)&negative_extent)/float(len(negative_extent)),
			'guarantee_accuracy':acc_guarantee,
			'guarantee_specificity':acc_crispiness,

		})
	#print 'support_size : ',to_return[0]['support_size']
	return to_return,HEADER




if __name__ == '__main__':
	#########DEFAULT_VALUES#######################
	DATASET_TO_USE='HABERMAN'#BALANCE
	NB_ATTRIBUTES=3
	WANTED_LABEL='1'
	COMPUTE_EXHAUSTIVE=True
	destinationCSVFile=DATASET_TO_USE+'_'+str(NB_ATTRIBUTES).zfill(2)
	PLOTFIGURE=False
	THRESHOLD_QUAL=0
	THRESHOLD_SUP=1#0.2*nb_pos
	THRESHOLD_SIM=0
	SELECTED_MEASURE='informedness'
	TOP_K=10
	SAVE_EXHAUSTIVE=False
	MAXIMUM_TIME=7200
	KEEP_PATTERNS_FOUND=False
	CRISPINESS_COMPUTATION=False
	#########DEFAULT_VALUES#######################

	if False:
		signal.signal(signal.SIGINT, signal_handler)
		print('Press Ctrl+C')
	#signal.pause()
	
	#print sys.argv[0]
	path, file = os.path.split(sys.argv[0])
	#print path, file
	#raw_input('....')
	parser = argparse.ArgumentParser(description='DiscretizeAndMine')
	parser.add_argument('--STATS',action='store_true',help='provide the stats of the considered datasets')
	parser.add_argument('--Q1',action='store_true',help='Answer to the question Q1 - how does the quality and the guarantee evolve through time')
	parser.add_argument('--Q2',action='store_true',help='Answer to the question Q2 - does our algorithm covers well the full search space (Globaility and diversity)')
	parser.add_argument('--Q3',action='store_true',help='Answer to the question Q3 - comparative study')
	parser.add_argument('--VIS',action='store_true',help='Visualize the beauty of RefineAndMine')

	parser.add_argument('--dataset',metavar='dataset',type=str,help='dataset name')
	parser.add_argument('--nbattr',metavar='nbattr',type=int,help='nb attributes')
	
	parser.add_argument('--dataset_file',metavar='dataset_file',type=str,help='dataset file path')
	parser.add_argument('--delimiter',metavar='delimiter',type=str,help='delimiter of the csv file',default='\t')
	parser.add_argument('--attributes',metavar='attributes',nargs='*',help='input the name of numerical attribute that you want to consider in the mining process')
	parser.add_argument('--label_attribute',metavar='label_attribute',type=str,help='input the name the label class that you want to consider')
	parser.add_argument('--wanted_label',metavar='wanted_label',type=str,help='considered label')
	parser.add_argument('--time_budget',metavar='time_budget',type=float,help='timebudget in seconds')
	parser.add_argument('--results_file',metavar='results_file',type=str,help='results file')


	parser.add_argument('--sigma_sup',metavar='sigma_sup',type=float,help='support threshold')
	parser.add_argument('--sigma_qual',metavar='sigma_qual',type=float,help='quality threshold')
	parser.add_argument('--sigma_sim',metavar='sigma_sim',type=float,help='similarity threshold')
	parser.add_argument('--top_k',metavar='top_k',type=float,help='number of patterns to keep')
	parser.add_argument('--quality_measure',metavar='quality_measure',type=str,help='quality measure (informedness, wracc, linearCorr)')
	parser.add_argument('--compute_crispiness',action='store_true',help='Compute crispiness on the fly')

	parser.add_argument('--plot',metavar='plot',type=str,help='csv file')
	parser.add_argument('--x_axis_attribute',metavar='x_axis_attribute',type=str,help='which attribute to vary')
	parser.add_argument('--methods',metavar='methods',type=str,nargs='+',help='which attribute to study')
	parser.add_argument('--exhaustive_info',action='store_true',help='First line of csv contains info about the exhaustive approach')



	parser.add_argument('--USE_ALGO',action='store_true',help='use the algorithm naturally to provide patterns')



	parser.add_argument('--save_exhaustive',action='store_true',help='save exhaustive top-k list in a file (ground truth)')

	args=parser.parse_args()


	if args.USE_ALGO:
		KEEP_PATTERNS_FOUND=True
		PLOTFIGURE=False

		dataset_file_to_consider=args.dataset_file
		attributes=args.attributes
		attr_label=args.label_attribute
		wanted_label=args.wanted_label if args.wanted_label is not None else WANTED_LABEL 
		delimiter=args.delimiter
		
		MAXIMUM_TIME=args.time_budget if args.time_budget is not None else 7200


		#raw_input('....')
		THRESHOLD_QUAL=args.sigma_qual if args.sigma_qual is not None else 0.
		THRESHOLD_SUP=args.sigma_sup if args.sigma_sup is not None else 1
		THRESHOLD_SIM=args.sigma_sim if args.sigma_sim is not None else 0.
		SELECTED_MEASURE=args.quality_measure if args.quality_measure is not None else 'informedness'
		TOP_K=int(args.top_k) if args.top_k is not None else 10
		CRISPINESS_COMPUTATION=args.compute_crispiness# if args.compute_crispiness is not None else CRISPINESS_COMPUTATION
		WRITE_RESULTS_FILE=args.results_file if args.results_file is not None else 'resultNow.csv'
		NB_ATTRIBUTES=args.nbattr if args.nbattr is not None else NB_ATTRIBUTES

		print '------------------------------------------------------------------------'
		print 'dataset:',dataset_file_to_consider
		print 'attributes:',attributes,type(attributes)
		print 'label_class:',attr_label
		print 'wanted_label:',wanted_label
		print 'delimiter:',delimiter

		print 'threshold_quality:',THRESHOLD_QUAL
		print 'threshold_support_pos:',THRESHOLD_SUP
		print 'Threshold similarity:',THRESHOLD_SIM
		print 'Selected Interestingness measure:',SELECTED_MEASURE

		print 'TOP_K:',TOP_K
		print 'Compute Specifity Guarantee:',CRISPINESS_COMPUTATION

		print 'Time budget:',MAXIMUM_TIME

		print 'Results file:',WRITE_RESULTS_FILE
		print '------------------------------------------------------------------------'




		# SELECTED_MEASURE=MEASURES_DICTIONNARY[SELECTED_MEASURE]
		# index_initialization=time()
		# dataset,h=readCSVwithHeader(dataset_file_to_consider,numberHeader=attributes,delimiter=delimiter)
		
		# dataset,positive_extent,negative_extent,alpha_ratio_class,statistics=transform_dataset(dataset,attributes,attr_label,wanted_label,verbose=False)
		# index_attr,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset=compute_index_all_attributes(dataset,attributes,positive_extent=positive_extent,negative_extent=negative_extent)
		# index_initialization=time()-index_initialization
		# nb_pos=float(len(positive_extent))
		
		# if THRESHOLD_SUP<1:
		# 	THRESHOLD_SUP=THRESHOLD_SUP*nb_pos


		# # timeserie_qual,timeserie_guarantee,all_patterns_found_exhaustive= next(exhaustive_mine(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
		# # 													positive_extent,negative_extent,alpha_ratio_class,DISCRIMINATION_MEASURE=SELECTED_MEASURE,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,MAXIMUM_TIME=MAXIMUM_TIME))
		# # print 'EXHAUSTIVE:',len(all_patterns_found_exhaustive)

		# for yielded_pattern_found,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,acc_guarantee,acc_crispiness  in discretize_and_mine_STRINGENT_2(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
		# 										positive_extent,negative_extent,alpha_ratio_class,
		# 										DISCRIMINATION_MEASURE=SELECTED_MEASURE,PROCEDURE_OF_SELECTION=select_next_cut_point_median_value_in_the_border,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,verbose=False,MAXIMUM_TIME=MAXIMUM_TIME,CRISPINESS_COMPUTATION=CRISPINESS_COMPUTATION,USE_ALGO=True):
		# 	#print acc_guarantee,acc_crispiness	
		# 	continue
		

		# print 'REFINEANDMINE:',len(yielded_pattern_found)

		# topk_patterns_refine_and_mine=[(a,c,c,d) for (a,b,c,d) in yielded_pattern_found]
		# # if TOP_K>=len(topk_patterns_refine_and_mine):
		# # 	return 
		# topk_patterns_refine_and_mine=get_top_k_div_from_a_pattern_set(topk_patterns_refine_and_mine,threshold_sim=THRESHOLD_SIM,k=TOP_K)


		Results,HEADER=Refine_and_mine_to_use(
			dataset_file_to_consider,
			attributes,
			attr_label,
			wanted_label,
			time_budget=MAXIMUM_TIME,
			selected_quality_measure=SELECTED_MEASURE,
			threshold_quality=THRESHOLD_QUAL,
			threshold_sup=THRESHOLD_SUP,
			threshold_sim=THRESHOLD_SIM,
			top_k=TOP_K,
			specifity_guarantee_computation=CRISPINESS_COMPUTATION,
			delimiter=delimiter)


		# topk_patterns_exhaustive=[(a,c,c,d) for (a,b,c,d) in all_patterns_found_exhaustive]
		# topk_patterns_exhaustive=get_top_k_div_from_a_pattern_set(topk_patterns_exhaustive,threshold_sim=THRESHOLD_SIM,k=TOP_K)

		

		#HEADER=['id_pattern','attributes','pattern','support_size','support_size_ratio','quality']



		# to_write=[]
		# for id_pattern,(p,p_sup,p_bitset,qual) in enumerate(topk_patterns_refine_and_mine):
		# 	print attributes,p,len(set(p_bitset)),qual #p,p_sup,set(p_bitset)
		# 	to_write.append({
		# 		'id_pattern':id_pattern,
		# 		'attributes':attributes,
		# 		'pattern':p,
		# 		'support_size':len(set(p_bitset)),
		# 		'support_size_ratio':len(set(p_bitset))/float(len(dataset)),
		# 		'quality':qual,
		# 	})

		writeCSVwithHeader(Results,WRITE_RESULTS_FILE,selectedHeader=HEADER,flagWriteHeader=True)
		print 'Results Written in the following path ', WRITE_RESULTS_FILE

		#raw_input('****************')






		# DATASET_TO_USE=args.dataset if args.dataset is not None else DATASET_TO_USE
	
		
		# COMPUTE_EXHAUSTIVE=COMPUTE_EXHAUSTIVE
		# destinationCSVFile=DATASET_TO_USE+'_'+str(NB_ATTRIBUTES).zfill(2)+'_'+str(WANTED_LABEL)
		
		

		
		


		# #if False:
		
		# print '--------------------------------------------Q1---------------------------------------------------'
		# print 'Dataset : ',DATASET_TO_USE
		# print 'NB Attributes : ',NB_ATTRIBUTES
		# print 'Wanted Label : ',WANTED_LABEL
		# print 'Selected Measure : ',SELECTED_MEASURE
		# print 'Quality Threshold : ',THRESHOLD_QUAL
		# print 'Support Threshold : ',THRESHOLD_SUP
		# print '--------------------------------------------Q1---------------------------------------------------'

		# if not os.path.exists('./tmp'):
		# 	os.makedirs('./tmp')

		# DATASET_ENTITY_CONSIDERED=DATASETDICTIONNARY[DATASET_TO_USE]
		# data_file=path+'//'+DATASET_ENTITY_CONSIDERED['data_file']
		# attributes=DATASET_ENTITY_CONSIDERED['attributes'][:NB_ATTRIBUTES]
		# attr_label=DATASET_ENTITY_CONSIDERED['attr_label']
		# wanted_label=DATASET_ENTITY_CONSIDERED['wanted_label'] if args.wanted_label is None else WANTED_LABEL
		# SELECTED_MEASURE=MEASURES_DICTIONNARY[SELECTED_MEASURE]
		
		# index_initialization=time()
		
		# if True:
		# 	transform_to_the_wanted_structure(data_file,selectedHeader=attributes+[attr_label],delimiter=',',nb_attributes=NB_ATTRIBUTES,class_label=attr_label,wanted_label=wanted_label)

		# dataset,h=readCSVwithHeader(data_file,numberHeader=attributes,delimiter=',')
		# dataset,positive_extent,negative_extent,alpha_ratio_class,statistics=transform_dataset(dataset,attributes,attr_label,wanted_label,verbose=False)
		# index_attr,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset=compute_index_all_attributes(dataset,attributes,positive_extent=positive_extent,negative_extent=negative_extent)
		# index_initialization=time()-index_initialization
		# nb_pos=float(len(positive_extent))
		



		# # timeserie_qual,timeserie_guarantee,all_patterns_found_exhaustive= next(exhaustive_mine(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
		# # 													positive_extent,negative_extent,alpha_ratio_class,DISCRIMINATION_MEASURE=SELECTED_MEASURE,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,MAXIMUM_TIME=MAXIMUM_TIME))
		# # print 'EXHAUSTIVE:',len(all_patterns_found_exhaustive)

		# for yielded_pattern_found,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new,acc_guarantee,acc_crispiness  in discretize_and_mine_STRINGENT_2(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
		# 										positive_extent,negative_extent,alpha_ratio_class,
		# 										DISCRIMINATION_MEASURE=SELECTED_MEASURE,PROCEDURE_OF_SELECTION=select_next_cut_point_median_value_in_the_border,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,verbose=False,MAXIMUM_TIME=MAXIMUM_TIME,CRISPINESS_COMPUTATION=CRISPINESS_COMPUTATION,USE_ALGO=True):
		# 	#print acc_guarantee,acc_crispiness	
		# 	continue
		

		# print 'REFINEANDMINE:',len(yielded_pattern_found)
		# #print 'EXHAUSTIVE:',len(all_patterns_found_exhaustive)	
		# #print TOP_K

		
		# # PAT_R_and_M=[c for (a,b,c,d) in yielded_pattern_found]
		# # PAT_EXHAUSTIVE=[c for (a,b,c,d) in all_patterns_found_exhaustive]
		# # print positive_extent
		# # for p in PAT_EXHAUSTIVE:
		# # 	if p not in PAT_R_and_M:
		# # 		print set(p),set(p)&positive_extent
		# # 		for 







		# topk_patterns_refine_and_mine=[(a,c,c,d) for (a,b,c,d) in yielded_pattern_found]
		# # if TOP_K>=len(topk_patterns_refine_and_mine):
		# # 	return 
		# topk_patterns_refine_and_mine=get_top_k_div_from_a_pattern_set(topk_patterns_refine_and_mine,threshold_sim=THRESHOLD_SIM,k=TOP_K)



		# # topk_patterns_exhaustive=[(a,c,c,d) for (a,b,c,d) in all_patterns_found_exhaustive]
		# # topk_patterns_exhaustive=get_top_k_div_from_a_pattern_set(topk_patterns_exhaustive,threshold_sim=THRESHOLD_SIM,k=TOP_K)

		

		# HEADER=['id_pattern','attributes','pattern','support_size','support_size_ratio','quality']


		# to_write=[]
		# for id_pattern,(p,p_sup,p_bitset,qual) in enumerate(topk_patterns_refine_and_mine):
		# 	print attributes,p,len(set(p_bitset)),qual #p,p_sup,set(p_bitset)
		# 	to_write.append({
		# 		'id_pattern':id_pattern,
		# 		'attributes':attributes,
		# 		'pattern':p,
		# 		'support_size':len(set(p_bitset)),
		# 		'support_size_ratio':len(set(p_bitset))/float(len(dataset)),
		# 		'quality':qual,


		# 	})

		# writeCSVwithHeader(to_write,WRITE_RESULTS_FILE,selectedHeader=HEADER,flagWriteHeader=True)
		# # raw_input('.....')
		# # print "EXHAUSTIVE NOW"
		# # raw_input('.....')
		# # for p,p_sup,p_bitset,qual in topk_patterns_exhaustive:
		# # 	print attributes,p,len(set(p_bitset)),qual #p,p_sup,set(p_bitset)
		









	if args.STATS:
		header=['Dataset','num','rows','intervals','intervalsGood','class','prevalence']
		writeCSVwithHeader([],'STATS'+'.csv',selectedHeader=header,flagWriteHeader=True)
		for d in sorted(DATASETDICTIONNARY):
			print d
			all_datasets_statistics=[]
			DATASET_ENTITY_CONSIDERED=DATASETDICTIONNARY[d]
			data_file=path+'//'+DATASET_ENTITY_CONSIDERED['data_file']
			attributes=DATASET_ENTITY_CONSIDERED['attributes']
			attr_label=DATASET_ENTITY_CONSIDERED['attr_label']
			dataset,h=readCSVwithHeader(data_file,numberHeader=attributes,delimiter=',')
			for nb_attr in range(0,len(attributes)):
				for wanted_label in sorted({x[attr_label] for x in dataset}):
					
					_,_,_,_,statistics = transform_dataset(dataset,attributes[:nb_attr+1],attr_label,wanted_label,verbose=False)
					d_stat={
						'Dataset':d+'\_'+str(nb_attr+1).zfill(2)+'\_'+str(wanted_label),
						'num':nb_attr+1,
						'rows':statistics['rows'],
						'intervals':statistics['intervals'],
						'intervalsGood':statistics['intervalsGood'],
						'class':wanted_label,
						'prevalence':statistics['alpha']
					}
					all_datasets_statistics.append(d_stat)

					writeCSVwithHeader([d_stat],'STATS'+'.csv',selectedHeader=header,flagWriteHeader=False)
	if args.Q1:

		KEEP_PATTERNS_FOUND=False
		DATASET_TO_USE=args.dataset if args.dataset is not None else DATASET_TO_USE
		NB_ATTRIBUTES=args.nbattr if args.nbattr is not None else NB_ATTRIBUTES
		WANTED_LABEL=args.wanted_label if args.wanted_label is not None else WANTED_LABEL 
		COMPUTE_EXHAUSTIVE=COMPUTE_EXHAUSTIVE
		destinationCSVFile=DATASET_TO_USE+'_'+str(NB_ATTRIBUTES).zfill(2)+'_'+str(WANTED_LABEL)
		PLOTFIGURE=PLOTFIGURE
		THRESHOLD_QUAL=args.sigma_qual if args.sigma_qual is not None else THRESHOLD_QUAL
		THRESHOLD_SUP=args.sigma_sup if args.sigma_sup is not None else THRESHOLD_SUP
		THRESHOLD_SIM=THRESHOLD_SIM
		SELECTED_MEASURE=args.quality_measure if args.quality_measure is not None else SELECTED_MEASURE
		TOP_K=TOP_K
		CRISPINESS_COMPUTATION=args.compute_crispiness# if args.compute_crispiness is not None else CRISPINESS_COMPUTATION

		PLOTFIGURE=False
		print '--------------------------------------------Q1---------------------------------------------------'
		print 'Dataset : ',DATASET_TO_USE
		print 'NB Attributes : ',NB_ATTRIBUTES
		print 'Wanted Label : ',WANTED_LABEL
		print 'Selected Measure : ',SELECTED_MEASURE
		print 'Quality Threshold : ',THRESHOLD_QUAL
		print 'Support Threshold : ',THRESHOLD_SUP
		print '--------------------------------------------Q1---------------------------------------------------'

		if not os.path.exists('./tmp'):
			os.makedirs('./tmp')


		DATASET_ENTITY_CONSIDERED=DATASETDICTIONNARY[DATASET_TO_USE]
		data_file=path+'//'+DATASET_ENTITY_CONSIDERED['data_file']
		attributes=DATASET_ENTITY_CONSIDERED['attributes'][:NB_ATTRIBUTES]
		attr_label=DATASET_ENTITY_CONSIDERED['attr_label']
		wanted_label=DATASET_ENTITY_CONSIDERED['wanted_label'] if args.wanted_label is None else WANTED_LABEL
		SELECTED_MEASURE=MEASURES_DICTIONNARY[SELECTED_MEASURE]
		
		index_initialization=time()
		
		if True:
			transform_to_the_wanted_structure(data_file,selectedHeader=attributes+[attr_label],delimiter=',',nb_attributes=NB_ATTRIBUTES,class_label=attr_label,wanted_label=wanted_label)

		dataset,h=readCSVwithHeader(data_file,numberHeader=attributes,delimiter=',')
		dataset,positive_extent,negative_extent,alpha_ratio_class,statistics=transform_dataset(dataset,attributes,attr_label,wanted_label,verbose=False)






		index_attr,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset=compute_index_all_attributes(dataset,attributes,positive_extent=positive_extent,negative_extent=negative_extent)
		
		index_initialization=time()-index_initialization

		nb_pos=float(len(positive_extent))
		

		exhaustive_time=None;confirmation_time=None;best_pattern_found_time=None
		PROFILING=False
		
		Header=['method','quality','guarantee','nb_patterns','crispiness','timespent','timeInitialize']
		writeCSVwithHeader([],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=True)

		if PROFILING:
			pr = cProfile.Profile()
			pr.enable()
		########################################TESTING#############################################
		if COMPUTE_EXHAUSTIVE:
			try:
				
				with open ('.//tmp//'+destinationCSVFile+'.exhaustive', 'rb') as fp:
					print 'Loading Exhaustive File ....'
					t=time()
					STATS=pickle.load(fp)
					piece_of_data=STATS
					all_patterns_found_exhaustive=STATS['all_patterns_found_exhaustive']
					writeCSVwithHeader([piece_of_data],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=False)
					print 'Time Spent : ', time()-t
			except Exception as e:
				
				exhaustive_time=time()
				timeserie_qual,timeserie_guarantee,all_patterns_found_exhaustive= next(exhaustive_mine(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
															positive_extent,negative_extent,alpha_ratio_class,DISCRIMINATION_MEASURE=SELECTED_MEASURE,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,MAXIMUM_TIME=MAXIMUM_TIME))
				exhaustive_time=time()-exhaustive_time
				piece_of_data={'method':'CbO','quality':timeserie_qual[-1][1],'guarantee':timeserie_guarantee[-1][1],'timespent':timeserie_qual[-1][0],'timeInitialize':index_initialization,'nb_patterns':timeserie_qual[-1][2],'crispiness':0.}
				writeCSVwithHeader([piece_of_data],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=False)
				if SAVE_EXHAUSTIVE:
					STATS={
						'method':'CbO',
						'quality':piece_of_data['quality'],
						'guarantee':piece_of_data['guarantee'],
						'timespent':piece_of_data['timespent'],
						'timeInitialize':piece_of_data['timeInitialize'],
						'nb_patterns':piece_of_data['nb_patterns'],
						'all_patterns_found_exhaustive':all_patterns_found_exhaustive
					}
					with open('.//tmp//'+destinationCSVFile+'.exhaustive', 'wb') as fp:
						pickle.dump(STATS, fp)

			#raw_input('...')

		TIMESERIE_QUALITY=[]
		TIMESERIE_GUARANTEE=[]
		TIMESERIE_GLOBALITY=[]
		for yielded_pattern_found,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new  in discretize_and_mine_STRINGENT_2(dataset,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
												positive_extent,negative_extent,alpha_ratio_class,
												DISCRIMINATION_MEASURE=SELECTED_MEASURE,PROCEDURE_OF_SELECTION=select_next_cut_point_median_value_in_the_border,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,verbose=False,MAXIMUM_TIME=MAXIMUM_TIME,CRISPINESS_COMPUTATION=CRISPINESS_COMPUTATION):
			
			piece_of_data={'method':'RefineAndMine','quality':TIMESERIE_QUALITY[-1][1],'guarantee':TIMESERIE_GUARANTEE[-1][1],'timespent':TIMESERIE_QUALITY[-1][0],'timeInitialize':index_initialization,'nb_patterns':TIMESERIE_QUALITY[-1][2],'crispiness':TIMESERIE_GLOBALITY[-1][1]}
			writeCSVwithHeader([piece_of_data],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=False)
			continue
		timeserie_qual,timeserie_guarantee,timeserie_globality= TIMESERIE_QUALITY,TIMESERIE_GUARANTEE,TIMESERIE_GLOBALITY
		
		if PLOTFIGURE:
			from plotter.plotter import plot_timeseries
			best_pattern_quality=timeserie_qual[-1][1]
			
			for (t1,x,_),(t2,y,_) in zip(timeserie_qual,timeserie_guarantee):
				if x==best_pattern_quality:
					if best_pattern_found_time is None:
						best_pattern_found_time=t1
					
				if x==y:
					confirmation_time=t1
					break
			#raw_input('........')
			if CRISPINESS_COMPUTATION:
				timeseries={'Quality':([x[0] for x in timeserie_qual],[x[1] for x in timeserie_qual]),'Guarantees':([x[0] for x in timeserie_guarantee],[x[1] for x in timeserie_guarantee]),'ZCrispiness':([x[0] for x in timeserie_globality],[x[1] for x in timeserie_globality])}
			else:
				timeseries={'Quality':([x[0] for x in timeserie_qual],[x[1] for x in timeserie_qual]),'Guarantees':([x[0] for x in timeserie_guarantee],[x[1] for x in timeserie_guarantee])}#,'FScore':([x[0] for x in timeserie_guarantee],RESULTS_SIMILARITIES)}
			plot_timeseries(timeseries,destination_file=destinationCSVFile,exhaustive_time=exhaustive_time,confirmation_time=confirmation_time,best_pattern_found_time=best_pattern_found_time,show_plot=False)

		########################################TESTING#############################################
		if PROFILING:
			pr.disable()
			ps = pstats.Stats(pr)
			ps.sort_stats('cumulative').print_stats(20) #time
		######################################################################################################################
	if args.Q2:
		KEEP_PATTERNS_FOUND=True
		DATASET_TO_USE=args.dataset if args.dataset is not None else DATASET_TO_USE
		NB_ATTRIBUTES=args.nbattr if args.nbattr is not None else NB_ATTRIBUTES
		WANTED_LABEL=args.wanted_label if args.wanted_label is not None else WANTED_LABEL
		COMPUTE_EXHAUSTIVE=True
		
		PLOTFIGURE=PLOTFIGURE
		THRESHOLD_QUAL=args.sigma_qual if args.sigma_qual is not None else THRESHOLD_QUAL
		THRESHOLD_SUP=args.sigma_sup if args.sigma_sup is not None else THRESHOLD_SUP
		THRESHOLD_SIM=args.sigma_sim if args.sigma_sim is not None else THRESHOLD_SIM
		SELECTED_MEASURE=args.quality_measure if args.quality_measure is not None else SELECTED_MEASURE
		TOP_K=args.top_k if args.top_k is not None else TOP_K
		CRISPINESS_COMPUTATION=args.compute_crispiness
		
		SAVE_EXHAUSTIVE=args.save_exhaustive

		###############TODO##############
		DATASET_ENTITY_CONSIDERED=DATASETDICTIONNARY[DATASET_TO_USE]
		data_file=path+'//'+DATASET_ENTITY_CONSIDERED['data_file']
		attributes=DATASET_ENTITY_CONSIDERED['attributes'][:NB_ATTRIBUTES]
		attr_label=DATASET_ENTITY_CONSIDERED['attr_label']
		wanted_label=DATASET_ENTITY_CONSIDERED['wanted_label'] if args.wanted_label is None else WANTED_LABEL
		SELECTED_MEASURE=MEASURES_DICTIONNARY[SELECTED_MEASURE]
		
		index_initialization=time()
		dataset,h=readCSVwithHeader(data_file,numberHeader=attributes,delimiter=',')
		dataset,positive_extent,negative_extent,alpha_ratio_class,statistics=transform_dataset(dataset,attributes,attr_label,wanted_label,verbose=False)
		index_attr,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset=compute_index_all_attributes(dataset,attributes,positive_extent=positive_extent,negative_extent=negative_extent)
		index_initialization=time()-index_initialization

		nb_pos=float(len(positive_extent))
		if THRESHOLD_SUP<1:
			THRESHOLD_SUP=THRESHOLD_SUP*nb_pos

		

		print '--------------------------------------------Q2---------------------------------------------------'
		print 'Dataset : ',DATASET_TO_USE
		print 'NB Attributes : ',NB_ATTRIBUTES
		print 'Wanted Label : ',WANTED_LABEL
		print 'Selected Measure : ',SELECTED_MEASURE
		print 'Quality Threshold : ',THRESHOLD_QUAL
		print 'Support Threshold : ',THRESHOLD_SUP
		print 'Similarity Threshold : ',THRESHOLD_SIM
		print 'TOP - K : ',TOP_K
		print '--------------------------------------------Q2---------------------------------------------------'


		LIGHTER_PATTERN_SET=True

		destinationCSVFile=DATASET_TO_USE+'_'+str(NB_ATTRIBUTES)+'_'+str(WANTED_LABEL)+'_'+str(THRESHOLD_SUP)+'_'+str(THRESHOLD_QUAL)+'_'+str(TOP_K)+'_'+str(THRESHOLD_SIM).zfill(2)
		exhaustive_time=None;confirmation_time=None;best_pattern_found_time=None
		PROFILING=False


		Header=['method','quality','guarantee','nb_patterns','timespent','timeInitialize','precision','recall','fscore','globality','crispiness','size_pattern_set']
		writeCSVwithHeader([],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=True)

		if COMPUTE_EXHAUSTIVE:
			try:
				
				with open ('.//tmp//'+destinationCSVFile+'.exhaustive', 'rb') as fp:
					print 'Loading Exhaustive File ....'
					t=time()
					STATS=pickle.load(fp)
					piece_of_data=STATS
					all_patterns_found_exhaustive=STATS['all_patterns_found_exhaustive']
					writeCSVwithHeader([piece_of_data],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=False)
					print 'Time Spent : ', time()-t
			except Exception as e:
				
				exhaustive_time=time()
				timeserie_qual,timeserie_guarantee,all_patterns_found_exhaustive= next(exhaustive_mine(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
															positive_extent,negative_extent,alpha_ratio_class,DISCRIMINATION_MEASURE=SELECTED_MEASURE,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,MAXIMUM_TIME=MAXIMUM_TIME,LIGHTER_PATTERN_SET=LIGHTER_PATTERN_SET,similarity_threshold=THRESHOLD_SIM))
				if not LIGHTER_PATTERN_SET:
					all_patterns_found_exhaustive=[(a,c,c,d) for (a,b,c,d) in all_patterns_found_exhaustive]
					all_patterns_found_exhaustive=get_top_k_div_from_a_pattern_set(all_patterns_found_exhaustive,threshold_sim=THRESHOLD_SIM,k=TOP_K)
				else:
					all_patterns_found_exhaustive=get_top_k_div_from_a_pattern_set_new(all_patterns_found_exhaustive,threshold_sim=THRESHOLD_SIM,k=TOP_K)

				exhaustive_time=time()-exhaustive_time
				piece_of_data={
					'method':'CbO',
					'quality':timeserie_qual[-1][1],
					'guarantee':timeserie_guarantee[-1][1],
					'timespent':timeserie_qual[-1][0],
					'timeInitialize':index_initialization,
					'nb_patterns':timeserie_qual[-1][2],
					'precision':1.,
					'recall':1.,
					'fscore':1.,
					'globality':0.,
					'crispiness':0.,

					'size_pattern_set':len(all_patterns_found_exhaustive)
				}
				writeCSVwithHeader([piece_of_data],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=False)
				
				if SAVE_EXHAUSTIVE:
					STATS={
						'method':'CbO',
						'quality':piece_of_data['quality'],
						'guarantee':piece_of_data['guarantee'],
						'timespent':piece_of_data['timespent'],
						'timeInitialize':piece_of_data['timeInitialize'],
						'nb_patterns':piece_of_data['nb_patterns'],
						'nb_patterns':timeserie_qual[-1][2],
						'quality_pattern_set':1.,
						'size_pattern_set':len(all_patterns_found_exhaustive),
						'all_patterns_found_exhaustive':all_patterns_found_exhaustive,
						'all_patterns_found_exhaustive_clear':[(set(x[0]),x[1]) for x in all_patterns_found_exhaustive]
					}
					with open('.//tmp//'+destinationCSVFile+'.exhaustive', 'wb') as fp:
						pickle.dump(STATS, fp)

					
					writeCSVwithHeader([{'s':stringifier(sorted(set(x[0])))} for x in all_patterns_found_exhaustive],'.//tmp//'+destinationCSVFile+'.csv',flagWriteHeader=False,ERASE_EXISTING_FILE=True)


		STARTING=time()
			
		OPTIMIZATION=True
		if not LIGHTER_PATTERN_SET:
			exhaustive_sets=[x[1] for x in all_patterns_found_exhaustive]
		else:
			exhaustive_sets=[x[0] for x in all_patterns_found_exhaustive]

		if OPTIMIZATION:
			exhaustive_dictionnary_jaccard_max={k:0. for k in range(len(exhaustive_sets))}
			exhaustive_dictionnary_symmetric_difference_min={k:1. for k in range(len(exhaustive_sets))}


		for yielded_pattern_found,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new  in discretize_and_mine_STRINGENT_2(dataset,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
												positive_extent,negative_extent,alpha_ratio_class,
												DISCRIMINATION_MEASURE=SELECTED_MEASURE,PROCEDURE_OF_SELECTION=select_next_cut_point_median_value_in_the_border,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,verbose=False,MAXIMUM_TIME=MAXIMUM_TIME,CRISPINESS_COMPUTATION=CRISPINESS_COMPUTATION,CLEAN_PATTERNS_FOUND_LIST=True):
			yielded_pattern_found=[(a,c,c,d) for (a,b,c,d) in yielded_pattern_found]
			#yielded_pattern_found=get_top_k_div_from_a_pattern_set(yielded_pattern_found,threshold_sim=THRESHOLD_SIM,k=TOP_K)
			
			observed_sets=[x[1] for x in yielded_pattern_found]
			



			###TODO#############
			fscore,precision,recall=similarity_between_patterns_set_new(exhaustive_sets,observed_sets,dictionnary_infos=exhaustive_dictionnary_jaccard_max)
			globality=similarity_between_patterns_set_globaility_new(exhaustive_sets,observed_sets,dataset,dictionnary_infos=exhaustive_dictionnary_symmetric_difference_min)
			piece_of_data={
				'method':'RefineAndMine',
				'quality':TIMESERIE_QUALITY[-1][1],
				'guarantee':TIMESERIE_GUARANTEE[-1][1],
				'timespent':TIMESERIE_QUALITY[-1][0],
				'timeInitialize':index_initialization,
				'nb_patterns':TIMESERIE_QUALITY[-1][2],
				'precision':precision,
				'recall':recall,
				'fscore':fscore,
				'globality':globality,
				'crispiness':TIMESERIE_GLOBALITY[-1][1],
				'size_pattern_set':len(yielded_pattern_found)
			}
			writeCSVwithHeader([piece_of_data],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=False)
			continue

		print 'Computation Time Full : ',time()-STARTING

	if args.Q3:

		
		EVERY_NB_ITERATION_KEEP_TRACK=1

		KEEP_PATTERNS_FOUND=True
		DATASET_TO_USE=args.dataset if args.dataset is not None else DATASET_TO_USE
		NB_ATTRIBUTES=args.nbattr if args.nbattr is not None else NB_ATTRIBUTES
		WANTED_LABEL=args.wanted_label if args.wanted_label is not None else WANTED_LABEL
		COMPUTE_EXHAUSTIVE=True
		
		PLOTFIGURE=PLOTFIGURE
		THRESHOLD_QUAL=args.sigma_qual if args.sigma_qual is not None else THRESHOLD_QUAL
		THRESHOLD_SUP=args.sigma_sup if args.sigma_sup is not None else THRESHOLD_SUP
		THRESHOLD_SIM=args.sigma_sim if args.sigma_sim is not None else THRESHOLD_SIM
		SELECTED_MEASURE=args.quality_measure if args.quality_measure is not None else SELECTED_MEASURE
		TOP_K=args.top_k if args.top_k is not None else TOP_K
		CRISPINESS_COMPUTATION=args.compute_crispiness
		
		SAVE_EXHAUSTIVE=args.save_exhaustive

		if not os.path.exists('./tmp'):
			os.makedirs('./tmp')


		###############TODO##############
		DATASET_ENTITY_CONSIDERED=DATASETDICTIONNARY[DATASET_TO_USE]
		data_file=path+'//'+DATASET_ENTITY_CONSIDERED['data_file']
		attributes=DATASET_ENTITY_CONSIDERED['attributes'][:NB_ATTRIBUTES]
		attr_label=DATASET_ENTITY_CONSIDERED['attr_label']
		wanted_label=DATASET_ENTITY_CONSIDERED['wanted_label'] if args.wanted_label is None else WANTED_LABEL
		SELECTED_MEASURE=MEASURES_DICTIONNARY[SELECTED_MEASURE]
		
		index_initialization=time()
		dataset,h=readCSVwithHeader(data_file,numberHeader=attributes,delimiter=',')
		dataset,positive_extent,negative_extent,alpha_ratio_class,statistics=transform_dataset(dataset,attributes,attr_label,wanted_label,verbose=False)
		index_attr,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset=compute_index_all_attributes(dataset,attributes,positive_extent=positive_extent,negative_extent=negative_extent)
		index_initialization=time()-index_initialization


		
		nb_pos=float(len(positive_extent))
		if THRESHOLD_SUP<1:
			THRESHOLD_SUP=THRESHOLD_SUP*nb_pos

		

		print '--------------------------------------------Q3---------------------------------------------------'
		print 'Dataset : ',DATASET_TO_USE
		print 'NB Attributes : ',NB_ATTRIBUTES
		print 'Wanted Label : ',WANTED_LABEL
		print 'Selected Measure : ',SELECTED_MEASURE
		print 'Quality Threshold : ',THRESHOLD_QUAL
		print 'Support Threshold : ',THRESHOLD_SUP
		print 'Similarity Threshold : ',THRESHOLD_SIM
		print 'TOP - K : ',TOP_K
		print '--------------------------------------------Q3---------------------------------------------------'


		LIGHTER_PATTERN_SET=True
		LIGHER_PATTERN_SET_FOR_DISCERETIZE=True

		destinationCSVFile=DATASET_TO_USE+'_'+str(NB_ATTRIBUTES)+'_'+str(WANTED_LABEL)#+'_'+str(THRESHOLD_SUP)+'_'+str(THRESHOLD_QUAL)+'_'+str(TOP_K)+'_'+str(THRESHOLD_SIM).zfill(2)
		exhaustive_time=None;confirmation_time=None;best_pattern_found_time=None
		PROFILING=False


		Header=['method','quality','guarantee','nb_patterns','timespent_full','timeInitialize','timespent','timePostProcessing','recall','size_pattern_set']
		writeCSVwithHeader([],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=True)

		if COMPUTE_EXHAUSTIVE:
			try:
				
				with open ('.//tmp//'+destinationCSVFile+'.exhaustive', 'rb') as fp:
					print 'Loading Exhaustive File ....'
					t=time()
					STATS=pickle.load(fp)
					piece_of_data=STATS
					all_patterns_found_exhaustive=STATS['all_patterns_found_exhaustive']
					writeCSVwithHeader([piece_of_data],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=False)
					print 'Loading Done, Time Spent : ', time()-t
			except Exception as e:
				
				exhaustive_time=time()
				timeserie_qual,timeserie_guarantee,all_patterns_found_exhaustive= next(exhaustive_mine(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
															positive_extent,negative_extent,alpha_ratio_class,DISCRIMINATION_MEASURE=SELECTED_MEASURE,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,MAXIMUM_TIME=MAXIMUM_TIME,LIGHTER_PATTERN_SET=LIGHTER_PATTERN_SET,similarity_threshold=THRESHOLD_SIM))
				timing_top_k_div=time()
				if not LIGHTER_PATTERN_SET:
					all_patterns_found_exhaustive=[(a,c,c,d) for (a,b,c,d) in all_patterns_found_exhaustive]
					all_patterns_found_exhaustive=get_top_k_div_from_a_pattern_set(all_patterns_found_exhaustive,threshold_sim=THRESHOLD_SIM,k=TOP_K)
				else:
					all_patterns_found_exhaustive=get_top_k_div_from_a_pattern_set_new(all_patterns_found_exhaustive,threshold_sim=THRESHOLD_SIM,k=TOP_K)
				timing_top_k_div=time()-timing_top_k_div
				exhaustive_time=time()-exhaustive_time
				piece_of_data={
					'method':'CbO',
					'quality':timeserie_qual[-1][1],
					'guarantee':timeserie_guarantee[-1][1],
					'timespent':timeserie_qual[-1][0],
					'timeInitialize':index_initialization,
					'timePostProcessing':timing_top_k_div,
					'timespent_full':index_initialization+timeserie_qual[-1][0]+timing_top_k_div,
					'nb_patterns':timeserie_qual[-1][2],
					'precision':1.,
					'recall':1.,
					'fscore':1.,
					'globality':0.,
					'crispiness':0.,

					'size_pattern_set':len(all_patterns_found_exhaustive)
				}
				writeCSVwithHeader([piece_of_data],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=False)
				
				if SAVE_EXHAUSTIVE:
					STATS={
						'method':'CbO',
						'quality':piece_of_data['quality'],
						'guarantee':piece_of_data['guarantee'],
						'timespent':piece_of_data['timespent'],
						'timeInitialize':piece_of_data['timeInitialize'],
						'timePostProcessing':timing_top_k_div,
						'timespent_full':index_initialization+timeserie_qual[-1][0]+timing_top_k_div,
						'nb_patterns':piece_of_data['nb_patterns'],
						'nb_patterns':timeserie_qual[-1][2],
						'quality_pattern_set':1.,
						'recall':1.,
						'size_pattern_set':len(all_patterns_found_exhaustive),
						'all_patterns_found_exhaustive':all_patterns_found_exhaustive,
						'all_patterns_found_exhaustive_clear':[(set(x[0]),x[1]) for x in all_patterns_found_exhaustive]
					}
					with open('.//tmp//'+destinationCSVFile+'.exhaustive', 'wb') as fp:
						pickle.dump(STATS, fp)

					
					writeCSVwithHeader([{'s':stringifier(sorted(set(x[0])))} for x in all_patterns_found_exhaustive],'.//tmp//'+destinationCSVFile+'.csv',flagWriteHeader=False,ERASE_EXISTING_FILE=True)


		STARTING=time()
			
		OPTIMIZATION=True
		if not LIGHER_PATTERN_SET_FOR_DISCERETIZE:
			exhaustive_sets=[x[1] for x in all_patterns_found_exhaustive]
		else:
			exhaustive_sets=[x[0] for x in all_patterns_found_exhaustive]

		if OPTIMIZATION:
			exhaustive_dictionnary_jaccard_max={k:0. for k in range(len(exhaustive_sets))}
			exhaustive_dictionnary_symmetric_difference_min={k:1. for k in range(len(exhaustive_sets))}


		step=0

		FULL_NB_ITERATION=sum(len(y) for x,y in partitions_to_dataset.iteritems())


		for yielded_pattern_found,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new  in discretize_and_mine_STRINGENT_2(dataset,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
												positive_extent,negative_extent,alpha_ratio_class,
												DISCRIMINATION_MEASURE=SELECTED_MEASURE,PROCEDURE_OF_SELECTION=select_next_cut_point_median_value_in_the_border,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=KEEP_PATTERNS_FOUND,verbose=False,MAXIMUM_TIME=MAXIMUM_TIME,CRISPINESS_COMPUTATION=CRISPINESS_COMPUTATION,CLEAN_PATTERNS_FOUND_LIST=False,LIGHTER_PATTERN_SET=LIGHER_PATTERN_SET_FOR_DISCERETIZE):
			
			if step==0:
				FULL_NB_ITERATION=FULL_NB_ITERATION-sum(len(y) for x,y in partitions_to_dataset_new.iteritems())	

			if step%EVERY_NB_ITERATION_KEEP_TRACK==0 or step==FULL_NB_ITERATION:
				

				if not LIGHER_PATTERN_SET_FOR_DISCERETIZE:
					yielded_pattern_found=[(a,c,c,d) for (a,b,c,d) in yielded_pattern_found]
					timing_top_k_div=time()
					yielded_pattern_found=get_top_k_div_from_a_pattern_set(yielded_pattern_found,threshold_sim=THRESHOLD_SIM,k=TOP_K)
					timing_top_k_div=time()-timing_top_k_div
					observed_sets=[x[1] for x in yielded_pattern_found]
				else:
					yielded_pattern_found=get_top_k_div_from_a_pattern_set_new(yielded_pattern_found,threshold_sim=THRESHOLD_SIM,k=TOP_K)
					observed_sets=[x[0] for x in yielded_pattern_found]
				



				###TODO#############
				fscore,precision,recall=similarity_between_patterns_set_new(exhaustive_sets,observed_sets,dictionnary_infos=exhaustive_dictionnary_jaccard_max)

				#globality=similarity_between_patterns_set_globaility_new(exhaustive_sets,observed_sets,dataset,dictionnary_infos=exhaustive_dictionnary_symmetric_difference_min)
				piece_of_data={
					'method':'RefineAndMine',
					'quality':TIMESERIE_QUALITY[-1][1],
					'guarantee':TIMESERIE_GUARANTEE[-1][1],
					'timespent':TIMESERIE_QUALITY[-1][0],
					'timeInitialize':index_initialization,
					'timePostProcessing':timing_top_k_div,
					'timespent_full':index_initialization+TIMESERIE_QUALITY[-1][0]+timing_top_k_div,
					'nb_patterns':TIMESERIE_QUALITY[-1][2],
					'precision':precision,
					'recall':recall,
					'fscore':fscore,
					'crispiness':TIMESERIE_GLOBALITY[-1][1],
					'size_pattern_set':len(yielded_pattern_found)
				}
				writeCSVwithHeader([piece_of_data],destinationCSVFile+'.csv',selectedHeader=Header,flagWriteHeader=False)

			step+=1
			continue

		print 'Computation Time Full : ',time()-STARTING

	if args.plot is not None:
		csv_file=args.plot
		print csv_file
		x_axis=args.x_axis_attribute
		methods=args.methods
		first_line_exhaustive=args.exhaustive_info
		dataset,h=readCSVwithHeader(csv_file,delimiter='\t')
		
		first_line_to_consider=1 if first_line_exhaustive else 0
		timeseries={}

		if False:
			for y_axis in methods:
				timeserie=([],[])
				for i in range(first_line_to_consider,len(dataset)):
					row=dataset[i]

					timeserie[0].append((float(row[x_axis])))
					timeserie[1].append((float(row[y_axis])))
				timeseries[y_axis]=timeserie
		else:
			
			
			for y_axis in methods:
				Hajimeta=False
				timeserie=([],[])
				considered_method=None
				for i in range(first_line_to_consider,len(dataset)):
					row=dataset[i]
					now_considered_method=row['method']

					if considered_method is None or considered_method==now_considered_method:

						timeserie[0].append((float(row[x_axis])))
						timeserie[1].append((float(row[y_axis])))
					else:
						Hajimeta=True
						#print now_considered_method
						timeseries[y_axis+'_'+considered_method]=timeserie
						timeserie=([],[])
						timeserie[0].append((float(row[x_axis])))
						timeserie[1].append((float(row[y_axis])))
					considered_method=now_considered_method

				#if Hajimeta:
				timeseries[y_axis+'_'+row['method']]=timeserie
			


		exhaustive_time=None
		best_pattern_found_time=None
		confirmation_time=None
		if first_line_exhaustive:
			exhaustive_time=float(dataset[0][x_axis])

		if methods[:2]==['quality','guarantee'] or methods==['quality','guarantee','crispiness']:
			QUAL='quality' if 'quality' in timeseries else 'quality_RefineAndMine' 
			GUAR='guarantee' if 'guarantee'  in timeseries else 'guarantee_RefineAndMine' 
			best_pattern_quality=timeseries[QUAL][1][-1]
			timeserie_time=timeseries[QUAL][0]
			timeserie_quality=timeseries[QUAL][1]
			timeserie_guarantee=timeseries[GUAR][1]
			for t1,x,y in zip(timeserie_time,timeserie_quality,timeserie_guarantee):
				if x==best_pattern_quality:
					if best_pattern_found_time is None:
						best_pattern_found_time=t1
					
				if x==y:
					confirmation_time=t1
					break


		from plotter.plotter import plot_timeseries
		filename, file_extension = os.path.splitext(csv_file)
		plot_timeseries(timeseries,destination_file=filename,confirmation_time=confirmation_time,best_pattern_found_time=best_pattern_found_time,exhaustive_time=exhaustive_time,show_plot=False)#,exhaustive_time=exhaustive_time,confirmation_time=confirmation_time,best_pattern_found_time=best_pattern_found_time,show_plot=False)
		# for row in dataset:
		# 	print row
	########################################PLOT AND DIVERSE#############################################
	if args.VIS:
		# from plotter.plotter import pdf_cat
		# pdf_cat(['./tmp/F02.pdf','./tmp/F27.pdf'], './tmp/MERGED.pdf')
		
		# raw_input('....')

		KEEP_PATTERNS_FOUND=False
		DATASET_TO_USE=args.dataset if args.dataset is not None else DATASET_TO_USE
		NB_ATTRIBUTES=args.nbattr if args.nbattr is not None else NB_ATTRIBUTES
		WANTED_LABEL=args.wanted_label if args.wanted_label is not None else WANTED_LABEL 
		COMPUTE_EXHAUSTIVE=COMPUTE_EXHAUSTIVE
		destinationCSVFile=DATASET_TO_USE+'_'+str(NB_ATTRIBUTES).zfill(2)+'_'+str(WANTED_LABEL)
		PLOTFIGURE=PLOTFIGURE
		THRESHOLD_QUAL=args.sigma_qual if args.sigma_qual is not None else THRESHOLD_QUAL
		THRESHOLD_SUP=args.sigma_sup if args.sigma_sup is not None else THRESHOLD_SUP
		THRESHOLD_SIM=args.sigma_sim if args.sigma_sim is not None else THRESHOLD_SIM
		SELECTED_MEASURE=args.quality_measure if args.quality_measure is not None else SELECTED_MEASURE
		TOP_K=args.top_k if args.top_k is not None else TOP_K
		CRISPINESS_COMPUTATION=args.compute_crispiness# if args.compute_crispiness is not None else CRISPINESS_COMPUTATION
		LIGHTER_PATTERN_SET=False
		LIGHTER_PATTERN_SET_DISCRETE=False
		PLOTFIGURE=False
		print '--------------------------------------------VIS---------------------------------------------------'
		print 'Dataset : ',DATASET_TO_USE
		print 'NB Attributes : ',NB_ATTRIBUTES
		print 'Wanted Label : ',WANTED_LABEL
		print 'Selected Measure : ',SELECTED_MEASURE
		print 'Quality Threshold : ',THRESHOLD_QUAL
		print 'Support Threshold : ',THRESHOLD_SUP
		print '--------------------------------------------VIS---------------------------------------------------'

		if not os.path.exists('./tmp'):
			os.makedirs('./tmp')


		DATASET_ENTITY_CONSIDERED=DATASETDICTIONNARY[DATASET_TO_USE]
		data_file=path+'//'+DATASET_ENTITY_CONSIDERED['data_file']
		attributes=DATASET_ENTITY_CONSIDERED['attributes'][:NB_ATTRIBUTES]
		attr_label=DATASET_ENTITY_CONSIDERED['attr_label']
		wanted_label=DATASET_ENTITY_CONSIDERED['wanted_label'] if args.wanted_label is None else WANTED_LABEL
		SELECTED_MEASURE=MEASURES_DICTIONNARY[SELECTED_MEASURE]
		
		index_initialization=time()
		
		
		dataset,h=readCSVwithHeader(data_file,numberHeader=attributes,delimiter=',')
		dataset,positive_extent,negative_extent,alpha_ratio_class,statistics=transform_dataset(dataset,attributes,attr_label,wanted_label,verbose=False)
		index_attr,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset=compute_index_all_attributes(dataset,attributes,positive_extent=positive_extent,negative_extent=negative_extent)
		


		t_spent=time()
		keep_patterns_found=True
		timeserie_qual,timeserie_guarantee,all_patterns_found_exhaustive= next(exhaustive_mine(dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
												positive_extent,negative_extent,alpha_ratio_class,DISCRIMINATION_MEASURE=SELECTED_MEASURE,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=keep_patterns_found,LIGHTER_PATTERN_SET=LIGHTER_PATTERN_SET))

		timespent_by_exhaustive=time()-t_spent
		# print timespent_by_exhaustive
		# raw_input('....')

		if not LIGHTER_PATTERN_SET:
			all_patterns_found_exhaustive=[(a,c,c,d) for (a,b,c,d) in all_patterns_found_exhaustive]
			all_supports_exhaustive_cleaned=get_top_k_div_from_a_pattern_set(all_patterns_found_exhaustive,threshold_sim=THRESHOLD_SIM,k=TOP_K)
		else:
			all_supports_exhaustive_cleaned=get_top_k_div_from_a_pattern_set_new(all_patterns_found_exhaustive,threshold_sim=THRESHOLD_SIM,k=TOP_K)
		
		print 'EXHAUSTIVE TOPK SIZE : ', len(all_supports_exhaustive_cleaned)

		# all_patterns_found_exhaustive=[(a,c,c,d) for (a,b,c,d) in all_patterns_found_exhaustive]
		# all_supports_exhaustive_cleaned=get_top_k_div_from_a_pattern_set(all_patterns_found_exhaustive,threshold_sim=THRESHOLD_SIM,k=TOP_K)
		TIMESERIE_GLOBAILITY_EMPIRIC=[] 
		TIMESERIE_GLOBAILITY_RECALL=[]
		TIMESERIE_TIME_XX=[0]
		exhaustive_sets=[x[1] for x in all_supports_exhaustive_cleaned]
		def discrete_process(compap=False):
			
			for yielded_pattern_found,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new  in discretize_and_mine_STRINGENT_2(dataset,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
												positive_extent,negative_extent,alpha_ratio_class,
												PROCEDURE_OF_SELECTION=select_next_cut_point_median_value_in_the_border,DISCRIMINATION_MEASURE=SELECTED_MEASURE,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=keep_patterns_found,LIGHTER_PATTERN_SET=LIGHTER_PATTERN_SET_DISCRETE,CRISPINESS_COMPUTATION=CRISPINESS_COMPUTATION,verbose=False):
				
				if compap:
					continue
				if not LIGHTER_PATTERN_SET_DISCRETE:
					yielded_pattern_found=[(a,c,c,d) for (a,b,c,d) in yielded_pattern_found]
					all_patterns_found_cleaned=get_top_k_div_from_a_pattern_set(yielded_pattern_found,threshold_sim=THRESHOLD_SIM,k=TOP_K)
				else:
					all_patterns_found_cleaned=get_top_k_div_from_a_pattern_set_new(yielded_pattern_found,threshold_sim=THRESHOLD_SIM,k=TOP_K)
				
				observed_sets=[x[1] for x in yielded_pattern_found]
				TIMESERIE_GLOBAILITY_EMPIRIC.append(similarity_between_patterns_set_globaility(exhaustive_sets,observed_sets,dataset,dictionnary_infos={}))
				fscore,precision,recall=similarity_between_patterns_set(exhaustive_sets,observed_sets,dictionnary_infos={})
				TIMESERIE_GLOBAILITY_RECALL.append(recall)
				

				cutty_points=[sorted([partitions_to_dataset_new[a][p]['val_min'] for p in sorted(partitions_to_dataset_new[a])]+[partitions_to_dataset_new[a][p]['val_max'] for p in sorted(partitions_to_dataset_new[a])]) for a in attributes ]
				timeserie_qual,timeserie_guarantee,timeserie_crispiness= TIMESERIE_QUALITY,TIMESERIE_GUARANTEE,TIMESERIE_GLOBALITY
				TIMESERIE_TIME_XX.append(timeserie_qual[-1][0])
				#timeseries={'quality':([x[1] for x in timeserie_qual]),'guarantee':([x[1] for x in timeserie_guarantee]),'crispiness':([x[1] for x in timeserie_crispiness]),'globality':TIMESERIE_GLOBAILITY_EMPIRIC,'recall':TIMESERIE_GLOBAILITY_RECALL}
				timeseries={'quality':([x[0] for x in timeserie_qual],[x[1] for x in timeserie_qual]),'guarantee':([x[0] for x in timeserie_qual],[x[1] for x in timeserie_guarantee]),'crispiness':([x[0] for x in timeserie_qual],[x[1] for x in timeserie_crispiness]),'globality':([x[0] for x in timeserie_qual],TIMESERIE_GLOBAILITY_EMPIRIC),'recall':([x[0] for x in timeserie_qual],TIMESERIE_GLOBAILITY_RECALL)}
				
				yield [x[0] for x in all_patterns_found_cleaned],cutty_points,timeseries,TIMESERIE_TIME_XX
			if compap:
				yield TIMESERIE_QUALITY[-1][0]
		
		
		time_refine_and_mine=timespent_by_exhaustive
		

		for yielded_pattern_found,dataset_to_partitions_new,partitions_to_dataset_new,partitions_to_dataset_bitset_new  in discretize_and_mine_STRINGENT_2(dataset,dataset_to_partitions,partitions_to_dataset,partitions_to_dataset_bitset,attributes,index_attr, \
												positive_extent,negative_extent,alpha_ratio_class,
												PROCEDURE_OF_SELECTION=select_next_cut_point_median_value_in_the_border,DISCRIMINATION_MEASURE=SELECTED_MEASURE,threshold_sup=THRESHOLD_SUP,threshold_qual=THRESHOLD_QUAL,keep_patterns_found=keep_patterns_found,LIGHTER_PATTERN_SET=LIGHTER_PATTERN_SET_DISCRETE,CRISPINESS_COMPUTATION=CRISPINESS_COMPUTATION,verbose=False):
			continue
		time_refine_and_mine=TIMESERIE_QUALITY[-1][0]/3.5
		print time_refine_and_mine
		TIMESERIE_QUALITY,TIMESERIE_GUARANTEE,TIMESERIE_GLOBALITY=[],[],[]
		# for x in discrete_process(True):
		
		#raw_input('....')

		from plotter.plotter import plot_patterns
		print destinationCSVFile
		plot_patterns(dataset,attributes,positive_extent,negative_extent,[x[0] for x in all_supports_exhaustive_cleaned],discrete_process(),title_for_figure=destinationCSVFile,timespent_by_exhaustive=timespent_by_exhaustive)
	########################################PLOT AND DIVERSE#############################################