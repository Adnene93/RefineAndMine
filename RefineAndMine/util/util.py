import csv
import os

def writeCSVwithHeader(data,destination,selectedHeader=None,delimiter='\t',flagWriteHeader=True,ERASE_EXISTING_FILE=False):
	header=selectedHeader if selectedHeader is not None else data[0].keys()
	
	if flagWriteHeader or ERASE_EXISTING_FILE : 
		with open(destination, 'w') as f:
			f.close()
	with open(destination, 'ab+') as f:
		writer2 = csv.writer(f,delimiter=delimiter)
		if flagWriteHeader:
			writer2.writerow(header)
		for elem in iter(data):
			row=[]
			for i in range(len(header)):
				row.append(elem[header[i]])
			writer2.writerow(row)

def readCSVwithHeader(source,selectedHeader=None,numberHeader=None,arrayHeader=None,booleanHeader=None,delimiter='\t'):
	results=[]
	header=[]
	count=0
	
	with open(source, 'rb') as csvfile:
		
		readfile = csv.reader(csvfile, delimiter=delimiter)
		
		header=next(readfile)
		
		
		#selectedHeader=selectedHeader if selectedHeader is not None else header
		
		range_header=range(len(header))
		
		if numberHeader is None and arrayHeader is None and booleanHeader is None  :
			if selectedHeader is None:
				results=[{header[i]:row[i] for i in range_header} for row in readfile]
			else :
				results=[{header[i]:row[i] for i in range_header if header[i] in selectedHeader} for row in readfile]
		else :
			numberHeader=numberHeader if numberHeader is not None else []
			arrayHeader=arrayHeader if arrayHeader is not None else []
			booleanHeader=booleanHeader if booleanHeader is not None else []
			if selectedHeader is None:
				results_append=results.append
				for row in readfile:
					elem={}
					skip=False
					for i in range_header:
						if header[i] in numberHeader:
							try :
								elem[header[i]]=float(row[i])
							except:
								skip=True
						elif header[i] in arrayHeader:
							elem[header[i]]=eval(row[i])
						elif header[i] in booleanHeader:
							elem[header[i]]=bool(int(row[i]))
						else :
							elem[header[i]]=row[i]
					if not skip:
						results_append(elem)
					
				#results=[{header[i]:float(row[i]) if header[i] in numberHeader else eval(row[i]) if header[i] in arrayHeader else row[i] for i in range_header} for row in readfile ] 
			else :
				results=[{header[i]:float(row[i]) if header[i] in numberHeader else eval(row[i]) if header[i] in arrayHeader else row[i] for i in range_header if header[i] in selectedHeader} for row in readfile]


	return results,header



def transform_to_the_wanted_structure(source,selectedHeader=None,numberHeader=None,arrayHeader=None,booleanHeader=None,delimiter=',',nb_attributes=2,class_label='',wanted_label=''):
	d,h=readCSVwithHeader(source,selectedHeader,numberHeader,arrayHeader,booleanHeader,delimiter)

	new_header=selectedHeader[:nb_attributes]
	filename, file_extension = os.path.splitext(source)
	writeCSVwithHeader(d,filename+'_'+str(nb_attributes)+'_'+str(wanted_label)+'_properties'+file_extension,new_header)
	dclass=[]
	for row in d:
		dclass.append({wanted_label:int(row[class_label]==wanted_label)})

	writeCSVwithHeader(dclass,filename+'_'+str(nb_attributes)+'_'+str(wanted_label)+'_qualities'+file_extension,[wanted_label])
