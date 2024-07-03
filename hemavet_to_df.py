# import the necessary packages
from google.oauth2 import service_account
from google.cloud import vision
import argparse
import cv2
import io
import numpy as np
import re
import os
import sys
import pdb;

"""
Please note:
Coordinates have teh top left corder as the origin. Numbers get higher as you go down and move to the right
First coordinate is the horizontal axis
Second coordinate is the vertical axis
Thus, the order in which coordiantes are provided is left top, right top, right bottom, left bottom
"""
def extract_text_annotations(response):
    annotations = response.text_annotations
    text_boxes = []
    for annotation in annotations[1:]:  # Skip the first annotation which is the full text
        vertices = annotation.bounding_poly.vertices
        text = annotation.description
        text_boxes.append({
            'text': text,
            'vertices': vertices
        })
    return text_boxes

def calculate_center_y(vertices):
    ys = [vertex.y for vertex in vertices]
    return np.mean(ys)

def calculate_center_x(vertices):
    xs = [vertex.x for vertex in vertices]
    return np.mean(xs)

#This was done in terms of x before
def group_by_rows(text_boxes, x_threshold=15):
    # Sort boxes by their vertical center
    text_boxes.sort(key=lambda box: calculate_center_x(box['vertices']))
    
    rows = []
    current_row = [text_boxes[0]]
    
    for box in text_boxes[1:]:
        if calculate_center_x(box['vertices']) - calculate_center_x(current_row[-1]['vertices']) < x_threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    
    if current_row:
        rows.append(current_row)
    
    return rows

def group_by_cols(text_boxes, y_threshold=15):
    # Sort boxes by their vertical center
    text_boxes.sort(key=lambda box: calculate_center_y(box['vertices']))
    
    rows = []
    current_row = [text_boxes[0]]
    
    for box in text_boxes[1:]:
        if calculate_center_y(box['vertices']) - calculate_center_y(current_row[-1]['vertices']) < y_threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    
    if current_row:
        rows.append(current_row)
    
    return rows

def sort_rows_by_x(rows):
    sorted_rows = []
    for row in rows:
        sorted_row = sorted(row, key=lambda box: box['vertices'][0].x)
        sorted_rows.append(sorted_row)
    return sorted_rows


def sort_cols_by_y(sorted_rows):
	sorted_cols = []
	for row in sorted_rows:
		row.sort(key=lambda box: calculate_center_y(box['vertices']),reverse=True)
		sorted_cols.append(row)
	return sorted_cols

def print_table(sorted_rows):
    for row in sorted_rows:
        print(" | ".join(box['text'] for box in row))

		   
#Specify a set of lines, a regex to extract per line and any prefixes that may go with it.
def segment_extract(data_range,regex_query,prefixes):
    ret_dict= dict()
    for leuk in prefixes:
        for line in data_range:
            #if leuk == "Hb" and line[0]=="H":
            #    pdb.set_trace()
            #    print("We are here")
            #    print(line)
            #    input()
            curr_query = leuk+regex_query
            result = re.search(curr_query, line)
            if (result is not None) and len(result.groups()) > 0:
                ret_dict[leuk] = result.group(1)
    return(ret_dict)


def header_extract(data_line,lookup_word, regex_query):
    if lookup_word in data_line:
          result = re.search(regex_query, data_line)
          if (result is not None) and len(result.groups()) > 0:
                #print(result)
                #input()
                return(result)
    return(None)      
        
  

def str_puller(str_line, id):
	cell_len = id.count(" ") + 1
	#easy case, nothing fancy here
	if cell_len == 1:
		if id not in str_line:
			return("")
		i=str_line.index(id)
		if len(str_line) == i + 1:
			print("We are at the end")
			print(str_line)
			input()
		q_i=i+2
		if str_line[i+1] != ":" : #Special case with a header
			print("Don't want this case, may be an error")
			print(str_line)
		if str_line[q_i] == "Species":
			#print(str_line)
			print("Unable to identify written sample number, sorry")
			return("")
		else:
		    return(str_line[q_i])
	if cell_len > 1 :
		#We are looking at a multiplle matches
		full_query=id.split(" ")
		if full_query[0] not in str_line:
			return("")
		i=str_line.index(full_query[0])
		present=True
		for q in full_query:
			if q == str_line[i]:
				i = i+1
			else:
				print("Not the query searching for. Keep looking")
				present=False
				return("")
		if present:
			if str_line[i] != ":" : #Special case with a header
				print("Don't want this case, may be an error")
				print(str_line)
				return("")
			return(str_line[i+1])	
#Get the header data...
def extract_header(all_data, header_query,lookup_index=0):
    first=header_query[lookup_index]
    ret_dat=""
    for i,l in enumerate(all_data):
            if first + "\t:" in l:
                ret_dat = dict(zip(header_query,[str_puller(l.split(), x) for x in header_query]))
                if first == "Sample" and ret_dat[first] == "":
                    #pdb.set_trace()
                    if all_data[i-1].split()[0].isnumeric():
                        ret_dat[first]=all_data[i-1].split()[0]
                break 
    return(ret_dat)


def extract_header_regex(all_data, header_match,regex_query):
    for i,l in enumerate(all_data):
            if header_match in l:
                print("Regex version match....")
                return(header_extract(l,header_match, regex_query))
    return("")



#helper function for calibration
def maxCoordDiff(vertex_list,i):
	max_dist=0
	for v in vertex_list:
		try:
			new_max = max([abs(v[i].y- x[i].y) for x in vertex_list])
		except IndexError:
			print("Index error")
			print(v)
			print(i)
			print(vertex_list)
			pdb.set_trace()
			input()
			return(max_dist)
		if new_max > max_dist:
			max_dist=new_max
    #print("Max detected across all is..."+ str(max_dist))
	return max_dist

def calibrate_threshold(text_boxes):
	time_coord=""
	test_coord=""
	ID_coord=""
	for box in text_boxes[1:]:
		if box['text'] == "Time":
			time_coord=box['vertices']
		if box['text'] == "Test":
			test_coord=box['vertices']
		if box['text'] == "ID":
			ID_coord=box['vertices']
	vl=[ID_coord, test_coord, time_coord]
	if len(ID_coord) == 0 or len(test_coord) == 0 or len(time_coord) == 0:
			print("Unusual case here... unable to find one of them....")
			if len(ID_coord) == 0: 
				vl=[test_coord, time_coord]
			if len(test_coord) == 0: 
				vl=[ID_coord, time_coord]     
			if len(time_coord) == 0: 
				vl=[ID_coord, test_coord] 
	out_thresh=max(maxCoordDiff(vl,0), maxCoordDiff(vl,3)) + 2
	if out_thresh > 15:
		print("Unusual case..")
		print(vl)
		input()
		out_thresh=15
	return( out_thresh)

def align_header(text_lines):
    move_line=-1
    merge_line = -1
    for i, line in enumerate(text_lines):          
          if line[0]["text"] == "Sample":
               if line[1]["text"] == ":":
                    if len(text_lines[i-1]) == 1 and text_lines[i-1][0]["text"].isnumeric():
                         #print(line[0])
                         merge_line = i
                         move_line =  i-1
                         break
    if merge_line != -1:
        to_insert=text_lines[move_line][0]
        #print(to_insert)
        text_lines[i].insert(2,to_insert)
        #print(text_lines[i])
    return(text_lines)
         
    


def create_string_list(sorted_rows):
	strings_all=[]
	for row in sorted_rows:
		strings_all.append("\t".join(box['text'] for box in row))
	return strings_all		
# construct the argument parser and parse the arguments


def transcribeImage(image_file, google_client):
      	
    with io.open(image_file, "rb") as f:
        byteImage = f.read()

    # create an image object from the binary file and then make a request
    # to the Google Cloud Vision API to OCR the input image
    print("[INFO] making request to Google Cloud Vision API...")
    image = vision.Image(content=byteImage)
    response = google_client.text_detection(image=image, image_context={"language_hints": ["en"]})  # Bengali))
	#this seems to help a bit....
    # check to see if there was an error when making a request to the API
    if response.error.message:
        raise Exception(
            "{}\nFor more info on errors, check:\n"
            "https://cloud.google.com/apis/design/errors".format(
                response.error.message))



    # Assuming 'response' is the result from Google Vision API text detection
    # response = client.text_detection(image=image)
    # Extract text annotations
    text_boxes = extract_text_annotations(response)

    # Group text boxes by rows
    threshold=calibrate_threshold(text_boxes)
    print(threshold)
    #print("Threshold choice: " + str(threshold))
    #rows = group_by_rows(text_boxes,x_threshold=threshold)
    rows = group_by_cols(text_boxes,y_threshold=threshold)
    #issue preceeds below
    # Sort text boxes within each row by x coordinate
    sorted_rows = sort_rows_by_x(rows)
    sorted_rows = align_header(sorted_rows)
    sorted_rows = sort_rows_by_x(rows)
    #sorted_cols = sort_cols_by_y(sorted_rows)
    #This command just messes stuff up.
    #print_table(sorted_rows)
    all_data = create_string_list(sorted_rows) #Something is wrong here-  the text is getting inverted

    #Now start parsing the data
    header_queries_one = ["Sample", "Species", "Date"]
    header_queries_two =["Sample ID", "Test No", "Time"]

    #header_extract(all_data[3],"Species", "Sample\t:\t(\d+)\tSpecies\t:\t(\w+)\tDate\t:\t([\d\/]+)")
    #alt = extract_header_regex(all_data,"Species", "Sample\t:\t([\d*]?)\tSpecies\t:\t(\w+)\tDate\t:\t([\d\/]+)$")
    full_dat = extract_header(all_data, header_queries_one,lookup_index=0)
    #full_dat = dict(zip(header_queries_one,[str_puller(all_data[3].split(), x) for x in header_queries_one]))
    #full_dat.update(dict(zip(header_queries_two,[str_puller(all_data[4].split(), x) for x in header_queries_two])))
    second_header = extract_header(all_data, header_queries_two,lookup_index=2)
    full_dat.update(second_header)
    #Get the starting index for the searches:
    leuk_index=-1  
    eryth_index=-1
    thromb_index=-1             
    for i, line in enumerate(all_data):
        if 'Leukocytes\t:' in line:
            leuk_index=i
        if 'Erythrocytes\t:' in line:
            eryth_index=i
        if 'Thrombocytes\t:' in line:
            thromb_index=i
    #WRONGIssues with starting point, for some; just do global min:
    #leuk_index = min(leuk_index,eryth_index,thromb_index )
 
    import re	
    leukocyte_measures = ["WBC", "NE", "LY", "MO", "EO", "BA", "NRBC"]
    measure_query="\\t\(\\tK\\t\/\\t\w+\\t\)\\t([0-9\.]+)\\t"
    leuk_quant_dict= segment_extract(all_data[leuk_index:eryth_index],measure_query,leukocyte_measures)
    full_dat.update(leuk_quant_dict)

    leukocyte_perc_keys = ["NE%", "LY%", "MO%", "EO%", "BA%"]
    leukocyte_perc_queries = ["NE", "LY", "MO", "EO", "BA"]
    percent_query="\\t\(\\t[\%8]\\t\)\\t([0-9\.]+)\\t"	
    print("Leukocyte percent dictionary now.....")
    #input()
    #input()
    #pdb.set_trace()
    #print(all_data[(leuk_index+5):eryth_index])
    leuk_perc_dict= segment_extract(all_data[(leuk_index+5):eryth_index],percent_query,leukocyte_perc_queries)
    #input()
    #print(leuk_perc_dict)
    #Need to get just the percentages
    full_dat.update(dict(zip(leukocyte_perc_keys,leuk_perc_dict.values())))

    #general_query="\\t\(\\t[\w\s.\\/%8]+\\t\)\\t([0-9\.]+)\\t"
    general_query="\\t\(\\t[\w\\t\/\s%8.]+\\t\)\\t([0-9\\.]+)[-\*]*\\t"
    erythrocyte_measures=["RBC", "Hb","HCT", "MCV", "MCH", "MCHC", "RDW"]
    erythrocyte_quant_dict= segment_extract(all_data[eryth_index:thromb_index],general_query,erythrocyte_measures)
    full_dat.update(erythrocyte_quant_dict)

    thrombocyte_perc = ["PLT", "MPV"]
    throm_quant_dict= segment_extract(all_data[thromb_index:],general_query,thrombocyte_perc)
    full_dat.update(throm_quant_dict)

    return(full_dat)    
#This is complete.



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_dir", required=True,
	help="path to input image that we'll submit to Google Vision API")
ap.add_argument("-c", "--client", default="/Users/aomdahl/.config/gcloud/application_default_credentials.json",
	help="path to input client ID JSON configuration file")
ap.add_argument("-o", "--output", default="/Users/aomdahl/OneDrive_JHU/Personal/Kay/hemavet_pdfs/processed_data/",
	help="path to output files")
args = vars(ap.parse_args())


client = vision.ImageAnnotatorClient()


# load the input image as a raw binary file (this file will be
# submitted to the Google Cloud Vision API)
from os import walk

filenames = next(walk(args["image_dir"]), (None, None, []))[2]
out_dict=dict()
for f in filenames:
    fdir=os.path.join(args["image_dir"], f)
    print(fdir)
    first_one = transcribeImage(fdir, client)
    print("successfully completed " + fdir)
    #print(first_one)
    out_dict[fdir]=first_one


#write it out with pandas.
import pandas as pd
df_out=pd.DataFrame(out_dict)
#old_names=list(df.columns)
new_names = [os.path.basename(x) for x in list(df_out.columns)]
#rename_dict = dict(zip(old_names,new_names))
#df = df.rename(columns=rename_dict)
df =df_out.transpose()
df.insert(0, "sample_image", new_names)
df.to_csv(args["output"])
#Save for latter looking
#import pickle
#with open(os.path.join(args["output"], "parsed.data.pickle"), 'wb') as f:  # open a text file
#    pickle.dump(out_dict, f) # serialize the list



#TODO:
#Figure out why the sample data isn't coming through correctly. It doesn't seem to want to pick it up at all Also in some cases we aren't getting the data.
#Maybe we need to make a regex for these instead. hiya.

## full run:
#python3.9 process_images_v2.py --image_dir hemavet_pdfs/separate_jpegs/ --output hemavet_pdfs/processed_data_updated.csv

##debug run:
#python3.9 process_images_v2.py --image_dir hemavet_pdfs/debug_jpegs/ --output hemavet_pdfs/debug_data.csv