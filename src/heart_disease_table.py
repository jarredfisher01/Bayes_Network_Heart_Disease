import pandas as pd
import csv 
from tqdm import tqdm
import numpy as np


def get_conditional_dictionary():
    #Creating the pandas dataframe of raw data
    df = pd.DataFrame(columns=['heart_disease','high_blood_pressure','high_cholesterol','smoker','heavy_alcohol','physical_activity'])
    df = pd.read_csv('./data/pandas_df.csv')

    #Numerator - Calculting joint probability distribution of 'heart_disease','high_blood_pressure','high_cholesterol','smoker','heavy_alcohol','physical_activity'
    combination_dict = {}
    for index, row in df.iterrows():
        row_tuple = (row['heart_disease'], row['high_blood_pressure'],row['high_cholesterol'],row['smoker'], row['heavy_alcohol'], row['physical_activity'])
        if row_tuple in combination_dict.keys():
            combination_dict[row_tuple] = combination_dict[row_tuple]+1
        else:
            combination_dict[row_tuple] = 1

    total_rows = 0 
    for x in combination_dict.values():
        total_rows+=x


    for key, value in combination_dict.items():
        combination_dict[key] = value/total_rows

    numerator_df  = pd.DataFrame(columns=['heart_disease','high_blood_pressure','high_cholesterol','smoker','heavy_alcohol','physical_activity', 'probability'])


    #Denominator - Calculting joing probability distribution of evidence: 'high_blood_pressure','high_cholesterol','smoker','heavy_alcohol','physical_activity'
    evidence_dict = {}
    for index, row in df.iterrows():
        row_tuple = (row['high_blood_pressure'],row['high_cholesterol'],row['smoker'], row['heavy_alcohol'], row['physical_activity'])
        if row_tuple in evidence_dict.keys():
            evidence_dict[row_tuple] = evidence_dict[row_tuple]+1
        else:
            evidence_dict[row_tuple] = 1

    total_rows = 0 
    for x in evidence_dict.values():
        total_rows+=x


    sum = 0 
    for key, value in evidence_dict.items():
        evidence_dict[key] = value/total_rows
        sum+=value/total_rows

    #Calculating conditional distribution  - using above values
    conditional_probability_table = {}
    for key, value in combination_dict.items():
        numerator_key = key 
        numerator_value = value 

        denominator_key = (numerator_key[1], numerator_key[2], numerator_key[3], numerator_key[4], numerator_key[5])
        denominator_value = evidence_dict[denominator_key]

        conditional_probability_table[key] = numerator_value/denominator_value

    #Add missing row that does not occur in DB 
    # conditional_probability_table[(True, False, True, False, True, False)] = 0.0

    #------------------------------------------------------------------------------------------------------------
    # Converting to correct format for PyAgrum

    # Converting from bools to ints
    formatted_conditional = {}
    for key, value in conditional_probability_table.items():
        new_key = (int(key[0]), int(key[1]), int(key[2]),int(key[3]),int(key[4]), int(key[5]))
        formatted_conditional[new_key] = value

    # Creating the correct format - first extract the evidence
    formatted = {}
    for key in formatted_conditional.keys():
        new_key = (int(key[1]), int(key[2]),int(key[3]),int(key[4]), int(key[5]))
        formatted[new_key] = (0,0)

    #Now we make the correct PyAgrum format
    for key in formatted.keys():
        #Negative Case  - i.e. 0 case
        negative_key = (0, key[0], key[1], key[2], key[3], key[4])
        negative_value = formatted_conditional[negative_key]
        #Positive Case - i.e. 1 case
        positive_key = (1, key[0], key[1], key[2], key[3], key[4])
        positive_value  = formatted_conditional[positive_key]

        #Add to formatted
        formatted[key] = (negative_value, positive_value)
    return formatted




#The following function was used to convert the health.csv file to a pandas dataframe that was saved to the pandas_df.csv
def convert_to_pandas_df():
        # Converting Raw data to pandas CSV
    with open('./data/health.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in tqdm(csv_reader):
            heart_disease = row[0]
            
            if heart_disease=='1.0':
                heart_disease = True
                
            else:
                heart_disease = False
            
            high_blood_pressure = row[1]

            if high_blood_pressure =='1.0':
                high_blood_pressure = True
            else:
                high_blood_pressure = False
            
            high_cholesterol = row[2]
            if high_cholesterol =='1.0':
                high_cholesterol = True
            else:
                high_cholesterol = False
            
            smoker = row[5]
            if smoker == '1.0':
                smoker = True
            else:
                smoker = False

            physical_actvity = row[8]
            if physical_actvity == '1.0':
                physical_actvity = True
            else:
                physical_actvity = False

            heavy_alcohol = row[11]
            if heavy_alcohol == '1.0':
                heavy_alcohol = True
            else:
                heavy_alcohol = False
        
            row_array = [heart_disease, high_blood_pressure, high_cholesterol, smoker, heavy_alcohol, physical_actvity]
            df.loc[len(df.index)] = row_array

