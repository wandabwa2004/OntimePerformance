# -*- coding: utf-8 -*-
"""
Created on Mon 11th Jan 2021
@author: Herman Wandabwa - herman.wandabwa@kiwirail.co.nz
"""

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('nexplorer.jpg')



def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]
    
model = load_model('Randomforestmodel')


st.title('Kiwirail Trains OnTime  Predictions')
st.write('A web-based app to predict whether a train will be late  or on time based on several features that you can see in the sidebar. Please adjust / select the value of each feature. After that, click on the Predict button at the bottom to  see the prediction of the model. Based on the  number of features at the moment, \
          accuracy in the predictions is about 91.81%. With more parameters, the model will improve.')

#origin_1_choices = {1:"Auckland",2: "Hamilton", 3: "Palmerston North"}
# def format_func(option)

train_id = st.sidebar.selectbox("Train", ('B21', '269X', '249', 'B15', 'B43', 'B49', 'F19', '263', '243',
                                          '217', '221', '229', '267', '239', '225', '241R', 'B25', '267M',
                                          '241', '263X', '261', '235R', 'B35', 'B23', '239X', '217X', '221X',
                                          '221Y', '225X', '229X', '229Y', '243X', '243Y', 'F17', 'F11',
                                          '263S', '269', '225S', '247'))

trainclass = st.sidebar.selectbox("Train Class", ("EX","-"))

base_train = st.sidebar.selectbox("Base Train", ('B21', '269', '249', 'B15', 'B43', 'B49', 'F19', '263', '243','217', '221', '229', '267', '239', '225', '241', 'B25', '261','235', 'B35', 'B23', 'F17', 'F11', '247'))                                         

train_category =  st.sidebar.selectbox('Train Category', ("Premium Freight","General Freight","Other"))

direction = st.sidebar.selectbox("Train Direction", ("N-S","-"))

traingroup = st.sidebar.selectbox("Train Group", ('Auckland - Christchurch', 'Other'))

AttachDetachKMs = st.sidebar.slider(label = 'Attached RollingStock Pull Distance in KMs', min_value = 0.00, max_value = 20000.00,  value = 0.00,  step = 1.0)

total_trainweight = st.sidebar.slider(label = 'Total Train Weight in Tonnes', min_value = 0.00, max_value = 20000.00,  value = 0.00,  step = 1.0)

origin =  st.sidebar.selectbox("Origin", ("Auckland","Hamilton","Palmerston North"))
CurrentLocation = st.sidebar.selectbox ("Current Location",('AUCK', 'HAM', 'MGWKA', 'MRTON', 'OHKNE', 'OTAKI', 'PKKHE', 'PNTH',
                                                             'PPKRA', 'PPRMU', 'PRIKI', 'PRRUA', 'TAUM', 'TKUTI', 'WIRI',
                                                                'WIURU', 'PMNST', 'OWHGO', 'NATPK', 'WIKNE', 'MRCER', 'POTRO',
                                                                    'OHUHU', 'HTRVL', 'OTRGA', 'RRIMU', 'TAMTU', 'RKHIA', 'HAMST',
                                                                    'LNBRN', 'SHNON', 'ONGRE', 'KPAKI', 'PKTTU', 'MAKTE', 'BNYTP',
                                                                    'OIO', 'TNGWI', 'TRAPA', 'OHAPO', 'OHAU', 'NGKHU', 'THAPE',
                                                                'WDVLE', 'HNTLY', 'LEVIN', 'HNGTK', 'HRPTO', 'WGTN', 'MNNUI','TKAWA', 'MAST'))

currentlocationtype = st.sidebar.selectbox ("Current Location Type",('ORIG', 'INTR', 'DEST'))                                                       

destination = st.sidebar.selectbox("Destination", ("Wellington","Auckland")) 

month = st.sidebar.selectbox("Month of The Year", ("January","February","March","April","May","June","July","August","September","October","November","December"))
week = st.sidebar.slider(label = 'Week of  the Year', min_value = 0, max_value = 52,  value = 0,  step = 1)
day_week = st.sidebar.selectbox("Day of the Week", ("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"))
   
# region = st.sidebar.selectbox("Region", ("NI-Upper","NI-Lower"))

IncidentType = st.sidebar.selectbox("LSM Incident Type:", ('None_LSM', 'Primary', 'Secondary'))

OccuranceType = st.sidebar.selectbox("LSM Occurance Type:",('None_LSM', 'Extra Services', 'Cancellation', 'Rail Operations',
                                                                                                 'Locomotive Incidents', 'Incidents/Accidents', 'Rolling Stock',
                                                                                                    'Line Closure', 'Cargo/Loading Irregularities', 'Derailments',
                                                                                                         'Collisions/Near Collisions', 'Operating Irregularities'))

Incident_location = st.sidebar.selectbox("Incident Location:",('None_LSM', 'PALMERSTON NORTH', 'HAMILTON', 'NATIONAL PARK',
                                                              'MARTON', 'AUCKLAND', 'KARIOI', 'WIRI', 'BURBUSH', 'PAPAKURA','OHAUPO', 'TAKANINI', 'OHAKUNE', 'MAKATOTE', 'KOPAKI',
                                                                'TE AWAMUTU', 'HANGATIKI', 'WAIMIHA', 'TE KUITI', 'PLIMMERTON','WELLINGTON', 'TAUMARUNUI', 'RAURIMU', 'TANGIWAI', 'PUKEKOHE',
                          'PAEKAKARIKI', 'WAIOURU', 'OWHANGO', 'LINTON', 'WHANGAMARINO','TE KAUWHATA', 'HOROPITO', 'OTOROHANGA', 'NGARUAWAHIA', 'KAKAHI',
                          'TAIHAPE', 'POREWA', 'TAUPIRI', 'AMOKURA', 'AUCKLAND PORT', 'MERCER', 'OTAHUHU', 'WESTFIELD', 'POROOTARAO', 'TE RAPA', 'MAEWA', 'BUNNYTHORPE', 'OTAKI', 'LEVIN', 'PARAPARAUMU', 'PORIRUA',
                          'TE KAWA', 'MANUNUI', 'HIHITAHI', 'OHAU', 'SHANNON (NI)', 'EDENDALE', 'MANGAONOHO', 'HOROTIU', 'TOKOMARU', 'HUNTERVILLE', 'FEILDING', 'PUKERUA BAY', 'WAIKANAE', 'LONGBURN', 'OIO',
                           'OKAHUKURA', 'UTIKU', 'PAERATA', 'HUNTLY', 'GREATFORD', 'ONGARUE', 'RUKUHIA', 'TUAKAU', 'PUKETUTU', 'MATAROA', 'Mercer', 'TE HORO', 'NGAURUKEHU', 'WAITETI', 'MANGAWEKA'))

trainline  = st.sidebar.selectbox("LSM Incident Train Line",('None_LSM', 'NIMT', 'MSL'))

lsm_delays  = st.sidebar.slider(label = 'Delayed Minutes by LSM', min_value = 0.00, max_value = 1000.00,  value = 0.00,  step = 1.0)

#Features in a DF 
features = {'TrainID': train_id, 'Train Class': trainclass,'Base Train': base_train,'Train Category': train_category,'Direction': direction,
            'Train Group': traingroup,'AttachDetachKMs': AttachDetachKMs,'Total_Weight': total_trainweight,'Train Origin': origin,
            'CurrentLocation': CurrentLocation,'WorkstationType': currentlocationtype, 'Destination': destination,'Month': month,
            'Week': week,'DayofWeek': day_week,'LSM Incident Type': IncidentType,'OccuranceType': OccuranceType,'Incident Location': Incident_location,
            'TrainLine': trainline,'DelayMins_LSM': lsm_delays}

# features = {'TrainID': train_id, 'Origin': origin,
#             'Dest': destination, 'Month': month,
#             'DayofWeek': day_week, 'Region': region,
#             'Train Category': train_category, 'BaseTrain': base_train,
#             'DelayMins_LSM': lsm_delays }
 

features_df  = pd.DataFrame([features])

#Change categorical values to encoded numerals  for prediction

features_df1 = features_df.copy()


st.table(features_df1)  

if st.button('Predict'):
    #Encode  trains  

    trains_encode = [
     (features_df1['TrainID'] == "B21"),
     (features_df1['TrainID'] == "269X"),
     (features_df1['TrainID'] == "249"),
     (features_df1['TrainID'] == "B15"),
     (features_df1['TrainID'] == "B43"),
     (features_df1['TrainID'] == "B49"),
     (features_df1['TrainID'] == "F19"),
     (features_df1['TrainID'] == "263"),
     (features_df1['TrainID'] == "243"),
     (features_df1['TrainID'] == "217"),
     (features_df1['TrainID'] == "221"),
     (features_df1['TrainID'] == "229"),
     (features_df1['TrainID'] == "267"),
     (features_df1['TrainID'] == "239"),
     (features_df1['TrainID'] == "225"),
     (features_df1['TrainID'] == "241R"),
     (features_df1['TrainID'] == "B25"),
     (features_df1['TrainID'] == "267M"),
     (features_df1['TrainID'] == "241"),
     (features_df1['TrainID'] == "263X"),
     (features_df1['TrainID'] == "261"),
     (features_df1['TrainID'] == "235R"),
     (features_df1['TrainID'] == "B35"),
     (features_df1['TrainID'] == "B23"),
     (features_df1['TrainID'] == "239X"),
     (features_df1['TrainID'] == "217X"),
     (features_df1['TrainID'] == "221X"),
     (features_df1['TrainID'] == "221Y"),
     (features_df1['TrainID'] == "225X"),
     (features_df1['TrainID'] == "229X"),
     (features_df1['TrainID'] == "229Y"),
     (features_df1['TrainID'] == "243X"),
     (features_df1['TrainID'] == "243Y"),
     (features_df1['TrainID'] == "F17"),
     (features_df1['TrainID'] == "F11"),
     (features_df1['TrainID'] == "263S"),
     (features_df1['TrainID'] == "269"),
     (features_df1['TrainID'] == "225S"),
     (features_df1['TrainID'] == "247") ]
    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_trains_encode = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39] 
    features_df1['TrainID'] = np.select(trains_encode, values_trains_encode)
    
    #Trains  current location

    trains_currentloc_encode = [
    
     (features_df1['CurrentLocation'] == "AUCK"),
     (features_df1['CurrentLocation'] == "HAM"),
     (features_df1['CurrentLocation'] == "MGWKA"),
     (features_df1['CurrentLocation'] == "MRTON"),
     (features_df1['CurrentLocation'] == "OHKNE"),
     (features_df1['CurrentLocation'] == "OTAKI"),
     (features_df1['CurrentLocation'] == "PKKHE"),
     (features_df1['CurrentLocation'] == "PNTH"),
     (features_df1['CurrentLocation'] == "PPKRA"),
     (features_df1['CurrentLocation'] == "PPRMU"),
     (features_df1['CurrentLocation'] == "PRIKI"),
     (features_df1['CurrentLocation'] == "PRRUA"),
     (features_df1['CurrentLocation'] == "TAUM"),
     (features_df1['CurrentLocation'] == "TKUTI"),
     (features_df1['CurrentLocation'] == "WIRI"),
     (features_df1['CurrentLocation'] == "WIURU"),
     (features_df1['CurrentLocation'] == "PMNST"),
     (features_df1['CurrentLocation'] == "OWHGO"),
     (features_df1['CurrentLocation'] == "NATPK"),
     (features_df1['CurrentLocation'] == "WIKNE"),
     (features_df1['CurrentLocation'] == "MRCER"),
     (features_df1['CurrentLocation'] == "POTRO"),
     (features_df1['CurrentLocation'] == "OHUHU"),
     (features_df1['CurrentLocation'] == "HTRVL"),
     (features_df1['CurrentLocation'] == "OTRGA"),
     (features_df1['CurrentLocation'] == "RRIMU"),
     (features_df1['CurrentLocation'] == "TAMTU"),
     (features_df1['CurrentLocation'] == "RKHIA"),
     (features_df1['CurrentLocation'] == "HAMST"),
     (features_df1['CurrentLocation'] == "LNBRN"),
     (features_df1['CurrentLocation'] == "SHNON"),
     (features_df1['CurrentLocation'] == "ONGRE"),
     (features_df1['CurrentLocation'] == "KPAKI"),
     (features_df1['CurrentLocation'] == "PKTTU"),
     (features_df1['CurrentLocation'] == "MAKTE"),
     (features_df1['CurrentLocation'] == "BNYTP"),
     (features_df1['CurrentLocation'] == "OIO"),
     (features_df1['CurrentLocation'] == "TNGWI"),
     (features_df1['CurrentLocation'] == "TRAPA"),
     (features_df1['CurrentLocation'] == "OHAPO"),
     (features_df1['CurrentLocation'] == "OHAU"),
     (features_df1['CurrentLocation'] == "NGKHU"),
     (features_df1['CurrentLocation'] == "THAPE"),
     (features_df1['CurrentLocation'] == "WDVLE"),
     (features_df1['CurrentLocation'] == "HNTLY"),
     (features_df1['CurrentLocation'] == "LEVIN"),
     (features_df1['CurrentLocation'] == "HNGTK"),
     (features_df1['CurrentLocation'] == "HRPTO"),
     (features_df1['CurrentLocation'] == "WGTN"),
     (features_df1['CurrentLocation'] == "MNNUI"),
     (features_df1['CurrentLocation'] == "TKAWA"),
     (features_df1['CurrentLocation'] == "MAST")]

    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_trains_currentloc_encode = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                            31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52] 

    # create a new column and use np.select to assign values to it using our lists as arguments
    features_df1['CurrentLocation'] = np.select(trains_currentloc_encode, values_trains_currentloc_encode)
    
    #Workstation Type 
    trains_workstationtype = [
        (features_df1['WorkstationType'] == "ORIG"),
        (features_df1['WorkstationType'] == "INTR"),
        (features_df1['WorkstationType'] == "DEST")]

    values_workstationtype_encode = [1,2,3]

    features_df1['WorkstationType'] = np.select(trains_workstationtype, values_workstationtype_encode)

    #Train  origins 
    trains_origins = [
        (features_df1['Train Origin'] == "Auckland"),
        (features_df1['Train Origin'] == "Hamilton"),
        (features_df1['Train Origin'] == "Palmerston North")]

    values_locations_encode = [1,2,3]
    features_df1['Origin'] = np.select(trains_origins, values_locations_encode)

    #Trains  destination(s)
    trains_dest = [
        (features_df1['Destination'] == "Wellington") ]

    values_dest_encode = [1]
    features_df1['Dest'] = np.select(trains_dest, values_dest_encode)

    #Train Class 
    trains_trainclass = [
     
     (features_df1['Train Class'] == "EX") ]

    values_trainclass_encode = [1]
    features_df1['TrainClass'] = np.select(trains_trainclass, values_trainclass_encode)

    #Incident Type  
    trains_incidencetypes = [
     (features_df1['LSM Incident Type'] == "None_LSM"),
     (features_df1['LSM Incident Type'] == "Primary"),
     (features_df1['LSM Incident Type'] == "Secondary")
     ]
    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_tincidences_encode = [1,2,3] 

    # create a new column and use np.select to assign values to it using our lists as arguments
    features_df1['IncidentType'] = np.select(trains_incidencetypes, values_tincidences_encode)

    #Train OccuranceTypes 
    trains_occurancetypes = [
     (features_df1['OccuranceType'] == "None_LSM"),
     (features_df1['OccuranceType'] == "Extra Services"),
     (features_df1['OccuranceType'] == "Cancellation"),
     (features_df1['OccuranceType'] == "Rail Operations"),
     (features_df1['OccuranceType'] == "Locomotive Incidents"),
     (features_df1['OccuranceType'] == "Incidents/Accidents"),
     (features_df1['OccuranceType'] == "Rolling Stock"),
     (features_df1['OccuranceType'] == "Line Closure"),
     (features_df1['OccuranceType'] == "Cargo/Loading Irregularities"),
     (features_df1['OccuranceType'] == "Derailments"),
     (features_df1['OccuranceType'] == "Collisions/Near Collisions"),
     (features_df1['OccuranceType'] == "Operating Irregularities")
     ]
    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_occurancetypes_encode = [1,2,3,4,5,6,7,8,9,10,11,12] 

    # create a new column and use np.select to assign values to it using our lists as arguments
    features_df1['OccuranceType'] = np.select(trains_occurancetypes, values_occurancetypes_encode)

    #Train Line 
    trains_trainline = [
     (features_df1['TrainLine'] == "None_LSM"),
     (features_df1['TrainLine'] == "NIMT"),
     (features_df1['TrainLine'] == "MSL")
     ]
    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_trainline_encode = [1,2,3] 

    # create a new column and use np.select to assign values to it using our lists as arguments
    features_df1['TrainLine'] = np.select(trains_trainline, values_trainline_encode)

    #Base Train Details - Other trains can be added but  only after retraining the  model
    trains_base_encode = [
     (features_df1['Base Train'] == "B21"),
     (features_df1['Base Train'] == "269"),
     (features_df1['Base Train'] == "249"),
     (features_df1['Base Train'] == "B15"),
     (features_df1['Base Train'] == "B43"),
     (features_df1['Base Train'] == "B49"),
     (features_df1['Base Train'] == "F19"),
     (features_df1['Base Train'] == "263"),
     (features_df1['Base Train'] == "243"),
     (features_df1['Base Train'] == "217"),
     (features_df1['Base Train'] == "221"),
     (features_df1['Base Train'] == "229"),
     (features_df1['Base Train'] == "267"),
     (features_df1['Base Train'] == "239"),
     (features_df1['Base Train'] == "225"),
     (features_df1['Base Train'] == "241"),
     (features_df1['Base Train'] == "B25"),
     (features_df1['Base Train'] == "261"),
     (features_df1['Base Train'] == "235"),
     (features_df1['Base Train'] == "B35"),
     (features_df1['Base Train'] == "B23"),
     (features_df1['Base Train'] == "F17"),
     (features_df1['Base Train'] == "F11"),
     (features_df1['Base Train'] == "247")
    ]
    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_basetrains_encode = [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] 

    # create a new column and use np.select to assign values to it using our lists as arguments
    features_df1['BaseTrain'] = np.select(trains_base_encode, values_basetrains_encode)

    trains_incident_loc_encode = [
    
     (features_df1['Incident Location'] == "None_LSM"),
     (features_df1['Incident Location'] == "PALMERSTON NORTH"),
     (features_df1['Incident Location'] == "HAMILTON"),
     (features_df1['Incident Location'] == "NATIONAL PARK"),
     (features_df1['Incident Location'] == "MARTON"),
     (features_df1['Incident Location'] == "AUCKLAND"),
     (features_df1['Incident Location'] == "KARIOI"),
     (features_df1['Incident Location'] == "WIRI"),
     (features_df1['Incident Location'] == "BURBUSH"),
     (features_df1['Incident Location'] == "PAPAKURA"),
     (features_df1['Incident Location'] == "OHAUPO"),
     (features_df1['Incident Location'] == "TAKANINI"),
     (features_df1['Incident Location'] == "OHAKUNE"),
     (features_df1['Incident Location'] == "MAKATOTE"),
     (features_df1['Incident Location'] == "KOPAKI"),
     (features_df1['Incident Location'] == "TE AWAMUTU"),
     (features_df1['Incident Location'] == "HANGATIKI"),
     (features_df1['Incident Location'] == "WAIMIHA"),
     (features_df1['Incident Location'] == "TE KUITI"),
     (features_df1['Incident Location'] == "PLIMMERTON"),
     (features_df1['Incident Location'] == "WELLINGTON"),
     (features_df1['Incident Location'] == "TAUMARUNUI"),
     (features_df1['Incident Location'] == "RAURIMU"),
     (features_df1['Incident Location'] == "TANGIWAI"),
     (features_df1['Incident Location'] == "PUKEKOHE"),
     (features_df1['Incident Location'] == "PAEKAKARIKI"),
     (features_df1['Incident Location'] == "WAIOURU"),
     (features_df1['Incident Location'] == "OWHANGO"),
     (features_df1['Incident Location'] == "LINTON"),
     (features_df1['Incident Location'] == "WHANGAMARINO"),
     (features_df1['Incident Location'] == "TE KAUWHATA"),
     (features_df1['Incident Location'] == "HOROPITO"),
     (features_df1['Incident Location'] == "OTOROHANGA"),
     (features_df1['Incident Location'] == "NGARUAWAHIA"),
     (features_df1['Incident Location'] == "KAKAHI"),
     (features_df1['Incident Location'] == "TAIHAPE"),
     (features_df1['Incident Location'] == "POREWA"),
     (features_df1['Incident Location'] == "TAUPIRI"),
     (features_df1['Incident Location'] == "AMOKURA"),
     (features_df1['Incident Location'] == "AUCKLAND PORT"),
     (features_df1['Incident Location'] == "MERCER"),
     (features_df1['Incident Location'] == "OTAHUHU"),
     (features_df1['Incident Location'] == "WESTFIELD"),
     (features_df1['Incident Location'] == "POROOTARAO"),
     (features_df1['Incident Location'] == "TE RAPA"),
     (features_df1['Incident Location'] == "MAEWA"),
     (features_df1['Incident Location'] == "BUNNYTHORPE"),
     (features_df1['Incident Location'] == "OTAKI"),
     (features_df1['Incident Location'] == "LEVIN"),
     (features_df1['Incident Location'] == "PARAPARAUMU"),
     (features_df1['Incident Location'] == "PORIRUA"),
     (features_df1['Incident Location'] == "TE KAWA"),
     (features_df1['Incident Location'] == "MANUNUI"),
     (features_df1['Incident Location'] == "HIHITAHI"),
     (features_df1['Incident Location'] == "OHAU"),
     (features_df1['Incident Location'] == "SHANNON (NI)"),
     (features_df1['Incident Location'] == "EDENDALE"),
     (features_df1['Incident Location'] == "MANGAONOHO"),
     (features_df1['Incident Location'] == "HOROTIU"),
     (features_df1['Incident Location'] == "TOKOMARU"),
     (features_df1['Incident Location'] == "HUNTERVILLE"),
     (features_df1['Incident Location'] == "FEILDING"),
     (features_df1['Incident Location'] == "PUKERUA BAY"),
     (features_df1['Incident Location'] == "WAIKANAE"),
     (features_df1['Incident Location'] == "LONGBURN"),
     (features_df1['Incident Location'] == "OIO"),
     (features_df1['Incident Location'] == "OKAHUKURA"),
     (features_df1['Incident Location'] == "UTIKU"),
     (features_df1['Incident Location'] == "PAERATA"),
     (features_df1['Incident Location'] == "HUNTLY"),
     (features_df1['Incident Location'] == "GREATFORD"),
     (features_df1['Incident Location'] == "ONGARUE"),
     (features_df1['Incident Location'] == "RUKUHIA"),
     (features_df1['Incident Location'] == "TUAKAU"),
     (features_df1['Incident Location'] == "PUKETUTU"),
     (features_df1['Incident Location'] == "MATAROA"),
     (features_df1['Incident Location'] == "Mercer"),
     (features_df1['Incident Location'] == "TE HORO"),
     (features_df1['Incident Location'] == "NGAURUKEHU"),
     (features_df1['Incident Location'] == "WAITETI"),
     (features_df1['Incident Location'] == "MANGAWEKA")]

    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_trains_incidentloc_encode = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                            31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
                                    61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81] 

    # create a new column and use np.select to assign values to it using our lists as arguments
    features_df1['Incident Location'] = np.select(trains_incident_loc_encode, values_trains_incidentloc_encode)

    #Train categories
    trains_category = [
     (features_df1['Train Category'] == "Premium Freight"),
     (features_df1['Train Category'] == "General Freight"),
     (features_df1['Train Category'] == "Other")
  
     ]
    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_tcategory_encode = [1,2,3] 

    # create a new column and use np.select to assign values to it using our lists as arguments
    features_df1['Train Category'] = np.select(trains_category, values_tcategory_encode)
    
    # Train Direction
    trains_traindirection = [
     (features_df1['Direction'] == "N-S") ]

    values_traindirection_encode = [1]
    features_df1['Direction'] = np.select(trains_traindirection, values_traindirection_encode)

    #Train Groups
    trains_traingroup = [
     (features_df1['Train Group'] == "Auckland - Christchurch"),
     (features_df1['Train Group'] == "Other")]

    values_traingroup_encode = [1,2]
    features_df1['Train Group'] = np.select(trains_traingroup, values_traingroup_encode)
    
    #Months  of operation data 
    trains_month = [
        (features_df1['Month'] == "January"),
        (features_df1['Month'] == "February"),
        (features_df1['Month'] == "March"),
        (features_df1['Month'] == "April"),
        (features_df1['Month'] == "May"),
        (features_df1['Month'] == "June"),
        (features_df1['Month'] == "July"),
        (features_df1['Month'] == "August"),
        (features_df1['Month'] == "September"),
        (features_df1['Month'] == "October"),
        (features_df1['Month'] == "November"),
        (features_df1['Month'] == "December")
        
        ]
    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_month_encode = [1, 2,3,4,5,6,7,8,9,10,11,12] 

    # create a new column and use np.select to assign values to it using our lists as arguments
    features_df1['Month'] = np.select(trains_month, values_month_encode)
    
    #Day of the Week 
    trains_dayweek = [
     (features_df1['DayofWeek'] == "Monday"),
     (features_df1['DayofWeek'] == "Tuesday"),
     (features_df1['DayofWeek'] == "Wednesday"),
     (features_df1['DayofWeek'] == "Thursday"),
     (features_df1['DayofWeek'] == "Friday"),
     (features_df1['DayofWeek'] == "Saturday"),
     (features_df1['DayofWeek'] == "Sunday") 
     
    ]
    # create a list of the values we want to assign for each condition.1 for  on time arrivals and departures and 0 for late arrivals and departures
    values_dayweek_encode = [1,2,3,4,5,6,7] 

    # create a new column and use np.select to assign values to it using our lists as arguments
    features_df1['WeekDayName'] = np.select(trains_dayweek, values_dayweek_encode)


    #Make the predictions  
    prediction = predict_quality(model, features_df1)
    
    st.subheader('*Based on the provided feature values the selected train is  likely to be*  '+  str(prediction))