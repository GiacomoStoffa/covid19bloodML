
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure
import streamlit.components.v1 as components
#from win32api import GetSystemMetrics
import matplotlib 
import xlrd
import openpyxl
import xlsxwriter
from openpyxl import load_workbook
import io
st.set_option('deprecation.showfileUploaderEncoding', False)
#Width = GetSystemMetrics(0)
#Height=GetSystemMetrics(1)

#########################COPIO FLEXIBLE.PY#################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
class FlexibleScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=None):
        self.scaler = scaler
        self.check = False
    def __assign_scaler(self):
        if self.scaler == 'min-max':
            self.method = MinMaxScaler()
        elif self.scaler == 'standard':
            self.method = StandardScaler()
        elif self.scaler == 'yeo-johnson':
            self.method = PowerTransformer(method='yeo-johnson')
        elif self.scaler == 'box-cox':
            self.method = PowerTransformer(method='box-cox')
        elif self.scaler == 'max-abs':
            self.method = MaxAbsScaler()
        elif self.scaler == 'robust':
            self.method = RobustScaler()
        elif self.scaler == 'normalize':
            self.method = Normalizer()
        else:
            self.method = None
        self.check = True
    def fit_transform(self, X, y=None, **fit_params):
        if not self.check:
            self.__assign_scaler()
        if self.method is None:
            return X
        return self.method.fit_transform(X, y, **fit_params)
    def fit(self, X):
        if not self.check:
            self.__assign_scaler()
        if self.method is None:
            return X
        self.method.fit(X)
    def transform(self, X):
        if not self.check:
            self.__assign_scaler()
        if self.method is None:
            return X
        return self.method.transform(X)
####################################################################################




st.beta_set_page_config(
     page_title="ML Covid Blood Test",
     page_icon=":link:",
     layout="centered",
     initial_sidebar_state="auto",
 )
st.write("""
# ML-based COVID-19 Test from routine blood test
""")






st.header('User Input Parameters: ')
st.sidebar.header('Select the type of dataset to use: ')
dataset = st.sidebar.radio(
    "",
    ('CBC', 'COVID predictive'))
st.sidebar.header('Select the model to use: ')
model = st.sidebar.radio(
    "",
    ('Ensemble', 'SVM','Random Forest','KNN','Logistic Regression','Naive Bayes'))

st.write("Select Model and Dataset in the left sidebar")
st.write(" ")
st.info("Fill in the all the fields of the following form or upload your CSV file")
st.info("*Leave the field blank if any actual value is not available*")

errore=0

def is_number(s):
   try:
       float(s)
       return True
   except ValueError:
       return False

def user_input_features_cbc():		
	errore=0
	
	gender = st.selectbox('GENDER',("Do not specify","Male","Female"))
	age = st.text_input("AGE", "")
	if is_number(age)==True:
		age=float(age)
	else:
		if age=="":
			age=None
		else:
			errore=1
			st.warning("Please insert a valid Number ")
	hct = st.text_input('HCT: Hematocrit (%)'"")
	if is_number(hct)==True:
		hct=float(hct)
	else:
		if hct=="":
			hct=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	hgb = st.text_input('HGB: Hemoglobin (g/dL)',"")
	if is_number(hgb)==True:
		hgb=float(hgb)
	else:
		if hgb=="":
			hgb=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	mch = st.text_input('MCH: Mean Corpuscolar Hemoglobin (pg/Cell)',"")
	if is_number(mch)==True:
		mch=float(mch)
	else:
		if mch=="":
			mch=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	mchc = st.text_input('MCHC: Mean Corpuscolar Hemoglobin Concentration (g Hb/dL)',"")
	if is_number(mchc)==True:
		mchc=float(mchc)
	else:
		if mchc=="":
			mchc=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx) ")
	mcv = st.text_input('MCV: Average Globular Volume (fL)',"")
	if is_number(mcv)==True:
		mcv=float(mcv)
	else:
		if mcv=="":
			mcv=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx) ")
	rbc = st.text_input('RBC: Red Blood Cells (10^12/L)',"")
	if is_number(rbc)==True:
		rbc=float(rbc)
	else:
		if rbc=="":
			rbc=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	wbc = st.text_input('WBC: White Blood Cells (10^9/L )',"")
	if is_number(wbc)==True:
		wbc=float(wbc)
	else:
		if wbc=="":
			wbc=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	plt1 = st.text_input('PLT1: Platelets (10^9/L ) ',"")
	if is_number(plt1)==True:
		plt1=float(plt1)
	else:
		if plt1=="":
			plt1=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ne = st.text_input('NE: Neutrophils count (%)',"")
	if is_number(ne)==True:
		ne=float(ne)
	else:
		if ne=="":
			ne=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ly = st.text_input('LY: Lymphocytes count (%)',"")
	if is_number(ly)==True:
		ly=float(ly)
	else:
		if ly=="":
			ly=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	mo = st.text_input('MO: Monocytes count (%)',"")
	if is_number(mo)==True:
		mo=float(mo)
	else:
		if mo=="":
			mo=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	eo = st.text_input('EO: Eosinophils count (%)',"")
	if is_number(eo)==True:
		eo=float(eo)
	else:
		if eo=="":
			eo=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ba = st.text_input('BA: Basophils count (%)',"")
	if is_number(ba)==True:
		ba=float(ba)
	else:
		if ba=="":
			ba=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	net = st.text_input('NET: Neutrophils count (10^9/L)',"")
	if is_number(net)==True:
		net=float(net)
	else:
		if net=="":
			net=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	lyt = st.text_input('LYT: Lymphocytes count (10^9/L)',"")
	if is_number(lyt)==True:
		lyt=float(lyt)
	else:
		if lyt=="":
			lyt=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	mot = st.text_input('MOT: Monocytes count (10^9/L)',"")
	if is_number(mot)==True:
		mot=float(mot)
	else:
		if mot=="":
			mot=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	eot = st.text_input('EOT: Eosinophils count (10^9/L) ',"")
	if is_number(eot)==True:
		eot=float(eot)
	else:
		if eot=="":
			eot=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	bat = st.text_input('BAT: Basophils count (10^9/L)',"")
	if is_number(bat)==True:
		bat=float(bat)
	else:
		if bat=="":
			bat=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	suspect = st.selectbox('SUSPECT: Presence of COVID-19 symptoms',("Do not specify","False","True"))

	if gender == "Male":
		gender=1
	else: 
		if gender == "Female":
			gender=0
		else:
			gender=None
		
	if suspect == "True":
		suspect=1
	else: 
		if suspect =="False":
			suspect=0
		else:
			suspect=None
			
	
	

	
	if errore==1:
		
		features = []
		
		features = np.array(features)
		features = pd.DataFrame(features)#, index=[0])
		return features
		
	else:
		
		data = {'Gender': gender,
		'Age': age,
		'HCT': hct,
		'HGB': hgb,
		'MCH': mch,
		'MCHC': mchc,
		'MCV': mcv,
		'RBC': rbc,
		'WBC': wbc,
		'PLT1': plt1,
		'NE': ne,
		'LY': ly,
		'MO': mo,
		'EO': eo,
		'BA': ba,
		'NET': net,
		'LYT': lyt,
		'MOT': mot,
		'EOT': eot,
		'BAT': bat,
		'Suspect': suspect}
		features=pd.DataFrame(data,index=["Value"])
		#features2=pd.DataFrame.transpose(features)
		# st.write(features2)
		return features
		

def user_input_features_covid():
	errore=0
	gender = st.selectbox('GENDER',("Do not specify","Male","Female"))
	age = st.text_input("AGE", "")
	if is_number(age)==True:
		age=float(age)
	else:
		if age=="":
			age=None
		else:
			errore=1
			st.warning("Please insert a valid Number ")
	ca = st.text_input('CA: Calcium (mmol/L)',"")
	if is_number(ca)==True:
		ca=float(ca)
	else:
		if ca=="":
			ca=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ck = st.text_input('CK: Creatine Kinase (U/L)',"")
	if is_number(ck)==True:
		ck=float(ck)
	else:
		if ck=="":
			ck=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	crea = st.text_input('CREA: Creatine (mg/dL)',"")
	if is_number(crea)==True:
		crea=float(crea)
	else:
		if crea=="":
			crea=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	alp = st.text_input('ALP: Alkaline Phosphatase (U/L)',"")
	if is_number(alp)==True:
		alp=float(alp)
	else:
		if alp=="":
			alp=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ggt = st.text_input('GGT: Gamma GlutamyItransferase (U/L)',"")
	if is_number(ggt)==True:
		ggt=float(ggt)
	else:
		if ggt=="":
			ggt=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	glu = st.text_input('GLU: Glucose (mg/dL)',"")
	if is_number(glu)==True:
		glu=float(glu)
	else:
		if glu=="":
			glu=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ast = st.text_input('AST: Aspartate Aminotransferase (U/L)',"")
	if is_number(ast)==True:
		ast=float(ast)
	else:
		if ast=="":
			ast=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	alt = st.text_input('ALT: Alanine Aminotransferase (U/L)',"")
	if is_number(alt)==True:
		alt=float(alt)
	else:
		if alt=="":
			alt=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ldh = st.text_input('LDH: Lactate Dehydrogenase (U/L)',"")
	if is_number(ldh)==True:
		ldh=float(ldh)
	else:
		if ldh=="":
			ldh=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	pcr = st.text_input('PCR: Polymerase Chain Reaction (mg/dL)',"")
	if is_number(pcr)==True:
		pcr=float(pcr)
	else:
		if pcr=="":
			pcr=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	kal = st.text_input('KAL: Potassium (mmol/L)',"")
	if is_number(kal)==True:
		kal=float(kal)
	else:
		if kal=="":
			kal=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	nat = st.text_input('NAT: Sodium (mmol/L)',"")
	if is_number(nat)==True:
		nat=float(nat)
	else:
		if nat=="":
			nat=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	urea = st.text_input('UREA: Urea (mg/dL)',"")
	if is_number(urea)==True:
		urea=float(urea)
	else:
		if urea=="":
			urea=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	wbc = st.text_input('WBC: White Blood Cells (10^9/L )',"")
	if is_number(wbc)==True:
		wbc=float(wbc)
	else:
		if wbc=="":
			wbc=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	rbc = st.text_input('RBC: Red Blood Cells (10^12/L)',"")
	if is_number(rbc)==True:
		rbc=float(rbc)
	else:
		if rbc=="":
			rbc=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	
	hgb = st.text_input('HGB: Hemoglobin (g/dL)',"")
	if is_number(hgb)==True:
		hgb=float(hgb)
	else:
		if hgb=="":
			hgb=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	hct = st.text_input('HCT: Hematocrit (%)'"")
	if is_number(hct)==True:
		hct=float(hct)
	else:
		if hct=="":
			hct=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	
	mcv = st.text_input('MCV: Average Globular Volume (fL)',"")
	if is_number(mcv)==True:
		mcv=float(mcv)
	else:
		if mcv=="":
			mcv=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	mch = st.text_input('MCH: Mean Corpuscolar Hemoglobin (pg/Cell)',"")
	if is_number(mch)==True:
		mch=float(mch)
	else:
		if mch=="":
			mch=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	mchc = st.text_input('MCHC: Mean Corpuscolar Hemoglobin Concentration (g Hb/dL)',"")
	if is_number(mchc)==True:
		mchc=float(mchc)
	else:
		if mchc=="":
			mchc=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx) ")
	plt1 = st.text_input('PLT1: Platelets (10^9/L ) ',"")
	if is_number(plt1)==True:
		plt1=float(plt1)
	else:
		if plt1=="":
			plt1=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ne = st.text_input('NE: Neutrophils count (%)',"")
	if is_number(ne)==True:
		ne=float(ne)
	else:
		if ne=="":
			ne=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ly = st.text_input('LY: Lymphocytes count (%)',"")
	if is_number(ly)==True:
		ly=float(ly)
	else:
		if ly=="":
			ly=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	mo = st.text_input('MO: Monocytes count (%)',"")
	if is_number(mo)==True:
		mo=float(mo)
	else:
		if mo=="":
			mo=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	eo = st.text_input('EO: Eosinophils count (%)',"")
	if is_number(eo)==True:
		eo=float(eo)
	else:
		if eo=="":
			eo=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	ba = st.text_input('BA: Basophils count (%)',"")
	if is_number(ba)==True:
		ba=float(ba)
	else:
		if ba=="":
			ba=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	net = st.text_input('NET: Neutrophils count (10^9/L)',"")
	if is_number(net)==True:
		net=float(net)
	else:
		if net=="":
			net=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	lyt = st.text_input('LYT: Lymphocytes count (10^9/L)',"")
	if is_number(lyt)==True:
		lyt=float(lyt)
	else:
		if lyt=="":
			lyt=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	mot = st.text_input('MOT: Monocytes count (10^9/L)',"")
	if is_number(mot)==True:
		mot=float(mot)
	else:
		if mot=="":
			mot=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	eot = st.text_input('EOT: Eosinophils count (10^9/L) ',"")
	if is_number(eot)==True:
		eot=float(eot)
	else:
		if eot=="":
			eot=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	bat = st.text_input('BAT: Basophils count (10^9/L)',"")
	if is_number(bat)==True:
		bat=float(bat)
	else:
		if bat=="":
			bat=None
		else:
			errore=1
			st.warning("Please insert a valid Number (es syntax: xx.xx)")
	suspect = st.selectbox('SUSPECT: Presence of COVID-19 symptoms',("Do not specify","False","True"))
	
	
	
	if gender == "Male":
		gender=1
	else: 
		if gender == "Female":
			gender=0
		else:
			gender=None
		
	if suspect == "True":
		suspect=1
	else: 
		if suspect =="False":
			suspect=0
		else:
			suspect=None
	
	if errore==1:
		#st.warning("Check the parameters before continuing")
		features = []
		#eatures.append([])
		features = np.array(features)
		features = pd.DataFrame(features)#, index=[0])
		return features
	else:
		data = {'Gender': gender,
			'Age': age,
			'CA': ca,
			'CK': ck,
			'CREA': crea,
			'ALP': alp,
			'GGT': ggt,
			'GLU': glu,
			'AST': ast,
			'ALT': alt,
			'LDH': ldh,
			'PCR': pcr,
			'KAL': kal,
			'NAT': nat,
			'UREA': urea,
			'WBC': wbc,
			'RBC': rbc,
			'HGB': hgb,
			'HCT': hct,
			'MCV': mcv,
			'MCH': mch,
			'MCHC': mchc,
			'PLT1': plt1,
			'NE': ne,
			'LY': ly,
			'MO': mo,
			'EO': eo,
			'BA': ba,
			'NET': net,
			'LYT': lyt,
			'MOT': mot,
			'EOT': eot,
			'BAT': bat,
			'Suspect': suspect}
			
		features = pd.DataFrame(data, index=["Value"])
		#features2=pd.DataFrame.transpose(features) 
		#st.write(features2)
		#st.bar_chart(features2)
		return features
	
	

	
	
	
	
	
if dataset == 'CBC':
	
	df = user_input_features_cbc()
	if df.empty:
		#st.warning("Please check the Parameters")
		errore=1
		#df = user_input_features_cbc()
	else:
		
		if model == 'Ensemble':
			st.sidebar.subheader('You selected CBC,  model: Ensemble')
			clf = joblib.load("ENS_fitted_CBC.joblib")
			#df = user_input_features_cbc()
		else:
			if model == 'SVM':
				st.sidebar.subheader('You selected CBC,  model: SVM')
			
				clf = joblib.load("SVM_fitted_CBC.joblib")
				#df = user_input_features_cbc()
			else:
				if model == 'Random Forest':
					st.sidebar.subheader('You selected CBC,  model: Random Forest')
				
					clf = joblib.load("RF_fitted_CBC.joblib")
					#df = user_input_features_cbc()
				else:
					if model == 'KNN':
						st.sidebar.subheader('You selected CBC,  model: KNN')
					
						clf = joblib.load("KNN_fitted_CBC.joblib")
						#df = user_input_features_cbc()
					else:
						if model == 'Logistic Regression':		
							st.sidebar.subheader('You selected CBC,  model: Logistic Regression')
						
							clf = joblib.load("LR_fitted_CBC.joblib")
							#df = user_input_features_cbc()
						else:
							st.sidebar.subheader('You selected CBC,  model: Naive Bayes')
						
							clf = joblib.load("NB_fitted_CBC.joblib")
							#df = user_input_features_cbc()
	#df = user_input_features_cbc()
	features2=pd.DataFrame.transpose(df) 
	#st.write("*Check the values below:  *")
	#st.dataframe(features2, width=150, height=550)
	
else:
	df = user_input_features_covid()
	if df.empty:
		errore=1
	else:
		
		if model == 'Ensemble':
			st.sidebar.subheader('You selected COVID,  model: Ensemble')
			clf = joblib.load("ENS_fitted_COVID.joblib")
		else:
			if model == 'SVM':
				st.sidebar.subheader('You selected COVID,  model: SVM')
				clf = joblib.load("SVM_fitted_COVID.joblib")
			else:
				if model == 'Random Forest':
					st.sidebar.subheader('You selected COVID,  model: Random Forest')
					clf = joblib.load("RF_fitted_COVID.joblib")
				else:
					if model == 'KNN':
						st.sidebar.subheader('You selected COVID,  model: KNN')
						clf = joblib.load("KNN_fitted_COVID.joblib")
					else:
						if model == 'Logistic Regression':		
							st.sidebar.subheader('You selected COVID,  model: Logistic Regression')
							clf = joblib.load("LR_fitted_COVID.joblib")
						else:
							st.sidebar.subheader('You selected COVID,  model: Naive Bayes')
							clf = joblib.load("NB_fitted_COVID.joblib")
						
	features2=pd.DataFrame.transpose(df) 
	#st.write("*Check the values below:  *")
	#st.dataframe(features2, width=150, height=880)

	
#prediction = clf.predict(df)
#prediction_proba = (clf.predict_proba(df))





#valore=1-prediction_proba[0,0]
#valore = str(valore)
st.write("")


if st.button("SUBMIT"):
	if errore==1:
		st.warning("Check the parameters before continuing")
	else:
		prediction = clf.predict(df)
		prediction_proba = (clf.predict_proba(df))
		valore=1-prediction_proba[0,0]
		valore = str(valore)
		if prediction[0]==0:
			Outcome="NEGATIVE"
			valueNeg=round((prediction_proba[0,0]*100),2)
			valuePos=round((prediction_proba[0,1]*100),2)
			valueNeg=str(valueNeg)
			valuePos=str(valuePos)
			st.subheader("  NEGATIVE:   "+ valueNeg + " %")
		else:
			Outcome="POSITIVE"
			valuePos=round((prediction_proba[0,1]*100),2)
			valueNeg=round((prediction_proba[0,0]*100),2)
			valuePos=str(valuePos)
			valueNeg=str(valueNeg)
			st.subheader("  POSITIVE:   "+valuePos+" %")
		#st.table(prediction_proba)
		x = prediction_proba[0]
	
		plt.figure(figsize=(1,1))
		label = ["Negativo","Positivo"]
		plt.pie(x, labels=label)
	
	
		components.html(
    """
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

    <body>

      <h1 class="h3 mb-3 font-weight-normal first">Results:</h1>
      <br>
      <div id = "res"> </div>
      <br>
      <br>
      <br>
	
	
			**if you are on mobile, scroll to the right or use desktop site version, to see the entire graph.**
      <h2 class="h3 mb-3 font-weight-normal first" >Notes:</h2>
      <p class="italic" align="justify" style="font-size:20px;"> The circle’s diameter represents the confidence interval of the prediction score; <br>
        Its position along the horizontal dimension the value of the prediction score. <br>
        The smaller the circle the higher the algorithm’s confidence
        on its output. <br> The closer to a response (pos/neg), the higher the model’s confidence on that response. <br> <br> <br> <br> </p>
      
	
  
 
 
 
 
 <style>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
 table {
  border-collapse: collapse;
  width: 40%;
  align: left
}

th, td {
  text-align: left;
  padding: 8px;
}

tr:nth-child(even){background-color: #f2f2f2}

th {
  background-color: lightgrey;
  color: black;
}
th {
  background-color: lightblue;
  color: black;
}
</style>
</head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<body>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<style>
.responsive {
  width: auto;
  height: auto;
}
</style>


<table>
  <tr>
    <th>OUTCOME:</th>
    <th>CONFIDENCE:</th>
    
  </tr>
  <tr>
    <td> POSITIVE </td>
    <td> """ + valuePos + "%" +"""</td>
	
    
  </tr>
   <tr>
    <td> NEGATIVE</td>
    <td> """ + valueNeg + "%" +"""</td>
	
    
  </tr>

</table>
 
 


 
 
 
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
      <script src="//code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
      <script src="//cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.6/umd/popper.min.js" integrity="sha384-wHAiFfRlMFy6i5SRaxvfOCifBUQy1xHdJ/yoi7FRNXMRBu5WHdZYu1hA6ZOblgut" crossorigin="anonymous"></script>
      <script src="//stackpath.bootstrapcdn.com/bootstrap/4.2.1/js/bootstrap.min.js" integrity="sha384-B0UglyR+jN6CkvvICOB2joaf5I4l3gm9GU6Hc1og6Ls7i6U/mkkaduKaBhlAXv9k" crossorigin="anonymous"></script>
      <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/d3/4.1.1/d3.js"></script>
      <script>
		
	  
		percentuale="""
		
		+ valore +"""
        var value_pos=percentuale
        var diametro = 3.92*Math.sqrt((value_pos*(1-value_pos))/100)   //diametro del cerchio
        var raggio = diametro/2
        var raggioInPixel = (raggio*60)/0.1   // dimensione del raggio epsressa in pixel,
                                          // ottenuta considerando la proporzione
                                          // 0.1:60=raggio:600
        //Creo elemento svg
        var svg = d3.select("#res")

			
          .append("svg")
          .attr("width", 1000)
		  


        // Create l'elemento scale
        var x = d3.scaleLinear()
          .domain([0, 1])            // L'asse ha valori compresi tra 0 e 1
          .range([50, 650]);         // Coordinate dove inserire l'asse

        // disegno l'asse x
        svg
	
          .append("g")
          .attr("transform", "translate(0,130)")   // Sposto l'asse
          .style("stroke-width", "0.3px")
          .call(d3.axisBottom(x).tickFormat(""));

        // disegno il rettangolo rosso
        svg
          .append("rect")
          .attr("x", x(0))            //Inizio del rettangolo rosso coincide con l'origine dell'asse
          .attr("y",10)
          .attr("height", 120 )
          .attr("width",  180 )
          .style("fill", "#ffcdca")
          .style("opacity", 0.6)

        //disegno il rettangolo grigio
        svg
          .append("rect")
          .attr("x", x(0.3))
          .attr("y",10)
          .attr("height", 120 )
          .attr("width",  240 )
          .style("fill", "#D0D0D0")
          .style("opacity", 1)

        //disegno il rettangolo blu
        svg
          .append("rect")
          .attr("x", x(0.7) )
          .attr("y",10)
          .attr("height", 120 )
          .attr("width",  180 )
          .style("fill", "#54caefff")
          .style("opacity", 0.7)

        // disegno il cerchio
        svg
          .append("circle")
          .attr("cx", x(1-value_pos))
          .attr("cy", 70)
          .attr("r", raggioInPixel)
          .attr("stroke", "black")
          .attr("stroke-width", "0.03%")
          .style("fill", "#D0D0D0")
          .style("opacity", 0.7)

        // Creo un rettangolo bianco, in modo da evitare che il cerchio fuoriesca
        // in caso la variabile assume valore pari (o prossimo) a 0.99
        svg
          .append("rect")
          .attr("x", 650)
          .attr("y",10)
          .attr("height", 120 )
          .attr("width",  20 )
          .style("fill", "#ffffff")
          .style("opacity", 1)

        // Come sopra, creo un rettangolo bianco, in modo da evitare che il cerchio
        // fuoriesca nel caso in cui la variabile assume valore pari (o prossimo)
        // a 0.01
        svg
          .append("rect")
          .attr("x", 30)
          .attr("y", 10)
          .attr("height", 120)
          .attr("width", 20)
          .style("fill", "#ffffff")
          .style("opacity",1)

        // creo la scritta pos
        svg
          .append("text")
          .attr("text-anchor", "end")
          .attr("x", 30)
          .attr("y", 75)
          .text("POS");

        //creo la scritta neg
        svg
          .append("text")
          .attr("text-anchor", "start")
          .attr("x", 670)
          .attr("y", 75)
          .text("NEG");
		 
      </script>
   </body>
    """,
	width=800,
	height=800,
	scrolling=True
	)
		
		
		
#lettura da csv
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.subheader("If you want to upload more instances:")
#st.write("Please remember to rename the first line with the name of the chosen template parameters")
#uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")






st.write("It is possible to change the dataset and model in the left sidebar.")
if dataset == 'CBC':
	numValori=21	
	
	if model == 'Ensemble':
		st.subheader('You selected CBC,  model: Ensemble')
		clf = joblib.load("ENS_fitted_CBC.joblib")
		#df = user_input_features_cbc()
	else:
		if model == 'SVM':
			st.subheader('You selected CBC,  model: SVM')
			
			clf = joblib.load("SVM_fitted_CBC.joblib")
				#df = user_input_features_cbc()
		else:
			if model == 'Random Forest':
				st.subheader('You selected CBC,  model: Random Forest')
				
				clf = joblib.load("RF_fitted_CBC.joblib")
					#df = user_input_features_cbc()
			else:
				if model == 'KNN':
					st.subheader('You selected CBC,  model: KNN')
					
					clf = joblib.load("KNN_fitted_CBC.joblib")
						#df = user_input_features_cbc()
				else:
					if model == 'Logistic Regression':		
						st.subheader('You selected CBC,  model: Logistic Regression')
						
						clf = joblib.load("LR_fitted_CBC.joblib")
							#df = user_input_features_cbc()
					else:
						st.subheader('You selected CBC,  model: Naive Bayes')
						
						clf = joblib.load("NB_fitted_CBC.joblib")
							#df = user_input_features_cbc()
	
	
else:
		
	numValori=34
	if model == 'Ensemble':
		st.subheader('You selected COVID,  model: Ensemble')
		clf = joblib.load("ENS_fitted_COVID.joblib")
	else:
		if model == 'SVM':
			st.subheader('You selected COVID,  model: SVM')
			clf = joblib.load("SVM_fitted_COVID.joblib")
		else:
			if model == 'Random Forest':
				st.subheader('You selected COVID,  model: Random Forest')
				clf = joblib.load("RF_fitted_COVID.joblib")
			else:
				if model == 'KNN':
					st.subheader('You selected COVID,  model: KNN')
					clf = joblib.load("KNN_fitted_COVID.joblib")
				else:
					if model == 'Logistic Regression':		
						st.subheader('You selected COVID,  model: Logistic Regression')
						clf = joblib.load("LR_fitted_COVID.joblib")
					else:
						st.subheader('You selected COVID,  model: Naive Bayes')
						clf = joblib.load("NB_fitted_COVID.joblib")

						
		


from enum import Enum
from io import BytesIO, StringIO
from typing import Union
st.write("The columns in the CSV file must be exactly in the order of the form. \n\r Values must be separeted by a delimiter")

file = st.file_uploader("You csv file must not have a header", type="csv")
						
import csv					
if st.button("COMPUTE"):
	if not file:
		st.warning("Please upload a valid .CSV file")
	else:
		
		data = pd.read_csv(file,sep=None,header=None)
		
		i=0
		
	
		
		#st.write(df)
		
		c=1
		for index, row in data.iterrows():
		#	i=i+1
			a=row.to_frame().T
			#st.write(a)
			st.write("ID "+str(c)+":")
			c=c+1
			if a.size>numValori:
				
				st.warning("Error: The number of values ​​in this file is greater than the selected template. \n\r Change dataset selection or change file.")
				break
			else:
				while a.size!=numValori:
					if a.size < numValori:
						b=a.size
						a[b]=None
				
				b=pd.DataFrame(a,index=[i])
			
			
			
			
					
				if dataset=="CBC":
					if (isinstance(b[0][i], float)):
						gender=float(b[0][i])
					else:
						gender=None
					
					
						
					if (isinstance(b[1][i], float)):
						age=float(b[1][i])
					else:
						age=None
					
					
					if (isinstance(b[2][i], float)):
						hct=float(b[2][i])
					else:
						hct=None
					
					if (isinstance(b[3][i], float)):
						hgb=float(b[3][i])
					else:
						hgb=None
					
					if (isinstance(b[4][i], float)):
						mch=float(b[4][i])
					else:
						mch=None
					
					if (isinstance(b[5][i], float)):
						mchc=float(b[5][i])
					else:
						mchc=None
					
					if (isinstance(b[6][i], float)):
						mcv=float(b[6][i])
					else:
						mcv=None
					
					if (isinstance(b[7][i], float)):
						rbc=float(b[7][i])
					else:
						rbc=None
					
					if (isinstance(b[8][i], float)):
						wbc=float(b[8][i])
					else:
						wbc=None
					
					if (isinstance(b[9][i], float)):
						plt1=float(b[9][i])
					else:
						plt1=None
						
					if (isinstance(b[10][i], float)):
						ne=float(b[10][i])
					else:
						ne=None
					
					if (isinstance(b[11][i], float)):
						ly=float(b[11][i])
					else:
						ly=None
					
					if (isinstance(b[12][i], float)):
						mo=float(b[12][i])
					else:
						mo=None
					
					if (isinstance(b[13][i], float)):
						eo=float(b[13][i])
					else:
						eo=None
					if (isinstance(b[14][i], float)):
						ba=float(b[14][i])
					else:
						ba=None
					if (isinstance(b[15][i], float)):
						net=float(b[15][i])
					else:
						net=None
					if (isinstance(b[16][i], float)):
						lyt=float(b[16][i])
					else:
						lyt=None
					if (isinstance(b[17][i], float)):
						mot=float(b[17][i])
					else:
						mot=None
					if (isinstance(b[18][i], float)):
						eot=float(b[18][i])
					else:
						eot=None
					if (isinstance(b[19][i], float)):
						bat=float(b[19][i])
					else:
						bat=None
					if (isinstance(b[20][i], float)):
						suspect=float(b[20][i])
					else:
						suspect=None
					
						
					data = {'Gender': gender,
					'Age': age,
					'HCT': hct,
					'HGB': hgb,
					'MCH': mch,
					'MCHC': mchc,
					'MCV': mcv,
					'RBC': rbc,
					'WBC': wbc,
					'PLT1': plt1,
					'NE': ne,
					'LY': ly,
					'MO': mo,
					'EO': eo,
					'BA': ba,
					'NET': net,
					'LYT': lyt,
					'MOT': mot,
					'EOT': eot,
					'BAT': bat,
					'Suspect': suspect}
				else:
					if (isinstance(b[0][i], float)):
						gender=float(b[0][i])
					else:
						gender=None
					
					
						
					if (isinstance(b[1][i], float)):
						age=float(b[1][i])
					else:
						age=None
					
					
					if (isinstance(b[2][i], float)):
						ca=float(b[2][i])
					else:
						ca=None
					
					if (isinstance(b[3][i], float)):
						ck=float(b[3][i])
					else:
						ck=None
					
					if (isinstance(b[4][i], float)):
						crea=float(b[4][i])
					else:
						crea=None
					
					if (isinstance(b[5][i], float)):
						alp=float(b[5][i])
					else:
						alp=None
					
					if (isinstance(b[6][i], float)):
						ggt=float(b[6][i])
					else:
						ggt=None
					
					if (isinstance(b[7][i], float)):
						glu=float(b[7][i])
					else:
						glu=None
					
					if (isinstance(b[8][i], float)):
						ast=float(b[8][i])
					else:
						ast=None
					
					if (isinstance(b[9][i], float)):
						alt=float(b[9][i])
					else:
						alt=None
						
					if (isinstance(b[10][i], float)):
						ldh=float(b[10][i])
					else:
						ldh=None
					
					if (isinstance(b[11][i], float)):
						pcr=float(b[11][i])
					else:
						pcr=None
					
					if (isinstance(b[12][i], float)):
						kal=float(b[12][i])
					else:
						kal=None
					
					if (isinstance(b[13][i], float)):
						nat=float(b[13][i])
					else:
						nat=None
					if (isinstance(b[14][i], float)):
						urea=float(b[14][i])
					else:
						urea=None
					if (isinstance(b[15][i], float)):
						wbc=float(b[15][i])
					else:
						wbc=None
					if (isinstance(b[16][i], float)):
						rbc=float(b[16][i])
					else:
						rbc=None
					if (isinstance(b[17][i], float)):
						hgb=float(b[17][i])
					else:
						hgb=None
					if (isinstance(b[18][i], float)):
						hct=float(b[18][i])
					else:
						hct=None
					if (isinstance(b[19][i], float)):
						mcv=float(b[19][i])
					else:
						mcv=None
						
					if (isinstance(b[20][i], float)):
						mch=float(b[20][i])
					else:
						mch=None
					if (isinstance(b[21][i], float)):
						mchc=float(b[21][i])
					else:
						mchc=None
					
					
						
					if (isinstance(b[22][i], float)):
						plt1=float(b[22][i])
					else:
						plt1=None
					
					
					if (isinstance(b[23][i], float)):
						ne=float(b[23][i])
					else:
						ne=None
					
					if (isinstance(b[24][i], float)):
						ly=float(b[24][i])
					else:
						ly=None
					
					if (isinstance(b[25][i], float)):
						mo=float(b[25][i])
					else:
						mo=None
					
					if (isinstance(b[26][i], float)):
						eo=float(b[26][i])
					else:
						eo=None
					
					if (isinstance(b[27][i], float)):
						ba=float(b[27][i])
					else:
						ba=None
					
					if (isinstance(b[28][i], float)):
						net=float(b[28][i])
					else:
						net=None
					
					if (isinstance(b[29][i], float)):
						lyt=float(b[29][i])
					else:
						lyt=None
					
					if (isinstance(b[30][i], float)):
						mot=float(b[30][i])
					else:
						mot=None
						
					if (isinstance(b[31][i], float)):
						eot=float(b[31][i])
					else:
						eot=None
					
					
					if (isinstance(b[32][i], float)):
						bat=float(b[32][i])
					else:
						bat=None
					
					if (isinstance(b[33][i], float)):
						suspect=float(b[33][i])
					else:
						suspect=None
					
					
					
					
					data = {'Gender': gender,
					'Age': age,
					'CA': ca,
					'CK': ck,
					'CREA': crea,
					'ALP': alp,
					'GGT': ggt,
					'GLU': glu,
					'AST': ast,
					'ALT': alt,
					'LDH': ldh,
					'PCR': pcr,
					'KAL': kal,
					'NAT': nat,
					'UREA': urea,
					'WBC': wbc,
					'RBC': rbc,
					'HGB': hgb,
					'HCT': hct,
					'MCV': mcv,
					'MCH': mch,
					'MCHC': mchc,
					'PLT1': plt1,
					'NE': ne,
					'LY': ly,
					'MO': mo,
					'EO': eo,
					'BA': ba,
					'NET': net,
					'LYT': lyt,
					'MOT': mot,
					'EOT': eot,
					'BAT': bat,
					'Suspect': suspect}
				df=pd.DataFrame(data,index=["Value"])
				st.dataframe(df)
				i=i+1
			
			
				
				prediction = clf.predict(df)
				prediction_proba = (clf.predict_proba(df))
				valore=1-prediction_proba[0,0]
				valore = str(valore)
				if prediction[0]==0:
					Outcome="NEGATIVE"
					valueNeg=round((prediction_proba[0,0]*100),2)
					valuePos=round((prediction_proba[0,1]*100),2)
					valueNeg=str(valueNeg)
					valuePos=str(valuePos)
					#st.subheader("  NEGATIVE:   "+ valueNeg + " %")
				else:
					Outcome="POSITIVE"
					valuePos=round((prediction_proba[0,1]*100),2)
					valueNeg=round((prediction_proba[0,0]*100),2)
					valuePos=str(valuePos)
					valueNeg=str(valueNeg)
					#st.subheader("  POSITIVE:   "+valuePos+" %")
								
			 
				components.html(""" <style>
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	 table {
	  border-collapse: collapse;
	  width: 40%;
	  align: left
	}

	th, td {
	  text-align: left;
	  padding: 8px;
	}

	tr:nth-child(even){background-color: #f2f2f2}

	th {
	  background-color: lightgrey;
	  color: black;
	}
	th {
	  background-color: lightblue;
	  color: black;
	}
	</style>
	</head>
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<body>
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<style>
	.responsive {
	  width: auto;
	  height: auto;
	}
	</style>


	<table>
	  <tr>
		<th>OUTCOME:</th>
		<th>CONFIDENCE:</th>
		
	  </tr>
	  <tr>
		<td> POSITIVE </td>
		<td> """ + valuePos + "%" +"""</td>
		
		
	  </tr>
	   <tr>
		<td> NEGATIVE</td>
		<td> """ + valueNeg + "%" +"""</td>
		
		
	  </tr>

	</table>""")



		
		
			
			









	
	#st.write("For technical details, please refer to: Brinati D, Campagner A, Ferrari D, Banfi G, Locatelli M, Cabitza F (2020) Detection of COVID-19 Infection from Routine Blood Exams with Machine Learning: a Feasibility Study. Journal of Medical Systems volume 44(135) ")
st.subheader("")
st.info(" For technical details, please refer to: Cabitza F, Campagner A, Ferrari D, Di Resta C, Ceriotti D, Sabetta E, Colombini A, De Vecchi E, Banfi G, Locatelli M, Carobene A (2020) Development, evaluation, and validation of machine learning models for COVID-19 detection based on routine blood tests. CCLM, doi: 10.1515/cclm-2020-1294.                \n\r  Open access: https://www.medrxiv.org/content/10.1101/2020.10.02.20205070v1Data available at: https://zenodo.org/record/4081318#.X4V3btD7TIX")
st.subheader("DISCLAIMER:")
st.warning("This calculator was created for research and testing purposes. Medical decisions must NOT be based on the results of this program, which cannot be considered a diagnostic tool. Although this program has been tested thoroughly, the accuracy of the information cannot be guaranteed and the authors shall not be liable for any claim, damages or other liability.")






			
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
			footer:after {
	content:'Credits: this tool has been in 2020 within the MUDI Lab @ University of Milano-Bicocca (https://www.mudilab.net/mudi/) on an idea by Federico Cabitza (PhD), with the scientific advice of Dr. Anna Carobene. CC BY-NC-ND 3.0 IT                                       Model: Andrea Campagner,2020. Web app: Giacomo Stoffa. Dataviz: Lorenzo Tomasoni. Conception: Federico Cabitza'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

