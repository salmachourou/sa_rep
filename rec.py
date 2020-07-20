import pandas as pd
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt
dbase = pd.read_csv("final_data.csv")
dbase.rename(columns=lambda x: x.replace('NomPrenom', 'nom_prenom'), inplace=True)
print(dbase.shape)
np.radians(90)
#dbase.sort_values(['rating'], ascending=[False],inplace=True)
def _calcul_score(coach_price,min_price,long1,lat1):
    
    long2=16.3061# exemple manuel  a longterme via gps
    lat2=49.661
    coef1=0.5
    coef2=0.5
    score = coef1 * ( coach_price - min_price ) 
    calcul_distance(long1,lat1,long2,lat2)
    return score

def calcul_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
    
    
l_coach=pd.DataFrame(columns=['id','rating','score']) 

#♥ x=dbase.loc[i,'prix']
    
#print (l_coach.shape)
l_coach['id']=dbase['numero'].copy()
l_coach['rating']=dbase['rating'].copy()

for i in range (30):
 l_coach.loc[i,'score']=_calcul_score(dbase.loc[i,'prix'],1,dbase.loc[i,'langitude'],dbase.loc[i,'latitude'])
#l_coach.sort_values(['score'], ascending=[False],inplace=True)
 
for i in range(5):
    j=0
    while(l_coach.loc[i,'id']==dbase.loc[j,'numero']):
     j=j+1
    print(dbase.loc[i,'nom_prenom']) #lenna lezmni n affichi l profil kol mech ken colone
