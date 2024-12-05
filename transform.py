import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder

Month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
DayOfWeek_map = {"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
DayOfWeekClaimed_map = DayOfWeek_map.copy()
MonthClaimed_map = Month_map.copy()
VehiclePrice_map = {'less than 20000':0,'20000 to 29000':1,'30000 to 39000':2,'40000 to 59000':3,'60000 to 69000':4,
                     'more than 69000':5}
Days_Policy_Accident_map = {'none':0,'1 to 7':1,'8 to 15':2,'15 to 30':3,'more than 30':4}
Days_Policy_Claim_map = {'none':0,'8 to 15':1,'15 to 30':2,'more than 30':3}
PastNumberOfClaims_map = {'none':0,'1':1,'2 to 4':2,'more than 4':3}
AgeOfVehicle_map = {'new':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,\
                   '6 years':6,'7 years':7,'more than 7':8}
NumberOfSuppliments_map = {'none':0, 'more than 5':4, '1 to 2':1, '3 to 5':3}
AddressChange_Claim_map = {'no change':0,'under 6 months':1,'1 year':2,'2 to 3 years':3,'4 to 8 years':4}
NumberOfCars_map={'1 vehicle':1, '2 vehicles':2, '3 to 4':3, '5 to 8':4, 'more than 8':5}


def process_df(df,new=False):
    x = df.drop(columns=['PolicyNumber','AgeOfPolicyHolder','BasePolicy','FraudFound_P'],axis=1)
    y = df['FraudFound_P']
    num_cols = x.select_dtypes('number').columns
    
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    df[num_cols] = scaler.transform(x[num_cols])
    
    dummy_cols = ['Make','AccidentArea','Sex','MaritalStatus','Fault','PolicyType','VehicleCategory',\
              'PoliceReportFiled','WitnessPresent','AgentType']
    
    ordinal_cols = ['Month', 'DayOfWeek','DayOfWeekClaimed','MonthClaimed','VehiclePrice',\
                'Days_Policy_Accident','Days_Policy_Claim','PastNumberOfClaims','AgeOfVehicle','NumberOfSuppliments',\
                'AddressChange_Claim','NumberOfCars']
    x = pd.get_dummies(data=x,columns=dummy_cols)
    
    for col in ordinal_cols:
        map_name = globals()[col+'_map']
        x[col] = x[col].map(map_name)
    missing_index = x[x.isnull().any(axis=1)].index
    x = x.drop(missing_index)
    y = y.drop(missing_index)
    if new:
        return x,y
    else:
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
        return x_train,x_test,y_train,y_test