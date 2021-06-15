###########################
#
#         WRAPPER 
#
###########################

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
import numpy as np

def load_dataset(path, data):
    '''Imports and prepares data set using one of helper functions'''
    
    if data == 'taiwan':
        df = load_taiwan(path)
        
    if data == 'german':
        df = load_german(path)
        
    if data == 'uk':
        df = load_uk(path)  
        
    if data == 'bene':
        df = load_bene(path)  
        
    if data == 'homecredit':
        df = load_homecredit(path)  
        
    if data == 'gmsc':
        df = load_gmsc(path)          
        
    if data == 'pkdd':
        df = load_pkdd(path)          

    return df



###########################
#
#         TAIWAN
#
###########################

def load_taiwan(filepath):
    '''Imports and prepares taiwan data set'''
    
    # read CSV
    df = pd.read_csv(filepath, sep = ',', na_values = [])
    
    # prepare features
    df = df.rename(columns = {'default.payment.next.month': 'TARGET'})
    del df['ID']
    df['AGE'] = df['AGE'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    df['CREDIT_AMNT'] = df['BILL_AMT1'] - df['PAY_AMT1']
    df = df.loc[df['CREDIT_AMNT'] > 0].reset_index(drop = True)
    df = df.sample(frac = 1).reset_index(drop = True)

    # feature lists
    XD_features = ["LIMIT_BAL", "SEX","EDUCATION","MARRIAGE","AGE",
                "PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1",
                "BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
                "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5",
                "PAY_AMT6", "CREDIT_AMNT"]
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', "PAY_0",
                            "PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    
    # protected attrbute
    privileged_class        = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}
        
    # define factor encoding
    def default_preprocessing(df):
        
        def label_sex(x):
            if x == 1:
                return 'Male'
            elif x == 2:
                return 'Female'
            else:
                return 'NA'

        def label_education(x):
            if x == 1:
                return 'graduate_school'
            elif x == 2:
                return 'university'
            elif x == 3:
                return 'high_school'
            elif x == 4:
                return 'others'
            elif x == 5:
                return 'others'
            elif x == 6:
                return 'others'
            else:
                return 'others'

        def label_marriage(x):
            if x == 1:
                return 'married'
            elif x == 2:
                return 'single'
            elif x == 3:
                return 'others'
            else:
                return 'others'
        
        def label_pay(x):
            if x in [-2,-1]:
                return 0
            else:
                return x
    
        # apply emcoding
        df['SEX']       = df['SEX'].apply(lambda x: label_sex(x))
        df['EDUCATION'] = df['EDUCATION'].apply(lambda x: label_education(x))
        df['MARRIAGE']  = df['MARRIAGE'].apply(lambda x: label_marriage(x))
        
        # pay features
        pay_col = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
        for p in pay_col:
            df[p] = df[p].apply(lambda x: label_pay(x))
        
        # target encoding
        status_map   = {0: 1.0, 1: 2.0}
        df['TARGET'] = df['TARGET'].replace(status_map)
                
        return df
    
    # convert DF
    df_standard = StandardDataset(
        df                        = df,
        label_name                = Y_features[0],
        favorable_classes         = [1],
        protected_attribute_names = D_features,
        privileged_classes        = [privileged_class["AGE"]],
        instance_weights_name     = None,
        categorical_features      = categorical_features,
        features_to_keep          = X_features + Y_features + D_features,
        metadata                  = {'label_maps':               [{1.0: 'Good', 2.0: 'Bad'}],
                                     'protected_attribute_maps': [protected_attribute_map]},
        custom_preprocessing      = default_preprocessing)
    
    return df_standard



###########################
#
#         GERMAN
#
###########################

def load_german(filepath):
    '''Imports and prepares german data set'''
    
    # read CSV
    df = pd.read_csv(filepath, sep=',', na_values=[], index_col=0)
    
    # prepare features
    df['AGE'] = df['age'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    del df['age']
    df['CREDIT_AMNT'] = df['amount']
    del df['amount']
    df['TARGET'] = df['BAD']
    del df['BAD']

    # feature lists
    XD_features = ["account_status", "duration","credit_history","purpose","CREDIT_AMNT","savings",
                "employment","installment_rate","status_gender","guarantors","resident_since","property",
                "AGE","other_plans","housing","num_credits","job",
                "people_maintenance","phone","foreign"]
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ['account_status', 'credit_history', 'purpose', "savings",
                            "employment","status_gender","guarantors","property","other_plans",
                           'housing', 'job', 'phone', 'foreign']

    # protected attribtue
    privileged_class        = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}

    # target encoding
    status_map   = {'GOOD': 1.0, 'BAD': 2.0}
    df['TARGET'] = df['TARGET'].replace(status_map)
    
    # convert DF
    df_standard = StandardDataset(
        df                        = df,
        label_name                = Y_features[0],
        favorable_classes         = [1],
        protected_attribute_names = D_features,
        privileged_classes        = [privileged_class["AGE"]],
        instance_weights_name     = None,
        categorical_features      = categorical_features,
        features_to_keep          = X_features + Y_features + D_features,
        metadata                  = {'label_maps':               [{1.0: 'Good', 2.0: 'Bad'}],
                                     'protected_attribute_maps': [protected_attribute_map]})
    
    return df_standard



###########################
#
#         UK
#
###########################

def load_uk(filepath):
    '''Imports and prepares uk data set'''
    
    # read CSV
    df = pd.read_csv(filepath, sep = ',', na_values = [])
    
    # prepare features
    df['AGE'] = df['Age'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    del df['Age']
    df['CREDIT_AMNT'] = df['Amount']
    del df['Amount']
    df['TARGET'] = df['BAD']
    del df['BAD']

    # feature lists
    XD_features = ["AGE", "CREDIT_AMNT", "Curadd", "Curremp", "Custgend", "Depchild", "Freqpaid", "Homephon",
                   "Insprem", "Loantype", "Marstat", "Term", "Homeowns", "Purpose"] 
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ['Custgend', 'Freqpaid', 'Homephon', "Loantype",
                            "Marstat","Homeowns","Purpose"]

    # protected attribute
    privileged_class        = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}

    # target encoding
    status_map   = {'GOOD': 1.0, 'BAD': 2.0}
    df['TARGET'] = df['TARGET'].replace(status_map)
    
    # convert DF
    df_standard = StandardDataset(
        df                        = df,
        label_name                = Y_features[0],
        favorable_classes         = [1],
        protected_attribute_names = D_features,
        privileged_classes        = [privileged_class["AGE"]],
        instance_weights_name     = None,
        categorical_features      = categorical_features,
        features_to_keep          = X_features + Y_features + D_features,
        metadata                  = {'label_maps':               [{1.0: 'Good', 2.0: 'Bad'}],
                                     'protected_attribute_maps': [protected_attribute_map]})
    
    return df_standard



###########################
#
#         BENE
#
###########################

def load_bene(filepath):
    '''Imports and prepares bene data set'''
    
    # read CSV
    df = pd.read_csv(filepath, sep = ',', na_values = [])
    
    # prepare features
    df['AGE'] = df['DGEBOA11'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    del df['DGEBOA11'], df['bin_DGEBOA11']
    df['CREDIT_AMNT'] = df['BLENIA1']
    del df['BLENIA1']
    df['TARGET'] = df['BAD']
    del df['BAD']

    # feature lists
    XD_features = ["AGE", "CREDIT_AMNT",
                    "BFACTA1",      "PLASTA1",     "PDUURA2",      "BMENSA1",      "BSPARA11",   
                    "BUITGA21",     "BINKOA11",     "DBEGIA21",     "DVERBA11",         "DVERBA21",    
                    "DCLIEA11",     "DLLENA11",     "ACONTCV1",     "ACONTTM1",     "ACONTHY1",     "APERSA11",   
                    "bin_BFACTA1",  "bin_PLASTA1",  "bin_PDUURA2",  "bin_CDOELA2",  "bin_CGEBRA1", 
                    "bin_BMENSA1",  "bin_BSPARA11", "bin_BINKOA11",  "bin_CPRIVA11", "bin_CBURGA11",
                    "bin_DVERBA21", "bin_CTROSA11", "bin_CEIGEA11", "bin_DCLIEA11", "bin_DLLENA11", "bin_ACONTCV1",
                    "bin_ACONTHY1", "bin_CECOTA11", "bin_CVOLTA11", "bin_CAANSA11"]
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ["bin_BFACTA1",  "bin_PLASTA1",  "bin_PDUURA2",  "bin_CDOELA2",  "bin_CGEBRA1", 
                    "bin_BMENSA1",  "bin_BSPARA11", "bin_BINKOA11",  "bin_CPRIVA11", "bin_CBURGA11",
                    "bin_DVERBA21", "bin_CTROSA11", "bin_CEIGEA11", "bin_DCLIEA11", "bin_DLLENA11", "bin_ACONTCV1",
                    "bin_ACONTHY1", "bin_CECOTA11", "bin_CVOLTA11", "bin_CAANSA11"]

    # protected attribute
    privileged_class        = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}

    # target encoding
    status_map   = {'GOOD': 1.0, 'BAD': 2.0}
    df['TARGET'] = df['TARGET'].replace(status_map)
    
    # convert DF
    df_standard = StandardDataset(
        df                        = df,
        label_name                = Y_features[0],
        favorable_classes         = [1],
        protected_attribute_names = D_features,
        privileged_classes        = [privileged_class["AGE"]],
        instance_weights_name     = None,
        categorical_features      = categorical_features,
        features_to_keep          = X_features + Y_features + D_features,
        metadata                  = {'label_maps':               [{1.0: 'Good', 2.0: 'Bad'}],
                                     'protected_attribute_maps': [protected_attribute_map]})
    
    return df_standard



###########################
#
#         HOMECREDIT
#
###########################

def load_homecredit(filepath):
    '''Imports and prepares homecredit data set'''
    
    # read CSV
    df = pd.read_csv(filepath, sep = ',', na_values = [])
    
    # prepare features
    df['AGE'] = df['app_DAYS_BIRTH'] / 12
    df['AGE'] = df['AGE'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    del df['app_DAYS_BIRTH']
    df['CREDIT_AMNT'] = df['app_AMT_CREDIT']
    del df['app_AMT_CREDIT']
    df['TARGET'] = df['BAD']
    del df['BAD']

    # feature lists
    XD_features = ["AGE", "CREDIT_AMNT",
                   'app_CNT_CHILDREN', 'app_AMT_INCOME_TOTAL', 'app_AMT_ANNUITY', 
                   'app_AMT_GOODS_PRICE', 'app_REGION_POPULATION_RELATIVE', 'app_DAYS_EMPLOYED',
                   'app_DAYS_REGISTRATION', 'app_DAYS_ID_PUBLISH', 'app_OWN_CAR_AGE', 'app_FLAG_MOBIL', 'app_FLAG_EMP_PHONE',
                   'app_FLAG_WORK_PHONE', 'app_FLAG_CONT_MOBILE', 'app_FLAG_PHONE', 'app_FLAG_EMAIL', 'app_CNT_FAM_MEMBERS',
                   'app_REGION_RATING_CLIENT', 'app_REGION_RATING_CLIENT_W_CITY', 'app_HOUR_APPR_PROCESS_START',
                   'app_REG_REGION_NOT_LIVE_REGION', 'app_REG_REGION_NOT_WORK_REGION', 'app_LIVE_REGION_NOT_WORK_REGION', 
                   'app_REG_CITY_NOT_LIVE_CITY', 'app_REG_CITY_NOT_WORK_CITY', 'app_LIVE_CITY_NOT_WORK_CITY', 'app_EXT_SOURCE_1',
                   'app_EXT_SOURCE_2', 'app_EXT_SOURCE_3', 'app_APARTMENTS_AVG', 'app_BASEMENTAREA_AVG', 
                   'app_YEARS_BEGINEXPLUATATION_AVG', 'app_YEARS_BUILD_AVG', 'app_COMMONAREA_AVG', 'app_ELEVATORS_AVG', 
                   'app_ENTRANCES_AVG', 'app_FLOORSMAX_AVG', 'app_FLOORSMIN_AVG', 'app_LANDAREA_AVG', 'app_LIVINGAPARTMENTS_AVG',
                   'app_LIVINGAREA_AVG', 'app_NONLIVINGAPARTMENTS_AVG', 'app_NONLIVINGAREA_AVG', 'app_YEARS_BUILD_MODE', 
                   'app_OBS_30_CNT_SOCIAL_CIRCLE', 'app_DEF_30_CNT_SOCIAL_CIRCLE', 'app_OBS_60_CNT_SOCIAL_CIRCLE', 
                   'app_DEF_60_CNT_SOCIAL_CIRCLE', 'app_DAYS_LAST_PHONE_CHANGE', 'app_FLAG_DOCUMENT_2', 'app_FLAG_DOCUMENT_3', 
                   'app_FLAG_DOCUMENT_4', 'app_FLAG_DOCUMENT_5', 'app_FLAG_DOCUMENT_6', 'app_FLAG_DOCUMENT_7', 
                   'app_FLAG_DOCUMENT_8', 'app_FLAG_DOCUMENT_9', 'app_FLAG_DOCUMENT_10', 'app_FLAG_DOCUMENT_11', 
                   'app_FLAG_DOCUMENT_12', 'app_FLAG_DOCUMENT_13', 'app_FLAG_DOCUMENT_14', 'app_FLAG_DOCUMENT_15', 
                   'app_FLAG_DOCUMENT_16', 'app_FLAG_DOCUMENT_17', 'app_FLAG_DOCUMENT_18', 'app_FLAG_DOCUMENT_19', 
                   'app_FLAG_DOCUMENT_20', 'app_FLAG_DOCUMENT_21', 'app_AMT_REQ_CREDIT_BUREAU_HOUR', 
                   'app_AMT_REQ_CREDIT_BUREAU_DAY', 'app_AMT_REQ_CREDIT_BUREAU_WEEK', 'app_AMT_REQ_CREDIT_BUREAU_MON', 
                   'app_AMT_REQ_CREDIT_BUREAU_QRT', 'app_AMT_REQ_CREDIT_BUREAU_YEAR', 'app_CREDIT_BY_INCOME', 
                   'app_ANNUITY_BY_INCOME', 'app_GOODS_PRICE_BY_INCOME', 'app_INCOME_PER_PERSON', 'app_PERCENT_WORKED', 
                   'app_CNT_ADULTS', 'app_CHILDREN_RATIO', 'app_ANNUITY.LENGTH', 'app_EXT_SOURCE_MEAN', 'app_NUM_EXT_SOURCES', 
                   'app_NUM_DOCUMENTS', 'app_OWN_CAR_AGE_RATIO', 'app_DAYS_ID_PUBLISHED_RATIO', 'app_DAYS_REGISTRATION_RATIO', 
                   'app_DAYS_LAST_PHONE_CHANGE_RATIO']
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = []

    # protected attribute
    privileged_class        = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}

    # target encoding
    status_map   = {'GOOD': 1.0, 'BAD': 2.0}
    df['TARGET'] = df['TARGET'].replace(status_map)
    
    # convert DF
    df_standard = StandardDataset(
        df                        = df,
        label_name                = Y_features[0],
        favorable_classes         = [1],
        protected_attribute_names = D_features,
        privileged_classes        = [privileged_class["AGE"]],
        instance_weights_name     = None,
        features_to_keep          = X_features + Y_features + D_features,
        metadata                  = {'label_maps':               [{1.0: 'Good', 2.0: 'Bad'}],
                                     'protected_attribute_maps': [protected_attribute_map]})
    
    return df_standard



###########################
#
#        GMSC
#
###########################

def load_gmsc(filepath):
    '''Imports and prepares gmsc data set'''
    
    # read CSV
    df = pd.read_csv(filepath, sep = ',', na_values = [])
    
    # prepare features
    df['AGE'] = df['age'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    del df['age']
    df['CREDIT_AMNT'] = df['CREDIT_AMNT']
    df['TARGET'] = df['BAD']
    del df['BAD']

    # feature lists
    XD_features = ["AGE", "CREDIT_AMNT",
                  'UnknownNumberOfDependents', 'UnknownMonthlyIncome', 'NoDependents', 'NoIncome', 'ZeroDebtRatio',
                   'UnknownIncomeDebtRatio', 'WeirdRevolvingUtilization', 'ZeroRevolvingUtilization', 'Log.Debt',
                   'RevolvingLines', 'HasRevolvingLines', 'HasRealEstateLoans', 'HasMultipleRealEstateLoans', 'EligibleSS',
                   'DTIOver33', 'DTIOver43', 'DisposableIncome', 'RevolvingToRealEstate',
                   'NumberOfTime30.59DaysPastDueNotWorseLarge', 'NumberOfTime30.59DaysPastDueNotWorse96', 
                   'Never30.59DaysPastDueNotWorse', 'Never60.89DaysPastDueNotWorse', 'Never90DaysLate', 'IncomeDivBy10',
                   'IncomeDivBy100', 'IncomeDivBy1000', 'IncomeDivBy5000', 'Weird0999Utilization', 'FullUtilization', 
                   'ExcessUtilization', 'NumberOfTime30.89DaysPastDueNotWorse', 'Never30.89DaysPastDueNotWorse', 'NeverPastDue',
                   'Log.RevolvingUtilizationTimesLines', 'Log.RevolvingUtilizationOfUnsecuredLines', 'DelinquenciesPerLine', 
                   'MajorDelinquenciesPerLine', 'MinorDelinquenciesPerLine', 'DelinquenciesPerRevolvingLine', 
                   'MajorDelinquenciesPerRevolvingLine', 'MinorDelinquenciesPerRevolvingLine', 'Log.DebtPerLine', 
                   'Log.DebtPerRealEstateLine', 'Log.DebtPerPerson', 'RevolvingLinesPerPerson', 'RealEstateLoansPerPerson', 
                   'YearsOfAgePerDependent', 'Log.MonthlyIncome', 'Log.IncomePerPerson', 'Log.NumberOfTimesPastDue', 
                   'Log.NumberOfTimes90DaysLate', 'Log.NumberOfTime30.59DaysPastDueNotWorse', 
                   'Log.NumberOfTime60.89DaysPastDueNotWorse', 'Log.Ratio90to30.59DaysLate', 'Log.Ratio90to60.89DaysLate', 
                   'AnyOpenCreditLinesOrLoans', 'Log.NumberOfOpenCreditLinesAndLoans', 
                   'Log.NumberOfOpenCreditLinesAndLoansPerPerson', 'Has.Dependents', 'Log.HouseholdSize', 'Log.DebtRatio', 
                   'Log.UnknownIncomeDebtRatio', 'Log.UnknownIncomeDebtRatioPerPerson', 'Log.UnknownIncomeDebtRatioPerLine', 
                   'Log.UnknownIncomeDebtRatioPerRealEstateLine', 'Log.NumberRealEstateLoansOrLines']
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = []

    # protected attribute
    privileged_class        = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}

    # target encoding
    status_map   = {'GOOD': 1.0, 'BAD': 2.0}
    df['TARGET'] = df['TARGET'].replace(status_map)
    
    # convert DF
    df_standard = StandardDataset(
        df                        = df,
        label_name                = Y_features[0],
        favorable_classes         = [1],
        protected_attribute_names = D_features,
        privileged_classes        = [privileged_class["AGE"]],
        instance_weights_name     = None,
        features_to_keep          = X_features + Y_features + D_features,
        metadata                  = {'label_maps':               [{1.0: 'Good', 2.0: 'Bad'}],
                                     'protected_attribute_maps': [protected_attribute_map]})
    
    return df_standard



###########################
#
#         PKDD
#
###########################

def load_pkdd(filepath):
    '''Imports and prepares pkdd data set'''
    
    # read CSV
    df = pd.read_csv(filepath, sep = ',', na_values = [])
    
    # prepare features
    df = df.sample(frac = 1).reset_index(drop = True)
    df['AGE'] = df['AGE'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    df['CREDIT_AMNT'] = df['CREDIT_AMNT']
    df['TARGET'] = df['BAD']
    del df['BAD']

    # feature lists
    XD_features = ["AGE", "CREDIT_AMNT",
                   'PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'SEX', 'MARITAL_STATUS', 'QUANT_DEPENDANTS', 'STATE_OF_BIRTH', 
                   'NATIONALITY', 'RESIDENTIAL_STATE', 'FLAG_RESIDENCIAL_PHONE', 'RESIDENCE_TYPE', 'MONTHS_IN_RESIDENCE', 
                   'FLAG_EMAIL', 'PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'FLAG_VISA', 'FLAG_MASTERCARD', 
                   'QUANT_SPECIAL_BANKING_ACCOUNTS', 'QUANT_CARS', 'COMPANY', 'PROFESSIONAL_STATE', 'FLAG_PROFESSIONAL_PHONE',
                   'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE', 'EDUCATION_LEVEL2', 'PRODUCT']
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ['PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'SEX', 'MARITAL_STATUS', 'STATE_OF_BIRTH',
                            'NATIONALITY', 'RESIDENTIAL_STATE', 'RESIDENCE_TYPE', 'PROFESSIONAL_STATE', 'PROFESSION_CODE',
                            'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE', 'EDUCATION_LEVEL2', 'PRODUCT']

    # protected attribute
    privileged_class        = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}

    # target encoding
    status_map   = {'GOOD': 1.0, 'BAD': 2.0}
    df['TARGET'] = df['TARGET'].replace(status_map)
    
    # convert DF
    df_standard = StandardDataset(
        df                        = df,
        label_name                = Y_features[0],
        favorable_classes         = [1],
        protected_attribute_names = D_features,
        privileged_classes        = [privileged_class["AGE"]],
        instance_weights_name     = None,
        categorical_features      = categorical_features,
        features_to_keep          = X_features + Y_features + D_features,
        metadata                  = {'label_maps':               [{1.0: 'Good', 2.0: 'Bad'}],
                                     'protected_attribute_maps': [protected_attribute_map]})
    
    return df_standard