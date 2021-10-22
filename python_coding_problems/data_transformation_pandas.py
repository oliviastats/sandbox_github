import pandas as pd
import numpy as np
from scipy import stats

data_path = 'gs://mgb-loremipsum-dev-data/olivia/'

train = pd.read_csv(data_path +'train.csv')
test = pd.read_csv(data_path +'plan_data.csv')
valid = pd.read_csv(data_path + 'charges_data.csv')


def exploratory_analysis(personal_data, plan_data, charges_data):

    trimmed_av = round(stats.trim_mean(charges_data['monthlyCharges'], 0.1))
    charges_data['monthlyCharges'] = charges_data['monthlyCharges'].fillna(trimmed_av)

    charges_data['totalCharges'] = charges_data.apply(
                lambda row: row['monthlyCharges']*row['tenure'] if np.isnan(row['totalCharges']) else row['totalCharges'],
                axis=1
    )
    
    bins = np.array([0,24,48,60, np.inf])
    labels = ['group1', 'group2', 'group3', 'group4']
    charges_data['tenureBinned'] = pd.cut(charges_data['tenure'], bins=bins, labels=labels)

    total_customer = charges_data['churn'].count()
    churned_customers= charges_data['churn'][charges_data['churn'] == 'Yes'].count()
    churn_rate = round((churned_customers/total_customer)*100)

    charges_df_with_personal = charges_data.merge(personal_data, on=['customerID'], how='inner')
    charges_df_with_personal_and_plan = charges_df_with_personal.merge(plan_data, on=['customerID'], how='left')

    total_customer_merged = len(charges_df_with_personal_and_plan['customerID'].unique())
    customer_above_60 = charges_df_with_personal_and_plan['customerID'][charges_df_with_personal_and_plan['age'] > 60].count()
    perc_customers_more_than_60 = round((customer_above_60/total_customer_merged)*100)

    check_uniqueness = len(charges_df_with_personal_and_plan['customerID'].unique())
    dict_unique_counts = charges_df_with_personal_and_plan[['customerID', 'internetService']].groupby('internetService').agg('count').to_dict()['customerID']

    results = {'monthly_charges_mean' : trimmed_av,
               'charges_data_updated' : charges_data,
               'churn_pct' : churn_rate,
               'data_merged' : charges_df_with_personal_and_plan,
               'pct_age_above_60': perc_customers_more_than_60,
               'internet_service_counts': dict_unique_counts
    }