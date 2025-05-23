from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
import pandas as pd

def association_with_missing_target(df:pd.DataFrame, target_col:str, p_threshold:float=0.05):
    """
        df: pd.DataFrame, is the dataset
        target_col: str, is the target column name
        p_threshold: float, if calculated p_value < p_threshold, then there is association
                    between the feature and the target missingness.
                    The default value is 0.05
        --------
        Return DataFrame with the first column the features names, and the second 
                of boolean type indicating whether theris association or not
    """
    df['is_target_missing'] = df[target_col].isna()

    categorical_cols = df.select_dtypes(include='object').columns
    results = []

    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df['is_target_missing'])
        if contingency.shape[0] < 2 or contingency.shape[1] != 2:
            continue
        try:
            _, p, _, _ = chi2_contingency(contingency)
            results.append({'feature': col, f'p_value < {p_threshold}': p})
        except:
            continue

    association_to_cat = pd.DataFrame(results).sort_values('p_value')



    numerical_cols = df.select_dtypes(include='number').columns.drop(target_col)
    results = []

    for col in numerical_cols:
        group1 = df[df['is_price_missing']][col].dropna() #rows where target is missing
        group2 = df[~df['is_price_missing']][col].dropna() #rows where  target is not missing
        
        if len(group1) > 0 and len(group2) > 0:
            _, p = ttest_ind(group1, group2, equal_var=False)
            results.append({'feature': col, f'p_value < {p_threshold}': p})

    association_to_num = pd.DataFrame(results).sort_values('p_value')
    return pd.concat([association_to_cat, association_to_num])
