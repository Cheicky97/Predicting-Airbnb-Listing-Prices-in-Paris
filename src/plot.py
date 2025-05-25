import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway


class PlotData:
    """
    Ploting dataset
    """
    def dtypes(self, df_dtypes, figsize=(5, 5)):
        """
        Plot data types distribution as a pie chart.
        
        Parameters:
        - df_dtypes: pd.Series, typically from df.dtypes.value_counts()
        - figsize: tuple, size of the figure
        """
        plt.figure(figsize=figsize)
        c_palette = sns.color_palette('bright') 
        plt.pie(x=df_dtypes.values, labels=df_dtypes.index, colors=c_palette, autopct='%1.1f%%')
        plt.title("Distribution of Data Types")
        plt.show()

class TestDataset:
    """
    Perform test on dataset
    """
    def __init__(self, df:pd.DataFrame, target='price'):
        """
        df: (pd.DataFrame) hte dataset
        """
        self.data = df
        self.target = target
        self.data['is_missing']= self.data[target].isnull()
        self.categorical_cols = self.data.select_dtypes(include='object').columns
        self.numerical_cols = self.data.select_dtypes(include='number').columns.drop('price')
        self.cat_data = self.data[self.categorical_cols]
        self.num_data = self.data[self.numerical_cols]
        self.results = []

    def change_target(self, target):
        """
        Change target
        """
        if target != self.target:
            missing_data = self.data[target].isnull()
        else:
            missing_data = self.data['is_missing']
        return missing_data

    def numeric_vs_categoricals(self, target:str=None):
        """
        Function for numeric vs categorical (ANOVA)
        -----
        target:str, optional
            if test for a different attribute than the target.
        """
        missing_data = self.change_target(target)
        for col in self.categorical_cols:
            groups = [missing_data[self.cat_data[col]==val] for val in self.cat_data[col].unique()]
            try:
                _, p = f_oneway(*groups)
                self.results.append({'feature': col, 'p_value': p})
            except:
                continue

    def categorical_to_categorical(self, target=None):
        """
        Chi-square test for categorical vs categoricals 
        """
        missing_data = self.change_target(target)
        for col in self.categorical_cols:
            contengency = pd.crosstab(self.data[col], missing_data)
            if contengency.shape[0] < 2 or contengency[1] != 2:
                continue
            try:
                _, p, _, _ = chi2_contingency(contengency)
                self.results.append({'feature': col, 'p_value': p})
            except:
                continue

    def test_num_to_num(self, target:str=None):
        results = []
        missing_data = self.change_target(target)
        for col in self.numerical_cols:
            group1 = self.data[missing_data][col].dropna() #rows where price is missing
            group2 = self.data[~missing_data][col].dropna() #rows where price is not missing
            
            if len(group1) > 0 and len(group2) > 0:
                _, p = ttest_ind(group1, group2, equal_var=False)
                results.append({'feature': col, 'p_value': p})

    def gather_results(self, pthreshold=0.05):
        """

        """
        self.total_results = pd.DataFrame(self.results).sort_values('p_value')
        print(f'missingness dependent on {(self.total_results['p_value'] < pthreshold).mean()}% of attributes')