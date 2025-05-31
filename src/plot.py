import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from matplotlib.patches import Patch
from sklearn.feature_selection import mutual_info_regression

class PlotData:
    """
    Ploting dataset
    """
    @staticmethod
    def plot_skewness(data_skew, figsize=(8, 5)):
        """
        data: pd.DataFrame
        column:list or str,
            the column to be plotted
        """
        # Define colors based on skewness thresholds
        def color_map(val):
            if abs(val) < 0.5:
                return 'skyblue'      # Nearly symmetric
            elif abs(val) < 1:
                return 'orange'       # Moderate skew
            else:
                return 'crimson'      # High skew

        colors = [color_map(val) for val in data_skew]

        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x=data_skew.index, y=data_skew.values, palette=colors)
        plt.axhline(0, color='gray', linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Skewness")
        plt.title("Skewness of Numerical Features", fontsize=14)
        plt.tight_layout()

            # Optional: zoom in y-axis if outliers flatten the plot
        max_abs_skew = max(abs(data_skew.min()), abs(data_skew.max()))
        if max_abs_skew > 2.5:
            plt.ylim(-2.5, 2.5)

        # Add custom legend
        legend_elements = [
            Patch(facecolor='skyblue', edgecolor='black', label='|Skew| < 0.5 (Symmetric)'),
            Patch(facecolor='orange', edgecolor='black', label='0.5 ≤ |Skew| < 1 (Moderate)'),
            Patch(facecolor='crimson', edgecolor='black', label='|Skew| ≥ 1 (High)')
        ]
        plt.legend(handles=legend_elements, title='Skewness Level', loc=0)
        
        plt.show()

    @staticmethod
    def dtypes(df_dtypes, figsize=(5, 5)):
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

    @staticmethod
    def countplot__cat_var(df:pd.DataFrame, categorical_cols:list, figsize=(15,20), max_ncols:int=3, rotation:float=50, **kwargs):
        """
        plot the count plot for a list of categorical columns that columns names is indicated in the argument
        -------
        df: pandas.DataFrame, the dataset
        categorical_cols: list of categorical columns to be plotted 
        max_ncols: int, the maximum number of axes per row (of the figure)
        rotation: float, rotation of xticks
        **kwargs: dict, argument for setting more optin in the sns.countplot
        """
        num_of_axes = len(categorical_cols)
        ncols = min(max_ncols, num_of_axes)
        nrows = num_of_axes // ncols
        num_of_axes_is_multiple_ncols = num_of_axes%ncols
        nrows = nrows + 1 if num_of_axes_is_multiple_ncols else nrows

        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
        axs = axs.flatten()
        axes = axs[:num_of_axes]
        
        # remove all axes for which there is no column data to be plotted
        if num_of_axes_is_multiple_ncols:
            for i in range(num_of_axes, len(axs)):
                fig.delaxes(axs[i])
        


        for ax, col in list(zip(axes, categorical_cols)):
            sns.countplot(data=df, x=col, ax=ax, **kwargs)
            ax.set_title(col)
            ax.tick_params(axis='x', rotation=rotation)
        plt.tight_layout()
        plt.show()


        
        
class TestDataset:
    """
    Perform test on dataset
    """
    def __init__(self, df:pd.DataFrame, target:str, drop_num:list=None, for_missing=True):
        """
        ----------
        df: (pd.DataFrame) the dataset
        target: str, target variable whose missingness is to analysed
        """
        #drop_cat: list of categorical type columns to be drop from the analysis
        #drop_num: list of numerical type columns to be drop from the analysis
        self.data = df.copy()
        self.target = target
        self.for_missing = for_missing
        self.targeted_data = self.data[target].isnull() if for_missing else self.data[target]
        self.categorical_cols = self.data.select_dtypes(include='object').columns
        if drop_num is not None:
            self.numerical_cols = self.data.select_dtypes(include=['int', 'float']).columns.drop(drop_num)
        else:
            self.numerical_cols = self.data.select_dtypes(include=['int', 'float']).columns
        self.data_desc_T = self.data[self.numerical_cols].describe().T
        self.numerical_cols = self.numerical_cols.drop(target)
        self.cat_data = self.data[self.categorical_cols]
        self.num_data = self.data[self.numerical_cols]
        self.results = []

    def change_target(self, target):
        """
        Change target
        """
        if target != None:
            targeted_data = self.data[target].isnull() if self.for_missing else self.data[target]
        else:
            targeted_data = self.targeted_data
        return targeted_data


    def numeric_vs_categoricals(self, target:str=None, return_p_value=False, list_cols=None):
        """
        Function for numeric vs categorical (ANOVA)
        -----
        target:str, optional
            if test for a different attribute than the target.
        """
        p_value = []
        targeted_data = self.change_target(target)
        cols = list_cols if list_cols is not None else self.categorical_cols
        for col in cols:
            groups = [group["price"].dropna().values for _, group in self.data.groupby(col)]
            try:
                _, p = f_oneway(*groups)
                self.results.append({'feature': col, 'p_value': p})
                p_value.append({'feature': col, 'p_value': p})
            except:
                continue
        if return_p_value:
            return pd.DataFrame(p_value).sort_values('p_value')

    def make_mi_scores(self, target):
        """
        Calculate mutual information between features and target variable
        ---------
        target: str, is the target variable name in the dataset
        """
        X = self.data.copy()
        y = X.pop(target)
        for col in self.data.select_dtypes("object").columns:
            X[col], _ = X[col].factorize()
        
        discrete_features = X.dtypes == int
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    def categorical_vs_categorical(self, target:str=None):
        """
        Chi-square test for categorical vs categoricals
        -------
        target:str, optional
            if test for a different attribute than the target.
        type_target:str, optional
            indicate the data type of the target column : either 'numeric' or 'categorical'.
        ------
        return
            None
            p_value are append to the variable results which is an attribute of this class
        """
        targeted_data = self.change_target(target)
        for col in self.categorical_cols:
            contengency = pd.crosstab(self.cat_data[col], targeted_data)
            if contengency.shape[0] < 2 or contengency.shape[1] != 2:
                continue
            try:
                _, p, _, _ = chi2_contingency(contengency)
                self.results.append({'feature': col, 'p_value': p})
            except:
                continue

    def numerical_vs_numerical(self, target:str=None):
        """
        t-test for numerical vs numericals
        --------
        target:str, optional
            if test for a different attribute than the target.
        ------
        return
            None
            p_value are append to the variable results which is an attribute of this class
        """
        targeted_data = self.change_target(target)
        for col in self.numerical_cols:
            group1 = self.data[targeted_data][col].dropna() #rows where price is missing
            group2 = self.data[~targeted_data][col].dropna() #rows where price is not missing
            
            if len(group1) > 0 and len(group2) > 0:
                _, p = ttest_ind(group1, group2, equal_var=False)
                self.results.append({'feature': col, 'p_value': p})

    def gather_results(self, pthreshold=0.05):
        """
        put results into a pd.DataFrame
        ------
        pthreshold:float, the threshold such that if p_value < pthreshold there is significant association
                    between corresponding attribute and the target.
        """
        self.total_results = pd.DataFrame(self.results).sort_values('p_value')
        print(f'missingness dependent on {(self.total_results['p_value'] < pthreshold).mean() * 100:2.0f}% of attributes')

    def detect_outliers_iqr(self):
        """
        Detect oulier in numerical features
        """
        q1 = self.data_desc_T['25%'].values
        q3 = self.data_desc_T['75%'].values
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        mask = self.data_desc_T['max'].values > upper_bound
        features = self.data_desc_T.index[mask]
        self.data_desc_T['outlier']=mask
        return features