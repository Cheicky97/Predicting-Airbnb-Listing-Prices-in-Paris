import matplotlib.pyplot as plt
#import pandas as pd
import seaborn as sns

class PlotData:
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

