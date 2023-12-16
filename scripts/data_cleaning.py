import numpy as np
from data_summary import DataSummarizer;


class DataCleaner:
    """
    class that handles data cleaning.
    """
    def __init__(self) -> None:
        self.summar = DataSummarizer() 


    def fill_outliers_mean(self, df, cols):
        df_temp = df.copy(deep=True)
        for col in cols:
            ''' Detection '''
            # IQR
            Q1 = df_temp[col].quantile(0.25)

            Q3 = df_temp[col].quantile(0.75)
            IQR = Q3 - Q1

            length=df_temp.shape[0]
            for index in range(length):
                if(df_temp.loc[index,col] >= (Q3+1.5*IQR)):
                    df_temp.loc[index,col] = np.nan

            ''' filling the Outliers '''
            df_temp = self.fill_missing_by_median(df_temp, cols)

        return df_temp


    def removeOutliers(self, df,cols):
        df_temp = df.copy(deep=True)
        for col in cols:
            ''' Detection '''
            # IQR
            Q1 = df_temp[col].quantile(0.25)

            Q3 = df_temp[col].quantile(0.75)
            IQR = Q3 - Q1
            rm_lis = []
            length=df_temp.shape[0]
            for index in range(length):
                if(df_temp.loc[index,col] >= (Q3+1.5*IQR)):
                    rm_lis.append(index)

            ''' Removing the Outliers '''
            df_temp.drop(rm_lis, inplace = True)

        return df_temp
    
    def remove_cols(self, df, cols, keep=False):
        """
        a functions that removes specified columns from dataframe
        """
        if(keep):
            r_df = df.loc[:,cols]
        else:
            r_df = df.drop(cols, axis=1)

        return r_df

    def reduce_dim_missing(self, df,limit):
        """
        removes columns with number of missing values greater than the provided limit
        """
        temp = self.summar.summ_columns(df)
        rem_lis = []
        for i in range(temp.shape[0]):
            if(temp.iloc[i,2] > limit):
                rem_lis.append(temp.iloc[i,0])
        r_df = df.drop(rem_lis, axis=1) 
        
        return r_df

    
    def fill_missing_by_mode(self, df, cols=None):
        """
        fills missing values by mode
        """
        mod_fill_list = []
        if(cols == None):
            temp = self.summar.summ_columns(df)
            for i in range(temp.shape[0]):
                if(temp.iloc[i,3] == "object"):
                    mod_fill_list.append(temp.iloc[i,0])
        else:
            for col in cols:
                mod_fill_list.append(col)
        
        for col in mod_fill_list:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df


    def fill_missing_by_mean(self, df, cols=None):
        """
        fills missing values by mean
        """
        temp = self.summar.summ_columns(df)
        mean_fill_list = []
        
        if cols is None:
            for i in range(temp.shape[0]):
                if(temp.iloc[i,3] == "float64"):
                    mean_fill_list.append(temp.iloc[i,0])
        else:
            for col in cols:
                mean_fill_list.append(col)
        
        for col in mean_fill_list:
            df[col] = df[col].fillna(df[col].median())
        
        return df

    def fill_missing_by_median(self, df, cols=None):
        """
        fills missing values by median.
        """
        temp = self.summar.summ_columns(df)
        median_fill_list = []

        if cols is None:
            for i in range(temp.shape[0]):
                if(temp.iloc[i,3] == "float64"):
                    median_fill_list.append(temp.iloc[i,0])
        else:
            for col in cols:
                median_fill_list.append(col)

        for col in median_fill_list:
            df[col] = df[col].fillna(df[col].median())
        return df


    def fill_missing_forward(self, df, cols):
        """
        fills missing values by value from next rows
        """
        for col in cols:
            df[col] = df[col].fillna(method='ffill')
        return df

    def fill_missing_backward(self, df, cols):
        """
        fills missing values by value from previous rows
        """
        for col in cols:
            df[col] = df[col].fillna(method='bfill')
        return df


    def format_number(self, df, cols):
        """
        format floating number variables
        """
        float_format_list = []

        for col in cols:
            float_format_list.append(col)
        for col in float_format_list:
            df[col] = df.apply(lambda row : f'{row[col]:,.2f}', axis = 1)

        return df


    def byte_to_mb(self, df, identifier):
        """
        converting byte to megabyte.
        """
        bytes_list = []
        megabyte = 1*10e+5
        temp = self.summar.summ_columns(df)
        
        for i in range(temp.shape[0]):
            if(identifier in temp.iloc[i,0]):
                bytes_list.append(temp.iloc[i,0])
        
        for col in bytes_list:
            df[col] = df[col]/megabyte
            df.rename(columns={col:col.replace(identifier,'(MB)')}, inplace=True)
        return df

    

    def ms_to_s(self, df, identifier):
        """
        converting ms to s.
        """
        ms_list = []
        second = 1000
        temp = self.summar.summ_columns(df)
        
        for i in range(temp.shape[0]):
            if(identifier in temp.iloc[i,0]):
                ms_list.append(temp.iloc[i,0])
        
        for col in ms_list:
            df[col] = df[col]/second
            df.rename(columns={col:col.replace(identifier,'(S)')}, inplace=True)
        return df