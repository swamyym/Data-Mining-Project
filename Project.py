import quandl
from sklearn import datasets,linear_model
import numpy as np
import pandas as pd
import simplejson
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

#==============================================================================
# states_data = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
# 
# main_df = pd.DataFrame()
# states=[]
# for abbv in states_data[0][0][1:]:
#     states.append(abbv)
#     query = "FMAC/HPI_"+str(abbv)
#     df=pd.DataFrame()
#     df = quandl.get(query, authtoken='Kwd-6xZfx5Vk6xtbiFPb')
#     df.rename(columns={'Value':str(abbv)}, inplace=True)
#    # df.columns = str(abbv)
# 
#     if main_df.empty:
#         main_df = df
#     else:
#         main_df = main_df.join(df)
#         
# states_data[0].to_pickle('states_data1')   
# states_data[1].to_pickle('states_data2')   
# states_data[2].to_pickle('states_data3')
# main_df.to_pickle('main_data')
# #states.to_csv('states')
# f=open('states.txt','w')
# simplejson.dump(states,f)
# f.close()
# 
# states_data[0].to_csv('states_data1.csv')   
# states_data[1].to_csv('states_data2.csv')   
# states_data[2].to_csv('states_data3.csv')
# main_df.to_csv('main_data.csv')
#==============================================================================
  
#housing_price_index = pd.read_pickle('main_data')
#==============================================================================
# print(housing_price_index.head())
# housing_price_index.plot()
# plt.show()
#==============================================================================
#==============================================================================
# 
# housing_percentage_change= housing_price_index.pct_change()
# 
# housing_percentage_change.plot()
# plt.show()
#==============================================================================
#print(housing_price_index.corr().head())
#==============================================================================
# 
# states_data = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
# 
# df_pc = pd.DataFrame()
# states=[]
# for abbv in states_data[0][0][1:]:
#     states.append(abbv)
#     query = "FMAC/HPI_"+str(abbv)
#     df=pd.DataFrame()
#     df = quandl.get(query, authtoken='Kwd-6xZfx5Vk6xtbiFPb')
#     
#     df.rename(columns={'Value':str(abbv)}, inplace=True)
#    # df.columns = str(abbv)
#     df[abbv] = (df[abbv]-df[abbv][0]) / df[abbv][0] * 100.0
# 
#     if df_pc.empty:
#         df_pc = df
#     else:
#         df_pc = df_pc.join(df)
#         
#         
# df_pc.to_pickle('df_pc.pickle')
# 
#==============================================================================

  
#df_pc= pd.read_pickle('df_pc.pickle')
#==============================================================================
# df_pc.plot()
# plt.title('Percentage changes from start')
# plt.legend().remove()
# plt.show()
# 
#==============================================================================
#==============================================================================
# adding United state agregate data
#==============================================================================
#==============================================================================
# main_df= pd.read_pickle('main_data.pickle')
# df = quandl.get("FMAC/HPI_USA", authtoken='Kwd-6xZfx5Vk6xtbiFPb')
# df.rename(columns={'Value':str("United States")}, inplace=True)
# 
# main_df= main_df.join(df)
# main_df.to_pickle('main_data_with_us.pickle')
# main_df.to_csv('main_data_with_us.csv')
#==============================================================================
# #==============================================================================
#==============================================================================
#==============================================================================
# df = quandl.get("FMAC/HPI_USA", authtoken='Kwd-6xZfx5Vk6xtbiFPb')
# df.rename(columns={'Value':str("United States")}, inplace=True)
# df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
# df.to_pickle('data_pc_us.pickle')
# df_pc = df_pc.join(df)
# df_pc.to_pickle('df_pc_with_us.pickle')
# df.plot()
# 
# plt.title('Percentage change of United States')
# 
# plt.show()
# 
# 
#==============================================================================
  
#==============================================================================
# df= pd.read_pickle('df_pc_with_us.pickle')
# 
# df.to_csv('df_pc_with_us.csv')
# 
#==============================================================================
#==============================================================================
#Plotting Percentage Change with United states
#==============================================================================
# states_df= pd.read_pickle('df_pc.pickle')
# us_df= pd.read_pickle('data_pc_us.pickle')
# ax1 = plt.subplot2grid((1,1), (0,0))
# states_df.plot(ax=ax1)
# us_df.plot(color='k',ax=ax1, linewidth=10)
# plt.title('Housing Price Index of United states and states')
# plt.legend().remove()
# plt.show()
# 
# 
#==============================================================================

#==============================================================================
#correlation between United states and States individual Price indexes
#==============================================================================
# df = pd.read_pickle('df_pc_with_us.pickle')
# df_correlation = df.corr()
# df_correlation.to_csv('df_correlation.csv')
# print(df_correlation)
#df= pd.read_csv('df_correlation.csv')
#print(df.describe())
#df.to_csv('df_correlation_describe.csv')
#==============================================================================
#Extracting Mortagage Rates 
#==============================================================================
# 
# df = quandl.get("FMAC/MORTG", trim_start="1975-01-01", authtoken='Kwd-6xZfx5Vk6xtbiFPb')
# df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
# df.rename(columns={'Value':str("Mortagage Value")}, inplace=True)
#print(df.head())
# 
# df=df.resample('1D').mean()
# df=df.resample('M').mean()
# df.to_pickle('df_mortagage.pickle')
# df.to_csv('df_mortagage.csv')
# df.plot()
# plt.title('Mortagage Values')
# plt.show()
# 
# 
#==============================================================================
#GDP EXTRACTION

#==============================================================================
# df = quandl.get("OECD/MEI_CLI_LORSGPNO_USA_M", authtoken="Kwd-6xZfx5Vk6xtbiFPb", start_date="1975-01-01")
# df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
# df=df.resample('M').mean()
# df.rename(columns={'Value':'GDP'}, inplace=True)
# df = df['GDP']
# print(df.head())
# df.to_pickle('df_gdp.pickle')
# df.to_csv('df_gdp.csv')
# df.plot()
# plt.title('GDP')
# plt.show()
#==============================================================================
#Unemployment Data
#==============================================================================
# 
# df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken='Kwd-6xZfx5Vk6xtbiFPb')
# df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
#print(df.head())
# df=df.resample('1D').mean()
# df=df.resample('M').mean()
# df.to_pickle('df_unemployment.pickle')
# df.to_csv('df_unemployment.csv')
# df.plot()
# plt.title('Unemployment')
# plt.show()
# 
# 
#==============================================================================
#Stock Index

#==============================================================================
# df=quandl.get("MULTPL/SP500_DIV_MONTH", authtoken="Kwd-6xZfx5Vk6xtbiFPb", start_date="1975-01-01")
# #print(df.head())
# df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
# df.rename(columns={'Value':'SP500'}, inplace=True)
# df.to_pickle('df_SP500.pickle')
# df.to_csv('df_SP500.csv')
# df.plot()
# plt.title('SP500')
# plt.show()
# 
#==============================================================================

#Merging US_Housing price Index, Mortagage , Unemployment rate, GDP and SP500
#==============================================================================
# df=pd.DataFrame()
# df_ushpi= pd.read_pickle('data_pc_us.pickle')
# df_gdp=pd.read_pickle('df_gdp.pickle')
# df_unemp=pd.read_pickle('df_unemployment.pickle')
# df_mort= pd.read_pickle('df_mortagage.pickle')
# df_sp500 = pd.read_pickle('df_SP500.pickle')
# 
# df=df_ushpi
# df=df.join(df_mort)
# df=df.join(df_gdp)
# df=df.join(df_unemp)
# df=df.join(df_sp500)
# df.to_pickle('df_economicdata.pickle')
# df.to_csv('df_economicdata.csv')
# df.plot()
# plt.title('All Economic Data')
# plt.show()
# 
# print(df.head())
# 
# 
#==============================================================================
#Correlation between econimic factors
#droping non number values
#==============================================================================
# df_economy= pd.read_pickle('df_economicdata.pickle')
# #print(df_economy)
# #--------->droping Nan
# df_economy_na=df_economy.dropna()
# df_economy_na.to_pickle('df_economy_na.pickle')
# df_economy_na.to_csv('df_economy_na.csv')
# print(df_economy_na)
# 
#==============================================================================
#==============================================================================
# df_economy_correlation = pd.read_pickle('df_economy_na.pickle').corr()
# df_economy_correlation.to_pickle('df_economy_correlation.pickle')
# df_economy_correlation.to_csv('df_economy_correlation.csv')
# df_economy_correlation_describe=df_economy_correlation.describe()
# print(df_economy_correlation_describe)
# df_economy_correlation_describe.to_pickle('df_economy_correlation_describe.pickle')
# df_economy_correlation_describe.to_csv('df_economy_correlation_describe.csv')
# 
#==============================================================================
#Applying regression to predict United states Housing price index with two sets seperately, the first set with highest state
#next with sp500
#df= pd.read_pickle('main_data_with_us.pickle')
#==============================================================================
# print(df['VA'])
#==============================================================================
#==============================================================================
# 
# hpi_X = df['VA'].reshape(-1,1)
# hpi_Y = df['United States'].reshape(-1,1)
# # Split the data into training/testing sets
# hpi_st_X_train = hpi_X[:-50]
# hpi_st_X_test = hpi_X[-50:]
# 
# # Split the targets into training/testing sets
# hpi_us_y_train = hpi_Y[:-50]
# hpi_us_y_test = hpi_Y[-50:]
# 
# # Create linear regression object
# regr = linear_model.LinearRegression()
# 
# # Train the model using the training sets
# regr.fit(hpi_st_X_train, hpi_us_y_train)
# 
# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % np.mean((regr.predict(hpi_st_X_test) - hpi_us_y_test) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(hpi_st_X_test, hpi_us_y_test))
# 
# # Plot outputs
# plt.scatter(hpi_st_X_test, hpi_us_y_test,  color='black')
# plt.plot(hpi_st_X_test, regr.predict(hpi_st_X_test), color='blue',
#          linewidth=3)
# 
# plt.xticks(())
# plt.yticks(())
# 
# plt.show()
# 
#==============================================================================
#==============================================================================
# df=quandl.get("MULTPL/SP500_DIV_MONTH", authtoken="Kwd-6xZfx5Vk6xtbiFPb", start_date="1975-01-01")
# df.rename(columns={'Value':'SP500'}, inplace=True)
# df2=pd.read_pickle('main_data_with_us.pickle')
# df3=pd.DataFrame()
# df3['United States']=df2['United States']
# df3=df3.join(df)
# df3.dropna(inplace=True)
# 
# 
# hpi_X = df3['SP500'].reshape(-1,1)
# hpi_Y = df3['United States'].reshape(-1,1)
# # Split the data into training/testing sets
# hpi_st_X_train = hpi_X[:-200]
# hpi_st_X_test = hpi_X[-200:]
# 
# # Split the targets into training/testing sets
# hpi_us_y_train = hpi_Y[:-200]
# hpi_us_y_test = hpi_Y[-200:]
# 
# # Create linear regression object
# regr = linear_model.LinearRegression()
# 
# # Train the model using the training sets
# regr.fit(hpi_st_X_train, hpi_us_y_train)
# 
# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % np.mean((regr.predict(hpi_st_X_test) - hpi_us_y_test) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(hpi_st_X_test, hpi_us_y_test))
# 
# # Plot outputs
# plt.scatter(hpi_st_X_test, hpi_us_y_test,  color='black')
# plt.plot(hpi_st_X_test, regr.predict(hpi_st_X_test), color='blue',
#          linewidth=3)
# 
# plt.xticks(())
# plt.yticks(())
# 
# plt.show()
# 
#==============================================================================
