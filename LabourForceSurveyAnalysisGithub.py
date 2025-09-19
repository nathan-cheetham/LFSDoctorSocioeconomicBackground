# -*- coding: utf-8 -*-
"""
Labour force survey socio-economic background of doctors analysis
"""

import numpy as np
import pandas as pd
import gc
import copy
from scipy.stats import chi2_contingency, sem, bootstrap, t, norm
from scipy.spatial import cKDTree
from itertools import combinations 
import sklearn
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.genmod.families.links import logit, identity, log
from statsmodels.stats.multitest import fdrcorrection
from sklearn import preprocessing, metrics
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_error, roc_auc_score, explained_variance_score, r2_score
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter, FormatStrFormatter, FixedLocator, FixedFormatter, (MultipleLocator, AutoMinorLocator)
from matplotlib import cm, colors
import ast
import sys
sys.path.append("nc_scripts") # add nc_scripts folder to path so modules contained within can be imported
import processing_functions as pf
import analysis_functions as af
plt.rc("font", size=12)
sns.set_style("whitegrid")

dictionary = {}

#%% Define functions
#------------------------------------------------------------------------------
# Creating categorical variable from continuous variable 
def add_grouped_field(df,original_fieldname,new_fieldname,value_lims):
    for n in range(0,len(value_lims)):
        if len(value_lims) >= 10:
            # print(n)
            if n < 9:
                if n == len(value_lims)-1:
                    df.loc[(df[original_fieldname] >= value_lims[n])
                                  , new_fieldname] = '0'+ str(n+1) + ': ' + str(value_lims[n]) + '+'
                else:
                    df.loc[(df[original_fieldname] >= value_lims[n])
                                  & (df[original_fieldname] < value_lims[n+1])
                                  , new_fieldname] = '0'+ str(n+1) + ': ' + str(value_lims[n]) + '-' + str(value_lims[n+1])
            else:
                if n == len(value_lims)-1:
                    df.loc[(df[original_fieldname] >= value_lims[n])
                                  , new_fieldname] = str(n+1) + ': ' + str(value_lims[n]) + '+'
                else:
                    df.loc[(df[original_fieldname] >= value_lims[n])
                                  & (df[original_fieldname] < value_lims[n+1])
                                  , new_fieldname] = str(n+1) + ': ' + str(value_lims[n]) + '-' + str(value_lims[n+1])
        else:
            if n == len(value_lims)-1:
                    df.loc[(df[original_fieldname] >= value_lims[n])
                                  , new_fieldname] = str(n+1) + ': ' + str(value_lims[n]) + '+'
            else:
                df.loc[(df[original_fieldname] >= value_lims[n])
                              & (df[original_fieldname] < value_lims[n+1])
                              , new_fieldname] = str(n+1) + ': ' + str(value_lims[n]) + '-' + str(value_lims[n+1])
    
    return df



# -----------------------------------------------------------------------------
### Add dummy variable fields to dataframe generated from un-ordered categoricals
def categorical_to_dummy(df, variable_list_categorical):
    """Create dummy variables from un-ordered categoricals"""
    # Create dummy variables
    dummy_var_list_full = []
    for var in variable_list_categorical:
        df[var] = df[var].fillna('NaN') # fill NaN with 'No data' so missing data can be distinguished from 0 results
        cat_list ='var'+'_'+var # variable name
        cat_list = pd.get_dummies(df[var], prefix=var) # create binary variable of category value
        df = df.join(cat_list) # join new column to dataframe
    
    return df

# -----------------------------------------------------------------------------
### Generate list of categorical dummy variables from original fieldname, deleting original fieldname and deleting reference variable using reference dummy variable list
def generate_dummy_list(original_fieldname_list, full_fieldname_list, reference_fieldname_list, delete_reference):
    """ Generate list of categorical dummy variables from original fieldname, deleting original fieldname. Option to also delete reference variable using reference dummy variable list (delete_reference = 'yes') """
    dummy_list = []
    for var in original_fieldname_list:
        # print(var)
        var_matching_all = [variable_name for variable_name in full_fieldname_list if var in variable_name]
        # drop original variable
        var_matching_all.remove(var)
        if delete_reference == 'yes':
            # drop reference variable
            var_matching_reference = [variable_name for variable_name in reference_fieldname_list if var in variable_name][0]
            # print('matching reference var')
            # print(var_matching_reference)            
            var_matching_all.remove(var_matching_reference)
        # add to overall list
        dummy_list += var_matching_all
    
    return dummy_list

# -----------------------------------------------------------------------------
# Function to generate weights from model
def generate_IPW(data, model_fit, weight_colname_suffix):
    """ Generate scaled inverse probability weights from logistic regression fit and add to dataset """
    # Generate propensity scores 
    propensity_score =  model_fit.fittedvalues
    # Add to input data
    data['probability_'+weight_colname_suffix] = propensity_score
    # Plot histogram of weights
    ax = plt.figure()
    ax = sns.histplot(data=data, x='probability_'+weight_colname_suffix, hue=outcome_var, element="poly")
    ax1 = plt.figure()
    ax1 = sns.histplot(data=data, x='probability_'+weight_colname_suffix, hue=outcome_var, element="poly", stat = "probability", common_norm = False)
    
    # Following guidance in CLS report "Handling non-response in COVID-19 surveys across five national longitudinal studies"
    # Calculate non-response weights as simply 1/probability of response 
    data['probability_'+weight_colname_suffix+'_inverse'] = np.divide(1, data['probability_'+weight_colname_suffix])
    # Scale weights so that sum = number of participants in sample
    scaling_factor = data.shape[0] / (data['probability_'+weight_colname_suffix+'_inverse'].sum())
    data['IPW_'+weight_colname_suffix] = scaling_factor * data['probability_'+weight_colname_suffix+'_inverse']
    
    return data

# -----------------------------------------------------------------------------
### Run logistic regression model with HC3 robust error, producing summary dataframe  
def sm_logreg_simple_HC3(x_data, y_data, CI_alpha, do_robust_se, use_weights, weight_data, do_poisson):
    """ Run logistic regression model with HC3 robust error, producing summary dataframe """
    # Add constant - default for sklearn but not statsmodels
    print('constant about to be added')
    x_data = sm.add_constant(x_data) 
    print('constant added')
    
    # Add weight data to x_data if weights specified
    if use_weights == 'yes':
        x_data['weight'] = weight_data
        
    # Set model parameters
    max_iterations = 35
    solver_method = 'newton' # use default lbfgs newton
    # model = sm.Logit(y_data, x_data, use_t = True) # Previous model - same results. Replaced with more general construction, as GLM allows weights to be included. 
    
    # Also run model on test and train split to assess predictive power
    # Generate test and train split
    print('test train split about to start')
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, stratify = y_data, test_size = 0.25, random_state = 0)
    print('test train split done')
    
    # Save weight data in x_train and the drop weight data 
    if use_weights == 'yes':
        weight_data_train = np.asarray(x_train['weight'].copy())
        # drop weight columns
        x_data = x_data.drop(columns = ['weight'])
        x_train = x_train.drop(columns = ['weight'])
        x_test = x_test.drop(columns = ['weight'])
    
    # Set up overall and test-train models
    if use_weights == 'yes':
        model = sm.GLM(y_data, x_data, 
                       var_weights = weight_data, 
                       family = sm.families.Binomial(link=sm.families.links.Logit()),
                       )
        model_testtrain = sm.GLM(y_train, x_train, 
                       var_weights = weight_data_train, 
                       family = sm.families.Binomial(link=sm.families.links.Logit()),
                       )
        if do_poisson == 'yes':
            model = sm.GLM(y_data, x_data, 
                       var_weights = weight_data,
                       family = sm.families.Poisson(link=sm.families.links.Log()), 
                       )
            model_testtrain = sm.GLM(y_train, x_train,
                       var_weights = weight_data_train,
                       family = sm.families.Poisson(link=sm.families.links.Log()),
                       )
    else:
        model = sm.GLM(y_data, x_data, 
                       family = sm.families.Binomial(link=sm.families.links.Logit()) 
                       )
        model_testtrain = sm.GLM(y_train, x_train,
                       family = sm.families.Binomial(link=sm.families.links.Logit()),
                       )
        if do_poisson == 'yes':
            model = sm.GLM(y_data, x_data, 
                       family = sm.families.Poisson(link=sm.families.links.Log()), 
                       )
            model_testtrain = sm.GLM(y_train, x_train,
                       family = sm.families.Poisson(link=sm.families.links.Log()),
                       )
                
    # Fit GLM OLS model
    print('about to run model')
    if do_robust_se == 'HC3':
        model_fit = model.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='HC3',
                          use_t=True)
        model_testtrain_fit = model_testtrain.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='HC3',
                          use_t=True)
    else:
        model_fit = model.fit(maxiter = max_iterations, 
                              method = solver_method, 
                              use_t=True)
        model_testtrain_fit = model_testtrain.fit(maxiter = max_iterations, 
                              method = solver_method, 
                              use_t=True) 
    print('model fit finished')
    print(model_fit.summary())
   
    # Calculate AUC and explained variance score of model
    y_prob = model_testtrain_fit.predict(x_test)
    if np.isnan(np.min(y_prob)) == False:
        model_auc = roc_auc_score(y_test, y_prob)
        model_explained_variance = explained_variance_score(y_test, y_prob)
        model_r2 = r2_score(y_test, y_prob)
    else:
        print(y_prob)
        model_auc = 0 # for when AUC failed - e.g. due to non-convergence of model
        model_explained_variance = np.nan
        model_r2 = np.nan
        
    # Extract coefficients and convert to Odds Ratios
    sm_coeff = model_fit.params
    sm_se = model_fit.bse
    sm_pvalue = model_fit.pvalues
    sm_coeff_CI = model_fit.conf_int(alpha=CI_alpha)
    sm_OR = np.exp(sm_coeff)
    sm_OR_CI = np.exp(sm_coeff_CI)
    
    # Create dataframe summarising results
    sm_summary = pd.DataFrame({'Variable': sm_coeff.index,
                               'Coefficients': sm_coeff,
                               'Standard Error': sm_se,
                               'P-value': sm_pvalue,
                               'Coefficient C.I. (lower)': sm_coeff_CI[0],
                               'Coefficient C.I. (upper)': sm_coeff_CI[1],
                               'Odds ratio': sm_OR,
                               'OR C.I. (lower)': sm_OR_CI[0],
                               'OR C.I. (upper)': sm_OR_CI[1],
                               'OR C.I. error (lower)': np.abs(sm_OR - sm_OR_CI[0]),
                               'OR C.I. error (upper)': np.abs(sm_OR - sm_OR_CI[1]),
                                })
    sm_summary = sm_summary.reset_index(drop = True)
    
    # Add total number of individuals in given model
    sm_summary['total_count_n'] = len(x_data)

    # Add number of observations for given variable in input and outcome datasets
    x_data_count = x_data.sum()
    x_data_count.name = "group_count"
    sm_summary = pd.merge(sm_summary,x_data_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # join x_data and y_data
    x_y_data = x_data.copy()
    x_y_data['y_data'] = y_data
    # Count observation where y_data = 1
    y_data_count = x_y_data[x_y_data['y_data'] == 1].sum()
    y_data_count.name = "outcome_count"
    sm_summary = pd.merge(sm_summary,y_data_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # Highlight variables where confidence intervals are both below 1 or both above 1
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.05)
                        ,'Significance'] = 'Significant (OR > 1), *, p < 0.05'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.01)
                        ,'Significance'] = 'Significant (OR > 1), **, p < 0.01'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.001)
                        ,'Significance'] = 'Significant (OR > 1), ***, p < 0.001'
    
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.05)
                        ,'Significance'] = 'Significant (OR < 1), *, p < 0.05'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.01)
                        ,'Significance'] = 'Significant (OR < 1), **, p < 0.01'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.001)
                        ,'Significance'] = 'Significant (OR < 1), ***, p < 0.001'
    
    
        
    return sm_summary, model_fit, model_auc, model_explained_variance, model_r2, model, model_testtrain


# -----------------------------------------------------------------------------
# Function to add results for reference categories
def add_reference_odds_ratio(data):
    reference_list = []
    for var_exposure in data['var_exposure'].unique():
        print(var_exposure)
        var_reference_list = [var for var in cols_categorical_reference if var_exposure in var]
        if len(var_reference_list) > 0: # if there's a match
            var_reference = var_reference_list[0]
            # print(var_reference)
            reference_list.append({'Variable':var_reference, 'Odds ratio':1, 'var_exposure':var_exposure})
    
    for n in range(0,len(reference_list),1):
        reference = reference_list[n]
        data = data.append(reference, ignore_index=True) # Append row to the dataframe
    
    return data

# -----------------------------------------------------------------------------
# Function to run series of logistic regression models in sequence
def run_logistic_regression_models(data, data_full_col_list, model_var_list, outcome_var, use_weights, weight_var, filter_missing, plot_model, cols_categorical_reference, do_poisson):
    """ Function to run logistic regression models, given lists of categorical and continuous input variables """
    model_input_list = []
    model_input_dummy_list = []
    model_auc_list= []
    model_summary_list = []
    model_fit_list = []
    model_prediction_df_list = []
    model_predict_df_list = []
    for sublist in model_var_list:
        var_continuous = sublist[0]
        var_categorical = sublist[1]
        var_exposure = sublist[2] # identify exposure variable being tested in model
        
        print('Exposure variable: ' + var_exposure)
        print('Categorical variables: ')
        print(var_categorical)
        print('Continuous variables: ')
        print(var_continuous)
        
        # Filter out missing or excluded data
        print('Individuals before filtering: ' + str(data.shape[0]))
        data_filterformodel = data.copy(deep = False)
        
        if filter_missing == 'yes':
            # Filter using original categorical variables
            for col in var_categorical:
                # print(col)
                data_filterformodel = data_filterformodel[~(data_filterformodel[col].isin(missing_data_values))]
            print('Individuals after filtering: ' + str(data_filterformodel.shape[0]))
            # Filter using original continuous variables
            for col in var_continuous:
                data_filterformodel = data_filterformodel[~(data_filterformodel[col].isin(missing_data_values))]
            print('Individuals after filtering: ' + str(data_filterformodel.shape[0]))
            
            # for col in input_var_control_test:
            #         data_filterformodel = data_filterformodel[~(data_filterformodel[col].isin(missing_data_values))]
            # print('Individuals after filtering: ' + str(data_filterformodel.shape[0]))
        
        # Generate list of dummy fields for complete fields
        var_categorical_dummy = generate_dummy_list(original_fieldname_list = var_categorical, 
                                                         full_fieldname_list = data_full_col_list, 
                                                         reference_fieldname_list = cols_categorical_reference,
                                                         delete_reference = 'yes')
        print('Before dropping MISSING DATA and EMPTY dummy cols: ')
        print(var_categorical_dummy)
        if filter_missing == 'yes':
            # Drop dummy columns with name ending in missing data values (categorical only)
            var_categorical_dummy_copy = var_categorical_dummy.copy()
            for col in var_categorical_dummy_copy:
                print(col)
                for missing_val in missing_data_values[1:]:
                    if '_'+str(missing_val) in col:
                        print('remove missing data dummy: ' + col)
                        var_categorical_dummy.remove(col)
            # print('After dropping MISSING DATA dummy cols: ')
            # print(var_categorical_dummy)
        
        # Drop dummy columns where sum of column = 0, 1 or 2 - i.e. no-one or low numbers from particular group - can cause 'Singular matrix' error when running model
        print('testing for columns with few observations')
        var_categorical_dummy_copy = var_categorical_dummy.copy()
        for col in var_categorical_dummy_copy: 
            print(col)
            if data_filterformodel.copy(deep = False)[col].values.sum() <= 2:
                print('remove empty/low number (<=2) dummy: ' + col)
                var_categorical_dummy.remove(col)
        
        # MOVED OUTSIDE OF LOOP - MANUALLY - AS TAKES TOO LONG
        # Drop dummy columns where no observations of outcome in group in column = 0 - i.e. no observations - can cause 'Singular matrix' error when running model
        print('testing for columns with no observations of outcome')
        var_categorical_dummy_copy = var_categorical_dummy.copy()
        
        # Generate boolean showing observations of outcome
        outcome_true = (data_filterformodel[outcome_var].isin([1]))
        # Sum observations of outcome for each column of interest
        outcome_true_sum = data_filterformodel[outcome_true][var_categorical_dummy_copy].sum()
        for col in var_categorical_dummy_copy: 
            if outcome_true_sum[col] < 1: # find cols with no observations of outcome
                print('remove dummy with no observations of outcome: ' + col)
                var_categorical_dummy.remove(col)
            
            
        print('After dropping MISSING DATA and EMPTY dummy cols: ')
        print(var_categorical_dummy)
        
        # Set variables to go into model
        input_var_control_test = var_continuous + var_categorical_dummy
        model_input_dummy = str(var_continuous + var_categorical_dummy)
        
        model_input = str(var_continuous + var_categorical)
        model_input_list.append(model_input)
        model_input_dummy_list.append(model_input_dummy)
        print('model input variables: ' + model_input)
        print('model input variables (dummy): ' + str(input_var_control_test))
        
        # generate x dataset for selected control and test dummy variables only
        print('dataset length before model: ' + str(data_filterformodel.shape[0]))
        logreg_data_x = data_filterformodel[input_var_control_test].copy(deep = False) #.reset_index(drop=True) # create input variable tables for models 
        # generate y datasets from selected number of vaccinations and outcome of interest
        logreg_data_y = data_filterformodel[outcome_var] #.reset_index(drop=True) # set output variable
        print('model ready to run')
        if use_weights == 'yes':
            logreg_data_weight = data_filterformodel[weight_var].reset_index(drop=True) # filter for weight variable
            # Do logistic regression (stats models) of control + test variables
            sm_summary, model_fit, model_auc, model_explained_variance, model_r2, model, model_testtrain = sm_logreg_simple_HC3(x_data = logreg_data_x, y_data = logreg_data_y, 
                                                     CI_alpha = 0.05, do_robust_se = 'HC3',
                                                     use_weights = use_weights, weight_data = np.asarray(logreg_data_weight),
                                                     do_poisson = do_poisson)
        else:
            sm_summary, model_fit, model_auc, model_explained_variance, model_r2, model, model_testtrain = sm_logreg_simple_HC3(x_data = logreg_data_x, y_data = logreg_data_y, 
                                                     CI_alpha = 0.05, do_robust_se = 'HC3',
                                                     use_weights = '', weight_data = '',
                                                     do_poisson = do_poisson)
            
        sm_summary['model_input'] = model_input
        sm_summary['model_input_dummy'] = model_input_dummy
        sm_summary['var_exposure'] = var_exposure
        sm_summary['outcome_variable'] = outcome_var
        model_summary_list.append(sm_summary)
        model_fit_list.append(model_fit)
        
        # Generate table of individual level predictions with confidence levels
        # https://www.statsmodels.org/dev/generated/statsmodels.genmod.generalized_linear_model.GLMResults.get_prediction.html
        # https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.PredictionResultsMean.html#statsmodels.genmod.generalized_linear_model.PredictionResultsMean
        model_predict_df = model_fit.predict()
        model_predict_df_list.append(model_predict_df)
        model_prediction_df = model_fit.get_prediction().summary_frame()
        model_prediction_df_list.append(model_prediction_df)
        
        # Print predictive power
        model_auc_list.append(model_auc) 
        print ('AUC: ' + str(model_auc))
        
        # Plot odds ratios
        if plot_model == 'yes':
            sm_summary_filter = sm_summary[sm_summary['Variable'] != 'const'] # filter out constant so not plotted
            fig = af.plot_OR_w_conf_int(sm_summary_filter, 'Variable', 'Odds ratio', ['OR C.I. error (lower)','OR C.I. error (upper)'], 'Odds Ratio', ylims = [], titlelabel = outcome_var)
            
        gc.collect() # release/remove data to open up RAM hopefully
        
    # -----------------------------------------------------------------------------
    # Combine model results tables together
    model_results_summary = pd.concat(model_summary_list)
    model_auc_summary = pd.DataFrame({'model_input':model_input_list,
                                      'model_auc':model_auc_list,})
    
    return model_results_summary, model_auc_summary, model_fit_list, model_prediction_df_list, model_predict_df_list, model_input_dummy_list

# -----------------------------------------------------------------------------
# Do manual winsorisation as scipy winsorize function has bugs (treats nan as high number rather than ignoring) which means upper end winsorisation doesn't work
def winsorization(data, winsorization_limits, winsorization_col_list, set_manual_limits, manual_limit_list):
    for n in range(0, len(winsorization_col_list), 1): 
        col = winsorization_col_list[n]
        # Create copy columns 
        data[col + '_winsorised'] = data[col].copy()
        
        if set_manual_limits == 'yes':
            winsorize_lower = manual_limit_list[n][0]
            winsorize_upper = manual_limit_list[n][1]
        else:
            # Calculate percentile
            winsorize_lower = data[col].quantile(winsorization_limits[0])
            winsorize_upper = data[col].quantile(winsorization_limits[1])
            
        print('lower limit = ' + str(winsorize_lower))
        print('higher limit = ' + str(winsorize_upper))
        # Replace lower and upper values with limited values
        data.loc[(data[col] < winsorize_lower), col + '_winsorised'] = winsorize_lower
        data.loc[(data[col] > winsorize_upper), col + '_winsorised'] = winsorize_upper
    return data


# -----------------------------------------------------------------------------
# 1 SERIES scatter plot
def plot_OR_w_conf_int(data1, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, xlims, ylims, titlelabel, width, height, y_pos_manual, color_list, fontsize, invert_axis, x_logscale, legend_offset, x_major_tick, x_minor_tick, poisson_reg, bold):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    if y_pos_manual == 'yes':
        data1['x_manual'] = (data1['y_pos_manual'])
    else:
        data1['x_manual'] = (np.arange(len(data1[x_fieldname])))
    
    # plot scatter points
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[0], s = 5, label = plot1_label, figsize=(width,height))
    # plot error bars
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 4, label = None, fmt = 'none', color = color_list[0])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data1['x_manual'], data1[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    
    if poisson_reg == 'yes':
        ax.set_xlabel('Relative risk ratio')
    else:
        ax.set_xlabel('Odds ratio')
    ax.yaxis.label.set_visible(False) # hide y axis title
    ax.set_title(titlelabel)
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
    
    if x_logscale == 'yes':
        ax.set_xscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twiny()
        secax.set_xscale('linear')
        secax.set_xlim(np.log(xlims[0]), np.log(xlims[1]))
        secax.grid(False, which="both", axis = 'x')
        secax.set_xlabel('Coefficient')
        secax.tick_params(axis='x', which='major', labelsize=10)
        secax.set_xlabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', labelsize=8.2)
        
        xlim_thresh = 1.5
        if xlims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',                                   
                                      '1.1','','1.3','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif xlims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '', '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        
        ax.xaxis.set_minor_formatter(x_formatter)
        ax.xaxis.set_minor_locator(x_locator)
                
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        
    if x_major_tick > 0:
        secax.xaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick))
        
    if bold == 1:
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
    
    return data1

# -----------------------------------------------------------------------------
# 2 SERIES scatter plot
def plot_OR_w_conf_int_2plots(data1, data2, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
        
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + offset 
        data2['x_manual'] = (data2['y_pos_manual']*scalar) - offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) - offset
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[1], s = 5, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data1['x_manual'] - (offset), data1[x_fieldname]) # set labels manually
    # plt.yticks(data2['x_manual'] + (offset), data2[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    
    if poisson_reg == 'yes':
        ax.set_xlabel('Relative risk ratio')
    else:
        ax.set_xlabel('Odds ratio')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way

    if x_logscale == 'yes':
        ax.set_xscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twiny()
        secax.set_xscale('linear')
        secax.set_xlim(np.log(xlims[0]), np.log(xlims[1]))
        secax.grid(False, which="both", axis = 'x')
        secax.set_xlabel('Coefficient')
        secax.tick_params(axis='x', which='major', labelsize=10)
        secax.set_xlabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', labelsize=8.2)
        
        xlim_thresh = 3.5
        if xlims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',                                   
                                      '1.1','','1.3','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif xlims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '3', '4', '', '6', '', '8', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
            
        ax.xaxis.set_minor_formatter(x_formatter)
        ax.xaxis.set_minor_locator(x_locator)
        
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if x_major_tick > 0:
        secax.xaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick))
        # secax.xaxis.grid(True, which='minor', color='#EEEEEE')

    return data1, data2


# -----------------------------------------------------------------------------
# 3 SERIES scatter plot
def plot_OR_w_conf_int_3plots(data1, data2, data3, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg):       
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + offset 
        data2['x_manual'] = (data2['y_pos_manual']*scalar)
        data3['x_manual'] = (data3['y_pos_manual']*scalar) - offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) - offset
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[1], s = 5, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    # Plot 3
    ax = data3.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "s", color = color_list[2], s = 6, label = plot3_label, ax = ax)
    error_bar3 = ax.errorbar(y = data3['x_manual'], x = data3[y_fieldname], xerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data3['x_manual'] + (offset), data3[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    if poisson_reg == 'yes':
        ax.set_xlabel('Relative risk ratio')
    else:
        ax.set_xlabel('Odds ratio')
        
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
    
    if x_logscale == 'yes':
        ax.set_xscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twiny()
        secax.set_xscale('linear')
        secax.set_xlim(np.log(xlims[0]), np.log(xlims[1]))
        secax.grid(False, which="both", axis = 'x')
        secax.set_xlabel('Coefficient')
        secax.tick_params(axis='x', which='major', labelsize=10)
        secax.set_xlabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', labelsize=8.2)
        xlim_thresh = 1.5
        if xlims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '', '0.8', '',                                   
                                      '','','','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif xlims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '', '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        
        ax.xaxis.set_minor_formatter(x_formatter)
        ax.xaxis.set_minor_locator(x_locator)
                
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    if x_major_tick > 0:
        secax.xaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick))

    return data1, data2, data3


# -----------------------------------------------------------------------------
# 2 SERIES scatter plot with lines, for time-series
def plot_OR_w_conf_int_1plot_timeseries(data1, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg, xlabel):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
        
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar)
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar)
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(height,width))
    error_bar1 = ax.errorbar(x = data1['x_manual'], y = data1[y_fieldname], yerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    data1.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[0], label = '_nolegend_', alpha = 0.7, ax = ax) 
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.xticks(data1['x_manual'], data1[x_fieldname]) # set labels manually
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    # plt.yticks(data2['x_manual'] + (offset), data2[x_fieldname]) # set labels manually 
    # ax.xaxis.label.set_visible(False) # hide y axis title
    ax.set_xlabel(xlabel)
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axhline(y = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    
    if poisson_reg == 'yes':
        ax.set_ylabel('Relative risk ratio')
    else:
        ax.set_ylabel('Odds ratio')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    
    
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way

    if x_logscale == 'yes':
        ax.set_yscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twinx()
        secax.set_yscale('linear')
        secax.set_ylim(np.log(ylims[0]), np.log(ylims[1]))
        secax.grid(False, which="both", axis = 'y')
        secax.set_ylabel('Coefficient')
        secax.tick_params(axis='y', which='major', labelsize=10)
        secax.set_ylabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', labelsize=8.2)
        
        xlim_thresh = 3.5
        if ylims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',                                   
                                      '1.1','','1.3','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif ylims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '3', '4', '', '6', '', '8', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
            
        ax.yaxis.set_minor_formatter(x_formatter)
        ax.yaxis.set_minor_locator(x_locator)
        
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if x_major_tick > 0:
        secax.yaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.yaxis.set_minor_locator(MultipleLocator(x_minor_tick))
        # secax.xaxis.grid(True, which='minor', color='#EEEEEE')

    return data1

# -----------------------------------------------------------------------------
# 2 SERIES scatter plot with lines, for time-series
def plot_OR_w_conf_int_2plots_timeseries(data1, data2, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg, xlabel, xtick_rotation, error_fill):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
        
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + offset
        data2['x_manual'] = (data2['y_pos_manual']*scalar) - offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) - offset
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(height,width))
    if error_fill == 'yes':
        ax.fill_between(x = data1['x_manual'], y1 = np.array(data1[conf_int_fieldnames[0]].transpose()), y2 = np.array(data1[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[0], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[0])
    else:
        error_bar1 = ax.errorbar(x = data1['x_manual'], y = data1[y_fieldname], yerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    data1.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[0], label = '_nolegend_', alpha = 0.7, ax = ax) 
    
    # Plot 2
    ax = data2.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "D", color = color_list[1], s = 5, label = plot2_label, alpha = 0.4, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data2['x_manual'], y1 = np.array(data2[conf_int_fieldnames[0]].transpose()), y2 = np.array(data2[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[1], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[1])
    else:
        error_bar2 = ax.errorbar(x = data2['x_manual'], y = data2[y_fieldname], yerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    data2.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[1], label = '_nolegend_', alpha = 0.7, ax = ax) 
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.xticks(data1['x_manual'] - offset, data1[x_fieldname]) # set labels manually
    if xtick_rotation < 80:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, rotation_mode='anchor', ha="right")
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    # plt.yticks(data2['x_manual'] + (offset), data2[x_fieldname]) # set labels manually 
    # ax.xaxis.label.set_visible(False) # hide y axis title
    ax.set_xlabel(xlabel)
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axhline(y = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    
    if poisson_reg == 'yes':
        ax.set_ylabel('Relative risk ratio')
    else:
        ax.set_ylabel('Odds ratio')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    
    
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way

    if x_logscale == 'yes':
        ax.set_yscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twinx()
        secax.set_yscale('linear')
        secax.set_ylim(np.log(ylims[0]), np.log(ylims[1]))
        secax.grid(False, which="both", axis = 'y')
        secax.set_ylabel('Coefficient')
        secax.tick_params(axis='y', which='major', labelsize=10)
        secax.set_ylabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', labelsize=8.2)
        
        xlim_thresh = 3.5
        if ylims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',                                   
                                      '1.1','','1.3','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif ylims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '3', '4', '', '6', '', '8', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
            
        ax.yaxis.set_minor_formatter(x_formatter)
        ax.yaxis.set_minor_locator(x_locator)
        
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if x_major_tick > 0:
        secax.yaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.yaxis.set_minor_locator(MultipleLocator(x_minor_tick))
        # secax.xaxis.grid(True, which='minor', color='#EEEEEE')

    return data1, data2

# -----------------------------------------------------------------------------
# 3 SERIES scatter plot with lines, for time-series
def plot_OR_w_conf_int_3plots_timeseries(data1, data2, data3, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg, xlabel, xtick_rotation, error_fill):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
        
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + (offset)
        data2['x_manual'] = (data2['y_pos_manual']*scalar)
        data3['x_manual'] = (data3['y_pos_manual']*scalar) - (offset)
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + (offset)
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) - (offset)
    
    # Plot 1     
    alpha = 0.7
    ax = data1.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(height,width))
    if error_fill == 'yes':
        ax.fill_between(x = data1['x_manual'], y1 = np.array(data1[conf_int_fieldnames[0]].transpose()), y2 = np.array(data1[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[0], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[0])
    else:
        error_bar1 = ax.errorbar(x = data1['x_manual'], y = data1[y_fieldname], yerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    data1.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[0], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    # Plot 2
    ax = data2.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "D", color = color_list[1], s = 7, label = plot2_label, alpha = 0.4, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data2['x_manual'], y1 = np.array(data2[conf_int_fieldnames[0]].transpose()), y2 = np.array(data2[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[1], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[1])
    else:
        error_bar2 = ax.errorbar(x = data2['x_manual'], y = data2[y_fieldname], yerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    data2.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[1], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    # Plot 3
    ax = data3.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "s", color = color_list[2], s = 6, label = plot3_label, alpha = 0.4, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data3['x_manual'], y1 = np.array(data3[conf_int_fieldnames[0]].transpose()), y2 = np.array(data3[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[2], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[2])
    else:
        error_bar3 = ax.errorbar(x = data3['x_manual'], y = data3[y_fieldname], yerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    data3.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[2], label = '_nolegend_', alpha = alpha, ax = ax) 
        
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.xticks(data1['x_manual'] - (offset), data1[x_fieldname]) # set labels manually
    if xtick_rotation < 80:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, rotation_mode='anchor', ha="right")
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
        
    # plt.yticks(data2['x_manual'] + (offset), data2[x_fieldname]) # set labels manually 
    # ax.xaxis.label.set_visible(False) # hide y axis title
    ax.set_xlabel(xlabel)
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axhline(y = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    
    if poisson_reg == 'yes':
        ax.set_ylabel('Relative risk ratio')
    else:
        ax.set_ylabel('Odds ratio')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    
    
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way

    if x_logscale == 'yes':
        ax.set_yscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twinx()
        secax.set_yscale('linear')
        secax.set_ylim(np.log(ylims[0]), np.log(ylims[1]))
        secax.grid(False, which="both", axis = 'y')
        secax.set_ylabel('Coefficient')
        secax.tick_params(axis='y', which='major', labelsize=10)
        secax.set_ylabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', labelsize=8.2)
        
        xlim_thresh = 3.5
        if ylims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',                                   
                                      '1.1','','1.3','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif ylims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '3', '4', '', '6', '', '8', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
            
        ax.yaxis.set_minor_formatter(x_formatter)
        ax.yaxis.set_minor_locator(x_locator)
        
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if x_logscale == 'yes':
        if x_major_tick > 0:
            secax.yaxis.set_major_locator(MultipleLocator(x_major_tick))
        if x_minor_tick > 0:
            secax.yaxis.set_minor_locator(MultipleLocator(x_minor_tick))
            # secax.xaxis.grid(True, which='minor', color='#EEEEEE')

    return data1, data2, data3

# -----------------------------------------------------------------------------
# 4 SERIES scatter plot with lines, for time-series
def plot_OR_w_conf_int_4plots_timeseries(data1, data2, data3, data4, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, plot4_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg, xlabel, xtick_rotation, error_fill):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
        
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + 2*(offset/2)
        data2['x_manual'] = (data2['y_pos_manual']*scalar) + (offset/2)
        data3['x_manual'] = (data3['y_pos_manual']*scalar) - (offset/2)
        data4['x_manual'] = (data4['y_pos_manual']*scalar) - 2*(offset/2)
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + 2*(offset/2)
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) + (offset/2)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) - (offset/2)
        data4['x_manual'] = (np.arange(len(data4[x_fieldname]))*scalar) - 2*(offset/2)
    
    # Plot 1     
    alpha = 0.7
    ax = data1.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(height,width))
    if error_fill == 'yes':
        ax.fill_between(x = data1['x_manual'], y1 = np.array(data1[conf_int_fieldnames[0]].transpose()), y2 = np.array(data1[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[0], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[0])
    else:
        error_bar1 = ax.errorbar(x = data1['x_manual'], y = data1[y_fieldname], yerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    data1.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[0], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    # Plot 2
    ax = data2.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "D", color = color_list[1], s = 8, label = plot2_label, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data2['x_manual'], y1 = np.array(data2[conf_int_fieldnames[0]].transpose()), y2 = np.array(data2[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[1], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[1])
    else:
        error_bar2 = ax.errorbar(x = data2['x_manual'], y = data2[y_fieldname], yerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    data2.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[1], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    # Plot 3
    ax = data3.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "s", color = color_list[2], s = 6, label = plot3_label, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data3['x_manual'], y1 = np.array(data3[conf_int_fieldnames[0]].transpose()), y2 = np.array(data3[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[2], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[2])
    else:
        error_bar3 = ax.errorbar(x = data3['x_manual'], y = data3[y_fieldname], yerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    data3.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[2], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    # Plot 4
    ax = data4.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "^", color = color_list[3], s = 9, label = plot4_label, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data4['x_manual'], y1 = np.array(data4[conf_int_fieldnames[0]].transpose()), y2 = np.array(data4[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[3], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[3])
    else:
        error_bar4 = ax.errorbar(x = data4['x_manual'], y = data4[y_fieldname], yerr = np.array(data4[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[3])
    data4.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[3], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.xticks(data1['x_manual'] - 2*(offset/2), data1[x_fieldname]) # set labels manually
    if xtick_rotation < 80:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, rotation_mode='anchor', ha="right")
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    # plt.yticks(data2['x_manual'] + (offset), data2[x_fieldname]) # set labels manually 
    # ax.xaxis.label.set_visible(False) # hide y axis title
    ax.set_xlabel(xlabel)
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axhline(y = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    
    if poisson_reg == 'yes':
        ax.set_ylabel('Relative risk ratio')
    else:
        ax.set_ylabel('Odds ratio')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    
    
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way

    if x_logscale == 'yes':
        ax.set_yscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twinx()
        secax.set_yscale('linear')
        secax.set_ylim(np.log(ylims[0]), np.log(ylims[1]))
        secax.grid(False, which="both", axis = 'y')
        secax.set_ylabel('Coefficient')
        secax.tick_params(axis='y', which='major', labelsize=10)
        secax.set_ylabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', labelsize=8.2)
        
        xlim_thresh = 3.5
        if ylims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',                                   
                                      '1.1','','1.3','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif ylims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '2', 
                                      '', '4', '', '6', '', '8', '',
                                      '20', '', '40', '', '60', '', '80', ''])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60, 70, 80, 90,])
            
        ax.yaxis.set_minor_formatter(x_formatter)
        ax.yaxis.set_minor_locator(x_locator)
        
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if x_logscale == 'yes':
        if x_major_tick > 0:
            secax.yaxis.set_major_locator(MultipleLocator(x_major_tick))
        if x_minor_tick > 0:
            secax.yaxis.set_minor_locator(MultipleLocator(x_minor_tick))
            # secax.xaxis.grid(True, which='minor', color='#EEEEEE')

    return data1, data2, data3, data4



# -----------------------------------------------------------------------------
# 5 SERIES scatter plot with lines, for time-series
def plot_OR_w_conf_int_5plots_timeseries(data1, data2, data3, data4, data5, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, plot4_label, plot5_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg, xlabel, xtick_rotation, error_fill):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
        
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + 2*(offset/2)
        data2['x_manual'] = (data2['y_pos_manual']*scalar) + (offset/2)
        data3['x_manual'] = (data3['y_pos_manual']*scalar) 
        data4['x_manual'] = (data4['y_pos_manual']*scalar) - (offset/2)
        data5['x_manual'] = (data5['y_pos_manual']*scalar) - 2*(offset/2)
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + 2*(offset/2)
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) + (offset/2)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar)
        data4['x_manual'] = (np.arange(len(data4[x_fieldname]))*scalar) - (offset/2)
        data5['x_manual'] = (np.arange(len(data5[x_fieldname]))*scalar) - 2*(offset/2)
    
    # Plot 1     
    alpha = 0.7
    ax = data1.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(height,width))
    if error_fill == 'yes':
        ax.fill_between(x = data1['x_manual'], y1 = np.array(data1[conf_int_fieldnames[0]].transpose()), y2 = np.array(data1[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[0], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[0])
    else:
        error_bar1 = ax.errorbar(x = data1['x_manual'], y = data1[y_fieldname], yerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    data1.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[0], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    # Plot 2
    ax = data2.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "D", color = color_list[1], s = 8, label = plot2_label, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data2['x_manual'], y1 = np.array(data2[conf_int_fieldnames[0]].transpose()), y2 = np.array(data2[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[1], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[1])
    else:
        error_bar2 = ax.errorbar(x = data2['x_manual'], y = data2[y_fieldname], yerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    data2.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[1], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    # Plot 3
    ax = data3.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "s", color = color_list[2], s = 6, label = plot3_label, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data3['x_manual'], y1 = np.array(data3[conf_int_fieldnames[0]].transpose()), y2 = np.array(data3[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[2], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[2])
    else:
        error_bar3 = ax.errorbar(x = data3['x_manual'], y = data3[y_fieldname], yerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    data3.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[2], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    # Plot 4
    ax = data4.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "^", color = color_list[3], s = 9, label = plot4_label, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data4['x_manual'], y1 = np.array(data4[conf_int_fieldnames[0]].transpose()), y2 = np.array(data4[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[3], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[3])
    else:
        error_bar4 = ax.errorbar(x = data4['x_manual'], y = data4[y_fieldname], yerr = np.array(data4[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[3])
    data4.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[3], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    # Plot 5
    ax = data5.plot(kind = 'scatter', x = 'x_manual', y = y_fieldname, marker = "v", color = color_list[4], s = 9, label = plot5_label, ax = ax)
    if error_fill == 'yes':
        ax.fill_between(x = data5['x_manual'], y1 = np.array(data5[conf_int_fieldnames[0]].transpose()), y2 = np.array(data5[conf_int_fieldnames[1]].transpose()), alpha = 0.1, facecolor=color_list[4], linestyle='dotted', linewidth = 1.5, edgecolor = color_list[4])
    else:
        error_bar5 = ax.errorbar(x = data5['x_manual'], y = data5[y_fieldname], yerr = np.array(data5[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[4])
    data5.plot.line(x = 'x_manual', y = y_fieldname, linestyle = 'dashed', color = color_list[4], label = '_nolegend_', alpha = alpha, ax = ax) 
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.xticks(data1['x_manual'] - 2*(offset/2), data1[x_fieldname]) # set labels manually
    if xtick_rotation < 80:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, rotation_mode='anchor', ha="right")
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    # plt.yticks(data2['x_manual'] + (offset), data2[x_fieldname]) # set labels manually 
    # ax.xaxis.label.set_visible(False) # hide y axis title
    ax.set_xlabel(xlabel)
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axhline(y = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    
    if poisson_reg == 'yes':
        ax.set_ylabel('Relative risk ratio')
    else:
        ax.set_ylabel('Odds ratio')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    
    
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way

    if x_logscale == 'yes':
        ax.set_yscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twinx()
        secax.set_yscale('linear')
        secax.set_ylim(np.log(ylims[0]), np.log(ylims[1]))
        secax.grid(False, which="both", axis = 'y')
        secax.set_ylabel('Coefficient')
        secax.tick_params(axis='y', which='major', labelsize=10)
        secax.set_ylabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', labelsize=8.2)
        
        xlim_thresh = 3.5
        if ylims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',                                   
                                      '1.1','','1.3','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif ylims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '3', '4', '', '6', '', '8', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
            
        ax.yaxis.set_minor_formatter(x_formatter)
        ax.yaxis.set_minor_locator(x_locator)
        
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'y')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if x_major_tick > 0:
        secax.yaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.yaxis.set_minor_locator(MultipleLocator(x_minor_tick))
        # secax.xaxis.grid(True, which='minor', color='#EEEEEE')

    return data1, data2, data3, data4, data5


#%% Load data
# -----------------------------------------------------------------------------
# Load Labour Force Survey quarterly datasets
# -----------------------------------------------------------------------------
# Specify which columns to keep
lfs_col_select_pre2021 = ['AGE', 'AGES', 'SEX', 'CRY12', 'ETHEWEUL', 'ETHGBEUL', 'ETHUKEUL', # Demographics - age, country of birth, ethnic group
                          'SMSOC101', 'SMSOC103', 'SM_NSEC10', 'SMISCO8MMN', 'SMEARNER', 'SMHCOMP', # Family background
                          'SC10MMN','NSECM10','NSECMJ10', 'SOC10M', # Current occupation
                          'SC10LMN', # Occupation at last job
                          'INDC07M', 'INDD07M', # Industry class (detailed) & division (broader)
                          'INECAC05','ILODEFR', # Economic activity
                          'REGWKR','URESMC', # Region of work and residence
                           'BANDG', # Gross pay estimate band (if specific not known)
                           'HOURPAY', # Gross hourly pay
                          'SNGDEGB', # Subject of degree (single-subject degree)
                          'LNGLST', # Long term health condition
                          'CASENOP', 'HSERIALP', # case number, household number
                          'HOUT', # Other admin variables to look for systematic reasons for missing SEB data
                          ] # specify which columns to keep across dataset - for older datasets using SOC 2010 job classification

lfs_col_select_post2021 = ['AGE', 'AGES', 'SEX', 'CRY12', 'ETHEWEUL', 'ETHGBEUL', 'ETHUKEUL', # Demographics - age, country of birth, ethnic group
                          'SMSOC201', 'SMSOC203', 'SM_NSEC20', 'SMEARNER', 'SMHCOMP', # Family background
                          'SC20MMN','NSECM20','NSECMJ20', 'SOC20M', # Current occupation
                          'SC20LMN', # Occupation at last job
                          'INDC07M', 'INDD07M', # Industry class (detailed) & division (broader)
                          'INECAC05', 'ILODEFR', # Economic activity
                          'REGWKR','URESMC', # Region of work and residence
                           'BANDG', # Gross pay estimate band (if specific not known)
                           'HOURPAY', # Gross hourly pay
                          'SNGDEGB', # Subject of degree (single-subject degree)
                          'LNGLST', # Long term health condition
                          'CASENOP', 'HSERIALP', # case number, household number
                          'HOUT', # Other admin variables to look for systematic reasons for missing SEB data
                          ] # specify which columns to keep across dataset - for newer datasets using SOC 2020 job classification

# Degree related columns (when degree is multi-subject)
degree_cols_pre2016 = ['FDCMBMA', 'CMBHDMA']
degree_cols_post2016 = ['UNCOMBMA', 'HICOMBMA']

# 2014
data_lfs_2014 = pd.read_csv(r"lfsp_js14_eul_pwt18.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_pre2021 + ['PWT18','HIQUL11D','HEALTH'] + degree_cols_pre2016 # Plus person weight field
                    )
data_lfs_2014['lfs_year'] = 2014
data_lfs_2014_cols = data_lfs_2014.columns.to_list()
# 2015 
data_lfs_2015 = pd.read_csv(r"lfsp_js15_eul_pwt18.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_pre2016 # Plus person weight field
                    )
data_lfs_2015['lfs_year'] = 2015
data_lfs_2015_cols = data_lfs_2015.columns.to_list()
# 2016 
data_lfs_2016 = pd.read_csv(r"lfsp_js16_eul_pwt18.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_post2016 # Plus person weight field
                    )
data_lfs_2016['lfs_year'] = 2016
data_lfs_2016_cols = data_lfs_2016.columns.to_list()
# 2017
data_lfs_2017 = pd.read_csv(r"lfsp_js17_eul_pwt18.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_post2016 # Plus person weight field
                    )
data_lfs_2017['lfs_year'] = 2017
data_lfs_2017_cols = data_lfs_2017.columns.to_list()
# 2018
data_lfs_2018 = pd.read_csv(r"lfsp_js18_eul_pwt18.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_post2016 # Plus person weight field
                    )
data_lfs_2018['lfs_year'] = 2018
data_lfs_2018_cols = data_lfs_2018.columns.to_list()
# 2019
data_lfs_2019 = pd.read_csv(r"lfsp_js19_eul_pwt18.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_post2016 # Plus person weight field
                    )
data_lfs_2019['lfs_year'] = 2019
data_lfs_2019_cols = data_lfs_2019.columns.to_list()
# 2020
data_lfs_2020 = pd.read_csv(r"lfsp_js20_eul_pwt22.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_pre2021 + ['PWT22','HIQUL15D','HEALTH20'] + degree_cols_post2016 # Plus person weight field
                    )
data_lfs_2020['lfs_year'] = 2020
data_lfs_2020_cols = data_lfs_2020.columns.to_list()
# 2021
data_lfs_2021 = pd.read_csv(r"lfsp_js21_eul_pwt22.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_post2021 + ['PWT22','HIQUL15D','HEALTH20'] + degree_cols_post2016 # Plus person weight field
                    )
data_lfs_2021['lfs_year'] = 2021
data_lfs_2021_cols = data_lfs_2021.columns.to_list()
# 2022
data_lfs_2022 = pd.read_csv(r"lfsp_js22_eul_pwt23.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_post2021 + ['PWT23','HIQUL22D','HEALTH20'] + degree_cols_post2016 # Plus person weight field
                    )
data_lfs_2022['lfs_year'] = 2022
data_lfs_2022_cols = data_lfs_2022.columns.to_list()
# 2023
data_lfs_2023 = pd.read_csv(r"lfsp_js23_eul_pwt23.tab", sep = '\t', 
                    # nrows = 10000, 
                    usecols = lfs_col_select_post2021 + ['PWT23','HIQUL22D','HEALTH20'] + degree_cols_post2016 # Plus person weight field
                    )
data_lfs_2023['lfs_year'] = 2023
data_lfs_2023_cols = data_lfs_2023.columns.to_list()


dictionary = {}


#%% Combining Labour Force Survey datasets
# -----------------------------------------------------------------------------
# Specify which columns to keep
lfs_col_select_pre2021 = lfs_col_select_pre2021 + ['lfs_year']
lfs_col_select_post2021 = lfs_col_select_post2021 + ['lfs_year']

# -----------------------------------------------------------------------------
# Convert string to float for SM_NSEC10 2017-2019
data_lfs_2017['SM_NSEC10'] = pd.to_numeric(data_lfs_2017['SM_NSEC10'], errors = 'coerce')
data_lfs_2017['SM_NSEC10'] = data_lfs_2017['SM_NSEC10'].fillna(-9)

data_lfs_2018['SM_NSEC10'] = pd.to_numeric(data_lfs_2018['SM_NSEC10'], errors = 'coerce')
data_lfs_2018['SM_NSEC10'] = data_lfs_2018['SM_NSEC10'].fillna(-9)

data_lfs_2019['SM_NSEC10'] = pd.to_numeric(data_lfs_2019['SM_NSEC10'], errors = 'coerce')
data_lfs_2019['SM_NSEC10'] = data_lfs_2019['SM_NSEC10'].fillna(-9)

# -----------------------------------------------------------------------------
# Append datasets together
data_lfs = data_lfs_2014[lfs_col_select_pre2021 + ['PWT18','HIQUL11D'] + degree_cols_pre2016].copy()
data_lfs = data_lfs.append(data_lfs_2015[lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_pre2016]).reset_index(drop = True)
data_lfs = data_lfs.append(data_lfs_2016[lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_post2016]).reset_index(drop = True)
data_lfs = data_lfs.append(data_lfs_2017[lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_post2016]).reset_index(drop = True)
data_lfs = data_lfs.append(data_lfs_2018[lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_post2016]).reset_index(drop = True)
data_lfs = data_lfs.append(data_lfs_2019[lfs_col_select_pre2021 + ['PWT18','HIQUL15D','HEALTH'] + degree_cols_post2016]).reset_index(drop = True)
data_lfs = data_lfs.append(data_lfs_2020[lfs_col_select_pre2021 + ['PWT22','HIQUL15D','HEALTH20'] + degree_cols_post2016]).reset_index(drop = True)
data_lfs = data_lfs.append(data_lfs_2021[lfs_col_select_post2021 + ['PWT22','HIQUL15D','HEALTH20'] + degree_cols_post2016]).reset_index(drop = True)
data_lfs = data_lfs.append(data_lfs_2022[lfs_col_select_post2021 + ['PWT23','HIQUL22D','HEALTH20'] + degree_cols_post2016]).reset_index(drop = True)
data_lfs = data_lfs.append(data_lfs_2023[lfs_col_select_post2021 + ['PWT23','HIQUL22D','HEALTH20'] + degree_cols_post2016]).reset_index(drop = True)

# -----------------------------------------------------------------------------
# Delete individual datasets
del data_lfs_2014, data_lfs_2015, data_lfs_2016, data_lfs_2017, data_lfs_2018, data_lfs_2019, data_lfs_2020, data_lfs_2021, data_lfs_2022, data_lfs_2023 

# -----------------------------------------------------------------------------
# Create ID column from index
data_lfs = data_lfs.reset_index(drop = True)
data_lfs['id'] = data_lfs.index
test = data_lfs['id']

#%% Processing Labour Force Survey dataset
# -----------------------------------------------------------------------------
# Create combined person weight field
data_lfs.loc[(data_lfs['lfs_year'] < 2020), 'person_weight'] = data_lfs['PWT18']
data_lfs.loc[(data_lfs['lfs_year'].isin([2020,2021])), 'person_weight'] = data_lfs['PWT22']
data_lfs.loc[(data_lfs['lfs_year'].isin([2022,2023])), 'person_weight'] = data_lfs['PWT23']

# -----------------------------------------------------------------------------
# Create combined education level field
data_lfs.loc[(data_lfs['lfs_year'] < 2015), 'education_level'] = data_lfs['HIQUL11D']
data_lfs.loc[(data_lfs['lfs_year'].isin([2015,2016,2017,2018,2019,2020,2021])), 'education_level'] = data_lfs['HIQUL15D']
data_lfs.loc[(data_lfs['lfs_year'].isin([2022,2023])), 'education_level'] = data_lfs['HIQUL22D']


# -----------------------------------------------------------------------------
# Create custom age group bands
value_lims_10yrbands = [0,10,20,30,40,50,60,70,80,90]
data_lfs = add_grouped_field(data_lfs,'AGE','age_10yrbands',value_lims_10yrbands)
value_lims_5yrbands = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
data_lfs = add_grouped_field(data_lfs,'AGE','age_5yrbands',value_lims_5yrbands)

# -----------------------------------------------------------------------------
# Use age and year of survey to derive years
# Year of birth
data_lfs['year_of_birth'] = data_lfs['lfs_year'] - data_lfs['AGE'] 
# Group into bands
value_lims_5yrbands = [1910,1915,1920,1925,1930,1935,1940,1945,1950,1955,1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020]
value_lims_5yrbands_midpoint = pd.Series(value_lims_5yrbands) + 2
value_lims_10yrbands = [1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020]
value_lims_3yrbands = [1950,1953,1956,1959,1962,1965,1968,1971,1974,1977,1980,1983,1986,1989,1992,1995,1998,2001,2004,2007,2010,2013,2016,2019]
value_lims_3yrbands_midpoint = pd.Series(value_lims_3yrbands) + 1
data_lfs = add_grouped_field(data_lfs,'year_of_birth','yearofbirth_5yrbands',value_lims_5yrbands)
data_lfs = add_grouped_field(data_lfs,'year_of_birth','yearofbirth_10yrbands',value_lims_10yrbands)


# Year at age 18 - likely age when applying for and entering university medical school
data_lfs['year_age18'] = data_lfs['year_of_birth'] + 18
# Group into bands
data_lfs = add_grouped_field(data_lfs,'year_age18','year_age18_5yrbands',value_lims_5yrbands)
data_lfs = add_grouped_field(data_lfs,'year_age18','year_age18_10yrbands',value_lims_10yrbands)
data_lfs = add_grouped_field(data_lfs,'year_age18','year_age18_3yrbands',value_lims_3yrbands)
data_lfs = add_grouped_field(data_lfs,'year_age18','year_age18_3yrbands',value_lims_3yrbands)

dictionary['year_age18_3yrbands_midpoint'] = {'01: 1950-1953':1951, 
                                              '02: 1953-1956':1954,
                                              '03: 1956-1959':1957,
                                              '04: 1959-1962':1960,
                                              '05: 1962-1965':1963,
                                              '20: 2007-2010':2008,
                                              '19: 2004-2007':2005,
                                              '18: 2001-2004':2002,
                                              '16: 1995-1998':1996,
                                              '15: 1992-1995':1993,
                                              '17: 1998-2001':1999,
                                              '14: 1989-1992':1990,
                                              '07: 1968-1971':1969,
                                              '08: 1971-1974':1972,
                                              '09: 1974-1977':1975, 
                                              '13: 1986-1989':1987,
                                              '10: 1977-1980':1978,
                                              '06: 1965-1968':1966,
                                              '12: 1983-1986':1984,
                                              '11: 1980-1983':1981,
                                              '21: 2010-2013':2011,
                                              '22: 2013-2016':2014,
                                              '23: 2016-2019':2017,
                                              '24: 2019+':2020,
                                              }
dictionary['year_age18_5yrbands_midpoint'] = {'05: 1930-1935':1932,
                                              '06: 1935-1940':1937,
                                              '07: 1940-1945':1942,
                                              '08: 1945-1950':1947,
                                              '09: 1950-1955':1952,
                                              '10: 1955-1960':1957,
                                              '11: 1960-1965':1962,
                                              '21: 2010-2015':2012,
                                              '22: 2015-2020':2017,
                                              '20: 2005-2010':2007,
                                              '19: 2000-2005':2002,
                                              '18: 1995-2000':1997,
                                              '17: 1990-1995':1992,
                                              '13: 1970-1975':1972,
                                              '16: 1985-1990':1987,
                                              '14: 1975-1980':1977,
                                              '12: 1965-1970':1967,
                                              '15: 1980-1985':1982,
                                              '23: 2020+':2022,
                                              }

# Manually aggregate edge bands - for current doctor as outcome
data_lfs['year_age18_5yrband_agg_doctoronly'] = data_lfs['year_age18_5yrbands'].copy()
data_lfs['year_age18_10yrband_agg_doctoronly'] = data_lfs['year_age18_10yrbands'].copy()
data_lfs.loc[(data_lfs['year_age18'] < 1970),'year_age18_5yrband_agg_doctoronly'] = '12: Pre-1970'
data_lfs.loc[(data_lfs['year_age18'] >= 2010),'year_age18_5yrband_agg_doctoronly'] = '21: Post-2010'
data_lfs.loc[(data_lfs['year_age18'] < 1970),'year_age18_10yrband_agg_doctoronly'] = '06: Pre-1970'
data_lfs.loc[(data_lfs['year_age18'] >= 2010),'year_age18_10yrband_agg_doctoronly'] = '11: Post-2010'

# Manually aggregate edge bands - for current doctor OR medicine degree as outcome
data_lfs['year_age18_5yrband_agg_doctorORdegree'] = data_lfs['year_age18_5yrbands'].copy()
data_lfs['year_age18_10yrband_agg_doctorORdegree'] = data_lfs['year_age18_10yrbands'].copy()
data_lfs.loc[(data_lfs['year_age18'] < 1965),'year_age18_5yrband_agg_doctorORdegree'] = '11: Pre-1965'
data_lfs.loc[(data_lfs['year_age18'] >= 2010),'year_age18_5yrband_agg_doctorORdegree'] = '21: Post-2010'
data_lfs.loc[(data_lfs['year_age18'] < 1970),'year_age18_10yrband_agg_doctorORdegree'] = '06: Pre-1970'
data_lfs.loc[(data_lfs['year_age18'] >= 2010),'year_age18_10yrband_agg_doctorORdegree'] = '11: Post-2010'


# -----------------------------------------------------------------------------
# Create aggregated country of birth field to combine categories with small values
data_lfs['CRY12'] = data_lfs['CRY12'].replace({-8:997}) # Recode no answer to other, as low sample numbers causes singular matrix
dictionary['CRY12'] = {921:921,
                       922:922,
                       923:923,
                       924:924,
                       926:921, # Re-code unknown UK to England as the modal category
                       997:997,
                       356:356,
                       616:997, # Re-code Poland to other
                       586:997,
                       372:997, # Re-code Republic of Ireland to other
                       -9:-9,
                       -8:-9,
                       }
data_lfs['countryofbirth_agg'] = data_lfs['CRY12'].map(dictionary['CRY12'])

# Re-code all UK nations to 921, and all non-UK to 997
dictionary['CRY12_binary'] = {921:921,
                       922:921,
                       923:921,
                       924:921,
                       926:921, 
                       997:997,
                       356:997,
                       616:997, # Re-code Poland to other
                       586:997,
                       372:997, # Re-code Republic of Ireland to other
                       -9:-9,
                       -8:-9,
                       }
data_lfs['countryofbirth_binary'] = data_lfs['CRY12'].map(dictionary['CRY12_binary'])

# -----------------------------------------------------------------------------
# Create aggregated ethnic group field to combine categories with small values
dictionary['ETHUKEUL'] = {1:1,
                           2:9, # Re-code Mixed/Multiple under Any other
                           3:3,
                           4:4,
                           5:7, # Re-code Bangladeshi under Other asian as small numbers
                           6:7, # Re-code Chinese under Other asian as small numbers
                           7:7,
                           8:8,
                           9:9,
                           -8:-8,
                           }
data_lfs['ethnicgroup_agg'] = data_lfs['ETHUKEUL'].map(dictionary['ETHUKEUL'])

# Code to category level used in Census
dictionary['ETHUKEUL_category'] = {1:1,
                           2:2, # Mixed/Multiple
                           3:7, # Asian/Asian British
                           4:7,
                           5:7, 
                           6:7, 
                           7:7,
                           8:8,
                           9:9,
                           -8:9, # Re-code missing to Any other to avoid model failing due to small numbers
                           }
data_lfs['ethnicgroup_category'] = data_lfs['ETHUKEUL'].map(dictionary['ETHUKEUL_category'])

# Code to binary racially minoritised groups vs White
dictionary['ETHUKEUL_binary'] = {1:1,
                           2:9, # Mixed/Multiple
                           3:9, # Asian/Asian British
                           4:9,
                           5:9, 
                           6:9, 
                           7:9,
                           8:9,
                           9:9,
                           -8:9, # Re-code missing to Any other to avoid model failing due to small numbers
                           }
data_lfs['ethnicgroup_binary'] = data_lfs['ETHUKEUL'].map(dictionary['ETHUKEUL_binary'])


# -----------------------------------------------------------------------------
# Map 2010 SOC minor group to 2020 based on relationship table file - soc2010soc2020relationshiptablesjuly2021
# https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/therelationshipbetweenstandardoccupationalclassification2010andstandardoccupationalclassification2020
# Map based on whichever 2020 category has the majority from the 2010 category - file gives % splits - most are close to 100%
dictionary['SOC2010_to_2020'] = {-9:-9,
                                 -8:-8,
                                111:111,
                                112:112,
                                113:113,
                                115:113,
                                116:124,
                                117:116,
                                118:117,
                                119:115,
                                121:121,
                                122:122,
                                124:123,
                                125:125,
                                211:211,
                                212:212,
                                213:213,
                                214:215,
                                215:216,
                                221:221,
                                222:222,
                                223:223,
                                231:231,
                                241:241,
                                242:243, # big split
                                243:245,
                                244:246,
                                245:247,
                                246:248,
                                247:249,
                                311:311,
                                312:312,
                                313:313,
                                321:321,
                                323:322,
                                331:331,
                                341:341,
                                342:342, # big split
                                344:343,
                                351:351,
                                352:352, # big split
                                353:353, # big split
                                354:355,
                                355:511,
                                356:357, # big split
                                411:411,
                                412:412,
                                413:413,
                                415:415,
                                416:414,
                                421:421,
                                511:511,
                                521:521,
                                522:522,
                                523:523,
                                524:524,
                                525:525,
                                531:531,
                                532:532,
                                533:533,
                                541:541,
                                542:542,
                                543:543,
                                544:544,
                                612:611,
                                613:612,
                                614:613,
                                621:621,
                                622:622,
                                623:623,
                                624:624,
                                711:711,
                                712:712,
                                713:713,
                                721:721,
                                722:414, # big split
                                811:811,
                                812:813, # big split
                                813:814,
                                814:815,
                                821:821,
                                822:822,
                                823:823,
                                911:911,
                                912:912,
                                913:913,
                                921:921,
                                923:922,
                                924:923,
                                925:924,
                                926:925,
                                927:926,
                                }
data_lfs['SMSOC103_mapped'] = data_lfs['SMSOC103'].map(dictionary['SOC2010_to_2020'])
data_lfs['SC10MMN_mapped'] = data_lfs['SC10MMN'].map(dictionary['SOC2010_to_2020'])

# -----------------------------------------------------------------------------
# Create unified job classification variables to combine those based on SOC 2010 and 2020 job classifications
# Family main earner NSSEC
year_NSSEC_change = 2021 # dataset year where changed from NSSEC 2010 to 2020 version
data_lfs.loc[(data_lfs['lfs_year'] < year_NSSEC_change), 'nssec_familybackground_full'] = data_lfs['SM_NSEC10']
data_lfs.loc[(data_lfs['lfs_year'] >= year_NSSEC_change), 'nssec_familybackground_full'] = data_lfs['SM_NSEC20']
# Family main earner occupation
data_lfs.loc[(data_lfs['lfs_year'] < year_NSSEC_change), 'SOC_occupation_familybackground'] = data_lfs['SMSOC103_mapped']
data_lfs.loc[(data_lfs['lfs_year'] >= year_NSSEC_change), 'SOC_occupation_familybackground'] = data_lfs['SMSOC203']

# Current NSSEC
data_lfs.loc[(data_lfs['lfs_year'] < year_NSSEC_change), 'nssec_current_full'] = data_lfs['NSECM10']
data_lfs.loc[(data_lfs['lfs_year'] >= year_NSSEC_change), 'nssec_current_full'] = data_lfs['NSECM20']
# Current Occupation (minor group)
data_lfs.loc[(data_lfs['lfs_year'] < year_NSSEC_change), 'SOC_occupation_current_minor'] = data_lfs['SC10MMN_mapped']
data_lfs.loc[(data_lfs['lfs_year'] >= year_NSSEC_change), 'SOC_occupation_current_minor'] = data_lfs['SC20MMN']
# Current Occupation (detailed occupation) - NEED TO MAP BETWEEN 2010 and 2020 FOR ANY DETAILED ANALYSIS
data_lfs.loc[(data_lfs['lfs_year'] < year_NSSEC_change), 'SOC_occupation_current_detailed'] = data_lfs['SOC10M']
data_lfs.loc[(data_lfs['lfs_year'] >= year_NSSEC_change), 'SOC_occupation_current_detailed'] = data_lfs['SOC20M']

# Last job Occupation (minor group)
data_lfs.loc[(data_lfs['lfs_year'] < year_NSSEC_change), 'SOC_occupation_last_minor'] = data_lfs['SC10LMN']
data_lfs.loc[(data_lfs['lfs_year'] >= year_NSSEC_change), 'SOC_occupation_last_minor'] = data_lfs['SC20LMN']


# -----------------------------------------------------------------------------
# Create unified variables relating to main subject of undergraduate degree if degree was multi-subject
year_degree_change = 2016
data_lfs.loc[(data_lfs['lfs_year'] < year_degree_change), 'undergrad_main'] = data_lfs['FDCMBMA']
data_lfs.loc[(data_lfs['lfs_year'] >= year_degree_change), 'undergrad_main'] = data_lfs['UNCOMBMA']

# main subject of higher degree if degree was multi-subject
data_lfs.loc[(data_lfs['lfs_year'] < year_degree_change), 'higherdegree_main'] = data_lfs['CMBHDMA']
data_lfs.loc[(data_lfs['lfs_year'] >= year_degree_change), 'higherdegree_main'] = data_lfs['HICOMBMA']


# -----------------------------------------------------------------------------
# Flag if retired - INECAC05 == 20, 31 
data_lfs.loc[(data_lfs['INECAC05'].isin([20,31])), 'Flag_Retired'] = 1

# -----------------------------------------------------------------------------
# Recode family NS-SEC to separate true missing/unknown from 'not applicable' situations where no one was earning, or wasn't living with parents from missing data
# Create true missing/unknown = -99 only if 'SMEARNER', 'SMHCOMP' and SMSOC203 all == -9 or -8. 
data_lfs.loc[(data_lfs['SMEARNER'].isin([-9,-8]))
             & (data_lfs['SMHCOMP'].isin([-9,-8]))
             & (data_lfs['SOC_occupation_familybackground'].isin([-9,-8])), 'nssec_familybackground_full'] = -99

# If not living with parents, SMHCOMP = 3, set to -95 = 'Does not apply'
data_lfs.loc[(data_lfs['SMHCOMP'].isin([3]))
             & (data_lfs['SOC_occupation_familybackground'].isin([-9,-8])), 'nssec_familybackground_full'] = -95

# If living with parents or family SMHCOMP = 1,2 or someone was earning SMEARNER 1,2,3,4, but family occupation missing = -8,-9, then set family background as -90 = 'Do not know'
data_lfs.loc[((data_lfs['SMEARNER'].isin([1,2,3,4]))
             | (data_lfs['SMHCOMP'].isin([1,2])))
             & (data_lfs['SOC_occupation_familybackground'].isin([-9,-8,999])), 'nssec_familybackground_full'] = -90

# If living with parents or family SMHCOMP = 1,2 or someone was earning SMEARNER 1,2,3,4, but family ns-sec missing = -8, (even if occupation not missing), then set family background as -90 = 'Do not know'
data_lfs.loc[((data_lfs['SMEARNER'].isin([1,2,3,4]))
             | (data_lfs['SMHCOMP'].isin([1,2])))
             & (data_lfs['nssec_familybackground_full'].isin([-8])), 'nssec_familybackground_full'] = -90

# If no-one was earning, SMEARNER = 5, set to NS-SEC full = 14 - long-term unemployed category
data_lfs.loc[(data_lfs['SMEARNER'].isin([5])), 'nssec_familybackground_full'] = 14


# -----------------------------------------------------------------------------
# Processing parent class measure SM_NSEC10 - map to L number based on https://www.ons.gov.uk/methodology/classificationsandstandards/otherclassifications/thenationalstatisticssocioeconomicclassificationnssecrebasedonsoc2010 table 2

# NS-SEC of family main earner at age 14 - 3, 5, 8 cat
test = data_lfs.groupby(['lfs_year','nssec_familybackground_full'])['lfs_year'].count()
# 9-category - 1 split into 1.1 and 1.2
data_lfs.loc[(data_lfs['nssec_familybackground_full'].isin([1,2])), 'nssec_familybackground_9cat'] = '1.1'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 3) & (data_lfs['nssec_familybackground_full'] < 4), 'nssec_familybackground_9cat'] = '1.2'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 4) & (data_lfs['nssec_familybackground_full'] < 7), 'nssec_familybackground_9cat'] = '2'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 7) & (data_lfs['nssec_familybackground_full'] < 8), 'nssec_familybackground_9cat'] = '3'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 8) & (data_lfs['nssec_familybackground_full'] < 10), 'nssec_familybackground_9cat'] = '4'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 10) & (data_lfs['nssec_familybackground_full'] < 12), 'nssec_familybackground_9cat'] = '5'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 12) & (data_lfs['nssec_familybackground_full'] < 13), 'nssec_familybackground_9cat'] = '6'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 13) & (data_lfs['nssec_familybackground_full'] < 14), 'nssec_familybackground_9cat'] = '7'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 14) & (data_lfs['nssec_familybackground_full'] < 15), 'nssec_familybackground_9cat'] = '8'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -99), 'nssec_familybackground_9cat'] = 'Unknown'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -95), 'nssec_familybackground_9cat'] = 'Not living with family'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -90), 'nssec_familybackground_9cat'] = 'Do not know'

# 8-category
data_lfs.loc[(data_lfs['nssec_familybackground_full'].isin([1,2])), 'nssec_familybackground_8cat'] = '1'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 3) & (data_lfs['nssec_familybackground_full'] < 4), 'nssec_familybackground_8cat'] = '1'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 4) & (data_lfs['nssec_familybackground_full'] < 7), 'nssec_familybackground_8cat'] = '2'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 7) & (data_lfs['nssec_familybackground_full'] < 8), 'nssec_familybackground_8cat'] = '3'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 8) & (data_lfs['nssec_familybackground_full'] < 10), 'nssec_familybackground_8cat'] = '4'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 10) & (data_lfs['nssec_familybackground_full'] < 12), 'nssec_familybackground_8cat'] = '5'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 12) & (data_lfs['nssec_familybackground_full'] < 13), 'nssec_familybackground_8cat'] = '6'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 13) & (data_lfs['nssec_familybackground_full'] < 14), 'nssec_familybackground_8cat'] = '7'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] >= 14) & (data_lfs['nssec_familybackground_full'] < 15), 'nssec_familybackground_8cat'] = '8'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -99), 'nssec_familybackground_8cat'] = 'Unknown'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -95), 'nssec_familybackground_8cat'] = 'Not living with family'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -90), 'nssec_familybackground_8cat'] = 'Do not know'

# 5-category
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['1','2'])), 'nssec_familybackground_5cat'] = '1-2'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['3'])), 'nssec_familybackground_5cat'] = '3'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['4'])), 'nssec_familybackground_5cat'] = '4'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['5'])), 'nssec_familybackground_5cat'] = '5'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['6','7'])), 'nssec_familybackground_5cat'] = '6-7'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['8'])), 'nssec_familybackground_5cat'] = '8'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -99), 'nssec_familybackground_5cat'] = 'Unknown'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -95), 'nssec_familybackground_5cat'] = 'Not living with family'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -90), 'nssec_familybackground_5cat'] = 'Do not know'

# 4-category
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['1','2'])), 'nssec_familybackground_4cat'] = '1-2'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['3','4'])), 'nssec_familybackground_4cat'] = '3-4'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['5','6','7'])), 'nssec_familybackground_4cat'] = '5-7'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['8'])), 'nssec_familybackground_4cat'] = '8'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -99), 'nssec_familybackground_4cat'] = 'Unknown'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -95), 'nssec_familybackground_4cat'] = 'Not living with family'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -90), 'nssec_familybackground_4cat'] = 'Do not know'

# 3-category - grouping 3 and 8
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['1','2'])), 'nssec_familybackground_3cat'] = '1-2'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['3','4'])), 'nssec_familybackground_3cat'] = '3-4'
data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['5','6','7','8'])), 'nssec_familybackground_3cat'] = '5-8'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -99), 'nssec_familybackground_3cat'] = 'Unknown'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -95), 'nssec_familybackground_3cat'] = '5-8' # Add this group to not working group, as was not living with family, and this group looks most similar to NS-SEC 5-8 family background in terms of current employment, and tends to have higher sickness rates - perhaps was in care or hospital # 'Does not apply'
data_lfs.loc[(data_lfs['nssec_familybackground_full'] == -90), 'nssec_familybackground_3cat'] = 'Do not know'


test = data_lfs[['nssec_familybackground_full','nssec_familybackground_8cat','nssec_familybackground_5cat','nssec_familybackground_3cat']]

# -----------------------------------------------------------------------------
# Flag if parents were in certain occupations 
# Category including doctor - SOC_occupation_familybackground = 221
# In 2010, 221 includes other occupations as well as doctors - narrow down by filtering for SMISCO8MMN == 221 ('Medical doctors') also 
# In 2020, 221 includes generalist and specialist medical practioner only - so SOC_occupation_familybackground == 221 is sufficient
data_lfs.loc[(data_lfs['lfs_year'] < year_NSSEC_change) 
             & (data_lfs['SOC_occupation_familybackground'] == 221)
             & (data_lfs['SMISCO8MMN'] == 221)
             ,'Flag_MainEarner_221_Doctor'] = 1
data_lfs.loc[(data_lfs['lfs_year'] >= year_NSSEC_change) 
             & (data_lfs['SOC_occupation_familybackground'] == 221)
             ,'Flag_MainEarner_221_Doctor'] = 1


# Create edited version of SOC_occupation_familybackground which separate doctors using same logic (need to do this as category 221 in 2010 is broader so need to use SMISCO8MMN to isolate doctors)
data_lfs['SOC_occupation_familyback_separatedoctor'] = data_lfs['SOC_occupation_familybackground'].copy()
data_lfs.loc[(data_lfs['Flag_MainEarner_221_Doctor'] == 1), 'SOC_occupation_familyback_separatedoctor'] = 2211

# -----------------------------------------------------------------------------
# Create versions of family background NSSEC which have separate category if main earner was Doctor
separate_doctor = 1
if separate_doctor == 1:    
    data_lfs['nssec_familybackground_separatedoctor_9cat'] = data_lfs['nssec_familybackground_9cat'].copy()
    data_lfs['nssec_familybackground_separatedoctor_8cat'] = data_lfs['nssec_familybackground_8cat'].copy()
    data_lfs['nssec_familybackground_separatedoctor_5cat'] = data_lfs['nssec_familybackground_5cat'].copy()
    data_lfs['nssec_familybackground_separatedoctor_4cat'] = data_lfs['nssec_familybackground_4cat'].copy()
    data_lfs['nssec_familybackground_separatedoctor_3cat'] = data_lfs['nssec_familybackground_3cat'].copy()
    
    data_lfs.loc[(data_lfs['nssec_familybackground_9cat'].isin(['1.2','2']))
                 & (data_lfs['Flag_MainEarner_221_Doctor'] == 1), 'nssec_familybackground_separatedoctor_9cat'] = 'Doctor'
    data_lfs.loc[(data_lfs['nssec_familybackground_8cat'].isin(['1','2']))
                 & (data_lfs['Flag_MainEarner_221_Doctor'] == 1), 'nssec_familybackground_separatedoctor_8cat'] = 'Doctor'
    data_lfs.loc[(data_lfs['nssec_familybackground_5cat'].isin(['1-2']))
                 & (data_lfs['Flag_MainEarner_221_Doctor'] == 1), 'nssec_familybackground_separatedoctor_5cat'] = 'Doctor'
    data_lfs.loc[(data_lfs['nssec_familybackground_4cat'].isin(['1-2']))
                 & (data_lfs['Flag_MainEarner_221_Doctor'] == 1), 'nssec_familybackground_separatedoctor_4cat'] = 'Doctor'
    data_lfs.loc[(data_lfs['nssec_familybackground_3cat'].isin(['1-2']))
                 & (data_lfs['Flag_MainEarner_221_Doctor'] == 1), 'nssec_familybackground_separatedoctor_3cat'] = 'Doctor'


# -----------------------------------------------------------------------------
# Flag if current main job is doctor 'medical practioner' - SOC_occupation_current_minor (SC10M = 2211 or SC20M = 2211 | 2212)
data_lfs.loc[(data_lfs['lfs_year'] < year_NSSEC_change) 
             & (data_lfs['SOC_occupation_current_detailed'] == 2211)
             ,'Flag_CurrentDoctor'] = 1
data_lfs.loc[(data_lfs['lfs_year'] >= year_NSSEC_change) 
             & (data_lfs['SOC_occupation_current_detailed'].isin([2211,2212]))
             ,'Flag_CurrentDoctor'] = 1


# -----------------------------------------------------------------------------
# Combine central and inner london, as not clear what central london includes
data_lfs.loc[(data_lfs['REGWKR'] == 8),'REGWKR'] = 9

# -----------------------------------------------------------------------------
# Mapping regions to be consistent across years 
# Mapping for region of place of work
dictionary['regionofwork'] = {1:'01_TyneAndWear',
                                   2:'02_RestofNorthEast',
                                   3:'03_SouthYorkshire',
                                   4:'04_WestYorkshire',
                                   5:'05_RestofYorksHumberside',
                                   6:'06_EastMidlands',
                                   7:'07_EastofEngland',
                                   # 8:'',
                                   9:'09_InnerLondon',
                                   # 10:'',
                                   11:'11_OuterLondon',
                                   12:'12_SouthEast',
                                   13:'13_SouthWest',
                                   14:'14_WestMidlandsMetropolitan',
                                   15:'15_RestofWestMidlands',
                                   16:'16_GreaterManchester',
                                   17:'17_Merseyside',
                                   18:'18_RestofNorthWest',
                                   19:'19_Wales',
                                   20:'20_Strathclyde',
                                   21:'21_RestofScotland',
                                   22:'22_NorthernIreland',
                                   23:'23_OutsideUK',
                                   -8:'m8_NoAnswer',
                                   -9:'m9_DoesNotApply',
                                   }
data_lfs['regionofwork'] = data_lfs['REGWKR'].map(dictionary['regionofwork'])

# Mapping for region of residence - numbering slightly different
dictionary['regionofresidence'] = {1:'01_TyneAndWear',
                                   2:'02_RestofNorthEast',
                                   3:'03_SouthYorkshire',
                                   4:'04_WestYorkshire',
                                   5:'05_RestofYorksHumberside',
                                   6:'06_EastMidlands',
                                   7:'07_EastofEngland',
                                   8:'09_InnerLondon',
                                   9:'11_OuterLondon',
                                   10:'12_SouthEast',
                                   11:'13_SouthWest',
                                   12:'14_WestMidlandsMetropolitan',
                                   13:'15_RestofWestMidlands',
                                   14:'16_GreaterManchester',
                                   15:'17_Merseyside',
                                   16:'18_RestofNorthWest',
                                   17:'19_Wales',
                                   18:'20_Strathclyde',
                                   19:'21_RestofScotland',
                                   20:'22_NorthernIreland',
                                   -8:'m8_NoAnswer',
                                   -9:'m9_DoesNotApply',
                                   }
data_lfs['regionofresidence'] = data_lfs['URESMC'].map(dictionary['regionofresidence'])

# -----------------------------------------------------------------------------
# For doctors who do not provide region of work, impute with region of residence
data_lfs.loc[(data_lfs['Flag_CurrentDoctor'] == 1) 
             & (data_lfs['regionofwork'] == 'm8_NoAnswer')
             ,'regionofwork'] = data_lfs['regionofresidence']


# -----------------------------------------------------------------------------
# Quantify current occupational class of residents using Labour Force Survey - nssec_current_full (NSECM10)
test = data_lfs.groupby(['lfs_year','nssec_current_full'])['lfs_year'].count()
# 9-category - 1 split into 1.1 and 1.2
data_lfs.loc[(data_lfs['nssec_current_full'].isin([1,2])), 'nssec_current_9cat'] = '1.1'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 3) & (data_lfs['nssec_current_full'] < 4), 'nssec_current_9cat'] = '1.2'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 4) & (data_lfs['nssec_current_full'] < 7), 'nssec_current_9cat'] = '2'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 7) & (data_lfs['nssec_current_full'] < 8), 'nssec_current_9cat'] = '3'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 8) & (data_lfs['nssec_current_full'] < 10), 'nssec_current_9cat'] = '4'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 10) & (data_lfs['nssec_current_full'] < 12), 'nssec_current_9cat'] = '5'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 12) & (data_lfs['nssec_current_full'] < 13), 'nssec_current_9cat'] = '6'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 13) & (data_lfs['nssec_current_full'] < 14), 'nssec_current_9cat'] = '7'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 14) & (data_lfs['nssec_current_full'] < 15), 'nssec_current_9cat'] = '8'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 15) & (data_lfs['nssec_current_full'] < 16), 'nssec_current_9cat'] = 'Full-time students'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 16) & (data_lfs['nssec_current_full'] <= 17), 'nssec_current_9cat'] = 'Not classifiable'
data_lfs.loc[(data_lfs['nssec_current_full'] == -9), 'nssec_current_9cat'] = 'Does not apply'
data_lfs.loc[(data_lfs['nssec_current_full'] == -8), 'nssec_current_9cat'] = 'No answer'

# 8-category
data_lfs.loc[(data_lfs['nssec_current_full'].isin([1,2])), 'nssec_current_8cat'] = '1'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 3) & (data_lfs['nssec_current_full'] < 4), 'nssec_current_8cat'] = '1'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 4) & (data_lfs['nssec_current_full'] < 7), 'nssec_current_8cat'] = '2'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 7) & (data_lfs['nssec_current_full'] < 8), 'nssec_current_8cat'] = '3'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 8) & (data_lfs['nssec_current_full'] < 10), 'nssec_current_8cat'] = '4'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 10) & (data_lfs['nssec_current_full'] < 12), 'nssec_current_8cat'] = '5'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 12) & (data_lfs['nssec_current_full'] < 13), 'nssec_current_8cat'] = '6'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 13) & (data_lfs['nssec_current_full'] < 14), 'nssec_current_8cat'] = '7'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 14) & (data_lfs['nssec_current_full'] < 15), 'nssec_current_8cat'] = '8'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 15) & (data_lfs['nssec_current_full'] < 16), 'nssec_current_8cat'] = 'Full-time students'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 16) & (data_lfs['nssec_current_full'] <= 17), 'nssec_current_8cat'] = 'Not classifiable'
data_lfs.loc[(data_lfs['nssec_current_full'] == -9), 'nssec_current_8cat'] = 'Does not apply'
data_lfs.loc[(data_lfs['nssec_current_full'] == -8), 'nssec_current_8cat'] = 'No answer'

# 5-category
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['1','2'])), 'nssec_current_5cat'] = '1-2'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['3'])), 'nssec_current_5cat'] = '3'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['4'])), 'nssec_current_5cat'] = '4'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['5'])), 'nssec_current_5cat'] = '5'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['6','7'])), 'nssec_current_5cat'] = '6-7'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['8'])), 'nssec_current_5cat'] = '8'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 15) & (data_lfs['nssec_current_full'] < 16), 'nssec_current_5cat'] = 'Full-time students'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 16) & (data_lfs['nssec_current_full'] <= 17), 'nssec_current_5cat'] = 'Not classifiable'
data_lfs.loc[(data_lfs['nssec_current_full'] == -9), 'nssec_current_5cat'] = 'Does not apply'
data_lfs.loc[(data_lfs['nssec_current_full'] == -8), 'nssec_current_5cat'] = 'No answer'

# 4-category
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['1','2'])), 'nssec_current_4cat'] = '1-2'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['3','4'])), 'nssec_current_4cat'] = '3-4'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['5','6','7'])), 'nssec_current_4cat'] = '5-7'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['8'])), 'nssec_current_4cat'] = '8'
data_lfs.loc[(data_lfs['nssec_current_full'] == -9), 'nssec_current_4cat'] = 'Does not apply'
data_lfs.loc[(data_lfs['nssec_current_full'] == -8), 'nssec_current_4cat'] = 'No answer'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 15) & (data_lfs['nssec_current_full'] < 16), 'nssec_current_4cat'] = 'Full-time students'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 16) & (data_lfs['nssec_current_full'] <= 17), 'nssec_current_4cat'] = 'Not classifiable'

# 3-category
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['1','2'])), 'nssec_current_3cat'] = '1-2'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['3','4'])), 'nssec_current_3cat'] = '3-4'
data_lfs.loc[(data_lfs['nssec_current_8cat'].isin(['5','6','7','8'])), 'nssec_current_3cat'] = '5-8'
data_lfs.loc[(data_lfs['nssec_current_full'] == -9), 'nssec_current_3cat'] = 'Does not apply'
data_lfs.loc[(data_lfs['nssec_current_full'] == -8), 'nssec_current_3cat'] = 'No answer'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 15) & (data_lfs['nssec_current_full'] < 16), 'nssec_current_3cat'] = 'Full-time students'
data_lfs.loc[(data_lfs['nssec_current_full'] >= 16) & (data_lfs['nssec_current_full'] <= 17), 'nssec_current_3cat'] = 'Not classifiable'

test = data_lfs[['nssec_current_full','NSECMJ10','nssec_current_9cat','nssec_current_8cat','nssec_current_5cat','nssec_current_3cat','nssec_familybackground_8cat','nssec_familybackground_5cat','nssec_familybackground_3cat']]


# -----------------------------------------------------------------------------
# Create flag to identify respondents with socio-economic background data - and so who will be selected into analysis sample
data_lfs.loc[~(data_lfs['nssec_familybackground_9cat'].isin(['Unknown','Do not know'])), 'SelectionIntoAnalysisFlag'] = 1
data_lfs['SelectionIntoAnalysisFlag'] = data_lfs['SelectionIntoAnalysisFlag'].fillna(0)




#%% Create dummy columns for variables of interest
# -----------------------------------------------------------------------------
# List of categorical input variables to create dummy variables for
data_lfs_cols_beforedummy = data_lfs.columns.to_list()

variable_list_categorical = [# nssec family background
                             # 'nssec_familybackground_full', 
                             'nssec_familybackground_9cat', 
                             'nssec_familybackground_8cat',
                             'nssec_familybackground_5cat',
                             'nssec_familybackground_4cat',
                             'nssec_familybackground_3cat',
                             
                             'nssec_familybackground_separatedoctor_9cat', 
                             'nssec_familybackground_separatedoctor_8cat',
                             'nssec_familybackground_separatedoctor_5cat',
                             'nssec_familybackground_separatedoctor_4cat',
                             'nssec_familybackground_separatedoctor_3cat',
                             
                             # nssec current
                              'nssec_current_full', 
                             # 'nssec_current_9cat',
                             'nssec_current_8cat',
                             'nssec_current_5cat',
                             'nssec_current_4cat',
                             'nssec_current_3cat',
                             
                             # Main earner occupation
                             'SOC_occupation_familyback_separatedoctor',
                             'Flag_MainEarner_221_Doctor',
                             
                             # Current occupation
                             'Flag_CurrentDoctor',
                             
                             # Demographics
                              'AGES',
                             'age_5yrbands',
                             'yearofbirth_5yrbands',
                             'year_age18_5yrband_agg_doctoronly',
                             'year_age18_10yrband_agg_doctoronly',
                             'year_age18_5yrband_agg_doctorORdegree',
                             'year_age18_10yrband_agg_doctorORdegree',
                             'SEX', 
                             'CRY12',
                             'countryofbirth_agg',
                             'countryofbirth_binary',
                             'ETHEWEUL', 
                             'ETHGBEUL', 
                             'ETHUKEUL',
                             'ethnicgroup_agg',
                             'ethnicgroup_category',
                             'ethnicgroup_binary',
                             'education_level',
                             'INECAC05', # economic activity detailed
                             'ILODEFR', # economic activity grouped
                             'regionofresidence',
                             
                             # Health
                             'LNGLST',
                             
                             # Survey year
                             'lfs_year',
                             
                             # Has socioeconomic background data
                             # 'SelectionIntoAnalysisFlag',
                             
                             ]

# Add dummy variables to datasets
data_lfs = categorical_to_dummy(data_lfs, variable_list_categorical)
data_lfs_cols = data_lfs.columns.to_list()


#%% Specify reference categories 
# -----------------------------------------------------------------------------
# List of dummy fields to drop from model, to use as reference category
cols_categorical_reference = [# nssec family background 
                              # Doctor as reference so consistent across groupings
                             'nssec_familybackground_9cat_1.1',
                             'nssec_familybackground_8cat_1',
                             'nssec_familybackground_5cat_1-2',
                             'nssec_familybackground_4cat_1-2',
                             'nssec_familybackground_3cat_1-2',
                             
                             'nssec_familybackground_separatedoctor_9cat_Doctor',
                             'nssec_familybackground_separatedoctor_8cat_Doctor',
                             'nssec_familybackground_separatedoctor_5cat_Doctor',
                             'nssec_familybackground_separatedoctor_4cat_Doctor',
                             'nssec_familybackground_separatedoctor_3cat_Doctor',
                             
                             # nssec current
                             'nssec_current_full_17.0',
                             'nssec_current_8cat_1',
                             'nssec_current_5cat_1',
                             'nssec_current_3cat_1',
                             
                             # Main earner occupation
                             'SOC_occupation_familyback_separatedoctor_2211.0',
                             'Flag_MainEarner_221_Doctor_NaN',
                             
                             # Demographics
                             'AGES_8',
                             'age_5yrbands_11: 50-55',
                             'yearofbirth_5yrbands_11: 1960-1965',
                             'year_age18_5yrband_agg_doctoronly_15: 1980-1985',
                             'year_age18_10yrband_agg_doctoronly_08: 1980-1990',
                             'year_age18_5yrband_agg_doctorORdegree_15: 1980-1985',
                             'year_age18_10yrband_agg_doctorORdegree_08: 1980-1990',
                             'SEX_1', 
                             'CRY12_921',
                             'countryofbirth_agg_921',
                             'countryofbirth_binary_921',
                             'ETHEWEUL_1', 
                             'ETHGBEUL_1', 
                             'ETHUKEUL_1',
                             'ethnicgroup_agg_1',
                             'ethnicgroup_category_1',
                             'ethnicgroup_binary_1',
                             'education_level_1.0',
                             'INECAC05_1',
                             'ILODEFR_1',
                             'regionofresidence_12_SouthEast',
                             
                             # Health
                             'LNGLST_2',
                             
                             # Survey year
                             'lfs_year_2018',
                                   ]


#%% Filter dataset for regression models
# -----------------------------------------------------------------------------
# Filter for age only, INCLUDE those with unknown family NSSEC
# Age 23+ only, as majority of doctors have 5 year undergraduate degree, then become 'junior doctor' at ~23+ for 2 years foundation training, then specialism training (3 years for GP, 5-8 years for other specialties)
# Exclude those age 65+ AND Retired, as retirement age 60-65 for most people, so doctors still working at age 65+ are only compared against others also working at 65+

data_lfs_beforedummy = data_lfs[data_lfs_cols_beforedummy].copy()

data_lfs_filter_include_unknown = data_lfs[(data_lfs['AGE'] >= 23) 
                                           & ~((data_lfs['AGE'] >= 65) 
                                               & (data_lfs['Flag_Retired'] == 1)
                                               ) # NOT 65+ AND Retired
                           ].copy()
data_lfs_filter_include_unknown_cols = data_lfs_filter_include_unknown.columns.to_list()

data_lfs_filter_include_unknown_beforedummy = data_lfs_filter_include_unknown[data_lfs_cols_beforedummy].copy()

# -----------------------------------------------------------------------------
# Filter for age only and EXCLUDE those with unknown family NSSEC
data_lfs_filter_exclude_unknown = data_lfs[(data_lfs['AGE'] >= 23)
                                            & ~((data_lfs['AGE'] >= 65) 
                                                & (data_lfs['Flag_Retired'] == 1)
                                                ) # NOT 65+ AND Retired
                            & ~(data_lfs['nssec_familybackground_9cat'].isin(['Unknown','Do not know']))
                           ].copy()
data_lfs_filter_exclude_unknown_cols = data_lfs_filter_exclude_unknown.columns.to_list()

data_lfs_filter_exclude_unknown_beforedummy = data_lfs_filter_exclude_unknown[data_lfs_cols_beforedummy].copy()



#%% Inverse probability weighting to account for missing family NSSEC
# Model that predicts not having family NSSEC - 'Unknown','Do not know', based on demogs - Age, Sex, Country of birth, Ethnic group, Education level, Economic activity
# -----------------------------------------------------------------------------
#%% Creating prediction model for selection into analysis set: Run forward sequential feature selection to identify most predictive set of features
# -----------------------------------------------------------------------------
# Set dataset and outcome variable to test
# 0 = Model name, 1 = dataset, 2 = full fieldname list, 3 = outcome variable
outcome_var = 'SelectionIntoAnalysisFlag'
do_poisson_IPW = ''
manual_sfs_run_list = [['Predicting availability of socio-economic background data', 
                        data_lfs_filter_include_unknown,
                        data_lfs_filter_include_unknown_cols,
                    outcome_var,],
                   ]

manual_sfs_result_list = []
best_model_vars_list = []

do_IPW_sequentialfeatureselection = 0
if do_IPW_sequentialfeatureselection == 1:
    for n in range(0, len(manual_sfs_run_list), 1):
        model_name = manual_sfs_run_list[n][0]
        dataset = manual_sfs_run_list[n][1]
        full_fieldname_list = manual_sfs_run_list[n][2]
        
        # -----------------------------------------------------------------------------
        # Set outcome field
        outcome_var = manual_sfs_run_list[n][3]
        
        # -----------------------------------------------------------------------------
        # Set input fields
        # List of categorical variables (non-dummy fields)
        input_var_categorical = [# Socio-demographics
                                'yearofbirth_5yrbands',
                                'education_level',
                                'CRY12',
                                'lfs_year',
                                'regionofresidence',
                                'LNGLST',
                                'nssec_current_full',
                                'ETHUKEUL',
                                'age_5yrbands',                                
                                'SEX',
                                'INECAC05',
                                ]
            
        # List of continuous variables
        input_var_continuous = []
        
        # -----------------------------------------------------------------------------
        # Generate list of categorical input dummy variables, by identifying all dummy variables with fieldname starting with a value in input_var_categorical and deleting reference variable using input_var_categorical_reference list
        
        # Generate list of dummy fields for complete fields
        input_var_categorical_formodel = generate_dummy_list(original_fieldname_list = input_var_categorical, 
                                                         full_fieldname_list = full_fieldname_list, 
                                                         reference_fieldname_list = cols_categorical_reference,
                                                         delete_reference = 'yes')
        
        
        # -----------------------------------------------------------------------------
        ### Run various models to measure importance of features
        # Set input fields
        # Combine categorical and continuous
        input_var_all = input_var_categorical_formodel + input_var_continuous
               
        # Drop dummy columns where sum of column = 0 - i.e. no-one from particular group - can cause 'Singular matrix' error when running model
        empty_cols  = []
        input_var_categorical_formodel_copy = input_var_categorical_formodel.copy()
        for col in input_var_categorical_formodel_copy: 
            # print(col)
            if dataset[col].sum() < 1:
                print('remove empty dummy: ' + col)
                input_var_categorical_formodel.remove(col)
                empty_cols.append(col)
        
        # Drop dummy columns where no observations of outcome in group in column = 0 - i.e. no observations - can cause 'Singular matrix' error when running model
        input_var_categorical_formodel_copy = input_var_categorical_formodel.copy()
        for col in input_var_categorical_formodel_copy: 
            if dataset[(dataset[outcome_var] == 1)][col].sum() < 1:
                print('remove dummy with no observations of outcome: ' + col)
                input_var_categorical_formodel.remove(col)
                empty_cols.append(col)
    
        print('After dropping MISSING DATA and EMPTY dummy cols: ')
        print(input_var_categorical_formodel)
        
        # Set variables to go into model
        input_var_all = input_var_continuous + input_var_categorical_formodel
        
        
        # Generate outcome and input data
        y_data = dataset[outcome_var].reset_index(drop=True) # set output variable
        x_data = dataset[input_var_all].reset_index(drop=True) # create input variable tables for models
    
        # Split data into test and train set
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, stratify = y_data, test_size = 0.25, random_state = 0)
        x_train = x_train.reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)
        x_test = x_test.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)
        

        # -----------------------------------------------------------------------------
        # Manual forward Sequential Feature Selection
        model_select = 'logreg' # rf, logreg, extratrees, boosting
        # List of potential variables
        var_list = input_var_categorical + input_var_continuous
        var_list_len = len(var_list)
        # Start with empty list of variables to add selected to
        selected_vars = []
        auc_max_list = []
        best_var_list = []
        for n in range (0,var_list_len,1):
            model_auc_list = []
            for var in var_list:
                # if variable categorical, pull out dummy variables for model
                if var in input_var_categorical:
                    # Generate list of dummy fields for complete fields
                    var_formodel = generate_dummy_list(original_fieldname_list = [var], 
                                                    full_fieldname_list = full_fieldname_list, 
                                                    reference_fieldname_list = cols_categorical_reference,
                                                    delete_reference = 'yes')
                else:
                    var_formodel = [var]
                
                vars_input = selected_vars + var_formodel
                
                # Drop any empty columns identified earlier
                for empty_col in empty_cols:
                    if empty_col in vars_input:
                        print('remove empty column: ' + empty_col + ' from input vars')
                        vars_input.remove(empty_col)
                
                if model_select in ['rf', 'boosting', 'extratrees']:
                    # Filter test and train data for selected variables
                    x_train_sfs = x_train[vars_input]
                    x_test_sfs = x_test[vars_input]
                    if model_select == 'rf':
                        model = RandomForestClassifier()
                    elif model_select == 'boosting':
                        model = GradientBoostingClassifier()
                    elif model_select == 'extratrees':
                        model = ExtraTreesClassifier()
                    model_fit = model.fit(x_train_sfs, y_train) # fit model with training data
                    model_y_pred = model.predict(x_test_sfs) # generate predictions for test data
                    model_auc = metrics.roc_auc_score(y_test, model_y_pred)
                
                elif model_select == 'logreg':
                    # Do logistic regression (stats models) of control + test variables
                    x_data_sfs = x_data[vars_input].copy()
                    sm_summary, model_fit, model_auc, model_explained_variance, model_r2, model, model_testtrain = sm_logreg_simple_HC3(x_data = x_data_sfs, y_data = y_data, 
                                                         CI_alpha = 0.05, do_robust_se = 'HC3',
                                                         use_weights = 'NA', weight_data = '',
                                                         do_poisson = do_poisson_IPW)
        
                print(str(vars_input)+': AUC: '+str(model_auc))
                # Save to list
                model_auc_list.append(model_auc)
            
            # Create dataframe
            auc_df = pd.DataFrame({'var':var_list,
                          'AUC':model_auc_list})
            
            # Identify variable with maximum AUC
            auc_max = auc_df['AUC'].max()
            auc_max_list.append(auc_max)
            best_var = auc_df[auc_df['AUC'] == auc_df['AUC'].max()]['var'].to_list()
            print('Round ' + str(n) + ', best variable: ' + str(best_var) + ' , AUC: ' + str(auc_df['AUC'].max()))
            # Add variable to selected variable list
            # if best variable categorical, pull out dummy variables for model
            if best_var[0] in input_var_categorical:
                # Generate list of dummy fields for complete fields
                best_var_formodel = generate_dummy_list(original_fieldname_list = best_var, 
                                                    full_fieldname_list = full_fieldname_list, 
                                                    reference_fieldname_list = cols_categorical_reference,
                                                    delete_reference = 'yes')
            else:
                best_var_formodel = best_var
            selected_vars = selected_vars + best_var_formodel
            # Drop best variable from list for next round
            var_list.remove(best_var[0])
            
            # Save best variable to list
            best_var_list.append(best_var)
            
        
        manual_sfs_result = pd.DataFrame({'Variable added':best_var_list,
                                          'AUC':auc_max_list})
        
        manual_sfs_result_list.append(manual_sfs_result)
        
        # -------------------------------------------------------------------------
        # Identify model with max AUC
        model_auc_max_idx = manual_sfs_result['AUC'].idxmax()
        best_model_vars = manual_sfs_result['Variable added'][0:model_auc_max_idx+1].to_list()
        best_model_vars = [item for sublist in best_model_vars for item in sublist]
        best_model_vars_list.append(best_model_vars)


# -----------------------------------------------------------------------------
# SET WHICH VARIABLES TO USE IN IPW MODELS
# Choose variables in model with highest AUC
var_categorical = [# Socio-demographics
                   'yearofbirth_5yrbands',
                    'education_level',
                    'CRY12',
                    'lfs_year',
                    'regionofresidence',
                    'INECAC05',
                    'LNGLST',
                    'nssec_current_full',
                    'ETHUKEUL',
                    'age_5yrbands',                                
                    'SEX',
                       ]

var_continuous = []

do_poisson_IPW = ''

# -----------------------------------------------------------------------------
# Run IPW model
# Set output variable
outcome_var = 'SelectionIntoAnalysisFlag'
logreg_model_var_list_forweight = [[var_continuous, var_categorical, 'NA_predictiveonly_forweight']]

# Run model
model_results_summary_forweight, model_auc_summary_forweight, model_fit_list_forweight, model_prediction_df_list_forweight, model_predict_df_list_forweight, model_input_dummy_list_forweight = run_logistic_regression_models(data = data_lfs_filter_include_unknown, 
                                                                          data_full_col_list = data_lfs_filter_include_unknown_cols, 
                                                                          model_var_list = logreg_model_var_list_forweight, 
                                                                          outcome_var = outcome_var, 
                                                                          use_weights = 'no', 
                                                                          weight_var = '', 
                                                                          filter_missing = '',
                                                                          plot_model = 'yes',
                                                                          cols_categorical_reference = cols_categorical_reference,
                                                                          do_poisson = do_poisson_IPW)

model_fit_forweight = model_fit_list_forweight[0]
model_prediction_df_forweight = model_prediction_df_list_forweight[0]



# Generate scaled inverse probability weights from model fit
data_lfs_filter_include_unknown = generate_IPW(data = data_lfs_filter_include_unknown,
                                             model_fit = model_fit_forweight,
                                             weight_colname_suffix = 'SelectionIntoAnalysis')

test = data_lfs_filter_include_unknown[['IPW_SelectionIntoAnalysis',
                                      ]]

# Plot histogram of weights
xlims = [0, 3]
titlelabel = 'Has family background data'
legend_offset = -0.5

ax = plt.figure()
ax = sns.histplot(data=data_lfs_filter_include_unknown, x='IPW_SelectionIntoAnalysis', hue=outcome_var, element="poly")
ax.set_xlim(xlims[0], xlims[1])
ax.set_xlabel('Inverse probability of selection into analysis weight')
ax.set_title(titlelabel)
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, legend_offset), 
                title='Participation: 0 = Not selected (data missing), 1 = Selected')

ax1 = plt.figure()
ax1 = sns.histplot(data=data_lfs_filter_include_unknown, x='IPW_SelectionIntoAnalysis', hue=outcome_var, element="poly", stat = "probability", common_norm = False)
ax1.set_xlim(xlims[0], xlims[1])
ax1.set_xlabel('Inverse probability of selection into analysis weight')
ax1.set_ylabel('Normalised count')
ax1.set_title(titlelabel)
sns.move_legend(ax1, "lower center", bbox_to_anchor=(0.5, legend_offset), 
                title='Participation: 0 = Not selected (data missing), 1 = Selected')


#%% Generate weights table
# -----------------------------------------------------------------------------
weights_selectionintoanalysis = data_lfs_filter_include_unknown[['id',
                                                'IPW_SelectionIntoAnalysis',
                                                ]].copy()


# -----------------------------------------------------------------------------
# Merge weight into analysis sample
data_lfs_filter_exclude_unknown = pd.merge(data_lfs_filter_exclude_unknown, weights_selectionintoanalysis, how = 'left',  on = 'id')

# -----------------------------------------------------------------------------
# Multiply LFS provided person weight with selection into analysis weight to generate 'master' weight for analysis
data_lfs_filter_exclude_unknown['IPW_ProvidedWeightPlusSelection'] = data_lfs_filter_exclude_unknown['person_weight'] * data_lfs_filter_exclude_unknown['IPW_SelectionIntoAnalysis']

# Scale weights so that sum = number of participants in sample
scaling_factor = data_lfs_filter_exclude_unknown.shape[0] / (data_lfs_filter_exclude_unknown['IPW_ProvidedWeightPlusSelection'].sum())
data_lfs_filter_exclude_unknown['IPW_ProvidedWeightPlusSelection'] = scaling_factor * data_lfs_filter_exclude_unknown['IPW_ProvidedWeightPlusSelection']

data_lfs_filter_exclude_unknown['IPW_ProvidedWeightPlusSelection'].sum()

test = data_lfs_filter_exclude_unknown[['id','person_weight','IPW_SelectionIntoAnalysis','IPW_ProvidedWeightPlusSelection']]
test_describe1 = test.describe()


# Winsorise weights to reduce importance of outliers (e.g. limit weight at value of 5th or 9th percentile)
# Calculate limits based on 5th and 95th percentiles among those with full participation
winsor_values = data_lfs_filter_exclude_unknown['IPW_ProvidedWeightPlusSelection'].quantile(q = [0.01, 0.05, 0.95, 0.99])

do_winsorisation = 1
if do_winsorisation == 1:
    # Select columns to do winsorization on
    weight_winsorization_list = [#
                                 'IPW_ProvidedWeightPlusSelection',
                                 ]
    # manual limits based on 5th and 95th percentiles among those with full participation
    manual_limit_list = [[winsor_values[0.05], winsor_values[0.95]]]
    # Do winsorization
    data_lfs_filter_exclude_unknown = winsorization(data = data_lfs_filter_exclude_unknown, 
                         winsorization_limits = [0.05,0.95],
                         winsorization_col_list = weight_winsorization_list,
                         set_manual_limits = 'yes',
                         manual_limit_list = manual_limit_list)
    
    # Re-scale again after winsorising
    # Scale weights so that sum = number of participants in sample
    scaling_factor = data_lfs_filter_exclude_unknown.shape[0] / (data_lfs_filter_exclude_unknown['IPW_ProvidedWeightPlusSelection' + '_winsorised'].sum())
    data_lfs_filter_exclude_unknown['IPW_ProvidedWeightPlusSelection' + '_winsorised'] = scaling_factor * data_lfs_filter_exclude_unknown['IPW_ProvidedWeightPlusSelection' + '_winsorised']
    
    test = data_lfs_filter_exclude_unknown[['id','IPW_ProvidedWeightPlusSelection', 'IPW_ProvidedWeightPlusSelection' + '_winsorised']]
    test_describe2 = test.describe()
    


#%% Rescale existing population weights to add up to respondent N
# -----------------------------------------------------------------------------
# Dataset including unknown NSSEC
data_lfs_filter_include_unknown['person_weight_rescaled'] = data_lfs_filter_include_unknown['person_weight']*(data_lfs_filter_include_unknown.shape[0]/data_lfs_filter_include_unknown['person_weight'].sum())
test1 = data_lfs_filter_include_unknown['person_weight_rescaled'].describe()
test1sum = data_lfs_filter_include_unknown['person_weight_rescaled'].sum()

# Winsorise weights to reduce importance of outliers (e.g. limit weight at value of 5th or 9th percentile)
# Calculate limits based on 5th and 95th percentiles among those with full participation
winsor_values = data_lfs_filter_include_unknown['person_weight_rescaled'].quantile(q = [0.01, 0.05, 0.95, 0.99])

do_winsorisation = 1
if do_winsorisation == 1:
    # Select columns to do winsorization on
    weight_winsorization_list = [#
                                 'person_weight_rescaled',
                                 ]
    # manual limits based on 5th and 95th percentiles among those with full participation
    manual_limit_list = [[winsor_values[0.05], winsor_values[0.95]]]
    # Do winsorization
    data_lfs_filter_include_unknown = winsorization(data = data_lfs_filter_include_unknown, 
                         winsorization_limits = [0.05,0.95],
                         winsorization_col_list = weight_winsorization_list,
                         set_manual_limits = 'yes',
                         manual_limit_list = manual_limit_list)
    
    # Re-scale again after winsorising
    # Scale weights so that sum = number of participants in sample
    scaling_factor = data_lfs_filter_include_unknown.shape[0] / (data_lfs_filter_include_unknown['person_weight_rescaled' + '_winsorised'].sum())
    data_lfs_filter_include_unknown['person_weight_rescaled' + '_winsorised'] = scaling_factor * data_lfs_filter_include_unknown['person_weight_rescaled' + '_winsorised']
    
    test = data_lfs_filter_include_unknown[['Flag_CurrentDoctor', 'person_weight_rescaled', 'person_weight_rescaled' + '_winsorised']]
    test_describe = test.describe()
    

# -----------------------------------------------------------------------------
# Dataset excluding unknown NSSEC
data_lfs_filter_exclude_unknown['person_weight_rescaled'] = data_lfs_filter_exclude_unknown['person_weight']*(data_lfs_filter_exclude_unknown.shape[0]/data_lfs_filter_exclude_unknown['person_weight'].sum())
test2 = data_lfs_filter_exclude_unknown['person_weight_rescaled'].describe()
test2sum = data_lfs_filter_exclude_unknown['person_weight_rescaled'].sum()

# Winsorise weights to reduce importance of outliers (e.g. limit weight at value of 5th or 9th percentile)
# Calculate limits based on 5th and 95th percentiles among those with full participation
winsor_values = data_lfs_filter_exclude_unknown['person_weight_rescaled'].quantile(q = [0.01, 0.05, 0.95, 0.99])

do_winsorisation = 1
if do_winsorisation == 1:
    # Select columns to do winsorization on
    weight_winsorization_list = [#
                                 'person_weight_rescaled',
                                 ]
    # manual limits based on 5th and 95th percentiles among those with full participation
    manual_limit_list = [[winsor_values[0.05], winsor_values[0.95]]]
    # Do winsorization
    data_lfs_filter_exclude_unknown = winsorization(data = data_lfs_filter_exclude_unknown, 
                         winsorization_limits = [0.05,0.95],
                         winsorization_col_list = weight_winsorization_list,
                         set_manual_limits = 'yes',
                         manual_limit_list = manual_limit_list)
    
    # Re-scale again after winsorising
    # Scale weights so that sum = number of participants in sample
    scaling_factor = data_lfs_filter_exclude_unknown.shape[0] / (data_lfs_filter_exclude_unknown['person_weight_rescaled' + '_winsorised'].sum())
    data_lfs_filter_exclude_unknown['person_weight_rescaled' + '_winsorised'] = scaling_factor * data_lfs_filter_exclude_unknown['person_weight_rescaled' + '_winsorised']
    
    test = data_lfs_filter_exclude_unknown[['Flag_CurrentDoctor', 'person_weight_rescaled', 'person_weight_rescaled' + '_winsorised']]
    test_describe3 = test.describe()


data_lfs_filter_exclude_unknown_beforedummy = data_lfs_filter_exclude_unknown[data_lfs_cols_beforedummy+['IPW_ProvidedWeightPlusSelection' + '_winsorised']].copy()



#%% Poisson regression model to test association between family social class background and likelihood of being doctor - individual-level
# -----------------------------------------------------------------------------
# Set up models for individual-level dataset
# -----------------------------------------------------------------------------
# Specify outcome variable
outcome_var_doctor = 'Flag_CurrentDoctor_1.0'
outcome_var = outcome_var_doctor

# Specify which version of aggregated year turned 18 to use based on outcome variable
if outcome_var == outcome_var_doctor:
    var_year18_5yr = 'year_age18_5yrband_agg_doctoronly'
    var_year18_10yr = 'year_age18_10yrband_agg_doctoronly'
    year18_5yr_list = ['12: Pre-1970', '13: 1970-1975','14: 1975-1980','15: 1980-1985','16: 1985-1990','17: 1990-1995','18: 1995-2000','19: 2000-2005','20: 2005-2010','21: Post-2010'] #'12: Pre-1970', '21: 2010-2015','22: 2015-2020']
    

model_var_list_year = [
                # --------------------------------------------------                        
                ### Mutually adjusting for year turned 18 and sex
                # Exposure - Year group - 5 year bands
                [[],['lfs_year', var_year18_5yr, 'SEX',
                   ],var_year18_5yr],            
                ]
                
# All exposure other than year
model_var_list_demogs = [               
                # --------------------------------------------------     
                ### Mutually adjusting for year turned 18 and sex
                # Exposure - Sex
                [[],['lfs_year', var_year18_5yr, 'SEX',
                   ],'SEX'],
                
                # ### Adjusting for year turned 18, Sex
                # # Exposure - Country of birth
                [[],['lfs_year', var_year18_5yr, 'SEX', 'CRY12',
                    ],'CRY12'],
                
                # ### Adjusting for year turned 18, Sex, Country of birth
                # # Exposure - Ethnic group
                [[],['lfs_year', var_year18_5yr, 'SEX', 
                     # 'CRY12', # removed as control variable after suggestion of reviewer
                     'ETHUKEUL',
                    ],'ETHUKEUL'],
                ]

model_var_list_demogs_noyear = [
                # --------------------------------------------------                        
                ### Mutually adjusting for year turned 18 and sex
                # Exposure - Sex
                [[],['lfs_year', 'SEX',
                   ],'SEX'],
                
                ### Adjusting for year turned 18, Sex
                # Exposure - Country of birth (some groups collapsed)
                [[],['lfs_year', 'SEX', 'countryofbirth_agg',
                    ],'countryofbirth_agg'],
                [[],['lfs_year', 'SEX', 'countryofbirth_binary',
                   ],'countryofbirth_binary'],
                
                ### Adjusting for year turned 18, Sex, Country of birth
                # Exposure - Ethnic group (some groups collapsed)
                [[],['lfs_year', 'SEX', 
                     # 'countryofbirth_agg', # removed as control variable after suggestion of reviewer
                     'ethnicgroup_category',
                   ],'ethnicgroup_category'],
                # Exposure - Ethnic group (binarised)
                [[],['lfs_year', 'SEX', 
                     # 'countryofbirth_binary', # removed as control variable after suggestion of reviewer
                     'ethnicgroup_binary',
                   ],'ethnicgroup_binary'],
                
                ]

model_var_list_occupations = [            
                ### Adjusting for year turned 18, Sex, Country of birth, Ethnic group               
                # Exposure - Main earner occupation group (with doctor separated)
                # Aggregated country and ethnic group controls to avoid singular matrix
                [[],['lfs_year', var_year18_5yr, 'SEX', 'countryofbirth_binary', 'ethnicgroup_category',  'SOC_occupation_familyback_separatedoctor',
                   ],'SOC_occupation_familyback_separatedoctor'],

                # Exposure - Main earner was doctor
                [[],['lfs_year', var_year18_5yr, 'SEX', 'CRY12', 'ETHUKEUL', 'Flag_MainEarner_221_Doctor',
                    ],'Flag_MainEarner_221_Doctor'],
                ]

model_var_list_occupations_noyear = [            
                ## Adjusting for year turned 18, Sex, Country of birth, Ethnic group
                # Exposure - Main earner was doctor
                [[],['lfs_year', 'SEX', 'CRY12', 'ETHUKEUL', 'Flag_MainEarner_221_Doctor',
                    ],'Flag_MainEarner_221_Doctor'],
                ]


model_var_list_NSSEC = [                
                ### Adjusting for year turned 18, Sex, Country of birth, Ethnic group
                # Exposure - NSSEC of main earner - 9-cat
                [[],['lfs_year', var_year18_5yr, 'SEX', 'CRY12', 'ETHUKEUL', 'nssec_familybackground_9cat',
                    ],'nssec_familybackground_9cat'],
                # Exposure - NSSEC of main earner - 3-cat
                [[],['lfs_year', var_year18_5yr, 'SEX', 'CRY12', 'ETHUKEUL', 'nssec_familybackground_3cat',
                   ],'nssec_familybackground_3cat'],
                
                # WHERE DOCTOR IS SEPARATED CATEGORY
                # Exposure - NSSEC of main earner - 9-cat
                [[],['lfs_year', var_year18_5yr, 'SEX', 'CRY12', 'ETHUKEUL', 'nssec_familybackground_separatedoctor_9cat',
                    ],'nssec_familybackground_separatedoctor_9cat'],
                # Exposure - NSSEC of main earner - 3-cat
                [[],['lfs_year', var_year18_5yr, 'SEX', 'CRY12', 'ETHUKEUL', 'nssec_familybackground_separatedoctor_3cat',
                   ],'nssec_familybackground_separatedoctor_3cat'],

                ]


model_var_list_NSSEC_noyear = [                
                ### Adjusting for year turned 18, Sex, Country of birth, Ethnic group
                # Exposure - NSSEC of main earner - 3-cat
                [[],['lfs_year', 'SEX', 'countryofbirth_binary', 'ethnicgroup_category', 'nssec_familybackground_3cat',
                   ],'nssec_familybackground_3cat'],
                
                # WHERE DOCTOR IS SEPARATED CATEGORY
                # Exposure - NSSEC of main earner - 3-cat
                [[],['lfs_year', 'SEX', 'countryofbirth_binary', 'ethnicgroup_category', 'nssec_familybackground_separatedoctor_3cat',
                    ],'nssec_familybackground_separatedoctor_3cat'],
                
                ]

model_var_list_all = model_var_list_year + model_var_list_demogs + model_var_list_occupations + model_var_list_NSSEC
model_var_list_all_noyear = model_var_list_demogs_noyear + model_var_list_occupations_noyear + model_var_list_NSSEC_noyear




#%% Specify sequence of models to run
# -----------------------------------------------------------------------------
# Linear regression models
model_var_list_choose = model_var_list_all
model_var_list_choose_noyear = model_var_list_all_noyear

logreg_model_inference_run_list = [#
                               # -------------------------------------------------
                        # WEIGHTED + IPW, SAMPLE: Excluding unknown
                        ['Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown', # model name
                          data_lfs_filter_exclude_unknown, # dataset
                          data_lfs_filter_exclude_unknown_cols, # full dataset fieldname list
                          outcome_var, # outcome variable
                          model_var_list_choose, # input variable list
                          'yes', # use weights
                            'IPW_ProvidedWeightPlusSelection' + '_winsorised', # weight variable
                          ],
                                                                     
                            ]

        
# Add lists to run models on samples stratified by year turned 18 - rolling window
add_stratified_year18_rollingwindow = 1
if outcome_var == outcome_var_doctor:
    # 5yr rolling: list(range(1964,2017,1))
    year18_rolling_centre_list = list(range(1964,2017,1))
    year18_rolling_lim = 2 # specify how many years either side    
if add_stratified_year18_rollingwindow == 1:
    for n in range(0,len(year18_rolling_centre_list),1):
        year = year18_rolling_centre_list[n]       
        # WEIGHTED + IPW, SAMPLE: Excluding unknown
        model_list_toadd = ['Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: ' + str(int(year)) + ' (centre of rolling)', # model name
                    data_lfs_filter_exclude_unknown[(data_lfs_filter_exclude_unknown['year_age18'] >= year-year18_rolling_lim)
                                                    & (data_lfs_filter_exclude_unknown['year_age18'] <= year+year18_rolling_lim)
                                                    ], # dataset
                    data_lfs_filter_exclude_unknown_cols, # full dataset fieldname list
                    outcome_var, # outcome variable
                    model_var_list_choose_noyear, # input variable list
                    'yes', # use weights
                    'IPW_ProvidedWeightPlusSelection' + '_winsorised', # weight variable
                    ]
        logreg_model_inference_run_list.append(model_list_toadd)


        
#%% Loop to run models
run_regression = 1
do_poisson_models = 'yes' # yes to do modified poisson regression instead of logistic regression, to avoid non-collapsibility which means odds ratios aren't comparable
if run_regression == 1:
    # -----------------------------------------------------------------------------
    # Logistic regression models for binary outcomes
    logreg_model_results_summary_list = []
    logreg_model_auc_summary_list = []
    logreg_model_fit_list_list = []
    logreg_model_prediction_df_list_list = []
    logreg_model_predict_df_list_list = []
    logreg_model_input_dummy_list_list = []
    for n in range(0, len(logreg_model_inference_run_list), 1):
        model_name = logreg_model_inference_run_list[n][0]
        # dataset = logreg_model_inference_run_list[n][1].copy(deep = False)
        full_fieldname_list = logreg_model_inference_run_list[n][2]
        outcome_var = logreg_model_inference_run_list[n][3]
        logreg_model_var_list = logreg_model_inference_run_list[n][4]
        use_weights = logreg_model_inference_run_list[n][5]
        weight_var = logreg_model_inference_run_list[n][6]
        
        # -----------------------------------------------------------------------------
        # Run models
        print('Logistic regression for outcome of: ' + model_name)
        model_results_summary, model_auc_summary, model_fit_list, model_prediction_df_list, model_predict_df_list, model_input_dummy_list = run_logistic_regression_models(data = logreg_model_inference_run_list[n][1].copy(deep = False), 
                                                            data_full_col_list = full_fieldname_list, 
                                                            model_var_list = logreg_model_var_list, 
                                                            outcome_var = outcome_var, 
                                                            use_weights = use_weights, 
                                                            weight_var = weight_var,
                                                            filter_missing = 'no',
                                                            plot_model = 'yes',
                                                            cols_categorical_reference = cols_categorical_reference,
                                                            do_poisson = do_poisson_models)
           
        # Add reference rows to model results
        add_reference = 1
        if add_reference == 1:
            model_results_summary = add_reference_odds_ratio(model_results_summary)
        
        # Label model
        model_results_summary['model_name'] = model_name 
        
        logreg_model_results_summary_list.append(model_results_summary)
        logreg_model_auc_summary_list.append(model_auc_summary)
        logreg_model_fit_list_list.append(model_fit_list)
        logreg_model_prediction_df_list_list.append(model_prediction_df_list)
        logreg_model_predict_df_list_list.append(model_predict_df_list)
        logreg_model_input_dummy_list_list.append(model_input_dummy_list)
    
    logreg_model_results_summary1 = logreg_model_results_summary_list[0]



    #%% Join model results together and process
    # -----------------------------------------------------------------------------
    # Logistic regression models for binary outcomes
    # Append together
    logreg_model_results_combined = logreg_model_results_summary_list[0]
    for n in range(1,len(logreg_model_results_summary_list),1):
        logreg_model_results_combined = logreg_model_results_combined.append(logreg_model_results_summary_list[n])
        
    # Drop constant
    logreg_model_results_combined = logreg_model_results_combined[~(logreg_model_results_combined['Variable'] == 'const')].reset_index(drop = True)
    
    # Drop rows that aren't the exposure variable
    logreg_model_results_combined['var_match'] = logreg_model_results_combined.apply(lambda x: x.var_exposure in x.Variable, axis = 1)
    logreg_model_results_combined = logreg_model_results_combined[(logreg_model_results_combined['var_match'] == True)].reset_index(drop = True)
    
    # Filter columns
    logreg_model_results_combined_cols = logreg_model_results_combined.columns.to_list()
    col_select = ['model_name', 'Variable', 'P-value', 'Odds ratio', 'OR C.I. (lower)', 'OR C.I. (upper)', 'OR C.I. error (lower)', 'OR C.I. error (upper)', 'total_count_n', 'group_count', 'outcome_count', 'Significance', 'outcome_variable']
    logreg_model_results_combined_filter = logreg_model_results_combined[col_select].copy()
    
    # For mapping and plotting
    full_variable_dummy_list = list(logreg_model_results_combined['Variable'].unique())
    full_variable_exposure_list = list(logreg_model_results_combined['var_exposure'].unique())
    
    # -----------------------------------------------------------------------------
    # Apply multiple testing adjustment to p-values of models
    # Filter for all exposures testing association with same outcome variable in turn
    outcome_var_list = logreg_model_results_combined_filter['model_name'].unique()
    logreg_model_results_combined_filter_list = []
    for var in outcome_var_list:
        logreg_model_results_combined_filter_slice = logreg_model_results_combined_filter[(logreg_model_results_combined_filter['model_name'] == var) 
                                                                            & ~(logreg_model_results_combined_filter['P-value'].isnull())].copy()
        multiple_test_correction = fdrcorrection(logreg_model_results_combined_filter_slice['P-value'], alpha=0.05, method='indep', is_sorted=False)
        logreg_model_results_combined_filter_slice['p_value_corrected'] = multiple_test_correction[1]
        logreg_model_results_combined_filter_list.append(logreg_model_results_combined_filter_slice)
    model_results_pvalue_corrected = pd.concat(logreg_model_results_combined_filter_list)
    
    logreg_model_results_combined_filter = pd.merge(logreg_model_results_combined_filter, model_results_pvalue_corrected['p_value_corrected'], how = 'left', left_index = True, right_index = True)
    
    # Redo significance column for corrected p value
    logreg_model_results_combined_filter.loc[(logreg_model_results_combined_filter['OR C.I. (lower)'] > 1.0)
                        & (logreg_model_results_combined_filter['OR C.I. (upper)'] > 1.0)
                        & (logreg_model_results_combined_filter['p_value_corrected'] < 0.05)
                        ,'Significance_p_corrected'] = 'Significant (OR > 1), *, p < 0.05'
    logreg_model_results_combined_filter.loc[(logreg_model_results_combined_filter['OR C.I. (lower)'] > 1.0)
                        & (logreg_model_results_combined_filter['OR C.I. (upper)'] > 1.0)
                        & (logreg_model_results_combined_filter['p_value_corrected'] < 0.01)
                        ,'Significance_p_corrected'] = 'Significant (OR > 1), **, p < 0.01'
    logreg_model_results_combined_filter.loc[(logreg_model_results_combined_filter['OR C.I. (lower)'] > 1.0)
                        & (logreg_model_results_combined_filter['OR C.I. (upper)'] > 1.0)
                        & (logreg_model_results_combined_filter['p_value_corrected'] < 0.001)
                        ,'Significance_p_corrected'] = 'Significant (OR > 1), ***, p < 0.001'
    
    logreg_model_results_combined_filter.loc[(logreg_model_results_combined_filter['OR C.I. (lower)'] < 1.0)
                        & (logreg_model_results_combined_filter['OR C.I. (upper)'] < 1.0)
                        & (logreg_model_results_combined_filter['p_value_corrected'] < 0.05)
                        ,'Significance_p_corrected'] = 'Significant (OR < 1), *, p < 0.05'
    logreg_model_results_combined_filter.loc[(logreg_model_results_combined_filter['OR C.I. (lower)'] < 1.0)
                        & (logreg_model_results_combined_filter['OR C.I. (upper)'] < 1.0)
                        & (logreg_model_results_combined_filter['p_value_corrected'] < 0.01)
                        ,'Significance_p_corrected'] = 'Significant (OR < 1), **, p < 0.01'
    logreg_model_results_combined_filter.loc[(logreg_model_results_combined_filter['OR C.I. (lower)'] < 1.0)
                        & (logreg_model_results_combined_filter['OR C.I. (upper)'] < 1.0)
                        & (logreg_model_results_combined_filter['p_value_corrected'] < 0.001)
                        ,'Significance_p_corrected'] = 'Significant (OR < 1), ***, p < 0.001'
    
    
    
    #%% Apply mapping
    import LabourForceSurveyCodebook
    mapping_var_name = LabourForceSurveyCodebook.dictionary['var_string']
    mapping_var_name_short = LabourForceSurveyCodebook.dictionary['var_string_short']
    mapping_y_pos = LabourForceSurveyCodebook.dictionary['var_number']
    logreg_model_results_combined_filter['Variable_tidy'] = logreg_model_results_combined_filter['Variable'].map(mapping_var_name)
    logreg_model_results_combined_filter['Variable_tidy_short'] = logreg_model_results_combined_filter['Variable'].map(mapping_var_name_short)
    logreg_model_results_combined_filter['y_pos_manual'] = logreg_model_results_combined_filter['Variable'].map(mapping_y_pos)
    
    
    #%% Plot results
    # -----------------------------------------------------------------------------
    # 1 - Limits, 2 - height, 3 - domain for title
    plot_list = [# socio-demogs
                    [[37.5,-1], 12, -0.13, 'Year, Sex, Country of birth, Ethnic group', [0.04, 10]],
                    [[37.5,10.5], 12, -0.13, 'Sex, Country of birth, Ethnic group', [0.1, 10]],
                    [[149,40], 12, -0.13, 'Main earner occupation (SOC minor group)', [0.001,2]],
                    [[175.5,159], 12, -0.13, 'Socio-economic background', [0.01,4]],
                    [[196.5,178], 12, -0.13, 'Socio-economic background \n (Doctor separated)', [0.01,4]],
                   ]
    
    # -----------------------------------------------------------------------------
    ### Outcome: Current doctor
    for n in range(0,len(plot_list),1):
        limits = plot_list[n][0]
        height = (limits[0] - limits[1])/4
        # height = plot_list[n][1]
        domain = plot_list[n][3]
        scalar = 0.8
        legend_offset = -0.075 - np.exp(-0.25*height) # follows 1-exp relationship. fitted from manual best fits
        # legend_offset = plot_list[n][2]     
        xlims = plot_list[n][4]
        data1 = plot_OR_w_conf_int(data1 = logreg_model_results_combined_filter[(logreg_model_results_combined_filter['model_name'] == 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown')],
                              x_fieldname = 'Variable_tidy_short',
                              y_fieldname = 'Odds ratio',
                              conf_int_fieldnames = ['OR C.I. error (lower)','OR C.I. error (upper)'],
                              plot1_label = 'Outcome: Currently working as a doctor',
                              xlims = xlims,  # [0.03, 30] [0.3, 3.33333]
                              ylims = limits,
                              titlelabel = 'Exposures: ' + domain, 
                              width = 5.5, 
                              height = height*1.2,
                              y_pos_manual = 'yes',
                              color_list = ['royalblue'],
                              fontsize = 12,
                              legend_offset = legend_offset,
                              invert_axis = 'yes',
                              x_logscale = 'yes',
                              x_major_tick = 1,
                              x_minor_tick = 0.5,
                              poisson_reg = do_poisson_models,
                              bold = 0
                              )
        
        
        # Sorted by Risk Ratio
        data_results_occupations = logreg_model_results_combined_filter[(logreg_model_results_combined_filter['model_name'] == 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown')
                                    & (logreg_model_results_combined_filter['Variable'].str.contains('SOC_occupation'))
                                    & ~((logreg_model_results_combined_filter['outcome_count'] < 1)
                                       | (logreg_model_results_combined_filter['group_count'] < 100))
                                    # filter out any occupations with 0 observations of doctors only, or if less than 100 in the occupation overall
                                    ].sort_values(by = 'Odds ratio').reset_index(drop = True)       
        
        # Filter by NS-SEC, based on modal NS-SEC associated with occupation group
        nsssec12_occupations_string = '111.0|112.0|113.0|114.0|115.0|116.0|117.0|123.0|124.0|211.0|212.0|213.0|214.0|215.0|216.0|221.0|222.0|223.0|224.0|225.0|231.0|232.0|241.0|242.0|243.0|244.0|245.0|246.0|247.0|248.0|249.0|311.0|313.0|322.0|324.0|341.0|343.0|351.0|353.0|354.0|355.0|356.0|357.0|358.0|414|525|722|2211.0'
        nsssec12_occupations = [111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 123.0, 124.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 221.0, 222.0, 223.0, 224.0, 225.0, 231.0, 232.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 311.0, 313.0, 322.0, 324.0, 341.0, 343.0, 351.0, 353.0, 354.0, 355.0, 356.0, 357.0, 358.0, 
                                414, 525,722,
                                2211.0,
                                
                                ]
        nsssec34_occupations_string = '121|122|125|312|321|323|331|342|352|411|412|413|415|421.0|511.0|531|532|611.0|625.0|712.0|713.0'
        nsssec34_occupations = [121,122,125,312,321,323,331,342,352,411,412,413,415,
                                 421.0, 511.0, 531, 532, 611.0,625.0,712.0,713.0,
                                 ]
        nsssec58_occupations_string = '521|522.0|523.0|524.0|533.0|541.0|542.0|543.0|544.0|612.0|613.0|621.0|622.0|623.0|624.0|631.0|711.0|721.0|811.0|812.0|813.0|814.0|815.0|816.0|821.0|822.0|823.0|911.0|912.0|913.0|921.0|922.0|923.0|924.0|925.0|926.0|-9.0'
        nsssec58_occupations = [521, 522.0, 523.0, 524.0, 533.0, 541.0, 542.0, 543.0, 544.0, 612.0, 613.0, 621.0, 622.0, 623.0, 624.0, 631.0, 711.0, 721.0, 811.0, 812.0, 813.0, 814.0, 815.0, 816.0, 821.0, 822.0, 823.0, 911.0, 912.0, 913.0, 921.0, 922.0, 923.0, 924.0, 925.0, 926.0, -9.0
                                ]
        
        occupation_list = list(data_lfs_filter_exclude_unknown['SOC_occupation_familyback_separatedoctor'].sort_values().unique())
        
        data_results_occupations_12 = data_results_occupations[(data_results_occupations['Variable'].str.contains(nsssec12_occupations_string))]
        data_results_occupations_34 = data_results_occupations[(data_results_occupations['Variable'].str.contains(nsssec34_occupations_string))]
        data_results_occupations_58 = data_results_occupations[(data_results_occupations['Variable'].str.contains(nsssec58_occupations_string))]
        
        data_results_occupations.loc[(data_results_occupations['Variable'].isin(data_results_occupations_12['Variable'])),'nssec'] = '1-2'
        data_results_occupations.loc[(data_results_occupations['Variable'].isin(data_results_occupations_34['Variable'])),'nssec'] = '3-4'
        data_results_occupations.loc[(data_results_occupations['Variable'].isin(data_results_occupations_58['Variable'])),'nssec'] = '5-8'
        
        data_results_occupations = data_results_occupations.reset_index()
        data_results_occupations['y_pos_manual'] = data_results_occupations['index']
        
        # Plot risk ratio
        n = 2
        limits = plot_list[n][0]
        height = (limits[0] - limits[1])/4
        # height = plot_list[n][1]
        domain = plot_list[n][3]
        scalar = 0.8
        legend_offset = -0.075 - np.exp(-0.25*height) # follows 1-exp relationship. fitted from manual best fits
        # legend_offset = plot_list[n][2]     
        xlims = plot_list[n][4]
        data1 = plot_OR_w_conf_int(data1 = data_results_occupations,
                              x_fieldname = 'Variable_tidy_short',
                              y_fieldname = 'Odds ratio',
                              conf_int_fieldnames = ['OR C.I. error (lower)','OR C.I. error (upper)'],
                              plot1_label = 'Outcome: Currently working as a doctor',
                              xlims = [0.001,2],  # [0.03, 30] [0.3, 3.33333]
                              ylims = [-2,98],
                              titlelabel = 'Exposures: ' + domain, 
                              width = 5.5, 
                              height = 15*1.2, # 27.25
                              y_pos_manual = '',
                              color_list = ['royalblue'],
                              fontsize = 12,
                              legend_offset = legend_offset,
                              invert_axis = 'yes',
                              x_logscale = 'yes',
                              x_major_tick = 1,
                              x_minor_tick = 0.5,
                              poisson_reg = do_poisson_models,
                              bold = 0
                              )
        
        # Plot risk ratio
        color_list1 = [
               'blueviolet', # blueviolet
               'mediumblue',
               'royalblue',
               'cornflowerblue',
               
               'green', #3
               'mediumseagreen', #4
               
               'gold', #5 'gold'
               'orange', # 6
               'darkorange', # 7
               'maroon', # 8
               
               'grey', # Not living with family
               ]
        
        n = 2
        limits = plot_list[n][0]
        height = (limits[0] - limits[1])/4
        # height = plot_list[n][1]
        domain = plot_list[n][3]
        scalar = 0.8
        legend_offset = -0.075 - np.exp(-0.25*height) # follows 1-exp relationship. fitted from manual best fits
        # legend_offset = plot_list[n][2]     
        xlims = plot_list[n][4]
        data1, data2, data3 = plot_OR_w_conf_int_3plots(data1 = data_results_occupations[(data_results_occupations['nssec'] == '1-2')],
                                          data2 = data_results_occupations[(data_results_occupations['nssec'] == '3-4')],
                                          data3 = data_results_occupations[(data_results_occupations['nssec'] == '5-8')],
                              x_fieldname = 'Variable_tidy_short',
                              y_fieldname = 'Odds ratio',
                              conf_int_fieldnames = ['OR C.I. error (lower)','OR C.I. error (upper)'],
                              plot1_label = 'NS-SEC 1-2. Professional occupations',
                              plot2_label = 'NS-SEC 3-4. Intermediate occupations',
                              plot3_label = 'NS-SEC 5-8. Working class occupations',
                              xlims = [0.001,2],  # [0.03, 30] [0.3, 3.33333]
                              ylims = [-2,98],
                              titlelabel = 'Exposures: ' + domain, 
                              width = 5.5, 
                              height = 15*1.2, # 27.25
                              y_pos_manual = 'yes',
                              color_list = [color_list1[2],color_list1[4],color_list1[8]],
                              fontsize = 12,
                              offset = 0,
                              legend_offset = -0.11,
                              invert_axis = 'yes',
                              x_logscale = 'yes',
                              x_major_tick = 1,
                              x_minor_tick = 0.5,
                              poisson_reg = do_poisson_models,
                              )
        plt.yticks(data_results_occupations['y_pos_manual'], data_results_occupations['Variable_tidy_short']) # set labels manually 
        ax.yaxis.label.set_visible(False) # hide y axis title
                
        
    ### Plots with BOLD variable labels
    # Add bold variable labels
    var_label_list = []
    var_label_list.append({'y_pos_manual':0, 'var_label': 'Yes', 'Variable_tidy_short':'Year turned 18'})
    var_label_list.append({'y_pos_manual':11.5, 'var_label': 'Yes', 'Variable_tidy_short':'Sex'})
    var_label_list.append({'y_pos_manual':15, 'var_label': 'Yes', 'Variable_tidy_short':'Country of birth'})
    var_label_list.append({'y_pos_manual':26.5, 'var_label': 'Yes', 'Variable_tidy_short':'Ethnic group'})
    var_label_list.append({'y_pos_manual':40, 'var_label': 'Yes', 'Variable_tidy_short':'Household main earner occupation'})
    var_label_list.append({'y_pos_manual':160, 'var_label': 'Yes', 'Variable_tidy_short':'Socio-economic background (NS-SEC 9-class)'})
    var_label_list.append({'y_pos_manual':171.5, 'var_label': 'Yes', 'Variable_tidy_short':'Socio-economic background (NS-SEC 3-class)'})
    var_label_list.append({'y_pos_manual':179, 'var_label': 'Yes', 'Variable_tidy_short':'Socio-economic background (NS-SEC 9-class, doctor separated)'})
    var_label_list.append({'y_pos_manual':191.5, 'var_label': 'Yes', 'Variable_tidy_short':'Socio-economic background (NS-SEC 3-class, doctor separated)'})

    logreg_model_results_combined_filter_forbold = logreg_model_results_combined_filter.copy()
    for n in range(0,len(var_label_list),1):
        var_label = var_label_list[n]
        logreg_model_results_combined_filter_forbold = logreg_model_results_combined_filter_forbold.append(var_label, ignore_index=True)
        
    for n in range(0,len(plot_list),1):
        limits = plot_list[n][0]
        height = (limits[0] - limits[1])/4
        # height = plot_list[n][1]
        domain = plot_list[n][3]
        scalar = 0.8
        legend_offset = -0.075 - np.exp(-0.25*height) # follows 1-exp relationship. fitted from manual best fits
        # legend_offset = plot_list[n][2]     
        xlims = plot_list[n][4]

        data1 = plot_OR_w_conf_int(data1 = logreg_model_results_combined_filter_forbold[(logreg_model_results_combined_filter_forbold['model_name'] == 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown')
                                      | (logreg_model_results_combined_filter_forbold['var_label'] == 'Yes')],
                          x_fieldname = 'Variable_tidy_short',
                          y_fieldname = 'Odds ratio',
                          conf_int_fieldnames = ['OR C.I. error (lower)','OR C.I. error (upper)'],
                          plot1_label = 'Outcome: Currently working as a doctor',
                          xlims = xlims,  # [0.03, 30] [0.3, 3.33333]
                          ylims = limits,
                          titlelabel = 'Exposures: ' + domain, 
                          width = 5.5, 
                          height = height*1.2,
                          y_pos_manual = 'yes',
                          color_list = ['royalblue'],
                          fontsize = 12,
                          legend_offset = legend_offset,
                          invert_axis = 'yes',
                          x_logscale = 'yes',
                          x_major_tick = 1,
                          x_minor_tick = 0.5,
                          poisson_reg = do_poisson_models,
                          bold = 1
                          )



# -----------------------------------------------------------------------------
### Plot Time series stratified model results
# Filter for stratified time-series models
logreg_model_results_timeseries = logreg_model_results_combined_filter[(logreg_model_results_combined_filter['model_name'].str.contains('Year 18'))]

# Add field with tidy strings of stratified year
model_name_list = list(logreg_model_results_timeseries['model_name'].unique())


timeseries_dict = {#
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1960 (centre of rolling)':'1960',
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1961 (centre of rolling)':'',#'1961',
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1962 (centre of rolling)':'1962',
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1963 (centre of rolling)':'',#'1963',
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1964 (centre of rolling)':'1964',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1965 (centre of rolling)':'',#'1965',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1966 (centre of rolling)':'1966',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1967 (centre of rolling)':'',#'1967',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1968 (centre of rolling)':'1968',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1969 (centre of rolling)':'',#'1969',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1970 (centre of rolling)':'1970',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1971 (centre of rolling)':'',#'1971',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1972 (centre of rolling)':'1972',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1973 (centre of rolling)':'',#'1973',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1974 (centre of rolling)':'1974',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1975 (centre of rolling)':'',#'1975',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1976 (centre of rolling)':'1976',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1977 (centre of rolling)':'',#'1977',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1978 (centre of rolling)':'1978',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1979 (centre of rolling)':'',#'1979',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1980 (centre of rolling)':'1980',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1981 (centre of rolling)':'',#'1981',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1982 (centre of rolling)':'1982',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1983 (centre of rolling)':'',#'1983',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1984 (centre of rolling)':'1984',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1985 (centre of rolling)':'',#'1985',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1986 (centre of rolling)':'1986',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1987 (centre of rolling)':'',#'1987',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1988 (centre of rolling)':'1988',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1989 (centre of rolling)':'',#'1989',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1990 (centre of rolling)':'1990',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1991 (centre of rolling)':'',#'1991',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1992 (centre of rolling)':'1992',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1993 (centre of rolling)':'',#'1993',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1994 (centre of rolling)':'1994',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1995 (centre of rolling)':'',#'1995',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1996 (centre of rolling)':'1996',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1997 (centre of rolling)':'',#'1997',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1998 (centre of rolling)':'1998',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1999 (centre of rolling)':'',#'1999',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2000 (centre of rolling)':'2000',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2001 (centre of rolling)':'',#'2001',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2002 (centre of rolling)':'2002',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2003 (centre of rolling)':'',#'2003',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2004 (centre of rolling)':'2004',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2005 (centre of rolling)':'',#'2005',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2006 (centre of rolling)':'2006',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2007 (centre of rolling)':'',#'2007',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2008 (centre of rolling)':'2008',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2009 (centre of rolling)':'',#'2009',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2010 (centre of rolling)':'2010',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2011 (centre of rolling)':'',#'2011',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2012 (centre of rolling)':'2012',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2013 (centre of rolling)':'',#'2013',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2014 (centre of rolling)':'2014',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2015 (centre of rolling)':'',#'2015',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2016 (centre of rolling)':'2016',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2017 (centre of rolling)':'',#'2017',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2018 (centre of rolling)':'2018',
 }

timeseries_dict = {#
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1960 (centre of rolling)':'1960',
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1961 (centre of rolling)':'',#'1961',
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1962 (centre of rolling)':'',#'1962',
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1963 (centre of rolling)':'',#'1963',
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1964 (centre of rolling)':'',#'1964',
  'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1965 (centre of rolling)':'1965',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1966 (centre of rolling)':'',#'1966',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1967 (centre of rolling)':'',#'1967',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1968 (centre of rolling)':'',#'1968',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1969 (centre of rolling)':'',#'1969',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1970 (centre of rolling)':'1970',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1971 (centre of rolling)':'',#'1971',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1972 (centre of rolling)':'',#'1972',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1973 (centre of rolling)':'',#'1973',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1974 (centre of rolling)':'',#'1974',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1975 (centre of rolling)':'1975',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1976 (centre of rolling)':'',#'1976',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1977 (centre of rolling)':'',#'1977',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1978 (centre of rolling)':'',#'1978',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1979 (centre of rolling)':'',#'1979',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1980 (centre of rolling)':'1980',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1981 (centre of rolling)':'',#'1981',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1982 (centre of rolling)':'',#'1982',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1983 (centre of rolling)':'',#'1983',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1984 (centre of rolling)':'',#'1984',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1985 (centre of rolling)':'1985',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1986 (centre of rolling)':'',#'1986',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1987 (centre of rolling)':'',#'1987',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1988 (centre of rolling)':'',#'1988',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1989 (centre of rolling)':'',#'1989',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1990 (centre of rolling)':'1990',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1991 (centre of rolling)':'',#'1991',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1992 (centre of rolling)':'',#'1992',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1993 (centre of rolling)':'',#'1993',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1994 (centre of rolling)':'',#'1994',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1995 (centre of rolling)':'1995',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1996 (centre of rolling)':'',#'1996',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1997 (centre of rolling)':'',#'1997',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1998 (centre of rolling)':'',#'1998',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1999 (centre of rolling)':'',#'1999',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2000 (centre of rolling)':'2000',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2001 (centre of rolling)':'',#'2001',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2002 (centre of rolling)':'',#'2002',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2003 (centre of rolling)':'',#'2003',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2004 (centre of rolling)':'',#'2004',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2005 (centre of rolling)':'2005',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2006 (centre of rolling)':'',#'2006',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2007 (centre of rolling)':'',#'2007',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2008 (centre of rolling)':'',#'2008',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2009 (centre of rolling)':'',#'2009',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2010 (centre of rolling)':'2010',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2011 (centre of rolling)':'',#'2011',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2012 (centre of rolling)':'',#'2012',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2013 (centre of rolling)':'',#'2013',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2014 (centre of rolling)':'',#'2014',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2015 (centre of rolling)':'2015',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2016 (centre of rolling)':'',#'2016',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2017 (centre of rolling)':'',#'2017',
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2018 (centre of rolling)':'2018',
 }

timeseries_dict_num = {#
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1960 (centre of rolling)':1960,
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1961 (centre of rolling)':1961,
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1962 (centre of rolling)':1962,
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1963 (centre of rolling)':1963,
                   'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1964 (centre of rolling)':1964,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1965 (centre of rolling)':1965,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1966 (centre of rolling)':1966,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1967 (centre of rolling)':1967,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1968 (centre of rolling)':1968,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1969 (centre of rolling)':1969,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1970 (centre of rolling)':1970,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1971 (centre of rolling)':1971,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1972 (centre of rolling)':1972,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1973 (centre of rolling)':1973,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1974 (centre of rolling)':1974,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1975 (centre of rolling)':1975,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1976 (centre of rolling)':1976,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1977 (centre of rolling)':1977,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1978 (centre of rolling)':1978,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1979 (centre of rolling)':1979,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1980 (centre of rolling)':1980,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1981 (centre of rolling)':1981,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1982 (centre of rolling)':1982,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1983 (centre of rolling)':1983,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1984 (centre of rolling)':1984,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1985 (centre of rolling)':1985,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1986 (centre of rolling)':1986,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1987 (centre of rolling)':1987,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1988 (centre of rolling)':1988,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1989 (centre of rolling)':1989,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1990 (centre of rolling)':1990,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1991 (centre of rolling)':1991,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1992 (centre of rolling)':1992,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1993 (centre of rolling)':1993,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1994 (centre of rolling)':1994,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1995 (centre of rolling)':1995,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1996 (centre of rolling)':1996,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1997 (centre of rolling)':1997,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1998 (centre of rolling)':1998,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 1999 (centre of rolling)':1999,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2000 (centre of rolling)':2000,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2001 (centre of rolling)':2001,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2002 (centre of rolling)':2002,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2003 (centre of rolling)':2003,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2004 (centre of rolling)':2004,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2005 (centre of rolling)':2005,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2006 (centre of rolling)':2006,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2007 (centre of rolling)':2007,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2008 (centre of rolling)':2008,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2009 (centre of rolling)':2009,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2010 (centre of rolling)':2010,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2011 (centre of rolling)':2011,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2012 (centre of rolling)':2012,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2013 (centre of rolling)':2013,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2014 (centre of rolling)':2014,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2015 (centre of rolling)':2015,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2016 (centre of rolling)':2016,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2017 (centre of rolling)':2017,
 'Weighted: YES w/IPW, Outcome: Doctor, Sample: Excluding unknown, Year 18: 2018 (centre of rolling)':2018,
                   
                   }

logreg_model_results_timeseries['model_name_tidy'] = logreg_model_results_timeseries['model_name'].map(timeseries_dict)
logreg_model_results_timeseries['y_pos_manual'] = logreg_model_results_timeseries['model_name'].map(timeseries_dict_num)

color_list1 = [
               'blueviolet', # blueviolet
               'mediumblue',
               'royalblue',
               'cornflowerblue',
               
               'green', #3
               'mediumseagreen', #4
               
               'gold', #5 'gold'
               'orange', # 6
               'darkorange', # 7
               'maroon', # 8
               
               'grey', # Not living with family
               ]

# Do plot
limits = [11,0]
height = (limits[0] - limits[1])/4
scalar = 0.8
legend_offset = -0.6 # follows 1-exp relationship. fitted from manual best fits
# legend_offset = plot_list[n][2]     
xlims = plot_list[n][4]


# Doctor background & NS-SEC - 3 class
var_list = ['Flag_MainEarner_221_Doctor_1.0',
            'nssec_familybackground_separatedoctor_3cat_1-2',
            'nssec_familybackground_separatedoctor_3cat_3-4',
            'nssec_familybackground_separatedoctor_3cat_5-8',]
label_list = ['Main earner occupation: Doctor (vs. All other occupations)',
              'Socio-economic background: NS-SEC 1-2. Other professional backgrounds (vs. Doctors)',
              'Socio-economic background: NS-SEC 3-4. Intermediate backgrounds (vs. Doctors)',
              'Socio-economic background: NS-SEC 5-8. Working class backgrounds (vs. Doctors)',
              ]
data1, data2, data3, data4 = plot_OR_w_conf_int_4plots_timeseries(data1 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[0]],
            data2 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[1]],
            data3 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[2]],
            data4 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[3]],
                      x_fieldname = 'model_name_tidy',
                      y_fieldname = 'Odds ratio',
                      conf_int_fieldnames = ['OR C.I. (lower)','OR C.I. (upper)'],
                      plot1_label = label_list[0],
                      plot2_label = label_list[1],
                      plot3_label = label_list[2],
                      plot4_label = label_list[3],
                      xlims = [1962, 2018],#limits,  # [0.03, 30]
                      ylims = [0.004, 80] ,
                      titlelabel = 'Stratified models: Socio-economic background (Doctor separated)', 
                      width = 7, 
                      height = 10,
                      offset = 0.0,
                      y_pos_manual = 'yes',
                      color_list = [color_list1[0], color_list1[2],color_list1[4],color_list1[8],],
                      fontsize = 12,
                      legend_offset = -0.35,
                      invert_axis = 'yes',
                      x_logscale = 'yes',
                      x_major_tick = 1,
                      x_minor_tick = 0.5,
                      poisson_reg = do_poisson_models,
                      xlabel = 'Year respondent turned 18 (centre of 5-year window)',
                      xtick_rotation = 45,
                      error_fill = 'yes'
                      )


# Sex, Country of birth (binary), Ethnic group (binary), NS-SEC
var_list = [
            'nssec_familybackground_3cat_3-4',
            'nssec_familybackground_3cat_5-8', 
            'SEX_2',
            'countryofbirth_binary_997',
            'ethnicgroup_binary_9',]
label_list = ['NS-SEC 3-4. Intermediate backgrounds (vs. Professional backgrounds)',
              'NS-SEC 5-8. Working class backgrounds (vs. Professional backgrounds)',
              'Sex: Female (vs. Male)',
              'Country of birth: non-UK (vs. UK nation)',
              'Ethnic group: Racially minoritised groups (vs. White)',
              ]

data1, data2, data3, data4, data5 = plot_OR_w_conf_int_5plots_timeseries(data1 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[0]],
                       data2 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[1]],
                       data3 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[2]],
                       data4 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[3]],
                       data5 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[4]],
                      x_fieldname = 'model_name_tidy',
                      y_fieldname = 'Odds ratio',
                      conf_int_fieldnames = ['OR C.I. (lower)','OR C.I. (upper)'],
                      plot1_label = label_list[0],
                      plot2_label = label_list[1],
                      plot3_label = label_list[2],
                      plot4_label = label_list[3],
                      plot5_label = label_list[4],
                      xlims = [1962, 2018],#limits,  # [0.03, 30]
                      ylims = [0.06, 30] ,
                      titlelabel = 'Stratified models: Socio-economic background (3-class), \n Sex, Country of birth, Ethnic group', 
                      width = 7, 
                      height = 10,
                      offset = 0.0,
                      y_pos_manual = 'yes',
                      color_list = [color_list1[4], color_list1[8], 'black', 'C6', 'C7', ],
                      fontsize = 12,
                      legend_offset = -0.4,
                      invert_axis = 'yes',
                      x_logscale = 'yes',
                      x_major_tick = 1,
                      x_minor_tick = 0.5,
                      poisson_reg = do_poisson_models,
                      xlabel = 'Year respondent turned 18 (centre of 5-year window)',
                      xtick_rotation = 45, 
                      error_fill = 'yes'
                      )


# Ethnic group (disaggregated)
var_list = ['ethnicgroup_category_7',
            'ethnicgroup_category_8',
            'ethnicgroup_category_2',
            'ethnicgroup_category_9']
label_list = ['Ethnic group: Asian/Asian British (vs. White)',
              'Ethnic group: Black/Black British (vs. White)',
              'Ethnic group: Mixed/Multiple (vs. White)',
              'Ethnic group: Any other groups (vs. White)',
              ]
data1, data2, data3, data4 = plot_OR_w_conf_int_4plots_timeseries(data1 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[0]],
            data2 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[1]],
            data3 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[2]],
            data4 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[3]],
                      x_fieldname = 'model_name_tidy',
                      y_fieldname = 'Odds ratio',
                      conf_int_fieldnames = ['OR C.I. (lower)','OR C.I. (upper)'],
                      plot1_label = label_list[0],
                      plot2_label = label_list[1],
                      plot3_label = label_list[2],
                      plot4_label = label_list[3],
                      xlims = [],#limits,  # [0.03, 30]
                      ylims = [0.1, 100] ,
                      titlelabel = 'Stratified models: Ethnic group (disaggregated)', 
                      width = 7, 
                      height = 10,
                      offset = 0,
                      y_pos_manual = 'yes',
                      color_list = ['C4', 'C1', 'C7', 'C9'],
                      fontsize = 12,
                      legend_offset = -0.4,
                      invert_axis = 'yes',
                      x_logscale = 'yes',
                      x_major_tick = 1,
                      x_minor_tick = 0.5,
                      poisson_reg = do_poisson_models,
                      xlabel = 'Year respondent turned 18', 
                      xtick_rotation = 45,
                      error_fill = 'yes'
                      )


# Country of birth (disaggregated)
var_list = ['countryofbirth_agg_922',
            'countryofbirth_agg_923',
            'countryofbirth_agg_924',
            'countryofbirth_agg_356',
            'countryofbirth_agg_997',]
label_list = ['Country of birth: Northern Ireland (vs. England)',
              'Country of birth: Scotland (vs. England)',
              'Country of birth: Wales (vs. England)',
              'Country of birth: India (vs. England)',
              'Country of birth: All other non-UK (vs. England)',
              
              ]
data1, data2, data3, data4, data5 = plot_OR_w_conf_int_5plots_timeseries(data1 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[0]],
                       data2 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[1]],
                       data3 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[2]],
                       data4 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[3]],
                       data5 = logreg_model_results_timeseries[logreg_model_results_timeseries['Variable'] == var_list[4]],
                      x_fieldname = 'model_name_tidy',
                      y_fieldname = 'Odds ratio',
                      conf_int_fieldnames = ['OR C.I. (lower)','OR C.I. (upper)'],
                      plot1_label = label_list[0],
                      plot2_label = label_list[1],
                      plot3_label = label_list[2],
                      plot4_label = label_list[3],
                      plot5_label = label_list[4],
                      xlims = [],#limits,  # [0.03, 30]
                      ylims = [0.1, 100] ,
                      titlelabel = 'Stratified models: Country of birth (disaggregated)', 
                      width = 7, 
                      height = 10,
                      offset = 0,
                      y_pos_manual = 'yes',
                      color_list = ['C4', 'C1', 'C7', 'C0', 'C3'],
                      fontsize = 12,
                      legend_offset = -0.4,
                      invert_axis = 'yes',
                      x_logscale = 'yes',
                      x_major_tick = 1,
                      x_minor_tick = 0.5,
                      poisson_reg = do_poisson_models,
                      xlabel = 'Year respondent turned 18',
                      xtick_rotation = 45,
                      error_fill = 'yes'
                      )



#%% Other plots
# -----------------------------------------------------------------------------
# Stacked bar chart of NSSEC family background for different years turned 18
# Sum 'IPW_ProvidedWeightPlusSelection' + '_winsorised'

# Create cross tab and proportions
row_col = 'year_age18_5yrband_agg_doctoronly'
sum_col = 'IPW_ProvidedWeightPlusSelection' + '_winsorised'
col_col = 'nssec_familybackground_separatedoctor_9cat'

import LabourForceSurveyCodebook
mapping_var_name_stacked = LabourForceSurveyCodebook.dictionary['var_string_stacked']
mapping_var_number_stacked = LabourForceSurveyCodebook.dictionary['var_number_stacked']

## Current doctors
nssec_background_crosstab_doctors = pd.pivot_table(data_lfs_filter_exclude_unknown[(data_lfs_filter_exclude_unknown['Flag_CurrentDoctor'] == 1)], values = sum_col, index = [row_col], columns = [col_col], aggfunc = "sum").fillna(0)

# Add total row
nssec_background_crosstab_doctors.loc['Overall'] = nssec_background_crosstab_doctors.sum()

nssec_background_crosstab_doctors_cols = nssec_background_crosstab_doctors.columns.to_list()

# Calculate proportions
nssec_background_crosstab_doctors['total'] = nssec_background_crosstab_doctors.sum(axis = 1)
for col in nssec_background_crosstab_doctors_cols:
    nssec_background_crosstab_doctors[col+'_prop'] = 100*(nssec_background_crosstab_doctors[col]/nssec_background_crosstab_doctors['total'])

# Add tidy variable name
nssec_background_crosstab_doctors = nssec_background_crosstab_doctors.reset_index()
nssec_background_crosstab_doctors['var_tidy'] = nssec_background_crosstab_doctors[row_col].map(mapping_var_name_stacked)
nssec_background_crosstab_doctors['y_pos'] = nssec_background_crosstab_doctors[row_col].map(mapping_var_number_stacked)


## NOT doctors
nssec_background_crosstab_NOTdoctors = pd.pivot_table(data_lfs_filter_exclude_unknown[~(data_lfs_filter_exclude_unknown['Flag_CurrentDoctor'] == 1)], values = sum_col, index = [row_col], columns = [col_col], aggfunc = "sum").fillna(0)
# Add total row
nssec_background_crosstab_NOTdoctors.loc['Overall'] = nssec_background_crosstab_NOTdoctors.sum()
# Calculate proportions
nssec_background_crosstab_NOTdoctors['total'] = nssec_background_crosstab_NOTdoctors.sum(axis = 1)
for col in nssec_background_crosstab_doctors_cols:
    nssec_background_crosstab_NOTdoctors[col+'_prop'] = 100*(nssec_background_crosstab_NOTdoctors[col]/nssec_background_crosstab_NOTdoctors['total'])

# Add tidy variable name
nssec_background_crosstab_NOTdoctors = nssec_background_crosstab_NOTdoctors.reset_index()
nssec_background_crosstab_NOTdoctors['var_tidy'] = nssec_background_crosstab_NOTdoctors[row_col].map(mapping_var_name_stacked)
nssec_background_crosstab_NOTdoctors['y_pos'] = nssec_background_crosstab_NOTdoctors[row_col].map(mapping_var_number_stacked)

prop_cols = ['1.1_prop', '1.2_prop', '2_prop', '3_prop', '4_prop', '5_prop', '6_prop', '7_prop', '8_prop', 'Doctor_prop', 'Not living with family_prop']

# Panel 1 - current doctors
color_list1 = [
               'blueviolet', # blueviolet
               'mediumblue',
               'royalblue',
               'cornflowerblue',
               
               'green', #3
               'mediumseagreen', #4
               
               'gold', #5 'gold'
               'orange', # 6
               'darkorange', # 7
               'maroon', # 8
               
               'grey', # Not living with family
               ]
# color_list1 = [cm.bwr(0.25),'linen',cm.bwr(0.75)] # 'lightcoral','lightgray','yellowgreen', 'linen'
label_list1 = [
               'NS-SEC 1.2. Doctor',
               'NS-SEC 1.1', 
               'NS-SEC 1.2. All other',
               'NS-SEC 2',
               'NS-SEC 3',
               'NS-SEC 4',
               'NS-SEC 5',
               'NS-SEC 6',
               'NS-SEC 7',
               'NS-SEC 8',
               'Not living with family',
               
               ]
titlelabel1 = 'Socio-economic background of current doctors\nstratified by year turned 18'
titlelabel2 = 'Socio-economic background of all respondents excluding doctors\nstratified by year turned 18'
xlim1 = [0,1]
legend_offset = 1.5
height = 4
width1a = 7

### DOCTORS PLOT
# Do plot
fig, ax1a = plt.subplots(figsize=(width1a,height))
barheight = 0.85
# 1.1
hbar1 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['Doctor_prop'], height = barheight, align='center', color = color_list1[0], label = label_list1[0], alpha = 1)
# 1.2 Doctors
hbar2 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['1.1_prop'], height = barheight, align='center', color = color_list1[1], label = label_list1[1], alpha = 1, left = nssec_background_crosstab_doctors['Doctor_prop'])
# 1.2 Other
hbar3 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['1.2_prop'], height = barheight, align='center', color = color_list1[2], label = label_list1[2], alpha = 1, left = nssec_background_crosstab_doctors['1.1_prop']+nssec_background_crosstab_doctors['Doctor_prop'])
# 2
hbar4 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['2_prop'], height = barheight, align='center', color = color_list1[3], label = label_list1[3], alpha = 1, left = nssec_background_crosstab_doctors['1.1_prop']+nssec_background_crosstab_doctors['Doctor_prop']+nssec_background_crosstab_doctors['1.2_prop'])
# 3
hbar5 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['3_prop'], height = barheight, align='center', color = color_list1[4], label = label_list1[4], alpha = 1, left = nssec_background_crosstab_doctors['1.1_prop']+nssec_background_crosstab_doctors['Doctor_prop']+nssec_background_crosstab_doctors['1.2_prop']+nssec_background_crosstab_doctors['2_prop'])
# 4
hbar6 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['4_prop'], height = barheight, align='center', color = color_list1[5], label = label_list1[5], alpha = 1, left = nssec_background_crosstab_doctors['1.1_prop']+nssec_background_crosstab_doctors['Doctor_prop']+nssec_background_crosstab_doctors['1.2_prop']+nssec_background_crosstab_doctors['2_prop']+nssec_background_crosstab_doctors['3_prop'])
# 5
hbar7 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['5_prop'], height = barheight, align='center', color = color_list1[6], label = label_list1[6], alpha = 1, left = nssec_background_crosstab_doctors['1.1_prop']+nssec_background_crosstab_doctors['Doctor_prop']+nssec_background_crosstab_doctors['1.2_prop']+nssec_background_crosstab_doctors['2_prop']+nssec_background_crosstab_doctors['3_prop']+nssec_background_crosstab_doctors['4_prop'])
# 6
hbar8 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['6_prop'], height = barheight, align='center', color = color_list1[7], label = label_list1[7], alpha = 1, left = nssec_background_crosstab_doctors['1.1_prop']+nssec_background_crosstab_doctors['Doctor_prop']+nssec_background_crosstab_doctors['1.2_prop']+nssec_background_crosstab_doctors['2_prop']+nssec_background_crosstab_doctors['3_prop']+nssec_background_crosstab_doctors['4_prop']+nssec_background_crosstab_doctors['5_prop'])
# 7
hbar9 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['7_prop'], height = barheight, align='center', color = color_list1[8], label = label_list1[8], alpha = 1, left = nssec_background_crosstab_doctors['1.1_prop']+nssec_background_crosstab_doctors['Doctor_prop']+nssec_background_crosstab_doctors['1.2_prop']+nssec_background_crosstab_doctors['2_prop']+nssec_background_crosstab_doctors['3_prop']+nssec_background_crosstab_doctors['4_prop']+nssec_background_crosstab_doctors['5_prop']+nssec_background_crosstab_doctors['6_prop'])
# 8
hbar10 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['8_prop'], height = barheight, align='center', color = color_list1[9], label = label_list1[9], alpha = 1, left = nssec_background_crosstab_doctors['1.1_prop']+nssec_background_crosstab_doctors['Doctor_prop']+nssec_background_crosstab_doctors['1.2_prop']+nssec_background_crosstab_doctors['2_prop']+nssec_background_crosstab_doctors['3_prop']+nssec_background_crosstab_doctors['4_prop']+nssec_background_crosstab_doctors['5_prop']+nssec_background_crosstab_doctors['6_prop']+nssec_background_crosstab_doctors['7_prop'])
# Not living with family
hbar11 = ax1a.barh(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['Not living with family_prop'], height = barheight, align='center', color = color_list1[10], label = label_list1[10], alpha = 1, left = nssec_background_crosstab_doctors['1.1_prop']+nssec_background_crosstab_doctors['Doctor_prop']+nssec_background_crosstab_doctors['1.2_prop']+nssec_background_crosstab_doctors['2_prop']+nssec_background_crosstab_doctors['3_prop']+nssec_background_crosstab_doctors['4_prop']+nssec_background_crosstab_doctors['5_prop']+nssec_background_crosstab_doctors['6_prop']+nssec_background_crosstab_doctors['7_prop']+nssec_background_crosstab_doctors['8_prop'])


# Add labels
fontsize = 10
# fmt='%.2f' for proportion, fmt='%.2f' for percentage
fmt = '%.0f'
ax1a.bar_label(hbar1, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar2, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar3, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar4, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar5, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar6, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar7, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar8, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar9, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar10, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar11, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')

ax1a.set_xlim(0,100)
ax1a.set_xlabel('Proportion (%)')
ax1a.set_ylabel('Year respondent turned 18')
ax1a.set_title(titlelabel1)
ax1a.legend(labels = label_list1, bbox_to_anchor=(legend_offset, 0.5), loc = 'center right') # move legend out of the way

# Final overall settings
# plt.gca().invert_yaxis() # invert axis   
plt.yticks(nssec_background_crosstab_doctors['y_pos'], nssec_background_crosstab_doctors['var_tidy']) # set labels manually

# Below if want to hide y axis title and labels
# ax1a.yaxis.label.set_visible(False) # hide y axis title
# ax1a.set_yticklabels([]) # hide y axis tick labels
# Panel 2 - All others


### NOT DOCTORS PLOT
# Do plot
fig, ax1a = plt.subplots(figsize=(width1a,height))
barheight = 0.85
# 1.1
hbar1 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['Doctor_prop'], height = barheight, align='center', color = color_list1[0], label = label_list1[0], alpha = 1)
# 1.2 Doctors
hbar2 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['1.1_prop'], height = barheight, align='center', color = color_list1[1], label = label_list1[1], alpha = 1, left = nssec_background_crosstab_NOTdoctors['Doctor_prop'])
# 1.2 Other
hbar3 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['1.2_prop'], height = barheight, align='center', color = color_list1[2], label = label_list1[2], alpha = 1, left = nssec_background_crosstab_NOTdoctors['1.1_prop']+nssec_background_crosstab_NOTdoctors['Doctor_prop'])
# 2
hbar4 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['2_prop'], height = barheight, align='center', color = color_list1[3], label = label_list1[3], alpha = 1, left = nssec_background_crosstab_NOTdoctors['1.1_prop']+nssec_background_crosstab_NOTdoctors['Doctor_prop']+nssec_background_crosstab_NOTdoctors['1.2_prop'])
# 3
hbar5 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['3_prop'], height = barheight, align='center', color = color_list1[4], label = label_list1[4], alpha = 1, left = nssec_background_crosstab_NOTdoctors['1.1_prop']+nssec_background_crosstab_NOTdoctors['Doctor_prop']+nssec_background_crosstab_NOTdoctors['1.2_prop']+nssec_background_crosstab_NOTdoctors['2_prop'])
# 4
hbar6 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['4_prop'], height = barheight, align='center', color = color_list1[5], label = label_list1[5], alpha = 1, left = nssec_background_crosstab_NOTdoctors['1.1_prop']+nssec_background_crosstab_NOTdoctors['Doctor_prop']+nssec_background_crosstab_NOTdoctors['1.2_prop']+nssec_background_crosstab_NOTdoctors['2_prop']+nssec_background_crosstab_NOTdoctors['3_prop'])
# 5
hbar7 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['5_prop'], height = barheight, align='center', color = color_list1[6], label = label_list1[6], alpha = 1, left = nssec_background_crosstab_NOTdoctors['1.1_prop']+nssec_background_crosstab_NOTdoctors['Doctor_prop']+nssec_background_crosstab_NOTdoctors['1.2_prop']+nssec_background_crosstab_NOTdoctors['2_prop']+nssec_background_crosstab_NOTdoctors['3_prop']+nssec_background_crosstab_NOTdoctors['4_prop'])
# 6
hbar8 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['6_prop'], height = barheight, align='center', color = color_list1[7], label = label_list1[7], alpha = 1, left = nssec_background_crosstab_NOTdoctors['1.1_prop']+nssec_background_crosstab_NOTdoctors['Doctor_prop']+nssec_background_crosstab_NOTdoctors['1.2_prop']+nssec_background_crosstab_NOTdoctors['2_prop']+nssec_background_crosstab_NOTdoctors['3_prop']+nssec_background_crosstab_NOTdoctors['4_prop']+nssec_background_crosstab_NOTdoctors['5_prop'])
# 7
hbar9 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['7_prop'], height = barheight, align='center', color = color_list1[8], label = label_list1[8], alpha = 1, left = nssec_background_crosstab_NOTdoctors['1.1_prop']+nssec_background_crosstab_NOTdoctors['Doctor_prop']+nssec_background_crosstab_NOTdoctors['1.2_prop']+nssec_background_crosstab_NOTdoctors['2_prop']+nssec_background_crosstab_NOTdoctors['3_prop']+nssec_background_crosstab_NOTdoctors['4_prop']+nssec_background_crosstab_NOTdoctors['5_prop']+nssec_background_crosstab_NOTdoctors['6_prop'])
# 8
hbar10 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['8_prop'], height = barheight, align='center', color = color_list1[9], label = label_list1[9], alpha = 1, left = nssec_background_crosstab_NOTdoctors['1.1_prop']+nssec_background_crosstab_NOTdoctors['Doctor_prop']+nssec_background_crosstab_NOTdoctors['1.2_prop']+nssec_background_crosstab_NOTdoctors['2_prop']+nssec_background_crosstab_NOTdoctors['3_prop']+nssec_background_crosstab_NOTdoctors['4_prop']+nssec_background_crosstab_NOTdoctors['5_prop']+nssec_background_crosstab_NOTdoctors['6_prop']+nssec_background_crosstab_NOTdoctors['7_prop'])
# Not living with family
hbar11 = ax1a.barh(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['Not living with family_prop'], height = barheight, align='center', color = color_list1[10], label = label_list1[10], alpha = 1, left = nssec_background_crosstab_NOTdoctors['1.1_prop']+nssec_background_crosstab_NOTdoctors['Doctor_prop']+nssec_background_crosstab_NOTdoctors['1.2_prop']+nssec_background_crosstab_NOTdoctors['2_prop']+nssec_background_crosstab_NOTdoctors['3_prop']+nssec_background_crosstab_NOTdoctors['4_prop']+nssec_background_crosstab_NOTdoctors['5_prop']+nssec_background_crosstab_NOTdoctors['6_prop']+nssec_background_crosstab_NOTdoctors['7_prop']+nssec_background_crosstab_NOTdoctors['8_prop'])


# Add labels
fontsize = 10
# fmt='%.2f' for proportion, fmt='%.2f' for percentage
fmt = '%.0f'
ax1a.bar_label(hbar1, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar2, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar3, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar4, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar5, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar6, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar7, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar8, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar9, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar10, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')
ax1a.bar_label(hbar11, fmt=fmt, padding = 0, fontsize = fontsize, label_type = 'center', color = 'white')

ax1a.set_xlim(0,100)
ax1a.set_xlabel('Proportion (%)')
ax1a.set_ylabel('Year respondent turned 18')
ax1a.set_title(titlelabel2)
ax1a.legend(labels = label_list1, bbox_to_anchor=(legend_offset, 0.5), loc = 'center right') # move legend out of the way

# Final overall settings
# plt.gca().invert_yaxis() # invert axis   
plt.yticks(nssec_background_crosstab_NOTdoctors['y_pos'], nssec_background_crosstab_NOTdoctors['var_tidy']) # set labels manually



#%% Other stats
describe_age = data_lfs_filter_exclude_unknown.groupby('Flag_CurrentDoctor')['AGE'].describe()
describe_age = data_lfs_filter_exclude_unknown['AGE'].describe()


