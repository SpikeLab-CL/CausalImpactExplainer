
import pandas as pd
import numpy as np
import streamlit as st
import subprocess
import pickle
import json
from utils import (plotly_time_series, estimate_model,
                   get_n_most_important_vars, plot_top_n_relevant_vars,
                   plot_statistics, send_parameters_to_r, texto)


st.title("Causal Impact Explainer")
texto('This app shows some info about jeje', nfont=11)

@st.cache
def load_example_data_dict() -> dict:
    with open("example_data/tablitas.pickle", 'rb') as file:
        data_dict = pickle.load(file)
    for key, df in data_dict.items():
        data_dict[key] = df.sort_values(by="date").loc[1::, :]
    return data_dict

#data_dict = load_example_data_dict()
#dataframe_names = list(data_dict.keys())

#selected_df_key = st.radio("Choose a dataframe",
#                                dataframe_names,
#                                 index=0)

#chosen_df = data_dict[selected_df_key]

@st.cache
def load_feather_dataframe() -> pd.DataFrame:
    df = pd.read_feather("example_data/input_causal_impact.feather")
    df['date'] = df.date.dt.strftime('%Y-%m-%d')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    return df



def plot_vars(df_experiment, vars_to_plot, time_var, beg_pre_period=None, end_pre_period=None,
                beg_eval_period=None, end_eval_period=None):
    """
    TODO: plot_vars should change with the training and evaluation periods
    """
    for var in vars_to_plot:
        df_experiment[f'{var}_scaled'] = (df_experiment[var] - df_experiment[var].mean())/df_experiment[var].std()

    scalled = st.checkbox('Plot scaled variables', value=False)
    if scalled:
        vars_to_plot = [f'{var}_scaled' for var in vars_to_plot]
    fig = plotly_time_series(df_experiment, time_var, vars_to_plot, beg_pre_period, end_pre_period, beg_eval_period, end_eval_period)
    st.plotly_chart(fig)

def sidebar(df_experiment : pd.DataFrame, 
            chosen_df : pd.DataFrame,
            time_var,
            y_var):

    texto('<b> Causal Impact Explainer </b>',
          nfont=16,
          color='black',
          line_height=1,
          sidebar=True)
    alpha = st.sidebar.slider("Select significance level", 0.01, 0.5, value=0.1)

    #TODO: dates if dataframe has dates (indices if not)
    last_data_point = len(df_experiment) - 1

    min_date = df_experiment[time_var].min().date()
    last_date = df_experiment[time_var].max().date()
    mid_point = int(len(df_experiment) / 2)


    beg_pre_period = st.sidebar.slider('Beginning Pre Period', min_date, last_date,
                                   value=df_experiment.loc[mid_point - 1, time_var].date())
    end_pre_period = st.sidebar.slider('End Pre Period', beg_pre_period, last_date, 
                                     value=df_experiment.loc[mid_point + 1, time_var].date())

    beg_eval_period = st.sidebar.slider('Beginning Evaluation Period',
                                        end_pre_period, last_date,
                                        value=df_experiment.loc[mid_point + 2, time_var].date())
    end_eval_period = st.sidebar.slider('End Evaluation Period', 
                                        beg_eval_period, last_date, value=last_date)

    st.sidebar.markdown("### Select variables")

    x_vars = [col for col in chosen_df.columns if col != y_var and col != time_var]
    selected_x_vars = st.sidebar.multiselect("Variable list", x_vars,
                        default=x_vars)
    strftime_format="%Y-%m-%d"
    parameters = {"alpha": alpha, 
                  "beg_pre_period": beg_pre_period.strftime(strftime_format),
                  "end_pre_period": end_pre_period.strftime(strftime_format),
                  "beg_eval_period": beg_eval_period.strftime(strftime_format),
                  "end_eval_period": end_eval_period.strftime(strftime_format),
                  "selected_x_vars": selected_x_vars,
                  "y_var": y_var,
                  "time_var": time_var,
                  #"experiment": selected_experiment
            }
    return parameters


def main():

    chosen_df = load_feather_dataframe()
    col1, col2 = st.beta_columns(2)
    with col1:
        time_var = st.selectbox("Choose the time variable",
                               chosen_df.columns,
                               index=0) #date

        experiment_var = st.selectbox("Which variable identifies the individual experiment",
                                  chosen_df.columns,
                                  index=2)

    with col2:

        y_var = st.selectbox("Choose the outcome variable (y)",
                     chosen_df.columns,
                     index=1) #sales
        experiments_to_eval = list(chosen_df[experiment_var].unique())

        selected_experiment = st.selectbox("Choose an experiment to evaluate",
                            experiments_to_eval,
                            index=0)


    

    df_experiment = chosen_df[chosen_df[experiment_var] == selected_experiment].copy() 
    df_experiment[time_var] = pd.to_datetime(df_experiment[time_var])
    df_experiment.sort_values(time_var, inplace=True)
    df_experiment.index = range(len(df_experiment))

    parameters = sidebar(df_experiment, chosen_df, time_var, y_var)

    st.markdown('---')

    with st.beta_expander('Ploting variables'):
        vars_to_plot = st.multiselect("Variables to plot",
                                  list(df_experiment.columns),
                                  default=df_experiment.columns[1])

        plot_vars(df_experiment, vars_to_plot, time_var, 
                  beg_pre_period=parameters['beg_pre_period'],
                  end_pre_period=parameters['end_pre_period'],
                  beg_eval_period=parameters['beg_eval_period'], 
                  end_eval_period=parameters['end_eval_period'],
                  )

    with st.beta_expander('Show dataframe'):
        st.table(df_experiment.head(5))
    
    if st.checkbox("Estimate Causal Impact model with R"):
        #Save and run R
        df_experiment.to_csv("example_data/input_causal_impact_one_experiment.csv")
        send_parameters_to_r("example_data/parameters_for_r.json")
        #TODO: manage errors while executing R script
        completed = subprocess.run(["Rscript", "causal_impact_one_experiment.R"],
                        capture_output=True)
        st.write("Output from R (for debugging)")
        st.write(completed)

        #Bring results from R
        results_from_r = pd.read_feather(
            "example_data/results_causal_impact_from_r.feather")
        st.write(results_from_r.head(5))

        fig = plot_statistics(results_from_r)
        st.plotly_chart(fig)

    if st.checkbox('Estimate Causal Impact model with python'):
        #TODO: must be redone!
        ci = estimate_model(chosen_df, y_var,
                            selected_x_vars,
                            beg_pre_period, end_pre_period, beg_eval_period,
                            end_eval_period)
        fig, _ = ci.plot()
        st.pyplot(fig)
        st.text(ci.summary())

        #st.text(ci.summary('report'))

        st.markdown("### Most important variables (according to coefficients)")

        top_n_vars = get_n_most_important_vars(ci, 1)

        y_and_top_vars = [y_var] + top_n_vars
        fig, _ = plot_top_n_relevant_vars(chosen_df, time_var, y_and_top_vars,
                                          beg_eval_period)
        st.pyplot(fig)

main()
