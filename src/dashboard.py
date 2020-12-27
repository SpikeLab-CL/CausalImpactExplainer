
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
#from utils import plotly_time_series, get_n_most_important_vars
from utils import (plotly_time_series, estimate_model,
                   get_n_most_important_vars, plot_top_n_relevant_vars)

np.random.seed(12345)

st.title("Causal Impact explainer")

@st.cache
def load_example_data() -> dict:
    with open("example_data/tablitas.pickle", 'rb') as file:
        data_dict = pickle.load(file)
    for key, df in data_dict.items():
        data_dict[key] = df.sort_values(by="date").loc[1::, :]
    return data_dict

data_dict = load_example_data()
dataframe_names = list(data_dict.keys())

selected_df_key = st.radio("Choose a dataframe",
                                 dataframe_names,
                                 index=0)

chosen_df = data_dict[selected_df_key]

time_var = st.selectbox("Choose the time variable",
                           chosen_df.columns,
                           index=0) #date

y_var = st.selectbox("Choose the outcome variable (y)",
                 chosen_df.columns,
                 index=1) #sales

#Assuming dataframe is sorted by date
chosen_df.index = range(len(chosen_df))

vars_to_plot = st.multiselect("Variables to plot",
                              list(chosen_df.columns),
                              default=chosen_df.columns[1])

for plot_var in vars_to_plot:
    fig = plotly_time_series(chosen_df, time_var, plot_var)
    st.plotly_chart(fig)

#TODO: selection should be done with dates if dataframe has dates
#(and indices if not    )
last_data_point = len(chosen_df) - 1

beg_pre_period = st.sidebar.slider('Beginning Pre Period', 0,
     last_data_point - 4, value=0)
end_pre_period = st.sidebar.slider(
    'End Pre Period', beg_pre_period + 1, last_data_point - 3, value=69)

beg_eval_period = st.sidebar.slider('Beginning Evaluation Period', end_pre_period + 1, last_data_point - 2,
                            value=70)
end_eval_period = st.sidebar.slider(
    'End Evaluation Period', beg_eval_period + 1, last_data_point, value=last_data_point)

st.sidebar.markdown("### Select variables")

x_vars = [col for col in chosen_df.columns if col != y_var and col != time_var]
selected_x_vars = st.sidebar.multiselect("Variable list", x_vars,
                       default=x_vars)



def main():
    
    if st.checkbox('Show dataframe'):
        st.write(chosen_df.head(5))
    
    if st.checkbox('Estimate Causal Impact model'):
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
