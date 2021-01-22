
import pandas as pd
import numpy as np
import streamlit as st
import subprocess
from PIL import Image
from utils import (plotly_time_series, estimate_model,
                   get_n_most_important_vars, plot_top_n_relevant_vars,
                   plot_statistics, send_parameters_to_r, texto, max_width_)


st.title("Causal Impact Explainer :volcano:")
texto("""This dashboard can help you explore various Causal Impact packages""",
   nfont=17)
texto('Choose the following parameters ')

@st.cache
def load_dataframe() -> pd.DataFrame:
    #file = st.file_uploader("I'm a file uploader")
    #Emission scandal breaks out on 18th of September 2015
    df = pd.read_feather("example_data/volks_data_clean.feather")
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
    plotly_time_series(df_experiment, time_var, vars_to_plot, beg_pre_period, end_pre_period, beg_eval_period, end_eval_period)
    

def sidebar(df_experiment : pd.DataFrame, 
            chosen_df : pd.DataFrame,
            time_var,
            y_var):

    image = Image.open('causal_impact_explainer_logo.png')
    st.sidebar.image(image, caption='', use_column_width=True)

    st.sidebar.markdown("#### Select the variables you will use as control")

    x_vars = [col for col in chosen_df.columns if col != y_var and col != time_var and col!='group']
    selected_x_vars = st.sidebar.multiselect("Better less than more", x_vars,
                        default=x_vars)

    st.sidebar.markdown("## Experiment setting")
    #alpha = st.sidebar.slider("Significance level", 0.01, 0.5, value=0.1)
    alpha = st.sidebar.number_input("Significance level", 0.01, 0.5, value=0.1, step=0.01)

    #TODO: indices dataframe doesn't have dates
    min_date = df_experiment[time_var].min().date()
    last_date = df_experiment[time_var].max().date()
    mid_point = int(len(df_experiment) / 2)

    st.sidebar.markdown("### Beginning and end pre period")

    beg_pre_period, end_pre_period = st.sidebar.slider('', min_date, last_date,
                                   value=(df_experiment.loc[mid_point - 150, time_var].date(),
                                   df_experiment.loc[mid_point + 1, time_var].date()))

    st.sidebar.markdown("### Beginning and end evaluation period")
    beg_eval_period, end_eval_period = st.sidebar.slider('',
                                        end_pre_period, last_date,
                                        value=(df_experiment.loc[mid_point + 20, time_var].date(),
                                        last_date))
    
    strftime_format="%Y-%m-%d"
    parameters = {"alpha": alpha, 
                  "beg_pre_period": beg_pre_period.strftime(strftime_format),
                  "end_pre_period": end_pre_period.strftime(strftime_format),
                  "beg_eval_period": beg_eval_period.strftime(strftime_format),
                  "end_eval_period": end_eval_period.strftime(strftime_format),
                  "selected_x_vars": selected_x_vars,
                  "y_var": y_var,
                  "time_var": time_var,
                 }

    return parameters


def main():
    max_width_(width=1000)
    chosen_df = load_dataframe()
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

    #TODO: make it optional to choose experiment groups
    df_experiment = chosen_df[chosen_df[experiment_var] == selected_experiment].copy()
    df_experiment[time_var] = pd.to_datetime(df_experiment[time_var])
    df_experiment.sort_values(time_var, inplace=True)
    df_experiment.index = range(len(df_experiment))

    parameters = sidebar(df_experiment, chosen_df, time_var, y_var)

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
    
    with st.beta_expander("Estimate Causal Impact model with R"):
        if st.checkbox("Run R", False):
            #Save and run R
            df_experiment.to_csv("example_data/input_causal_impact_one_experiment.csv")
            send_parameters_to_r("example_data/parameters_for_r.json", parameters, selected_experiment)
            #TODO: manage errors while executing R script
            completed = subprocess.run(["Rscript", "causal_impact_one_experiment.R"],
                            capture_output=True)
            if st.checkbox("Show output from R (for debugging)"):
                st.write(completed)

            #Bring results from R
            results_from_r = pd.read_feather(
                "example_data/results_causal_impact_from_r.feather")
            if st.checkbox("Show results dataframe"):
                st.write(results_from_r.head(5))

            fig = plot_statistics(results_from_r, index_col=parameters["time_var"])
            st.plotly_chart(fig)

    with st.beta_expander('Estimate Causal Impact model with python'):
        if st.checkbox("Run Python", False):
            #Where is the input for time_var
            x_vars = [parameters["y_var"], "x0", "x1"]
            df__ = df_experiment[x_vars]
            ci = estimate_model(df__, parameters["y_var"],
                                x_vars,
                                3,
                                170,
                                171,
                                220)
            fig, _ = ci.plot()
            st.pyplot(fig)
            st.text(ci.summary())
#
            st.text(ci.summary('report'))
    #
            st.markdown("### Most important variables (according to coefficients)")
    #
            top_n_vars = get_n_most_important_vars(ci, 1)
    #
            y_and_top_vars = [parameters["y_var"]] + top_n_vars
            fig, _ = plot_top_n_relevant_vars(df__,
                     parameters["time_var"], y_and_top_vars,
                                              171)
            st.pyplot(fig)

main()
