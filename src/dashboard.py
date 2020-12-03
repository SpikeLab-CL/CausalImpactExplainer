
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import streamlit as st
from utils import myCausalImpact
from utils import plotly_time_series

np.random.seed(12345)

st.title("Causal Impact explainer")

@st.cache
def generate_data() -> pd.DataFrame:
    n = 100
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    x1 = 100 + arma_process.generate_sample(nsample=n)
    x2 = 115 + arma_process.generate_sample(nsample=n)
    y = 1.2 * x1 -0.3*x2 + np.random.normal(size=n)
    t = range(1, n + 1)
    y[70:] += 5

    data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 't': t})

    return data

example_data = generate_data() #type: ignore






vars_to_plot = st.multiselect("Variables to plot",
                 list(example_data.columns),
               default=example_data.columns[0])

time_var = "t"
for plot_var in vars_to_plot:
    fig = plotly_time_series(example_data, time_var, plot_var)
    st.plotly_chart(fig)

last_data_point = 99

beg_pre_period = st.sidebar.slider('Beginning Pre Period', 0, last_data_point - 4, value=0)
end_pre_period = st.sidebar.slider(
    'End Pre Period', beg_pre_period + 1, last_data_point - 3, value=69)

beg_eval_period = st.sidebar.slider('Beginning Evaluation Period', end_pre_period + 1, last_data_point - 2,
                            value=70)
end_eval_period = st.sidebar.slider(
    'End Evaluation Period', beg_eval_period + 1, last_data_point, value=last_data_point)

st.sidebar.markdown("### Select variables")

x_vars = [col for col in example_data.columns if col != 'y']
selected_x_vars = st.sidebar.multiselect("Variable list", x_vars,
                       default=x_vars)
#@st.cache
def estimate_model(df):
    pre_period = [beg_pre_period, end_pre_period]
    eval_period = [beg_eval_period, end_eval_period]
    selected_x_vars_plus_target = ['y'] + selected_x_vars
    ci = myCausalImpact(
        df[selected_x_vars_plus_target], pre_period, eval_period)
    return ci



def main():
    
    if st.checkbox('Show dataframe'):
        st.write(example_data.head(5))
    

    if st.checkbox('Estimate Causal Impact model'):
        ci = estimate_model(example_data)
        fig, axes = ci.plot()
        st.pyplot(fig)
        st.text(ci.summary())

main()
