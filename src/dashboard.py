
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from utils import myCausalImpact
np.random.seed(12345)


@st.cache
def generate_data() -> pd.DataFrame:
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    X = 100 + arma_process.generate_sample(nsample=100)
    y = 1.2 * X + np.random.normal(size=100)
    y[70:] += 5

    data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])

    return data

example_data = generate_data()

last_data_point = 99

beg_pre_period = st.slider('Beginning Pre Period', 0, last_data_point - 4, value=0)
end_pre_period = st.slider(
    'End Pre Period', beg_pre_period + 1, last_data_point - 3, value=69)

beg_eval_period = st.slider('Beginning Evaluation Period', end_pre_period + 1, last_data_point - 2,
                            value=70)
end_eval_period = st.slider(
    'End Evaluation Period', beg_eval_period + 1, last_data_point, value=last_data_point)


#@st.cache
def estimate_model():
    pre_period = [beg_pre_period, end_pre_period]
    eval_period = [beg_eval_period, end_eval_period]
    ci = myCausalImpact(example_data, pre_period, eval_period)
    return ci

ci = estimate_model()

def main():
    st.title("Causal Impact explainer")

    if st.checkbox('Show dataframe'):
        st.write(example_data.head(5))
    
    fig, axes = ci.plot()
    st.pyplot(fig)
    

    #st.write(ci.summary())
    st.markdown(ci.summary())
    
main()
