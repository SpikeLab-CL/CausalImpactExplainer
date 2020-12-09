import matplotlib.pyplot as plt
from causalimpact import CausalImpact
import pandas as pd
import plotly.express as px
import numpy as np
from typing import List

class myCausalImpact(CausalImpact):
    def __init__(self, data, pre_period, post_period, model=None, alpha=0.05, **kwargs):
        super(myCausalImpact, self).__init__(data, pre_period,
                                             post_period, model, alpha, **kwargs)
        #checked_input = self._process_input_data(
        #    data, pre_period, post_period, model, alpha, **kwargs
        #)


    def plot(self, panels=['original', 'pointwise', 'cumulative'], figsize=(15, 12)):
        """Plots inferences results related to causal impact analysis.
            Args
            ----
            panels: list.
                Indicates which plot should be considered in the graphics.
            figsize: tuple.
                Changes the size of the graphics plotted.
            Raises
            ------
            RuntimeError: if inferences were not computed yet.
            """
        
        fig = plt.figure(figsize=figsize)
        if self.summary_data is None:
            raise RuntimeError(
                'Please first run inferences before plotting results')

        valid_panels = ['original', 'pointwise', 'cumulative']
        for panel in panels:
            if panel not in valid_panels:
                raise ValueError(
                    '"{}" is not a valid panel. Valid panels are: {}.'.format(
                        panel, ', '.join(['"{}"'.format(e)
                                            for e in valid_panels])
                    )
                )

        # First points can be noisy due approximation techniques used in the likelihood
        # optimizaion process. We remove those points from the plots.
        llb = self.trained_model.filter_results.loglikelihood_burn #type: ignore
        inferences = self.inferences.iloc[llb:]

        intervention_idx = inferences.index.get_loc(self.post_period[0])
        n_panels = len(panels)
        ax = plt.subplot(n_panels, 1, 1)
        idx = 1

        if 'original' in panels:
            ax.plot(pd.concat([self.pre_data.iloc[llb:, 0], self.post_data.iloc[:, 0]]),  # type: ignore
                    'k', label='y') 
            ax.plot(inferences['preds'], 'b--',
                    label='Predicted')  # type: ignore
            ax.axvline(
                inferences.index[intervention_idx - 1], c='k', linestyle='--')
            ax.fill_between(
                self.pre_data.index[llb:].union(self.post_data.index),
                inferences['preds_lower'],
                inferences['preds_upper'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            ax.grid(True, linestyle='--')
            ax.legend()
            if idx != n_panels:
                plt.setp(ax.get_xticklabels(), visible=False)
            idx += 1

        if 'pointwise' in panels:
            ax = plt.subplot(n_panels, 1, idx, sharex=ax)
            ax.plot(inferences['point_effects'], 'b--', label='Point Effects')
            ax.axvline(
                inferences.index[intervention_idx - 1], c='k', linestyle='--')
            ax.fill_between(
                inferences['point_effects'].index,
                inferences['point_effects_lower'],
                inferences['point_effects_upper'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            ax.axhline(y=0, color='k', linestyle='--')
            ax.grid(True, linestyle='--')
            ax.legend()
            if idx != n_panels:
                plt.setp(ax.get_xticklabels(), visible=False)  # type: ignore
            idx += 1

        if 'cumulative' in panels:
            ax = plt.subplot(n_panels, 1, idx, sharex=ax)
            ax.plot(inferences['post_cum_effects'], 'b--',
                    label='Cumulative Effect')
            ax.axvline(
                inferences.index[intervention_idx - 1], c='k', linestyle='--')
            ax.fill_between(
                inferences['post_cum_effects'].index,
                inferences['post_cum_effects_lower'],
                inferences['post_cum_effects_upper'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            ax.grid(True, linestyle='--')  # type: ignore
            ax.axhline(y=0, color='k', linestyle='--')  # type: ignore
            ax.legend()  # type: ignore

        # Alert if points were removed due to loglikelihood burning data
        if llb > 0:
            text = ('Note: The first {} observations were removed due to approximate '
                    'diffuse initialization.'.format(llb))
            fig.text(0.1, 0.01, text, fontsize='large')  # type: ignore

        return fig, fig.axes  # type: ignore


def plotly_time_series(df, time_var, plot_var):
    fig = px.line(df.sort_values(by=time_var),
                  x=time_var,
                  y=plot_var)

    fig.update_layout(height=200,
                      width=800,
                      xaxis_title="Time",
                      yaxis_title=plot_var)

    return fig


def estimate_model(df: pd.DataFrame, y_var_name: str, x_vars: List,
                 beg_pre_period, end_pre_period, beg_eval_period,
                   end_eval_period) -> myCausalImpact:
    pre_period = [beg_pre_period, end_pre_period]
    eval_period = [beg_eval_period, end_eval_period]
    selected_x_vars_plus_target = [y_var_name] + x_vars
    ci = myCausalImpact(
        df[selected_x_vars_plus_target], pre_period, eval_period)
    return ci


def get_n_most_important_vars(trained_c_impact: myCausalImpact, top_n: int):
    """
    Get the names of the n most important variables in the training of the causal impact
    model.
    Most important is given by the absolute value of the coefficient
    (I THINK that data is standardized beforehand so scale of X shouldn't matter)
    """
    params: pd.Series = trained_c_impact.trained_model.params #type: ignore
    contains_beta = params.index.str.contains("beta")
    does_not_contain_t = params.index != "beta.t"
    params = params[contains_beta & does_not_contain_t]
    params = np.abs(params)

    top_n_vars = params.sort_values(ascending=False).index.values[:top_n]

    top_n_vars = [var.split(".")[1] for var in top_n_vars]
    return top_n_vars


def plot_top_n_relevant_vars(df, time_var, y_and_top_vars: List[str],
                            beg_eval_period):
    n_total_vars = len(y_and_top_vars)
    fig, axes = plt.subplots(n_total_vars, 1,
                             figsize=(6, 1.5*n_total_vars))
    for ax, var_name in zip(axes, y_and_top_vars):  # type: ignore
        ax.plot(df[time_var], df[var_name])
        ax.set_title(var_name)
        ax.axvline(beg_eval_period, c='k', linestyle='--')

    return fig, axes
