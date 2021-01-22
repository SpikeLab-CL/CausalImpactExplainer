library(CausalImpact)
library(feather)
library(ggplot2)
library(rjson)

options(error = function() {
  sink(stderr())
  on.exit(sink(NULL))
  traceback(3, max.lines = 1L)
  if (!interactive()) {
    q(status = 1)
  }
})

parameters = fromJSON(file="example_data/parameters_for_r.json")
alpha = parameters$alpha
y_var = parameters$y_var
x_vars = parameters$selected_x_vars
time_var = parameters$time_var
experiment = parameters$experiment

#TODO: these dates aren't used yet
beg_pre_period = as.Date(parameters$beg_pre_period)
end_pre_period = as.Date(parameters$end_pre_period)
beg_eval_period = as.Date(parameters$beg_eval_period)
end_eval_period = as.Date(parameters$end_eval_period)

output_file = "example_data/results_causal_impact_from_r.feather"

df = read.csv("example_data/input_causal_impact_one_experiment.csv")
df[ , time_var] = as.POSIXct(as.Date(df[ , time_var]))

pre.period <- as.POSIXct(c(beg_pre_period, end_pre_period))
post.period <- as.POSIXct(c(beg_eval_period, end_eval_period))

x = df[ , x_vars]
data = cbind(y=df[ , y_var], x)
data <- zoo(data, df[ , time_var])
    
impact <- CausalImpact(data, pre.period, post.period, alpha=alpha)
#, model.args = list(nseason=52))
#impact.plot <- plot(impact)
#impact.plot <- impact.plot + theme_bw(base_size = 10) +
#             ggtitle(paste("Impact for ", group, sep=" "))
#print(impact.plot)
results = data.frame(impact[[1]])
results$experiment_name = experiment
results[ , time_var] = df[ , time_var]

write_feather(results, output_file)





