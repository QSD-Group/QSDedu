# Wastewater Soft Sensor

Create a model for prediction of sulfur compounds concentration in wastewater.

Materials and data for this module are based on the [Wastewater Soft Sensor](https://www.kaggle.com/competitions/wastewater-soft-sensor) Kaggle competition.


## Introduction
Wastewater treatment is essential to protect public health. Untreated wastewater can contain harmful bacteria, viruses, and parasites that can cause serious illness, but also residual chemicals from different industrial plants. By removing these contaminants, wastewater treatment plants help to ensure that our water supplies are safe to drink and recreate in.

One wastewater treatment plant from Croatia implemented software sensors (soft sensors) and advanced process control for the purpose of increasing their treatment efficiency. Soft sensors are not physical instruments but computer models that estimate wastewater parameters based on existing data. This allows for continuous monitoring and optimization of treatment processes without the need for additional hardware sensors. Although soft sensors have a lot of potential in industrial application, the model on which it is based needs to be robust and accurate.

Process engineers of that wastewater treatment plant have concluded that this is not the case for their model. Namely, the displayed concentrations of sulfur compounds in the wastewater of chemical plants show much different values than those shown by the analysis carried out in the laboratory.


## Problem Statement
Your team has been tasked with fixing this. Process engineers from the plant came up with the idea to solve the problem using artificial intelligence. They employed you to develop a new model from the data they have collected through two months of successful plant operation. They have not specified which machine learning technique you have to use, they left that decision to you.

### Hint
There are two main branches of machine learning, supervised and unsupervised learning, you have been given data that allows you to take both paths, but be sure to correctly prepare your data.


## Data
Three datasets are available in CSV format.

**plant_data.csv: wastewater treatment plant data**
| Parameter | Description | Data Type |
| --- | --- | --- |
| `DT` | Date and time | datetime |
| `TI1` | Temperature of water inlet to rector 1, in Kelvin | float |
| `T1` | Temperature in reactor 1, in Kelvin | float |
| `WF` | Flow of water, in kg/h | float |
| `TI2` | Temperature of water inlet to rector 2, in Kelvin | float |
| `TO` | Temperature of water outlet, in Kelvin | float |
| `S` | Concentration of sulfur compounds predicted by the old soft sensor, in ppm | float |

**lab_data.csv: data from the analytical laboratory**

*This is the "real" concentration of the sulfur compounds.*

| Parameter | Description | Data Type |
| --- | --- | --- |
| `DT` | Date and time | datetime |
| `LS` | Concentration of sulfur compounds analytically determined, in ppm | float |


## Evaluation
The accuracy of your results will be evaluated using Root Mean Squared Error (RMSE). It measures the average squared difference between the predicted values and the actual values. It takes the square root of the MSE, which makes it easier to interpret because it is in the same units as the dependent variable.

```math
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n}*\sum_{i=1}^n*(y_i-\hat{y_i})^2}
```

## Prompting Questions
* How does your model compare with the old soft sensor?
* In addition to RMSE, are there other criteria that can be used to evaluate the accuracy of the model?
* When developing a model, are there considerations in addition to accuracy?
* Does your model perform fairly well for all data? Or are there a couple of outliers?


## Citation
Marko Sejdić. (2024). Wastewater soft sensor. Kaggle. https://kaggle.com/competitions/wastewater-soft-sensor
