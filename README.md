# NeXLUncertainties

| **Documentation**                        | **Build Status**                  |
|:----------------------------------------:|:---------------------------------:|
| [![][docs-stable-img]][docs-stable-url]  | [![][travis-img]][travis-url]     |


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://pages.nist.gov/NeXLUncertainties.jl
[travis-img]: https://travis-ci.com/usnistgov/NeXLUncertainties.jl.svg?branch=master
[travis-url]: https://travis-ci.com/usnistgov/NeXLUncertainties.jl

NeXLUncertainties implements propagation of uncertainties for multivariate measurement models.  A multivariate measurement model represents a class of models in which many input parameters (measured values or model parameters) are mapped into more than one output parameter.  Calculating these kinds of models using a univariate measurement model ignores correlations in the output parameters that result from sharing the same input parameters.

If your measurement model involves multiple inputs but only one output then [Measurements.jl](https://github.com/JuliaPhysics/Measurements.jl) is probably a better choice.

Multivariate measurement models are discussed in [JCGM 102:2011 Evaluation of measurement data – Supplement 2 to the “Guide to the expression of uncertainty in measurement” – Extension to any number of output quantities](https://www.bipm.org/utils/common/documents/jcgm/JCGM_102_2011_E.pdf) which serves as inspiration for this library.  This document presents a model based on a first-order Taylor series approximation. In the multivariate case, the relationship between the input and output uncertainties is represented by a Jacobian matrix - a matrix of partial derivatives of the output parameters with respect to the input parameters.  A detailed discussion and some simple examples are presented in [Embracing Uncertainty: Modeling the Standard Uncertainty in Electron Probe Microanalysis—Part I ](https://www.cambridge.org/core/journals/microscopy-and-microanalysis/article/embracing-uncertainty-modeling-the-standard-uncertainty-in-electron-probe-microanalysispart-i/3C65B4F344444F26A32E6C321FC85B62).

NeXLUncertainties primarily provides two data types:  UncertainValue and UncertainValues

## UncertainValue
Carries a single variable and the associated uncertainty. Does not address correlation between variables
  * UncertainValue implements functions of a single variable like sin(), cos(), exp(), log()
  * UncertainValue does not implement functions of two (or more) UncertainValue like *, /, +, - because it doesn't know whether the variable are correlated

## UncertainValues
Carries a set of variables and the covariance matrix that represented the uncertainties and the correlations between the variables.
  * UncertainValues implements an array of values and a correlation matrix the variable in which are identified by Label-derived structures.
  * Operations on UncertainValues are propagated using functions and Jacobian matrices.
