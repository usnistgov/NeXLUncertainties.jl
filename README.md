# NeXLUncertainties

NeXLUncertainties has two parts:  UncertainValue and UncertainValues

## UncertainValue
Carries a single variable and the associated uncertainty. Does not address correlation between variables
  * UncertainValue implements functions of a single variable like sin(), cos(), exp(), log()
  * UncertainValue does not implement functions of two (or more) UncertainValue like *, /, +, - because it doesn't know whether the variable are correlated

## UncertainValues
Carries a set of variables and the covariance matrix that represented the uncertainties and the correlations between the variables.
  * UncertainValues implements an array of values and a correlation matrix the variable in which are identified by Label-derived structures.
  * Operations on UncertainValues are propagated using functions and Jacobian matrices.

This is very much a work in progress.  It has passed basic testing but remains a little tenuous.
