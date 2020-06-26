# NeXLUncertainties

NeXLUncertainies is a library for propagating the uncertainty in multivariate
measurement models. What does this mean?

A measurement model is a way to transform the measured values into the desired
output values.  A very simple measurement model would be the following:
  * You measure a length in inches but need to transform the measured value by multiplying by 25.4 mm/inch to report in millimeters.
  * Slightly more complex, you measure the length and width of a rectangle and need to report the area.

These models are considered univariate because there is a single output value.

A multivariate measurement model is a model in which there are multiple outputs.
![Comparing univariate and multivariate measurement models](univsmulti.png)

A simple example of multivariate measurement model:
  * You measure the length and width of a rectangle and need to report the area, the perimeter and the aspect ratio.

If you have a model of the univariate type [Measurements.jl](https://github.com/JuliaPhysics/Measurements.jl)
is probably what you want.  If you have a model of the multivariate-type
`NeXLUncertainties` might be a better choice.

Often the distinction doesn't matter.  However, if, in our simple multivariate
example, the area, perimeter and aspect ratio are used in subsequent calculations,
ignoring the correlations between these values is likely to mis-estimate
the resulting uncertainty.

The distinctions can matter for X-ray microanalysis where many k-ratios are
measured and many compositional metrics are reported.  In between, there is
a complex measurement model that depends upon many measured inputs and
many physical parameters.  Hence this library was developed as part of the
NeXL collection of X-ray microanalysis algorithms.

## Getting Started
This [page](gettingstarted.html) will help you to get started.
