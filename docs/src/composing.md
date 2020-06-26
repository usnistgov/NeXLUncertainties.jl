## Composing and Parallelizing

#### Sequential Steps

When one sits down to perform uncertainty propagation, it can be overwhelming.
The measurement models can be complex.  The thought of computing the necessary
partial derivatives can be intimidating.  However, many models can be factored
into a series of simple steps - compute this and then compute that from this.
Taken one-by-one, it is likely to be a lot easier to compute the partial
derivatives for the simple steps than for the full model.

Fortunately, we have the chain-rule of differential calculus.  The chain rule
says that

$\frac{δf(g(x))}{δx} = \left. \frac{δf(y)}{δy}\right |_{y=g(x)} \frac{δg(x)}{δx}$

The key fact here is that we can break a complex problem into a series of simpler
steps.  We don't need to compute $\frac{δf(g(x))}{δx}$ directly.  We can
compute $\frac{δf(y)}{δy}$ and $\frac{δg(x)}{δx}$ and use the chain rule to
compute $\frac{δf(g(x))}{δx}$.

In the multivariate world, it gets even better. If $F(X)$ and $G(Y)$ are a vector
functions of a vector variable and $J[F(X)]$ and $J[G(Y)]$ are the Jacobians of
$F(X)$ and $G(Y)$ respectively then

$J[G(F(X))] = \left. J[G(Y)] \right |_{Y=G(X)} J[F(X)]$

That is to say, if we compute the Jacobian matrix for each step, we can
compute the Jacobian for the entire calculation by taking the product of the
Jacobians. All the bookkeeping necessary handle all the variables is performed
by matrix products.  This is far simpler than the univariate measurement
model case outlined in the [BIPM GUM](https://www.bipm.org/en/publications/guides/gum.html).
For comparison, consider equation 13 on page 21 of the GUM.  Equation 13 is
exactly equivalent to the Jacobian expression in the univariate case but
the bookkeeping is so much less clear.

The equivalent of equation 13 in the multivariate case is

$U(F(X)) = J[F(X)] U(X) J[F(X)]^T$

Clean, simple, straighforward.

From a practical perspective, this library allows you to implementing
`MeasurementModel` types to represent each step in a calculation.  So
let's say we implement measurement models `MM1`, `MM2` and `MM3`.  The
output of `MM1` is a superset of the values required as input to `MM2`
and the output of `MM2` is a superset of the values required as input
to `MM3`. We can create an object to represent the composition of these
models using the "compose operator" ∘ - `MM1to3 = MM3 ∘ MM2 ∘ MM1`.
`MM1to3(X)` is conceptually equivalent to `MM3(MM2(MM1(X)))`. (The ∘
operator is syntactic sugar for the `ComposedMeasurementModel` type.)

Like we did on the [Getting Started](gettingstarted.html) page, it is
possible to apply the composed `MeasurementModel` `MM1to3` to either input
represented by an `UncertainValues` object (the full uncertainty calculation)
or a `LabeledValues` object (just the function evaluation). The computational
cost of the entire calculation is roughly the sum of the cost of the individual
steps plus the matrix products of the Jacobians.

One subtlety - sometimes it is necessary to pass variables unmodified from
one step to the next.  This is of course handled trivially by the identity
function with derivative of unity.  Since this is common, there is a special
mechanism to handle this using the `MaintainInputs` or `AllInputs`
`MeasurementModel`s combined with the next concept - parallel steps.

See [this page](resistors.html) for an example of a simplish multi-step computation.

#### Parallel Steps

In some measurement models, the same calculation is repeated on multiple sets
of input data.  In this case, it is useful to be able to apply the same
`MeasurementModel` to different sub-sets of the data.  For example, if you
want to compute the X-ray mass absorption coefficient (MAC) for a series
of different X-rays lines, you might define a generic `MeasurementModel` `ComputeMAC`
that computes the MAC for a single X-ray line.  You could apply the model
sequentially as described in "sequential steps".  However, it would often be
more efficient to compute the MACs in parallel as a single step.

There is a syntax for that using the "¦" operator which for this purpose I'll
call the "parallel operator".  If we have X-rays represented by `x1`, `x2` and
`x3` then we could create a single step `x123 = x1 ¦ x2 ¦ x3`


There are two operators to handle steps that can be calculated in parallel
and combined to look like a single step = "¦" and "|".  The first of these,
\\brokenbar or "¦", uses `@threads` to parallelize the calculation. The second
performs the calculation sequentially.  If the calculation is simple and quick
use "|".  If the calculation is longer and more complex, "¦" *may* be faster
depending upon the relative cost of the calculation and the cost of spinning
up multiple threads.  "|" and "¦" are syntactic sugar for the
"ParallelMeasurementModel" type.

There is a second use for the "|" and "¦" operators (or the
`ParallelMeasurementModel` type) is to pass variables unmodified from one step
to the next using the `MaintainInputs` and `AllInputs` `MeasurementModel` types.
