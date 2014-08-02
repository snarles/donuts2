
The shape of the kernel is determined by the ratio of the
axial and radial signals. The predicted signal in all directions from a single
fiber depends on this ratio, while an additional weight \beta is a free
parameter of the model.

Therefore, we try to find a shape parameter \kappa defined as follows:

\kappa = log(\frac{S_{ortho}}{S_{para}})

where S_{ortho} and S_{para} are the signals measured orthogonal to and
parallel to a single fiber population, respectively.

=>

\kappa = log(\frac{S_0 e^{-b RD}}{{S_0 e^{-b AD}})

\kappa = log(\frac{e^{-b RD}}{{e^{-b AD}})

\kappa = log(e^{-b RD - (-b AD)}

\kappa = log(e^{b AD - b RD})

\kappa = log(e^{b (AD - RD)})

\kappa = b (AD - RD)
