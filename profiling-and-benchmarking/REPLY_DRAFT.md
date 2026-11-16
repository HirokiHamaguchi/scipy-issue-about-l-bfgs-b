# Reply to https://github.com/scipy/scipy/issues/26137#issuecomment-5697689243

Thank you.

I agree that changing the BLAS thread count is not a good solution.
I would therefore like to address the `formk` and `subsm` issues.

One point I would like to clarify is that the main idea from the previous post may also be beneficial for bound-constrained problems.
When no variables are active, the subspace problem is effectively a full-space problem.

In the early stages of optimization, a bound-constrained problem can behave much like an unconstrained problem, and this phase can account for a substantial portion of the optimization in practical applications.

Even with the default BLAS thread count, the approach appears to speed up the optimization by a factor of 2–3 (or at least around 1.5×), as suggested by @MohammadSadeghSalehi's and my experiments.

I think this speedup is significant enough to be worth considering.
Thus, I believe the situations in which this approach is beneficial are common enough to warrant the change.

Of course, if the `structural change` is preferable to these changes, I would be happy to implement it.

One concern I have is that it may be difficult to match the current implementation at the level of floating-point roundoff, so this could slightly affect backward compatibility.

Please let me know if there are any additional benchmarks that would be useful.
I am interested in this issue and would be happy to help further.
