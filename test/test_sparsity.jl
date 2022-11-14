using Dynare

context = @dynare "test/models/irbc/irbc2";
Dynare.sparsity(context)
context = @dynare "test/models/irbc/irbc20";
Dynare.sparsity(context)
context = @dynare "test/models/irbc/irbc100";
Dynare.sparsity(context)

