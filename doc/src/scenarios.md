## Interface for complex simulations scenarios

- scenario block
- information set <date>
 - periods <date> [- <date>]
 - <symbol> <value>
 - instruments <exogenous variables>... (if endogenous known in the
   future)
   
### Algorithms
- perfect foresight: adjust residuals and Jacobian matrix
- linear model:
  - future exogenous variables: augmented linear rational expectations
  - future endogenous variables: stacking special periods + linear solution 
  - future parameters: stacking special periods + linear solution
  - unicity of a stable trajectory
    - stack model must be full rank
	- unicity of stable trajectory in terminal model
