"""
An interface for an agent model which can make state-based predictions as well as update with new information.

We can take the state + operators approach. It works for state because all scenarios have a similar underlying
representation: state features and values. But a model?

Is it just a state-action mapping? No. It's a state x obs history -> action mapping. Just like a POMDP.
Is dynamic programming relevant here, as with belief states?

Is it a calculation? It can be, but it may be potentially computationally intensive.

Should we cache the already computed components? How does a model update affect cached results?

Is a model Markovian? 30 heads/70 tails -> 30% prediction, but 3/7 is updated very differently.

Eventually, the MDP planner will run with state being scenario state + current model. It needs to be hashable.

Other goal: override components with communication. Single instant commitments + multiple trial trends.

In effect, a model is a distribution over possible instances. Communication highlights which instant we are in.
A partially observable case with observations in two forms.

Update instance model with every observation/communication.
Update global model... between trials?

There are dependent components and conditionally independent components. A factored model?

Multiple issues here:
    - Deterministic teammate or probabilistic
        + obs/comms are gold truths for deterministic
        + obs/comms are instances of possible plans in probabilistic
    - Global vs individual model
        + global model can be updated later
        + individual model should be updated online

What should I choose?
    - I like the idea of stochastic models
    - general model to individual is some weird interpolation

Global model -> a distribution over potential instances of behavior   (a prior)
Observations and communication -> adjust probabilities, improving expectation (?)
Learning -> what is the association between the global model and the individual model?

A big problem is what to do when an agent is different from anything seen, when it defies expectation.

What we want:
    - when behavior aligns with known instances: converge to plan
    - when behavior is different: remain uncertain (let comm handle it)
"""


class Model:

    def predict(self, state):
        raise NotImplementedError('Have not implemented predict method of Model class ' + str(self.__class__))

    def update(self, state, action):
        raise NotImplementedError('Have not implemented update method of Model class ' + str(self.__class__))



