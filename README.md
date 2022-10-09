# AI_Ass1
Assignment 1 for AI
Group:
Jarred Fisher - FSHJAR002
Joshua Rosenthal - RSNJOS005
Shai Aarons - ARNSHA011

Bayesian Network for Heart Disease and Heart Attack risk profiling.

Decision Network for deciding whether or not to call ambulance, based on utility value calculations.

The two models can be found in the project root directory. "MI_bayes.bif" (Bayesian network) and "MI_decision.bifxml" (decision network).

## SETUP and RUN

If you would to build the models from scratch please see the following instructions:

Please run `make` in the project directory. This will setup the python virtual environment and install necessary dependencies.

Then run `make run`, this will run the wrapper file, which will build the bayesian and decision networks. It will also print some posterior distribution information. At this point the models are made and are stored in the project root directory.

A more interactive session can be found in the "src/visualisation.ipynb" which is a jupyter notebook. All the various diagrams and inferences found in the report, are present there for you to view. You should be able to run the notebook straight out of vscode (we tested this), or perhaps through some other IDE of your liking. Please run the first code block before any of the other use cases.