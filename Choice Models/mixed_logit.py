import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.messaging as msg
from biogeme import models
from biogeme.expressions import (
    Beta,
    bioDraws,
    Variable,
    PanelLikelihoodTrajectory,
    MonteCarlo,
    log,
)


sele = ['scenario', 'choice']
data_path = 'final_choice_dataset.csv'
raw_data = pd.read_csv(data_path).dropna()
raw_data = raw_data.drop(columns = sele)
raw_data = raw_data.sort_values(by=["rid"])

database = db.Database("mode_choice_ml", raw_data)
database.panel("rid")
globals().update(database.variables)

R = 5000

# Define transformed variables
CARPSAFE = database.define_variable("CARPSAFE", Variable("carpsafe") - 4)
EBIKEPSAFE = database.define_variable("EBIKEPSAFE", Variable("ebikepsafe") - 4)
ESCOOTPSAFE = database.define_variable("ESCOOTPSAFE", Variable("escootpsafe") - 4)
WALKPSAFE = database.define_variable("WALKPSAFE", Variable("walkpsafe") - 4)
CHOICE = database.define_variable("CHOICE", Variable("intchoice"))

# Coefficients (means)
ASC_CAR = Beta("ASC_CAR", 0, -1000, 1000, 0)
ASC_EBIKE = Beta("ASC_EBIKE", 0, -1000, 1000, 0)
ASC_ESCOOT = Beta("ASC_ESCOOT", 0, -1000, 1000, 0)
ASC_WALK = Beta("ASC_WALK", 0, -1000, 1000, 0)

B_CARTIME = Beta("B_CARTIME", 0, -1000, 1000, 0)
B_CARCOST = Beta("B_CARCOST", 0, -1000, 1000, 0)
B_CARPSAFE = Beta("B_CARPSAFE", 0, -1000, 1000, 0)

B_EBIKETIME = Beta("B_EBIKETIME", 0, -1000, 1000, 0)
B_EBIKECOST = Beta("B_EBIKECOST", 0, -1000, 1000, 0)
B_EBIKEPSAFE = Beta("B_EBIKEPSAFE", 0, -1000, 1000, 0)

B_ESCOOTIME = Beta("B_ESCOOTIME", 0, -1000, 1000, 0)
B_ESCOOTCOST = Beta("B_ESCOOTCOST", 0, -1000, 1000, 0)
B_ESCOOTPSAFE = Beta("B_ESCOOTPSAFE", 0, -1000, 1000, 0)

B_WALKTIME = Beta("B_WALKTIME", 0, -1000, 1000, 0)
B_WALKPSAFE = Beta("B_WALKPSAFE", 0, -1000, 1000, 0)

# Random components (standard deviations)
S_CARPSAFE = Beta("S_CARPSAFE", 1, -1000, 1000, 0)
S_EBIKEPSAFE = Beta("S_EBIKEPSAFE", 1, -1000, 1000, 0)
S_ESCOOTPSAFE = Beta("S_ESCOOTPSAFE", 1, -1000, 1000, 0)
S_WALKPSAFE = Beta("S_WALKPSAFE", 1, -1000, 1000, 0)

# Random parameters
R_CARPSAFE = B_CARPSAFE + S_CARPSAFE * bioDraws("R_CARPSAFE", "NORMAL_HALTON2")
R_EBIKEPSAFE = B_EBIKEPSAFE + S_EBIKEPSAFE * bioDraws("R_EBIKEPSAFE", "NORMAL_HALTON2")
R_ESCOOTPSAFE = B_ESCOOTPSAFE + S_ESCOOTPSAFE * bioDraws("R_ESCOOTPSAFE", "NORMAL_HALTON2")
R_WALKPSAFE = B_WALKPSAFE + S_WALKPSAFE * bioDraws("R_WALKPSAFE", "NORMAL_HALTON2")

# Utility functions
V = {
    1: ASC_WALK + B_WALKTIME * Variable("walktime") + R_WALKPSAFE * WALKPSAFE,
    2: ASC_ESCOOT + B_ESCOOTIME * Variable("escoottime") + B_ESCOOTCOST * Variable("escootcost") + R_ESCOOTPSAFE * ESCOOTPSAFE,
    3: ASC_EBIKE + B_EBIKETIME * Variable("ebiketime") + B_EBIKECOST * Variable("ebikecost") + R_EBIKEPSAFE * EBIKEPSAFE,
    4: ASC_CAR + B_CARTIME * Variable("cartime") + B_CARCOST * Variable("carcost") + R_CARPSAFE * CARPSAFE,
}

# Availability: assume all modes are available
av = {1: 1, 2: 1, 3: 1, 4: 1}

# Model estimation
obs_logprob = models.logit(V, av, CHOICE)
panel_prob = PanelLikelihoodTrajectory(obs_logprob)
logprob = log(MonteCarlo(panel_prob))

biogeme = bio.BIOGEME(database, logprob, numberOfDraws=R)
biogeme.modelName = "MixedLogit_ModeChoice"

logger = msg.bioMessage()
logger.setDetailed()
biogeme.generateHtml = True

results = biogeme.estimate()
print(results.getEstimatedParameters())