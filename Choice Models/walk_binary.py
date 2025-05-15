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
from biogeme.expressions.elementary_expressions import DefineVariable

sele = ['scenario', 'choice']
data_path = 'final_choice_dataset.csv'
raw_data = pd.read_csv(data_path).dropna()
raw_data = raw_data.drop(columns = sele)
raw_data = raw_data.sort_values(by=["rid"])

database = db.Database("walk_bin", raw_data)
database.panel("rid")
globals().update(database.variables)
R= 100

WALKTIME = database.define_variable("WALKTIME", Variable("walktime"))

WALKPSAFE = database.define_variable("WALKPSAFE", Variable("walkpsafe") - 4)
CHOICE = database.define_variable("CHOICE", Variable("binchoice4"))

ASC_WALK = Beta("ASC_WALK", 0, -1000, 1000, 0)
B_WALKTIME = Beta("B_WALKTIME", 0, -1000, 1000, 0)

B_WALKPSAFE = Beta("B_WALKPSAFE", 0, -1000, 1000, 0)

S_WALKTIME = Beta("S_WALKTIME", 1, -1000, 1000, 0)

S_WALKPSAFE = Beta("S_WALKPSAFE", 1, -1000, 1000, 0)

R_WALKTIME = B_WALKTIME + S_WALKTIME * bioDraws("R_WALKTIME", "NORMAL_HALTON2")

R_WALKPSAFE = B_WALKPSAFE + S_WALKPSAFE * bioDraws("R_WALKPSAFE", "NORMAL_HALTON2")

V0 = 0
V1 = ASC_WALK + R_WALKTIME * WALKTIME + R_WALKPSAFE * WALKPSAFE
V = {0: V0, 1: V1}
av = {0: 1, 1: 1}

obs_logprob = models.logit(V, av, CHOICE)
panel_prob = PanelLikelihoodTrajectory(obs_logprob)
logprob = log(MonteCarlo(panel_prob))
biogeme = bio.BIOGEME(database, logprob, numberOfDraws=R)
biogeme.modelName = "walk_bin"
logger = msg.bioMessage()
logger.setDetailed()
biogeme.generateHtml = True
results = biogeme.estimate()
print(results.getEstimatedParameters())