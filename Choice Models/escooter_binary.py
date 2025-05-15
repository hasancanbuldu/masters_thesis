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

database = db.Database("escoot_bin", raw_data)
database.panel("rid")
globals().update(database.variables)
R= 100

ESCOOTTIME = database.define_variable("ESCOOTTIME", Variable("escoottime"))
ESCOOTCOST = database.define_variable("ESCOOTCOST", Variable("escootcost"))
ESCOOTPSAFE = database.define_variable("ESCOOTPSAFE", Variable("escootpsafe") - 4)
CHOICE = database.define_variable("CHOICE", Variable("binchoice3"))

ASC_ESCOOT = Beta("ASC_ESCOOT", 0, -1000, 1000, 0)
B_ESCOOTTIME = Beta("B_ESCOOTTIME", 0, -1000, 1000, 0)
B_ESCOOTCOST = Beta("B_ESCOOTCOST", 0, -1000, 1000, 0)
B_ESCOOTPSAFE = Beta("B_ESCOOTPSAFE", 0, -1000, 1000, 0)

S_ESCOOTTIME = Beta("S_ESCOOTTIME", 1, -1000, 1000, 0)
S_ESCOOTCOST = Beta("S_ESCOOTCOST", 1, -1000, 1000, 0)
S_ESCOOTPSAFE = Beta("S_ESCOOTPSAFE", 1, -1000, 1000, 0)

R_ESCOOTTIME = B_ESCOOTTIME + S_ESCOOTTIME * bioDraws("R_ESCOOTTIME", "NORMAL_HALTON2")
R_ESCOOTCOST = B_ESCOOTCOST + S_ESCOOTCOST * bioDraws("R_ESCOOTCOST", "NORMAL_HALTON2")
R_ESCOOTPSAFE = B_ESCOOTPSAFE + S_ESCOOTPSAFE * bioDraws("R_ESCOOTPSAFE", "NORMAL_HALTON2")

V0 = 0
V1 = ASC_ESCOOT + R_ESCOOTTIME * ESCOOTTIME + R_ESCOOTCOST * ESCOOTCOST + R_ESCOOTPSAFE * ESCOOTPSAFE
V = {0: V0, 1: V1}
av = {0: 1, 1: 1}

obs_logprob = models.logit(V, av, CHOICE)
panel_prob = PanelLikelihoodTrajectory(obs_logprob)
logprob = log(MonteCarlo(panel_prob))
biogeme = bio.BIOGEME(database, logprob, numberOfDraws=R)
biogeme.modelName = "escoot_bin"
logger = msg.bioMessage()
logger.setDetailed()
biogeme.generateHtml = True
results = biogeme.estimate()
print(results.getEstimatedParameters())