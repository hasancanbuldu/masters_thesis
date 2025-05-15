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

database = db.Database("ebike_bin", raw_data)
database.panel("rid")
globals().update(database.variables)
R= 100

EBIKETIME = database.define_variable("EBIKETIME", Variable("ebiketime"))
EBIKECOST = database.define_variable("EBIKECOST", Variable("ebikecost"))
EBIKEPSAFE = database.define_variable("EBIKEPSAFE", Variable("ebikepsafe") - 4)
CHOICE = database.define_variable("CHOICE", Variable("binchoice2"))

ASC_EBIKE = Beta("ASC_EBIKE", 0, -1000, 1000, 0)
B_EBIKETIME = Beta("B_EBIKETIME", 0, -1000, 1000, 0)
B_EBIKECOST = Beta("B_EBIKECOST", 0, -1000, 1000, 0)
B_EBIKEPSAFE = Beta("B_EBIKEPSAFE", 0, -1000, 1000, 0)

S_EBIKETIME = Beta("S_EBIKETIME", 1, -1000, 1000, 0)
S_EBIKECOST = Beta("S_EBIKECOST", 1, -1000, 1000, 0)
S_EBIKEPSAFE = Beta("S_EBIKEPSAFE", 1, -1000, 1000, 0)

R_EBIKETIME = B_EBIKETIME + S_EBIKETIME * bioDraws("R_EBIKETIME", "NORMAL_HALTON2")
R_EBIKECOST = B_EBIKECOST + S_EBIKECOST * bioDraws("R_EBIKECOST", "NORMAL_HALTON2")
R_EBIKEPSAFE = B_EBIKEPSAFE + S_EBIKEPSAFE * bioDraws("R_EBIKEPSAFE", "NORMAL_HALTON2")

V0 = 0
V1 = ASC_EBIKE + R_EBIKETIME * EBIKETIME + R_EBIKECOST * EBIKECOST + R_EBIKEPSAFE * EBIKEPSAFE
V = {0: V0, 1: V1}
av = {0: 1, 1: 1}

obs_logprob = models.logit(V, av, CHOICE)
panel_prob = PanelLikelihoodTrajectory(obs_logprob)
logprob = log(MonteCarlo(panel_prob))
biogeme = bio.BIOGEME(database, logprob, numberOfDraws=R)
biogeme.modelName = "ebike_binary"
logger = msg.bioMessage()
logger.setDetailed()
biogeme.generateHtml = True
results = biogeme.estimate()
print(results.getEstimatedParameters())