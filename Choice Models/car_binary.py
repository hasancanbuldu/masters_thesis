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

database = db.Database("car_bin", raw_data)
database.panel("rid")
globals().update(database.variables)
R= 5000

CARTIME = database.define_variable("CARTIME", Variable("cartime"))
CARCOST = database.define_variable("CARCOST", Variable("carcost"))
CARPSAFE = database.define_variable("CARPSAFE", Variable("carpsafe") - 4)
CHOICE = database.define_variable("CHOICE", Variable("binchoice1"))

ASC_CAR = Beta("ASC_CAR", 0, -1000, 1000, 0)
B_CARTIME = Beta("B_CARTIME", 0, -1000, 1000, 0)
B_CARCOST = Beta("B_CARCOST", 0, -1000, 1000, 0)
B_CARPSAFE = Beta("B_CARPSAFE", 0, -1000, 1000, 0)

S_CARTIME = Beta("S_CARTIME", 1, -1000, 1000, 0)
S_CARCOST = Beta("S_CARCOST", 1, -1000, 1000, 0)
S_CARPSAFE = Beta("S_CARPSAFE", 1, -1000, 1000, 0)

R_CARTIME = B_CARTIME + S_CARTIME * bioDraws("R_CARTIME", "NORMAL_HALTON2")
R_CARCOST = B_CARCOST + S_CARCOST * bioDraws("R_CARCOST", "NORMAL_HALTON2")
R_CARPSAFE = B_CARPSAFE + S_CARPSAFE * bioDraws("R_CARPSAFE", "NORMAL_HALTON2")

V0 = 0
V1 = ASC_CAR + R_CARTIME * CARTIME + R_CARCOST * CARCOST + R_CARPSAFE * CARPSAFE
V = {0: V0, 1: V1}
av = {0: 1, 1: 1}

obs_logprob = models.logit(V, av, CHOICE)
panel_prob = PanelLikelihoodTrajectory(obs_logprob)
logprob = log(MonteCarlo(panel_prob))
biogeme = bio.BIOGEME(database, logprob, numberOfDraws=R)
biogeme.modelName = "Car_Binary"
logger = msg.bioMessage()
logger.setDetailed()
biogeme.generateHtml = True
results = biogeme.estimate()
print(results.getEstimatedParameters())