import pandas  as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.expressions as database
import seaborn as sns
import matplotlib.pyplot as plt
from biogeme.expressions.elementary_expressions import DefineVariable
from biogeme.expressions import Variable
from biogeme.expressions import Beta, log, Elem
import biogeme.biogeme_logging as blog
from biogeme.models import ordered_logit
from biogeme.expressions import bioDraws
from biogeme.expressions import MonteCarlo
# Load data

df = pd.read_csv("final_psafe_dataset.csv")
modes_to_keep = ['escooter']
filtered_df = df[df['mode'].isin(modes_to_keep)].copy()
sele = ['scenario', 'mode']
filtered_df = filtered_df.drop(columns = sele)
filtered_df['psafe'] = filtered_df['psafe'].astype(int)
database = db.Database("psafe", filtered_df)

globals().update(database.variables)

#Dependent Variable
psafe = Variable('psafe')

#Independent Variables
license_own= Variable('license_own')
age = Variable('age')
income = Variable('income')
type1 = Variable('type1')
type2 = Variable('type2') 
type4 = Variable('type4')
pav = Variable('pav')
cross1 = Variable('cross1')
cross2 = Variable('cross2')
obst = Variable('obst')
veh = Variable('veh')
bike= Variable('bike')
ped = Variable('ped')

#Betas
B_LICENSE = Beta('B_license_own', 0, None, None, 0)
B_AGE = Beta('B_AGE', 0, None, None, 0)
B_INCOME = Beta('B_INCOME', 0, None, None, 0)
B_PAV = Beta('B_PAV', 0, None, None, 0)
B_CROSS1 = Beta('B_CROSS1', 0, None, None, 0)
B_CROSS2 = Beta('B_CROSS2', 0, None, None, 0)
B_OBST = Beta('B_OBST', 0, None, None, 0)
B_VEH = Beta('B_VEH', 0, None, None, 0)
B_BIKE = Beta('B_BIKE', 0, None, None, 0)
B_PED = Beta('B_PED', 0, None, None, 0)

#Randoms
MU_TYPE1 = Beta('MU_TYPE1', 0, None, None, 0)
MU_TYPE2 = Beta('MU_TYPE2', 0, None, None, 0)
MU_TYPE4 = Beta('MU_TYPE4', 0, None, None, 0)

SIGMA_TYPE1 = Beta('SIGMA_TYPE1', 1, None, None, 0)
SIGMA_TYPE2 = Beta('SIGMA_TYPE2', 1, None, None, 0)
SIGMA_TYPE4 = Beta('SIGMA_TYPE4', 1, None, None, 0)

B_TYPE1 = MU_TYPE1 + SIGMA_TYPE1 * bioDraws('B_TYPE1_draw', 'NORMAL')
B_TYPE2 = MU_TYPE2 + SIGMA_TYPE2 * bioDraws('B_TYPE2_draw', 'NORMAL')
B_TYPE4 = MU_TYPE4 + SIGMA_TYPE4 * bioDraws('B_TYPE4_draw', 'NORMAL')

#thresholds with monotonicity
kappa1 = Beta('kappa1', -7, None, None, 0)
delta2 = Beta('delta2', 1, 1e-6, None, 0)
delta3 = Beta('delta3', 1, 1e-6, None, 0)
delta4 = Beta('delta4', 1, 1e-6, None, 0)
delta5 = Beta('delta5', 1, 1e-6, None, 0)
delta6 = Beta('delta6', 1, 1e-6, None, 0)

kappa2 = kappa1 + delta2
kappa3 = kappa2 + delta3
kappa4 = kappa3 + delta4
kappa5 = kappa4 + delta5
kappa6 = kappa5 + delta6

V = B_LICENSE * license_own + B_AGE * age + B_INCOME * income + B_TYPE1 * type1 + B_TYPE2 * type2 + B_TYPE4 * type4 + B_PAV * pav + B_CROSS1 * cross1 + B_CROSS2 * cross2 + B_OBST * obst + B_VEH * veh + B_BIKE * bike + B_PED * ped
    
the_proba = models.ordered_logit(
    V,      
    [1, 2, 3, 4, 5, 6, 7],
    kappa1,
    
)

the_chosen_proba = Elem(the_proba, psafe)
logprob = log(MonteCarlo(the_chosen_proba))

biogeme = bio.BIOGEME(database, logprob, number_of_draws=2000)
biogeme.modelName = 'ORD_LOG_escoot_Munich_HALTON'
biogeme.generateHtml = True
biogeme.draws = {'draws': 'HALTON'}
results = biogeme.estimate()
print(results.short_summary())
pandas_results = results.get_estimated_parameters()
pandas_results.round(3)
