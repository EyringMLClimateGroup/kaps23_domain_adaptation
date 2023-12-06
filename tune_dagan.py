"""
HPO wrapper around the domain adaptation
"""

from ray import tune,remote,nodes,cluster_resources
from ray.tune.schedulers import ASHAScheduler
from ray import init as rayinit
from ray.tune.tuner import Tuner
from ray.air import session
from ray.air.config import RunConfig,ScalingConfig
from ray.tune.tune_config import TuneConfig
import sys
from datetime import datetime
import numpy as np
import os 
import traceback
from ray.tune import CLIReporter,ProgressReporter
from ray.tune.experiment.trial import Trial
from ray.air.checkpoint import Checkpoint
from ray.air import FailureConfig
from ray.tune.search.hyperopt import HyperOptSearch as Algo
#import nevergrad as ng
import luigi
from train_cycle_gans_da import TrainCycleGAN
import time
import glob

rayinit(address="auto",)
#tunable parameters
parameters = { 
                "gradient_penalty_mode": tune.choice(["none", "gulrajani"]),
                "gradient_penalty_weight": tune.qloguniform(0.1,100,0.1),
                "df_gen" : tune.qrandint(24,256,4),
                "df_disc" : tune.randint(4,13),
                "identity_lambda" : tune.quniform(0,10,0.5),
                "cycled_lambda" : tune.quniform(0,10,0.5),
                "gan_lambda" : tune.quniform(0,50,1),
                "segmentation_consistency_lambda" : tune.quniform(0,5,.1),
                "normtype" : tune.choice(["batchnorm", "instancenorm", "no"]),
                "pretrain" : tune.qrandint(0,2,1),
                "activation" : tune.choice(["relu","softplus","linear"]),
                "reg" : tune.choice([False,True]),
                "const" : tune.quniform(1e-4,0.1,0.0001),
                "clr" : tune.choice([False,True]),
                "lr" : tune.loguniform(1e-9,0.0001),
                "water_path_lambda" : tune.uniform(1e-5,2),
                "n_critic" : tune.choice([1,5,15,50]),
    }
#fixed parameters
general= {    "folder" : os.path.join(os.environ["SCR"],"checkpoints"),
                "dataset_folder" : os.path.join(os.environ["WORK"],"dataset_dt"),
                "seed" : 123,
                "suffix": "tuning",
                "appl" : False,
                "drop" : True,
                "explog" : "tuning",
                "variables" : [0,1,2,3,4,5,6,7,8],
                "gen_groups" :1,
                "disc_groups" : 1,
                "patch_size" : 32,
                "resume" : False,
                "timelimit" : "02:00:00",
                "wstein" : True,
                "batch_size" : 800,
                "epochs" : 500,
                "full" : True,
                "dummy": 0,
                "tune" : True
            }
variable_names = ["cwp","lwp","iwp","cerl","ceri","cod","ptop","tsurf","clt"]
class my_reporter(CLIReporter):
    def __init__(self):
        super(my_reporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated

rep = my_reporter()

def mk_tname(trial):
    return trial.trial_id+(str(datetime.today()).replace("-","").replace(":","").replace(" ",""))[:14]

name = "diss_tuning_dagan"
general["tune_name"] = name
path = os.path.join(os.environ["SCR"],"ray_results",name )
if not os.path.exists(path):
    os.makedirs(path)
    name_new = name#where to save
    name_prev= name#what to load
else:
    it = 1
    while True:
        if not os.path.exists(path+"_"+str(it)):
            name_new = name+"_"+str(it)
            os.makedirs(path+"_"+str(it))
            break
        name_prev = name+"_"+str(it)
        it+=1
if not os.path.exists(os.path.join(path,"../checkpoints")):
    os.makedirs(os.path.join(path,"../checkpoints"))

def fun_to_tune(config):
    """config is combination of general and tuning choice"""
    config.update(general)
    tr_id = session.get_trial_id()    
    config["suffix"]+=str(tr_id)
    start = time.time()
    luigi.build([TrainCycleGAN(**config)], local_scheduler=True, log_level="ERROR")
    duration = time.time()-start
    time.sleep(2)
    outfolder = os.path.join(os.environ["SCR"],"ray_results",name)
    file = glob.glob(os.path.join(outfolder,"*cycle_{}_{}_{}*".format(config["df_gen"],
                                                                            config["df_disc"],config["suffix"])))
    if len(file)==1:
        out = np.load(file[0])
        with np.printoptions(precision=2, suppress=True):
            print("SUCCEEDED",tr_id, out)
        met_dict = {"mean_JSD_impr" : np.nanmean(out[:-1]),"CWP_const":out[-1],
                "duration":duration,
                "total improvement" : int(np.nansum(out[:-1]>0))}
        indiv_dict = {variable_names[x]:out[x] for x in config["variables"]}
        met_dict.update(indiv_dict)
        session.report(metrics=met_dict,
                        checkpoint=Checkpoint(os.path.join(path,"../checkpoints")))
    else:
        
        print("Failed",tr_id, file)
        met_dict = {"mean_JSD_impr" : -100,"CWP_const":1e5,
                "duration":duration,
                "total improvement" : 0}
        indiv_dict = {variable_names[x]:-100 for x in config["variables"]}
        met_dict.update(indiv_dict)
        session.report(metrics=met_dict,
                        checkpoint=Checkpoint(os.path.join(path,"../checkpoints")))
        os.system("rm -r {}/logs/cycle*{}*".format(os.environ["SCR"],tr_id))
        
#not using most features of the scheduler because each choice reports exactly once
#could not implement differently because of the luigi wrapper around everything
#even without that I wouldnt really know how to report and keep going
asha = ASHAScheduler(
    time_attr='duration',#useless in terms of tuning
    grace_period=10, #dont think that does anything
    reduction_factor=3,#by which the number of trials is reduced at each optimization step
    brackets=1,#dont really get what this would change. maybe if i would use this as scheduler and not just searcher
)

res = cluster_resources()
print(res)
num_gpu=int(res["GPU"])
num_a100=int(res["accelerator_type:A100"])

try:
    previous = Tuner.restore(trainable = fun_to_tune,path=os.path.join(os.environ["SCR"],"ray_results", name_prev),
                restart_errored=True)
    try:
        best_config = previous.get_results().get_best_result(metric="mean_JSD_impr", mode="max").config
        print("best",best_config)
        algo=Algo(space=parameters,
                 metric = "mean_JSD_impr",
                 mode = "max", 
                 points_to_evaluate = [best_config] )
    except RuntimeError:
        algo=Algo(space=parameters,
                 metric = "mean_JSD_impr",
                 mode = "max", )
    algo.restore_from_dir( os.path.join(os.environ["SCR"],"ray_results", name_prev))
    print("algo restored")

except RuntimeError:
    traceback.print_exc()
    print("SADFACE, ALGO NOT RESTORED",flush=True)
    algo=Algo(space=parameters,
                 metric = "mean_JSD_impr",
                 mode = "max",  )

tuner = Tuner(tune.with_resources(
            trainable=fun_to_tune,
            resources = tune.PlacementGroupFactory([{"CPU": 0, "GPU": 4}])),#{"cpu" : 0, "gpu" : 4}),
            #param_space = parameters,
            run_config = RunConfig(name_new, verbose=3,
                                    local_dir= os.path.join(os.environ["SCR"],"ray_results"),
                                    progress_reporter = rep,
                                    #failure_config=FailureConfig(fail_fast=True),
                                    log_to_file =os.path.join(os.environ["SCR"],"tuning_logs.txt"),
                                    sync_config=tune.SyncConfig( syncer=None)),
            tune_config = TuneConfig( search_alg=algo,
                                        num_samples = -1, time_budget_s = 3600*3,
                                        trial_dirname_creator = mk_tname,
                                        max_concurrent_trials=num_gpu//4,
                                        
                                        #scheduler = asha
                                        ))
print("max concurrent {}".format(num_gpu//4))  
if "best_config" in globals():
    analysis = tuner.fit()
elif "previous" in globals():
    analysis = previous.fit()
else:
    analysis = tuner.fit()
df = analysis.get_dataframe()
print("Number trials",len(df))
print("best mean impr: ", analysis.get_best_result(metric="mean_JSD_impr", mode="max"))
print("best number impr: ", analysis.get_best_result(metric="total improvement", mode="max"))
print("best best cwp const: ", analysis.get_best_result(metric="CWP_const", mode="min"))
