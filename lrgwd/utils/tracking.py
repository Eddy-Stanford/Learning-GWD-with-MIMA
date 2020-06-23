import os
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Union

import mlflow
from lrgwd.utils.logger import logger


@contextmanager
def tracking(
    experiment: Union[str, None] = None, 
    params: Dict[str, Any] = {}, 
    local_dir: Union[str, os.PathLike] = None,
    artifact_path: Union[str, os.PathLike] = None,
    tracking: bool = True,
):
    logger.debug("TRACKING")
    if tracking: 
        mlflow.set_experiment(experiment)
        with mlflow.start_run():
            mlflow.log_params(params)
    try: 
        yield
    except: 
        traceback.print_exc()
        logger.error("FAILURE")    
    else: 
        logger.debug("SUCCESS")    
                
    if local_dir and tracking:
        mlflow.log_artifact(local_dir, artifact_path)
