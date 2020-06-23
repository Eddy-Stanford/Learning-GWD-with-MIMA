import json
import os
import pickle
import traceback
from typing import Any, Dict, Union

from lrgwd.utils.logger import logger


def to_json(
    path: Union[str, os.PathLike], 
    obj: Union[Dict, str],
) -> None:
    try:
        with open(path, 'w+') as fp: 
            json.dump(obj, fp)
    except:  
        traceback.print_exc()
        logger.error(f"Failed to json save {path}")


def to_pickle(
    path: Union[str, os.PathLike], 
    obj: Any,
) -> None:
    try: 
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:  
        traceback.print_exc()
        logger.error(f"Failed to pickle {path}")


def from_pickle(path: Union[str, os.PathLike]) -> Any:
    try: 
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
    except:  
        traceback.print_exc()
        logger.error(f"Failed to load pickle {path}")
    else: 
        return b
