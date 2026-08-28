"""
Validation entry point — delegates to local_lib.models.val.validate.
"""
import sys
sys.path.insert(0, ".")
from local_lib.models.val import validate
from local_lib.utils.set_parse import parse_args_val


if __name__ == "__main__":
    validate(parse_args_val())
