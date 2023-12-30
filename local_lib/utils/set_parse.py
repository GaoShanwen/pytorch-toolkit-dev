######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.12.11
# filenaem: args_parse.py
# function: load options parses to args.
######################################################
import argparse
import os

import yaml


def merge_from_dict(args, merge_key="options"):
    args_dict = vars(args)
    if merge_key not in args:
        print(f"{merge_key} not in args")
        return args_dict
    merge_value = args_dict.pop(merge_key)
    for add_key, add_v in merge_value.items():
        args_dict[add_key] = add_v
    return args_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Validate a model")
    parser.add_argument("--config", type=str, help="train config file path")
    parser.add_argument("--resume", type=str, default="", help="resume from checkpoint path directory.")
    parser.add_argument("--amp", action="store_true", help="enable automatic-mixed-precision training")
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair in xxx=yyy format will "
        "be merged into config file. If the value to be overwritten is a list, it should be like "
        'key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space is allowed.",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    # Do we have a config file to parse?
    args_config, remaining = parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    # use options to overwrite original cfgs!
    if args_config.options:
        cfg = merge_from_dict(args, merge_key="options")
        parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


class DictAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    def __init__(self, *args, **kwargs):
        super(DictAction, self).__init__(*args, **kwargs)
        self.nargs = "*"

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val == "None":
            return None
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (char == ",") and (pre.count("(") == pre.count(")")) and (pre.count("[") == pre.count("]")):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


if __name__ == "__main__":
    args, args_text = parse_args()
    print(args_text)
