######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.27
# function: argument parsing for RF-DETR training.
######################################################
import argparse
import logging

_logger = logging.getLogger("set_parse")


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
        is_dict = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif val.startswith("{") and val.endswith("}"):
            is_dict = True
            val = val[1:-1]
        elif "," not in val:
            sep = ":" if ":" in val else "=" if "=" in val else None
            if sep and val.count(sep) == 1:
                k, v = val.split(sep, 1)
                return (DictAction._parse_int_float_bool(k), DictAction._parse_int_float_bool(v))
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        elif is_dict:
            result = {}
            for item in values:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                k, v = item
                result[k] = v
            return result
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        i = 0
        while i < len(values):
            kv = values[i]
            key, val = kv.split("=", maxsplit=1)
            bracket_depth = val.count("{") - val.count("}")
            j = i + 1
            while j < len(values):
                nxt = values[j]
                nxt_key = nxt.split("=", maxsplit=1)[0]
                if nxt_key != key:
                    break
                nxt_val = nxt.split("=", maxsplit=1)[1]
                val += "," + nxt_val
                bracket_depth += nxt_val.count("{") - nxt_val.count("}")
                if bracket_depth <= 0:
                    j += 1
                    break
                j += 1
            i = j
            if bracket_depth > 0 and not val.startswith("{"):
                val = "{" + val + "}"
            elif bracket_depth == 0 and not val.startswith("{") and not val.startswith("[") and not val.startswith("(") and ":" in val and "," in val:
                val = "{" + val + "}"
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


def merge_from_dict(args, merge_key="options"):
    args_dict = vars(args)
    if merge_key not in args:
        _logger.info(f"{merge_key} not in args")
        return args_dict
    merge_value = args_dict.pop(merge_key)
    for add_key, add_v in merge_value.items():
        args_dict[add_key] = add_v
    return args_dict


def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR training script")
    parser.add_argument("--data", type=str, required=True, help="path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[576], help="input image size(HxW)")
    parser.add_argument("--device", type=str, default="0", help="GPU IDs")
    parser.add_argument("--project", type=str, default="ckpts", help="project name")
    parser.add_argument("--name", type=str, default="train", help="run name")
    parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--model", type=str, default="medium",
                        choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"],
                        help="model size")
    parser.add_argument("--pretrained", type=str, default=None, help="path to pretrained checkpoint")
    parser.add_argument("--options", nargs="+", action=DictAction, default=None,
                       help="extra key=value pairs merged into args")

    args = parser.parse_args()

    try:
        import re
        dev = getattr(args, 'device', None)
        if isinstance(dev, str):
            s = dev.strip()
            if s == '-1' or s.lower() == 'cpu':
                args.device = 'cpu'
            else:
                if re.fullmatch(r'[0-9,\s]+', s):
                    ids = [x for x in re.split(r'[,\s]+', s.strip()) if x != '']
                    if len(ids) == 1:
                        args.device = f"cuda:{ids[0]}"
                    else:
                        args.device = 'cuda'
    except Exception:
        pass

    try:
        if getattr(args, 'options', None):
            cfg = merge_from_dict(args, merge_key="options")
            for k, v in cfg.items():
                setattr(args, k, v)
    except Exception:
        pass

    return args


def parse_args_val():
    parser = argparse.ArgumentParser(description="RF-DETR validation script")
    parser.add_argument(
        "--model", type=str,
        default="ckpts/detect/BakingRecognizeCOCO/202608141803/checkpoint_best_total.pth",
        help="path to checkpoint",
    )
    parser.add_argument(
        "--data", type=str,
        default="data/det-dataset/BakingRecognizeCOCO",
        help="path to dataset directory or dataset.yaml",
    )
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="input image size")
    parser.add_argument("--device", type=str, default="0", help="device, e.g. 0, cpu, cuda:0")
    parser.add_argument("--workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--project", type=str, default="runs", help="project directory for validation outputs")
    parser.add_argument("--name", type=str, default="val", help="run name under project directory")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="dataset split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.3, help="confidence threshold for saved visualizations")
    parser.add_argument("--no-save-vis", action="store_true", help="disable prediction visualizations")
    parser.add_argument("--trust-checkpoint", action="store_true", help="allow unsafe checkpoint deserialization")
    parser.add_argument("--options", nargs="+", action=DictAction, default=None,
                       help="extra key=value pairs (e.g. class_mapping=5:0,6:1,7:2)")

    args = parser.parse_args()

    try:
        import re
        dev = getattr(args, 'device', None)
        if isinstance(dev, str):
            s = dev.strip()
            if s == '-1' or s.lower() == 'cpu':
                args.device = 'cpu'
            else:
                if re.fullmatch(r'[0-9,\s]+', s):
                    ids = [x for x in re.split(r'[,\s]+', s.strip()) if x != '']
                    if len(ids) == 1:
                        args.device = f"cuda:{ids[0]}"
                    else:
                        args.device = 'cuda'
    except Exception:
        pass

    return args