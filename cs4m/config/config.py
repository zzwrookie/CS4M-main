import argparse
import os
import sys
from copy import deepcopy

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional for DB-only utilities
    yaml = None

try:
    from yacs.config import CfgNode as CN
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for utility scripts
    class CN(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

        def merge_from_file(self, yml_file: str) -> None:
            if yaml is None:
                raise ModuleNotFoundError("pyyaml is required to load YAML config files.")
            with open(yml_file, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            self._merge_dict(data)

        def _merge_dict(self, data: dict) -> None:
            for key, value in data.items():
                if isinstance(value, dict):
                    node = self.get(key)
                    if not isinstance(node, CN):
                        node = CN()
                    node._merge_dict(value)
                    self[key] = node
                else:
                    self[key] = value


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)


def _repo_dir(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def _ensure_trailing_sep(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


ROOT_GROUND_TRUTH_DIR = _repo_dir("ground_truth")
ROOT_DATA_DIR = os.path.abspath(os.getenv("CLAD_DATA_ROOT", os.path.join(WORKSPACE_ROOT, "data")))


def _dataset_raw_dir(dataset_folder: str) -> str:
    return _ensure_trailing_sep(os.path.join(ROOT_DATA_DIR, dataset_folder))


DATABASE_DEFAULT_CONFIG = {
    "host": os.getenv("CLAD_DB_HOST", "localhost"),
    "user": os.getenv("CLAD_DB_USER", "postgres"),
    "password": os.getenv("CLAD_DB_PASSWORD", ""),
    "port": os.getenv("CLAD_DB_PORT", "5432"),
}


# 当前 CLAD 主流程只需要这一个运行期开关：
# - True: 使用导入全量原始日志后对应的数据库
# - False: 使用子集数据库
CONFIG_SCHEMA = {
    "runtime": {
        "use_all_db_files": bool,
        "event_type_filter": bool,
    },
}


# 这里的 split 标识仅用于按天切分数据集，不承载图结构语义。
DATASET_DEFAULT_CONFIG = {
    "THEIA_E5": {
        "raw_dir": _dataset_raw_dir("theia"),
        "db_name": "theia_e5_new",
        "db_name_all": "theia_e5_new",
        "year_month": "2019-05",
        "day_range": (8, 18),
        "train_splits": ["day_8", "day_9", "day_10"],
        "val_splits": ["day_11"],
        "test_splits": ["day_14", "day_15"],
        "unused_splits": ["day_12", "day_13", "day_16", "day_17"],
        "ground_truth_paths": [
            "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv",
        ],
        "attack_windows": [
            ["E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv", "2019-05-15 14:47:00", "2019-05-15 15:08:00"],
        ],
    },
    # Full THEIA_E3 split used by the current project:
    # - train: Apr-02 ~ Apr-08
    # - val: Apr-09
    # - test: Apr-10 / Apr-12 / Apr-13
    # - unused: Apr-11
    "THEIA_E3": {
        "raw_dir": _dataset_raw_dir("theia"),
        "db_name": "theia_e3",
        "db_name_all": "theia_e3",
        "year_month": "2018-04",
        "day_range": (2, 14),
        "train_splits": ["day_2", "day_3", "day_4", "day_5", "day_6", "day_7", "day_8"],
        "val_splits": ["day_9"],
        "test_splits": ["day_10", "day_12", "day_13"],
        "unused_splits": ["day_11"],
        "ground_truth_paths": [
            "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
            "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
        ],
        "attack_windows": [
            ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv", "2018-04-12 12:40:00", "2018-04-12 13:30:00"],
            ["E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv", "2018-04-10 14:30:00", "2018-04-10 15:00:00"],
        ],
    },
    # Minimal reproducible slice for quick iteration:
    # - train on Apr-09 benign stream
    # - carve validation from Apr-09 by process
    # - test on Apr-10 attack day
    "THEIA_E3_MINI": {
        "raw_dir": _dataset_raw_dir("theia"),
        "db_name": "theia_e3",
        "db_name_all": "theia_e3",
        "year_month": "2018-04",
        "day_range": (9, 11),
        "train_splits": ["day_9"],
        "val_splits": [],
        "test_splits": ["day_10"],
        "unused_splits": [],
        "val_ratio_from_train": 0.15,
        "val_split_seed": 42,
        "ground_truth_paths": [
            "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
            "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
        ],
        "attack_windows": [
            ["E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv", "2018-04-10 14:30:00", "2018-04-10 15:00:00"],
        ],
    },
    "CADETS_E5": {
        "raw_dir": _dataset_raw_dir("cadets"),
        "db_name": "cadets_e5",
        "db_name_all": "cadets_e5",
        "year_month": "2019-05",
        "day_range": (8, 18),
        "train_splits": ["day_8", "day_9", "day_11"],
        "val_splits": ["day_12"],
        "test_splits": ["day_16", "day_17"],
        "unused_splits": ["day_15", "day_10", "day_13", "day_14"],
        "ground_truth_paths": [
            "E5-CADETS/node_Nginx_Drakon_APT.csv",
            "E5-CADETS/node_Nginx_Drakon_APT_17.csv",
        ],
        "attack_windows": [
            ["E5-CADETS/node_Nginx_Drakon_APT.csv", "2019-05-16 09:31:00", "2019-05-16 10:12:00"],
            ["E5-CADETS/node_Nginx_Drakon_APT_17.csv", "2019-05-17 10:15:00", "2019-05-17 15:33:00"],
        ],
    },
    "CADETS_E3": {
        "raw_dir": _dataset_raw_dir("cadets"),
        "db_name": "cadets_e3",
        "db_name_all": "cadets_e3",
        "year_month": "2018-04",
        "day_range": (2, 14),
        "train_splits": ["day_2", "day_3","day_4", "day_5", "day_7", "day_8", "day_9"],
        "val_splits": [ "day_10"],
        "test_splits": ["day_6", "day_11", "day_12", "day_13"],
        "unused_splits": [],
        "ground_truth_paths": [
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            # "E3-CADETS/node_Nginx_Backdoor_11.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
            "E3-CADETS/node_Nginx_Backdoor_13.csv",
        ],
        "attack_windows": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            # ["E3-CADETS/node_Nginx_Backdoor_11.csv", "2018-04-11 15:07:00", "2018-04-11 15:16:00"],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
            ["E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"],
        ],
    },
    "CLEARSCOPE_E5": {
        "raw_dir": _dataset_raw_dir("clearscope"),
        "db_name": "clearscope_e5",
        "db_name_all": "clearscope_e5",
        "year_month": "2019-05",
        "day_range": (8, 18),
        "train_splits": ["day_8", "day_9"],
        "val_splits": ["day_11"],
        "test_splits": ["day_14", "day_15", "day_17"],
        "unused_splits": ["day_10", "day_12", "day_13", "day_16"],
        "ground_truth_paths": [
            "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
            "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
            "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
        ],
        "attack_windows": [
            ["E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv", "2019-05-15 15:38:00", "2019-05-15 16:19:00"],
            ["E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv", "2019-05-17 15:48:00", "2019-05-17 16:01:00"],
            ["E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv", "2019-05-17 16:20:00", "2019-05-17 16:28:00"],
        ],
    },
    "CLEARSCOPE_E3": {
        "raw_dir": _dataset_raw_dir("clearscope"),
        "db_name": "clearscope_e3",
        "db_name_all": "clearscope_e3",
        "year_month": "2018-04",
        "day_range": (2, 14),
        "train_splits": ["day_3", "day_4", "day_5", "day_7", "day_8", "day_9", "day_10"],
        "val_splits": ["day_2"],
        "test_splits": ["day_11", "day_12"],
        "unused_splits": ["day_6", "day_13"],
        "ground_truth_paths": [
            "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
        ],
        "attack_windows": [
            ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv", "2018-04-11 13:54:00", "2018-04-11 14:48:00"],
        ],
    },

    "OPTC_201": {
        "raw_dir": "",
        "db_name": "optc_201",
        "db_name_all": "optc_201",
        "num_node_types": 3,
        "num_edge_types": 10,
        "year_month": "2019-09",
        "day_range": (15, 26),
        "train_splits": ["day_19", "day_20", "day_21"],
        "val_splits": ["day_22"],
        "test_splits": ["day_23", "day_24", "day_25"],
        "unused_splits": ["day_16", "day_17", "day_18"],
        "ground_truth_paths": [
            "h201/node_h201_0923.csv",
        ],
        "attack_windows": [
            ["h201/node_h201_0923.csv", "2019-09-23 11:23:00", "2019-09-23 13:25:00"],
        ],
    },
    "OPTC_501": {
        "raw_dir": "",
        "db_name": "optc_501",
        "db_name_all": "optc_501",
        "num_node_types": 3,
        "num_edge_types": 10,
        "year_month": "2019-09",
        "day_range": (15, 26),
        "train_splits": ["day_19", "day_20", "day_21"],
        "val_splits": ["day_22"],
        "test_splits": ["day_23", "day_24", "day_25"],
        "unused_splits": ["day_16", "day_17", "day_18"],
        "ground_truth_paths": [
            "h501/node_h501_0924.csv",
        ],
        "attack_windows": [
            ["h501/node_h501_0924.csv", "2019-09-24 10:28:00", "2019-09-24 15:29:00"],
        ],
    },
    "OPTC_051": {
        "raw_dir": "",
        "db_name": "optc_051",
        "db_name_all": "optc_051",
        "num_node_types": 3,
        "num_edge_types": 10,
        "year_month": "2019-09",
        "day_range": (15, 26),
        "train_splits": ["day_19", "day_20", "day_21"],
        "val_splits": ["day_22"],
        "test_splits": ["day_23", "day_24", "day_25"],
        "unused_splits": ["day_16", "day_17", "day_18"],
        "ground_truth_paths": [
            "h051/node_h051_0925.csv",
        ],
        "attack_windows": [
            ["h051/node_h051_0925.csv", "2019-09-25 10:29:00", "2019-09-25 14:25:00"],
        ],
    },

}


def _create_cfg_recursive(cfg: CN, schema_dict: dict) -> None:
    for key, value in schema_dict.items():
        if isinstance(value, dict):
            setattr(cfg, key, CN())
            _create_cfg_recursive(getattr(cfg, key), value)
        else:
            setattr(cfg, key, None)
def get_default_cfg(args) -> CN:
    cfg = CN()
    cfg._use_cpu = args.cpu
    cfg._ground_truth_dir = ROOT_GROUND_TRUTH_DIR

    cfg.database = CN()
    for attr, value in DATABASE_DEFAULT_CONFIG.items():
        setattr(cfg.database, attr, value)

    cfg.dataset = CN()
    cfg.dataset.name = args.dataset
    for attr, value in DATASET_DEFAULT_CONFIG[cfg.dataset.name].items():
        setattr(cfg.dataset, attr, value)

    _create_cfg_recursive(cfg, CONFIG_SCHEMA)
    return cfg


def get_runtime_required_args(return_unknown_args: bool = False, args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset name.")
    parser.add_argument("--cpu", action="store_true", help="Run CPU-only utilities without assuming CUDA.")
    parser = add_cfg_args_to_parser(CONFIG_SCHEMA, parser)

    try:
        parsed_args, unknown_args = parser.parse_known_args(args)
    except Exception:
        parser.print_help()
        sys.exit(1)

    parsed_args.model = "clad"

    if return_unknown_args:
        return parsed_args, unknown_args
    return parsed_args


def overwrite_cfg_with_args(cfg: CN, args) -> None:
    for arg, value in args.__dict__.items():
        if "." in arg and value is not None:
            cfg_ptr = cfg
            dots = arg.split(".")
            path, attr_name = dots[:-1], dots[-1]

            for attr in path:
                cfg_ptr = getattr(cfg_ptr, attr)
            setattr(cfg_ptr, attr_name, value)


def validate_config_file(yml_file: str) -> None:
    if yaml is None:
        raise ModuleNotFoundError("pyyaml is required to load YAML config files.")

    with open(yml_file, "r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file)

    def validate_config(user_cfg, schema_cfg, path=None):
        if path is None:
            path = []
        if user_cfg is None:
            raise ValueError(f"Config at {' > '.join(path) or '<root>'} is empty but should not be.")

        for key, value_type in schema_cfg.items():
            if key not in user_cfg:
                continue

            sub_cfg = user_cfg[key]
            if isinstance(value_type, dict):
                if not isinstance(sub_cfg, dict):
                    raise TypeError(f"Section '{' > '.join(path + [key])}' should be a mapping.")
                validate_config(sub_cfg, value_type, path + [key])
            else:
                if sub_cfg is None:
                    raise ValueError(f"Parameter '{' > '.join(path + [key])}' should not be None.")
                if not isinstance(sub_cfg, value_type):
                    raise TypeError(
                        f"Parameter '{' > '.join(path + [key])}' should be of type {value_type.__name__}."
                    )

    validate_config(user_config, CONFIG_SCHEMA)


def check_args(args) -> None:
    available_datasets = DATASET_DEFAULT_CONFIG.keys()
    if args.dataset not in available_datasets:
        raise ValueError(f"Unknown dataset {args.dataset}. Available datasets are {available_datasets}")


def get_yml_cfg(args) -> CN:
    check_args(args)

    cfg = get_default_cfg(args)

    yml_file = _repo_dir("cs4m", "config", f"{args.model}.yml")
    validate_config_file(yml_file)

    cfg.merge_from_file(yml_file)
    overwrite_cfg_with_args(cfg, args)

    return cfg


def add_cfg_args_to_parser(cfg_schema, parser):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v == "None":
            return None
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    def nested_dict_to_separator_dict(nested_dict, separator="."):
        def _create_separator_dict(x, key="", separator_dict=None):
            if separator_dict is None:
                separator_dict = {}
            if isinstance(x, dict):
                for nested_key, nested_value in x.items():
                    full_key = f"{key}{separator}{nested_key}" if key else nested_key
                    _create_separator_dict(nested_value, full_key, separator_dict)
            else:
                separator_dict[key] = x
            return separator_dict

        return _create_separator_dict(deepcopy(nested_dict))

    separator_dict = nested_dict_to_separator_dict(cfg_schema)

    for key, value in separator_dict.items():
        is_bool = value == type(True)
        dtype = str2bool if is_bool else value
        parser.add_argument(f"--{key}", type=dtype)

    return parser
