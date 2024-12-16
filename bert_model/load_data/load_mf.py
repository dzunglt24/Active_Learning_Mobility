import os
import csv
import datasets
import pandas as pd

logger = datasets.logging.get_logger(__name__)


class MFDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for MFDataset"""
    week: str = None
    augment: bool = False
    has_dev: bool = False

    def __init__(self, week, augment, has_dev, **kwargs):
        """BuilderConfig for MFDataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MFDatasetConfig, self).__init__(**kwargs)
        self.week = week
        self.augment = augment
        self.has_dev = has_dev


class MFDataset(datasets.GeneratorBasedBuilder):
    """MBDataset dataset."""
    BUILDER_CONFIG_CLASS = MFDatasetConfig
    BUILDER_CONFIGS = [
        
        MFDatasetConfig(name="Action", 
                        version=datasets.Version("1.0.0"), 
                        description="Action dataset",
                        week="Week0",
                        augment=False,
                        has_dev=False),
        MFDatasetConfig(name="Mobility", 
                        version=datasets.Version("1.1.0"), 
                        description="Mobility dataset",
                        week="Week0",
                        augment=False,
                        has_dev=False),
        MFDatasetConfig(name="Assistance", 
                        version=datasets.Version("1.2.0"), 
                        description="Assistance dataset",
                        week="Week0",
                        augment=False,
                        has_dev=False),
        MFDatasetConfig(name="Quantification", 
                        version=datasets.Version("1.3.0"), 
                        description="Quantification dataset",
                        week="Week0",
                        augment=False,
                        has_dev=False),
    ]

    def _info(self):
        print(f"{self.config.week},  Augment: {self.config.augment}, Has dev set: {self.config.has_dev}")
        _TRAINING_FILE =  os.path.join(self.config.data_dir, self.config.week, 
                                    self.config.name, "train.tsv")
        if self.config.augment:
            _TRAINING_FILE =  os.path.join(self.config.data_dir, self.config.week, 
                                        self.config.name, "train_aug.tsv")
        _TEST_FILE = os.path.join(self.config.data_dir, self.config.week, 
                                        self.config.name, "test.tsv")

        if self.config.has_dev:
            _DEV_FILE = os.path.join(self.config.data_dir, self.config.week, 
                                        self.config.name, "dev.tsv")
            print("Using DEV set")
        else:
            _DEV_FILE = os.path.join(self.config.data_dir, self.config.week, 
                                        self.config.name, "test.tsv")
        train_df = pd.read_csv(_TRAINING_FILE, sep="\t", names=["token", "tag"], quoting=csv.QUOTE_NONE)#, error_bad_lines=False, engine="python")
        dev_df = pd.read_csv(_DEV_FILE, sep="\t", names=["token", "tag"], quoting=csv.QUOTE_NONE)#error_bad_lines=False, engine="python")
        test_df = pd.read_csv(_TEST_FILE, sep="\t", names=["token", "tag"], quoting=csv.QUOTE_NONE)#error_bad_lines=False, engine="python")
        label_list = sorted(list(set(list(train_df["tag"].unique()) \
            + list(dev_df["tag"].unique()) + list(test_df["tag"].unique()))))

        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=label_list
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        _TRAINING_FILE =  os.path.join(self.config.data_dir, self.config.week, 
                                    self.config.name, "train.tsv")
        if self.config.augment:
            _TRAINING_FILE =  os.path.join(self.config.data_dir, self.config.week, 
                                        self.config.name, "train_aug.tsv")
        _TEST_FILE = os.path.join(self.config.data_dir, self.config.week, 
                                        self.config.name, "test.tsv")

        if self.config.has_dev:
            _DEV_FILE = os.path.join(self.config.data_dir, self.config.week, 
                                        self.config.name, "dev.tsv")
        else:
            _DEV_FILE = os.path.join(self.config.data_dir, self.config.week, 
                                        self.config.name, "test.tsv")

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": _TRAINING_FILE}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepath": _DEV_FILE}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                                    "filepath": _TEST_FILE}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # MFDataset tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
