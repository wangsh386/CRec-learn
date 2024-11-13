import logging

from pyhealth.datasets import (MIMIC3Dataset, MIMIC4Dataset, collate_fn_dict,
                               eICUDataset, split_by_patient)
from pyhealth.tasks import (drug_recommendation_eicu_fn,
                            drug_recommendation_mimic3_fn,
                            drug_recommendation_mimic4_fn)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def build_dataloader(
    dataset_name: str,
    batch_size: int = 64,
    n_workers: int = 8,
    dev: bool = False,
    refresh_cache=False,
    seed=None,
):
    if dataset_name == "mimic3":
        data_root = "dataset/mimic-iii-clinical-database-1.4"
        dataset = MIMIC3Dataset(
            root=data_root,
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=dev,
            refresh_cache=refresh_cache,
        )
        task_fn = drug_recommendation_mimic3_fn
    elif dataset_name == "mimic4":
        data_root = "dataset/mimic-iv-2.2/hosp"
        dataset = MIMIC4Dataset(
            root=data_root,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=dev,
            refresh_cache=refresh_cache,
        )
        task_fn = drug_recommendation_mimic4_fn
    elif dataset_name == "eicu":
        data_root = "dataset/eicu-collaborative-research-database-2.0"
        dataset = eICUDataset(
            root=data_root,
            tables=["diagnosis", "physicalExam", "medication"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=dev,
            refresh_cache=refresh_cache,
        )
        task_fn = drug_recommendation_eicu_fn
    else:
        raise ValueError
    dataset = dataset.set_task(task_fn=task_fn)

    train_dataset, val_dataset, test_dataset = split_by_patient(
        dataset, [0.8, 0.1, 0.1], seed=seed
    )
    logger.info(f"len of train_dataset:{len(train_dataset)}")
    logger.info(f"len of val_dataset:{len(val_dataset)}")
    logger.info(f"len of test_dataset:{len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=collate_fn_dict,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=collate_fn_dict,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=collate_fn_dict,
    )

    return dataset, train_loader, val_loader, test_loader
