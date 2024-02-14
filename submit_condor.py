import os
import argparse
import subprocess
from typing import Dict, List, Optional
import utils
from utils.helpers.modules.package_names import get_package_names
from pathlib import Path

python_executable = "/work/phummler/miniconda3/envs/gnn_efficiency/bin/python"


def get_bash_run_id_command(bootstrap: bool):
    return f"""RUNID=$(python -c "from utils.helpers.run_id_handler import RunIdHandler; print(RunIdHandler.generate_run_id(bootstrap={bootstrap}));" >&1)"""


def get_commands(
    task: str,
    dataset: str,
    dataset_handling: Optional[str],
    working_points_set: Optional[str],
    model: Optional[str],
    bootstrap: bool,
    run_id: Optional[str],
    prediction_dataset_handling: Optional[str],
    evaluation_model_selection: Optional[str],
    evaluation_data_manipulation_list: Optional[List[str]],
    save_predictions_after_training_prediction_dataset_handling: Optional[List[str]],
) -> List[str]:
    main_script = "main.py"
    cmd = f"python {main_script}"
    cmd += f" {task}"
    cmd += f" --dataset={dataset}"
    if dataset_handling is not None:
        cmd += f" --dataset_handling={dataset_handling}"
    if working_points_set is not None:
        cmd += f" --working_points_set={working_points_set}"
    if model is not None:
        cmd += f" --model={model}"
    if bootstrap is True:
        cmd += " --bootstrap"
    if run_id is not None:
        cmd += f" --run_id={run_id}"
    if prediction_dataset_handling is not None:
        cmd += f" --prediction_dataset_handling={prediction_dataset_handling}"
    if evaluation_model_selection is not None:
        cmd += f" --evaluation_model_selection={evaluation_model_selection}"
    if evaluation_data_manipulation_list is not None:
        cmd += f" --evaluation_data_manipulation {' '.join(evaluation_data_manipulation_list)}"
    if save_predictions_after_training_prediction_dataset_handling is not None:
        cmd += " --run_id=$RUNID"
        commands = [
            get_bash_run_id_command(bootstrap=bootstrap),
            cmd,
            *[
                cmd
                for e in save_predictions_after_training_prediction_dataset_handling
                for cmd in get_commands(
                    task="save_predictions",
                    dataset=dataset,
                    dataset_handling=dataset_handling,
                    working_points_set=working_points_set,
                    model=model,
                    bootstrap=False,  # False as bootstrap is not an argument for parser_save_predictions
                    run_id="$RUNID",
                    prediction_dataset_handling=e,
                    evaluation_model_selection=None,  # None, as it is not an argument for parser_save_predictions
                    evaluation_data_manipulation_list=None,  # None, as it is not an argument for parser_save_predictions
                    save_predictions_after_training_prediction_dataset_handling=None,
                )
            ],
        ]
    else:
        commands = [cmd]
    return commands


def get_condor_kwargs(
    task: str,
    dataset: str,
    dataset_handling: Optional[str],
    working_points_set: Optional[str],
    model: Optional[str],
    bootstrap: Optional[bool],
    run_id: Optional[str],
    prediction_dataset_handling: Optional[str],
    evaluation_model_selection: Optional[str],
    evaluation_data_manipulation_list: Optional[List[str]],
    array: Optional[int],
):
    job_name = task

    is_test_dataset = dataset.endswith("_test")

    if task == "extract":
        job_name += f"_{dataset}"
        cpus_per_task = 8
        n_gpus = 0
        if is_test_dataset:
            job_flavour = "microcentury"
            mem_gb = 10
        else:
            job_flavour = "tomorrow"
            mem_gb = 80
    elif task == "train":
        job_name += (
            f"_{dataset}"
            f"_{dataset_handling}"
            f"_{working_points_set}"
            f"_{model}"
            f"_{bootstrap}"
        )
        if model.startswith("gnn"):
            cpus_per_task = 2
            n_gpus = 1
            if is_test_dataset:
                job_flavour = "microcentury"
                mem_gb = 10
            else:
                if "single_wp_mode" in model:
                    job_flavour = "testmatch"
                else:
                    job_flavour = "tomorrow"
                if dataset.startswith("TTTo2L2Nu"):
                    mem_gb = 80
                else:
                    mem_gb = 48
        elif model.startswith("eff_map"):
            cpus_per_task = 4
            n_gpus = 0
            if is_test_dataset:
                job_flavour = "microcentury"
                mem_gb = 10
            else:
                job_flavour = "microcentury"
                mem_gb = 60
        elif model.startswith("direct_tagging"):
            cpus_per_task = 4
            n_gpus = 0
            if is_test_dataset:
                job_flavour = "microcentury"
                mem_gb = 10
            else:
                job_flavour = "microcentury"
                mem_gb = 60
        else:
            raise ValueError(f"Unknown model: {model}")
    elif task == "save_predictions":
        job_name += (
            f"_{dataset}_"
            f"{dataset_handling}_"
            f"{working_points_set}_"
            f"{model}_"
            f"{run_id}_"
            f"{prediction_dataset_handling}"
        )
        if model.startswith("gnn"):
            cpus_per_task = 2
            n_gpus = 1
            if is_test_dataset:
                job_flavour = "microcentury"
                mem_gb = 10
            else:
                job_flavour = "workday"
                mem_gb = 48
        elif model.startswith("eff_map"):
            cpus_per_task = 4
            n_gpus = 0
            if is_test_dataset:
                job_flavour = "microcentury"
                mem_gb = 10
            else:
                time_hours = "microcentury"
                mem_gb = 60
        elif model.startswith("direct_tagging"):
            cpus_per_task = 4
            n_gpus = 0
            if is_test_dataset:
                job_flavour = "microcentury"
                mem_gb = 10
            else:
                job_flavour = "microcentury"
                mem_gb = 60
        else:
            raise ValueError(f"Unknown model: {model}")
    elif task == "evaluate":
        job_name += (
            f"_{dataset}_"
            f"{dataset_handling}_"
            f"{working_points_set}_"
            f"{prediction_dataset_handling}_"
            f"{evaluation_model_selection}_"
            f"{'__'.join(evaluation_data_manipulation_list)}"
        )
        cpus_per_task = 2
        n_gpus = 0
        if is_test_dataset:
            job_flavour = "microcentury"
            mem_gb = 10
        else:
            if dataset.startswith("TTTo2L2Nu"):
                job_flavour = "tomorrow"
                if evaluation_model_selection.startswith("individual"):
                    mem_gb = 220
                else:
                    mem_gb = 120
            else:
                job_flavour = "longlunch"
                mem_gb = 20
    else:
        raise ValueError(f"Unknown task: {task}")
    gres = None
    if n_gpus > 0:
        requested_gpus = 4
    else:
        requested_gpus = 0
    mem = f"{int(mem_gb)}G"
    # output = "condor_"
    # error = "condor_%j-%x.err"

    main_dir = Path.cwd()
    condor_dir = Path(f"{main_dir}/condor")

    # create logs and output directories

    log_dir = Path(str(condor_dir) + "/logs")
    if not log_dir.exists():
        log_dir.mkdir()
    # define output directories

    out_dir = Path(f"{condor_dir}/out/")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    # define EOS area to move output files

    username = os.environ["USER"]
    eos_dir = Path(f"/eos/user/{username[0]}/{username}/btv")
    if not eos_dir.exists():
        eos_dir.mkdir(parents=True)

    res = {
        "job_flavour": job_flavour,
        "requested_gpus": requested_gpus,
        "job_name": job_name,
        "condor_dir": str(condor_dir),
        "eos_dir": str(eos_dir),
        "main_dir": str(main_dir),
    }

    if gres is not None:
        res["gres"] = gres
    if array is not None:
        assert isinstance(array, int)
        res["array"] = f"0-{array - 1}"
    return res


def submit_to_lxplus(commands: List[str], condor_kwargs: Dict[str, str]):
    # try:
    #    subprocess.run(["sinfo", "--version"])
    # except FileNotFoundError as file_not_found_error:
    #    raise OSError(
    #        "Slurm seems to be not available. Make sure to run this on T3"
    #    ) from file_not_found_error

    print("\n")
    print(f"{commands=}\n")
    print(f"{condor_kwargs=}\n")

    # make condor file

    if condor_kwargs["requested_gpus"]:
        condor_template_file = open(f"{condor_kwargs['condor_dir']}/submit_gpu.sub")
    else:
        condor_template_file = open(f"{condor_kwargs['condor_dir']}/submit.sub")
    local_condor = f"condor/{condor_kwargs['job_name']}.sub"
    condor_file = open(local_condor, "w")
    for line in condor_template_file:
        line = line.replace("DIRECTORY", condor_kwargs["condor_dir"])
        line = line.replace("JOBNAME", condor_kwargs["job_name"])
        line = line.replace("JOBFLAVOUR", condor_kwargs["job_flavour"])
        condor_file.write(line)
    condor_file.close()
    condor_template_file.close()

    # make executable file

    sh_template_file = open(f"{condor_kwargs['condor_dir']}/submit.sh")
    local_sh = f"{condor_kwargs['condor_dir']}/{condor_kwargs['job_name']}.sh"
    sh_file = open(local_sh, "w")
    for line in sh_template_file:
        line = line.replace("MAINDIRECTORY", condor_kwargs["main_dir"])
        line = line.replace("EOSDIR", condor_kwargs["eos_dir"])
        line = line.replace("COMMANDS", commands[0])
        sh_file.write(line)
    sh_file.close()
    sh_template_file.close()
    # submit condor jobs
    #subprocess.run(["condor_submit", local_condor], shell=True)


def submit(
    task: str,
    dataset: str,
    dataset_handling: Optional[str],
    working_points_set: Optional[str],
    model: Optional[str],
    bootstrap: Optional[bool],
    run_id: Optional[str],
    prediction_dataset_handling: Optional[str],
    evaluation_model_selection: Optional[str],
    evaluation_data_manipulation_list: Optional[List[str]],
    save_predictions_after_training_prediction_dataset_handling: Optional[List[str]],
    array: Optional[int],
):
    commands = get_commands(
        task=task,
        dataset=dataset,
        dataset_handling=dataset_handling,
        working_points_set=working_points_set,
        model=model,
        bootstrap=bootstrap,
        run_id=run_id,
        prediction_dataset_handling=prediction_dataset_handling,
        evaluation_model_selection=evaluation_model_selection,
        evaluation_data_manipulation_list=evaluation_data_manipulation_list,
        save_predictions_after_training_prediction_dataset_handling=save_predictions_after_training_prediction_dataset_handling,
    )

    condor_kwargs = get_condor_kwargs(
        task=task,
        dataset=dataset,
        dataset_handling=dataset_handling,
        working_points_set=working_points_set,
        model=model,
        bootstrap=bootstrap,
        run_id=run_id,
        prediction_dataset_handling=prediction_dataset_handling,
        evaluation_model_selection=evaluation_model_selection,
        evaluation_data_manipulation_list=evaluation_data_manipulation_list,
        array=array,
    )

    submit_to_lxplus(
        commands=commands,
        condor_kwargs=condor_kwargs,
    )


def get_parsers():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="task", help="task to do", required=True)

    parser_train = subparsers.add_parser(
        name="train",
        help="train a model",
    )
    parser_save_predictions = subparsers.add_parser(
        name="save_predictions",
        help="save predictions of a model",
    )
    parser_evaluate = subparsers.add_parser(
        name="evaluate",
        help="run an evaluation",
    )
    parser_extract = subparsers.add_parser(
        name="extract",
        help="extract a dataset",
    )

    for p in [parser_train, parser_save_predictions, parser_evaluate, parser_extract]:
        p.add_argument(
            "--dataset",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="dataset", config_name="foo", mkdir=False
                ).parent
            ),
            help="name of the dataset config",
            required=True,
        )
    for p in [parser_train, parser_save_predictions, parser_evaluate]:
        p.add_argument(
            "--dataset_handling",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="dataset_handling", config_name="foo", mkdir=False
                ).parent
            ),
            help="name of the dataset handling config",
            required=True,
        )
    for p in [parser_train, parser_save_predictions, parser_evaluate]:
        p.add_argument(
            "--working_points_set",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="working_points_set", config_name="foo", mkdir=False
                ).parent
            ),
            help="name of the working points set config",
            required=True,
        )
    for p in [parser_train, parser_save_predictions]:
        p.add_argument(
            "--model",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="model", config_name="foo", mkdir=False
                ).parent
            ),
            help="name of the model config",
            required=True,
        )
    parser_train.add_argument(
        "--bootstrap",
        action="store_true",
        help="whether its a bootstrap run",
    )

    parser_save_predictions.add_argument(
        "--run_id",
        help="run_id of the model",
        required=True,
    )

    for p in [parser_save_predictions, parser_evaluate]:
        p.add_argument(
            "--prediction_dataset_handling",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="prediction_dataset_handling",
                    config_name="foo",
                    mkdir=False,
                ).parent
            ),
            help="name of the prediction dataset handling configs",
            required=True,
        )
    parser_evaluate.add_argument(
        "--evaluation_model_selection",
        choices=get_package_names(
            dir_path=utils.paths.config(
                config_type="evaluation_model_selection",
                config_name="foo",
                mkdir=False,
            ).parent
        ),
        help="name of the evaluation model selection config",
        required=True,
    )

    parser_evaluate.add_argument(
        "--evaluation_data_manipulation",
        nargs="+",
        choices=get_package_names(
            dir_path=utils.paths.config(
                config_type="evaluation_data_manipulation",
                config_name="foo",
                mkdir=False,
            ).parent
        ),
        help="names of the evaluation data manipulation configs",
        required=True,
    )

    return (
        parser,
        parser_train,
        parser_save_predictions,
        parser_evaluate,
        parser_extract,
    )


def main():
    (
        parser,
        parser_train,
        parser_save_predictions,
        parser_evaluate,
        parser_extract,
    ) = get_parsers()

    parser_train.add_argument(
        "--save_predictions_after_training_prediction_dataset_handling",
        nargs="*",
        choices=get_package_names(
            dir_path=utils.paths.config(
                config_type="prediction_dataset_handling",
                config_name="foo",
                mkdir=False,
            ).parent
        ),
        help="names of the prediction dataset handling config",
    )

    parser_train.add_argument(
        "--array",
        type=int,
        choices=range(1, 25 + 1),
        help="spawn a job array",
    )

    args = parser.parse_args()

    task = args.task

    dataset = args.dataset

    try:
        save_predictions_after_training_prediction_dataset_handling = (
            args.save_predictions_after_training_prediction_dataset_handling
        )
    except AttributeError:
        save_predictions_after_training_prediction_dataset_handling = None
    try:
        array = args.array
    except AttributeError:
        array = None
    try:
        dataset_handling = args.dataset_handling
    except AttributeError:
        dataset_handling = None
    try:
        working_points_set = args.working_points_set
    except AttributeError:
        working_points_set = None
    try:
        model = args.model
    except AttributeError:
        model = None
    try:
        bootstrap = args.bootstrap
    except AttributeError:
        bootstrap = None
    try:
        run_id = args.run_id
    except AttributeError:
        run_id = None
    try:
        prediction_dataset_handling = args.prediction_dataset_handling
    except AttributeError:
        prediction_dataset_handling = None
    try:
        evaluation_model_selection = args.evaluation_model_selection
    except AttributeError:
        evaluation_model_selection = None
    try:
        evaluation_data_manipulation_list = args.evaluation_data_manipulation
    except AttributeError:
        evaluation_data_manipulation_list = None
    submit(
        task=task,
        dataset=dataset,
        dataset_handling=dataset_handling,
        working_points_set=working_points_set,
        model=model,
        bootstrap=bootstrap,
        run_id=run_id,
        prediction_dataset_handling=prediction_dataset_handling,
        evaluation_model_selection=evaluation_model_selection,
        evaluation_data_manipulation_list=evaluation_data_manipulation_list,
        save_predictions_after_training_prediction_dataset_handling=save_predictions_after_training_prediction_dataset_handling,
        array=array,
    )


if __name__ == "__main__":
    main()

