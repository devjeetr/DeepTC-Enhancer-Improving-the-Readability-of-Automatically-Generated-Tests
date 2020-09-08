import argparse
import os
import shlex
import subprocess
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from preprocessing.context_utils import (
    path_iterator,
    get_variables_from_contexts,
    mask_variables_in_contexts
)
from preprocessing.create_variable_name_dataset import generate_variable_name_data

def join_args(args):
    return " ".join([f"{k} {v}" for k, v in args.items()])


def run_java_extractor(target_file, java_extractor_path, outdir, max_path_length=8, max_path_width=2):
    args = {
        "--file": target_file,
        "--num_threads": 1,
        "--max_path_length": max_path_length,
        "--max_path_width": max_path_width,
    }

    cmd = f"java -cp {java_extractor_path} JavaExtractor.App {join_args(args)}"

    with open(outdir, "w") as f:
        subprocess.call(shlex.split(cmd), stdout=f)


def run_method_isolator(jar, source, destination):
    cmd = f"java -jar {jar} {source} {destination}"
    subprocess.call(shlex.split(cmd))


def run_dataset_transformer(target_file, outdir, executable_path):
    cmd = f"{executable_path} -i {target_file} -o {outdir} --dict-prefix prefix"
    subprocess.call(shlex.split(cmd))


def mask_variables_for_method_name(src_folder, out_folder):
    raw_src_files = os.listdir(src_folder)
    raw_src_files = map(lambda p: os.path.join(src_folder, p), raw_src_files)

    for raw_src_file in raw_src_files:
        with open(raw_src_file, "r") as f:
            lines = map(lambda s: s.strip(), f.readlines())
            masked_paths = []
            for label, contexts in path_iterator(lines):
                masked_contexts = []
                masked_contexts = mask_variables_in_contexts(contexts)
                masked_contexts = [",".join(masked_context) for masked_context in masked_contexts]
                masked_contexts = " ".join(masked_contexts)
                masked_paths.append(f"{label} {masked_contexts}")

            out_file = os.path.join(out_folder, os.path.basename(raw_src_file))
            with open(out_file, "w") as out:
                for path in masked_paths:
                    out.write(f"{path}\n")



def generate_variable_data(input_folder, output_folder):
    input_files = os.listdir(input_folder)

    for input_file in input_files:
        input_file = os.path.join(input_folder, input_file)
        generate_variable_name_data(
            Namespace(
                i=input_file,
                o=os.path.join(output_folder, os.path.basename(input_file)),
                n_workers=1,
            )
        )


def print_if_verbose(msg, verbose=False):
    if verbose:
        print(msg)

def create_raw_dataset_for_inference(
    src_file, out_folder, java_method_extractor="", jextractor_path="", variables=False, verbose=False
):
    ISOLATED_SRC_FOLDER = "isolated-method-src"
    RAW_C2S_FOLDER = "raw-c2s"
    if not os.path.exists(src_file):
        raise Exception(f"Source file doesnt exist: {src_file}")
    if not os.path.exists(java_method_extractor):
        raise Exception(f"Java method isolator jar path invalid:\n{java_method_extractor}")

    if not os.path.exists(jextractor_path):
        raise Exception(f"Java extractor jar path invalid\n{jextractor_path}")

    print_if_verbose("Creating tmp dir", verbose=verbose)
    # first isolate methods
    with tempfile.TemporaryDirectory() as tmp_dir:
        isolated_src_path = os.path.join(tmp_dir, ISOLATED_SRC_FOLDER)
        os.mkdir(isolated_src_path)

        # step 1 run method isolator
        print_if_verbose("Running method isolator", verbose=verbose)
        run_method_isolator(java_method_extractor, src_file, isolated_src_path)
        method_files = os.listdir(isolated_src_path)
        method_files = map(lambda p: os.path.join(isolated_src_path, p), method_files)

        # step 2 run c2s java extractor
        print_if_verbose("running java extractor", verbose=verbose)
        raw_c2s_path = os.path.join(tmp_dir, RAW_C2S_FOLDER)
        os.mkdir(raw_c2s_path)
        for method_file in method_files:
            run_java_extractor(
                method_file,
                jextractor_path,
                os.path.join(raw_c2s_path, os.path.basename(method_file)),
                max_path_width=2, # TODO
                max_path_length=8, # TODO
            )

        # raise Exception("Check above")

        # step 3 mask
        print_if_verbose("Masking", verbose=verbose)
        if variables:
            generate_variable_data(raw_c2s_path, out_folder)
        else:
            mask_variables_for_method_name(raw_c2s_path, out_folder)

        print_if_verbose("Completed!", verbose=verbose)


def main(args):

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    if os.path.isdir(args.i):
        source_root = Path(args.i)
        java_files = list(map(str, source_root.glob("**/*.java")))
        output_root = args.o

        if args.variables:
            output_root = os.path.join(output_root, "variables")
        else:
            output_root = os.path.join(output_root, "methods")

        for java_file in java_files:
            current_outfolder = os.path.join(output_root, os.path.basename(java_file))

            if not os.path.exists(current_outfolder):
                print(f"Making folder {current_outfolder}")
                os.makedirs(current_outfolder)

            create_raw_dataset_for_inference(
                java_file,
                current_outfolder,
                java_method_extractor=args.method_isolator_path,
                jextractor_path=args.jpredict_path,
                variables=args.variables,
                verbose=args.verbose
            )
    else:
        create_raw_dataset_for_inference(
            args.i,
            args.o,
            java_method_extractor=args.method_isolator_path,
            jextractor_path=args.jpredict_path,
            variables=args.variables,
            verbose=args.verbose
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--i")
    parser.add_argument("--o")
    parser.add_argument("--variables", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--jpredict_path", default="./interactive_predictions/jars/jpredict.jar"
    )
    parser.add_argument(
        "--method_isolator_path",
        default="./interactive_predictions/jars/isolate-methods.jar",
    )
    args = parser.parse_args()

    main(args)
